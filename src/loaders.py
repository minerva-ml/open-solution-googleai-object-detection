import os
import random
import torch
import numpy as np
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from PIL import Image
from steppy.base import BaseTransformer

from .retinanet import DataEncoder
from .pipeline_config import MEAN, STD, SEED
from .utils import get_target_size


class FixedSizeSampler(Sampler):
    def __init__(self, images_data, *,  sample_size=None, **kwargs):
        self.images_data = images_data.sample(frac=1, random_state=SEED)
        self.sample_size = len(self.images_data) if sample_size is None else min(sample_size, len(self.images_data))

    def __iter__(self):
        indices = random.sample(range(len(self.images_data)), self.sample_size)
        return iter(indices)

    def __len__(self):
        return self.sample_size


class AspectRatioSampler(Sampler):
    def __init__(self, images_data, *, batch_size=8, sample_size=None):
        self.images_data = images_data
        self.batch_size = batch_size
        self.sample_size = len(self.images_data) if sample_size is None else min(sample_size, len(self.images_data))
        self.sample_size = self.sample_size // self.batch_size * self.batch_size

        self.indices = self.images_data.sort_values('aspect_ratio').index.tolist()

    def __iter__(self):
        subset = sorted(random.sample(range(len(self.indices)), self.sample_size))
        indices = np.array([self.indices[i] for i in subset])\
            .reshape(self.sample_size // self.batch_size, self.batch_size)
        np.random.seed()
        np.random.shuffle(indices)
        indices = indices.flatten()

        return iter(indices)

    def __len__(self):
        return self.sample_size


class ImageDetectionDataset(Dataset):
    def __init__(self, images_data, annotations, annotations_human_labels, target_encoder, train_mode,
                 short_dim, long_dim, image_h, image_w, sampler_name, image_transform):
        super().__init__()
        self.images_data = images_data
        self.annotations = annotations
        self.annotations_human_labels = annotations_human_labels
        self.target_encoder = target_encoder
        self.train_mode = train_mode

        self.short_dim = short_dim
        self.long_dim = long_dim
        self.image_h = image_h
        self.image_w = image_w
        self.sampler_name = sampler_name
        self.image_transform = image_transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):

        Xi = self.load_from_disk(index)
        Xi = self.resize_image(Xi)

        if self.annotations is not None:
            w, h = Xi.size
            yi = self.load_target(index, (h, w))
            return Xi, yi
        else:
            return self.image_transform(Xi)

    def load_from_disk(self, index):
        img_path = self.images_data.iloc[index]['image_path']
        return self.load_image(img_path)

    def load_image(self, img_filepath, grayscale=False):
        image = Image.open(img_filepath, 'r')
        if not grayscale:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        return image

    def load_target(self, index, img_shape):
        imgId = self.images_data.iloc[index]['ImageID']
        boxes_rows = self.annotations[self.annotations['ImageID'] == imgId]
        return self.get_boxes_and_labels(boxes_rows, img_shape)

    def get_boxes_and_labels(self, boxes_rows, img_shape):
        boxes, labels = [], []
        h, w = img_shape
        for _, box_row in boxes_rows.iterrows():
            x_min = box_row['XMin'] * w
            x_max = box_row['XMax'] * w
            y_min = box_row['YMin'] * h
            y_max = box_row['YMax'] * h
            label = box_row['LabelName']
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
        return torch.FloatTensor(boxes), torch.FloatTensor(labels)

    def resize_image(self, image):
        if self.sampler_name == 'fixed':
            w, h = self.image_w, self.image_h
        else:
            org_w, org_h = image.size
            w, h = get_target_size(aspect_ratio=org_w/float(org_h), short_dim=self.short_dim, long_dim=self.long_dim)
        resize = transforms.Resize((h, w))
        return resize(image)

    def alling_images(self, images):
        max_h, max_w = 0, 0
        min_h, min_w = 1e10, 1e10
        for image in images:
            w, h = image.size
            max_h, max_w = max(h, max_h), max(w, max_w)
            min_h, min_w = min(h, min_h), min(w, min_w)

        resize = transforms.Resize((max_h, max_w))
        allinged_images = []
        for image in images:
            allinged_images.append(resize(image))

        return allinged_images

    def collate_fn(self, batch):
        """Encode targets.

        Args:
          batch: (list) of images, bbox_targets, clf_targets.

        Returns:
          images, stacked bbox_targets, stacked clf_targets.
        """
        imgs = [x[0] for x in batch]
        boxes = [x[1][0] for x in batch]
        labels = [x[1][1] for x in batch]

        imgs = self.alling_images(imgs)
        imgs = [self.image_transform(img) for img in imgs]

        inputs = torch.stack(imgs)
        _, _, h, w = inputs.size()
        bbox_targets, clf_targets = [], []
        for box, label in zip(boxes, labels):
            bbox_target, clf_target = self.target_encoder.encode(box, label, input_size=(w,h))
            bbox_targets.append(bbox_target)
            clf_targets.append(clf_target)

        bbox_targets, clf_targets = torch.stack(bbox_targets), torch.stack(clf_targets)
        clf_targets = clf_targets.unsqueeze(-1)
        targets = torch.cat((bbox_targets, clf_targets), 2)

        return inputs, targets


class ImageDetectionLoader(BaseTransformer):
    def __init__(self, train_mode, loader_params, dataset_params):
        super().__init__()
        self.train_mode = train_mode
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        sampler_name = self.dataset_params.sampler_name
        if sampler_name == 'fixed':
            self.sampler = FixedSizeSampler
        elif sampler_name == 'aspect ratio':
            self.sampler = AspectRatioSampler
        else:
            msg = "expected sampler name from (fixed, aspect ratio), got {} instead".format(sampler_name)
            raise Exception(msg)

        self.target_encoder = DataEncoder(**self.dataset_params.data_encoder)
        self.dataset = ImageDetectionDataset

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def transform(self, images_data, annotations=None, annotations_human_labels=None, valid_images_data=None):
        if self.train_mode and annotations is not None:
            flow, steps = self.get_datagen(images_data, annotations, annotations_human_labels, True,
                                           self.loader_params.training)
        else:
            flow, steps = self.get_datagen(images_data, None, None, False, self.loader_params.inference)

        if valid_images_data is not None:
            valid_flow, valid_steps = self.get_datagen(valid_images_data, annotations, annotations_human_labels, False,
                                                       self.loader_params.inference)
        else:
            valid_flow = None
            valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, images_data, annotations, annotations_human_labels, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(images_data,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   target_encoder=self.target_encoder,
                                   train_mode=True,
                                   short_dim=self.dataset_params.short_dim,
                                   long_dim=self.dataset_params.long_dim,
                                   image_h=self.dataset_params.image_h,
                                   image_w=self.dataset_params.image_w,
                                   sampler_name=self.dataset_params.sampler_name,
                                   image_transform=self.image_transform)

            datagen = DataLoader(dataset, **loader_params,
                                 sampler=self.sampler(images_data=images_data,
                                                      sample_size=self.dataset_params.sample_size,
                                                      batch_size=loader_params.batch_size),
                                 collate_fn=dataset.collate_fn)
        else:
            dataset = self.dataset(images_data,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   target_encoder=self.target_encoder,
                                   train_mode=False,
                                   short_dim=self.dataset_params.short_dim,
                                   long_dim=self.dataset_params.long_dim,
                                   image_h=self.dataset_params.image_h,
                                   image_w=self.dataset_params.image_w,
                                   sampler_name=self.dataset_params.sampler_name,
                                   image_transform=self.image_transform)

            if annotations is not None:
                datagen = DataLoader(dataset, **loader_params,
                                     sampler=self.sampler(images_data=images_data,
                                                          batch_size=loader_params.batch_size),
                                     collate_fn=dataset.collate_fn)
            else:
                datagen = DataLoader(dataset, **loader_params)

        steps = len(datagen)
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)
