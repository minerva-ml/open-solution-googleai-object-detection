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
from .pipeline_config import MEAN, STD


class RandomSubsetSampler(Sampler):
    def __init__(self, images_data, sample_size, batch_size):
        self.images_data = images_data
        self.batch_size = batch_size
        self.sample_size = min(sample_size, len(self.images_data))
        self.sample_size = self.sample_size // self.batch_size * self.batch_size

        self.indicies = self.images_data.sort_values('aspect_ratio').index.tolist()

    def __iter__(self):
        subset = sorted(random.sample(range(len(self.indicies)), self.sample_size))
        indicies = np.array([self.indicies[i] for i in subset])\
            .reshape(self.sample_size // self.batch_size, self.batch_size)
        np.random.shuffle(indicies)
        indicies = indicies.flatten()

        return iter(indicies)

    def __len__(self):
        return self.sample_size


class ImageDetectionDataset(Dataset):
    def __init__(self, images_data, annotations, annotations_human_labels, target_encoder, train_mode,
                 short_dim, long_dim, image_transform=None):
        super().__init__()
        self.images_data = images_data
        self.annotations = annotations
        self.annotations_human_labels = annotations_human_labels
        self.target_encoder = target_encoder
        self.train_mode = train_mode

        self.short_dim = short_dim
        self.long_dim = long_dim
        self.image_transform = image_transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):

        Xi = self.load_from_disk(index)
        Xi = self.resize_image(Xi)
        if not self.train_mode:
            Xi = self.image_transform(Xi)

        if self.annotations is not None:
            h, w = Xi.size
            yi = self.load_target(index, (h, w))
            return Xi, yi
        else:
            return Xi

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
            x_min = box_row['XMin'] * h
            x_max = box_row['XMax'] * h
            y_min = box_row['YMin'] * w
            y_max = box_row['YMax'] * w
            label = box_row['LabelName']
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
        return torch.FloatTensor(boxes), torch.FloatTensor(labels)

    def resize_image(self, image):
        h, w = image.size
        x, y = min(h, w), max(h, w)     # x < y
        if y*self.short_dim > x*self.long_dim:
            target_x = x * self.long_dim // y
            target_y = self.long_dim
        else:
            target_x = self.short_dim
            target_y = y * self.short_dim // x

        if h > w:
            target_x, target_y = target_y, target_x
        target_x, target_y = target_x // 4 * 4, target_y // 4 * 4

        resize = transforms.Resize((target_x, target_y))

        return resize(image)

    def alling_images(self, images):
        max_h, max_w = 0, 0
        min_h, min_w = 1000, 1000
        for image in images:
            h, w = image.size
            max_h, max_w = max(h, max_h), max(w, max_w)
            min_h, min_w = min(h, min_h), min(w, min_w)

        print('h: [{}, {}], w: [{}, {}]'.format(min_h, max_h, min_w, max_w))

        resize = transforms.Resize((max_h, max_w))
        allinged_images = []
        for image in images:
            # h, w = image.size
            # pad = transforms.Pad((0, 0, max_h - h, max_w - w), fill=255, padding_mode='constant')
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
        input_size = torch.Tensor(list(inputs.size()[-2:]))
        bbox_targets, clf_targets = [], []
        for box, label in zip(boxes, labels):
            bbox_target, clf_target = self.target_encoder.encode(box, label, input_size=input_size)
            bbox_targets.append(bbox_target)
            clf_targets.append(clf_target)

        bbox_targets, clf_targets = torch.stack(bbox_targets), torch.stack(clf_targets)
        clf_targets = clf_targets.unsqueeze(-1)
        targets = torch.cat((bbox_targets, clf_targets), 2)

        # print("inputs: {}, targets: {}".format(inputs.size(), targets.size()))
        return inputs, targets


class ImageDetectionLoader(BaseTransformer):
    def __init__(self, train_mode, loader_params, dataset_params):
        super().__init__()
        self.train_mode = train_mode
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

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
                                                       self.loader_params.training)
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
                                   image_transform=self.image_transform)

            datagen = DataLoader(dataset, **loader_params,
                                 sampler=RandomSubsetSampler(images_data=images_data,
                                                             sample_size=self.dataset_params.sample_size,
                                                             batch_size=self.loader_params.training.batch_size),
                                 collate_fn=dataset.collate_fn)
        else:
            dataset = self.dataset(images_data,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   target_encoder=self.target_encoder,
                                   train_mode=False,
                                   short_dim=self.dataset_params.short_dim,
                                   long_dim=self.dataset_params.long_dim,
                                   image_transform=self.image_transform)

            if annotations is not None:
                datagen = DataLoader(dataset, **loader_params, collate_fn=dataset.collate_fn)
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
