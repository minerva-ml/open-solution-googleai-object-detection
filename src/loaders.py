import os
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
from .logging import LOGGER


class BaseSampler(Sampler):
    def __init__(self, images_data, *args, sample_size=None, shuffle=True,
                 even_class_sampling=False, annotations=None, seed=None, **kwargs):
        self.shuffle = shuffle
        self.even_class_sampling = even_class_sampling
        self.seed = seed
        self.annotations = annotations
        self.images_data = self._prepare_data(images_data)
        self.sample_size = len(self.images_data) if sample_size is None else min(sample_size, len(self.images_data))
        assert self.even_class_sampling is False or self.annotations is not None,\
            'Annotations are required for class sampling'

        if self.even_class_sampling:
            self.class_labels = self.annotations['LabelName'].unique()
            self.num_classes = len(self.class_labels)
            division = self.sample_size / float(self.num_classes)
            self.samples_per_class = [round(division * (i + 1)) - round(division * i) for i in range(self.num_classes)]

            self.class_indices = {}
            for class_label in self.class_labels:
                imgIds = self.annotations[self.annotations['LabelName'] == class_label]['ImageID'].unique()
                indices = self.images_data[self.images_data['ImageID'].isin(imgIds)].index.tolist()
                self.class_indices[class_label] = indices

    def __iter__(self):
        return iter(self._get_indices(self._get_sample()))

    def __len__(self):
        return self.sample_size

    def _get_sample(self):
        np.random.seed(self.seed)

        if self.even_class_sampling:
            sample = []
            for class_label, sample_size in zip(self.class_labels, self.samples_per_class):
                indices = self.class_indices[class_label]
                if len(indices) == 0:
                    LOGGER.info('No instances from class {}'.format(class_label))
                    continue
                if len(indices) < sample_size:
                    LOGGER.info('Not enough {} class instances to get {} samples, got {} instead'\
                          .format(class_label, sample_size, len(indices)))
                sample.append(np.random.choice(indices, min(sample_size, len(indices)), replace=False))

            sample = np.concatenate(sample)
            np.random.shuffle(sample)
            return sample
        else:
            return np.random.choice(len(self.images_data), self.sample_size, replace=False)

    def _prepare_data(self, data):
        raise NotImplementedError

    def _get_indices(self, sample):
        raise NotImplementedError


class FixedSizeSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_data(self, data):
        data = data.sample(frac=1, random_state=SEED).reset_index(drop=True) if self.shuffle\
            else data.reset_index(drop=True)
        return data

    def _get_indices(self, sample):
        return sample if self.shuffle else np.sort(sample)


class AspectRatioSampler(BaseSampler):
    def __init__(self, *args, batch_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _prepare_data(self, data):
        data = data.sort_values('aspect_ratio').reset_index(drop=True)
        return data

    def _get_indices(self, sample):
        indices = np.sort(sample)
        if self.shuffle:
            indices = self._shuffle_batches(indices)
        return indices

    def _shuffle_batches(self, indices):
        num_batches = len(indices) // self.batch_size
        split_point = num_batches * self.batch_size
        head, tail = indices[:split_point], indices[split_point:]
        head = head.reshape(num_batches, self.batch_size)
        np.random.seed(self.seed)
        np.random.shuffle(head)
        head = head.flatten()
        indices = np.concatenate((head, tail))
        return indices


class ImageDetectionDataset(Dataset):
    def __init__(self, images_data, images_dir, annotations, annotations_human_labels, target_encoder, train_mode,
                 short_dim, long_dim, fixed_h, fixed_w, sampler_name, image_transform):
        super().__init__()
        self.images_data = images_data
        self.images_dir = images_dir
        self.annotations = annotations
        self.annotations_human_labels = annotations_human_labels
        self.target_encoder = target_encoder
        self.train_mode = train_mode

        self.short_dim = short_dim
        self.long_dim = long_dim
        self.fixed_h = fixed_h
        self.fixed_w = fixed_w
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
        imgId = self.images_data.iloc[index]['ImageID']
        img_path = os.path.join(self.images_dir, imgId + '.jpg')
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
            w, h = self.fixed_w, self.fixed_h
        else:
            org_w, org_h = image.size
            w, h = get_target_size(aspect_ratio=org_w/float(org_h), short_dim=self.short_dim, long_dim=self.long_dim)
        resize = transforms.Resize((h, w))
        return resize(image)

    def align_images(self, images):
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

        imgs = self.align_images(imgs)
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
                                   images_dir=self.dataset_params.images_dir,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   target_encoder=self.target_encoder,
                                   train_mode=True,
                                   short_dim=self.dataset_params.short_dim,
                                   long_dim=self.dataset_params.long_dim,
                                   fixed_h=self.dataset_params.fixed_h,
                                   fixed_w=self.dataset_params.fixed_w,
                                   sampler_name=self.dataset_params.sampler_name,
                                   image_transform=self.image_transform)

            datagen = DataLoader(dataset, **loader_params,
                                 sampler=self.sampler(images_data=images_data,
                                                      annotations=annotations,
                                                      even_class_sampling=self.dataset_params.even_class_sampling,
                                                      sample_size=self.dataset_params.sample_size,
                                                      batch_size=loader_params.batch_size,
                                                      shuffle=True),
                                 collate_fn=dataset.collate_fn)
        else:
            dataset = self.dataset(images_data,
                                   images_dir=self.dataset_params.images_dir,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   target_encoder=self.target_encoder,
                                   train_mode=False,
                                   short_dim=self.dataset_params.short_dim,
                                   long_dim=self.dataset_params.long_dim,
                                   fixed_h=self.dataset_params.fixed_h,
                                   fixed_w=self.dataset_params.fixed_w,
                                   sampler_name=self.dataset_params.sampler_name,
                                   image_transform=self.image_transform)

            if annotations is not None:
                datagen = DataLoader(dataset, **loader_params,
                                     sampler=self.sampler(images_data=images_data,
                                                          annotations=annotations,
                                                          even_class_sampling=True,
                                                          sample_size=self.dataset_params.valid_sample_size,
                                                          batch_size=loader_params.batch_size,
                                                          shuffle=False,
                                                          seed=SEED),
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
