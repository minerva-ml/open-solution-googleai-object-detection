import os
import torch
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from PIL import Image
from steppy.base import BaseTransformer
import numpy as np

from .retinanet import DataEncoder
from .pipeline_config import MEAN, STD


class RandomSubsetSampler(Sampler):
    def __init__(self, data_size, sample_size):
        self.data_size = data_size
        self.sample_size = sample_size

    def __iter__(self):
        return iter(torch.randperm(self.data_size)[:self.sample_size].long())

    def __len__(self):
        return self.sample_size


class ImageDetectionDataset(Dataset):
    def __init__(self, ids, annotations, annotations_human_labels, images_dir, target_encoder, train_mode, num_classes=20,
                 image_transform=None):
        super().__init__()
        self.ids = ids
        self.annotations = annotations
        self.annotations_human_labels = annotations_human_labels
        self.images_dir = images_dir
        self.target_encoder = target_encoder
        self.train_mode = train_mode

        self.num_classes = num_classes
        self.image_transform = image_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        Xi = self.load_from_disk(index)

        Xi = self.image_transform(Xi)

        if self.annotations is not None:
            _, h, w = Xi.size()
            yi = self.load_target(index, (h, w))
            return Xi, yi
        else:
            return Xi

    def load_from_disk(self, index):
        imgId = self.ids[index]
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
        imgId = self.ids[index]
        boxes_rows = self.annotations.query('ImageID == @imgId')
        human_labels = self.annotations_human_labels.query('ImageID == @imgId')
        return self.get_boxes_and_labels(boxes_rows, img_shape) + self.get_human_labels(human_labels)

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

    def get_human_labels(self, annotations):
        labels = list(annotations['LabelName'].values)
        labels_ = torch.zeros(self.num_classes)
        labels_[[int(label) for label in labels]] = 1
        return labels_,

    def collate_fn(self, batch):
        """Encode targets.

        Args:
          batch: (list) of images, (bbox_targets, clf_targets, human_labels).

        Returns:
          images, stacked bbox_targets, stacked clf_targets.
        """
        imgs = [x[0] for x in batch]
        boxes = [x[1][0] for x in batch]
        labels = [x[1][1] for x in batch]
        human_labels = [x[1][2] for x in batch]

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

        targets = self.join_target_with_labels(targets, human_labels)
        return inputs, targets

    def join_target_with_labels(self, targets, human_labels):
        human_labels = torch.stack(human_labels)
        human_labels = human_labels.unsqueeze(-1)
        human_labels = torch.cat([human_labels,]*targets.size(-1), dim=-1)
        targets = torch.cat([targets, human_labels], dim=1)
        return targets


class ImageDetectionLoader(BaseTransformer):
    def __init__(self, train_mode, loader_params, dataset_params):
        super().__init__()
        self.train_mode = train_mode
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.target_encoder = DataEncoder()
        self.dataset = ImageDetectionDataset

        self.image_transform = transforms.Compose([transforms.Resize((self.dataset_params.h, self.dataset_params.w)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=MEAN, std=STD),
                                                   ])

    def transform(self, ids, annotations=None, annotations_human_labels=None, valid_ids=None):
        if self.train_mode and annotations is not None:
            flow, steps = self.get_datagen(ids, annotations, annotations_human_labels, True,
                                           self.loader_params.training)
        else:
            flow, steps = self.get_datagen(ids, None, None, False, self.loader_params.training)

        if valid_ids is not None:
            valid_flow, valid_steps = self.get_datagen(valid_ids, annotations, annotations_human_labels, False,
                                                       self.loader_params.training)
        else:
            valid_flow = None
            valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, ids, annotations, annotations_human_labels, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(ids,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   images_dir=self.dataset_params.images_dir,
                                   target_encoder=self.target_encoder,
                                   train_mode=True,
                                   num_classes=self.dataset_params.num_classes,
                                   image_transform=self.image_transform)
            datagen = DataLoader(dataset, **loader_params,
                                 sampler=RandomSubsetSampler(data_size=len(dataset),
                                                             sample_size=self.dataset_params.sample_size),
                                 collate_fn=dataset.collate_fn)
        else:
            dataset = self.dataset(ids,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   images_dir=self.dataset_params.images_dir,
                                   target_encoder=self.target_encoder,
                                   train_mode=False,
                                   num_classes=self.dataset_params.num_classes,
                                   image_transform=self.image_transform)

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
