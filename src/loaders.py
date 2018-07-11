import os
import torch
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image
from steppy.base import BaseTransformer

from .retinanet import DataEncoder


class RandomSubsetSampler(Sampler):
    def __init__(self, data_size, subset_size):
        super().__init__()
        self.data_size = data_size
        self.subset_size = subset_size

    def __iter__(self):
        return iter(torch.randperm(self.data_size)[:self.subset_size].long())

    def __len__(self):
        return self.subset_size


class ImageDetectionDataset(Dataset):
    def __init__(self, X, y, labels, images_dir, encoder, train_mode):
        super().__init__()
        self.X = X
        if y is not None:
            self.y = y
            self.labels = labels
        else:
            self.y = None
            self.labels = None
        self.images_dir = images_dir
        self.encoder = encoder
        self.train_mode = train_mode

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):

        Xi = self.load_from_disk(index)

        if self.y is not None:
            Bi, Li = self.load_target(index, Xi.size)
            return Xi, Bi, Li
        else:
            return Xi

    def load_from_disk(self, index):
        imgId = self.X[index]
        img_path = os.path.join(self.images_dir, imgId + '.png')
        return self.load_from_disk(img_path)

    def load_image(self, img_filepath, grayscale=False):
        image = Image.open(img_filepath, 'r')
        if not grayscale:
            image = image.convert('RGB')
        else:
            image = image.convert('L')
        return image

    def load_target(self, index, img_shape):
        imgId = self.X[index]
        boxes_rows = self.y[self.y['ImageID'] == imgId]
        return self.get_boxes_and_labels(boxes_rows, img_shape)

    def get_boxes_and_labels(self, boxes_rows, img_shape):
        boxes = []
        labels = []
        h, w = img_shape
        for box_row in boxes_rows:
            x_min = box_row['XMin'].values[0] * h
            x_max = box_row['XMax'].values[0] * h
            y_min = box_row['YMin'].values[0] * w
            y_max = box_row['YMax'].values[0] * w
            label = box_row['LabelName'].values[0]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)
        return torch.FloatTensor(boxes), torch.LongTensor(labels)

    def collate_fn(self, batch):
        """Encode targets.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          images, stacked cls_targets, stacked loc_targets.
        """
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        inputs = torch.stack(imgs)
        loc_targets = []
        cls_targets = []
        for i in range(len(imgs)):
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=inputs.size()[2:])
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)


class ImageDetectionLoader(BaseTransformer):
    def __init__(self, labels, loader_params, dataset_params):
        super().__init__()
        self.labels = labels
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.encoder = DataEncoder()
        self.dataset = ImageDetectionDataset

    def transform(self, X, y, X_valid=None, y_valid=None, train_mode=True):
        if train_mode and y is not None:
            flow, steps = self.get_datagen(X, y, True, self.loader_params.training)
        else:
            flow, steps = self.get_datagen(X, None, False, self.loader_params.inference)

        if X_valid is not None and y_valid is not None:
            valid_flow, valid_steps = self.get_datagen(X_valid, y_valid, False, self.loader_params.inference)
        else:
            valid_flow = None
            valid_steps = None
        return {'datagen': (flow, steps),
                'validation_datagen': (valid_flow, valid_steps)}

    def get_datagen(self, X, y, train_mode, loader_params):
        if train_mode:
            dataset = self.dataset(X, y,
                                   labels=self.labels,
                                   images_dir=self.dataset_params.images_dir,
                                   encoder=self.encoder,
                                   train_mode=True)

            datagen = DataLoader(dataset, **loader_params,
                                 sampler=RandomSubsetSampler(len(dataset), self.loader_params.training.subset_size),
                                 collate_fn=dataset.collate_fn)
        else:
            dataset = self.dataset(X, y,
                                   labels=self.labels,
                                   images_dir=self.dataset_params.images_dir,
                                   encoder=self.encoder,
                                   train_mode=False)

        datagen = DataLoader(dataset, **loader_params,
                             collate_fn=dataset.collate_fn)
        steps = len(datagen)
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)
