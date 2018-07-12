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
    def __init__(self, data_size, sample_size):
        self.data_size = data_size
        self.sample_size = sample_size

    def __iter__(self):
        return iter(torch.randperm(self.data_size)[:self.sample_size].long())

    def __len__(self):
        return self.sample_size


class ImageDetectionDataset(Dataset):
    def __init__(self, ids, annotations, annotations_human_labels, images_dir, output_encoder, train_mode):
        super().__init__()
        self.ids = ids
        self.annotations = annotations
        self.annotations_human_labels = annotations_human_labels
        self.images_dir = images_dir
        self.output_encoder = output_encoder
        self.train_mode = train_mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        Xi = self.load_from_disk(index)

        if self.annotations is not None:
            yi = self.load_target(index, Xi.size)
            return Xi, yi
        else:
            return Xi

    def load_from_disk(self, index):
        imgId = self.ids[index]
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
        imgId = self.ids[index]
        boxes_rows = self.annotations.query('ImageID == {}'.format(imgId))
        return self.get_boxes_and_labels(boxes_rows, img_shape)

    def get_boxes_and_labels(self, boxes_rows, img_shape):
        boxes, labels = [], []
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
          batch: (list) of images, bbox_targets, clf_targets.

        Returns:
          images, stacked bbox_targets, stacked clf_targets.
        """
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        inputs = torch.stack(imgs)
        bbox_targets, clf_targets = [], []
        for i in range(len(imgs)):
            bbox_target, clf_target = self.output_encoder.encode(boxes[i], labels[i], input_size=inputs.size()[2:])
            bbox_targets.append(bbox_target)
            clf_targets.append(clf_target)
        return inputs, torch.stack(bbox_targets), torch.stack(clf_targets)


class ImageDetectionLoader(BaseTransformer):
    def __init__(self, train_mode, loader_params, dataset_params):
        super().__init__()
        self.train_mode = train_mode
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.output_encoder = DataEncoder()
        self.dataset = ImageDetectionDataset

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
                                   output_encoder=self.output_encoder,
                                   train_mode=True)
            datagen = DataLoader(dataset, **loader_params,
                                 sampler=RandomSubsetSampler(data_size=len(dataset),
                                                             sample_size=self.loader_params.training.sample_size),
                                 collate_fn=dataset.collate_fn)
        else:
            dataset = self.dataset(ids,
                                   annotations=annotations,
                                   annotations_human_labels=annotations_human_labels,
                                   images_dir=self.dataset_params.images_dir,
                                   output_encoder=self.output_encoder,
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
