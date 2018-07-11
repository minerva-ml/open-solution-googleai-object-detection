import os
from math import sqrt
import torch
from attrdict import AttrDict
from sklearn.externals import joblib
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image

from .steps.base import BaseTransformer
from .utils import meshgrid, box_iou, box_nms, change_box_order


class RandomSubsetSampler(Sampler):
    def __init__(self, data_size, subset_size):
        super().__init__()
        self.data_size = data_size
        self.subset_size = subset_size

    def __iter__(self):
        return iter(torch.randperm(self.data_size)[:self.subset_size].long())

    def __len__(self):
        return self.subset_size


class DetectionDataset(Dataset):
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
        '''Encode targets.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          images, stacked cls_targets, stacked loc_targets.
        '''
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


class ImageSegmentationLoaderBasic(BaseTransformer):
    def __init__(self, labels, loader_params, dataset_params):
        super().__init__()
        self.labels = labels
        self.loader_params = AttrDict(loader_params)
        self.dataset_params = AttrDict(dataset_params)

        self.encoder = DataEncoder()
        self.dataset = None

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
        else:
            dataset = self.dataset(X, y,
                                   labels=self.labels,
                                   images_dir=self.dataset_params.images_dir,
                                   encoder=self.encoder,
                                   train_mode=False)

        datagen = DataLoader(dataset, **loader_params, sampler=RandomSubsetSampler(len(dataset), 10))
        steps = len(datagen)
        return datagen, steps

    def load(self, filepath):
        params = joblib.load(filepath)
        self.loader_params = params['loader_params']
        return self

    def save(self, filepath):
        params = {'loader_params': self.loader_params}
        joblib.dump(params, filepath)


class DataEncoder(object):
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_hw()

    def _get_anchor_hw(self):
        '''Compute anchor height and width for each feature map.

        Returns:
          anchor_hw: (tensor) anchor hw, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_hw = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_hw.append([anchor_h, anchor_w])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_hw).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (h, w).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_h, fm_w = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_h,fm_w) + 0.5  # [fm_h*fm_w, 2]
            xy = (xy*grid_size).view(fm_w,fm_h,1,2).expand(fm_w,fm_h,9,2)
            hw = self.anchor_wh[i].view(1,1,9,2).expand(fm_w,fm_h,9,2)
            box = torch.cat([xy,hw], 3)  # [x,y,w,h]
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
        cls_targets[ignore] = -1  # for now just mark ignored to -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = 0.5
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

        score, labels = cls_preds.sigmoid().max(1)          # [#anchors,]
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()             # [#obj,]
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep]
