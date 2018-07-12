import multiprocessing as mp

import numpy as np
from skimage.transform import resize
from skimage.morphology import erosion, dilation, rectangle
import pandas as pd
import torch

from .steps.base import BaseTransformer


class SubmissionProducer(BaseTransformer):

    def transform(self, image_ids, results, target_sizes, decoder_dict):
        self.decoder_dict = decoder_dict
        prediction_strings = []
        for (bboxes, labels), target_size in zip(results, target_sizes):
            prediction_strings.append(self.get_prediction_string(bboxes, labels, target_size))
        submission = pd.DataFrame({'ImageId': image_ids, 'PredictionString': prediction_strings})
        return {'submission': submission}

    def get_prediction_string(self, bboxes, labels, size):
        prediction_list = []
        for bbox, label in zip(bboxes, labels):
            prediction_list.append(self.get_class_id(label))
            prediction_list.extend(self.get_bbox_relative(bbox, size))
        prediction_string = " ".join(prediction_list)
        return prediction_string


    def get_class_id(self, label):
        return self.decoder_dict[label]


    def get_bbox_relative(self, bbox, size):
        x = size[0]
        y = size[1]
        x_min = bbox[0]/x
        x_max = bbox[1]/x
        y_min = bbox[2]/y
        y_max = bbox[3]/y
        #print(x, y, bbox)#test
        result = [x_min, x_max, y_min, y_max]
        return [str(r) for r in result]


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.
    Returns:
      keep: (tensor) selected indices.
    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
      https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)