import json
import logging
import math
import os
import ntpath
import random
import sys
import time
from itertools import chain
from collections import defaultdict, Iterable

import glob
from deepsense import neptune
import numpy as np
import pandas as pd
import torch
import yaml
import imgaug as ia
from PIL import Image
from attrdict import AttrDict
from pycocotools import mask as cocomask
from pycocotools.coco import COCO
from tqdm import tqdm
from scipy import ndimage as ndi
from .cocoeval import COCOeval
from .steppy.base import BaseTransformer


# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class NeptuneContext(_Borg):
    def __init__(self, fallback_file='configs/neptune_local.yaml'):
        _Borg.__init__(self)

        self.ctx = neptune.Context()
        self.fallback_file = fallback_file
        self.params = self._read_params()
        self.numeric_channel = neptune.ChannelType.NUMERIC
        self.image_channel = neptune.ChannelType.IMAGE
        self.text_channel = neptune.ChannelType.TEXT

    def _read_params(self):
        if self.ctx.params.__class__.__name__ == 'OfflineContextParams':
            params = self._read_yaml().parameters
        else:
            params = self.ctx.params
        return params

    def _read_yaml(self):
        with open(self.fallback_file) as f:
            config = yaml.load(f)
        return AttrDict(config)


def read_yaml(filepath):
    with open(filepath) as f:
        config = yaml.load(f)
    return AttrDict(config)


def init_logger():
    logger = logging.getLogger('google-ai-odt')
    logger.setLevel(logging.INFO)
    message_format = logging.Formatter(fmt='%(asctime)s %(name)s >>> %(message)s',
                                       datefmt='%Y-%m-%d %H-%M-%S')

    # console handler for validation info
    ch_va = logging.StreamHandler(sys.stdout)
    ch_va.setLevel(logging.INFO)

    ch_va.setFormatter(fmt=message_format)

    # add the handlers to the logger
    logger.addHandler(ch_va)

    return logger


def get_logger():
    return logging.getLogger('google-ai-odt')


def get_img_ids_from_folder(dirpath, n_ids=None):
    ids = []
    for i, filepath in enumerate(glob.glob('{}/*'.format(dirpath))):
        idx = os.path.basename(filepath).split('.')[0]
        ids.append(idx)

        if n_ids == i:
            break
    return ids


def create_annotations(meta, predictions, logger, category_ids, category_layers, save=False, experiment_dir='./'):
    """

    Args:
        meta: pd.DataFrame with metadata
        predictions: list of labeled masks or numpy array of size [n_images, im_height, im_width]
        logger: logging object
        category_ids: list with ids of categories,
            e.g. [None, 100] means, that no annotations will be created from category 0 data, and annotations
            from category 1 will be created with category_id=100
        category_layers:
        save: True, if one want to save submission, False if one want to return it
        experiment_dir: directory of experiment to save annotations, relevant if save==True

    Returns: submission if save==False else True

    """
    annotations = []
    logger.info('Creating annotations')
    category_layers_inds = np.cumsum(category_layers)
    for image_id, (prediction, image_scores) in zip(meta["ImageId"].values, predictions):
        for category_ind, (category_instances, category_scores) in enumerate(zip(prediction, image_scores)):
            category_nr = np.searchsorted(category_layers_inds, category_ind, side='right')
            if category_ids[category_nr] != None:
                masks = decompose(category_instances)
                for mask_nr, (mask, score) in enumerate(zip(masks, category_scores)):
                    annotation = {}
                    annotation["image_id"] = int(image_id)
                    annotation["category_id"] = category_ids[category_nr]
                    annotation["score"] = score
                    annotation["segmentation"] = rle_from_binary(mask.astype('uint8'))
                    annotation['segmentation']['counts'] = annotation['segmentation']['counts'].decode("UTF-8")
                    annotation["bbox"] = bounding_box_from_rle(rle_from_binary(mask.astype('uint8')))
                    annotations.append(annotation)
    if save:
        submission_filepath = os.path.join(experiment_dir, 'submission.json')
        with open(submission_filepath, "w") as fp:
            fp.write(str(json.dumps(annotations)))
            logger.info("Submission saved to {}".format(submission_filepath))
            logger.info('submission head \n\n{}'.format(annotations[0]))
        return True
    else:
        return annotations


def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def bounding_box_from_rle(rle):
    return list(cocomask.toBbox(rle))


def read_params(ctx, fallback_file):
    if ctx.params.__class__.__name__ == 'OfflineContextParams':
        neptune_config = read_yaml(fallback_file)
        params = neptune_config.parameters
    else:
        params = ctx.params
    return params


def softmax(X, theta=1.0, axis=None):
    """
    https://nolanbconaway.github.io/blog/2017/softmax-numpy
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def from_pil(*images):
    images = [np.array(image) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def to_pil(*images):
    images = [Image.fromarray((image).astype(np.uint8)) for image in images]
    if len(images) == 1:
        return images[0]
    else:
        return images


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_data_frame_chunks(meta, chunk_size):
    n_rows = meta.shape[0]
    chunk_nr = math.ceil(n_rows / chunk_size)
    for i in tqdm(range(chunk_nr)):
        meta_chunk = meta.iloc[i * chunk_size:(i + 1) * chunk_size]
        yield meta_chunk


def map_evaluation(ground_truth, prediction):
    return NotImplementedError


# def coco_evaluation(gt_filepath, prediction_filepath, image_ids, category_ids, small_annotations_size):
#     coco = COCO(gt_filepath)
#     coco_results = coco.loadRes(prediction_filepath)
#     cocoEval = COCOeval(coco, coco_results)
#     cocoEval.params.imgIds = image_ids
#     cocoEval.params.catIds = category_ids
#     cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, small_annotations_size ** 2],
#                                [small_annotations_size ** 2, 1e5 ** 2]]
#     cocoEval.params.areaRngLbl = ['all', 'small', 'large']
#     cocoEval.evaluate()
#     cocoEval.accumulate()
#     cocoEval.summarize()
#
#     return cocoEval.stats[0], cocoEval.stats[3]


def make_apply_transformer(func, output_name='output', apply_on=None):
    class StaticApplyTransformer(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)

            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            output = []
            for func_args in tqdm(iterator, total=self.get_arg_length(*args, **kwargs)):
                output.append(func(*func_args))
            return {output_name: output}

        @staticmethod
        def check_input(*args, **kwargs):
            if len(args) and len(kwargs) == 0:
                raise Exception('Input must not be empty')

            arg_length = None
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')
                arg_length_loc = None
                try:
                    arg_length_loc = len(arg)
                except:
                    pass
                if arg_length_loc is not None:
                    if arg_length is None:
                        arg_length = arg_length_loc
                    elif arg_length_loc != arg_length:
                        raise Exception('All inputs must be the same length')

        @staticmethod
        def get_arg_length(*args, **kwargs):
            arg_length = None
            for arg in chain(args, kwargs.values()):
                if arg_length is None:
                    try:
                        arg_length = len(arg)
                    except:
                        pass
                if arg_length is not None:
                    return arg_length

    return StaticApplyTransformer()


def make_apply_transformer_stream(func, output_name='output', apply_on=None):
    class StaticApplyTransformerStream(BaseTransformer):
        def transform(self, *args, **kwargs):
            self.check_input(*args, **kwargs)
            return {output_name: self._transform(*args, **kwargs)}

        def _transform(self, *args, **kwargs):
            if not apply_on:
                iterator = zip(*args, *kwargs.values())
            else:
                iterator = zip(*args, *[kwargs[key] for key in apply_on])

            for func_args in tqdm(iterator):
                yield func(*func_args)

        @staticmethod
        def check_input(*args, **kwargs):
            for arg in chain(args, kwargs.values()):
                if not isinstance(arg, Iterable):
                    raise Exception('All inputs must be iterable')

    return StaticApplyTransformerStream()

