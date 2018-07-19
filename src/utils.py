import glob
import logging
import math
import os
import pathlib
import random
import subprocess
import sys
from collections import Iterable
from itertools import chain

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from attrdict import AttrDict
from deepsense import neptune
from pycocotools import mask as cocomask
from steppy.base import BaseTransformer
from tqdm import tqdm

neptune_config_path = str(pathlib.Path(__file__).resolve().parents[1] / 'configs' / 'neptune_config_local.yaml')


# Alex Martelli's 'Borg'
# http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class _Borg:
    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state


class NeptuneContext(_Borg):
    def __init__(self, fallback_file=neptune_config_path):
        _Borg.__init__(self)

        self.ctx = neptune.Context()
        self.fallback_file = fallback_file
        self.params = self._read_params()
        self.numeric_channel = neptune.ChannelType.NUMERIC
        self.image_channel = neptune.ChannelType.IMAGE
        self.text_channel = neptune.ChannelType.TEXT

    def channel_send(self, *args, **kwargs):
        self.ctx.channel_send(*args, **kwargs)

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


def parameter_eval(param):
    try:
        return eval(param)
    except Exception:
        return param


def get_img_ids_from_folder(dirpath, n_ids=None):
    ids = []
    for i, filepath in enumerate(sorted(glob.glob('{}/*'.format(dirpath)))):
        idx = os.path.basename(filepath).split('.')[0]
        ids.append(idx)

        if n_ids == i:
            break
        # print(filepath, idx)
        # exit()
    return ids


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


def generate_list_chunks(meta, chunk_size):
    n_rows = len(meta)
    chunk_nr = int(math.ceil(n_rows / chunk_size))
    for i in tqdm(range(chunk_nr)):
        meta_chunk = meta[i * chunk_size:(i + 1) * chunk_size]
        yield meta_chunk


def denormalize_img(image, mean, std):
    return image * np.array(std).reshape(3, 1, 1) + np.array(mean).reshape(3, 1, 1)


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


def competition_metric_evaluation(annotation_filepath,
                                  annotations_human_labels_filepath,
                                  prediction_filepath,
                                  label_hierarchy_filepath,
                                  metrics_filepath,
                                  list_of_desired_classes=None,
                                  mappings_filepath=None
                                  ):
    expanded_annotations_filepath = annotation_filepath.replace('.csv', '_expanded.csv')
    expand_annotations_cmd = ["python src/object_detection/dataset_tools/oid_hierarchical_labels_expansion.py",
                              "--json_hierarchy_file={}".format(label_hierarchy_filepath),
                              "--input_annotations={}".format(annotation_filepath),
                              "--output_annotations={}".format(expanded_annotations_filepath),
                              "--annotation_type=1"]
    expand_annotations_cmd = ' '.join(expand_annotations_cmd)

    expanded_annotations_human_labels_filepath = annotations_human_labels_filepath.replace('.csv', '_expanded.csv')
    expand_annotations_human_labels_cmd = [
        "python src/object_detection/dataset_tools/oid_hierarchical_labels_expansion.py",
        "--json_hierarchy_file={}".format(label_hierarchy_filepath),
        "--input_annotations={}".format(annotations_human_labels_filepath),
        "--output_annotations={}".format(
            expanded_annotations_human_labels_filepath),
        "--annotation_type=2"]
    expand_annotations_human_labels_cmd = ' '.join(expand_annotations_human_labels_cmd)

    run_evaluation_cmd = ["python src/object_detection/metrics/oid_od_challenge_evaluation.py",
                          "--input_annotations_boxes={}".format(expanded_annotations_filepath),
                          "--input_annotations_labels={}".format(expanded_annotations_human_labels_filepath),
                          "--input_class_labelmap=src/object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt",
                          "--input_predictions={}".format(prediction_filepath),
                          "--output_metrics={}".format(metrics_filepath)]
    run_evaluation_cmd = ' '.join(run_evaluation_cmd)

    subprocess.call(expand_annotations_cmd, shell=True)
    subprocess.call(expand_annotations_human_labels_cmd, shell=True)
    subprocess.call(run_evaluation_cmd, shell=True)

    map_score = calculate_map(metrics_filepath, list_of_desired_classes, mappings_filepath)
    return map_score


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def submission_formatting(submission):
    prediction_formatted = {}
    for i, row in submission.iterrows():
        image_id = row.ImageId
        prediction = row.PredictionString
        for pred in chunker(prediction.split(), 6):
            label, score, x_min, y_min, x_max, y_max = pred
            prediction_formatted.setdefault('ImageID', []).append(image_id)
            prediction_formatted.setdefault('LabelName', []).append(label)
            prediction_formatted.setdefault('XMin', []).append(x_min)
            prediction_formatted.setdefault('YMin', []).append(y_min)
            prediction_formatted.setdefault('XMax', []).append(x_max)
            prediction_formatted.setdefault('YMax', []).append(y_max)
            prediction_formatted.setdefault('Score', []).append(score)
    prediction_formatted = pd.DataFrame(prediction_formatted)
    return prediction_formatted


def calculate_map(metrics_filepath, list_of_desired_classes=None, mappings_file=None):
    with open(metrics_filepath) as f:
        metrics = f.read().splitlines()

    if list_of_desired_classes:
        codes2names, names2codes = get_class_mappings(mappings_file)
        if not all([cls.startswith('/') for cls in list_of_desired_classes]):
            list_of_desired_classes = [names2codes.get(cls_name, 'notfound')
                                       for cls_name in list_of_desired_classes]

        assert all(
            [cls_code in codes2names for cls_code in list_of_desired_classes]), "One or More Class names/codes are " \
                                                                                "invalid "

    label_scores = []
    for label_score in metrics:
        score = float(label_score.split(',')[1])
        score = 0.0 if np.isnan(score) else score

        if list_of_desired_classes:
            label = label_score.split(',')[0].split('AP@0.5IOU/')[-1]
            if label in list_of_desired_classes:
                label_scores.append(score)
        else:
            label_scores.append(score)
    return np.mean(label_scores)


def get_class_mappings(mappings_file):
    codes2names = pd.read_csv(mappings_file, header=None).set_index(0).to_dict()[1]
    names2codes = {v: k for k, v in codes2names.items()}
    return codes2names, names2codes


def reduce_number_of_classes(annotations_df, list_of_desired_classes, mappings_file):
    """
    Filters a dataframe based on provided classes

    Args:
        annotations_df: Loaded annotation file (pd.DataFrame)
        list_of_desired_classes: List of codes or names of Open Images v4 classes
        mappings_file: codes to names csv

    Returns:

    """

    codes2names, names2codes = get_class_mappings(mappings_file)
    if not all([cls.startswith('/') for cls in list_of_desired_classes]):
        list_of_desired_classes = [names2codes.get(cls_name, 'notfound') for cls_name in list_of_desired_classes]

    assert all([cls_code in codes2names for cls_code in list_of_desired_classes]), "One or More Class names/codes are " \
                                                                                   "invalid "
    subset_df = annotations_df[annotations_df.LabelName.isin(list_of_desired_classes)]

    assert not subset_df.empty, "There is not enough data left after filtering for {} classes. This can happen when a " \
                                "small sample is used"

    return subset_df.reset_index(drop=True)


def add_missing_image_ids(submission, sample_submission):
    submission['ImageId'] = submission['ImageId'].astype(str)
    sample_submission['ImageId'] = sample_submission['ImageId'].astype(str)
    fixed_submission = pd.merge(sample_submission[['ImageId']], submission, on=['ImageId'], how='outer')
    return fixed_submission
