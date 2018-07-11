import os
import shutil

import pandas as pd
from deepsense import neptune
import json
from pycocotools.coco import COCO

from .pipeline_config import SOLUTION_CONFIG, CATEGORY_IDS, SEED, CATEGORY_LAYERS, ID_COLUMN
from .pipelines import PIPELINES
from .utils import init_logger, read_params, set_seed, get_img_ids_from_folder, map_evaluation, create_annotations, \
    generate_data_frame_chunks, NeptuneContext

neptune_ctx = NeptuneContext()
params = neptune_ctx.params
ctx = neptune_ctx.ctx

set_seed(SEED)
logger = init_logger()


class PipelineManager():
    def __init__(self):
        self.logger = init_logger()
        self.seed = SEED
        set_seed(self.seed)
        self.ctx = neptune.Context()
        self.params = read_params(self.ctx, fallback_file='neptune.yaml')

    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode, self.logger, self.params, self.seed)

    def evaluate(self, pipeline_name, dev_mode, chunk_size):
        evaluate(pipeline_name, dev_mode, chunk_size, self.logger, self.params, self.seed, self.ctx)

    def predict(self, pipeline_name, dev_mode, submit_predictions, chunk_size):
        predict(pipeline_name, dev_mode, submit_predictions, chunk_size, self.logger, self.params, self.seed)

    def make_submission(self, submission_filepath):
        make_submission(submission_filepath)


def train(pipeline_name, dev_mode, logger, params, seed):
    logger.info('training')
    if bool(params.overwrite) and os.path.isdir(params.experiment_dir):
        shutil.rmtree(params.experiment_dir)

    nrows = 100 if dev_mode else None
    annotations = pd.read_csv(params.annotations_filepath, nrows=nrows)
    annotations_human_labels = pd.read_csv(params.annotations_human_verification_filepath, nrows=nrows)

    if params.default_valid_ids:
        valid_ids_data = pd.read_csv(params.valid_ids_filepath, nrows=nrows)
        valid_img_ids = valid_ids_data[ID_COLUMN].tolist()
        train_img_ids = list(set(annotations[ID_COLUMN].unique()) - set(valid_img_ids))
    else:
        raise NotImplementedError

    data = {'input': {'img_ids': train_img_ids
                      },
            'metadata': {'annotations': annotations,
                         'annotations_human_labels': annotations_human_labels
                         },
            'callback_input': {'valid_img_ids': valid_img_ids
                               }
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode, chunk_size, logger, params, seed, ctx):
    logger.info('evaluating')

    nrows = 100 if dev_mode else None
    annotations = pd.read_csv(params.annotations_filepath, nrows=nrows)

    if params.default_valid_ids:
        valid_ids_data = pd.read_csv(params.valid_ids_filepath, nrows=nrows)
        valid_img_ids = valid_ids_data[ID_COLUMN].tolist()
    else:
        raise NotImplementedError

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(valid_img_ids,
                                     pipeline, logger, CATEGORY_IDS, chunk_size)

    logger.info('Calculating mean average precision')
    mean_average_precision = map_evaluation(annotations, prediction)
    logger.info('MAP on validation is {}'.format(mean_average_precision))
    ctx.channel_send('MAP', 0, mean_average_precision)


def predict(pipeline_name, dev_mode, submit_predictions, chunk_size, logger, params, seed):
    logger.info('predicting')

    n_ids = 100 if dev_mode else None
    test_img_ids = get_img_ids_from_folder(params.test_imgs_dir, n_ids=n_ids)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(test_img_ids,
                                     pipeline, logger, CATEGORY_IDS, chunk_size)

    submission = prediction
    submission_filepath = os.path.join(params.experiment_dir, 'submission.json')
    with open(submission_filepath, "w") as fp:
        fp.write(json.dumps(submission))
        logger.info('submission saved to {}'.format(submission_filepath))
        logger.info('submission head \n\n{}'.format(submission[0]))

    if submit_predictions:
        make_submission(submission_filepath)


def make_submission(submission_filepath):
    logger.info('Making Kaggle submit...')
    os.system('kaggle competitions submit -c google-ai-open-images-object-detection-track -f {} -m {}'.format(
        submission_filepath, params.kaggle_message))
    logger.info('Kaggle submit completed')


def generate_prediction(img_ids, pipeline, logger, category_ids, chunk_size):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(img_ids, pipeline, logger, category_ids, chunk_size)
    else:
        return _generate_prediction(img_ids, pipeline, logger, category_ids)


def _generate_prediction(img_ids, pipeline, logger, category_ids):
    data = {'input': {'img_ids': img_ids
                      },
            }

    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    prediction = create_annotations(img_ids, y_pred, logger, category_ids, CATEGORY_LAYERS)
    return prediction


def _generate_prediction_in_chunks(img_ids, pipeline, logger, category_ids, chunk_size):
    prediction = []
    for img_ids_chunk in generate_data_frame_chunks(img_ids, chunk_size):
        data = {'input': {'img_ids': img_ids_chunk
                          },

                }

        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        prediction_chunk = create_annotations(img_ids_chunk, y_pred, logger, category_ids, CATEGORY_LAYERS)
        prediction.extend(prediction_chunk)

    return prediction


def _get_scoring_model_data(data_dir, meta, num_training_examples, random_seed):
    annotation_file_path = os.path.join(data_dir, 'train', "annotation.json")
    coco = COCO(annotation_file_path)
    meta = meta.sample(num_training_examples, random_state=random_seed)
    annotations = []
    for image_id in meta['ImageId'].values:
        image_annotations = {}
        for category_id in CATEGORY_IDS:
            annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=category_id)
            category_annotations = coco.loadAnns(annotation_ids)
            image_annotations[category_id] = category_annotations
        annotations.append(image_annotations)
    return meta, annotations
