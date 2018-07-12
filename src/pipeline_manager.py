import os
import shutil

import pandas as pd
import json
from pycocotools.coco import COCO

from .pipeline_config import SOLUTION_CONFIG, SEED, ID_COLUMN
from .pipelines import PIPELINES
from .utils import init_logger, set_seed, get_img_ids_from_folder, map_evaluation, create_annotations, \
    generate_data_frame_chunks, NeptuneContext

LOGGER = init_logger()
CTX = NeptuneContext()
PARAMS = CTX.params
set_seed(SEED)


class PipelineManager:
    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode, chunk_size):
        evaluate(pipeline_name, dev_mode, chunk_size)

    def predict(self, pipeline_name, dev_mode, submit_predictions, chunk_size):
        predict(pipeline_name, dev_mode, submit_predictions, chunk_size)

    def make_submission(self, submission_filepath):
        make_submission(submission_filepath)


def train(pipeline_name, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.clean_experiment_directory_before_training) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    annotations = pd.read_csv(PARAMS.annotations_filepath)
    annotations_human_labels = pd.read_csv(PARAMS.annotations_human_verification_filepath)

    if PARAMS.default_valid_ids:
        valid_ids_data = pd.read_csv(PARAMS.valid_ids_filepath)
        valid_img_ids = valid_ids_data[ID_COLUMN].tolist()
        train_img_ids = list(set(annotations[ID_COLUMN].unique()) - set(valid_img_ids))
    else:
        raise NotImplementedError

    if dev_mode:
        train_img_ids = train_img_ids[:100]
        valid_img_ids = valid_img_ids[:20]

    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.train_imgs_dir

    data = {'input': {'img_ids': train_img_ids
                      },
            'validation_input': {'valid_img_ids': valid_img_ids,
                                 },
            'metadata': {'annotations': annotations,
                         'annotations_human_labels': annotations_human_labels
                         }
            }

    pipeline = PIPELINES[pipeline_name]['train'](SOLUTION_CONFIG)
    pipeline.clean_cache()
    pipeline.fit_transform(data)
    pipeline.clean_cache()


def evaluate(pipeline_name, dev_mode, chunk_size):
    LOGGER.info('evaluating')

    annotations = pd.read_csv(PARAMS.annotations_filepath)

    if PARAMS.default_valid_ids:
        valid_ids_data = pd.read_csv(PARAMS.valid_ids_filepath)
        valid_img_ids = valid_ids_data[ID_COLUMN].tolist()
    else:
        raise NotImplementedError

    if dev_mode:
        valid_img_ids = valid_img_ids[:20]

    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.train_imgs_dir

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(valid_img_ids, pipeline, chunk_size)

    LOGGER.info('Calculating mean average precision')
    mean_average_precision = map_evaluation(annotations, prediction)
    LOGGER.info('MAP on validation is {}'.format(mean_average_precision))
    CTX.ctx.channel_send('MAP', 0, mean_average_precision)


def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    LOGGER.info('predicting')

    n_ids = 100 if dev_mode else None
    test_img_ids = get_img_ids_from_folder(PARAMS.test_imgs_dir, n_ids=n_ids)

    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.test_imgs_dir

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(test_img_ids, pipeline, chunk_size)

    submission = prediction
    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.json')
    with open(submission_filepath, "w") as fp:
        fp.write(json.dumps(submission))
        LOGGER.info('submission saved to {}'.format(submission_filepath))
        LOGGER.info('submission head \n\n{}'.format(submission[0]))

    if submit_predictions:
        make_submission(submission_filepath)


def make_submission(submission_filepath):
    LOGGER.info('Making Kaggle submit...')
    os.system(
        'kaggle competitions submit -c google-ai-open-images-object-detection-track -f {}'.format(submission_filepath))
    LOGGER.info('Kaggle submit completed')


def generate_prediction(img_ids, pipeline, chunk_size):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(img_ids, pipeline, chunk_size)
    else:
        return _generate_prediction(img_ids, pipeline)


def _generate_prediction(img_ids, pipeline):
    data = {'input': {'img_ids': img_ids,
                      'image_size': (256, 256)
                      },
            }

    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    y_pred = output['y_pred']

    prediction = create_annotations(img_ids, y_pred)
    return prediction


def _generate_prediction_in_chunks(img_ids, pipeline, chunk_size):
    prediction = []
    for img_ids_chunk in generate_data_frame_chunks(img_ids, chunk_size):
        data = {'input': {'img_ids': img_ids_chunk,
                          'image_size': (256, 256)
                          },

                }

        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        y_pred = output['y_pred']

        prediction_chunk = create_annotations(img_ids_chunk, y_pred)
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
