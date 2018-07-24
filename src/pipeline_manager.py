import os
import shutil

import pandas as pd
import numpy as np
import math
import cv2
from PIL import Image

from deepsense import neptune
from .pipeline_config import DESIRED_CLASS_SUBSET, ID_COLUMN, SEED, SOLUTION_CONFIG
from .pipelines import PIPELINES
from .utils import NeptuneContext, competition_metric_evaluation, generate_list_chunks, get_img_ids_from_folder, \
    init_logger, reduce_number_of_classes, set_seed, submission_formatting, add_missing_image_ids

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

    def visualize(self, pipeline_name, image_dir=None, single_image=None, n_files=16, show_popups=False):
        visualize(pipeline_name=pipeline_name, image_dir=image_dir, single_image=single_image, n_files=n_files,
                  show_popups=show_popups)


def train(pipeline_name, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.clean_experiment_directory_before_training) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)

    annotations = pd.read_csv(PARAMS.annotations_filepath)
    annotations_human_labels = pd.read_csv(PARAMS.annotations_human_labels_filepath)
    valid_ids_data = pd.read_csv(PARAMS.valid_ids_filepath)

    if DESIRED_CLASS_SUBSET:
        LOGGER.info("Training on a reduced class subset: {}".format(DESIRED_CLASS_SUBSET))
        annotations = reduce_number_of_classes(annotations,
                                               DESIRED_CLASS_SUBSET,
                                               PARAMS.class_mappings_filepath)

        annotations_human_labels = reduce_number_of_classes(annotations_human_labels,
                                                            DESIRED_CLASS_SUBSET,
                                                            PARAMS.class_mappings_filepath)

        img_ids_in_reduced_annotations = annotations[ID_COLUMN].unique()
        valid_ids_data = valid_ids_data[valid_ids_data[ID_COLUMN].isin(img_ids_in_reduced_annotations)].reset_index(
            drop=True)

    if PARAMS.default_valid_ids:
        if valid_ids_data.shape[0] < PARAMS.validation_sample_size:
            LOGGER.warning("Validation sample-size is smaller then desired validation safmple size ... clipping")
            PARAMS.validation_sample_size = np.clip(PARAMS.validation_sample_size,
                                                    a_max=valid_ids_data.shape[0], a_min=0)

        valid_ids_data = valid_ids_data.sample(PARAMS.validation_sample_size, random_state=SEED)
        valid_img_ids = valid_ids_data[ID_COLUMN].tolist()
        train_img_ids = list(set(annotations[ID_COLUMN].values) - set(valid_img_ids))
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
    annotations_human_labels = pd.read_csv(PARAMS.annotations_human_labels_filepath)

    if DESIRED_CLASS_SUBSET:
        LOGGER.info("Evaluating on a reduced class subset: {}".format(DESIRED_CLASS_SUBSET))
        annotations = reduce_number_of_classes(annotations,
                                               DESIRED_CLASS_SUBSET,
                                               PARAMS.class_mappings_filepath)

        annotations_human_labels = reduce_number_of_classes(annotations_human_labels,
                                                            DESIRED_CLASS_SUBSET,
                                                            PARAMS.class_mappings_filepath)

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

    LOGGER.info('Formatting prediction')
    prediction = submission_formatting(prediction)
    prediction_filepath = os.path.join(PARAMS.experiment_dir, 'evaluation_prediction.csv')
    prediction.to_csv(prediction_filepath, index=None)

    if prediction.empty:
        LOGGER.info('Background predicted for all the images. Metric cannot be calculated')
    else:
        LOGGER.info('Calculating mean average precision')
        validation_annotations = annotations[annotations[ID_COLUMN].isin(valid_img_ids)]
        validation_annotations_human_labels = annotations_human_labels[
            annotations_human_labels[ID_COLUMN].isin(valid_img_ids)]
        validation_annotations_filepath = os.path.join(PARAMS.experiment_dir, 'validation_annotations.csv')
        validation_annotations.to_csv(validation_annotations_filepath, index=None)
        validation_annotations_human_labels_filepath = os.path.join(PARAMS.experiment_dir,
                                                                    'validation_annotations_human_labels.csv')

        validation_annotations_human_labels.to_csv(validation_annotations_human_labels_filepath, index=None)
        metrics_filepath = os.path.join(PARAMS.experiment_dir, 'validation_metrics')
        mean_average_precision = competition_metric_evaluation(annotation_filepath=validation_annotations_filepath,
                                                               annotations_human_labels_filepath=validation_annotations_human_labels_filepath,
                                                               prediction_filepath=prediction_filepath,
                                                               label_hierarchy_filepath=PARAMS.bbox_hierarchy_filepath,
                                                               metrics_filepath=metrics_filepath,
                                                               list_of_desired_classes=DESIRED_CLASS_SUBSET,
                                                               mappings_filepath=PARAMS.class_mappings_filepath
                                                               )
        LOGGER.info('MAP on validation is {}'.format(mean_average_precision))
        CTX.channel_send('MAP', 0, mean_average_precision)


def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    LOGGER.info('predicting')

    n_ids = 100 if dev_mode else None
    test_img_ids = get_img_ids_from_folder(PARAMS.test_imgs_dir, n_ids=n_ids)

    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.test_imgs_dir

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(test_img_ids, pipeline, chunk_size)

    sample_submission = pd.read_csv(PARAMS.sample_submission)
    prediction = add_missing_image_ids(prediction, sample_submission)
    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.csv')
    prediction.to_csv(submission_filepath, index=None)
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(prediction.head()))

    if submit_predictions:
        make_submission(submission_filepath)


def visualize(pipeline_name, image_dir=None, single_image=None, n_files=16, show_popups=False):
    if image_dir:
        SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = image_dir
    else:
        SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.test_imgs_dir

    if single_image:
        raise NotImplemented

    test_img_ids = get_img_ids_from_folder(SOLUTION_CONFIG['loader']['dataset_params']['images_dir'], n_ids=n_files)
    pipeline = PIPELINES[pipeline_name]['visualize'](SOLUTION_CONFIG)

    data = {'input': {'img_ids': test_img_ids
                      },
            'metadata': {'annotations': None,
                         'annotations_human_labels': None
                         }
            }

    images = pipeline.fit_transform(data)
    for img in images[0]:
        basewidth = 600
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS) # we have to make them smaller bc of neptune limitations
        CTX.channel_send("my_image_channel", neptune.Image(
            name="image’s name",
            description="this image depicts a cat",
            data=img))

        if show_popups:
            cv2.imshow('sample image', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)  # waits until a key is pressed
            cv2.destroyAllWindows()  # destroys the window showing image


def make_submission(submission_filepath):
    LOGGER.info('Making Kaggle submit...')
    os.system(
        'kaggle competitions submit -c google-ai-open-images-object-detection-track -f {} -m {}'.format(
            submission_filepath, PARAMS.kaggle_message))
    LOGGER.info('Kaggle submit completed')


def generate_prediction(img_ids, pipeline, chunk_size):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(img_ids, pipeline, chunk_size)
    else:
        return _generate_prediction(img_ids, pipeline)


def _generate_prediction(img_ids, pipeline):
    data = {'input': {'img_ids': img_ids
                      },
            'metadata': {'annotations': None,
                         'annotations_human_labels': None
                         }
            }

    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    return output['y_pred']


def _generate_prediction_in_chunks(img_ids, pipeline, chunk_size):
    chunk_nr = int(math.ceil(len(img_ids) / chunk_size))
    predictions = []
    for i, img_ids_chunk in enumerate(generate_list_chunks(img_ids, chunk_size)):
        LOGGER.info('Processing chunk {}/{}'.format(i, chunk_nr))
        data = {'input': {'img_ids': img_ids_chunk
                          },
                'metadata': {'annotations': None,
                             'annotations_human_labels': None
                             }
                }

        pipeline.clean_cache()
        output = pipeline.transform(data)
        pipeline.clean_cache()
        predictions.append(output['y_pred'])

    predictions = pd.concat(predictions)
    return predictions
