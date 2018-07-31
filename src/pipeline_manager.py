import os
import shutil

from deepsense import neptune
import pandas as pd
import numpy as np
import math
import cv2
from PIL import Image

from .pipeline_config import DESIRED_CLASS_SUBSET, N_SUB_CLASSES, ID_COLUMN, LABEL_COLUMN, SEED, SOLUTION_CONFIG
from .pipelines import PIPELINES

from .utils import NeptuneContext, competition_metric_evaluation, generate_data_frame_chunks, get_img_ids_from_folder, \
    init_logger, reduce_number_of_classes, set_seed, submission_formatting, add_missing_image_ids, map_per_class, \
    generate_metadata, prepare_metadata

LOGGER = init_logger()
CTX = NeptuneContext()
PARAMS = CTX.params
set_seed(SEED)


class PipelineManager:
    def __init__(self):
        if not os.path.isfile(PARAMS.metadata_filepath):
            self.prepare_metadata()

    def prepare_metadata(self):
        prepare_metadata(annotations_filepath=PARAMS.annotations_filepath,
                         valid_ids_filepath=PARAMS.valid_ids_filepath,
                         default_valid_ids=PARAMS.default_valid_ids,
                         metadata_filepath=PARAMS.metadata_filepath,
                         train_image_dir=PARAMS.train_imgs_dir,
                         valid_image_dir=PARAMS.train_imgs_dir,
                         test_image_dir=PARAMS.test_imgs_dir,
                         id_column=ID_COLUMN,
                         num_threads=PARAMS.num_threads,
                         logger=LOGGER)

    def train(self, pipeline_name, dev_mode):
        train(pipeline_name, dev_mode)

    def evaluate(self, pipeline_name, dev_mode, chunk_size):
        evaluate(pipeline_name, dev_mode, chunk_size)

    def predict(self, pipeline_name, dev_mode, submit_predictions, chunk_size):
        predict(pipeline_name, dev_mode, submit_predictions, chunk_size)

    def make_submission(self, submission_filepath):
        make_submission(submission_filepath)

    def visualize(self, pipeline_name, image_dir=None, single_image=None, n_files=16, show_popups=False,
                  classes_to_visualize=None, nms_threshold=None, classification_threshold=None):
        visualize(pipeline_name=pipeline_name, image_dir=image_dir, single_image=single_image, n_files=n_files,
                  show_popups=show_popups, classes_to_visualize=classes_to_visualize, nms_threshold=nms_threshold,
                  classification_threshold=classification_threshold)


def train(pipeline_name, dev_mode):
    LOGGER.info('training')
    if bool(PARAMS.clean_experiment_directory_before_training) and os.path.isdir(PARAMS.experiment_dir):
        shutil.rmtree(PARAMS.experiment_dir)
    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.train_imgs_dir

    annotations, annotations_human_labels, train_data, valid_data = _get_input_data(dev_mode)

    data = {'input': {'images_data': train_data
                      },
            'validation_input': {'valid_images_data': valid_data,
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
    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.train_imgs_dir

    annotations, annotations_human_labels, _, valid_data = _get_input_data(dev_mode)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(valid_data, pipeline, chunk_size)

    LOGGER.info('Formatting prediction')
    prediction = submission_formatting(prediction)
    prediction_filepath = os.path.join(PARAMS.experiment_dir, 'evaluation_prediction.csv')
    prediction.to_csv(prediction_filepath, index=None)

    if prediction.empty:
        LOGGER.info('Background predicted for all the images. Metric cannot be calculated')
    else:
        LOGGER.info('Calculating mean average precision')
        valid_img_ids = valid_data['ImageID'].values
        validation_annotations = annotations[annotations[ID_COLUMN].isin(valid_img_ids)]
        validation_annotations_human_labels = annotations_human_labels[
            annotations_human_labels[ID_COLUMN].isin(valid_img_ids)]
        validation_annotations_filepath = os.path.join(PARAMS.experiment_dir, 'validation_annotations.csv')
        validation_annotations.to_csv(validation_annotations_filepath, index=None)
        validation_annotations_human_labels_filepath = os.path.join(PARAMS.experiment_dir,
                                                                    'validation_annotations_human_labels.csv')
        validation_annotations_human_labels.to_csv(validation_annotations_human_labels_filepath, index=None)
        metrics_filepath = os.path.join(PARAMS.experiment_dir, 'validation_metrics')

        if dev_mode:
            class_subset = annotations[annotations[ID_COLUMN].isin(valid_img_ids)][
                LABEL_COLUMN].unique().tolist()
        else:
            class_subset = DESIRED_CLASS_SUBSET

        mean_average_precision = competition_metric_evaluation(annotation_filepath=validation_annotations_filepath,
                                                               annotations_human_labels_filepath=validation_annotations_human_labels_filepath,
                                                               prediction_filepath=prediction_filepath,
                                                               label_hierarchy_filepath=PARAMS.bbox_hierarchy_filepath,
                                                               metrics_filepath=metrics_filepath,
                                                               list_of_desired_classes=class_subset,
                                                               mappings_filepath=PARAMS.class_mappings_filepath
                                                               )

        for class_name, map_score in map_per_class(metrics_filepath, class_subset, PARAMS.class_mappings_filepath):
            LOGGER.info('MAP on validation {} is {}'.format(class_name, map_score))
        LOGGER.info('MAP on validation mean is {}'.format(mean_average_precision))
        CTX.channel_send('MAP', 0, mean_average_precision)


def predict(pipeline_name, dev_mode, submit_predictions, chunk_size):
    LOGGER.info('predicting')
    SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.test_imgs_dir

    metadata = pd.read_csv(PARAMS.metadata_filepath)
    meta_test = metadata[metadata['is_test'] == 1]
    if dev_mode:
        meta_test = meta_test.sample(100, random_state=SEED)
    meta_test = meta_test.reset_index(drop=True)

    pipeline = PIPELINES[pipeline_name]['inference'](SOLUTION_CONFIG)
    prediction = generate_prediction(meta_test, pipeline, chunk_size)

    sample_submission = pd.read_csv(PARAMS.sample_submission)
    prediction = add_missing_image_ids(prediction, sample_submission)
    submission_filepath = os.path.join(PARAMS.experiment_dir, 'submission.csv')
    prediction.to_csv(submission_filepath, index=None)
    LOGGER.info('submission saved to {}'.format(submission_filepath))
    LOGGER.info('submission head \n\n{}'.format(prediction.head()))

    if submit_predictions:
        make_submission(submission_filepath)


def visualize(pipeline_name,
              image_dir=None,
              single_image=None,
              n_files=16,
              show_popups=False,
              classes_to_visualize=None,
              classification_threshold=None,
              nms_threshold=None):
    if classification_threshold:
        SOLUTION_CONFIG['postprocessing']['data_decoder']['cls_thrs'] = classification_threshold
    if nms_threshold:
        SOLUTION_CONFIG['postprocessing']['data_decoder']['nms_thrs'] = nms_threshold

    if image_dir:
        SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = image_dir
        img_ids = get_img_ids_from_folder(image_dir, n_ids=n_files)
        metadata = generate_metadata(valid_image_ids=img_ids, valid_image_dir=image_dir)
        *_, images_data = _get_input_data(metadata=metadata, reduce=False)
    else:
        SOLUTION_CONFIG['loader']['dataset_params']['images_dir'] = PARAMS.train_imgs_dir
        *_, images_data = _get_input_data(classes_to_visualize=classes_to_visualize)
        images_data = images_data.sample(n_files, random_state=SEED, replace=len(images_data) < n_files)

    if single_image:
        raise NotImplemented

    pipeline = PIPELINES[pipeline_name]['visualize'](SOLUTION_CONFIG)

    data = {'input': {'images_data': images_data
                      },
            'metadata': {'annotations': None,
                         'annotations_human_labels': None
                         }
            }

    images_with_drawn_boxes = pipeline.transform(data)
    for img in images_with_drawn_boxes:
        basewidth = 500
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)  # we have to make them smaller bc of neptune limitations
        CTX.channel_send("my_image_channel", neptune.Image(name="", description="", data=img))
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


def generate_prediction(images_data, pipeline, chunk_size):
    if chunk_size is not None:
        return _generate_prediction_in_chunks(images_data, pipeline, chunk_size)
    else:
        return _generate_prediction(images_data, pipeline)


def _generate_prediction(images_data, pipeline):
    data = {'input': {'images_data': images_data
                      },
            'metadata': {'annotations': None,
                         'annotations_human_labels': None
                         }
            }

    pipeline.clean_cache()
    output = pipeline.transform(data)
    pipeline.clean_cache()
    return output['y_pred']


def _generate_prediction_in_chunks(images_data, pipeline, chunk_size):
    chunk_nr = int(math.ceil(len(images_data) / chunk_size))
    predictions = []
    for i, images_data_chunk in enumerate(generate_data_frame_chunks(images_data, chunk_size)):
        LOGGER.info('Processing chunk {}/{}'.format(i, chunk_nr))
        data = {'input': {'images_data': images_data_chunk
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


def _get_input_data(dev_mode=False, metadata=None, reduce=True, classes_to_visualize=None):
    annotations = pd.read_csv(PARAMS.annotations_filepath)
    annotations_human_labels = pd.read_csv(PARAMS.annotations_human_labels_filepath)

    classes_to_visualize = classes_to_visualize or DESIRED_CLASS_SUBSET

    if metadata is None:
        metadata = pd.read_csv(PARAMS.metadata_filepath)

    if classes_to_visualize and reduce:
        LOGGER.info("Training on a reduced class subset: {}".format(DESIRED_CLASS_SUBSET))
        annotations = reduce_number_of_classes(annotations,
                                               classes_to_visualize,
                                               PARAMS.class_mappings_filepath)

        annotations_human_labels = reduce_number_of_classes(annotations_human_labels,
                                                            classes_to_visualize,
                                                            PARAMS.class_mappings_filepath)

        img_ids_in_reduced_annotations = annotations[ID_COLUMN].unique()
        metadata = metadata[metadata['ImageID'].isin(img_ids_in_reduced_annotations)]

    meta_train = metadata[metadata['is_train'] == 1]
    meta_valid = metadata[metadata['is_valid'] == 1]

    if dev_mode:
        meta_train = meta_train.sample(100, random_state=SEED)
        meta_valid = meta_valid.sample(20, random_state=SEED)

    return annotations, annotations_human_labels, meta_train.reset_index(drop=True), meta_valid.reset_index(drop=True)
