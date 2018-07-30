import os

import PIL
import numpy as np
import pandas as pd
from steppy.base import BaseTransformer

from src.logging import LOGGER
from .pipeline_config import CODES2NAMES, SOLUTION_CONFIG, params
from .utils import visualize_bboxes


class PredictionFormatter(BaseTransformer):
    def transform(self, images_data, results, decoder_dict):
        self.decoder_dict = decoder_dict
        image_ids = images_data['ImageID'].values.tolist()
        prediction_strings = []
        for bboxes, labels, scores in results:
            prediction_strings.append(self._get_prediction_string(bboxes, labels, scores))
        submission = pd.DataFrame({'ImageId': image_ids, 'PredictionString': prediction_strings})
        return {'submission': submission}

    def _get_prediction_string(self, bboxes, labels, scores):
        prediction_list = []
        for bbox, label, score in zip(bboxes, labels, scores):
            prediction_list.append(self._get_class_id(label))
            prediction_list.append(str(score))
            prediction_list.extend([str(coord) for coord in bbox])
        prediction_string = " ".join(prediction_list)
        return prediction_string

    def _get_class_id(self, label):
        return self.decoder_dict[label]


class Visualizer(BaseTransformer):
    def transform(self, images_data, results, decoder_dict):
        image_ids = images_data['ImageID'].values.tolist()
        decoder_dict = decoder_dict
        all_detections, all_boxes = [], []
        for i, (image_id, detections) in enumerate(zip(image_ids, results)):
            if not bool(detections[0].size()):
                continue
            LOGGER.info("Drawing boxes on image {}/{}".format(i, len(results)))
            image = PIL.Image.open(
                os.path.join(SOLUTION_CONFIG['loader']['dataset_params']['images_dir'], image_id + '.jpg'))
            width, height = image.size  # original image size
            box = detections[0].numpy()
            classes = detections[1].numpy()
            scores = detections[2].numpy()

            df = pd.DataFrame(np.column_stack([box, classes, scores]))
            df.columns = ['x1', 'y1', 'x2', 'y2', 'class_id', 'score']
            df['class_name'] = df.class_id.map(decoder_dict)
            df.class_name = df.class_name.map(CODES2NAMES)

            # to absolute
            df['x1'] = df['x1'] * width
            df['x2'] = df['x2'] * width
            df['y1'] = df['y1'] * height
            df['y2'] = df['y2'] * height

            pil_image_detections = visualize_bboxes(image, df)
            all_detections.append(pil_image_detections)
            all_boxes.append(box)
        return all_detections
