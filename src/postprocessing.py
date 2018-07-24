import numpy as np
import pandas as pd

from steppy.base import BaseTransformer


class PredictionFormatter(BaseTransformer):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def transform(self, image_ids, results, decoder_dict):
        self.decoder_dict = decoder_dict
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
            prediction_list.extend(self._get_bbox_relative(bbox))
        prediction_string = " ".join(prediction_list)
        return prediction_string

    def _get_class_id(self, label):
        return self.decoder_dict[label]

    def _get_bbox_relative(self, bbox):
        h = self.image_size[0]
        w = self.image_size[1]
        x_min = np.clip(bbox[0] / h, 0.0, 1.0)
        y_min = np.clip(bbox[1] / w, 0.0, 1.0)
        x_max = np.clip(bbox[2] / h, 0.0, 1.0)
        y_max = np.clip(bbox[3] / w, 0.0, 1.0)
        result = [x_min, y_min, x_max, y_max]
        return [str(r) for r in result]
