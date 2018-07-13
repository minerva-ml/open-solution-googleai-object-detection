import pandas as pd

from steppy.base import BaseTransformer


class PredictionFormatter(BaseTransformer):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def transform(self, image_ids, results, decoder_dict):
        self.decoder_dict = decoder_dict
        prediction_strings = []
        for bboxes, labels in results:
            prediction_strings.append(self._get_prediction_string(bboxes, labels))
        submission = pd.DataFrame({'ImageId': image_ids, 'PredictionString': prediction_strings})
        return {'submission': submission}

    def _get_prediction_string(self, bboxes, labels):
        prediction_list = []
        for bbox, label in zip(bboxes, labels):
            prediction_list.append(self._get_class_id(label))
            prediction_list.extend(self._get_bbox_relative(bbox))
        prediction_string = " ".join(prediction_list)
        return prediction_string

    def _get_class_id(self, label):
        return self.decoder_dict[label]

    def _get_bbox_relative(self, bbox):
        x = self.image_size[0]
        y = self.image_size[1]
        x_center = bbox[0] / x
        y_center = bbox[1] / y
        h = bbox[2] / x
        w = bbox[3] / y
        result = [x_center, y_center, h, w]
        return [str(r) for r in result]
