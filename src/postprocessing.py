import numpy as np
import pandas as pd
from steppy.base import BaseTransformer

from src.logging import LOGGER
from .pipeline_config import params
from .utils import visualize_bboxes, get_class_mappings
import os
import PIL
import numpy as np

codes2names, names2codes = get_class_mappings(mappings_file=params.class_mappings_filepath)


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


class Visualizer(BaseTransformer):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def transform(self, image_ids, results, decoder_dict):

        decoder_dict = decoder_dict
        image_dir = params.train_imgs_dir

        all_detections = []
        all_box = []
        for i, (image_id, detections) in enumerate(zip(image_ids, results)):
            if not bool(detections[0].size()):
                continue
            LOGGER.info("Drawing boxes on image {}/{}".format(i, len(results)))
            image = PIL.Image.open(os.path.join(image_dir, image_id + '.jpg'))
            h, w = self.image_size  # resize size
            width, heighth = image.size  # original image size
            box = detections[0].numpy()
            classes = detections[1].numpy()
            scores = detections[2].numpy()

            df = pd.DataFrame(np.column_stack([box, classes, scores]))
            df.columns = ['x1', 'y1', 'x2', 'y2', 'class_id', 'score']
            df['class_name'] = df.class_id.map(decoder_dict)
            df.class_name = df.class_name.map(codes2names)

            # revert resize

            df.x1 = df.x1 / h
            df.y1 = df.y1 / w
            df.x2 = df.x2 / h
            df.y2 = df.y2 / w

            # to absolute

            df['x1'] = df['x1'] * width
            df['x2'] = df['x2'] * width
            df['y1'] = df['y1'] * heighth
            df['y2'] = df['y2'] * heighth

            pil_image_detections = visualize_bboxes(image, df)
            all_detections.append(pil_image_detections)
            all_box.append(box)
        return all_detections, all_box
