from category_encoders.ordinal import OrdinalEncoder
from sklearn.externals import joblib
from steppy.base import BaseTransformer


class GoogleAiLabelEncoder(BaseTransformer):
    def __init__(self, colname):
        self.colname = colname
        self.encoder = OrdinalEncoder()

    def fit(self, annotations, **kwargs):
        self.encoder.fit(annotations[self.colname].values)
        return self

    def transform(self, annotations, annotations_human_labels, **kwargs):
        if annotations is not None:
            annotations[self.colname] = self.encoder.transform(annotations[self.colname].values)
            annotations_human_labels[self.colname] = self.encoder.transform(
                annotations_human_labels[self.colname].values)
            return {'annotations': annotations,
                    'annotations_human_labels': annotations_human_labels}
        else:
            return {'mapping': self.encoder.category_mapping[0]['mapping']}

    def load(self, filepath):
        self.encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.encoder, filepath)


class GoogleAiLabelDecoder(BaseTransformer):

    def transform(self, mapping, **kwargs):
        inverse_mapping = {val: name for name, val in mapping}
        return {'inverse_mapping': inverse_mapping}
