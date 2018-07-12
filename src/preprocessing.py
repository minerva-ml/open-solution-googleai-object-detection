from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from steppy.base import BaseTransformer


class GoogleAiLabelEncoder(BaseTransformer):
    def __init__(self, colname):
        self.colname = colname
        self.encoder = LabelEncoder()

    def fit(self, annotations, annotations_human_labels, **kwargs):
        self.encoder.fit(annotations[self.colname])
        return self

    def transform(self, annotations, annotations_human_labels, **kwargs):
        annotations[self.colname] = self.encoder.transform(annotations[self.colname])
        annotations_human_labels[self.colname] = self.encoder.transform(annotations_human_labels[self.colname])
        print(annotations.head())
        return {'annotations': annotations,
                'annotations_human_labels': annotations_human_labels}

    def load(self, filepath):
        self.encoder = joblib.load(filepath)
        return self

    def persist(self, filepath):
        joblib.dump(self.encoder, filepath)


class GoogleAiLabelDecoder(BaseTransformer):
    def transform(self, label_encoder):
        inverse_mapping = label_encoder
        return {'inverse_mapping': inverse_mapping}
