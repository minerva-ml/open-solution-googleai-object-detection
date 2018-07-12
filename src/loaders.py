from steppy.base import BaseTransformer


class DetectionLoader(BaseTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)