from toolkit.pytorch_transformers.models import Model


class BaseRetina(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
