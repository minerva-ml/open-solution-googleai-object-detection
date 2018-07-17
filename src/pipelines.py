from functools import partial

from steppy.base import IdentityOperation
from steppy.adapter import Adapter, E

from .loaders import ImageDetectionLoader
from .steppy_dev.base import Step
from .models import Retina
from .retinanet import DataDecoder
from .postprocessing import PredictionFormatter
from .preprocessing import GoogleAiLabelEncoder, GoogleAiLabelDecoder


def retinanet(config, train_mode):
    persist_output = False
    load_persisted_output = False

    loader = preprocessing_generator(config, is_train=train_mode)

    retinanet = Step(name='retinanet',
                     transformer=Retina(**config.retinanet, train_mode=train_mode),
                     input_steps=[loader],
                     experiment_directory=config.env.cache_dirpath,
                     persist_output=persist_output,
                     is_trainable=True,
                     load_persisted_output=load_persisted_output)

    if train_mode:
        return retinanet

    postprocessor = postprocessing(retinanet, loader.get_step('label_encoder'), config)

    output = Step(name='output',
                  transformer=IdentityOperation(),
                  input_steps=[postprocessor],
                  adapter=Adapter({'y_pred': E(postprocessor.name, 'submission')}),
                  experiment_directory=config.env.cache_dirpath,
                  persist_output=persist_output,
                  load_persisted_output=load_persisted_output)
    return output


def preprocessing_generator(config, is_train):
    label_encoder = Step(name='label_encoder',
                         transformer=GoogleAiLabelEncoder(**config.label_encoder),
                         input_data=['metadata'],
                         adapter=Adapter({'annotations': E('metadata', 'annotations'),
                                          'annotations_human_labels': E('metadata', 'annotations_human_labels')
                                          }),
                         is_trainable=True,
                         experiment_directory=config.env.cache_dirpath)

    if is_train:
        loader = Step(name='loader',
                      transformer=ImageDetectionLoader(train_mode=True, **config.loader),
                      input_data=['input', 'validation_input'],
                      input_steps=[label_encoder],
                      adapter=Adapter({'ids': E('input', 'img_ids'),
                                       'valid_ids': E('validation_input', 'valid_img_ids'),
                                       'annotations': E(label_encoder.name, 'annotations'),
                                       'annotations_human_labels': E(label_encoder.name, 'annotations_human_labels'),
                                       }),
                      experiment_directory=config.env.cache_dirpath)

    else:
        loader = Step(name='loader',
                      transformer=ImageDetectionLoader(train_mode=False, **config.loader),
                      input_data=['input'],
                      input_steps=[label_encoder],
                      adapter=Adapter({'ids': E('input', 'img_ids'),
                                       'annotations': None,
                                       'annotations_human_labels': None,
                                       }),
                      experiment_directory=config.env.cache_dirpath)
    return loader


def postprocessing(model, label_encoder, config):
    label_decoder = Step(name='label_decoder',
                         transformer=GoogleAiLabelDecoder(),
                         input_steps=[label_encoder, ],
                         experiment_directory=config.env.cache_dirpath)

    decoder = Step(name='decoder',
                   transformer=DataDecoder(**config.postprocessing.data_decoder),
                   input_steps=[model, ],
                   experiment_directory=config.env.cache_dirpath)

    submission_producer = Step(name='submission_producer',
                               transformer=PredictionFormatter(**config.postprocessing.prediction_formatter),
                               input_steps=[label_decoder, decoder],
                               input_data=['input'],
                               adapter=Adapter({'image_ids': E('input', 'img_ids'),
                                                'results': E(decoder.name, 'results'),
                                                'decoder_dict': E(label_decoder.name, 'inverse_mapping')}),
                               experiment_directory=config.env.cache_dirpath)
    return submission_producer


PIPELINES = {'retinanet': {'train': partial(retinanet, train_mode=True),
                           'inference': partial(retinanet, train_mode=False),
                           },
             }
