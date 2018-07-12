from functools import partial

from steppy.base import IdentityOperation

from . import loaders
from .steppy.base import Step
from .models import BaseRetina
from .preprocessing import GoogleAiLabelEncoder, GoogleAiLabelDecoder


def retinanet(config, train_mode):
    save_output = False
    load_saved_output = False

    loader, label_encoder = preprocessing_generator(config, is_train=train_mode)

    retinanet = Step(name='retinanet',
                     transformer=BaseRetina(**config.retinanet),
                     input_data=['callback_input'],
                     input_steps=[loader],
                     cache_dirpath=config.env.cache_dirpath,
                     save_output=save_output,
                     is_trainable=True,
                     load_saved_output=load_saved_output)

    postprocessor = postprocessing(retinanet, label_encoder, config, save_output=save_output)

    output = Step(name='output',
                  transformer=IdentityOperation(),
                  input_steps=[postprocessor],
                  adapter={'y_pred': ([(postprocessor.name, 'postprocessed_images')]),
                           },
                  cache_dirpath=config.env.cache_dirpath,
                  save_output=save_output, load_saved_output=load_saved_output)
    return output


def preprocessing_generator(config, is_train):
    if config.execution.loader_mode == 'crop_and_pad':
        Loader = loaders.DetectionLoader
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.DetectionLoader
    else:
        raise NotImplementedError('only crop_and_pad and resize options available')

    label_encoder = Step(name='label_encoder',
                         transformer=GoogleAiLabelEncoder(),
                         input_data=['metadata'],
                         adapter={'annotations': ([('metadata', 'annotations')]),
                                  'annotations_human_labels': ([('metadata', 'annotations_human_labels')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath)

    if is_train:
        loader = Step(name='loader',
                      transformer=Loader(**config.loader),
                      input_data=['input', 'callback_input'],
                      input_steps=[label_encoder],
                      adapter={'X': ([('input', 'img_ids')]),
                               'annotations': ([(label_encoder.name, 'annotations')]),
                               'annotations_human_labels': ([(label_encoder.name, 'annotations_human_labels')]),
                               'X_valid': ([('callback_input', 'valid_img_ids')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    else:
        loader = Step(name='loader',
                      transformer=Loader(**config.loader),
                      input_data=['input'],
                      adapter={'X': ([('input', 'img_ids')]),
                               'annotations': None,
                               'annotations_human_labels': None,
                               'X_valid': None,
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return loader, label_encoder


def postprocessing(model, label_encoder, config, **kwargs):
    decoder = Step(name='decoder',
                   transformer=GoogleAiLabelDecoder(label_encoder),
                   input_steps=[model],
                   cache_dirpath=config.env.cache_dirpath, **kwargs)

    postprocessor = Step(name='postprocessor',
                         transformer=IdentityOperation(),
                         input_steps=[model, decoder],
                         adapter={'predictions': ([('model', 'predictions')]),
                                  'label_mapping': ([('decoder', 'label_mapping')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath, **kwargs)
    return postprocessor


PIPELINES = {'retinanet': {'train': partial(retinanet, train_mode=True),
                           'inference': partial(retinanet, train_mode=False),
                           },
             }
