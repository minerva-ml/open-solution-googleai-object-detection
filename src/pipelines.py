from functools import partial

from steppy.base import IdentityOperation

from . import loaders
from .steppy.base import Step
from .steppy.preprocessing.misc import XYSplit
from .utils import squeeze_inputs
from .models import BaseRetina


def retinanet(config, train_mode):
    save_output = False
    load_saved_output = False

    loader = preprocessing_generator(config, is_train=train_mode)

    retinanet = Step(name='retinanet',
                     transformer=BaseRetina(**config.retinanet),
                     input_data=['callback_input'],
                     input_steps=[loader],
                     cache_dirpath=config.env.cache_dirpath,
                     save_output=save_output,
                     is_trainable=True,
                     load_saved_output=load_saved_output)

    postprocessor = postprocessing(retinanet, config, save_output=save_output)

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
        Loader = loaders.MetadataImageSegmentationLoaderCropPad
    elif config.execution.loader_mode == 'resize':
        Loader = loaders.MetadataImageSegmentationLoaderResize
    else:
        raise NotImplementedError('only crop_and_pad and resize options available')

    if is_train:
        xy_train = Step(name='xy_train',
                        transformer=XYSplit(**config.xy_splitter),
                        input_data=['input', 'specs'],
                        adapter={'meta': ([('input', 'meta')]),
                                 'train_mode': ([('specs', 'train_mode')])
                                 },
                        cache_dirpath=config.env.cache_dirpath)

        xy_inference = Step(name='xy_inference',
                            transformer=XYSplit(**config.xy_splitter),
                            input_data=['callback_input', 'specs'],
                            adapter={'meta': ([('callback_input', 'meta_valid')]),
                                     'train_mode': ([('specs', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=Loader(**config.loader),
                      input_data=['specs'],
                      input_steps=[xy_train, xy_inference],
                      adapter={'X': ([('xy_train', 'X')], squeeze_inputs),
                               'y': ([('xy_train', 'y')], squeeze_inputs),
                               'train_mode': ([('specs', 'train_mode')]),
                               'X_valid': ([('xy_inference', 'X')], squeeze_inputs),
                               'y_valid': ([('xy_inference', 'y')], squeeze_inputs),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    else:
        xy_inference = Step(name='xy_inference',
                            transformer=XYSplit(**config.xy_splitter),
                            input_data=['input', 'specs'],
                            adapter={'meta': ([('input', 'meta')]),
                                     'train_mode': ([('specs', 'train_mode')])
                                     },
                            cache_dirpath=config.env.cache_dirpath)

        loader = Step(name='loader',
                      transformer=Loader(**config.loader),
                      input_data=['specs'],
                      input_steps=[xy_inference, xy_inference],
                      adapter={'X': ([('xy_inference', 'X')], squeeze_inputs),
                               'y': ([('xy_inference', 'y')], squeeze_inputs),
                               'train_mode': ([('specs', 'train_mode')]),
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return loader


def postprocessing(model, config, **kwargs):
    postprocessor = Step(name='postprocessor',
                         transformer=IdentityOperation(),
                         input_steps=[model],
                         cache_dirpath=config.env.cache_dirpath, **kwargs)
    return postprocessor


PIPELINES = {'retinanet': {'train': partial(retinanet, train_mode=True),
                           'inference': partial(retinanet, train_mode=False),
                           },
             }
