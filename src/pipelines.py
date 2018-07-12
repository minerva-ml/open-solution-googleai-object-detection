from functools import partial

from steppy.base import IdentityOperation

from .loaders import ImageDetectionLoader
from .steppy.base import Step
from .models import Retina
from .retinanet import DataDecoder
from .postprocessing import SubmissionProducer, resize_bboxes
from .utils import make_apply_transformer, get_image_size


def retinanet(config, train_mode):
    save_output = False
    load_saved_output = False

    loader = preprocessing_generator(config, is_train=train_mode)

    retinanet = Step(name='retinanet',
                     transformer=Retina(**config.retinanet, train_mode=train_mode),
                     input_data=['callback_input'],
                     input_steps=[loader],
                     cache_dirpath=config.env.cache_dirpath,
                     save_output=save_output,
                     is_trainable=True,
                     load_saved_output=load_saved_output)

    if train_mode:
        return retinanet

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
    if is_train:
        loader = Step(name='loader',
                      transformer=ImageDetectionLoader(train_mode=True, **config.loader),
                      input_data=['input', 'validation_input', 'metadata'],
                      adapter={'X': ([('input', 'img_ids')]),
                               'y': ([('metadata', 'annotations')]),
                               'X_valid': ([('validation_input', 'valid_img_ids')]),
                               'y_valid': ([('metadata', 'annotations')])
                               },
                      cache_dirpath=config.env.cache_dirpath)
    else:
        loader = Step(name='loader',
                      transformer=ImageDetectionLoader(train_mode=False, **config.loader),
                      input_data=['specs'],
                      adapter={'X': ([('input', 'img_ids')]),
                               'y': ([('metadata', 'annotations')])
                               },
                      cache_dirpath=config.env.cache_dirpath)
    return loader


def postprocessing(model, config, **kwargs):
    decoder = Step(name='decoder',
                   transformer=DataDecoder(),
                   input_steps=[model, ],
                   cache_dirpath=config.env.cache_dirpath, **kwargs
                   )

    resizer = Step(name='resizer',
                   transformer=make_apply_transformer(func=resize_bboxes,
                                                      output_name='resized_results',
                                                      ),
                   input_steps=[decoder, ],
                   cache_dirpath=config.env.cache_dirpath, **kwargs
                   )

    submission_producer = Step(name='submission_producer',
                               transformer=SubmissionProducer(),
                               input_steps=[resizer, label_encoder],
                               input_data=['input', ],
                               adapter={'image_ids': ([('input', 'img_ids')]),
                                        'results': ([(resizer.name, 'results')]),
                                        'image_size': ([('input', 'image_size')]),
                                        'decoder_dict': ([(label_encoder.name, 'dict')])},
                               cache_dirpath=config.env.cache_dirpath, **kwargs
                               )
    return submission_producer


PIPELINES = {'retinanet': {'train': partial(retinanet, train_mode=True),
                           'inference': partial(retinanet, train_mode=False),
                           },
             }
