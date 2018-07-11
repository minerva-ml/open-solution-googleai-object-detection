from functools import partial

from . import loaders
from .steppy.base import Step, Dummy
from .steppy.preprocessing.misc import XYSplit
from .utils import squeeze_inputs, make_apply_transformer, make_apply_transformer_stream
from .models import PyTorchUNet, PyTorchUNetWeighted, PyTorchUNetStream, PyTorchUNetWeightedStream, ScoringLightGBM, \
    ScoringRandomForest
from . import postprocessing as post


def retinanet(config, train_mode):
    save_output = False
    load_saved_output = False

    make_apply_transformer_ = make_apply_transformer_stream if config.execution.stream_mode else make_apply_transformer

    loader = preprocessing_generator(config, is_train=train_mode)
    unet = Step(name='unet',
                transformer=PyTorchUNetStream(**config.unet) if config.execution.stream_mode
                else PyTorchUNet(**config.unet),
                input_data=['callback_input'],
                input_steps=[loader],
                cache_dirpath=config.env.cache_dirpath,
                save_output=save_output,
                is_trainable=True,
                load_saved_output=load_saved_output)

    mask_postprocessed = mask_postprocessing(unet, config, make_apply_transformer_, save_output=save_output)

    output = Step(name='output',
                  transformer=Dummy(),
                  input_steps=[mask_postprocessed],
                  adapter={'y_pred': ([(mask_postprocessed.name, 'images_with_scores')]),
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


def mask_postprocessing(model, config, make_transformer, **kwargs):
    mask_resize = Step(name='mask_resize',
                       transformer=make_transformer(post.resize_image,
                                                    output_name='resized_images',
                                                    apply_on=['images', 'target_sizes']),
                       input_data=['input'],
                       input_steps=[model],
                       adapter={'images': ([(model.name, 'multichannel_map_prediction')]),
                                'target_sizes': ([('input', 'target_sizes')]),
                                },
                       cache_dirpath=config.env.cache_dirpath,
                       cache_output=not config.execution.stream_mode, **kwargs)

    category_mapper = Step(name='category_mapper',
                           transformer=make_transformer(post.categorize_multilayer_image,
                                                        output_name='categorized_images'),
                           input_steps=[mask_resize],
                           adapter={'images': ([('mask_resize', 'resized_images')]),
                                    },
                           cache_dirpath=config.env.cache_dirpath, **kwargs)

    mask_erosion = Step(name='mask_erosion',
                        transformer=make_transformer(partial(post.erode_image, **config.postprocessor.mask_erosion),
                                                     output_name='eroded_images'),
                        input_steps=[category_mapper],
                        adapter={'images': ([(category_mapper.name, 'categorized_images')]),
                                 },
                        cache_dirpath=config.env.cache_dirpath, **kwargs)

    labeler = Step(name='labeler',
                   transformer=make_transformer(post.label_multilayer_image,
                                                output_name='labeled_images'),
                   input_steps=[mask_erosion],
                   adapter={'images': ([(mask_erosion.name, 'eroded_images')]),
                            },
                   cache_dirpath=config.env.cache_dirpath, **kwargs)

    mask_dilation = Step(name='mask_dilation',
                         transformer=make_transformer(partial(post.dilate_image, **config.postprocessor.mask_dilation),
                                                      output_name='dilated_images'),
                         input_steps=[labeler],
                         adapter={'images': ([(labeler.name, 'labeled_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath, **kwargs)

    score_builder = Step(name='score_builder',
                         transformer=make_transformer(post.build_score,
                                                      output_name='images_with_scores',
                                                      apply_on=['images', 'probabilities']),
                         input_steps=[mask_dilation, mask_resize],
                         adapter={'images': ([(mask_dilation.name, 'dilated_images')]),
                                  'probabilities': ([(mask_resize.name, 'resized_images')]),
                                  },
                         cache_dirpath=config.env.cache_dirpath, **kwargs)

    return score_builder


PIPELINES = {'retinanet': {'train': partial(retinanet, train_mode=True),
                           'inference': partial(retinanet, train_mode=False),
                           },
             }
