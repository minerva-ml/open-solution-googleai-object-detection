import os

from attrdict import AttrDict

from .utils import NeptuneContext

neptune_ctx = NeptuneContext()
params = neptune_ctx.params
ctx = neptune_ctx.ctx

ID_COLUMN = 'ImageID'
LABEL_COLUMN = 'LabelName'
SEED = 1234
CATEGORY_IDS = [None, 100]
CATEGORY_LAYERS = [1, 19]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ASPECT_RATIOS = [1/2., 1/1., 2/1.]
# SCALE_RATIOS = [1., pow(2,1/3.), pow(2,2/3.)]
ASPECT_RATIOS = [1 / 1.]
SCALE_RATIOS = [1., pow(2, 1 / 2.)]

GLOBAL_CONFIG = {'exp_root': params.experiment_dir,
                 'load_in_memory': params.load_in_memory,
                 'num_workers': params.num_workers,
                 'num_classes': 2,
                 'img_H-W': (params.image_h, params.image_w),
                 'batch_size_train': params.batch_size_train,
                 'batch_size_inference': params.batch_size_inference,
                 'loader_mode': params.loader_mode,
                 'stream_mode': params.stream_mode
                 }

SOLUTION_CONFIG = AttrDict({
    'env': {'cache_dirpath': params.experiment_dir},
    'execution': GLOBAL_CONFIG,

    'label_encoder': {'colname': LABEL_COLUMN
                      },
    'loader': {'dataset_params': {'h': params.image_h,
                                  'w': params.image_w,
                                  'pad_method': params.pad_method,
                                  'images_dir': None,
                                  'sample_size': params.training_sample_size,
                                  'data_encoder': {'aspect_ratios': ASPECT_RATIOS,
                                                   'scale_ratios': SCALE_RATIOS,
                                                   'num_anchors': len(ASPECT_RATIOS) * len(SCALE_RATIOS)}
                                  },
               'loader_params': {'training': {'batch_size': params.batch_size_train,
                                              'num_workers': params.num_workers,
                                              'pin_memory': params.pin_memory
                                              },
                                 'inference': {'batch_size': params.batch_size_inference,
                                               'shuffle': False,
                                               'num_workers': params.num_workers,
                                               'pin_memory': params.pin_memory
                                               },
                                 },
               },

    'retinanet': {
        'architecture_config': {'model_params': {'encoder_depth': params.encoder_depth,
                                                 'num_classes': params.num_classes,
                                                 'num_anchors': len(ASPECT_RATIOS) * len(SCALE_RATIOS),
                                                 'pretrained_encoder': params.pretrained_encoder
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_conv,
                                                       },
                                'weights_init': {'function': 'he',
                                                 'pi': params.pi
                                                 }
                                },
        'training_config': {'epochs': params.epochs_nr,
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'retinanet', 'best.torch'),
                'epoch_every': 1,
                # 'minimize': not params.validate_with_map
            },
            'exp_lr_scheduler': {'gamma': params.gamma,
                                 'epoch_every': 1},
            'plateau_lr_scheduler': {'lr_factor': params.lr_factor,
                                     'lr_patience': params.lr_patience,
                                     'epoch_every': 1},
            'training_monitor': {'batch_every': 1,
                                 'epoch_every': 1},
            'experiment_timing': {'batch_every': 10,
                                  'epoch_every': 1},
            'validation_monitor': {
                'epoch_every': 1,
                # 'data_dir': params.train_imgs_dir,
                # 'validate_with_map': params.validate_with_map,
                # 'small_annotations_size': params.small_annotations_size,
            },
            'neptune_monitor': {'model_name': 'unet',
                                # 'image_nr': 16,
                                # 'image_resize': 0.2,
                                # 'outputs_to_plot': params.unet_outputs_to_plot
                                },
            'early_stopping': {'patience': params.patience,
                               # 'minimize': not params.validate_with_map
                               },
        },
    },
    'postprocessing': {
        'data_decoder': {
            'input_size': (params.image_h, params.image_w),
            'aspect_ratios': ASPECT_RATIOS,
            'scale_ratios': SCALE_RATIOS,
            'num_anchors': len(ASPECT_RATIOS) * len(SCALE_RATIOS)
        },
        'prediction_formatter': {
            'image_size': (params.image_h, params.image_w)
        }
    },
})
