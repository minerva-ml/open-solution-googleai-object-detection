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
    'loader': {'dataset_params': {'h_pad': params.h_pad,
                                  'w_pad': params.w_pad,
                                  'h': params.image_h,
                                  'w': params.image_w,
                                  'pad_method': params.pad_method,
                                  'images_dir': None,
                                  'sample_size': params.training_sample_size
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
        'architecture_config': {'model_params': {'n_filters': params.n_filters,
                                                 'conv_kernel': params.conv_kernel,
                                                 'pool_kernel': params.pool_kernel,
                                                 'pool_stride': params.pool_stride,
                                                 'repeat_blocks': params.repeat_blocks,
                                                 'batch_norm': params.use_batch_norm,
                                                 'dropout': params.dropout_conv,
                                                 'in_channels': params.image_channels,
                                                 'out_channels': params.channels_per_output,
                                                 'nr_outputs': params.nr_unet_outputs,
                                                 'encoder_depth': params.encoder_depth,
                                                 'num_classes': params.num_classes,
                                                 'pretrained': params.pretrained
                                                 },
                                'optimizer_params': {'lr': params.lr,
                                                     },
                                'regularizer_params': {'regularize': True,
                                                       'weight_decay_conv2d': params.l2_reg_conv,
                                                       },
                                'weights_init': {'function': 'he',
                                                 }
                                },
        'training_config': {'epochs': params.epochs_nr,
                            },
        'callbacks_config': {
            'model_checkpoint': {
                'filepath': os.path.join(GLOBAL_CONFIG['exp_root'], 'checkpoints', 'unet', 'best.torch'),
                'epoch_every': 1,
                'minimize': not params.validate_with_map
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
                               'minimize': not params.validate_with_map
                               },
        },
    },
    'postprocessing': {
        'prediction_formatter': {
            'image_size': (params.image_h, params.image_w)
        }
    },
})
