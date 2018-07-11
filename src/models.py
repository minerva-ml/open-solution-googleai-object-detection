import torch
from torch.autograd import Variable
from torch import optim

from .steppy.pytorch.callbacks import CallbackList, TrainingMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping, NeptuneMonitor, ValidationMonitor
from .steppy.pytorch.models import Model

from .retinanet import RetinaNet, RetinaLoss


class Retina(Model):
    def __init__(self, architecture_config, training_config, callbacks_config):
        """
        """
        super().__init__(architecture_config, training_config, callbacks_config)
        self.encoder_depth = self.architecture_config['model_params']['encoder_depth']
        self.num_classes = self.architecture_config['model_params']['num_classes']
        self.pretrained = self.architecture_config['model_params']['pretrained']
        self.set_model()
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('FocalLoss', RetinaLoss(num_classes=self.num_classes), 1.0)]
        self.callbacks = callbacks(self.callbacks_config)

    def transform(self, datagen, *args, **kwargs):
        self.model.eval()

        batch_gen, steps = datagen
        boxes = []
        labels = []
        for batch_id, data in enumerate(batch_gen):
            if isinstance(data, list):
                X = data[0]
            else:
                X = data

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
            else:
                X = Variable(X, volatile=True)

            boxes_batch, labels_batch = self.model(X)
            boxes.extend([box for box in boxes_batch])
            labels.extend([label for label in labels_batch])
            if batch_id == steps:
                break
        self.model.train()
        outputs = {'boxes_prediction': boxes,
                   'labels_prediction': labels}
        return outputs

    def set_model(self):
        self.model = RetinaNet(encoder_depth=self.encoder_depth,
                               num_classes=self.num_classes,
                               pretrained_encoder=self.pretrained)

    def _initialize_model_weights(self):
        # TODO: implement weights initialization from Retina paper
        self.model.freeze_bn()


def weight_regularization(model, regularize, weight_decay_conv2d):
    if regularize:
        parameter_list = [{'params': model.parameters(), 'weight_decay': weight_decay_conv2d}]
    else:
        parameter_list = [model.parameters()]
    return parameter_list


def callbacks(callbacks_config):
    experiment_timing = ExperimentTiming(**callbacks_config['experiment_timing'])
    model_checkpoints = ModelCheckpoint(**callbacks_config['model_checkpoint'])
    lr_scheduler = ExponentialLRScheduler(**callbacks_config['exp_lr_scheduler'])
    training_monitor = TrainingMonitor(**callbacks_config['training_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitor(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, early_stopping, neptune_monitor,
                   ])
