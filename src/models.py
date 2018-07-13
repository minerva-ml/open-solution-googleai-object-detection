import torch
from torch.autograd import Variable
from torch import optim
from .steppy.pytorch.models import Model
from toolkit.pytorch_transformers.callbacks import CallbackList, TrainingMonitor, ModelCheckpoint, \
    ExperimentTiming, ExponentialLRScheduler, EarlyStopping, NeptuneMonitor, ValidationMonitor
from .steppy.pytorch.parallel import DataParallelCriterion

from .callbacks import NeptuneMonitorDetection, ValidationMonitorDetection
from .retinanet import RetinaNet, RetinaLoss


class Retina(Model):
    def __init__(self, architecture_config, training_config, callbacks_config, train_mode=False):
        """
        """
        super().__init__(architecture_config, training_config, callbacks_config)
        self.train_mode = train_mode
        self.encoder_depth = self.architecture_config['model_params']['encoder_depth']
        self.num_classes = self.architecture_config['model_params']['num_classes']
        self.pretrained = self.architecture_config['model_params']['pretrained']
        self.set_model()
        self.weight_regularization = weight_regularization
        self.optimizer = optim.Adam(self.weight_regularization(self.model, **architecture_config['regularizer_params']),
                                    **architecture_config['optimizer_params'])
        self.loss_function = [('FocalLoss', DataParallelCriterion(RetinaLoss(num_classes=self.num_classes)), 1.0)]
        self.callbacks = callbacks(self.callbacks_config)

    def transform(self, datagen, *args, **kwargs):
        if self.train_mode:
            return self

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

    def _fit_loop(self, data):
        X = data[0]
        targets_tensors = data[1:]

        if torch.cuda.is_available():
            X = Variable(X).cuda()
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor).cuda())
        else:
            X = Variable(X)
            targets_var = []
            for target_tensor in targets_tensors:
                targets_var.append(Variable(target_tensor))

        self.optimizer.zero_grad()
        outputs_batch = self.model(X)
        partial_batch_losses = {}

        if len(self.output_names) == 1:
            for (name, loss_function, weight), target in zip(self.loss_function, targets_var):
                batch_loss = loss_function(outputs_batch, target) * weight
        else:
            for (name, loss_function, weight), output, target in zip(self.loss_function, outputs_batch, targets_var):
                partial_batch_losses[name] = loss_function(output, target) * weight
            batch_loss = sum(partial_batch_losses.values())
        partial_batch_losses['sum'] = batch_loss
        batch_loss.backward()
        self.optimizer.step()

        return partial_batch_losses

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
    # validation_monitor = ValidationMonitorDetection(**callbacks_config['validation_monitor'])
    # neptune_monitor = NeptuneMonitorDetection(**callbacks_config['neptune_monitor'])
    validation_monitor = ValidationMonitor(**callbacks_config['validation_monitor'])
    neptune_monitor = NeptuneMonitor(**callbacks_config['neptune_monitor'])
    early_stopping = EarlyStopping(**callbacks_config['early_stopping'])

    return CallbackList(
        callbacks=[experiment_timing, training_monitor, validation_monitor,
                   model_checkpoints, lr_scheduler, early_stopping, neptune_monitor,
                   ])
