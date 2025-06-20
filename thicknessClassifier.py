import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from models.thicknessModel import ThicknessModel
from models.FocalLoss import FocalLoss


class ThicknessClassifier(pl.LightningModule):
    def __init__(self, pretrained=True, classification=False, backbone='', dropout=0.1, out=2, lr=0.03,
                 wd=0.01, momentum=0.1, gamma=0.99, focal_gamma=2):
        super().__init__()
        self.lr = lr

        self.wd = wd
        self.momentum = momentum
        self.gamma = gamma

        self.model = ThicknessModel(pretrained=pretrained, backbone=backbone, num_classes=out, dropout=dropout)

        self.classification = classification
        if self.classification:
            # self.criterion = nn.CrossEntropyLoss()
            self.criterion = FocalLoss(gamma=focal_gamma, alpha=0.25)
            self.validation_metric = Accuracy(task="multiclass", num_classes=out)
            self.train_metric = Accuracy(task="multiclass", num_classes=out)
            self.best_validation_metric = 0
        else:
            self.criterion = nn.MSELoss()
            self.validation_metric = MeanAbsoluteError()
            self.train_metric = MeanAbsoluteError()

        self.layerhook = []

        self.gradients = None
        self.tensorhook = []
        self.selected_out = None
        self.save_hyperparameters()
        self.validation_predictions = []
        self.validation_gt = []
        print(self.model)
        # self.layerhook.append(self.resnet.layer4.register_forward_hook(self.forward_hook()))

    def training_step(self, batch, batch_idx):
        input_batch, label_batch, _ = batch

        predicted = self.model(input_batch)

        loss = self.criterion(predicted.squeeze(), label_batch)
        # print(loss)

        self.log("train_loss", loss)
        self.train_metric(predicted.squeeze(), label_batch)
        self.log('train_metric', self.train_metric, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_batch, label_batch, _ = batch

        predicted = self.model(input_batch)

        loss = self.criterion(predicted.squeeze(), label_batch)
        # print(loss)

        self.validation_metric(predicted.squeeze(), label_batch)
        self.log('validation_metric', self.validation_metric, on_epoch=True, on_step=False)
        self.log("validation_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.validation_predictions.append(predicted.squeeze())
        self.validation_gt.append(label_batch)

    def on_validation_epoch_end(self):
        tb = self.logger.experiment
        preds = torch.concatenate(self.validation_predictions, 0)
        gt = torch.concatenate(self.validation_gt, 0)
        epoch_acc = self.validation_metric.compute()
        print("on epoch end: ", self.best_validation_metric, epoch_acc)
        if self.best_validation_metric < epoch_acc:
            self.best_validation_metric = epoch_acc
            self.log("best_validation_metric", self.best_validation_metric)
        malignant = preds[(gt == 1).nonzero()]
        bening = preds[(gt == 0).nonzero()]
        if malignant.size()[0] > 0 and bening.size()[0] > 0:
            tb.add_histogram("malignant", malignant, self.current_epoch)
            tb.add_histogram("bening", bening, self.current_epoch)

        self.validation_predictions = []
        self.validation_gt = []

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.lr, weight_decay=self.wd, momentum=self.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        y = self.model(x)
        return y

    def activations_hook(self, grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))

        return hook
