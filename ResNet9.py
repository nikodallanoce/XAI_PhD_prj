from typing import Tuple, Any

import torch.nn as nn  # for creating  neural networks
import torch.nn.functional as F  # for functions for calculating loss
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import Accuracy, MeanMetric, Precision, Recall
from torchvision.ops import DropBlock2d


class ResNet9(LightningModule):
    def __init__(self, in_channels, num_diseases):
        super().__init__()

        self.accuracy = Accuracy("multiclass", num_classes=num_diseases)
        self.train_accuracy = Accuracy("multiclass", num_classes=num_diseases)
        # self.precision = Precision("multiclass", num_classes=num_diseases)
        # self.recall = Recall("multiclass", num_classes=num_diseases)
        self.mean_loss = MeanMetric()

        self.conv1 = self.ConvBlock(in_channels, 64)
        self.conv2 = self.ConvBlock(64, 128, pool=True)  # out_dim : 128 x 64 x 64
        self.res1 = nn.Sequential(self.ConvBlock(128, 128),
                                  self.ConvBlock(128, 128))

        self.conv3 = self.ConvBlock(128, 256, pool=True)  # out_dim : 256 x 16 x 16
        self.conv4 = self.ConvBlock(256, 512, pool=True)  # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(self.ConvBlock(512, 512),
                                  self.ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    # convolution block with BatchNormalization
    @staticmethod
    def ConvBlock(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.BatchNorm2d(out_channels),
                  nn.GELU()]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)

    def forward(self, xb):  # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

    # def set_optimizer_scheduler(self, optimizer: torch.optim.Optimizer,
    #                             lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
    #     self.optimizer = optimizer
    #     self.lr_scheduler = lr_scheduler

    def configure_optimizers(self):
        # return [self.optimizer(self.parameters(), self.lr)], [
        #     {"scheduler": self.lr_scheduler(self.optimizer, self.lr,
        #                                     self.trainer.estimated_stepping_batches), "interval": "step"}]
        lr = 1.5e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
        #                                               total_iters=self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                                        total_steps=self.trainer.estimated_stepping_batches + 1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        # loss = F.cross_entropy(out, labels, label_smoothing=0.1)  # Calculate loss
        loss = F.cross_entropy(out, labels)
        self.train_accuracy.update(out, labels)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        # self.log("train/lr", self.lr_schedulers().get_last_lr(), prog_bar=False, sync_dist=True)
        return loss

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     self.lr_schedulers().step()

    def on_train_epoch_end(self) -> None:
        self.log("train/accuracy", self.train_accuracy.compute(), sync_dist=True)
        self.train_accuracy.reset()

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/accuracy", acc, sync_dist=True)
        return {"loss": loss, "accuracy": acc}
