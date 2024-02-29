from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy, MeanMetric, ConfusionMatrix
from torchvision.models import EfficientNet, efficientnet_b0


class CustomEfficientNet(LightningModule):

    def __init__(self, num_diseases, weights=None):
        super().__init__()

        model = efficientnet_b0(weights)

        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_diseases,
                                              bias=True)
        self.model = model
        self.train_accuracy = Accuracy("multiclass", num_classes=num_diseases)
        self.valid_accuracy = Accuracy("multiclass", num_classes=num_diseases)
        self.conf_matrix = ConfusionMatrix("multiclass", num_classes=num_diseases)
        self.mean = MeanMetric()
        self.save_hyperparameters()

    def configure_optimizers(self):
        lr = 8e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.trainer.estimated_stepping_batches + 1,
                                                               eta_min=1e-5)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch):
        images, labels = batch
        out = self.model(images)  # Generate predictions
        # loss = F.cross_entropy(out, labels, label_smoothing=0.1)  # Calculate loss
        loss = F.cross_entropy(out, labels, label_smoothing=0.1)
        self.train_accuracy(out, labels)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/accuracy", self.train_accuracy, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/lr", self.lr_schedulers().get_last_lr(), prog_bar=False, sync_dist=True)
        return loss

    def forward(self, batch) -> Any:
        return self.model(batch)

    def test_step(self, batch):
        images, labels = batch
        out = self.model(images)  # Generate prediction
        # classes = torch.argmax(out, dim=1)
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.valid_accuracy(out, labels)
        self.conf_matrix(out, labels)
        self.log("test/loss", loss, sync_dist=True)
        self.log("test/accuracy", acc, sync_dist=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, batch):
        images, labels = batch
        out = self.model(images)  # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.valid_accuracy(out, labels)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/accuracy", acc, sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss, "accuracy": acc}
