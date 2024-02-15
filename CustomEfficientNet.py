from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import Accuracy
from torchvision.models import EfficientNet, efficientnet_b0


class CustomEfficientNet(LightningModule):

    def __init__(self, num_diseases, weights=None):
        super().__init__()

        model = efficientnet_b0(weights)
        model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_diseases,
                                              bias=True)
        self.model = model
        self.accuracy = Accuracy("multiclass", num_classes=num_diseases)
        self.train_accuracy = Accuracy("multiclass", num_classes=num_diseases)
        self.save_hyperparameters()

    def configure_optimizers(self):
        # return [self.optimizer(self.parameters(), self.lr)], [
        #     {"scheduler": self.lr_scheduler(self.optimizer, self.lr,
        #                                     self.trainer.estimated_stepping_batches), "interval": "step"}]
        lr = 0.8e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        # optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, momentum=0.89, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.trainer.estimated_stepping_batches + 1,
                                                               eta_min=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        #                                                        T_0=(self.trainer.estimated_stepping_batches + 1)//2)
        # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        # loss = F.cross_entropy(out, labels, label_smoothing=0.1)  # Calculate loss
        loss = F.cross_entropy(out, labels, label_smoothing=0.1)
        self.train_accuracy.update(out, labels)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        # self.log("train/lr", self.lr_schedulers().get_last_lr(), prog_bar=False, sync_dist=True)
        return loss

    def forward(self, batch) -> Any:
        return self(batch)

    # def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
    #     self.lr_schedulers().step()

    def on_train_epoch_end(self) -> None:
        self.log("train/accuracy", self.train_accuracy.compute(), sync_dist=True)
        self.train_accuracy.reset()

    def validation_step(self, batch):
        images, labels = batch
        out = self.model(images)  # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)
        self.log("val/loss", loss, sync_dist=True)
        self.log("val/accuracy", acc, sync_dist=True)
        return {"loss": loss, "accuracy": acc}
