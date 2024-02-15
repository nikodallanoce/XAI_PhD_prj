import os  # for working with files
import numpy as np  # for numerical computationss
import pandas as pd  # for working with dataframes
import torch  # Pytorch module
import matplotlib.pyplot as plt  # for plotting informations on graph and images using tensors
import torch.nn as nn  # for creating  neural networks
from torch.utils.data import DataLoader  # for dataloaders
from PIL import Image  # for checking images
import torch.nn.functional as F  # for functions for calculating loss
import torchvision.transforms as transforms  # for transforming images into tensors
from torchvision.utils import make_grid  # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
import os
from ResNet9 import ResNet9
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from CustomEfficientNet import CustomEfficientNet
from torchvision.models.efficientnet import EfficientNet_B0_Weights

torch.set_float32_matmul_precision('medium')


def show_image(image, label):
    print("Label :" + train.classes[label] + "(" + str(label) + ")")
    plt.imshow(image.permute(1, 2, 0))


# helper function to show a batch of training instances
def show_batch(data):
    for images, labels in data:
        fig, ax = plt.subplots(figsize=(30, 30))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break


if __name__ == '__main__':
    data_dir = "New Plant Diseases Dataset(Augmented)"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    diseases = os.listdir(train_dir)

    plants = []
    number_of_diseases = 0
    for plant in diseases:
        if plant.split('___')[0] not in plants:
            plants.append(plant.split('___')[0])
        if plant.split('___')[1] != 'healthy':
            number_of_diseases += 1

    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

    # Setting the seed value
    random_seed = 7
    torch.manual_seed(random_seed)
    # setting the batch size
    batch_size = 256

    # DataLoaders for training and validation
    train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)
    checkpoint = ModelCheckpoint(dirpath="./checkpoints", filename="efficient_net_scratch", monitor='val/loss',
                                 mode="min")
    early_stop = EarlyStopping(monitor="val/loss", mode="min", patience=20)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # show_batch(train_dl)

    # model = ResNet9(in_channels=3, num_diseases=len(train.classes))
    model = CustomEfficientNet(num_diseases=len(train.classes), weights=None)
    # model = ResNetCustom(num_classes=len(train.classes))
    compiled_model = torch.compile(model, mode="max-autotune")
    trainer = Trainer(accelerator="auto", devices="auto", precision="16-mixed", max_epochs=50,
                      accumulate_grad_batches=2,
                      logger=WandbLogger(name="EfficientNet_scratch", project="XAI", log_model=False), log_every_n_steps=10,
                      gradient_clip_val=1,
                      enable_progress_bar=True, callbacks=[checkpoint, lr_monitor, early_stop])
    trainer.fit(model, train_dl, valid_dl)
