import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from lightning.pytorch import Trainer

from CustomEfficientNet import CustomEfficientNet
from ResNet9 import ResNet9

if __name__ == '__main__':
    data_dir = "New Plant Diseases Dataset(Augmented)"
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")
    diseases = os.listdir(train_dir)

    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid = ImageFolder(valid_dir, transform=transforms.ToTensor())
    test = ImageFolder(test_dir, transform=transforms.ToTensor())
    for k in test.class_to_idx:
        test.class_to_idx[k] = valid.class_to_idx[k]
    test.samples = test.make_dataset(test_dir, test.class_to_idx,
                                     extensions=(
                                     ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"))

    batch_size = 128

    # DataLoaders for training and validation
    train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_dl = DataLoader(valid, batch_size, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test, batch_size, num_workers=4, pin_memory=True)

    # model = ResNet9(in_channels=3, num_diseases=len(train.classes))
    # model = CustomEfficientNet(num_diseases=len(train.classes))
    model = CustomEfficientNet.load_from_checkpoint("checkpoints/efficient_net_scratch-v15.ckpt")

    compiled_model = torch.compile(model, mode="max-autotune")
    trainer = Trainer(devices=[0,1])
    # metrics = trainer.validate(model, valid_dl)
    trainer.validate(model, valid_dl)
    # print(metrics)
