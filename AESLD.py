import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from XAI_course_2024.externals.ABELE.autoencoders.autoencoder import Autoencoder
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        # self.flatten = nn.Flatten()

    def forward(self, x):
        x1 = F.gelu(self.conv1(x))
        x2 = F.gelu(self.conv2(x1))
        x3 = F.gelu(self.conv3(x2))
        x4 = F.gelu(self.conv4(x3))
        # x4 = self.flatten(x4)
        return x4


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        self.t_conv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x4):
        xd = F.gelu(self.t_conv1(x4))
        xd = self.dropout(xd)
        xd = F.gelu(self.t_conv2(xd))
        xd = self.dropout(xd)
        xd = F.gelu(self.t_conv3(xd))
        xd = self.dropout(xd)
        x_out = F.sigmoid(self.t_conv4(xd))
        return x_out


class AE(LightningModule, Autoencoder):

    def __init__(self):
        super(AE, self).__init__()
        self.latent_dim_shape = (64, 32, 32)
        self.latent_dim = self.latent_dim_shape[0] * self.latent_dim_shape[1] * self.latent_dim_shape[2]  # latent_dim
        self.discriminator = None
        ## encoder layers ##
        self.encoder = Encoder()

        ## decoder layers ##
        self.decoder = Decoder()

    def forward(self, x):
        x4 = self.encoder(x)
        # x4 = self.pool(x4)
        x_out = self.decoder(x4)
        return x_out

    def encode(self, image):
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                ris = self.encoder(torch.from_numpy(image).to(self.device))
                ris = ris.detach().cpu().view(image.shape[0], -1).numpy()
                # ris = np.expand_dims(ris, axis=0)
            elif isinstance(image, torch.Tensor):
                ris = self.encoder(image)
                ris = ris.unsqueeze(0)
                # ris = np.expand_dims(ris, axis=0)
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
        return ris

    def decode(self, batch):
        with torch.no_grad():
            if len(batch.shape) == 2 and batch.shape[-1] == self.latent_dim:
                batch_shape = batch.shape
                s1, s2, s3 = self.latent_dim_shape
                if isinstance(batch, np.ndarray):
                    ris = self.decoder(
                        torch.from_numpy(batch).view(batch_shape[0], s1, s2, s3).to(self.device).float())
                    ris = ris.detach().to("cpu").numpy()
                else:
                    ris = batch.view(batch_shape[0], s1, s2, s3)
            else:
                ris = self.decoder(batch)
        return ris

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x, x_hat)
        # loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def training_step(self, batch):
        # image, _ = batch
        # out = self(image)
        # loss = torch.nn.functional.mse_loss(out, image)
        loss = self._get_reconstruction_loss(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch):
        # image, _ = batch
        # out = self(image)
        # loss = torch.nn.functional.mse_loss(out, image)
        loss = self._get_reconstruction_loss(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self):
        lr = 2e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.trainer.estimated_stepping_batches + 1,
                                                               eta_min=1e-5)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     img = torch.randn(2, 3, 256, 256)
#     img = img.to(device)
#     # Initialize the autoencoder
#     # model = AE(512).to(device)
#     model = AE().to(device)  # .load_from_checkpoint("./checkpoints_ae/ae_g2c.ckpt")
#     ris = model(img)
#     r = model.encode(img.to("cpu").numpy())
#     model.decode(model.encoder(img).reshape(8, -1))
#
#     print()
