import torch
import torch.nn as nn
from src.model.generator import Generator
from src.model.discriminator import Discriminator

class SkipGANomaly(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, feature_channels=[64,128,256,512]):
        super().__init__()

        # Generator wraps encoder + decoder with skip connections
        self.generator = Generator(
            in_channels=in_channels,
            feature_channels=feature_channels
        )

        # Discriminator
        self.discriminator = Discriminator(
            data_dim=in_channels,
            hidden_dim=base_channels
        )

    def forward(self, x):
        # Generator forward
        x_hat = self.generator(x)  

        # Discriminator forward
        disc_out, disc_features = self.discriminator(x_hat)

        return x_hat, disc_out, disc_features
