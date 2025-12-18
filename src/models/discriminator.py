import torch
import torch.nn as nn
from src.layers.conv_block import ConvBlock  

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Convolutional feature extractor
        self.conv1 = ConvBlock(in_channels, base_channels, kernel_size=4, stride=2, padding=1, batch_norm=False)  # 64x64 -> 32x32
        self.conv2 = ConvBlock(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = ConvBlock(base_channels*2, base_channels*4, kernel_size=4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv4 = ConvBlock(base_channels*4, base_channels*8, kernel_size=4, stride=2, padding=1)  # 8x8 -> 4x4

        # Final classification layer
        self.final = nn.Conv2d(base_channels*8, 1, kernel_size=4, stride=1, padding=0)  # 4x4 -> 1x1

        # Sigmoid for probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        features = out  # latent features for latent loss

        out = self.final(out)
        out = self.sigmoid(out)  # probability between 0-1
        out = out.view(out.size(0), -1)  # flatten to [batch, 1]
        return out, features
