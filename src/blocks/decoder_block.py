import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU(), use_bn=True):
        super().__init__()
        layers = []

        # ConvTranspose2d 
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))

        # BatchNorm 
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation 
        if activation is not None:
            layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_connection=None):
        x = self.block(x)
        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
        return x
