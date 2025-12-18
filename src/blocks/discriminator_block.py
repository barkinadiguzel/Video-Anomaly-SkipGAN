import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2), use_bn=True):
        super().__init__()
        layers = []

        # Conv2d 
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))

        # BatchNorm 
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activation 
        if activation is not None:
            layers.append(activation)

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
