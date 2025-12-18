import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        feature_channels=(64, 128, 256, 512),
        use_bn=True,
        activation=nn.LeakyReLU(0.2, inplace=True)
    ):
        super().__init__()

        layers = []
        prev_c = in_channels

        for out_c in feature_channels:
            layers.append(
                nn.Conv2d(
                    prev_c,
                    out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=not use_bn
                )
            )

            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))

            layers.append(activation)
            prev_c = out_c

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
