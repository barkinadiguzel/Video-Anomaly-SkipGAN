import torch
import torch.nn as nn

class LatentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, features_real, features_fake):
        return self.mse_loss(features_fake, features_real)
