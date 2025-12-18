import torch
import torch.nn as nn

class ContextualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, x_hat):
        return self.l1_loss(x_hat, x)
