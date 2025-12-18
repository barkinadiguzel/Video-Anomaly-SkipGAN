import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, disc_real, disc_fake):
        real_labels = torch.ones_like(disc_real)
        fake_labels = torch.zeros_like(disc_fake)

        loss_real = self.bce_loss(disc_real, real_labels)
        loss_fake = self.bce_loss(disc_fake, fake_labels)
        return loss_real + loss_fake
