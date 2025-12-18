import torch
import torch.nn as nn
from src.blocks.encoder_block import EncoderBlock
from src.blocks.decoder_block import DecoderBlock

class Generator(nn.Module):
    def __init__(self, in_channels=3, feature_channels=[64, 128, 256, 512]):
        super().__init__()

        # Encoder stack
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for ch in feature_channels:
            self.encoder_blocks.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        # Decoder stack
        self.decoder_blocks = nn.ModuleList()
        rev_channels = feature_channels[::-1]
        for i in range(len(rev_channels)-1):
            # decoder input = bottleneck + skip connection
            self.decoder_blocks.append(
                DecoderBlock(rev_channels[i]*2, rev_channels[i+1])
            )

        # Final layer
        self.final_layer = nn.ConvTranspose2d(rev_channels[-1]*2, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward
        out = x
        for enc in self.encoder_blocks:
            out = enc(out)
            skip_connections.append(out)

        # Bottleneck
        out = skip_connections[-1]
        skip_connections = skip_connections[:-1][::-1]

        # Decoder forward
        for i, dec in enumerate(self.decoder_blocks):
            out = torch.cat([out, skip_connections[i]], dim=1)  # skip connection
            out = dec(out)

        # Final reconstruction
        out = torch.cat([out, skip_connections[-1]], dim=1)
        out = self.final_layer(out)
        out = torch.tanh(out)  # normalize to [-1,1]
        return out
