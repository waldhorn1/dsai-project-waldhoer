"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MyModel(nn.Module):
    def __init__(self, n_in_channels: int):
        super(MyModel, self).__init__()
        
        # Encoder
        self.inc = DoubleConv(n_in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck
        self.bot = DoubleConv(256, 512)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(384, 256) # 384 = 256 (von up1) + 128 (von skip connection)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(192, 128) # 192 = 128 (von up2) + 64 (von skip connection)

        self.outc = nn.Conv2d(128, 3, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, 4, 100, 100]
        
        x1 = self.inc(x)        # -> [B, 64, 100, 100]
        x2 = self.down1[0](x1)  # MaxPool
        x2 = self.down1[1](x2)  # DoubleConv -> [B, 128, 50, 50]
        
        x3 = self.down2[0](x2)  # MaxPool
        x3 = self.down2[1](x3)  # DoubleConv -> [B, 256, 25, 25]
        
        x4 = self.bot(x3)       # -> [B, 512, 25, 25]
        
        # Up 1
        x = self.up1(x4)        # -> [B, 256, 50, 50]
        x = torch.cat([x2, x], dim=1) # Skip Connection concatenation
        x = self.conv_up1(x)    # -> [B, 256, 50, 50]
        
        # Up 2
        x = self.up2(x)         # -> [B, 128, 100, 100]
        x = torch.cat([x1, x], dim=1) # Skip Connection concatenation
        x = self.conv_up2(x)    # -> [B, 128, 100, 100]
        
        logits = self.outc(x)   # -> [B, 3, 100, 100]
        
        # WICHTIG: Sigmoid zwingt den Output zwischen 0.0 und 1.0 (Pixelwerte)
        return torch.sigmoid(logits)