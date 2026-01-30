"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch

class MyModel(torch.nn.Module):
    def __init__(self, n_in_channels: int):
        super().__init__()

        # Encoder path
        self.cnn1 = torch.nn.Conv2d(in_channels=n_in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()
        
        self.cnn2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        
        # Middle layers
        self.cnn3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU()
        
        self.cnn4 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.relu4 = torch.nn.ReLU()
        
        # Decoder path
        self.cnn5 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.relu5 = torch.nn.ReLU()
        
        self.cnn6 = torch.nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Skip connection for later use
        x_skip = x
        
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.cnn5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = self.cnn6(x)
        
        return x
