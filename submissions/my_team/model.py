"""
Student model architecture.

Required behavior:
    input:  torch.Tensor of shape [batch_size, 3, height, width]
    output: torch.Tensor of shape [batch_size, 20]
"""
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A standard Convolutional block with optional Max Pooling.
    Used to quickly reduce spatial dimensions and save compute time.
    """
    def __init__(self, in_channels, out_channels, pool=False):
        super(ConvBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleResBlock(nn.Module):
    """
    A lightweight Residual Block containing exactly 2 convolutional layers.
    Maintains spatial dimensions while learning deeper features.
    """
    def __init__(self, channels):
        super(SimpleResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Skip connection: adds the original input to the output of the convolutions
        return self.relu(self.net(x) + x)


class ModelArchitecture(nn.Module):
    def __init__(self):
        super(ModelArchitecture, self).__init__()

        # --- LAYER 1 ---
        # Stem: Input (3, 224, 224) -> Output (64, 112, 112)
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, pool=True)

        # --- LAYER 2 ---
        # Feature Extraction: (64, 112, 112) -> Output (128, 56, 56)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, pool=True)

        # --- LAYERS 3 & 4 ---
        # Residual Stage 1: (128, 56, 56) -> Output (128, 56, 56)
        self.res1 = SimpleResBlock(channels=128)

        # --- LAYER 5 ---
        # Downsample: (128, 56, 56) -> Output (256, 28, 28)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, pool=True)

        # --- LAYERS 6 & 7 ---
        # Residual Stage 2: (256, 28, 28) -> Output (256, 28, 28)
        self.res2 = SimpleResBlock(channels=256)

        # Global Average Pooling: Shrinks (256, 28, 28) into a flat (256, 1, 1) vector
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # --- LAYER 8 ---
        # Classifier: Maps the 256 features to the 20 hackathon classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=20)
        )

    def forward(self, x):
        """
        Forward pass optimized for maximum throughput.
        """
        # Sequential standard convolutions for fast downsampling
        x = self.conv1(x)
        x = self.conv2(x)

        # First residual block
        x = self.res1(x)

        # Final downsample
        x = self.conv3(x)

        # Second residual block
        x = self.res2(x)

        # Pool, flatten, and classify
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x