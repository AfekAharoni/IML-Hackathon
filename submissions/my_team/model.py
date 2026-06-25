"""
Student model architecture.

Required behavior:
    input:  torch.Tensor of shape [batch_size, 3, height, width]
    output: torch.Tensor of shape [batch_size, 20]
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Basic ResNet block:
        conv -> batch norm -> ReLU -> conv -> batch norm -> skip add -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Run one residual block.

        The convolution path learns new features, and the shortcut keeps the
        previous representation available for easier gradient flow.
        """
        out = self.conv_path(x)
        shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)
        return out


class ModelArchitecture(nn.Module):
    def __init__(self):
        super(ModelArchitecture, self).__init__()

        # Stem: Input size (3, 224, 224) -> Output size (32, 224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Pool 1: (32, 224, 224) -> (32, 112, 112)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 1: (32, 112, 112) -> (32, 112, 112)
        self.stage1 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=32),
            ResidualBlock(in_channels=32, out_channels=32)
        )

        # Pool 2: (32, 112, 112) -> (32, 56, 56)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: (32, 56, 56) -> (64, 56, 56)
        self.stage2 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=64),
            ResidualBlock(in_channels=64, out_channels=64)
        )

        # Pool 3: (64, 56, 56) -> (64, 28, 28)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: (64, 28, 28) -> (128, 28, 28)
        self.stage3 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128),
            ResidualBlock(in_channels=128, out_channels=128)
        )

        # Pool 4: (128, 28, 28) -> (128, 14, 14)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 4: (128, 14, 14) -> (256, 14, 14)
        self.stage4 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256),
            ResidualBlock(in_channels=256, out_channels=256)
        )

        # Global Average Pooling: (256, 14, 14) -> (256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier: 256 extracted features -> 20 class logits
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=20)
        )

    def forward(self, x):
        """
        Run a forward pass through a half-width ResNet-18 style CNN.

        Args:
            x: Input image batch as a tensor with shape
               [batch_size, 3, 224, 224]. The 3 channels are RGB.

        Returns:
            A tensor with shape [batch_size, 20]. Each row contains raw
            class scores, called logits, for the 20 hackathon classes.

        Flow:
            1. Stem extracts first low-level features.
            2. MaxPool layers reduce spatial size and increase receptive field.
            3. Residual stages learn deeper visual features.
            4. Global average pooling summarizes each feature map.
            5. The classifier converts 256 features into 20 logits.
        """
        x = self.stem(x)

        x = self.pool1(x)
        x = self.stage1(x)

        x = self.pool2(x)
        x = self.stage2(x)

        x = self.pool3(x)
        x = self.stage3(x)

        x = self.pool4(x)
        x = self.stage4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
