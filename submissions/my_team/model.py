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
    A small ResNet-style block.

    The main path learns new features with two convolution layers.
    The shortcut path keeps the original input information and adds it back.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Run the residual block.

        Args:
            x: Input tensor with shape [batch_size, in_channels, height, width].

        Returns:
            Tensor with shape [batch_size, out_channels, new_height, new_width].

        The output is:
            convolution path output + shortcut path output
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
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # Stage 1: Input size (32, 224, 224) -> Output size (32, 224, 224)
        self.stage1 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=32, stride=1),
            ResidualBlock(in_channels=32, out_channels=32, stride=1)
        )
        # Stage 2: Input size (32, 224, 224) -> Output size (64, 112, 112)
        self.stage2 = nn.Sequential(
            ResidualBlock(in_channels=32, out_channels=64, stride=2),
            ResidualBlock(in_channels=64, out_channels=64, stride=1)
        )
        # Stage 3: Input size (64, 112, 112) -> Output size (128, 56, 56)
        self.stage3 = nn.Sequential(
            ResidualBlock(in_channels=64, out_channels=128, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, stride=1)
        )
        # Stage 4: Input size (128, 56, 56) -> Output size (256, 28, 28)
        self.stage4 = nn.Sequential(
            ResidualBlock(in_channels=128, out_channels=256, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, stride=1)
        )
        # Global Average Pooling reduces (256, 28, 28) to (256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Final classifier: 256 extracted features -> 20 class logits
        self.classifier = nn.Linear(in_features=256, out_features=20)

    def forward(self, x):
        """
        Run a forward pass through the ResNet-style CNN.

        Args:
            x: Input image batch as a tensor with shape
               [batch_size, 3, 224, 224]. The 3 channels are RGB.

        Returns:
            A tensor with shape [batch_size, 20]. Each row contains raw
            class scores, called logits, for the 20 hackathon classes.

        Flow:
            1. The stem converts RGB pixels into 32 low-level feature maps.
            2. Four residual stages extract deeper visual features.
            3. Adaptive average pooling summarizes each feature map.
            4. The classifier converts the final 256 features into 20 logits.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
