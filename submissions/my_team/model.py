"""
Student model architecture.

Required behavior:
    input:  torch.Tensor of shape [batch_size, 3, height, width]
    output: torch.Tensor of shape [batch_size, 20]
"""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    Small helper block:
        convolution -> batch normalization -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=1):
        super(ConvBNReLU, self).__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV2-style block.

    It first expands the number of channels, then applies a cheap depthwise
    spatial convolution, then projects back to the requested output channels.
    If the input and output shapes match, it adds a residual skip connection.
    """

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(InvertedResidualBlock, self).__init__()

        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels

        layers = []

        if expansion != 1:
            # Pointwise expansion: more channels for richer feature mixing
            layers.append(
                ConvBNReLU(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=1
                )
            )

        # Depthwise convolution: one spatial filter per channel, much cheaper
        # than a full convolution at the same channel count.
        layers.append(
            ConvBNReLU(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=stride,
                groups=hidden_channels
            )
        )

        # Pointwise projection: combine channels and choose output width.
        # No ReLU here; MobileNetV2 uses a linear bottleneck to avoid losing
        # information in the narrow output representation.
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        )

        self.conv_path = nn.Sequential(*layers)

    def forward(self, x):
        """
        Run the inverted residual block.

        Args:
            x: Tensor with shape [batch_size, in_channels, height, width].

        Returns:
            Tensor with shape [batch_size, out_channels, new_height, new_width].
        """
        out = self.conv_path(x)

        if self.use_residual:
            out = out + x

        return out


class ModelArchitecture(nn.Module):
    def __init__(self):
        super(ModelArchitecture, self).__init__()

        # Stem: Input size (3, 224, 224) -> Output size (24, 112, 112)
        self.stem = ConvBNReLU(
            in_channels=3,
            out_channels=24,
            kernel_size=3,
            stride=2
        )

        # Stage 1: Input size (24, 112, 112) -> Output size (24, 112, 112)
        self.stage1 = nn.Sequential(
            InvertedResidualBlock(
                in_channels=24,
                out_channels=24,
                stride=1,
                expansion=1
            )
        )

        # Stage 2: Input size (24, 112, 112) -> Output size (32, 56, 56)
        self.stage2 = nn.Sequential(
            InvertedResidualBlock(
                in_channels=24,
                out_channels=32,
                stride=2,
                expansion=4
            ),
            InvertedResidualBlock(
                in_channels=32,
                out_channels=32,
                stride=1,
                expansion=4
            )
        )

        # Stage 3: Input size (32, 56, 56) -> Output size (48, 28, 28)
        self.stage3 = nn.Sequential(
            InvertedResidualBlock(
                in_channels=32,
                out_channels=48,
                stride=2,
                expansion=4
            ),
            InvertedResidualBlock(
                in_channels=48,
                out_channels=48,
                stride=1,
                expansion=4
            )
        )

        # Stage 4: Input size (48, 28, 28) -> Output size (80, 14, 14)
        self.stage4 = nn.Sequential(
            InvertedResidualBlock(
                in_channels=48,
                out_channels=80,
                stride=2,
                expansion=4
            ),
            InvertedResidualBlock(
                in_channels=80,
                out_channels=80,
                stride=1,
                expansion=4
            ),
            InvertedResidualBlock(
                in_channels=80,
                out_channels=80,
                stride=1,
                expansion=4
            )
        )

        # Stage 5: Input size (80, 14, 14) -> Output size (128, 14, 14)
        self.stage5 = nn.Sequential(
            InvertedResidualBlock(
                in_channels=80,
                out_channels=128,
                stride=1,
                expansion=4
            ),
            InvertedResidualBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
                expansion=4
            )
        )

        # Final feature mixing: (128, 14, 14) -> (256, 14, 14)
        self.final_conv = ConvBNReLU(
            in_channels=128,
            out_channels=256,
            kernel_size=1,
            stride=1
        )

        # Global Average Pooling reduces (256, 14, 14) to (256, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Small classifier head: 256 features -> 20 class logits
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256, out_features=20)
        )

    def forward(self, x):
        """
        Run a forward pass through the lightweight MobileNetV2-style CNN.

        Args:
            x: Input image batch as a tensor with shape
               [batch_size, 3, 224, 224]. The 3 channels are RGB.

        Returns:
            A tensor with shape [batch_size, 20]. Each row contains raw
            class scores, called logits, for the 20 hackathon classes.

        Flow:
            1. The stem downsamples the image and extracts simple features.
            2. Inverted residual stages learn richer features efficiently.
            3. A final 1x1 convolution mixes high-level features.
            4. Adaptive average pooling summarizes each feature map.
            5. The classifier converts the final 256 features into 20 logits.
        """
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.final_conv(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
