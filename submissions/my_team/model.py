
"""
    Student model architecture.

    Students should define their model here.

    Required behavior:
        input:  torch.Tensor of shape [batch_size, 3, height, width]
        output: torch.Tensor of shape [batch_size, 20]
    """
import torch
import torch.nn as nn


class ModelArchitecture(nn.Module):
    def __init__(self):
        super(ModelArchitecture, self).__init__()

        # Block 1: Input size (3, 128, 128) -> Output size (32, 64, 64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Input size (32, 64, 64) -> Output size (64, 32, 32)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Input size (64, 32, 32) -> Output size (128, 16, 16)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: Input size (128, 16, 16) -> Output size (256, 8, 8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 5: Input size (256, 8, 8) -> Output size (512, 4, 4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Average Pooling reduces the (512, 4, 4) spatial dimensions to (512, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected classifier with updated input features
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            # Keep Dropout at 50% to prevent overfitting on this deeper network
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=20)
        )

    def forward(self, x):
        # Pass input through the 5 convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Pool and flatten for the dense layers
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Generate the final 20 class logits
        x = self.classifier(x)

        return x