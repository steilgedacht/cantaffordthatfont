
import torch.nn as nn
from torchvision import models


class GoogleFontsClassifier_ResNet50(nn.Module):
    def __init__(self, output_classes, dropout=0.5):
        super(GoogleFontsClassifier_ResNet50, self).__init__()

        resnet50 = models.resnet50(weights="IMAGENET1K_V1")
        # Change first conv layer to accept 1 channel (grayscale)
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the fully connected layer
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, output_classes)
        )
        self.resnet = resnet50

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, H, W)
        x = self.resnet(x)
        return x


class GoogleFontsClassifier_ResNet34(nn.Module):
    def __init__(self, output_classes, dropout=0.5):
        super(GoogleFontsClassifier_ResNet34, self).__init__()

        resnet34 = models.resnet34(weights="IMAGENET1K_V1")
        # Change first conv layer to accept 1 channel (grayscale)
        resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the fully connected layer
        num_ftrs = resnet34.fc.in_features
        resnet34.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, output_classes)
        )
        self.resnet = resnet34

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, H, W)
        x = self.resnet(x)
        return x


class GoogleFontsClassifier_ResNet18(nn.Module):
    def __init__(self, output_classes, dropout=0.5):
        super(GoogleFontsClassifier_ResNet18, self).__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")
        # Change first conv layer to accept 1 channel (grayscale)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace the fully connected layer
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, output_classes)
        )
        self.resnet = resnet

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, H, W)
        x = self.resnet(x)
        return x

class GoogleFontsClassifier(nn.Module):
    def __init__(self, output_classes, dropout=0.5):
        super(GoogleFontsClassifier, self).__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        self.resnet = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.avgpool,
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, H, W)
        x = self.resnet(x)
        return x
