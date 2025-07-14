
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
