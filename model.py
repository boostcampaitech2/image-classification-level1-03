import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet import efficientnet_b3


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        self.feature = efficientnet_b3(pretrained=True, progress=True, num_classes=num_classes).features
        self.classifier1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536, 3)) # mask classifier
        self.classifier2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536, 2)) # gender classifier
        self.classifier3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True) # age classifier
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.mean([2, 3])
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x3 = self.classifier3(x)
        return (x1, x2, x3)

