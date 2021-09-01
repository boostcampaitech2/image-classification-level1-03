import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm


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


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet18(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet34(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet101(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet152(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class VitBaseResnet50_384(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net  = timm.create_model(model_name="vit_base_resnet50_384", num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.net(x)


class VitBasePatch16_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net  = timm.create_model(model_name="vit_base_patch16_224", num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.net(x)


class MaskResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class GenderResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)


class AgeResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = torchvision.models.resnet50(pretrained=True)
        self.net.fc = nn.Linear(in_features=self.net.fc.weight.shape[1], out_features=num_classes)

    def forward(self, x):
        return self.net(x)
