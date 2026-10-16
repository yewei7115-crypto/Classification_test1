from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if num_classes != 10:
        raise ValueError("CIFAR-10 requires num_classes=10")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    # Adapt the ImageNet architecture to 32x32 CIFAR images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

