import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

ANIMAL_TYPE_CLASSES = [
    "Кіт",
    "Пес",
    "Бобер"
]

SIZE_CLASSES = [
    "Маленький",
    "Середній",
    "Великий"
]

COLOR_CLASSES = [
    "Чорне",
    "Біле",
    "Чорно-Біле",
    "Сіре",
    "Руде",
    "Коричневе",
    "Змішане"
]

class ShelterPetResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.type_head = nn.Linear(in_features, len(ANIMAL_TYPE_CLASSES))
        self.size_head = nn.Linear(in_features, len(SIZE_CLASSES))
        self.color_head = nn.Linear(in_features, len(COLOR_CLASSES))

    def forward(self, x):
        features = self.backbone(x)

        return {
            "animal_type": self.type_head(features),
            "size": self.size_head(features),
            "color": self.color_head(features)
        }