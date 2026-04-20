import torch.nn as nn
from torchvision import models


def get_model(model_name="resnet18", num_classes=2, pretrained=True):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model: '{model_name}'. Available models: ['resnet18', 'densenet121']")

    return model