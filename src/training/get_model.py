import torch.nn as nn
from torchvision import models


def build_head(in_features, num_classes, head_type="simple", dropout=0.3):
    if head_type == "simple":
        return nn.Linear(in_features, num_classes)

    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    raise ValueError(
        f"Unsupported head_type: '{head_type}'. Available options: ['simple', 'mlp']"
    )


def get_model(model_name="resnet18", num_classes=2, pretrained=True, head_type="simple", dropout=0.3):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = build_head(in_features, num_classes, head_type, dropout)

    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier.in_features
        model.classifier = build_head(in_features, num_classes, head_type, dropout)

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. Available models: ['resnet18', 'densenet121']"
        )

    return model