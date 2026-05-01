import torch
import torch.nn as nn
import torch.optim as optim
from src.training.get_model import get_model
from src.utils.logger import get_logger

logger = get_logger("classifier")


class ImageClassifier:
    def __init__(self, model_name="resnet18", num_classes=2, pretrained=True, head_type="simple",
                 dropout=0.3, learning_rate=1e-4, optimizer_name="adamw", weight_decay=1e-4, momentum=0.9, freeze_backbone=True, device=None):
        try:
            self.device = device or self._get_device()

            logger.info(
                f"Initializing {model_name} --> "
                f"pretrained={pretrained}, head={head_type}, "
                f"freeze_backbone={freeze_backbone}, device={self.device}"
            )

            self.model = get_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained, head_type=head_type, dropout=dropout).to(self.device)

            if freeze_backbone:
                self._freeze_backbone()

            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = self._build_optimizer(optimizer_name=optimizer_name, learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum)

            logger.info("ImageClassifier initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize ImageClassifier: {e}")
            raise

    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _freeze_backbone(self):
        logger.info("Freezing backbone parameters")

        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, "fc"):  # ResNet
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "classifier"):  # DenseNet
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Could not identify classifier head to unfreeze")

    def _build_optimizer(self, optimizer_name, learning_rate, weight_decay, momentum):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer_name = optimizer_name.lower()

        if optimizer_name == "adam":
            return optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return optim.SGD(trainable_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Available options: ['adam', 'adamw', 'sgd']" )

    def train_step(self, images, labels):
        if images is None or labels is None:
            raise ValueError("images or labels is None")

        self.model.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)

        return {
            "loss": loss.item(),
            "correct": correct,
            "total": total,
            "predictions": predictions.detach().cpu(),
            "labels": labels.detach().cpu()
        }

    def eval_step(self, images, labels):
        if images is None or labels is None:
            raise ValueError("images or labels is None")

        self.model.eval()

        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == labels).sum().item()
        total = labels.size(0)

        return {
            "loss": loss.item(),
            "correct": correct,
            "total": total,
            "predictions": predictions.detach().cpu(),
            "labels": labels.detach().cpu()
        }

    def predict(self, images):
        self.model.eval()

        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions.cpu(), probabilities.cpu()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")