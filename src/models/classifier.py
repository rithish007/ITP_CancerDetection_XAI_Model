import torch
import torch.nn as nn
import torch.optim as optim
from src.models.get_model import get_model
from src.utils.logger import get_logger

logger = get_logger("classifier")


class ImageClassifier:
    def __init__(self, model_name="resnet18", num_classes=2, pretrained=True, learning_rate=1e-4, device=None):
        logger.info(f"Initializing model: {model_name}")
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = get_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            logger.info(f"Model initialized successfully on device={self.device}")

        except Exception as e:
            logger.exception(f"Failed to initialize ImageClassifier: {e}")
            raise

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def train_step(self, batch, labels):
        try:
            if batch is None or labels is None:
                raise ValueError("batch or labels is None")

            if len(batch) == 0:
                raise ValueError("batch is empty")

            self.model.train()

            batch = batch.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == labels).float().mean().item()

            return {"loss": loss.item(), "accuracy": accuracy, "predictions": predictions.detach().cpu().tolist(),}

        except Exception as e:
            logger.exception(f"Training error: {e}")
            raise

    def predict(self, batch):
        self.model.eval()
        batch = batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu(), outputs.cpu()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)