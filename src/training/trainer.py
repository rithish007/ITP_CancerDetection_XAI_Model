import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.logger import get_logger

logger = get_logger("trainer")


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_one_epoch(classifier, train_loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in train_loader:
        result = classifier.train_step(images, labels)

        batch_size = result["total"]

        total_loss += result["loss"] * batch_size
        total_correct += result["correct"]
        total_samples += batch_size

        all_preds.extend(result["predictions"].numpy())
        all_labels.extend(result["labels"].numpy())

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = epoch_loss
    metrics["accuracy"] = epoch_accuracy

    return metrics


def validate(classifier, val_loader):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        result = classifier.eval_step(images, labels)

        batch_size = result["total"]

        total_loss += result["loss"] * batch_size
        total_correct += result["correct"]
        total_samples += batch_size

        all_preds.extend(result["predictions"].numpy())
        all_labels.extend(result["labels"].numpy())

    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    metrics = calculate_metrics(all_labels, all_preds)
    metrics["loss"] = epoch_loss
    metrics["accuracy"] = epoch_accuracy

    return metrics


def train_model(classifier, train_loader, val_loader, epochs=10, save_path=None):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    } # Used to store the training metrics and copare for best model

    best_val_f1 = 0.0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        train_metrics = train_one_epoch(classifier, train_loader)
        val_metrics = validate(classifier, val_loader)

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            f"Train Loss: {train_metrics['loss']:.4f} -- "
            f"Train Acc: {train_metrics['accuracy']:.4f} -- "
            f"Val Loss: {val_metrics['loss']:.4f} -- "
            f"Val Acc: {val_metrics['accuracy']:.4f} -- "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if save_path is not None and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            classifier.save_model(save_path)
            logger.info(f"Best model updated with Val F1: {best_val_f1:.4f}")

    return history