import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.utils.paths import TEST_DIR
from src.utils.logger import get_logger
from src.data.preprocessor import create_test_dataloader
from src.training.classifier import ImageClassifier

logger = get_logger("test_model")


def calculate_metrics(y_true, y_pred, y_probs=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "specificity": specificity,
        "confusion_matrix": cm.tolist()
    }
    if y_probs is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
    return metrics


def evaluate_model(classifier, test_loader):
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in test_loader:
        result = classifier.eval_step(images, labels)
        all_preds.extend(result["predictions"].numpy())
        all_labels.extend(result["labels"].numpy())
        all_probs.extend(result["probabilities"].numpy())

    return calculate_metrics(all_labels, all_preds, all_probs)


def main(full_model=True):
    try:
        logger.info("Starting testing pipeline")

        # Testing Model
        models = ["resnet18", "densenet121"]
        model_name = models[0] # resnet18, densenet121
        head_type = "mlp" # mlp, basic
        freeze = False
        lr = 0.01
        wd = 1e-4

        if full_model:
            model_path = f"../../data/outputs/models/{model_name}_{head_type}_freeze{freeze}_lr{lr}_wd{wd}_FINAL.pt"
            config_path = f"../../data/outputs/models/{model_name}_{head_type}_freeze{freeze}_lr{lr}_wd{wd}_FINAL_config.json"
        else:
            model_path = f"../../data/outputs/models/{model_name}_{head_type}_freeze{freeze}_lr{lr}_wd{wd}.pt"
            config_path = f"../../data/outputs/models/{model_name}_{head_type}_freeze{freeze}_lr{lr}_wd{wd}_config.json"



        with open(config_path, "r") as f:
            config = json.load(f)

        test_loader = create_test_dataloader(
            TEST_DIR,
            batch_size=config["batch_size"],
            image_size=224,
            mean=config["mean"],
            std=config["std"]
        )

        classifier = ImageClassifier(
            model_name=config["model_name"],
            num_classes=config["num_classes"],
            pretrained=config["pretrained"],
            head_type=config["head_type"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            optimizer_name=config["optimizer_name"],
            weight_decay=config["weight_decay"],
            freeze_backbone=config["freeze_backbone"]
        )

        classifier.load_model(model_path)

        metrics = evaluate_model(classifier, test_loader)

        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1: {metrics['f1']:.4f}")
        logger.info(f"Confusion Matrix: {metrics['confusion_matrix']}")
        logger.info(f"Test Specificity: {metrics['specificity']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")

        logger.info("Testing completed successfully")

    except Exception as e:
        logger.exception(f"Testing pipeline error: {e}")
        raise


if __name__ == "__main__":
    main(False)