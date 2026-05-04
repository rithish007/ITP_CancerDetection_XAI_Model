import os
import json

from src.utils.paths import TRAIN_VAL_DIR
from src.utils.logger import get_logger
from src.data.preprocessor import create_dataloaders, MicroscopyDataset, build_dataloaders_from_subsets
from src.training.classifier import ImageClassifier
from src.training.trainer import train_model, train_one_epoch
from src.utils.plots import plot_training_history

logger = get_logger("train_model")


def build_model_name(config):
    return (
        f"{config['model_name']}_"
        f"{config['head_type']}_"
        f"freeze{config['freeze_backbone']}_"
        f"lr{config['learning_rate']}_"
        f"wd{config['weight_decay']}"
    )


def create_classifier(config):
    return ImageClassifier(
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


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main(train_full=False):
    try:
        logger.info("Starting training pipeline")

        config = {
            "model_name": "resnet18", # resnet18, densenet121
            "num_classes": 2,
            "pretrained": True,
            "head_type": "mlp", # mlp, basic
            "dropout": 0.3,
            "learning_rate": 0.01,
            "optimizer_name": "adamw",
            "weight_decay": 1e-4,
            "freeze_backbone": False,
            "epochs": 20,
            "batch_size": 32,
            "image_size": 224
        }

        model_name = build_model_name(config)
        save_dir = "../data/outputs/models"
        os.makedirs(save_dir, exist_ok=True)

        if not train_full:
            logger.info("Running training experimentation with train/val split")

            train_loader, val_loader, mean, std = create_dataloaders(
                TRAIN_VAL_DIR,
                batch_size=config["batch_size"],
                image_size=config["image_size"],
                mode="train",
                compute_stats=True
            )

            classifier = create_classifier(config)

            config["mean"] = mean
            config["std"] = std

            save_path = os.path.join(save_dir, f"{model_name}.pt")
            config_path = os.path.join(save_dir, f"{model_name}_config.json")
            history_path = os.path.join(save_dir, f"{model_name}_history.json")
            plot_path = os.path.join(save_dir, f"{model_name}_plot.png")

            save_json(config, config_path)
            logger.info(f"Config saved to {config_path}")

            history = train_model(
                classifier,
                train_loader,
                val_loader,
                epochs=config["epochs"],
                save_path=save_path
            )

            save_json(history, history_path)
            logger.info(f"Training history saved to {history_path}")

            plot_training_history(history, save_path=plot_path)
            logger.info(f"Training plot saved to {plot_path}")

            logger.info("Training experimentation completed successfully")

        else:
            logger.info("Training final model with full train+val dataset")

            full_dataset = MicroscopyDataset(TRAIN_VAL_DIR)

            # Full training: stats are computed using the full train_val dataset
            full_train_loader, _, mean, std = build_dataloaders_from_subsets(
                train_subset=full_dataset,
                val_subset=full_dataset,  # dummy, not used
                batch_size=config["batch_size"],
                image_size=config["image_size"],
                mode="train",
                compute_stats=True
            )

            final_classifier = create_classifier(config)

            final_history = {
                "train_loss": [],
                "train_accuracy": [],
                "train_precision": [],
                "train_recall": [],
                "train_f1": []
            }

            for epoch in range(config["epochs"]):
                train_metrics = train_one_epoch(final_classifier, full_train_loader)

                final_history["train_loss"].append(train_metrics["loss"])
                final_history["train_accuracy"].append(train_metrics["accuracy"])
                final_history["train_precision"].append(train_metrics["precision"])
                final_history["train_recall"].append(train_metrics["recall"])
                final_history["train_f1"].append(train_metrics["f1"])

                logger.info(
                    f"[FINAL MODEL] Epoch {epoch + 1}/{config['epochs']} -- "
                    f"Loss: {train_metrics['loss']:.4f} -- "
                    f"Acc: {train_metrics['accuracy']:.4f} -- "
                    f"F1: {train_metrics['f1']:.4f}"
                )

            config["mean"] = mean
            config["std"] = std
            config["training_mode"] = "full_train_val"

            final_model_path = os.path.join(save_dir, f"{model_name}_FINAL.pt")
            final_config_path = os.path.join(save_dir, f"{model_name}_FINAL_config.json")
            final_history_path = os.path.join(save_dir, f"{model_name}_FINAL_history.json")

            final_classifier.save_model(final_model_path)
            save_json(config, final_config_path)
            save_json(final_history, final_history_path)

            logger.info(f"Final model saved to {final_model_path}")
            logger.info(f"Final config saved to {final_config_path}")
            logger.info(f"Final history saved to {final_history_path}")

        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.exception(f"Training pipeline error: {e}")
        raise


if __name__ == "__main__":
    main(train_full=False)