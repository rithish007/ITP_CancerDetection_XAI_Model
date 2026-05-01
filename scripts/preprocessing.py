from src.utils.paths import TRAIN_VAL_DIR
from src.utils.logger import get_logger
from src.data.preprocessor import create_dataloaders

logger = get_logger("preprocessing")

def main():
    try:
        logger.info("Starting preprocessing stage")

        train_loader, val_loader, mean, std = create_dataloaders(
            TRAIN_VAL_DIR,
            batch_size=32, # Use "full" the entire training dataset or any number for batched size
            mode="train", # Use "train" for random order of images or "debug" for fixed order
            compute_stats=True
        )

        for images, labels in train_loader:
            logger.info(f"Train batch shape: {images.shape}")
            logger.info(f"Train labels shape: {labels.shape}")
            logger.info(f"Train labels: {labels}")
            break

        for images, labels in val_loader:
            logger.info(f"Validation batch shape: {images.shape}")
            logger.info(f"Validation labels shape: {labels.shape}")
            logger.info(f"Validation labels: {labels}")
            break

        logger.info("Preprocessing stage completed successfully")

    except Exception as e:
        logger.exception(f"Preprocessing error: {e}")
        raise


if __name__ == "__main__":
    main()