from src.utils.paths import TRAIN_VAL_DIR
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.preprocessor import create_dataloaders

logger = get_logger("preprocessing")

def main():
    try:
        logger.info("Starting preprocessing stage")

        DATA_DIR = TRAIN_VAL_DIR

        train_loader, val_loader = create_dataloaders(
            DATA_DIR,
            batch_size="full",  # Use "full" the entire training dataset or any number for batched size
            mode="train",  # Use "train" for random order of images or "debug" for fixed order
            compute_stats=True
        )

        for images, labels in train_loader:
            logger.info(f"Batch shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Labels: {labels}")
            break

        logger.info("Preprocessing stage completed successfully")

    except Exception:
        logger.exception("Preprocessing error: {e}")
        raise



if __name__ == "__main__":
    main()