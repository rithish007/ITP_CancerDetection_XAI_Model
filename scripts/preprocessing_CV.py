# scripts/run_preprocessing_cv.py

from src.utils.paths import TRAIN_VAL_DIR
from src.utils.logger import get_logger
from src.data.preprocessor import create_cv_dataloaders

logger = get_logger("preprocessing_cv")


def main():
    try:
        logger.info("Starting CV preprocessing test")

        cv_loaders = create_cv_dataloaders(
            TRAIN_VAL_DIR,
            n_splits=5,
            batch_size=32,
            mode="train",
            compute_stats=True
        )

        for fold, train_loader, val_loader in cv_loaders:
            logger.info(f"Testing fold {fold}")

            """for images, labels in train_loader:
                logger.info(f"Fold {fold} train batch shape: {images.shape}")
                logger.info(f"Fold {fold} train labels shape: {labels.shape}")
                logger.info(f"Fold {fold} train labels: {labels}")
                break

            for images, labels in val_loader:
                logger.info(f"Fold {fold} validation batch shape: {images.shape}")
                logger.info(f"Fold {fold} validation labels shape: {labels.shape}")
                logger.info(f"Fold {fold} validation labels: {labels}")
                break"""

        logger.info("CV preprocessing test completed successfully")

    except Exception as e:
        logger.exception(f"CV preprocessing error: {e}")
        raise


if __name__ == "__main__":
    main()