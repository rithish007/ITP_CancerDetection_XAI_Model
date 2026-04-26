from src.data.preprocessor_sofia import Preprocessor
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("preprocessing")


def main():
    try:
        logger.info("Starting preprocessing stage")

        set_seed(42)

        preprocessor = Preprocessor(image_size=224)
        batch, labels = preprocessor.load_sample_batch()

        logger.info(f"Batch shape: {batch.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Labels: {labels.tolist()}")

        logger.info("Preprocessing stage completed successfully")

    except Exception:
        logger.exception("Preprocessing error: {e}")
        raise


if __name__ == "__main__":
    main()