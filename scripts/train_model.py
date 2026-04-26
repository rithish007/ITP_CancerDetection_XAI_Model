from src.data.preprocessor_sofia import Preprocessor
from src.models.classifier import ImageClassifier
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.utils.paths import MODELS_DIR

logger = get_logger("training")


def main():
    try:
        model_name = "resnet18" # **** CHANGE MODEL NAME TO DESIRED MODEL (resnet18/densenet121)
        logger.info("Starting training stage")
        set_seed(42)

        logger.info("Loading training batch")
        preprocessor = Preprocessor(image_size=224)
        batch, labels = preprocessor.load_sample_batch()

        logger.info("Initializing classifier")
        classifier = ImageClassifier(model_name=model_name, num_classes=2, pretrained=True, learning_rate=1e-4,)

        num_epochs = 20
        result = None

        for epoch in range(num_epochs):
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
            result = classifier.train_step(batch, labels)
            logger.info(f"Loss: {result['loss']:.4f} --- Accuracy: {result['accuracy']:.4f}")

        logger.info(f"Finished training. Predictions: {result['predictions']}")

        model_path = MODELS_DIR / f"{model_name}_baseline.pth"
        # classifier.save_model(model_path) # **** Uncomment to save new model, leave as is to avoid overwriting saved model

        logger.info("Training completed successfully")

    except Exception as e:
        logger.exception(f"Training error: {e}")
        raise

if __name__ == "__main__":
    main()