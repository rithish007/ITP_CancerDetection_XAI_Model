from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
TRAIN_VAL_DIR = SAMPLE_IMAGES_DIR / "train_and_validate"
OUTPUTS_DIR = DATA_DIR / "outputs"

MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
XAI_DIR = OUTPUTS_DIR / "xai"