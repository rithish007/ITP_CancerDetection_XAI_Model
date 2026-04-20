import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.utils.logger import get_logger
from src.utils.paths import SAMPLE_IMAGES_DIR

logger = get_logger("preprocessing")


class Preprocessor:
    def __init__(self, image_size=224, class_to_idx=None):
        self.image_size = image_size
        self.class_to_idx = class_to_idx or {"normal": 0, "cancer": 1}
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(),])

    def get_image_paths(self, root_dir=SAMPLE_IMAGES_DIR):
        root_dir = Path(root_dir)

        if not root_dir.exists():
            logger.warning(f"Image directory does not exist: {root_dir}")
            return []

        image_paths = list(root_dir.glob("*/*.tif")) + list(root_dir.glob("*/*.tiff"))
        image_paths = sorted(image_paths)

        logger.info(f"Found {len(image_paths)} TIFF images in {root_dir}")
        return image_paths

    def load_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor

    def get_label_from_path(self, image_path):
        image_path = Path(image_path)
        class_name = image_path.parent.name

        if class_name not in self.class_to_idx:
            raise ValueError(
                f"Class '{class_name}' not found in class_to_idx mapping."
            )

        return self.class_to_idx[class_name]

    def load_image_with_label(self, image_path):
        image_path = Path(image_path)
        img_tensor = self.load_image(image_path)
        label = self.get_label_from_path(image_path)
        return img_tensor, label

    def load_batch_with_labels(self, image_paths):
        logger.info(f"Creating batch with {len(image_paths)} images")

        tensors = []
        labels = []

        for path in image_paths:
            img_tensor, label = self.load_image_with_label(path)
            tensors.append(img_tensor)
            labels.append(label)

        batch = torch.stack(tensors)
        labels = torch.tensor(labels, dtype=torch.long)

        logger.info("Batch preprocessing completed")
        return batch, labels

    def load_sample_batch(self, root_dir=SAMPLE_IMAGES_DIR):
        try:
            image_paths = self.get_image_paths(root_dir=root_dir)

            if not image_paths:
                logger.warning("No TIFF images found.")
                return None, None

            selected_paths = image_paths[:10] # **** REMOVE [:#] IT IS ONLY FOR QUICK TESTING
            logger.info(f"Loading sample batch of {len(selected_paths)} images")

            return self.load_batch_with_labels(selected_paths)
        except Exception as e:
            logger.exception(f"Preprocessor error: {e}")
            raise