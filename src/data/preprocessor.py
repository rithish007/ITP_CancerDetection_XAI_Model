import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

from src.utils.seed import set_seed
from src.utils.paths import TRAIN_VAL_DIR
from src.utils.logger import get_logger

logger = get_logger("data_pipeline")


# 1. DATASET
class MicroscopyDataset(Dataset):
    def __init__(self, root_dir, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx or {"normal": 0, "cancer": 1}

        self.image_paths = sorted(
            list(self.root_dir.glob("*/*.tif")) +
            list(self.root_dir.glob("*/*.tiff"))
        )

        logger.info(f"Found {len(self.image_paths)} images in {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        try:
            image = Image.open(path).convert("RGB")
            label = self.class_to_idx[path.parent.name]
            return image, label

        except Exception as e:
            logger.warning(f"Skipping corrupted image: {path} | {e}")
            new_idx = (idx + 1) % len(self.image_paths)
            return self.__getitem__(new_idx)


# 2. RESIZE WITH PADDING
class ResizeWithPadding:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size

        scale = self.size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = F.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )

        return F.pad(img, padding, fill=0)


# 3. TRANSFORM WRAPPER
class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)
        return image, label


# 3b. DETERMINISTIC AUGMENTED DATASET (n * 4 images)

# Each image produces exactly 4 fixed variants:
#   0 → original
#   1 → horizontal flip
#   2 → rotation (fixed angle) - change rotation angle here
#   3 → gaussian blur (simulates denoising) - change blur kernel and sigma values here
class DeterministicAugmentedDataset(Dataset):
    VARIANTS = ["original", "hflip", "rotation", "gaussian_blur"]

    def __init__(self, dataset, plain_tf, image_size, mean, std,
                 rotation_angle=90, blur_kernel=7, blur_sigma=1.0):
        self.dataset = dataset
        self.plain_tf = plain_tf
        self.image_size = image_size
        self.rotation_angle = rotation_angle
        self.blur_kernel = blur_kernel      # must be odd integer
        self.blur_sigma = blur_sigma

        resize_pad = ResizeWithPadding(image_size)
        to_tensor_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Each augmentation is fully deterministic — no random ops
        self.aug_transforms = {
            "original": self.plain_tf,

            "hflip": transforms.Compose([
                resize_pad,
                transforms.Lambda(lambda img: F.hflip(img)),   # fixed, always flipped
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),

            "rotation": transforms.Compose([
                resize_pad,
                transforms.Lambda(lambda img: F.rotate(img, rotation_angle)),  # fixed angle
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),

            "gaussian_blur": transforms.Compose([
                resize_pad,
                transforms.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma),  # fixed sigma
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        }

        logger.info(
            f"DeterministicAugmentedDataset: {len(dataset)} base images × "
            f"{len(self.VARIANTS)} variants = {len(dataset) * len(self.VARIANTS)} total"
        )

    def __len__(self):
        return len(self.dataset) * len(self.VARIANTS)

    def __getitem__(self, idx):
        variant_idx = idx % len(self.VARIANTS)         # 0, 1, 2, or 3
        real_idx = idx // len(self.VARIANTS)           # which base image

        variant_name = self.VARIANTS[variant_idx]

        image, label = self.dataset[real_idx]
        image = self.aug_transforms[variant_name](image)

        return image, label


# 4. COMPUTE MEAN / STD
def compute_mean_std(dataset, image_size=224, batch_size=32, num_batches=None):
    logger.info("Computing dataset mean and std...")

    temp_transform = transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.ToTensor()
    ])

    temp_dataset = TransformDataset(dataset, temp_transform)

    loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for i, (images, _) in enumerate(loader):
        if num_batches and i >= num_batches:
            break

        b = images.size(0)
        images = images.view(b, 3, -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

        total_images += b

    mean /= total_images
    std /= total_images

    logger.info(f"Mean: {mean.tolist()}")
    logger.info(f"Std: {std.tolist()}")

    return mean.tolist(), std.tolist()


# 5. TRANSFORMS
def get_transforms(image_size=224, mean=None, std=None):
    resize_pad = ResizeWithPadding(image_size)

    mean = mean if mean else [0.5]*3
    std = std if std else [0.5]*3

    plain_tf = transforms.Compose([        # original, no augmentation
        resize_pad,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        resize_pad,
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return plain_tf, val_tf


# 6. DATA LOADER PIPELINE
def create_dataloaders(
    root_dir,
    batch_size=32,
    image_size=224,
    val_split=0.2,
    seed=42,
    num_workers=0,
    mode="train",
    compute_stats=True
):
    logger.info(f"Initializing dataset pipeline (mode={mode})")

    if mode == "debug":
        logger.info("DEBUG MODE: deterministic")
        set_seed(seed)

    base_dataset = MicroscopyDataset(root_dir)

    if len(base_dataset) == 0:
        logger.warning("Empty dataset")
        return None, None

    # Split FIRST before computing stats
    generator = torch.Generator().manual_seed(seed)

    train_size = int((1 - val_split) * len(base_dataset))
    val_size = len(base_dataset) - train_size

    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=generator
    )

    logger.info(f"Split: {train_size} train / {val_size} val")

    # Compute stats from training data ONLY
    if compute_stats:
        mean, std = compute_mean_std(train_subset, image_size=image_size)
    else:
        mean, std = [0.5]*3, [0.5]*3

    # Transforms
    plain_tf, val_tf = get_transforms(image_size, mean, std)

    if mode == "debug":
        logger.info("Debug mode: using original images only, no augmentation")
        train_dataset = TransformDataset(train_subset, plain_tf)
    else:
        train_dataset = DeterministicAugmentedDataset(
            train_subset,
            plain_tf=plain_tf,
            image_size=image_size,
            mean=mean,
            std=std,
            rotation_angle=15,    # fixed rotation angle, always the same
            blur_kernel=5,        # must be odd
            blur_sigma=1.0        # fixed sigma, always the same
        )

    val_dataset = TransformDataset(val_subset, val_tf)

    # Full batch support
    if batch_size == "full":
        logger.info("Using full dataset as one batch")
        batch_size = len(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers if mode == "train" else 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info("DataLoaders ready")

    return train_loader, val_loader


"""
# TEST
if __name__ == "__main__":
    DATA_DIR = TRAIN_VAL_DIR

    train_loader, val_loader = create_dataloaders(
        DATA_DIR,
        batch_size="full", # Use "full" the entire training dataset or any number for batched size
        mode="train", # Use "train" for random order of images or "debug" for fixed order
        compute_stats=True
    )

    for images, labels in train_loader:
        logger.info(f"Batch shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Labels: {labels}")
        break

"""