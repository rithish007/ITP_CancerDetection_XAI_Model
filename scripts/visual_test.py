import matplotlib.pyplot as plt
import torch
import random
from torch.utils.data import Subset

from src.data.preprocessor import (
    MicroscopyDataset,
    DeterministicAugmentedDataset,
    compute_mean_std,
    get_transforms
)
from src.utils.paths import TRAIN_VAL_DIR


def denormalize(tensor, mean, std):
    """Reverse normalization so pixel values are viewable."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def visualize_augmentations(root_dir, num_images=3, image_size=224):
    # ── Load dataset ──────────────────────────────────────────────
    base_dataset = MicroscopyDataset(root_dir)

    # ✅ Random indices different every run (no fixed seed)
    all_indices = list(range(len(base_dataset)))
    selected_indices = random.sample(all_indices, num_images)
    print(f"Selected image indices: {selected_indices}")

    subset = Subset(base_dataset, selected_indices)

    # ── Compute stats from these images ───────────────────────────
    mean, std = compute_mean_std(subset, image_size=image_size)

    # ── Build transforms ──────────────────────────────────────────
    plain_tf, val_tf = get_transforms(image_size, mean, std)

    aug_dataset = DeterministicAugmentedDataset(
        subset,
        plain_tf=plain_tf,
        image_size=image_size,
        mean=mean,
        std=std,
        rotation_angle=90,
        blur_kernel=7,
        blur_sigma=1.0
    )

    # ── Plot ──────────────────────────────────────────────────────
    variants = DeterministicAugmentedDataset.VARIANTS
    num_variants = len(variants)

    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=num_variants,
        figsize=(4 * num_variants, 4 * num_images)
    )

    # Ensure axes is always 2D even if num_images=1
    if num_images == 1:
        axes = [axes]

    for img_idx in range(num_images):
        for var_idx, variant_name in enumerate(variants):

            flat_idx = img_idx * num_variants + var_idx
            image_tensor, label = aug_dataset[flat_idx]

            # ✅ Denormalize for display
            image_display = denormalize(image_tensor, mean, std)
            image_np = image_display.permute(1, 2, 0).numpy()

            ax = axes[img_idx][var_idx]
            ax.imshow(image_np)
            ax.axis("off")

            # ✅ Label burned into every subplot title
            label_name = "normal" if label == 0 else "cancer"
            ax.set_title(
                f"{variant_name}\n[{label_name}]",
                fontsize=11,
                fontweight="bold",
                pad=6,
                color="green" if label == 0 else "red"   # green=normal, red=cancer
            )

        # ✅ Row label showing which original file was used
        original_path = base_dataset.image_paths[selected_indices[img_idx]]
        axes[img_idx][0].set_ylabel(
            f"Image {selected_indices[img_idx]}\n{original_path.name}",
            fontsize=9,
            rotation=0,
            labelpad=100,
            va="center"
        )

    plt.suptitle("Original vs Augmented Variants", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    #plt.savefig("augmentation_preview.png", bbox_inches="tight", dpi=150)
    plt.show()
    #print("Saved to augmentation_preview.png")


if __name__ == "__main__":
    visualize_augmentations(
        root_dir=TRAIN_VAL_DIR,
        num_images=3,
        image_size=224
    )