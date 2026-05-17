
import sys
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data.preprocessor import get_transforms, ResizeWithPadding
from src.training.classifier import ImageClassifier
from src.utils.logger import get_logger
from src.utils.seed import set_seed

# Settings

MODEL_PATH  = PROJECT_ROOT / "data" / "outputs" / "models" / "resnet18_mlp_freezeFalse_lr0.01_wd0.0001.pt"
IMAGE_DIR   = PROJECT_ROOT / "data" / "test_images"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "outputs" / "xai_gradcam_pkg"

CLASS_NAMES  = {0: "Normal (Type 1)", 1: "Cancer (Type 4)"}
CLASS_DIRS   = {"normal": 0, "cancer": 1}
CLASS_SUBDIR = {0: "normal", 1: "cancer"}

# Helpers

def load_config(model_path):
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def get_target_layer(model):
    if hasattr(model, "layer4"):
        return [model.layer4[-1]]
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return [model.features.denseblock4]
    raise ValueError(f"Cannot auto-detect target layer for {type(model).__name__}.")


def collect_all_images(image_dir):
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    samples = []
    for subdir, class_idx in CLASS_DIRS.items():
        folder = image_dir / subdir
        if not folder.exists():
            continue
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)
        for p in imgs:
            samples.append((p, class_idx))
    return samples


def resize_original(image_pil, image_size):
    return np.array(ResizeWithPadding(image_size)(image_pil))


def build_content_mask(image_pil, image_size):
    img_np = resize_original(image_pil, image_size)
    return (img_np.max(axis=2) > 10).astype(np.float32)


def apply_content_mask(heatmap, content_mask):
    masked = heatmap * content_mask
    if masked.max() > 0:
        masked = masked / masked.max()
    return masked.astype(np.float32)


def masked_confidence(model, image_tensor, heatmap, pred_idx, device):
    mask   = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    mask   = mask.expand_as(image_tensor).to(device)
    masked = image_tensor * mask
    with torch.no_grad():
        logits = model(masked)
        probs  = torch.softmax(logits, dim=1)
    return probs[0, pred_idx].item()


def save_comparison_plot(original_rgb, overlays, image_stem, img_name, img_index, total_images,
                         true_label, pred_label, output_dir):
    method_order = ["Grad-CAM", "Grad-CAM++", "EigenCAM", "HiResCAM"]
    fig, axes = plt.subplots(1, 5, figsize=(28, 5))

    fig.suptitle(
        f"Image {img_index}/{total_images}  |  {img_name}  |  "
        f"True: {true_label}  |  Predicted: {pred_label}",
        fontsize=11, fontweight="bold", y=1.02,
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for ax, method in zip(axes[1:], method_order):
        ax.imshow(overlays[method])
        ax.set_title(method, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    save_path = output_dir / f"compare_{image_stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# Main

def main():
    logger = get_logger("run_gradcam_pkg")
    set_seed(42)

    config = load_config(MODEL_PATH)
    logger.info(f"Config: model={config['model_name']}, head={config['head_type']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    else:
        logger.info("GPU: not available — running on CPU")

    classifier = ImageClassifier(
        model_name=config["model_name"],
        num_classes=config["num_classes"],
        pretrained=False,
        head_type=config["head_type"],
        dropout=config["dropout"],
        freeze_backbone=False,
    )
    classifier.load_model(str(MODEL_PATH))
    model = classifier.model
    model.to(device)
    model.eval()
    logger.info("Model loaded.")
    logger.info("Target layer: layer4[-1] (last conv block — standard Grad-CAM target)")

    _, val_transform = get_transforms(
        image_size=config["image_size"],
        mean=config["mean"],
        std=config["std"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in CLASS_SUBDIR.values():
        (OUTPUT_DIR / sub).mkdir(exist_ok=True)

    samples = collect_all_images(IMAGE_DIR)
    logger.info(f"Test images: {len(samples)} total ({IMAGE_DIR})")

    target_layers = get_target_layer(model)

    cam_methods = {
        "Grad-CAM":   GradCAM(model=model,         target_layers=target_layers),
        "Grad-CAM++": GradCAMPlusPlus(model=model, target_layers=target_layers),
        "EigenCAM":   EigenCAM(model=model,         target_layers=target_layers),
        "HiResCAM":   HiResCAM(model=model,         target_layers=target_layers),
    }

    rows = []

    for i, (img_path, true_class) in enumerate(samples, 1):
        logger.info(f"[{i}/{len(samples)}] {img_path.name}")
        try:
            image        = Image.open(str(img_path)).convert("RGB")
            image_tensor = val_transform(image).unsqueeze(0).to(device)

            content_mask = build_content_mask(image, config["image_size"])
            original_rgb = resize_original(image, config["image_size"])
            float_rgb    = original_rgb.astype(np.float32) / 255.0

            with torch.no_grad():
                logits = model(image_tensor)
                probs  = torch.softmax(logits, dim=1)
            pred_idx   = logits.argmax(dim=1).item()
            orig_conf  = probs[0, pred_idx].item()
            pred_label = CLASS_NAMES.get(pred_idx, str(pred_idx))

            targets  = [ClassifierOutputTarget(pred_idx)]
            overlays = {}
            confs    = {}
            drops    = {}
            ics      = {}

            for method_name, cam_obj in cam_methods.items():
                use_eigen_smooth = method_name == "EigenCAM"

                grayscale_cam = cam_obj(
                    input_tensor=image_tensor,
                    targets=targets,
                    aug_smooth=True,
                    eigen_smooth=use_eigen_smooth,
                )
                heatmap = grayscale_cam[0]
                heatmap = apply_content_mask(heatmap, content_mask)

                overlay = show_cam_on_image(float_rgb, heatmap, use_rgb=True)

                conf = masked_confidence(model, image_tensor, heatmap, pred_idx, device)
                drop = max(0.0, (orig_conf - conf) / orig_conf) * 100
                ic   = max(0.0, (conf - orig_conf) / orig_conf) * 100

                overlays[method_name] = overlay
                confs[method_name]    = conf
                drops[method_name]    = drop
                ics[method_name]      = ic

            out_subdir = OUTPUT_DIR / CLASS_SUBDIR[true_class]
            save_comparison_plot(
                original_rgb, overlays, img_path.stem, img_path.name,
                i, len(samples),
                CLASS_NAMES[true_class], pred_label, out_subdir,
            )

            row = {
                "image":         img_path.name,
                "true_class":    CLASS_NAMES[true_class],
                "pred_class":    pred_label,
                "original_conf": round(orig_conf * 100, 2),
            }
            for method_name in cam_methods:
                key = method_name.lower().replace("-", "").replace("+", "p").replace(" ", "_")
                row[f"{key}_masked_conf"] = round(confs[method_name] * 100, 2)
                row[f"{key}_drop_pct"]    = round(drops[method_name], 2)
                row[f"{key}_ic_pct"]      = round(ics[method_name],   2)
            rows.append(row)

            logger.info(
                f"  orig={orig_conf*100:.1f}%  |  "
                + "  |  ".join(f"{m} drop={drops[m]:.1f}% IC={ics[m]:.1f}%" for m in cam_methods)
            )

        except Exception as e:
            logger.error(f"Failed on {img_path.name}: {e}", exc_info=True)

    # Release CAM hooks
    for cam_obj in cam_methods.values():
        cam_obj.__exit__(None, None, None)

    if not rows:
        logger.error("No results — check IMAGE_DIR and model path.")
        return

    # CSV
    csv_path = OUTPUT_DIR / "metrics_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved: {csv_path}")

    # Summary
    method_keys = {
        "Grad-CAM":   "gradcam",
        "Grad-CAM++": "gradcampp",
        "EigenCAM":   "eigencam",
        "HiResCAM":   "hirescam",
    }

    sep = "=" * 76
    logger.info(f"\n{sep}")
    logger.info(f"  {'Method':<14}  {'Drop Mean':>10}  {'Drop Std':>9}  {'IC Mean':>9}  {'IC Std':>8}")
    logger.info(f"  {'-'*72}")
    for display_name, key in method_keys.items():
        drop_vals = [r[f"{key}_drop_pct"] for r in rows]
        ic_vals   = [r[f"{key}_ic_pct"]   for r in rows]
        logger.info(
            f"  {display_name:<14}  {float(np.mean(drop_vals)):>9.2f}%  {float(np.std(drop_vals)):>8.2f}%"
            f"  {float(np.mean(ic_vals)):>8.2f}%  {float(np.std(ic_vals)):>7.2f}%"
        )
    logger.info(sep)
    logger.info("  Drop%: lower = better localisation")
    logger.info("  IC%:   lower = less spurious confidence boost")
    logger.info(sep)
    logger.info(f"Outputs: {OUTPUT_DIR}")

    # Save metrics summary text
    summary_path = OUTPUT_DIR / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 76 + "\n")
        f.write(" XAI Metrics Summary  --  All Test Images (pytorch-grad-cam)\n")
        f.write(f" Model: {MODEL_PATH.name}\n")
        f.write(f" Images: {len(rows)} ({IMAGE_DIR})\n")
        f.write("=" * 76 + "\n\n")
        f.write(f"  {'Method':<14}  {'Drop Mean':>10}  {'Drop Std':>9}  {'IC Mean':>9}  {'IC Std':>8}\n")
        f.write(f"  {'-'*72}\n")
        for display_name, key in method_keys.items():
            drop_vals = [r[f"{key}_drop_pct"] for r in rows]
            ic_vals   = [r[f"{key}_ic_pct"]   for r in rows]
            f.write(
                f"  {display_name:<14}  {float(np.mean(drop_vals)):>9.2f}%  {float(np.std(drop_vals)):>8.2f}%"
                f"  {float(np.mean(ic_vals)):>8.2f}%  {float(np.std(ic_vals)):>7.2f}%\n"
            )
        f.write("=" * 76 + "\n\n")
        f.write("  Drop%: lower = better localisation\n")
        f.write("  IC%:   lower = less spurious confidence boost\n")
    logger.info(f"Metrics summary saved: {summary_path}")


if __name__ == "__main__":
    main()
