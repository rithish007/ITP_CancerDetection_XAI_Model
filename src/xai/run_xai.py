
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

from src.data.preprocessor import get_transforms, ResizeWithPadding
from src.training.classifier import ImageClassifier
from src.xai.gradcam import generate_heatmap, overlay_heatmap
from src.utils.logger import get_logger
from src.utils.seed import set_seed

# Settings

MODEL_PATH = PROJECT_ROOT / "data" / "outputs" / "models" / "resnet18_mlp_freezeFalse_lr0.01_wd0.0001.pt"
TEST_DIR   = PROJECT_ROOT / "data" / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "xai_gradcam"

CLASS_DIRS = {
    "cancer": ("cancer_test", 1),
    "normal": ("normal_test", 0),
}
CLASS_NAMES = {0: "Normal (Type 1)", 1: "Cancer (Type 4)"}
EXTENSIONS  = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Helpers

def load_config(model_path):
    config_path = model_path.parent / (model_path.stem + "_config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def get_target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return model.features.denseblock4
    raise ValueError(f"Cannot auto-detect Grad-CAM layer for {type(model).__name__}.")


def resize_original(image_pil, image_size):
    """Resize with aspect-ratio padding — same spatial layout the model sees, no normalisation."""
    return np.array(ResizeWithPadding(image_size)(image_pil))


def masked_confidence(model, image_tensor, heatmap, pred_idx, device):
    mask   = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    mask   = mask.expand_as(image_tensor).to(device)
    masked = image_tensor * mask
    with torch.no_grad():
        logits = model(masked)
        probs  = torch.softmax(logits, dim=1)
    return probs[0, pred_idx].item()


def save_figure(original_np, heatmap, overlay, true_label, pred_label, confidence, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(original_np)
    axes[0].set_title(f"Original\nTrue: {true_label}", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nPred: {pred_label}  ({confidence:.1f}%)", fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_on_image(image_path, true_class_idx, model, val_transform, image_size,
                 device, target_layer, output_subdir, logger):
    image = Image.open(str(image_path)).convert("RGB")

    original_np  = resize_original(image, image_size)
    image_tensor = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.softmax(logits, dim=1)
    pred_idx   = logits.argmax(dim=1).item()
    orig_conf  = probs[0, pred_idx].item()
    pred_label = CLASS_NAMES.get(pred_idx, str(pred_idx))
    true_label = CLASS_NAMES.get(true_class_idx, str(true_class_idx))

    with torch.enable_grad():
        heatmap = generate_heatmap(model, image_tensor, target_layer, class_idx=pred_idx)

    gc_conf = masked_confidence(model, image_tensor, heatmap, pred_idx, device)
    gc_drop = max(0.0, (orig_conf - gc_conf) / orig_conf) * 100
    gc_ic   = max(0.0, (gc_conf  - orig_conf) / orig_conf) * 100

    overlay   = overlay_heatmap(heatmap, original_np, alpha=0.45)
    correct   = "OK" if pred_idx == true_class_idx else "WRONG"
    save_path = output_subdir / f"{image_path.stem}_pred-{pred_idx}_conf-{orig_conf*100:.0f}pct_{correct}.png"
    save_figure(original_np, heatmap, overlay, true_label, pred_label, orig_conf * 100, save_path)

    logger.info(
        f"  {image_path.name:<40}  true={true_label}  pred={pred_label}"
        f"  conf={orig_conf*100:.1f}%  drop={gc_drop:.1f}%  IC={gc_ic:.1f}%  [{correct}]"
    )

    return pred_idx == true_class_idx, {
        "image":          image_path.name,
        "true_class":     CLASS_NAMES[true_class_idx],
        "pred_class":     pred_label,
        "original_conf":  round(orig_conf * 100, 2),
        "gc_masked_conf": round(gc_conf    * 100, 2),
        "gc_drop_pct":    round(gc_drop,   2),
        "gc_ic_pct":      round(gc_ic,     2),
    }


# Main

def main():
    logger = get_logger("run_xai")
    set_seed(42)

    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    config = load_config(MODEL_PATH)
    logger.info(
        f"Config: model={config['model_name']}  head={config['head_type']}"
        f"  image_size={config['image_size']}"
    )

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

    _, val_transform = get_transforms(
        image_size=config["image_size"],
        mean=config["mean"],
        std=config["std"],
    )

    target_layer = get_target_layer(model)

    total, correct_total = 0, 0
    class_results = {}
    rows = []

    for subdir_name, (out_name, true_class_idx) in CLASS_DIRS.items():
        src_folder = TEST_DIR / subdir_name
        out_folder = OUTPUT_DIR / out_name
        out_folder.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in src_folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        logger.info(f"\n--- {subdir_name.upper()}  ({len(images)} images) -> {out_name}/ ---")

        class_correct = 0
        for i, img_path in enumerate(images, 1):
            logger.info(f"[{i}/{len(images)}]")
            try:
                ok, row = run_on_image(
                    img_path, true_class_idx, model, val_transform,
                    config["image_size"], device, target_layer, out_folder, logger,
                )
                class_correct += int(ok)
                rows.append(row)
            except Exception as e:
                logger.error(f"  Failed on {img_path.name}: {e}")

        acc = class_correct / len(images) * 100 if images else 0.0
        logger.info(f"  {subdir_name} accuracy: {class_correct}/{len(images)} ({acc:.1f}%)")
        class_results[subdir_name] = (class_correct, len(images), acc)
        total += len(images)
        correct_total += class_correct

    overall = correct_total / total * 100 if total else 0.0
    logger.info(f"\nOverall: {correct_total}/{total} correct ({overall:.1f}%)")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")

    drop_vals = [r["gc_drop_pct"] for r in rows]
    ic_vals   = [r["gc_ic_pct"]   for r in rows]

    # Save CSV
    if rows:
        csv_path = OUTPUT_DIR / "gradcam_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"CSV saved: {csv_path}")
        sep = "=" * 60
        logger.info(f"\n{sep}")
        logger.info(f"  {'Method':<12}  {'Drop Mean':>10}  {'Drop Std':>9}  {'IC Mean':>9}  {'IC Std':>8}")
        logger.info(f"  {'-'*56}")
        logger.info(
            f"  {'Grad-CAM':<12}  {float(np.mean(drop_vals)):>9.2f}%  {float(np.std(drop_vals)):>8.2f}%"
            f"  {float(np.mean(ic_vals)):>8.2f}%  {float(np.std(ic_vals)):>7.2f}%"
        )
        logger.info(sep)

    # Write accuracy + metrics summary to output folder
    summary_path = OUTPUT_DIR / "accuracy_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 52 + "\n")
        f.write(" Grad-CAM Test Run  --  Accuracy Summary\n")
        f.write(f" Model: {MODEL_PATH.name}\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"  {'Class':<20}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>9}\n")
        f.write(f"  {'-'*48}\n")
        for cls, (n_correct, n_total, acc) in class_results.items():
            label = CLASS_NAMES.get({"cancer": 1, "normal": 0}[cls], cls)
            f.write(f"  {label:<20}  {n_correct:>7}  {n_total:>5}  {acc:>8.1f}%\n")
        f.write(f"  {'-'*48}\n")
        f.write(f"  {'Overall':<20}  {correct_total:>7}  {total:>5}  {overall:>8.1f}%\n")
        f.write("\n" + "=" * 52 + "\n")
        if rows:
            f.write("\n Grad-CAM Metrics\n")
            f.write("=" * 52 + "\n")
            f.write(f"  {'Method':<12}  {'Drop Mean':>10}  {'Drop Std':>9}  {'IC Mean':>9}  {'IC Std':>8}\n")
            f.write(f"  {'-'*48}\n")
            f.write(
                f"  {'Grad-CAM':<12}  {float(np.mean(drop_vals)):>9.2f}%  {float(np.std(drop_vals)):>8.2f}%"
                f"  {float(np.mean(ic_vals)):>8.2f}%  {float(np.std(ic_vals)):>7.2f}%\n"
            )
            f.write("=" * 52 + "\n")
            f.write("  Drop%: lower = better localisation\n")
            f.write("  IC%:   lower = less spurious confidence boost\n")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()