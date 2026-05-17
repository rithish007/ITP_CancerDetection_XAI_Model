
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
from src.xai.gradcam_plus_plus import generate_heatmap as generate_heatmap_pp
from src.xai.eigencam import generate_heatmap as generate_heatmap_eigen
from src.xai.scorecam import generate_heatmap as generate_heatmap_sc
from src.utils.logger import get_logger
from src.utils.seed import set_seed

# Settings

MODEL_PATH  = PROJECT_ROOT / "data" / "outputs" / "models" / "resnet18_mlp_freezeFalse_lr0.01_wd0.0001.pt"
IMAGE_DIR   = PROJECT_ROOT / "data" / "test_images"
OUTPUT_DIR  = PROJECT_ROOT / "data" / "outputs" / "xai_test"

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
        return model.layer4
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return model.features.denseblock4
    raise ValueError(f"Cannot auto-detect Grad-CAM layer for {type(model).__name__}.")


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


def masked_confidence(model, image_tensor, heatmap, pred_idx, device):
    mask   = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    mask   = mask.expand_as(image_tensor).to(device)
    masked = image_tensor * mask
    with torch.no_grad():
        logits = model(masked)
        probs  = torch.softmax(logits, dim=1)
    return probs[0, pred_idx].item()


def resize_original(image_pil, image_size):
    """Resize with aspect-ratio padding — same spatial layout the model sees, no normalisation."""
    return np.array(ResizeWithPadding(image_size)(image_pil))



def save_comparison_plot(original_rgb, overlays,
                         image_stem, img_name, img_index, total_images,
                         true_label, pred_label, output_dir):
    method_order = ["Grad-CAM", "Grad-CAM++", "EigenCAM", "Score-CAM"]
    fig, axes = plt.subplots(1, 5, figsize=(32, 7))

    fig.suptitle(
        f"Image {img_index}/{total_images}  |  {img_name}  |  "
        f"True: {true_label}  |  Predicted: {pred_label}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=13, fontweight="bold", pad=10)
    axes[0].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])

    for ax, method in zip(axes[1:], method_order):
        ax.imshow(overlays[method])
        ax.set_title(method, fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.85,
                     label="Saliency intensity")

    plt.tight_layout()
    save_path = output_dir / f"compare_{image_stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# Main

def main():
    logger = get_logger("run_metrics")
    set_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in CLASS_SUBDIR.values():
        (OUTPUT_DIR / sub).mkdir(exist_ok=True)

    config = load_config(MODEL_PATH)
    logger.info(f"Config loaded: model={config['model_name']}, head={config['head_type']}")

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

    samples = collect_all_images(IMAGE_DIR)
    logger.info(f"Test images: {len(samples)} total ({IMAGE_DIR})")

    target_layer = get_target_layer(model)
    rows = []

    for i, (img_path, true_class) in enumerate(samples, 1):
        logger.info(f"[{i}/{len(samples)}] {img_path.name}")
        try:
            image        = Image.open(str(img_path)).convert("RGB")
            image_tensor = val_transform(image).unsqueeze(0).to(device)

            # Original confidence
            with torch.no_grad():
                logits = model(image_tensor)
                probs  = torch.softmax(logits, dim=1)
            pred_idx  = logits.argmax(dim=1).item()
            orig_conf = probs[0, pred_idx].item()
            pred_label = CLASS_NAMES.get(pred_idx, str(pred_idx))

            # Grad-CAM (gradients required)
            with torch.enable_grad():
                heatmap_gc = generate_heatmap(
                    model, image_tensor, target_layer, class_idx=pred_idx
                )
            gc_conf = masked_confidence(model, image_tensor, heatmap_gc, pred_idx, device)
            gc_drop = max(0.0, (orig_conf - gc_conf) / orig_conf) * 100
            gc_ic   = max(0.0, (gc_conf - orig_conf) / orig_conf) * 100

            # Grad-CAM++ (gradients required)
            with torch.enable_grad():
                heatmap_pp = generate_heatmap_pp(
                    model, image_tensor, target_layer, class_idx=pred_idx
                )
            pp_conf = masked_confidence(model, image_tensor, heatmap_pp, pred_idx, device)
            pp_drop = max(0.0, (orig_conf - pp_conf) / orig_conf) * 100
            pp_ic   = max(0.0, (pp_conf - orig_conf) / orig_conf) * 100

            # EigenCAM (no gradients needed)
            heatmap_eigen = generate_heatmap_eigen(
                model, image_tensor, target_layer
            )
            eigen_conf = masked_confidence(model, image_tensor, heatmap_eigen, pred_idx, device)
            eigen_drop = max(0.0, (orig_conf - eigen_conf) / orig_conf) * 100
            eigen_ic   = max(0.0, (eigen_conf - orig_conf) / orig_conf) * 100

            # Score-CAM (gradient-free, perturbation-based)
            heatmap_sc = generate_heatmap_sc(
                model, image_tensor, target_layer, class_idx=pred_idx
            )
            sc_conf = masked_confidence(model, image_tensor, heatmap_sc, pred_idx, device)
            sc_drop = max(0.0, (orig_conf - sc_conf) / orig_conf) * 100
            sc_ic   = max(0.0, (sc_conf - orig_conf) / orig_conf) * 100

            # Comparison plot — original pixel values, no normalisation distortion
            original_rgb = resize_original(image, config["image_size"])
            heatmaps = {
                "Grad-CAM":   heatmap_gc,
                "Grad-CAM++": heatmap_pp,
                "EigenCAM":   heatmap_eigen,
                "Score-CAM":  heatmap_sc,
            }
            overlays = {
                name: overlay_heatmap(hm, original_rgb, alpha=0.45)
                for name, hm in heatmaps.items()
            }
            out_subdir = OUTPUT_DIR / CLASS_SUBDIR[true_class]
            save_comparison_plot(
                original_rgb, overlays,
                img_path.stem, img_path.name, i, len(samples),
                CLASS_NAMES[true_class], pred_label, out_subdir,
            )

            rows.append({
                "image":             img_path.name,
                "true_class":        CLASS_NAMES[true_class],
                "pred_class":        pred_label,
                "original_conf":     round(orig_conf * 100, 2),
                "gc_masked_conf":    round(gc_conf    * 100, 2),
                "gc_drop_pct":       round(gc_drop,   2),
                "gc_ic_pct":         round(gc_ic,     2),
                "gcpp_masked_conf":  round(pp_conf    * 100, 2),
                "gcpp_drop_pct":     round(pp_drop,   2),
                "gcpp_ic_pct":       round(pp_ic,     2),
                "eigen_masked_conf": round(eigen_conf * 100, 2),
                "eigen_drop_pct":    round(eigen_drop, 2),
                "eigen_ic_pct":      round(eigen_ic,  2),
                "sc_masked_conf":    round(sc_conf    * 100, 2),
                "sc_drop_pct":       round(sc_drop,   2),
                "sc_ic_pct":         round(sc_ic,     2),
            })

            logger.info(
                f"  orig={orig_conf*100:.1f}%  |  "
                f"GC drop={gc_drop:.1f}% IC={gc_ic:.1f}%  |  "
                f"GC++ drop={pp_drop:.1f}% IC={pp_ic:.1f}%  |  "
                f"Eigen drop={eigen_drop:.1f}% IC={eigen_ic:.1f}%  |  "
                f"Score-CAM drop={sc_drop:.1f}% IC={sc_ic:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed on {img_path.name}: {e}")

    if not rows:
        logger.error("No results — check IMAGE_DIR and model path.")
        return

    # Save CSV
    csv_path = OUTPUT_DIR / "metrics_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved: {csv_path}")

    # Summary table
    method_keys = {
        "Grad-CAM":   ("gc_drop_pct",    "gc_ic_pct"),
        "Grad-CAM++": ("gcpp_drop_pct",  "gcpp_ic_pct"),
        "EigenCAM":   ("eigen_drop_pct", "eigen_ic_pct"),
        "Score-CAM":  ("sc_drop_pct",    "sc_ic_pct"),
    }

    sep = "=" * 76
    logger.info(f"\n{sep}")
    logger.info(f"  {'Method':<14}  {'Drop Mean':>10}  {'Drop Std':>9}  {'IC Mean':>9}  {'IC Std':>8}")
    logger.info(f"  {'-'*72}")
    for method, (dk, ik) in method_keys.items():
        drop_vals = [r[dk] for r in rows]
        ic_vals   = [r[ik] for r in rows]
        logger.info(
            f"  {method:<14}  {float(np.mean(drop_vals)):>9.2f}%  {float(np.std(drop_vals)):>8.2f}%"
            f"  {float(np.mean(ic_vals)):>8.2f}%  {float(np.std(ic_vals)):>7.2f}%"
        )
    logger.info(sep)
    logger.info("  Drop%: lower = better localisation (CAM captures what the model uses)")
    logger.info("  IC%:   lower = less spurious confidence boost")
    logger.info(sep)
    logger.info(f"Outputs: {OUTPUT_DIR}")

    # Save metrics summary to output folder
    summary_path = OUTPUT_DIR / "metrics_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 76 + "\n")
        f.write(" XAI Metrics Summary  --  All Test Images\n")
        f.write(f" Model: {MODEL_PATH.name}\n")
        f.write(f" Images: {len(rows)} ({IMAGE_DIR})\n")
        f.write("=" * 76 + "\n\n")
        f.write(f"  {'Method':<14}  {'Drop Mean':>10}  {'Drop Std':>9}  {'IC Mean':>9}  {'IC Std':>8}\n")
        f.write(f"  {'-'*72}\n")
        for method, (dk, ik) in method_keys.items():
            drop_vals = [r[dk] for r in rows]
            ic_vals   = [r[ik] for r in rows]
            f.write(
                f"  {method:<14}  {float(np.mean(drop_vals)):>9.2f}%  {float(np.std(drop_vals)):>8.2f}%"
                f"  {float(np.mean(ic_vals)):>8.2f}%  {float(np.std(ic_vals)):>7.2f}%\n"
            )
        f.write("=" * 76 + "\n\n")
        f.write("  Drop%: lower = better localisation\n")
        f.write("  IC%:   lower = less spurious confidence boost\n")
    logger.info(f"Metrics summary saved: {summary_path}")


if __name__ == "__main__":
    main()
