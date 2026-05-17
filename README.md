# ITP_CancerDetection_XAI_Model

Deep learning pipeline for cancer vs normal cell classification using microscopy images, including explainable AI (XAI) analysis for model interpretability.

Developed as part of an Industry Training Programme (ITP) project using PyTorch and pretrained convolutional neural networks.

---

# Pipeline Overview

The project is divided into four main stages:

```text
Preprocessing
    ↓
Training / Fine-Tuning
    ↓
Evaluation / Testing
    ↓
XAI Visualization
```

---

# Project Structure

```text
project/
│
├── data/
│   ├── train_val/
│   │   ├── normal/
│   │   └── cancer/
│   │
│   ├── test/
│   │   ├── normal/
│   │   └── cancer/
│   │
│   └── outputs/
│       ├── models/
│       ├── logs/
│       └── xai/
│
├── src/
│   ├── data/
│   │   └── preprocessor.py
│   │
│   ├── training/
│   │   ├── classifier.py
│   │   ├── get_model.py
│   │   └── trainer.py
│   │
│   ├── evaluation/
│   │   └── test_model.py
│   │
│   ├── xai/
│   │   └── gradcam.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── paths.py
│       ├── plots.py
│       └── seed.py
│
├── scripts/
│   ├── preprocessing.py
│   ├── preprocessing_CV.py
│   ├── train_model.py
│   └── visual_test.py
│
├── requirements.txt
└── README.md
```

---

# Setup

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Structure

The dataset must follow this structure:

```text
train_val/
├── normal/
└── cancer/

test/
├── normal/
└── cancer/
```

Supported image formats:

```text
.tif
.tiff
```

---

# Preprocessing

The preprocessing pipeline handles:

* Dataset loading
* Train/validation splitting
* Image resizing with padding
* Dataset normalization
* Deterministic data augmentation
* Batch creation using PyTorch DataLoaders

Implemented augmentations include:

* Horizontal flip
* Rotation
* Gaussian blur
* Elastic deformation

Run preprocessing only scripts:

```bash
python scripts/preprocessing.py
python scripts/visual_test.py
```

This scripts can be used to verify:

* Dataset loading
* Batch shapes
* Labels
* Transformations
* Data pipeline behavior

---

# Training / Fine-Tuning

The training pipeline supports pretrained CNN models using ImageNet weights.

Currently supported models:

* ResNet18
* DenseNet121

Supported features:

* Pretrained backbone loading
* Backbone freezing / unfreezing
* Simple or MLP classification heads
* Adam / AdamW optimizers
* Weight decay (L2 regularization)
* Automatic best model checkpoint saving
* Training history logging
* Loss and accuracy plotting

---

## Training Modes

### 1. Validation Training Mode

Uses a train/validation split to evaluate configurations and automatically saves the best validation checkpoint.

Run:

```bash
python scripts/train_model.py
```

Outputs generated:

```text
model.pt
model_config.json
model_history.json
model_plot.png
```

---

### 2. Full Training Mode

Retrains the selected configuration using the full train-validation dataset.

This mode was intended for experimentation and comparison only.

---

# Evaluation / Testing

Testing is performed on a completely separate test dataset.

Implemented evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* ROC-AUC
* Confusion Matrix

Run evaluation:

```bash
python src/evaluation/test_model.py
```

The testing pipeline automatically:

* Loads the trained model
* Loads the saved training configuration
* Reuses the original training normalization statistics
* Evaluates the model on the test dataset

---

# Explainable AI (XAI)

The project includes explainability methods for visualizing model attention and decision regions.

Current XAI implementations:

* Grad-CAM

Generated visualizations are stored in:

```text
data/outputs/xai/
```

Run XAI visualization:

```bash
python scripts/run_xai.py?
```

---

# Outputs

Generated outputs are stored in:

```text
data/outputs/
├── models/   # trained models and configs
├── logs/     # module logs
└── xai/      # Grad-CAM visualizations
```

Typical generated files:

```text
.pt                 → trained model
_config.json        → training configuration
_history.json       → training metrics history
_plot.png           → training curves
```

---

# Configuration Notes

Most training configurations can be modified directly from the training script, including:

* Model selection
* Learning rate
* Batch size
* Freeze/unfreeze backbone
* Optimizer selection
* Classification head type
* Weight decay
* Number of epochs

