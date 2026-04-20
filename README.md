# ITP_CancerDetection_XAI_Model

This project implements a deep learning pipeline for classifying cancer vs normal cells using microscopy images, along with explainability methods (XAI) to understand model decisions.

Developed as part of an Industry Training Programme (ITP), focusing on building a modular and extensible ML pipeline using PyTorch.

---

## Setup

Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

## How to run

### 1. Preprocessing (Test data pipeline)

```
python scripts/run_preprocessing.py
```

This will:
- Load sample images
- Apply transformations
- Output batch shape and labels

### 2. Training (Baseline model)
```
python scripts/run_training.py
```
This will:
- Load sample batch
- Train a model (ResNet18 by default)
- Save the trained model

## Outputs

Generated outputs are stored in:
```
data/outputs/
├── models/   # trained models (.pth)
├── logs/     # logs per module (training, preprocessing, etc.)
└── xai/      # explanation outputs (gradcam)
```

## Notes
### Code Notes

- `****` marks configurable sections in the code (e.g., model selection, dataset size).
- Trained model saving code is commented to avoid overwriting existing files.
