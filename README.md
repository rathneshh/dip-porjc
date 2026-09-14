# COVID-19 vs. Viral Pneumonia X-Ray Classification: Data Pipeline

**Role:** Data Engineer (Member 1)

This repository contains the foundational data ingestion, preprocessing, and validation architecture for differentiating COVID-19 and Viral Pneumonia from Chest X-rays.

## Data Engineering Metrics

| Metric | Result |
| :--- | :--- |
| **Dataset Volume** | 15,153 validated X-ray images (70% Train, 15% Val, 15% Test) |
| **Data Quality Score** | 100% (0 corrupted files detected during scan) |
| **Class Imbalance** | Raw 7.5:1 ratio (Normal:Pneumonia) balanced to 1:1:1 per batch via `WeightedRandomSampler` |
| **Transformation Speed** | ~0.076 seconds per batch (Size 32) via Albumentations & OpenCV |
| **Augmentations** | CLAHE (Clip 2.0), Resize (224x224), Horizontal Flips (p=0.5), Rotations (±10°) |

## Repository Structure

```text
dip-porjc/
├── checkpoints/
│   └── best_resnet50.pth
├── results/
│   └── confusion_matrix.png
├── src/
│   ├── dip_porjc/
│   │   └── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── split_data.py
│   ├── train.py
│   └── visnverify_preprocess.py
├── trained/
│   └── best_resnet50.pth
├── .gitignore
├── .python-version
├── covid19-radiography-database.zip
├── list_of_commands.md
├── pyproject.toml
├── README.md
└── uv.lock
```

## Setup & Pipeline Execution (Cross-Platform)

The project relies on a universal `uv.lock` file configured with OS markers. It will automatically install Apple-optimized PyTorch binaries on macOS and CUDA-accelerated binaries on Windows.

**1. Sync the Environment**
Ensure `uv` is installed, then build the isolated environment:

```bash
uv sync

```

**2. Kaggle Authentication**
Generate an access token from Kaggle and securely store it.

* **macOS:**

```bash
mkdir -p ~/.kaggle && echo your-api-key > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

```

* **Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "your-api-key"

```

**3. Dataset Download & Extraction**

* **Download (Cross-Platform):**

```bash
uv run kaggle datasets download -d tawsifurrahman/covid19-radiography-database

```

* **Extract (macOS):**

```bash
unzip covid19-radiography-database.zip -d raw_data

```

* **Extract (Windows PowerShell):**

```powershell
New-Item -ItemType Directory -Force -Path "raw_data"
tar -xf covid19-radiography-database.zip -C raw_data
```

**4. Run Data Pipeline Scripts**
Flatten the nested Kaggle directories, drop out-of-scope classes, generate the stratified splits, and verify the tensor transformations:

```bash
uv run src/split_data.py
uv run src/data.py
uv run src/visualize_precprocess.py

```

## Model Engineer Handoff

The PyTorch data loaders are ready for immediate use. They handle CLAHE enhancement, ImageNet normalization, and class-balanced sampling dynamically in memory.

```python
from src.data import train_loader, val_loader, test_loader

# Iterate through perfectly balanced, augmented tensor batches
for images, labels in train_loader:
    # images shape: [32, 3, 224, 224] (float32)
    # labels shape: [32] (long)
    pass

```

## Model Engineering (Training Phase)

**Role:** ML Engineer (Member 2)
**Architecture:** ResNet50 (Transfer Learning)
**Hyperparameters:** Adam Optimizer (lr=0.001), Weighted CrossEntropyLoss (COVID: 2.5, Normal: 1.0, Viral Pneumonia: 2.5)

To train the model locally and generate the weights:
```bash
uv run src/train.py

```

### Training Phase Metrics

| Metric | Result |
| --- | --- |
| **Best Validation Accuracy** | **98.41%** (Epoch 10) |
| **Final Training Loss** | 0.0468 |
| **Final Validation Loss** | 0.0496 |
| **Compute Time Utilization** | 19 minutes, 12 seconds (NVIDIA CUDA) |
| **Epochs to Convergence** | Peaked at Epoch 10 (out of 10 total epochs) |
| **Model Checkpoints Generated** | 1 (`checkpoints/best_resnet50.pth`) |
| **Loss Curve Stability** | Training loss decreased steadily; Validation loss exhibited moderate oscillation before converging at Epoch 10. |

---

**Note to Test Engineer:** The trained weights are saved locally. Please download `best_resnet50.pth` from the provided Drive link and place it in the `checkpoints/` directory before running your evaluation scripts.
