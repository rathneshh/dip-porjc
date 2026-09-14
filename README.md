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
├── raw_data/                    # Ignored in git (Raw Kaggle downloads)
├── dataset_split/               # Ignored in git (Stratified splits)
├── src/
│   ├── split_data.py            # Flattens Kaggle structure & stratifies classes
│   ├── data.py                  # PyTorch Dataset, CLAHE transforms, DataLoaders
│   ├── visnverify_preprocess.py # Validation visualizer & tensor assertions
│   └── train.py                 # ResNet50 training loop
├── pyproject.toml               # Python dependencies & OS-specific GPU routing
├── uv.lock                      # Universal cross-platform lockfile
└── README.md

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
Expand-Archive -Path "covid19-radiography-database.zip" -DestinationPath "raw_data" -Force

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