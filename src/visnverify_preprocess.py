import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure src directory is on Python path
sys.path.append(str(Path(__file__).resolve().parent))
from data import CovidXrayDataset, get_balanced_sampler, train_transform

BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "dataset_split"

def denormalize_image(tensor):
    """Reverses ImageNet normalization for plotting (C, H, W) -> (H, W, C)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    return np.clip(img, 0, 1)

def run_tensor_assertions(train_loader, train_dataset):
    print("\n" + "="*50)
    print("1. RUNNING PROGRAMMATIC ASSERTIONS")
    print("="*50)
    
    images, labels = next(iter(train_loader))
    
    # 1. Dimensions check
    assert images.shape == (32, 3, 224, 224), f"Shape failure: Expected (32, 3, 224, 224), got {images.shape}"
    assert labels.shape == (32,), f"Label shape failure: Expected (32,), got {labels.shape}"
    print("[PASS] Tensor Shapes: Batch shape is (32, 3, 224, 224), Labels are (32,)")

    # 2. Data type check
    assert images.dtype == torch.float32, f"Dtype failure: Expected torch.float32, got {images.dtype}"
    assert labels.dtype == torch.long, f"Label dtype failure: Expected torch.long, got {labels.dtype}"
    print("[PASS] Data Types: Float32 for images, Long for class indices")

    # 3. Normalization value check
    min_val, max_val = images.min().item(), images.max().item()
    assert -3.0 < min_val < 0.0 and 1.0 < max_val < 3.0, f"Normalization out of bounds: min={min_val}, max={max_val}"
    print(f"[PASS] Value Normalization: ImageNet bounds verified (min: {min_val:.2f}, max: {max_val:.2f})")

    # 4. Sampler balance check across multiple batches
    print("\nChecking Class Balance via WeightedRandomSampler across 5 batches (160 samples):")
    total_counts = np.zeros(len(train_dataset.classes), dtype=int)
    for i, (_, batch_labels) in enumerate(train_loader):
        if i >= 5:
            break
        unique, counts = torch.unique(batch_labels, return_counts=True)
        for u, c in zip(unique.tolist(), counts.tolist()):
            total_counts[u] += c

    for idx, class_name in enumerate(train_dataset.classes):
        print(f"  - {class_name:16}: {total_counts[idx]} samples ({total_counts[idx]/160:.1%})")

def plot_visual_inspection(sample_img_path):
    print("\n" + "="*50)
    print("2. GENERATING COMPARATIVE VISUALIZATIONS")
    print("="*50)

    raw_bgr = cv2.imread(str(sample_img_path))
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    # Generate CLAHE alone
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe_op.apply(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY))
    
    # Generate full pipeline output (CLAHE + Rotation/Flip + Normalization)
    augmented = train_transform(image=raw_rgb)["image"]
    denorm_aug = denormalize_image(augmented)

    # Figure 1: Visual comparison across processing stages
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(raw_rgb)
    axes[0].set_title("1. Original Raw X-Ray")
    axes[0].axis("off")

    axes[1].imshow(clahe_gray, cmap="gray")
    axes[1].set_title("2. CLAHE Local Contrast Enhanced")
    axes[1].axis("off")

    axes[2].imshow(denorm_aug)
    axes[2].set_title("3. Full Pipeline (Augmented + Scaled)")
    axes[2].axis("off")

    plt.suptitle(f"Preprocessing Pipeline Inspection: {sample_img_path.name}", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Figure 2: Histogram redistribution check
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(9, 4))
    plt.plot(cv2.calcHist([raw_gray], [0], None, [256], [0, 256]), color="dimgray", label="Raw Histogram")
    plt.plot(cv2.calcHist([clahe_gray], [0], None, [256], [0, 256]), color="dodgerblue", label="CLAHE Redistributed")
    plt.title("Pixel Intensity Redistribution (CLAHE Verification)")
    plt.xlabel("Pixel Intensity [0 - 255]")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_dataloader_batch(train_loader, train_dataset):
    # Figure 3: Inspect what the model receives in a live batch
    images, labels = next(iter(train_loader))
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img = denormalize_image(images[i])
        ax.imshow(img)
        ax.set_title(f"{train_dataset.classes[labels[i]]}")
        ax.axis("off")
        
    plt.suptitle("Live Model Input Batch (8-sample preview with Sampler)", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_dataset = CovidXrayDataset(SPLIT_DIR / "train", transform=train_transform)
    sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)

    # 1. Assertions & Class Balance Verification
    run_tensor_assertions(train_loader, train_dataset)

    # 2. Visual comparison & Histogram Check (picks a random COVID X-ray)
    covid_samples = list((SPLIT_DIR / "train" / "COVID").glob("*.png"))
    sample_path = covid_samples[0]
    plot_visual_inspection(sample_path)

    # 3. Model batch preview
    plot_dataloader_batch(train_loader, train_dataset)