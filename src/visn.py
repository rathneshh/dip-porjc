import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import your datasets and transforms
from data import CovidXrayDataset, train_transform, val_test_transform

def denormalize(tensor):
    """Reverses ImageNet normalization so matplotlib can display the image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert tensor (C, H, W) to numpy (H, W, C)
    img = tensor.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def inspect_random_samples():
    BASE_DIR = Path(__file__).resolve().parent.parent
    SPLIT_DIR = BASE_DIR / "dataset_split"
    
    print("Loading datasets...")
    train_data = CovidXrayDataset(SPLIT_DIR / "train", transform=train_transform)
    val_data = CovidXrayDataset(SPLIT_DIR / "val", transform=val_test_transform)
    test_data = CovidXrayDataset(SPLIT_DIR / "test", transform=val_test_transform)
    
    # Select 2 random indices from each
    train_idx = random.sample(range(len(train_data)), 2)
    val_idx = random.sample(range(len(val_data)), 2)
    test_idx = random.sample(range(len(test_data)), 2)
    
    samples = [
        ("Train", train_data[train_idx[0]]), ("Train", train_data[train_idx[1]]),
        ("Val", val_data[val_idx[0]]), ("Val", val_data[val_idx[1]]),
        ("Test", test_data[test_idx[0]]), ("Test", test_data[test_idx[1]])
    ]
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle("Preprocessed X-Ray Samples (Grad-CAM Inspection)", fontsize=16)
    
    for ax, (split_name, (image_tensor, label)) in zip(axes.flatten(), samples):
        img = denormalize(image_tensor)
        class_name = list(train_data.class_to_idx.keys())[label.item()]
        
        ax.imshow(img)
        ax.set_title(f"{split_name} | {class_name}")
        ax.axis("off")
        
    plt.tight_layout()
    save_path = BASE_DIR / "results" / "preprocessing_inspection.png"
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Inspection grid saved to {save_path}")

if __name__ == "__main__":
    inspect_random_samples()