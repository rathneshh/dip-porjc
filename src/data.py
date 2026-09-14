import time
from pathlib import Path
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CovidXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in cls_dir.glob("*.png"):
                self.image_paths.append(str(img_path))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)

    def check_data_quality(self):
        """Scans for corrupted images and calculates the Data Quality Score."""
        valid_paths, valid_labels = [], []
        corrupted = 0
        
        print("Scanning dataset for corrupted files...")
        for img_path, label in zip(self.image_paths, self.labels):
            img = cv2.imread(img_path)
            if img is None:
                corrupted += 1
            else:
                valid_paths.append(img_path)
                valid_labels.append(label)
                
        self.image_paths = valid_paths
        self.labels = valid_labels
        
        quality_score = (len(self.image_paths) / (len(self.image_paths) + corrupted)) * 100
        print(f"Data Quality Score: {quality_score:.2f}% ({corrupted} corrupted files removed)")


def get_balanced_sampler(dataset):
    """Calculates weights for each class to handle imbalance."""
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    print(f"Class counts (Imbalance Ratio): {dict(zip(dataset.classes, class_counts))}")
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# --- TRANSFORMS DEFINED GLOBALLY FOR EXTERNAL IMPORT ---
train_transform = A.Compose([
    A.Resize(224, 224),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_test_transform = A.Compose([
    A.Resize(224, 224),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    SPLIT_DIR = BASE_DIR / "dataset_split"
    
    # 1. Initialize Train Dataset & Check Quality
    train_dataset = CovidXrayDataset(SPLIT_DIR / "train", transform=train_transform)
    train_dataset.check_data_quality()
    
    # 2. Handle Imbalance with Sampler
    sampler = get_balanced_sampler(train_dataset)
    
    # 3. Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    
    # 4. Benchmark Transformation Pipeline Speed
    print("\nBenchmarking Transformation Pipeline Speed...")
    start_time = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx == 10: # Test only the first 10 batches
            break
    end_time = time.time()
    print(f"Pipeline Speed: {(end_time - start_time):.2f} seconds per 10 batches (Size 32).")

    # 5. Initialize Validation and Test DataLoaders
    val_dataset = CovidXrayDataset(SPLIT_DIR / "val", transform=val_test_transform)
    test_dataset = CovidXrayDataset(SPLIT_DIR / "test", transform=val_test_transform)

    # Validation and Test sets do not use the sampler and are not shuffled
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Pipeline ready: {len(train_dataset)} Train | {len(val_dataset)} Val | {len(test_dataset)} Test images.")