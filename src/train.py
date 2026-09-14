import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pathlib import Path
from torch.utils.data import DataLoader

# 1. Import the building blocks from Member 1's script
from data import CovidXrayDataset, get_balanced_sampler, train_transform, val_test_transform

def train_model():
    # --- NEW: Build the DataLoaders here instead ---
    print("Setting up data loaders...")
    BASE_DIR = Path(__file__).resolve().parent.parent
    SPLIT_DIR = BASE_DIR / "dataset_split"
    
    train_dataset = CovidXrayDataset(SPLIT_DIR / "train", transform=train_transform)
    sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
    
    val_dataset = CovidXrayDataset(SPLIT_DIR / "val", transform=val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    # -----------------------------------------------

    # 2. Setup Device (Uses Nvidia GPU if available, else standard CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on device: {device}")

    # 3. Model Architecture (ResNet50)
    print("Loading ResNet50 model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3) 
    model = model.to(device)

    # 4. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training Loop Setup
    num_epochs = 10
    best_val_acc = 0.0
    checkpoint_dir = BASE_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # 6. Model Checkpoint Saving
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            save_path = checkpoint_dir / "best_resnet50.pth"
            torch.save(model.state_dict(), save_path)
            print(f"*** New best model saved to {save_path} ***")

if __name__ == "__main__":
    train_model()