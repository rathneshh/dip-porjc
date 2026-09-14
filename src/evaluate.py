import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Import your dataset blueprints
from data import CovidXrayDataset, val_test_transform

def evaluate_model():
    print("Setting up validation data...")
    BASE_DIR = Path(__file__).resolve().parent.parent
    SPLIT_DIR = BASE_DIR / "dataset_split"
    
    # Load the validation data
    test_dataset = CovidXrayDataset(SPLIT_DIR / "test", transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load the blank ResNet50 architecture
    print("Loading trained ResNet50 model weights...")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)
    
    # Inject the "brain" (weights) you saved during training
    checkpoint_path = BASE_DIR / "checkpoints" / "best_resnet50.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval() # Lock the model for testing

    all_preds = []
    all_labels = []

    print("Analyzing predictions (this might take a minute)...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*70)
    print("           SCIENTIFICALLY VALID DIAGNOSTIC EVALUATION")
    print("="*70)
    print("MEDICAL METRICS CONTEXT:")
    print("- Recall (Sensitivity): Critical to avoid sending sick patients home (False Negatives).")
    print("- Precision (PPV): Measures False Positives to prevent unnecessary clinical stress.")
    print("- F1-Score: Harmonic mean providing a robust single score to detect over-prediction.")
    print("="*70 + "\n")
    
    # Generate Text Report
    target_names = ["COVID", "Normal", "Viral Pneumonia"]
    report = classification_report(all_labels, all_preds, digits=4, target_names=target_names)
    print(report)

    # Generate Visual Confusion Matrix
    print("Generating visual confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Diagnostic Confusion Matrix')
    plt.tight_layout()

    # Save the plot securely (better for remote development)
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    save_path = results_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300)
    print(f"*** Visual matrix saved successfully to {save_path} ***")

if __name__ == "__main__":
    evaluate_model()