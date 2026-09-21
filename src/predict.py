import argparse
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from pathlib import Path

# Import your validation transform so the image is processed identically
from data import val_test_transform

def predict_image(image_path):
    BASE_DIR = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load the trained model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    checkpoint_path = BASE_DIR / "checkpoints" / "best_resnet50.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 2. Load and preprocess the unseen image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = val_test_transform(image=img)
    img_tensor = augmented['image'].unsqueeze(0).to(device) # Add batch dimension

    # 3. Run Inference
    class_names = ["COVID", "Normal", "Viral Pneumonia"]
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # 4. Display Results
    print(f"\n--- Prediction Results for {Path(image_path).name} ---")
    results = {class_names[i]: float(probabilities[i]) * 100 for i in range(3)}
    
    # Sort by highest confidence
    for disease, conf in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{disease}: {conf:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict disease from a chest X-ray.")
    parser.add_argument("image_path", type=str, help="Path to the unknown X-ray image file")
    args = parser.parse_args()
    
    predict_image(args.image_path)