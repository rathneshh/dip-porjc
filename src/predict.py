import argparse
import torch
import torch.nn as nn
from torchvision import models
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Force strict determinism at the PyTorch & Hardware level
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def auto_crop_xray(img):
    """Dynamically crops out pure black backgrounds and applies a 5% inner margin in memory."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(thresh)
    
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        
        x_tight = x + margin_x
        y_tight = y + margin_y
        w_tight = w - (2 * margin_x)
        h_tight = h - (2 * margin_y)
        
        return img[y_tight:y_tight+h_tight, x_tight:x_tight+w_tight]
    
    return img

def predict_image(image_path):
    BASE_DIR = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    

    # 2. Define transform LOCALLY.
    val_test_transform = A.Compose([
        A.Resize(height=224, width=224),
        # Fix: Force CLAHE to use exactly 2.0 instead of a random range
        A.CLAHE(clip_limit=(2.0, 2.0), tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 3. Load the trained model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    checkpoint_path = BASE_DIR / "checkpoints" / "best_resnet50.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 4. Load and preprocess the unseen image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    print(f"Applying automatic lung cropping to {Path(image_path).name}...")
    img = auto_crop_xray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"debug_cropped_{Path(image_path).name}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Apply strict deterministic validation transform
    augmented = val_test_transform(image=img)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    class_names = ["COVID", "Normal", "Viral Pneumonia"]
    
    # 5. Run Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    print(f"\n--- Prediction Results for {Path(image_path).name} ---")
    results = {class_names[i]: float(probabilities[i]) * 100 for i in range(3)}
    
    for disease, conf in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{disease}: {conf:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict disease from a raw chest X-ray.")
    parser.add_argument("image_path", type=str, help="Path to the unknown X-ray image file")
    args = parser.parse_args()
    
    predict_image(args.image_path)