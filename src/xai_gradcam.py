import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader, Subset
from torchvision import models

# Import the testing data from the sibling data script when run directly.
from data import CovidXrayDataset, val_test_transform

# ---------------------------------------------------------
# 1. Grad-CAM Implementation
# ---------------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target_score = output[0, class_idx]
        target_score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, class_idx

# ---------------------------------------------------------
# 2. Image Helper
# ---------------------------------------------------------
def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor.cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# ---------------------------------------------------------
# 3. Execution Pipeline (Using OpenCV for Saving)
# ---------------------------------------------------------
def main(args):
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    print("Loading model...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3) 
    
    checkpoint = torch.load(
        base_dir / "checkpoints" / "best_resnet50.pth",
        map_location=device,
        weights_only=True,
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    class_names = ["COVID-19", "Normal", "Viral Pneumonia"]
    
    print("Extracting images from the test set...")
    test_dir = base_dir / "dataset_split" / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test data not found at {test_dir}. Run `uv run src/split_data.py` first."
        )
    test_dataset = CovidXrayDataset(
        test_dir,
        transform=val_test_transform,
    )
    selected_indices = []
    for class_idx in range(len(class_names)):
        class_indices = [
            index for index, label in enumerate(test_dataset.labels)
            if label == class_idx
        ]
        if args.all:
            selected_indices.extend(class_indices)
        else:
            selected_indices.extend(class_indices[:args.per_class])

    selected_dataset = Subset(test_dataset, selected_indices)
    test_loader = DataLoader(
        selected_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"Generating Grad-CAM for {len(selected_dataset)} images...")
    
    for i, (images, labels) in enumerate(test_loader, start=1):
        img_tensor = images.to(device)
        true_label = labels.item()
        
        heatmap, pred_label = grad_cam.generate(img_tensor)
        
        raw_img = unnormalize(images[0])
        raw_img = np.transpose(raw_img, (1, 2, 0))
        raw_img_uint8 = np.uint8(255 * raw_img)
        
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Hide weak activations so the overlay emphasizes meaningful peaks.
        activation_threshold = 0.35
        heatmap_strength = np.clip(
            (heatmap - activation_threshold) / (1.0 - activation_threshold),
            0.0,
            1.0,
        )
        alpha = (0.55 * heatmap_strength)[..., np.newaxis]
        overlay = (
            raw_img_uint8.astype(np.float32) * (1.0 - alpha)
            + heatmap_colored.astype(np.float32) * alpha
        ).astype(np.uint8)
        
        # --- NEW OpenCV Visualization Logic ---
        # OpenCV saves in BGR format, so we convert from RGB
        raw_bgr = cv2.cvtColor(raw_img_uint8, cv2.COLOR_RGB2BGR)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Add text labels to the images
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(raw_bgr, f"True: {class_names[true_label]}", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(overlay_bgr, f"Pred: {class_names[pred_label]}", (10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Stitch the two images side-by-side horizontally
        combined = cv2.hconcat([raw_bgr, overlay_bgr])
        
        save_path = output_dir / f"gradcam_{i:03d}.png"
        cv2.imwrite(save_path, combined)
        
        print(f"Successfully saved: {save_path}")

    print("\nAll done! Check the 'outputs' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate class-balanced Grad-CAM images.")
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Number of images to explain per class (default: 5).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explain every image in the test set.",
    )
    args = parser.parse_args()
    main(args)
