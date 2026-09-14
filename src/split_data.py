from pathlib import Path
import splitfolders
import shutil

# Dynamically gets the project root (one folder up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "raw_data" / "COVID-19_Radiography_Dataset"

# 1. Remove loose metadata files (.xlsx, .txt) from the root so split-folders ignores them
for item in dataset_path.iterdir():
    if item.is_file():
        item.unlink()

# 2. Flatten the dataset (Move images up, delete masks)
for class_dir in dataset_path.iterdir():
    if class_dir.is_dir():
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"
        
        if images_dir.exists():
            # Move all image files into the parent class folder
            for img_file in images_dir.iterdir():
                shutil.move(str(img_file), str(class_dir / img_file.name))
            
            # Delete the now-empty 'images' and unused 'masks' folders
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(masks_dir, ignore_errors=True)

# 3. Remove Lung Opacity (Since your scope is COVID vs. Pneumonia)
lung_opacity_dir = dataset_path / "Lung_Opacity"
if lung_opacity_dir.exists():
    shutil.rmtree(lung_opacity_dir, ignore_errors=True)

# 4. Execute the stratified split
splitfolders.ratio(
    input=dataset_path, 
    output=BASE_DIR / "dataset_split", 
    seed=42, 
    ratio=(0.70, 0.15, 0.15)
)
print("Data formatting and splitting complete.")