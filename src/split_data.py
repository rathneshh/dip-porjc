from pathlib import Path
import splitfolders
import shutil
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
dataset_path = BASE_DIR / "raw_data" / "COVID-19_Radiography_Dataset"

print("Cleaning metadata...")
for item in dataset_path.iterdir():
    if item.is_file():
        item.unlink()

print("Cropping images to lung bounding boxes...")
for class_dir in dataset_path.iterdir():
    if class_dir.is_dir() and class_dir.name != "Lung_Opacity":
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"
        
        if images_dir.exists() and masks_dir.exists():
            for img_file in images_dir.iterdir():
                mask_file = masks_dir / img_file.name
                
                if mask_file.exists():
                    img = cv2.imread(str(img_file))
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None and mask is not None:
                        # Force dimensions to match if necessary
                        if img.shape[:2] != mask.shape:
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # Find coordinates of all white pixels in the mask
                        points = cv2.findNonZero(mask)
                        
                        if points is not None:
                            # Calculate the tightest bounding box around the lungs
                            x, y, w, h = cv2.boundingRect(points)
                            
                            # Crop the original image (removing peripheral text and empty space)
                            cropped_img = img[y:y+h, x:x+w]
                            cv2.imwrite(str(class_dir / img_file.name), cropped_img)
                        else:
                            # Fallback just in case a mask is completely blank
                            cv2.imwrite(str(class_dir / img_file.name), img)
            
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(masks_dir, ignore_errors=True)

lung_opacity_dir = dataset_path / "Lung_Opacity"
if lung_opacity_dir.exists():
    shutil.rmtree(lung_opacity_dir, ignore_errors=True)

print("Executing stratified split...")
splitfolders.ratio(
    input=dataset_path, 
    output=BASE_DIR / "dataset_split", 
    seed=42, 
    ratio=(0.70, 0.15, 0.15)
)
print("Data formatting and splitting complete.")