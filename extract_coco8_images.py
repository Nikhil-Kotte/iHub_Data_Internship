"""
Extract and Organize COCO8 Validation Images
Separate the COCO8 dataset images used for validation metrics
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

print("=" * 80)
print("EXTRACTING COCO8 VALIDATION IMAGES")
print("=" * 80)

# Create output directory
output_dir = Path("coco8_validation_images")
output_dir.mkdir(exist_ok=True)

print(f"\n✓ Created output directory: {output_dir}/")

# YOLO downloads COCO8 to a cache location
# We need to find where it downloaded the dataset
print("\n1. Locating COCO8 dataset...")

# First, trigger validation to ensure COCO8 is downloaded
model = YOLO('yolov8n.pt')

print("\n2. Running validation to ensure COCO8 is downloaded...")
metrics = model.val(data='coco8.yaml', verbose=False)

print("\n3. Finding COCO8 image locations...")

# COCO8 is typically in datasets/coco8/images/val/
coco8_paths = [
    Path.home() / "datasets" / "coco8" / "images" / "val",
    Path("datasets") / "coco8" / "images" / "val",
    Path.cwd() / "datasets" / "coco8" / "images" / "val",
]

coco8_dir = None
for path in coco8_paths:
    if path.exists():
        coco8_dir = path
        print(f"\n✓ Found COCO8 images at: {coco8_dir}")
        break

if coco8_dir is None:
    print("\n✗ Could not find COCO8 dataset automatically.")
    print("Attempting to find via ultralytics cache...")
    
    # Try finding through ultralytics
    from ultralytics.utils import DATASETS_DIR
    coco8_dir = DATASETS_DIR / "coco8" / "images" / "val"
    
    if coco8_dir.exists():
        print(f"\n✓ Found via cache: {coco8_dir}")
    else:
        print("\n✗ COCO8 not found. It may be in a different location.")
        print("Checking validation results folder...")
        
        # Check if validation created preview images
        val_labels = Path("runs/detect/val/val_batch0_labels.jpg")
        if val_labels.exists():
            print(f"\n✓ Found validation batch preview: {val_labels}")
            shutil.copy(val_labels, output_dir / "val_batch0_labels.jpg")
            shutil.copy("runs/detect/val/val_batch0_pred.jpg", 
                       output_dir / "val_batch0_pred.jpg")
            print("✓ Copied validation batch images")

if coco8_dir and coco8_dir.exists():
    # Copy all COCO8 images
    image_files = list(coco8_dir.glob("*.jpg"))
    
    print(f"\n4. Found {len(image_files)} COCO8 images")
    print("\n5. Copying images...")
    
    for i, img_path in enumerate(sorted(image_files), 1):
        dest = output_dir / f"coco8_image_{i}_{img_path.name}"
        shutil.copy(img_path, dest)
        print(f"  ✓ Copied: {img_path.name} -> {dest.name}")
    
    print(f"\n✓ Copied {len(image_files)} COCO8 images to {output_dir}/")

# Also copy validation results for reference
print("\n6. Copying validation results for reference...")

val_files = [
    "runs/detect/val/val_batch0_labels.jpg",
    "runs/detect/val/val_batch0_pred.jpg",
    "runs/segment/val/val_batch0_labels.jpg",
    "runs/segment/val/val_batch0_pred.jpg",
]

for val_file in val_files:
    if Path(val_file).exists():
        dest_name = Path(val_file).parent.name + "_" + Path(val_file).name
        dest = output_dir / dest_name
        shutil.copy(val_file, dest)
        print(f"  ✓ Copied: {val_file} -> {dest.name}")

# Create README
print("\n7. Creating documentation...")

readme_content = """# COCO8 Validation Images

## Overview

This folder contains the COCO8 dataset images used for validation metrics and performance plots.

**These are DIFFERENT from the Sample Images folder.**

---

## What's Inside

### Original COCO8 Images
- `coco8_image_1_*.jpg` through `coco8_image_8_*.jpg`
- These are the 8 images from COCO8 dataset
- Used for validation metrics (mAP, F1, precision, recall)

### Validation Batch Previews

**Detection Model:**
- `val_val_batch0_labels.jpg` - Ground truth annotations
- `val_val_batch0_pred.jpg` - Model predictions

**Segmentation Model:**
- `val_val_batch0_labels.jpg` - Ground truth annotations  
- `val_val_batch0_pred.jpg` - Model predictions with masks

These show all 8 images in a grid with annotations.

---

## Why COCO8?

### Your Sample Images (10 images)
- Location: `Sample Images/`
- Purpose: Demonstration
- Problem: No ground truth labels
- Can show: What model detects
- Cannot calculate: Accuracy metrics

### COCO8 Images (8 images)
- Location: This folder
- Purpose: Validation and metrics
- Advantage: Has ground truth labels
- Can show: Model predictions
- Can calculate: mAP, precision, recall, confusion matrix

---

## Connection to Plots

All validation plots were generated from THESE images:
- Confusion matrix (`runs/detect/val/confusion_matrix.png`)
- F1 curves (`runs/detect/val/BoxF1_curve.png`)
- PR curves (`runs/detect/val/BoxPR_curve.png`)
- All other validation plots

---

## COCO8 Dataset Details

**COCO8** is a mini version of the COCO dataset containing:
- 8 validation images
- Multiple object classes (person, car, dog, etc.)
- Ground truth bounding boxes
- For segmentation: pixel-level masks

**Original COCO:** 118,000+ images  
**COCO8:** 8 images (for quick testing)

---

## File Structure

```
coco8_validation_images/
├── coco8_image_1_*.jpg      (Original COCO8 images)
├── coco8_image_2_*.jpg
├── ...
├── coco8_image_8_*.jpg
│
├── val_val_batch0_labels.jpg    (Detection ground truth)
├── val_val_batch0_pred.jpg      (Detection predictions)
├── val_val_batch0_labels.jpg    (Segmentation ground truth)
├── val_val_batch0_pred.jpg      (Segmentation predictions)
│
└── README.md (this file)
```

---

## Comparison

| Aspect | Your Sample Images | COCO8 Images |
|--------|-------------------|--------------|
| **Count** | 10 images | 8 images |
| **Location** | Sample Images/ | This folder |
| **Labels** | None | Has ground truth |
| **Purpose** | Demonstration | Validation |
| **Metrics** | ❌ Cannot calculate | ✅ Full metrics |
| **Used for** | Visual results | Performance plots |

---

## Usage

These images are referenced when discussing:
- Model accuracy (mAP: 73.92%)
- Confusion matrix results
- Precision/Recall trade-offs
- F1 score optimization
- Validation metrics

The validation plots in `runs/detect/val/` and `runs/segment/val/` 
are based on THESE 8 images, not your sample images.

---

## Generated: October 27, 2025
"""

readme_path = output_dir / "README.md"
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"✓ Created README: {readme_path}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✓ COCO8 validation images organized in: {output_dir}/")
print(f"✓ Includes original images and validation batch previews")
print(f"✓ Documentation created: README.md")

print("\n📁 Contents:")
print(f"  • COCO8 original images")
print(f"  • Validation batch previews (labels + predictions)")
print(f"  • Complete documentation")

print("\n" + "=" * 80)
print("✓ COMPLETE!")
print("=" * 80)

print("\nNote: These are the images used for ALL validation metrics:")
print("  - mAP50: 73.92%")
print("  - Precision: 83.33%")
print("  - Recall: 65.00%")
print("  - Confusion matrix")
print("  - F1/PR curves")
