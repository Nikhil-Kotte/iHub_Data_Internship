# COCO8 Validation Images

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
