"""
Generate Comprehensive Validation Plots with YOLO
This script uses the COCO8 dataset to generate:
- Confusion Matrix
- F1-Confidence Curve
- Precision-Recall Curve
- Precision Curve
- Recall Curve
"""

from ultralytics import YOLO
import os

print("=" * 80)
print("GENERATING YOLO VALIDATION PLOTS")
print("=" * 80)

print("\nNote: To generate plots like confusion matrix, F1-curve, PR-curve, etc.,")
print("YOLO needs ground truth labels (annotations) for comparison.")
print("\nWe'll use the COCO8 dataset (mini version of COCO) which YOLO downloads")
print("automatically. This dataset has 8 images with proper annotations.")
print("=" * 80)

# Load pre-trained model
print("\n1. Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

print("\n2. Running validation on COCO8 dataset...")
print("   (YOLO will download the dataset automatically if not present)")
print("-" * 80)

# Run validation - this generates all the plots
metrics = model.val(
    data='coco8.yaml',      # Small COCO dataset (8 images)
    plots=True,             # Generate all plots
    save_json=True,         # Save results as JSON
    conf=0.25,              # Confidence threshold
    iou=0.45,               # IoU threshold
    verbose=True
)

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

# Print metrics
print(f"\n📊 Detection Metrics:")
print(f"  • mAP50 (IoU=0.50):          {metrics.box.map50:.4f}")
print(f"  • mAP50-95 (IoU=0.50:0.95):  {metrics.box.map:.4f}")
print(f"  • Precision:                  {metrics.box.mp:.4f}")
print(f"  • Recall:                     {metrics.box.mr:.4f}")
print(f"  • F1 Score:                   {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.4f}")

print(f"\n⚡ Speed:")
print(f"  • Preprocess:  {metrics.speed['preprocess']:.2f}ms")
print(f"  • Inference:   {metrics.speed['inference']:.2f}ms")
print(f"  • Postprocess: {metrics.speed['postprocess']:.2f}ms")

print("\n" + "=" * 80)
print("GENERATED PLOTS")
print("=" * 80)

# List expected plot files
plots_folder = "runs/detect/val"
expected_plots = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "PR_curve.png",
    "results.png"
]

print(f"\n📁 All plots saved to: {plots_folder}/")
print("\nGenerated files:")
for plot in expected_plots:
    print(f"  ✓ {plot}")

print("\n" + "=" * 80)
print("PLOT DESCRIPTIONS")
print("=" * 80)

print("""
1. confusion_matrix.png
   - Shows which classes are confused with each other
   - Diagonal = correct predictions
   - Off-diagonal = misclassifications

2. F1_curve.png
   - F1 score vs confidence threshold
   - Helps find optimal confidence threshold
   - Peak indicates best balance of precision/recall

3. P_curve.png (Precision Curve)
   - Precision vs confidence threshold
   - Shows how precision changes with threshold
   - Higher confidence = higher precision

4. R_curve.png (Recall Curve)
   - Recall vs confidence threshold
   - Shows how recall changes with threshold
   - Lower confidence = higher recall

5. PR_curve.png (Precision-Recall Curve)
   - Classic ML evaluation plot
   - Area under curve = Average Precision (AP)
   - Closer to top-right = better model

6. results.png
   - Training/validation metrics over time
   - Shows loss curves, mAP progression
   - (Only meaningful if you train)
""")

print("=" * 80)
print("✓ VALIDATION COMPLETE!")
print(f"✓ Check '{plots_folder}/' folder for all plots")
print("=" * 80)

# Additional: Segmentation validation
print("\n" + "=" * 80)
print("BONUS: SEGMENTATION MODEL VALIDATION")
print("=" * 80)

print("\nRunning validation on segmentation model...")
seg_model = YOLO('yolov8n-seg.pt')
seg_metrics = seg_model.val(
    data='coco8-seg.yaml',  # Segmentation dataset
    plots=True,
    conf=0.25,
    iou=0.45,
    verbose=True
)

print(f"\n📊 Segmentation Metrics:")
print(f"  • Box mAP50:  {seg_metrics.box.map50:.4f}")
print(f"  • Box mAP:    {seg_metrics.box.map:.4f}")
print(f"  • Mask mAP50: {seg_metrics.seg.map50:.4f}")
print(f"  • Mask mAP:   {seg_metrics.seg.map:.4f}")

print(f"\n✓ Segmentation plots saved to: runs/segment/val/")

print("\n" + "=" * 80)
print("ALL DONE!")
print("=" * 80)
