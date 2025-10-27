"""
Sample Images Analysis - What's Possible Without Ground Truth

This script demonstrates what metrics CAN and CANNOT be calculated
from your sample images without ground truth annotations.
"""

from ultralytics import YOLO
from pathlib import Path
import json
import time

print("=" * 80)
print("SAMPLE IMAGES ANALYSIS")
print("Understanding Metrics Limitations")
print("=" * 80)

print("\n⚠️  IMPORTANT LIMITATION:")
print("Your sample images do NOT have ground truth labels (annotations).")
print("This means we CANNOT calculate:")
print("  ❌ Confusion Matrix (need actual vs predicted)")
print("  ❌ Precision/Recall (need true positives/false positives)")
print("  ❌ mAP scores (need ground truth to compare)")
print("  ❌ F1 curves (derived from precision/recall)")
print("\n✅ What we CAN calculate:")
print("  ✓ Inference timing metrics")
print("  ✓ Detection counts and distributions")
print("  ✓ Comparison between models")
print("  ✓ FPS and throughput")

print("\n" + "=" * 80)
print("AVAILABLE METRICS FOR YOUR IMAGES")
print("=" * 80)

# Load models
model_det = YOLO('yolov8n.pt')
model_seg = YOLO('yolov8n-seg.pt')

# Get sample images
image_folder = Path("Sample Images")
images = sorted(image_folder.glob("*.jpg"))

print(f"\nProcessing {len(images)} images...")

# Collect metrics
metrics = {
    'detection': {'images': [], 'timing': []},
    'segmentation': {'images': [], 'timing': []}
}

# Process detection
print("\n1. Detection Model Processing...")
for img in images:
    start = time.time()
    results = model_det(str(img), verbose=False)
    elapsed = time.time() - start
    
    result = results[0]
    boxes = result.boxes
    
    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model_det.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    metrics['detection']['images'].append({
        'name': img.name,
        'objects': len(boxes),
        'classes': class_counts
    })
    metrics['detection']['timing'].append(elapsed)

# Process segmentation
print("2. Segmentation Model Processing...")
for img in images:
    start = time.time()
    results = model_seg(str(img), verbose=False)
    elapsed = time.time() - start
    
    result = results[0]
    boxes = result.boxes
    masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
    
    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = model_seg.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    metrics['segmentation']['images'].append({
        'name': img.name,
        'objects': len(boxes),
        'masks': len(masks.data) if masks else 0,
        'classes': class_counts
    })
    metrics['segmentation']['timing'].append(elapsed)

# Calculate summary statistics
print("\n" + "=" * 80)
print("✅ AVAILABLE METRICS SUMMARY")
print("=" * 80)

det_avg_time = sum(metrics['detection']['timing']) / len(metrics['detection']['timing'])
seg_avg_time = sum(metrics['segmentation']['timing']) / len(metrics['segmentation']['timing'])

det_total_objects = sum(m['objects'] for m in metrics['detection']['images'])
seg_total_objects = sum(m['objects'] for m in metrics['segmentation']['images'])

print(f"\n📊 TIMING METRICS (Available)")
print(f"{'─' * 80}")
print(f"\nDetection Model:")
print(f"  • Average time: {det_avg_time:.4f}s per image")
print(f"  • FPS: {1/det_avg_time:.2f} frames/second")
print(f"  • Total time: {sum(metrics['detection']['timing']):.4f}s")

print(f"\nSegmentation Model:")
print(f"  • Average time: {seg_avg_time:.4f}s per image")
print(f"  • FPS: {1/seg_avg_time:.2f} frames/second")
print(f"  • Total time: {sum(metrics['segmentation']['timing']):.4f}s")

time_diff = ((seg_avg_time - det_avg_time) / det_avg_time) * 100
print(f"\n⚡ Performance Comparison:")
print(f"  • Segmentation is {time_diff:.1f}% slower than detection")

print(f"\n\n📦 DETECTION COUNTS (Available)")
print(f"{'─' * 80}")
print(f"  • Total objects detected: {det_total_objects}")
print(f"  • Average per image: {det_total_objects/len(images):.1f}")

print(f"\n📦 SEGMENTATION COUNTS (Available)")
print(f"{'─' * 80}")
print(f"  • Total objects detected: {seg_total_objects}")
print(f"  • Average per image: {seg_total_objects/len(images):.1f}")

# Class distribution
from collections import Counter
all_classes_det = Counter()
all_classes_seg = Counter()

for m in metrics['detection']['images']:
    for cls, count in m['classes'].items():
        all_classes_det[cls] += count

for m in metrics['segmentation']['images']:
    for cls, count in m['classes'].items():
        all_classes_seg[cls] += count

print(f"\n🏷️  CLASS DISTRIBUTION (Available)")
print(f"{'─' * 80}")
print(f"\nDetection Model - Most Common:")
for cls, count in all_classes_det.most_common(5):
    print(f"  • {cls}: {count} instances")

print(f"\nSegmentation Model - Most Common:")
for cls, count in all_classes_seg.most_common(5):
    print(f"  • {cls}: {count} instances")

# Save metrics
output = Path("sample_images_metrics.json")
with open(output, 'w') as f:
    json.dump({
        'available_metrics': {
            'detection': {
                'avg_time_sec': det_avg_time,
                'fps': 1/det_avg_time,
                'total_objects': det_total_objects,
                'class_distribution': dict(all_classes_det)
            },
            'segmentation': {
                'avg_time_sec': seg_avg_time,
                'fps': 1/seg_avg_time,
                'total_objects': seg_total_objects,
                'class_distribution': dict(all_classes_seg)
            }
        },
        'per_image_results': metrics
    }, f, indent=2)

print(f"\n✓ Metrics saved to: {output}")

print("\n" + "=" * 80)
print("❌ UNAVAILABLE METRICS (Need Ground Truth)")
print("=" * 80)

print("""
The following metrics CANNOT be calculated without ground truth annotations:

1. Confusion Matrix
   - Requires: Knowing which objects SHOULD be in each image
   - Shows: Actual class vs Predicted class
   - Example: Can't tell if "cat" detection is correct without label

2. Precision
   - Requires: True Positives + False Positives
   - Formula: TP / (TP + FP)
   - Need to know: Which detections are actually correct

3. Recall
   - Requires: True Positives + False Negatives  
   - Formula: TP / (TP + FN)
   - Need to know: How many objects we missed

4. mAP (mean Average Precision)
   - Requires: Precision-Recall curve across IoU thresholds
   - Need: Ground truth boxes to calculate IoU

5. F1 Score
   - Requires: Precision and Recall
   - Formula: 2 × (P × R) / (P + R)

6. IoU (Intersection over Union)
   - Requires: Ground truth boxes
   - Cannot compare prediction to unknown ground truth
""")

print("\n" + "=" * 80)
print("💡 SOLUTION: CREATE ANNOTATIONS")
print("=" * 80)

print("""
To get full validation metrics for YOUR images, you need to:

Option 1: Manual Annotation
  1. Use tools like CVAT, LabelImg, or Roboflow
  2. Draw bounding boxes around all objects
  3. Label each object with correct class
  4. Export in YOLO format
  5. Create dataset YAML file
  6. Run: model.val(data='your_data.yaml', plots=True)

Option 2: Use Pre-annotated Dataset (Current Approach)
  ✅ Use COCO8 for validation metrics
  ✅ Use your images for demonstration
  ✅ Both serve different but valid purposes

Option 3: Semi-Automatic Annotation
  1. Use model predictions as starting point
  2. Manually correct and verify
  3. Export as ground truth
  4. Then run validation
""")

print("\n" + "=" * 80)
print("📝 RECOMMENDATION")
print("=" * 80)

print("""
Your current approach is CORRECT and PROFESSIONAL:

✅ Sample Images (yours):
   - Show what the model can do
   - Demonstrate on real-world data
   - Visual results in individual_results/
   
✅ COCO8 Dataset:
   - Validate model accuracy
   - Generate all performance metrics
   - Standard benchmark comparison

This is how it's done in industry!
- Demo dataset: Shows capabilities
- Validation dataset: Proves accuracy
""")

print("\n" + "=" * 80)
print("✅ COMPLETE")
print("=" * 80)

print("\nAvailable metrics for your sample images saved to:")
print(f"  • {output}")
print(f"  • individual_results/ (visual results)")
print("\nFor full validation metrics, see:")
print(f"  • runs/detect/val/ (COCO8 metrics)")
print(f"  • coco8_validation_images/ (validation images)")
