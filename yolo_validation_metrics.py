"""
Using YOLO's Built-in Validation to Generate Performance Metrics
This demonstrates YOLO's native evaluation capabilities
"""

from ultralytics import YOLO
import os

print("=" * 80)
print("YOLO BUILT-IN VALIDATION & METRICS")
print("=" * 80)

# Load models
print("\n1. Loading Models...")
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8n-seg.pt')

# Method 1: Validation on custom images
print("\n2. Running Validation on Sample Images...")
print("-" * 80)

# YOLO's val() method generates comprehensive metrics
# It expects a dataset format, but we can use predict with save=True for similar results

# Method 2: Predict with detailed results
print("\nRunning predictions with detailed metrics...")

image_folder = "Sample Images"

# Detection with metrics
print("\n📊 Detection Model Validation:")
det_results = detection_model.predict(
    source=image_folder,
    save=True,
    project="runs/detect",
    name="validation_metrics",
    conf=0.25,
    verbose=True
)

print(f"\n✓ Detection results saved to: runs/detect/validation_metrics/")

# Segmentation with metrics
print("\n📊 Segmentation Model Validation:")
seg_results = segmentation_model.predict(
    source=image_folder,
    save=True,
    project="runs/segment",
    name="validation_metrics",
    conf=0.25,
    verbose=True
)

print(f"\n✓ Segmentation results saved to: runs/segment/validation_metrics/")

print("\n" + "=" * 80)
print("ANALYZING PREDICTION RESULTS")
print("=" * 80)

# Extract detailed metrics from results
print("\nDetection Model Summary:")
print("-" * 80)

total_det_time = 0
for i, result in enumerate(det_results):
    speed = result.speed
    boxes = result.boxes
    
    preprocess = speed['preprocess']
    inference = speed['inference']
    postprocess = speed['postprocess']
    total = preprocess + inference + postprocess
    total_det_time += total
    
    print(f"\nImage {i+1}: {result.path}")
    print(f"  Objects detected: {len(boxes)}")
    print(f"  Preprocess:  {preprocess:.2f}ms")
    print(f"  Inference:   {inference:.2f}ms")
    print(f"  Postprocess: {postprocess:.2f}ms")
    print(f"  Total:       {total:.2f}ms")

avg_det_time = total_det_time / len(det_results)
print(f"\n📈 Detection Average: {avg_det_time:.2f}ms per image")
print(f"📈 Detection FPS: {1000/avg_det_time:.2f} frames/sec")

print("\n" + "=" * 80)
print("Segmentation Model Summary:")
print("-" * 80)

total_seg_time = 0
for i, result in enumerate(seg_results):
    speed = result.speed
    boxes = result.boxes
    masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
    
    preprocess = speed['preprocess']
    inference = speed['inference']
    postprocess = speed['postprocess']
    total = preprocess + inference + postprocess
    total_seg_time += total
    
    print(f"\nImage {i+1}: {result.path}")
    print(f"  Objects detected: {len(boxes)}")
    print(f"  Masks generated: {len(masks.data) if masks else 0}")
    print(f"  Preprocess:  {preprocess:.2f}ms")
    print(f"  Inference:   {inference:.2f}ms")
    print(f"  Postprocess: {postprocess:.2f}ms")
    print(f"  Total:       {total:.2f}ms")

avg_seg_time = total_seg_time / len(seg_results)
print(f"\n📈 Segmentation Average: {avg_seg_time:.2f}ms per image")
print(f"📈 Segmentation FPS: {1000/avg_seg_time:.2f} frames/sec")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
time_diff = avg_seg_time - avg_det_time
percent_diff = (time_diff / avg_det_time) * 100

print(f"\nSegmentation is {percent_diff:.1f}% slower than detection")
print(f"Time difference: {time_diff:.2f}ms per image")

print("\n" + "=" * 80)
print("✓ COMPLETE - Check the following folders for results:")
print("  • runs/detect/validation_metrics/")
print("  • runs/segment/validation_metrics/")
print("=" * 80)
