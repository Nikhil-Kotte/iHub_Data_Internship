"""
Assignment 2: Multi-Image Object Detection and Segmentation
Process multiple images and analyze performance metrics
"""

from ultralytics import YOLO
import os
import glob
import time
import json
from pathlib import Path

# Create output directory
output_dir = Path("results_assignment2")
output_dir.mkdir(exist_ok=True)

# Initialize models
print("=" * 60)
print("Loading YOLO Models...")
print("=" * 60)

detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8n-seg.pt')

# Get all images from Sample Images folder
image_folder = "Sample Images"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
image_files.sort()  # Sort by name

print(f"\nFound {len(image_files)} images to process\n")

# Storage for metrics
all_metrics = {
    'detection': [],
    'segmentation': [],
    'summary': {}
}

print("=" * 60)
print("OBJECT DETECTION - Processing Multiple Images")
print("=" * 60)

detection_times = []
for img_path in image_files:
    img_name = os.path.basename(img_path)
    print(f"\nProcessing: {img_name}")
    
    # Run detection
    start_time = time.time()
    results = detection_model(img_path, verbose=False)
    inference_time = time.time() - start_time
    detection_times.append(inference_time)
    
    # Extract metrics
    result = results[0]
    boxes = result.boxes
    
    # Count detections by class
    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Store metrics
    img_metrics = {
        'image': img_name,
        'total_objects': len(boxes) if boxes is not None else 0,
        'detected_classes': class_counts,
        'inference_time_sec': round(inference_time, 4),
        'speed_ms': round(result.speed['inference'], 2) if hasattr(result, 'speed') else 0
    }
    all_metrics['detection'].append(img_metrics)
    
    # Print results
    print(f"  Objects detected: {img_metrics['total_objects']}")
    print(f"  Classes: {class_counts}")
    print(f"  Inference time: {img_metrics['inference_time_sec']:.4f}s")
    
    # Save result
    save_path = output_dir / f"detection_{img_name}"
    result.save(filename=str(save_path))

print("\n" + "=" * 60)
print("SEGMENTATION - Processing Multiple Images")
print("=" * 60)

segmentation_times = []
for img_path in image_files:
    img_name = os.path.basename(img_path)
    print(f"\nProcessing: {img_name}")
    
    # Run segmentation
    start_time = time.time()
    results = segmentation_model(img_path, verbose=False)
    inference_time = time.time() - start_time
    segmentation_times.append(inference_time)
    
    # Extract metrics
    result = results[0]
    boxes = result.boxes
    
    # Count detections by class
    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = segmentation_model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Check if masks are available
    has_masks = hasattr(result, 'masks') and result.masks is not None
    num_masks = len(result.masks.data) if has_masks else 0
    
    # Store metrics
    img_metrics = {
        'image': img_name,
        'total_objects': len(boxes) if boxes is not None else 0,
        'detected_classes': class_counts,
        'num_masks': num_masks,
        'inference_time_sec': round(inference_time, 4),
        'speed_ms': round(result.speed['inference'], 2) if hasattr(result, 'speed') else 0
    }
    all_metrics['segmentation'].append(img_metrics)
    
    # Print results
    print(f"  Objects detected: {img_metrics['total_objects']}")
    print(f"  Classes: {class_counts}")
    print(f"  Masks generated: {num_masks}")
    print(f"  Inference time: {img_metrics['inference_time_sec']:.4f}s")
    
    # Save result
    save_path = output_dir / f"segmentation_{img_name}"
    result.save(filename=str(save_path))

# Calculate summary statistics
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)

# Detection summary
det_avg_time = sum(detection_times) / len(detection_times)
det_total_objects = sum(m['total_objects'] for m in all_metrics['detection'])

# Segmentation summary
seg_avg_time = sum(segmentation_times) / len(segmentation_times)
seg_total_objects = sum(m['total_objects'] for m in all_metrics['segmentation'])
seg_total_masks = sum(m['num_masks'] for m in all_metrics['segmentation'])

all_metrics['summary'] = {
    'total_images_processed': len(image_files),
    'detection': {
        'avg_inference_time_sec': round(det_avg_time, 4),
        'total_time_sec': round(sum(detection_times), 4),
        'total_objects_detected': det_total_objects,
        'avg_objects_per_image': round(det_total_objects / len(image_files), 2)
    },
    'segmentation': {
        'avg_inference_time_sec': round(seg_avg_time, 4),
        'total_time_sec': round(sum(segmentation_times), 4),
        'total_objects_detected': seg_total_objects,
        'total_masks_generated': seg_total_masks,
        'avg_objects_per_image': round(seg_total_objects / len(image_files), 2)
    }
}

# Print summary
print(f"\nImages processed: {len(image_files)}")
print(f"\nDetection Model:")
print(f"  Average inference time: {det_avg_time:.4f}s")
print(f"  Total objects detected: {det_total_objects}")
print(f"  Average objects per image: {all_metrics['summary']['detection']['avg_objects_per_image']}")

print(f"\nSegmentation Model:")
print(f"  Average inference time: {seg_avg_time:.4f}s")
print(f"  Total objects detected: {seg_total_objects}")
print(f"  Total masks generated: {seg_total_masks}")
print(f"  Average objects per image: {all_metrics['summary']['segmentation']['avg_objects_per_image']}")

# Save metrics to JSON
metrics_file = output_dir / "metrics.json"
with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f"\n✓ All results saved to: {output_dir}/")
print(f"✓ Metrics saved to: {metrics_file}")
print("=" * 60)
