"""
Task 2: Multi-Image Object Detection and Segmentation
Process multiple images and collect performance metrics
"""

from ultralytics import YOLO
import os
import glob
import time
import json
from pathlib import Path

# Create output directory
output_dir = Path("task-2/results_assignment2")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize models
detection_model = YOLO('yolov8x.pt')
segmentation_model = YOLO('yolov8x-seg.pt')

# Get all images from images folder
image_folder = "images"
image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
image_files.extend(glob.glob(os.path.join(image_folder, "*.jpeg")))
image_files.sort()

# Storage for metrics
all_metrics = {'detection': [], 'segmentation': [], 'summary': {}}

for img_path in image_files:
    img_name = os.path.basename(img_path)

    # Detection
    start_time = time.time()
    results = detection_model(img_path, verbose=False)
    inference_time = time.time() - start_time
    result = results[0]
    boxes = result.boxes

    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    img_metrics = {
        'image': img_name,
        'total_objects': len(boxes) if boxes is not None else 0,
        'detected_classes': class_counts,
        'inference_time_sec': round(inference_time, 4)
    }
    all_metrics['detection'].append(img_metrics)

    # Save detection annotated result
    save_path = output_dir / f"detection_{img_name}"
    result.save(filename=str(save_path))

# Segmentation pass
for img_path in image_files:
    img_name = os.path.basename(img_path)
    start_time = time.time()
    results = segmentation_model(img_path, verbose=False)
    inference_time = time.time() - start_time
    result = results[0]
    boxes = result.boxes

    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = segmentation_model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    has_masks = hasattr(result, 'masks') and result.masks is not None
    num_masks = len(result.masks.data) if has_masks else 0

    img_metrics = {
        'image': img_name,
        'total_objects': len(boxes) if boxes is not None else 0,
        'detected_classes': class_counts,
        'num_masks': num_masks,
        'inference_time_sec': round(inference_time, 4)
    }
    all_metrics['segmentation'].append(img_metrics)

    save_path = output_dir / f"segmentation_{img_name}"
    result.save(filename=str(save_path))

# Save metrics
metrics_file = output_dir / "metrics.json"
with open(metrics_file, 'w') as f:
    json.dump(all_metrics, f, indent=2)

print(f"Results and metrics saved to: {output_dir}")
