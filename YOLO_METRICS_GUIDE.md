# YOLO Built-in Performance Metrics Guide

## Overview
YOLO (Ultralytics) has powerful built-in capabilities for performance evaluation and metrics visualization. This guide demonstrates how to use these native features.

---

## Method 1: Using Python API

### Basic Prediction with Metrics

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Predict with automatic metrics
results = model.predict(
    source='Sample Images',
    save=True,
    project='runs/detect',
    name='experiment',
    conf=0.25,
    verbose=True
)

# Access metrics from results
for result in results:
    speed = result.speed  # Dictionary with timing info
    print(f"Preprocess: {speed['preprocess']:.2f}ms")
    print(f"Inference: {speed['inference']:.2f}ms")
    print(f"Postprocess: {speed['postprocess']:.2f}ms")
```

### Key Metrics Available

**From `result.speed` dictionary:**
- `preprocess`: Image preprocessing time (ms)
- `inference`: Model inference time (ms)
- `postprocess`: Post-processing time (ms)

**From `result` object:**
- `boxes`: Detected bounding boxes
- `masks`: Segmentation masks (if segmentation model)
- `probs`: Class probabilities
- `keypoints`: Pose keypoints (if pose model)

---

## Method 2: Using YOLO CLI

### Command Line Prediction

```bash
# Detection
yolo detect predict model=yolov8n.pt source="Sample Images" save=true

# Segmentation
yolo segment predict model=yolov8n-seg.pt source="Sample Images" save=true

# With custom parameters
yolo detect predict model=yolov8n.pt source="Sample Images" conf=0.25 iou=0.45 save=true
```

### CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Path to model | Required |
| `source` | Image/video/folder | Required |
| `conf` | Confidence threshold | 0.25 |
| `iou` | IoU threshold for NMS | 0.45 |
| `save` | Save results | False |
| `project` | Project folder | runs |
| `name` | Experiment name | predict |
| `imgsz` | Image size | 640 |

---

## Method 3: Validation on Dataset (Training Metrics)

### For Pre-trained Models

If you want comprehensive metrics like mAP, precision, recall, you need to validate on a proper dataset:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Validate on COCO dataset (or custom dataset)
metrics = model.val(data='coco8.yaml')

# Access validation metrics
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

### CLI Validation

```bash
yolo detect val model=yolov8n.pt data=coco8.yaml
```

**This generates:**
- Confusion matrix
- Precision-Recall curves
- F1-Confidence curves
- mAP metrics
- Saved plots in `runs/detect/val/`

---

## Method 4: Training with Automatic Metrics

### Train and Generate Plots

```python
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')  # or load pre-trained

# Train - automatically generates comprehensive metrics
results = model.train(
    data='coco8.yaml',
    epochs=10,
    imgsz=640,
    plots=True
)
```

**Automatically generated plots:**
- `results.png` - Training curves (loss, mAP, etc.)
- `confusion_matrix.png` - Confusion matrix
- `P_curve.png` - Precision curve
- `R_curve.png` - Recall curve
- `F1_curve.png` - F1 score curve
- `PR_curve.png` - Precision-Recall curve

All saved in `runs/detect/train/`

---

## Our Implementation Results

### Metrics from `yolo_validation_metrics.py`

**Detection Model (YOLOv8n):**
- Average total time: 207.88ms per image
- Preprocess: 3.4ms avg
- Inference: 201.4ms avg
- Postprocess: 3.0ms avg
- **Throughput: 4.81 FPS**

**Segmentation Model (YOLOv8n-seg):**
- Average total time: 273.54ms per image
- Preprocess: 4.8ms avg
- Inference: 258.9ms avg
- Postprocess: 9.9ms avg
- **Throughput: 3.66 FPS**

**Comparison:**
- Segmentation is **31.6% slower** than detection
- Time difference: **65.66ms** per image

---

## Understanding YOLO's Metrics Pipeline

### Timing Breakdown

```
Total Time = Preprocess + Inference + Postprocess
```

1. **Preprocess** (2-6ms)
   - Image resizing
   - Normalization
   - Padding
   - Tensor conversion

2. **Inference** (150-300ms on CPU)
   - Forward pass through neural network
   - Feature extraction
   - Detection/segmentation head computation

3. **Postprocess** (1-24ms)
   - Non-Maximum Suppression (NMS)
   - Coordinate scaling
   - Mask generation (segmentation only)

---

## Speed vs Accuracy Trade-offs

### Model Size Variants

| Model | Size | Speed | mAP |
|-------|------|-------|-----|
| YOLOv8n | 3.2M | Fastest | 37.3 |
| YOLOv8s | 11.2M | Fast | 44.9 |
| YOLOv8m | 25.9M | Medium | 50.2 |
| YOLOv8l | 43.7M | Slow | 52.9 |
| YOLOv8x | 68.2M | Slowest | 53.9 |

Choose based on your requirements:
- **Real-time video:** Use YOLOv8n or YOLOv8s
- **Accuracy critical:** Use YOLOv8l or YOLOv8x
- **Balanced:** Use YOLOv8m

---

## Generating Visual Plots (Requires Training Data)

To get plots like confusion matrices, precision-recall curves, you need to:

### Option 1: Train on Custom Data

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='custom_data.yaml',
    epochs=10,
    plots=True
)
```

### Option 2: Validate on Standard Dataset

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
metrics = model.val(
    data='coco8.yaml',  # Use COCO dataset
    plots=True
)
```

This generates:
- `confusion_matrix.png`
- `F1_curve.png`
- `P_curve.png`
- `R_curve.png`
- `PR_curve.png`

---

## Practical Usage Examples

### Example 1: Benchmark Different Models

```python
from ultralytics import YOLO
import time

models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
source = 'Sample Images'

for model_name in models:
    model = YOLO(model_name)
    start = time.time()
    results = model.predict(source, verbose=False)
    elapsed = time.time() - start
    
    print(f"{model_name}: {elapsed:.2f}s for 10 images")
    print(f"  FPS: {10/elapsed:.2f}")
```

### Example 2: Export Metrics to JSON

```python
from ultralytics import YOLO
import json

model = YOLO('yolov8n.pt')
results = model.predict('Sample Images', verbose=False)

metrics = []
for result in results:
    metrics.append({
        'image': result.path,
        'objects': len(result.boxes),
        'speed': result.speed,
        'classes': [model.names[int(c)] for c in result.boxes.cls]
    })

with open('yolo_metrics.json', 'w') as f:
    json.dump(metrics, indent=2)
```

### Example 3: Live Video Metrics

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# Process video with FPS counter
results = model.predict(
    source=0,  # Webcam
    show=True,  # Display
    stream=True  # Generator for live processing
)

for result in results:
    fps = 1000 / result.speed['inference']
    print(f"Live FPS: {fps:.2f}")
```

---

## Comparison: Manual vs Built-in Metrics

### Our Manual Implementation
✅ Custom metrics collection  
✅ Flexible analysis  
✅ Educational value  
❌ More code to maintain  

### YOLO Built-in
✅ Zero additional code  
✅ Standardized metrics  
✅ Automatic plots (with training)  
❌ Less customization  

**Best Practice:** Use YOLO's built-in metrics for quick evaluation, create custom analysis for specific insights.

---

## Output Folder Structure

When using YOLO's built-in features:

```
runs/
├── detect/
│   ├── predict/          # Default predictions
│   ├── validation_metrics/  # Our named experiment
│   ├── cli_metrics/      # CLI results
│   ├── train/            # Training results
│   │   ├── weights/
│   │   ├── results.png
│   │   ├── confusion_matrix.png
│   │   └── *.csv
│   └── val/              # Validation results
│       ├── confusion_matrix.png
│       └── *.png
└── segment/
    └── [same structure]
```

---

## Summary

### Key Takeaways

1. **Built-in Metrics:** YOLO automatically tracks preprocessing, inference, and postprocessing times
2. **CLI Power:** Single command to process images with metrics: `yolo detect predict model=yolov8n.pt source="images"`
3. **Validation Plots:** Use `model.val()` on a dataset to get mAP, confusion matrices, and curves
4. **Training Plots:** Training automatically generates comprehensive performance visualizations
5. **Python API:** Access all metrics programmatically via `result.speed`, `result.boxes`, etc.

### For Your Assignment

You can enhance your submission by:
- ✅ Using `yolo_validation_metrics.py` to show native YOLO metrics
- ✅ Documenting the automatic timing breakdown (preprocess/inference/postprocess)
- ✅ Comparing manual metrics vs YOLO built-in metrics
- ✅ Showing CLI usage for quick evaluations

---

**Resources:**
- [Ultralytics Docs - Predict Mode](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics Docs - Val Mode](https://docs.ultralytics.com/modes/val/)
- [Ultralytics Docs - Train Mode](https://docs.ultralytics.com/modes/train/)
