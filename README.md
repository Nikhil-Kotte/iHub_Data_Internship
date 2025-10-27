# iHub Data Internship 2025-26
## Vision Domain - Object Detection and Segmentation with YOLOv8

### Project Overview
This repository documents my work in the Vision domain internship program, focusing on object detection and segmentation using pre-trained YOLO models from Ultralytics.

---

## Assignment 1: Installation and Initial Testing

**Date:** October 27, 2025  
**Status:** ✅ Completed

### Setup Process

#### 1. Environment Setup
- **Platform:** Windows  
- **Python Version:** 3.14.0  
- **Virtual Environment:** Created using `python -m venv venv`

#### 2. Installation Steps
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install numpy (to avoid compilation issues)
pip install numpy

# Install ultralytics with pre-built binaries
pip install ultralytics --only-binary=:all:
```

#### 3. Verification
```bash
yolo version
# Output: 8.3.221
```

### Testing Results

#### Object Detection Test
- **Model:** YOLOv8n (nano - smallest, fastest)
- **Test Image:** https://ultralytics.com/images/bus.jpg
- **Detected Objects:**
  - 4 persons
  - 1 bus
  - 1 stop sign
- **Performance:**
  - Preprocess: 2.7ms
  - Inference: 71.1ms
  - Postprocess: 1.8ms

#### Segmentation Test
- **Model:** YOLOv8n-seg (segmentation model)
- **Test Image:** Same bus image
- **Detected & Segmented Objects:**
  - 4 persons
  - 1 bus
  - 1 skateboard
- **Performance:**
  - Preprocess: 1.7ms
  - Inference: 86.2ms
  - Postprocess: 6.3ms

### Files Generated
- `test_yolo.py` - Test script for detection and segmentation
- `detection_result.jpg` - Object detection output
- `segmentation_result.jpg` - Segmentation output
- `yolov8n.pt` - Pre-trained detection model (6.2 MB)
- `yolov8n-seg.pt` - Pre-trained segmentation model (6.7 MB)

### Key Observations
1. ✅ Successfully installed ultralytics package
2. ✅ Both detection and segmentation models work correctly
3. ✅ Models can process online images
4. ✅ Real-time inference is fast enough for practical applications
5. The segmentation model detected an additional object (skateboard) compared to the detection model

---

## Assignment 2: Multi-Image Processing and Performance Metrics

**Date:** October 27, 2025  
**Status:** ✅ Completed

### Overview
Processed 10 sample images using both detection and segmentation models, collected detailed performance metrics, and analyzed the results.

### Implementation

#### Scripts Created
1. **`multi_image_detection.py`** - Batch processes multiple images
   - Loads all images from `Sample Images` folder
   - Runs both detection and segmentation
   - Saves annotated results to `results_assignment2/`
   - Generates JSON metrics file

2. **`analyze_metrics.py`** - Analyzes collected metrics
   - Calculates summary statistics
   - Compares model performance
   - Identifies patterns and insights

### Results Summary

#### Overall Statistics
- **Total Images Processed:** 10
- **Total Objects Detected (Detection):** 52 objects
- **Total Objects Detected (Segmentation):** 55 objects + 55 masks
- **Unique Classes (Detection):** 13 classes
- **Unique Classes (Segmentation):** 14 classes

#### Performance Metrics

##### Detection Model (YOLOv8n)
| Metric | Value |
|--------|-------|
| Average inference time | 0.0598s |
| Total processing time | 0.5978s |
| Throughput | 16.73 images/sec |
| Average objects per image | 5.2 |

##### Segmentation Model (YOLOv8n-seg)
| Metric | Value |
|--------|-------|
| Average inference time | 0.0761s |
| Total processing time | 0.7614s |
| Throughput | 13.13 images/sec |
| Average objects per image | 5.5 |
| Average masks per image | 5.5 |

### Performance Analysis

#### 1. Speed Comparison
- **Detection is 27.3% faster** than segmentation
- Segmentation takes 0.0163s more per image
- **Reason:** Segmentation generates pixel-level masks, requiring more computational resources

#### 2. Detection Accuracy
- Segmentation model found **5.8% more objects** (3 additional objects)
- Different models may have:
  - Different confidence thresholds
  - Varying sensitivity to object boundaries
  - Architectural differences affecting detection

#### 3. Class Distribution (Top 5)

**Detection Model:**
1. Person: 23 instances
2. Car: 7 instances
3. Traffic light: 5 instances
4. Dog: 3 instances
5. Chair: 3 instances

**Segmentation Model:**
1. Person: 24 instances
2. Car: 6 instances
3. Book: 4 instances
4. Chair: 4 instances
5. Traffic light: 4 instances

#### 4. Interesting Observations

**Most Complex Image (8.jpg - Street Scene):**
- Detection: 24 objects (12 persons, 7 cars, 5 traffic lights)
- Segmentation: 25 objects (15 persons, 6 cars, 4 traffic lights)
- Both models handled complex multi-object scenes well

**Processing Time Variance:**
- Fastest: 10.jpg (0.0417s) - 4 objects
- Slowest: 1.jpg (0.1256s) - 1 object
- **Note:** First image is slower due to model initialization/warm-up

**Model Detection Differences:**
- Classes only in Detection: `cat`, `dining table`
- Classes only in Segmentation: `bear`, `cell phone`, `tv`
- Shows models can have different classification behaviors

### Key Insights

1. **✅ Batch Processing Efficient:** Both models can process 10+ images/second on CPU
2. **✅ Consistent Performance:** After warm-up, inference time is stable (40-70ms)
3. **✅ Real-time Capable:** Detection model achieves ~17 fps, suitable for video
4. **✅ Trade-offs Clear:** Segmentation provides more detail (+masks) at 27% time cost
5. **✅ Model Reliability:** Both models detected objects across diverse image types

### Metrics Explanation

#### Understanding the Metrics

**Inference Time:**
- Time taken by model to process one image
- Excludes file I/O and pre/post-processing overhead
- Critical metric for real-time applications

**Throughput (images/sec):**
- Total images divided by total processing time
- Real-world metric including all overhead
- Better indicator of production performance

**Objects per Image:**
- Average number of detected objects
- Indicates scene complexity
- Higher values = more complex scenes

**Masks (Segmentation only):**
- Pixel-level object boundaries
- One mask per detected object
- Enables precise object extraction

### Files Generated
- `results_assignment2/` - Folder with 20 annotated images (10 detection + 10 segmentation)
- `results_assignment2/metrics.json` - Complete metrics data in JSON format
- `multi_image_detection.py` - Batch processing script
- `analyze_metrics.py` - Metrics analysis script

### Usage
```bash
# Process multiple images
python multi_image_detection.py

# Analyze metrics
python analyze_metrics.py

# YOLO built-in metrics
python yolo_validation_metrics.py
```

### YOLO Built-in Metrics (Bonus)

YOLO provides native performance evaluation with detailed timing breakdown:

**Script:** `yolo_validation_metrics.py`

**Automatic Metrics Tracked:**
- ⏱️ **Preprocess time:** Image resizing, normalization (2-6ms)
- ⏱️ **Inference time:** Neural network forward pass (150-300ms CPU)
- ⏱️ **Postprocess time:** NMS, coordinate scaling, mask generation (1-24ms)

**Results from Built-in Metrics:**

| Model | Total Time | Preprocess | Inference | Postprocess | FPS |
|-------|------------|------------|-----------|-------------|-----|
| Detection | 207.88ms | 3.4ms | 201.4ms | 3.0ms | 4.81 |
| Segmentation | 273.54ms | 4.8ms | 258.9ms | 9.9ms | 3.66 |

**Key Finding:** Segmentation postprocessing takes 3.3x longer due to mask generation.

**CLI Usage:**
```bash
# Quick prediction with automatic metrics
yolo detect predict model=yolov8n.pt source="Sample Images" save=true
```

See `YOLO_METRICS_GUIDE.md` for comprehensive documentation on YOLO's built-in evaluation features.

---

## Assignment 3: Validation Plots & Advanced Metrics

**Date:** October 27, 2025  
**Status:** ✅ Completed

### Overview
Generated comprehensive validation plots (confusion matrix, F1-curve, PR-curve, etc.) using YOLO's validation mode on the COCO8 dataset.

### Why COCO8 Dataset?

Our sample images don't have **ground truth annotations** (labels). To generate validation plots like confusion matrices and precision-recall curves, we need:
- Images with known object locations
- Ground truth bounding boxes
- Class labels for each object

YOLO automatically downloads and uses COCO8 (8 annotated images from COCO dataset) for validation.

### Implementation

**Script:** `generate_validation_plots.py`

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
metrics = model.val(data='coco8.yaml', plots=True)
```

### Generated Plots

#### Detection Model (`runs/detect/val/`)
1. ✅ **Confusion Matrix** - Classification accuracy across classes
2. ✅ **F1-Confidence Curve** - Optimal confidence threshold
3. ✅ **Precision-Recall Curve** - Performance trade-off
4. ✅ **Precision Curve** - Accuracy vs confidence
5. ✅ **Recall Curve** - Detection coverage vs confidence
6. ✅ **Validation Batches** - Visual comparison (predictions vs labels)

#### Segmentation Model (`runs/segment/val/`)
All detection plots **PLUS:**
7. ✅ **Mask F1 Curve** - Segmentation F1 scores
8. ✅ **Mask PR Curve** - Mask precision-recall
9. ✅ **Mask Precision/Recall** - Mask-specific metrics

### Validation Metrics

#### Detection Model (YOLOv8n on COCO8)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **mAP50** | 73.92% | Good detection at IoU=0.5 |
| **mAP50-95** | 50.98% | Solid across IoU thresholds |
| **Precision** | 83.33% | 83% of predictions correct |
| **Recall** | 65.00% | Found 65% of all objects |
| **F1 Score** | 73.03% | Balanced performance |

**Speed:**
- Preprocess: 0.62ms
- Inference: 53.72ms
- Postprocess: 0.52ms
- **Total: 54.86ms (18.2 FPS)**

#### Segmentation Model (YOLOv8n-seg on COCO8)

| Metric | Box | Mask |
|--------|-----|------|
| **mAP50** | 45.54% | 32.04% |
| **mAP50-95** | 28.96% | 22.70% |

### Key Insights from Plots

#### 1. Confusion Matrix
- Shows which classes are misclassified
- Diagonal elements = correct predictions
- Strong per-class separation observed
- Minimal confusion between dissimilar objects

#### 2. F1-Confidence Curve
- **Optimal confidence threshold: 0.42**
- Peak F1 score: 0.73
- Trade-off between precision and recall
- Wider plateau = more robust model

#### 3. Precision-Recall Curve
- Area under curve = Average Precision
- mAP50-95 = 50.98% (good performance)
- Some classes achieve near-perfect AP
- Smaller objects more challenging

#### 4. Precision vs Recall Trade-off

**At confidence 0.25:**
- Precision: 83.33% (few false positives)
- Recall: 65.00% (misses some objects)

**Interpretation:**
- Model favors **accuracy over coverage**
- Suitable for applications where false alarms costly
- Lower confidence threshold would improve recall

### Understanding the Metrics

#### mAP (mean Average Precision)
- **mAP50:** IoU threshold = 0.5 (lenient)
- **mAP50-95:** Average across IoU 0.5 to 0.95 (COCO standard)
- Higher = better detection accuracy

#### IoU (Intersection over Union)
```
IoU = Overlap Area / Union Area
IoU > 0.5 = Good detection
IoU < 0.5 = Poor detection
```

#### Precision vs Recall
```
Precision = TP / (TP + FP)  # Accuracy of predictions
Recall = TP / (TP + FN)     # Coverage of all objects

High Precision → Few false alarms
High Recall → Find everything
```

### Comparison: Detection vs Segmentation

| Aspect | Detection | Segmentation |
|--------|-----------|-------------|
| **mAP50** | 73.92% | 45.54% (box), 32.04% (mask) |
| **Task** | Bounding boxes | Pixel-level masks |
| **Difficulty** | Easier | Harder |
| **Use Case** | Object counting | Precise boundaries |

**Why segmentation mAP is lower:**
- Pixel-level accuracy required
- More sensitive to IoU threshold
- Harder task overall

### Practical Applications

**Use these plots to:**
1. **Choose confidence threshold** - F1 curve peak
2. **Identify problem classes** - Confusion matrix
3. **Compare models** - mAP scores
4. **Debug issues** - Visual inspection of predictions

### Files Generated
- `runs/detect/val/` - 9 detection validation plots
- `runs/segment/val/` - 12 segmentation validation plots  
- `generate_validation_plots.py` - Validation script
- `VALIDATION_PLOTS_GUIDE.md` - Complete guide (415 lines)

### Usage
```bash
# Generate all validation plots
python generate_validation_plots.py

# CLI method
yolo detect val model=yolov8n.pt data=coco8.yaml plots=true
```

See `VALIDATION_PLOTS_GUIDE.md` for detailed explanations of each plot and metric.

---

## Dependencies
```
ultralytics==8.3.221
torch==2.9.0
torchvision==0.24.0
numpy==2.3.4
opencv-python==4.11.0.86
pillow==12.0.0
```

## Usage
To run the test:
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run test script
python test_yolo.py
```
