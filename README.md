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
```

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
