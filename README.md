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

## Next Steps
- Assignment 2: Process multiple images in a single program
- Evaluate and document performance metrics from `runs/detect/train` folder

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
