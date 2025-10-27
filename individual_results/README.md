# Individual Detection and Segmentation Results

## Overview

This folder contains individual annotated images for each of the 10 sample images.

**Total files:** 20 images (10 detection + 10 segmentation)

---

## Folder Structure

```
individual_results/
  detection/
    detection_1.jpg
    detection_2.jpg
    ... (10 files total)
  
  segmentation/
    segmentation_1.jpg
    segmentation_2.jpg
    ... (10 files total)
  
  README.md (this file)
```

---

## Detection Results

**Location:** `detection/`

- Shows bounding boxes around detected objects
- Each box labeled with class name and confidence score
- Model: YOLOv8n (detection)
- File naming: `detection_<number>.jpg`

---

## Segmentation Results

**Location:** `segmentation/`

- Shows pixel-level segmentation masks
- Colored masks overlay exact object boundaries
- Also includes bounding boxes
- Model: YOLOv8n-seg (segmentation)
- File naming: `segmentation_<number>.jpg`

---

## Image Details

| Image | Detection | Segmentation | Notes |
|-------|-----------|--------------|-------|
| 1.jpg | 1 cat | 1 bear | Different classification |
| 2.jpg | 1 dog | 1 dog | Same |
| 3.jpg | 3 chairs | 4 chairs | Seg found 1 more |
| 4.jpg | 5 (2 person, 3 ball) | 4 (2 person, 2 ball) | Different |
| 5.jpg | 7 (5 person, 2 kite) | 8 (5 person, 1 ball, 2 kite) | Seg found ball |
| 6.jpg | 1 bird | 1 bird | Same |
| 7.jpg | 4 (2 dog, 2 person) | 2 (1 dog, 1 person) | Det found more |
| 8.jpg | 24 objects | 25 objects | Complex scene |
| 9.jpg | 2 person | 2 (1 person, 1 tv) | Seg found TV |
| 10.jpg | 4 objects | 7 objects | Seg found more |

---

## Usage

These images can be used for:

- Visual comparison of detection vs segmentation
- Documentation and reports
- Presentations
- Side-by-side analysis

---

## Generation

Generated using: `generate_individual_results.py`

Date: October 27, 2025
