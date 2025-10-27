# Box vs Mask Plots - Complete Explanation

## 🎯 Overview

When using **segmentation models** (like YOLOv8n-seg), YOLO generates **TWO sets** of performance plots:

1. **Box plots** - Evaluate bounding box detection
2. **Mask plots** - Evaluate pixel-level segmentation

---

## 📊 The Files You Have

### From `runs/segment/val/`:

**Box Plots (4 files):**
- `BoxF1_curve.png` - F1 score for bounding boxes
- `BoxPR_curve.png` - Precision-Recall for boxes
- `BoxP_curve.png` - Precision curve for boxes
- `BoxR_curve.png` - Recall curve for boxes

**Mask Plots (4 files):**
- `MaskF1_curve.png` - F1 score for segmentation masks
- `MaskPR_curve.png` - Precision-Recall for masks
- `MaskP_curve.png` - Precision curve for masks
- `MaskR_curve.png` - Recall curve for masks

---

## 🔍 What's the Difference?

### Box Plots (Bounding Box Evaluation)

**What they measure:**
```
┌─────────────────┐
│                 │
│   🐕 DOG       │  ← Bounding Box
│                 │
└─────────────────┘
```

**Evaluation criteria:**
- Is there a box around the object? ✅
- Is the box in the right location?
- How much overlap with ground truth? (IoU)

**IoU Calculation (Box):**
```
Box IoU = Area of overlap / Area of union

Good box: IoU > 0.5 (box covers 50%+ of object)
```

**Example:**
```
Ground Truth Box: [x=10, y=10, w=50, h=50]
Predicted Box:    [x=12, y=12, w=48, h=48]
Box IoU = 0.82 ← Good detection!
```

---

### Mask Plots (Pixel-Level Evaluation)

**What they measure:**
```
┌─────────────────┐
│  🟦🟦           │
│🟦🟦🟦🟦🟦       │  ← Pixel-level Mask
│  🟦🟦🟦🟦       │     (exact object shape)
│    🟦🟦         │
└─────────────────┘
```

**Evaluation criteria:**
- Which pixels belong to the object?
- Pixel-by-pixel accuracy
- Exact shape matching

**IoU Calculation (Mask):**
```
Mask IoU = Overlapping pixels / Total pixels

Good mask: IoU > 0.5 (50%+ pixel overlap)
```

**Example:**
```
Ground Truth Pixels: 1000 pixels (dog shape)
Predicted Pixels:    950 pixels
Overlap:             800 pixels

Mask IoU = 800 / (1000 + 950 - 800) = 0.696
```

---

## 📈 Key Differences

| Aspect | Box Plots | Mask Plots |
|--------|-----------|------------|
| **What** | Bounding box rectangles | Pixel-level shapes |
| **Precision** | Coarse (rectangle) | Fine (exact shape) |
| **Difficulty** | Easier | Harder |
| **IoU Type** | Box overlap | Pixel overlap |
| **Typical mAP** | Higher (easier task) | Lower (harder task) |
| **Use Case** | Object counting, location | Precise boundaries, segmentation |

---

## 🎓 Your Results

### From COCO8 Segmentation Validation:

**Box Metrics:**
```
Box mAP50:   45.54%
Box mAP:     28.96%
```

**Mask Metrics:**
```
Mask mAP50:  32.04%
Mask mAP:    22.70%
```

### Why Mask mAP is Lower:

**Box task (easier):**
- Just needs a rectangle around object
- Allows some slack (as long as IoU > 0.5)
- Rectangle can be slightly off and still count

**Mask task (harder):**
- Needs exact pixel boundaries
- Every pixel must be correct
- Much more sensitive to small errors
- Background vs foreground decision per pixel

---

## 📊 What Each Plot Shows

### 1. F1 Curves (BoxF1 vs MaskF1)

**BoxF1_curve.png:**
- Shows F1 score for **bounding box** detection
- X-axis: Confidence threshold
- Y-axis: F1 score (box accuracy)
- Peak = optimal threshold for boxes

**MaskF1_curve.png:**
- Shows F1 score for **segmentation mask** accuracy
- X-axis: Confidence threshold
- Y-axis: F1 score (mask accuracy)
- Peak = optimal threshold for masks

**Comparison:**
```
Box F1:  Usually higher (easier task)
Mask F1: Usually lower (harder task)
```

---

### 2. PR Curves (BoxPR vs MaskPR)

**BoxPR_curve.png:**
- Precision-Recall for **box** detection
- Area under curve = Box mAP
- Shows box detection accuracy vs coverage

**MaskPR_curve.png:**
- Precision-Recall for **mask** segmentation
- Area under curve = Mask mAP
- Shows mask accuracy vs coverage

**Typical Behavior:**
```
Box PR curve:   Closer to top-right corner
Mask PR curve:  Lower and left (harder task)
```

---

### 3. Precision Curves (BoxP vs MaskP)

**BoxP_curve.png:**
- How box precision changes with confidence
- Higher confidence → fewer false positive boxes

**MaskP_curve.png:**
- How mask precision changes with confidence
- Higher confidence → fewer false positive pixels

---

### 4. Recall Curves (BoxR vs MaskR)

**BoxR_curve.png:**
- How box recall changes with confidence
- Lower confidence → find more objects (boxes)

**MaskR_curve.png:**
- How mask recall changes with confidence
- Lower confidence → capture more pixels

---

## 🔬 Detailed Example

### Scenario: Detecting a Dog

**Ground Truth:**
```
┌─────────────────┐
│  Ground Truth   │
│                 │
│   🟦🟦🟦       │  ← Exact dog pixels
│ 🟦🟦🟦🟦🟦     │     (irregular shape)
│   🟦🟦🟦🟦     │
│     🟦🟦       │
└─────────────────┘
```

**Predicted (Box Task):**
```
┌─────────────────┐
│  ┌───────────┐  │
│  │ 🟦🟦🟦    │  │  ← Rectangle around dog
│  │🟦🟦🟦🟦🟦  │  │     (includes background)
│  │ 🟦🟦🟦🟦   │  │
│  │   🟦🟦     │  │
│  └───────────┘  │
└─────────────────┘

Box IoU = 0.75 (Good! Counts as True Positive)
```

**Predicted (Mask Task):**
```
┌─────────────────┐
│                 │
│   🟩🟩🟩       │  ← Predicted pixels
│ 🟩🟩🟩🟩🟩     │     (tries to match exact shape)
│   🟩🟩  🟩     │     (missing some pixels)
│       🟩       │
└─────────────────┘

Mask IoU = 0.65 (Okay, but not perfect)
```

**Results:**
- Box: ✅ True Positive (IoU 0.75 > 0.5)
- Mask: ✅ True Positive (IoU 0.65 > 0.5)
- **But** mask has lower IoU (harder task)

---

## 💡 Why Both Matter

### Box Metrics Answer:
- "Did we **find** the object?"
- "Is the **location** correct?"
- "How many objects did we **detect**?"

### Mask Metrics Answer:
- "How **precisely** did we segment?"
- "Are the **boundaries** accurate?"
- "Can we **extract** the object cleanly?"

---

## 🎯 Practical Applications

### When Box Metrics Are Enough:
- Object counting (how many cars?)
- Object tracking (follow the person)
- General detection (is there a dog?)
- Bounding box annotations

### When Mask Metrics Matter:
- Image editing (remove background)
- Medical imaging (tumor boundaries)
- Robotics (grasp planning)
- AR/VR (realistic overlays)
- Precise measurements

---

## 📊 Interpreting Your Plots

### Compare Box vs Mask:

**If Box mAP >> Mask mAP:**
- Model good at finding objects ✅
- Struggles with exact boundaries ⚠️
- **Your case:** Box 45.54% vs Mask 32.04%

**If both are low:**
- Model struggles at detection fundamentally
- Need more training

**If both are high:**
- Excellent model!
- Good at both tasks

---

## 🔢 Your Results Explained

### From COCO8-Seg:

```
Box mAP50:   45.54%  ← Box detection accuracy
Mask mAP50:  32.04%  ← Mask segmentation accuracy

Difference:  13.5 percentage points
```

**This is NORMAL and expected:**

1. **Box task easier:**
   - Just draw rectangles
   - More forgiving (can be approximate)

2. **Mask task harder:**
   - Pixel-perfect boundaries
   - Every pixel matters
   - Background separation critical

3. **Performance gap typical:**
   - Usually 10-20% difference
   - Larger gap = model better at detection than segmentation
   - Smaller gap = model good at both

---

## 🎓 For Your Assignment

### Key Points to Document:

**1. Dual Evaluation:**
```
Segmentation models evaluated on TWO tasks:
- Box detection (bounding boxes)
- Mask segmentation (pixel-level)
```

**2. Box Plots:**
```
- Evaluate rectangular bounding boxes
- Coarser evaluation (easier)
- mAP: 45.54% @ IoU=0.5
```

**3. Mask Plots:**
```
- Evaluate pixel-level masks
- Finer evaluation (harder)
- mAP: 32.04% @ IoU=0.5
```

**4. Why Different:**
```
- Box: Rectangle around object (approximate)
- Mask: Exact pixel boundaries (precise)
- Mask inherently harder → lower scores
```

**5. Both Important:**
```
- Box metrics: Object detection quality
- Mask metrics: Segmentation precision
- Together: Complete model evaluation
```

---

## 📈 Visualization Concept

```
┌──────────────────────────────────┐
│                                  │
│  BOX EVALUATION (BoxF1, BoxPR)  │
│  ┌─────────────┐                │
│  │ ┏━━━━━━━┓   │                │
│  │ ┃🐕 DOG  ┃   │ ← Just the box │
│  │ ┗━━━━━━━┛   │                │
│  └─────────────┘                │
│                                  │
│  MASK EVALUATION (MaskF1, MaskPR)│
│  ┌─────────────┐                │
│  │   🟦🟦🟦    │                │
│  │ 🟦🟦🟦🟦🟦  │ ← Exact pixels │
│  │   🟦🟦🟦🟦  │                │
│  └─────────────┘                │
│                                  │
└──────────────────────────────────┘
```

---

## ✅ Summary

**Box Plots** = Evaluate bounding box detection (rectangles)
- BoxF1_curve.png, BoxPR_curve.png, BoxP_curve.png, BoxR_curve.png
- Easier task, higher mAP (45.54%)
- Good for: object counting, location

**Mask Plots** = Evaluate pixel-level segmentation (exact shapes)
- MaskF1_curve.png, MaskPR_curve.png, MaskP_curve.png, MaskR_curve.png
- Harder task, lower mAP (32.04%)
- Good for: precise boundaries, extraction

**Why Both?**
- Complete model evaluation
- Box = "Did we find it?"
- Mask = "How precisely did we outline it?"

**Your Results:**
- Box mAP > Mask mAP ✅ (This is normal!)
- Shows model better at detection than pixel-perfect segmentation
- Both together give complete performance picture

Perfect for your documentation! 🎯
