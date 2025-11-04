# Box vs Mask Plots - Explanation

Segmentation models produce box-based and mask-based evaluation plots.

Box plots evaluate bounding box detection (IoU on rectangles).
Mask plots evaluate pixel-level segmentation (IoU on masks).

Typical workflow:

```python
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
metrics = model.val(data='coco8-seg.yaml', plots=True)
```

Compare box and mask mAP to assess detection vs segmentation performance.
