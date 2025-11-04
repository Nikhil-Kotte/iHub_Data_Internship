from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # n = nano (smallest, fastest)

results = model('https://ultralytics.com/images/bus.jpg')

for result in results:
    result.save(filename='task-1/detection_result.jpg')

print("Object detection completed.")

# segmentation with YOLOv8 segmentation model
seg_model = YOLO('yolov8n-seg.pt')
seg_results = seg_model('https://ultralytics.com/images/bus.jpg')

for result in seg_results:
    result.save(filename='task-1/segmentation_result.jpg')

print("Segmentation completed. Results saved to: task-1/segmentation_result.jpg")
