from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # n = nano (smallest, fastest)

# Test with an online image
results = model('https://ultralytics.com/images/bus.jpg')

# Display results
for result in results:
    result.show()  # Display the image with detections
    result.save(filename='detection_result.jpg')  # Save the result

print("✓ Object detection completed!")
print(f"✓ Results saved to: detection_result.jpg")

# Test segmentation with YOLOv8 segmentation model
seg_model = YOLO('yolov8n-seg.pt')  # segmentation model

# Run segmentation
seg_results = seg_model('https://ultralytics.com/images/bus.jpg')

# Display and save segmentation results
for result in seg_results:
    result.show()
    result.save(filename='segmentation_result.jpg')

print("✓ Segmentation completed!")
print(f"✓ Results saved to: segmentation_result.jpg")
