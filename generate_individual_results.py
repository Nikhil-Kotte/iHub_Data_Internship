"""
Generate Individual Detection and Segmentation Results
Creates separate annotated image files for each input image
"""

from ultralytics import YOLO
import os
from pathlib import Path

print("=" * 80)
print("GENERATING INDIVIDUAL RESULT IMAGES")
print("=" * 80)

# Create output directories
detection_output = Path("individual_results/detection")
segmentation_output = Path("individual_results/segmentation")

detection_output.mkdir(parents=True, exist_ok=True)
segmentation_output.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Created output directories:")
print(f"  • {detection_output}")
print(f"  • {segmentation_output}")

# Load models
print("\n1. Loading models...")
detection_model = YOLO('yolov8n.pt')
segmentation_model = YOLO('yolov8n-seg.pt')
print("✓ Models loaded")

# Get all sample images
image_folder = "Sample Images"
image_files = sorted(Path(image_folder).glob("*.jpg"))

print(f"\n2. Found {len(image_files)} images to process")
print("=" * 80)

# Process each image for DETECTION
print("\n📊 DETECTION RESULTS")
print("-" * 80)

for img_path in image_files:
    img_name = img_path.name
    print(f"\nProcessing: {img_name}")
    
    # Run detection
    results = detection_model(str(img_path), verbose=False)
    result = results[0]
    
    # Get detection info
    boxes = result.boxes
    num_objects = len(boxes)
    
    # Count by class
    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = detection_model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Save annotated image
    output_path = detection_output / f"detection_{img_name}"
    result.save(filename=str(output_path))
    
    # Print results
    classes_str = ', '.join([f"{v} {k}" for k, v in class_counts.items()]) or "none"
    print(f"  Objects: {num_objects}")
    print(f"  Classes: {classes_str}")
    print(f"  ✓ Saved: {output_path}")

print("\n" + "=" * 80)
print("📊 SEGMENTATION RESULTS")
print("-" * 80)

# Process each image for SEGMENTATION
for img_path in image_files:
    img_name = img_path.name
    print(f"\nProcessing: {img_name}")
    
    # Run segmentation
    results = segmentation_model(str(img_path), verbose=False)
    result = results[0]
    
    # Get segmentation info
    boxes = result.boxes
    masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
    num_objects = len(boxes)
    num_masks = len(masks.data) if masks else 0
    
    # Count by class
    class_counts = {}
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = segmentation_model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Save annotated image
    output_path = segmentation_output / f"segmentation_{img_name}"
    result.save(filename=str(output_path))
    
    # Print results
    classes_str = ', '.join([f"{v} {k}" for k, v in class_counts.items()]) or "none"
    print(f"  Objects: {num_objects}")
    print(f"  Masks: {num_masks}")
    print(f"  Classes: {classes_str}")
    print(f"  ✓ Saved: {output_path}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n✓ Processed {len(image_files)} images")
print(f"\n📁 Output locations:")
print(f"  • Detection results:    {detection_output}/")
print(f"     - detection_1.jpg")
print(f"     - detection_2.jpg")
print(f"     - ... (10 total)")
print(f"\n  • Segmentation results: {segmentation_output}/")
print(f"     - segmentation_1.jpg")
print(f"     - segmentation_2.jpg")
print(f"     - ... (10 total)")

print(f"\n✓ Total files created: {len(image_files) * 2} images")
print(f"   ({len(image_files)} detection + {len(image_files)} segmentation)")

print("\n" + "=" * 80)
print("✓ COMPLETE!")
print("=" * 80)

# Create an index file
index_file = Path("individual_results/README.md")
with open(index_file, 'w') as f:
    f.write("# Individual Detection and Segmentation Results\n\n")
    f.write(f"## Overview\n\n")
    f.write(f"This folder contains individual annotated images for each of the {len(image_files)} sample images.\n\n")
    
    f.write("## Folder Structure\n\n")
    f.write("```\n")
    f.write("individual_results/\n")
    f.write("├── detection/\n")
    f.write("│   ├── detection_1.jpg\n")
    f.write("│   ├── detection_2.jpg\n")
    f.write("│   └── ... (10 files)\n")
    f.write("├── segmentation/\n")
    f.write("│   ├── segmentation_1.jpg\n")
    f.write("│   ├── segmentation_2.jpg\n")
    f.write("│   └── ... (10 files)\n")
    f.write("└── README.md (this file)\n")
    f.write("```\n\n")
    
    f.write("## Detection Results\n\n")
    f.write("Location: `detection/`\n\n")
    f.write("- Shows bounding boxes around detected objects\n")
    f.write("- Each box labeled with class name and confidence score\n")
    f.write("- Model: YOLOv8n (detection)\n\n")
    
    f.write("## Segmentation Results\n\n")
    f.write("Location: `segmentation/`\n\n")
    f.write("- Shows pixel-level segmentation masks\n")
    f.write("- Colored masks overlay exact object boundaries\n")
    f.write("- Also includes bounding boxes\n")
    f.write("- Model: YOLOv8n-seg (segmentation)\n\n")
    
    f.write("## File Naming\n\n")
    f.write("- Detection: `detection_<original_name>.jpg`\n")
    f.write("- Segmentation: `segmentation_<original_name>.jpg`\n\n")
    
    f.write("## Usage\n\n")
    f.write("These images can be used for:\n")
    f.write("- Visual comparison of detection vs segmentation\n")
    f.write("- Documentation and reports\n")
    f.write("- Presentations\n")
    f.write("- Side-by-side analysis\n")

print(f"\n✓ Created index file: {index_file}")
