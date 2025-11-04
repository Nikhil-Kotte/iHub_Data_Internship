"""
Extract COCO8 validation images and copy validation previews to task-3 folder.
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

output_dir = Path('task-3/coco8_validation_images')
output_dir.mkdir(parents=True, exist_ok=True)

model = YOLO('yolov8n.pt')
model.val(data='coco8.yaml', verbose=False)

possible = [Path.cwd() / 'datasets' / 'coco8' / 'images' / 'val']
for p in possible:
    if p.exists():
        for i, img in enumerate(sorted(p.glob('*.jpg')), 1):
            shutil.copy(img, output_dir / f'coco8_image_{i}_{img.name}')
        break

print(f'COCO8 images copied to: {output_dir}')
