"""
Create video from detection/segmentation result images (task-3)
"""

import cv2
import glob
from pathlib import Path

results_folder = Path("task-3/results_assignment2")
output_video_detection = "task-3/detection_results.mp4"
fps = 5

detection_images = sorted(glob.glob(str(results_folder / "detection_*.jpeg")))

if not detection_images:
    print('No detection images found in task-3/results_assignment2/')
else:
    first_img = cv2.imread(detection_images[0])
    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_detection, fourcc, fps, (w, h))
    for img_path in detection_images:
        img = cv2.imread(img_path)
        video.write(img)
    video.release()
    print(f'Detection video saved: {output_video_detection}')
