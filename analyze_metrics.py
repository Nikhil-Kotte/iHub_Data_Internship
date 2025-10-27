"""
Assignment 2: Detailed Metrics Analysis
Analyze performance metrics and generate insights
"""

import json
from pathlib import Path
from collections import Counter

# Load metrics
metrics_file = Path("results_assignment2/metrics.json")
with open(metrics_file, 'r') as f:
    metrics = json.load(f)

print("=" * 80)
print("DETAILED METRICS ANALYSIS - ASSIGNMENT 2")
print("=" * 80)

# Summary statistics
summary = metrics['summary']
print(f"\n📊 OVERALL STATISTICS")
print(f"{'─' * 80}")
print(f"Total images processed: {summary['total_images_processed']}")
print(f"\nDetection Model Performance:")
print(f"  • Total objects detected: {summary['detection']['total_objects_detected']}")
print(f"  • Average objects per image: {summary['detection']['avg_objects_per_image']}")
print(f"  • Total processing time: {summary['detection']['total_time_sec']}s")
print(f"  • Average inference time: {summary['detection']['avg_inference_time_sec']}s")
print(f"  • Throughput: {round(summary['total_images_processed']/summary['detection']['total_time_sec'], 2)} images/sec")

print(f"\nSegmentation Model Performance:")
print(f"  • Total objects detected: {summary['segmentation']['total_objects_detected']}")
print(f"  • Total masks generated: {summary['segmentation']['total_masks_generated']}")
print(f"  • Average objects per image: {summary['segmentation']['avg_objects_per_image']}")
print(f"  • Total processing time: {summary['segmentation']['total_time_sec']}s")
print(f"  • Average inference time: {summary['segmentation']['avg_inference_time_sec']}s")
print(f"  • Throughput: {round(summary['total_images_processed']/summary['segmentation']['total_time_sec'], 2)} images/sec")

# Performance comparison
print(f"\n⚡ PERFORMANCE COMPARISON")
print(f"{'─' * 80}")
det_time = summary['detection']['avg_inference_time_sec']
seg_time = summary['segmentation']['avg_inference_time_sec']
time_diff = seg_time - det_time
percent_diff = (time_diff / det_time) * 100

print(f"Detection vs Segmentation:")
print(f"  • Detection is faster by: {abs(time_diff):.4f}s per image")
print(f"  • Segmentation takes {percent_diff:.1f}% more time")
print(f"  • Reason: Segmentation generates pixel-level masks (more compute intensive)")

# Object detection analysis
print(f"\n🔍 OBJECT DETECTION ANALYSIS")
print(f"{'─' * 80}")

all_detected_classes_det = Counter()
for img in metrics['detection']:
    for cls, count in img['detected_classes'].items():
        all_detected_classes_det[cls] += count

print(f"Unique classes detected: {len(all_detected_classes_det)}")
print(f"\nClass distribution (Detection):")
for cls, count in all_detected_classes_det.most_common():
    print(f"  • {cls}: {count} instances")

# Segmentation analysis
print(f"\n🎯 SEGMENTATION ANALYSIS")
print(f"{'─' * 80}")

all_detected_classes_seg = Counter()
for img in metrics['segmentation']:
    for cls, count in img['detected_classes'].items():
        all_detected_classes_seg[cls] += count

print(f"Unique classes detected: {len(all_detected_classes_seg)}")
print(f"\nClass distribution (Segmentation):")
for cls, count in all_detected_classes_seg.most_common():
    print(f"  • {cls}: {count} instances")

# Per-image analysis
print(f"\n📸 PER-IMAGE BREAKDOWN")
print(f"{'─' * 80}")

print(f"\n{'Image':<12} {'Detection':<25} {'Segmentation':<25} {'Time (Det/Seg)':<20}")
print(f"{'─' * 80}")

for i in range(len(metrics['detection'])):
    det = metrics['detection'][i]
    seg = metrics['segmentation'][i]
    
    det_classes = ', '.join([f"{v} {k}" for k, v in det['detected_classes'].items()]) or "none"
    seg_classes = ', '.join([f"{v} {k}" for k, v in seg['detected_classes'].items()]) or "none"
    
    print(f"{det['image']:<12} {det_classes:<25} {seg_classes:<25} {det['inference_time_sec']:.4f}s / {seg['inference_time_sec']:.4f}s")

# Find extremes
print(f"\n📈 INTERESTING OBSERVATIONS")
print(f"{'─' * 80}")

# Most objects detected
max_det = max(metrics['detection'], key=lambda x: x['total_objects'])
print(f"\nMost objects in single image (Detection):")
print(f"  • Image: {max_det['image']}")
print(f"  • Objects: {max_det['total_objects']}")
print(f"  • Classes: {max_det['detected_classes']}")

max_seg = max(metrics['segmentation'], key=lambda x: x['total_objects'])
print(f"\nMost objects in single image (Segmentation):")
print(f"  • Image: {max_seg['image']}")
print(f"  • Objects: {max_seg['total_objects']}")
print(f"  • Masks: {max_seg['num_masks']}")
print(f"  • Classes: {max_seg['detected_classes']}")

# Slowest/fastest processing
slowest_det = max(metrics['detection'], key=lambda x: x['inference_time_sec'])
fastest_det = min(metrics['detection'], key=lambda x: x['inference_time_sec'])

print(f"\nProcessing time range (Detection):")
print(f"  • Fastest: {fastest_det['image']} ({fastest_det['inference_time_sec']}s, {fastest_det['total_objects']} objects)")
print(f"  • Slowest: {slowest_det['image']} ({slowest_det['inference_time_sec']}s, {slowest_det['total_objects']} objects)")

# Detection differences
print(f"\n🔬 MODEL COMPARISON INSIGHTS")
print(f"{'─' * 80}")

det_total = summary['detection']['total_objects_detected']
seg_total = summary['segmentation']['total_objects_detected']
diff = seg_total - det_total

print(f"\nObject detection differences:")
print(f"  • Segmentation found {diff} more objects than detection")
print(f"  • This is {round((diff/det_total)*100, 1)}% more objects")
print(f"  • Possible reasons:")
print(f"    - Different confidence thresholds")
print(f"    - Better boundary detection in segmentation model")
print(f"    - Model architectural differences")

# Class differences
det_classes = set(all_detected_classes_det.keys())
seg_classes = set(all_detected_classes_seg.keys())

only_in_det = det_classes - seg_classes
only_in_seg = seg_classes - det_classes

if only_in_det:
    print(f"\n  • Classes only detected by detection model: {', '.join(only_in_det)}")
if only_in_seg:
    print(f"  • Classes only detected by segmentation model: {', '.join(only_in_seg)}")

print(f"\n{'═' * 80}")
print(f"✓ Analysis complete!")
print(f"{'═' * 80}\n")
