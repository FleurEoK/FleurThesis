import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy
from collections import defaultdict

# Load results
results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth')

# Load your config to see the thresholds
from FALcon_config_test_as_WSOL import FALcon_config
config = FALcon_config

print("Current thresholds:")
print(f"  switch_location_th: {config.switch_location_th}")
print(f"  objectness_based_nms_th: {config.objectness_based_nms_th}")
print(f"  glimpse_change_th: {config.glimpse_change_th}")

# Visualize a sample with detection
sample_idx = 3
sample = results[sample_idx]

print(f"\nSample {sample_idx}:")
print(f"Ground truth bbox (xyxy): {sample['gt_bboxes']}")
if len(sample['detections']) > 0:
    for i, det in enumerate(sample['detections']):
        print(f"Predicted bbox {i} (xyxy): {det['bbox_xyxy']}")
        print(f"  Objectness: {det['objectness_score']:.3f}")


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in xyxy format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Calculate overall and per-class localization accuracy
ious = []
class_ious = defaultdict(list)
class_counts = defaultdict(int)
class_names = {}  # Store synset names

for sample_idx, sample in results.items():
    # Get class information
    if 'gt_synsets' in sample:
        class_id = sample['gt_synsets'][0]  # Use synset as class identifier
    elif 'gt_labels' in sample:
        class_id = sample['gt_labels'][0]  # Use label as class identifier
    else:
        class_id = 'unknown'
    
    class_counts[class_id] += 1
    
    if len(sample['detections']) > 0:
        # Take the top detection (highest objectness)
        best_det = max(sample['detections'], key=lambda x: x['objectness_score'])
        gt_bbox = sample['gt_bboxes'][0]
        pred_bbox = best_det['bbox_xyxy'].float().cpu()
        
        iou = calculate_iou(gt_bbox, pred_bbox)
        ious.append(iou)
        class_ious[class_id].append(iou)

ious = numpy.array(ious)

# Overall statistics
print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)
print(f"Total samples: {len(results)}")
print(f"Samples with detections: {len(ious)} ({100*len(ious)/len(results):.1f}%)")
print(f"\nIoU Statistics (for samples with detections):")
print(f"  Mean IoU: {ious.mean():.3f}")
print(f"  Median IoU: {numpy.median(ious):.3f}")
print(f"  IoU >= 0.3: {(ious >= 0.3).sum()} ({100*(ious >= 0.3).mean():.1f}%)")
print(f"  IoU >= 0.5: {(ious >= 0.5).sum()} ({100*(ious >= 0.5).mean():.1f}%)")

# CorLoc: percentage of images with at least one detection with IoU >= 0.5
corloc = (ious >= 0.5).sum() / len(results) * 100
print(f"\nCorLoc (Correct Localization): {corloc:.1f}%")

# Per-class statistics
print("\n" + "="*60)
print("PER-CLASS STATISTICS")
print("="*60)
print(f"Number of unique classes: {len(class_counts)}")

# Sort classes by number of samples
sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'Class':<15} {'Total':<8} {'W/Det':<8} {'Mean IoU':<10} {'IoU>=0.5':<10} {'CorLoc%':<10}")
print("-" * 80)

per_class_metrics = []
for class_id, total_count in sorted_classes:
    class_iou_values = numpy.array(class_ious[class_id]) if class_id in class_ious else numpy.array([])
    num_with_det = len(class_iou_values)
    
    if num_with_det > 0:
        mean_iou = class_iou_values.mean()
        iou_05_count = (class_iou_values >= 0.5).sum()
        corloc_class = (iou_05_count / total_count) * 100
        per_class_metrics.append({
            'class_id': class_id,
            'total': total_count,
            'with_det': num_with_det,
            'mean_iou': mean_iou,
            'iou_05_count': iou_05_count,
            'corloc': corloc_class
        })
        print(f"{str(class_id):<15} {total_count:<8} {num_with_det:<8} {mean_iou:<10.3f} {iou_05_count:<10} {corloc_class:<10.1f}")
    else:
        print(f"{str(class_id):<15} {total_count:<8} {0:<8} {'N/A':<10} {0:<10} {0.0:<10.1f}")

# Summary statistics
print("\n" + "="*60)
print("PER-CLASS SUMMARY")
print("="*60)
if per_class_metrics:
    mean_ious_per_class = [m['mean_iou'] for m in per_class_metrics]
    corlocs_per_class = [m['corloc'] for m in per_class_metrics]
    
    print(f"Mean IoU across classes: {numpy.mean(mean_ious_per_class):.3f}")
    print(f"Std IoU across classes: {numpy.std(mean_ious_per_class):.3f}")
    print(f"Mean CorLoc across classes: {numpy.mean(corlocs_per_class):.1f}%")
    print(f"Std CorLoc across classes: {numpy.std(corlocs_per_class):.1f}%")
    
    # Best and worst performing classes
    best_class = max(per_class_metrics, key=lambda x: x['mean_iou'])
    worst_class = min(per_class_metrics, key=lambda x: x['mean_iou'])
    
    print(f"\nBest performing class: {best_class['class_id']} (Mean IoU: {best_class['mean_iou']:.3f})")
    print(f"Worst performing class: {worst_class['class_id']} (Mean IoU: {worst_class['mean_iou']:.3f})")