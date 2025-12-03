import torch

# Load the results
results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500.pth')

# Analyze detection statistics
total_samples = len(results)
samples_with_detections = 0
samples_without_detections = 0
total_detections = 0
detection_counts = []

for sample_idx, sample in results.items():
    num_detections = len(sample['detections'])
    detection_counts.append(num_detections)
    total_detections += num_detections
    
    if num_detections > 0:
        samples_with_detections += 1
    else:
        samples_without_detections += 1

print(f"Total samples: {total_samples}")
print(f"Samples with detections: {samples_with_detections} ({100*samples_with_detections/total_samples:.1f}%)")
print(f"Samples without detections: {samples_without_detections} ({100*samples_without_detections/total_samples:.1f}%)")
print(f"Total detections: {total_detections}")
print(f"Average detections per sample: {total_detections/total_samples:.2f}")
print(f"Max detections in a sample: {max(detection_counts)}")
print(f"Min detections in a sample: {min(detection_counts)}")

# Show a sample WITH detections
for sample_idx, sample in results.items():
    if len(sample['detections']) > 0:
        print(f"\n\nExample sample {sample_idx} WITH detections:")
        print(f"  Ground truth bboxes: {sample['gt_bboxes']}")
        print(f"  Number of detections: {len(sample['detections'])}")
        for i, det in enumerate(sample['detections']):
            print(f"  Detection {i}: bbox_xyxy={det['bbox_xyxy']}, objectness={det['objectness_score']:.3f}")
        break