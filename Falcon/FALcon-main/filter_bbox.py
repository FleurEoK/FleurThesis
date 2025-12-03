import torch

# Load the results
results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500.pth')

print(f"Original total samples: {len(results)}")

# Filter out samples without detections
filtered_results = {
    sample_idx: sample 
    for sample_idx, sample in results.items() 
    if len(sample['detections']) > 0
}

print(f"Samples after filtering: {len(filtered_results)}")
print(f"Removed {len(results) - len(filtered_results)} samples without detections")

# Analyze the filtered results
total_detections = sum(len(sample['detections']) for sample in filtered_results.values())
print(f"Total detections: {total_detections}")
print(f"Average detections per sample (with detections): {total_detections/len(filtered_results):.2f}")

# Save the filtered results
output_path = '/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth'
torch.save(filtered_results, output_path)
print(f"\nFiltered results saved to: {output_path}")

# Show example
sample_idx = next(iter(filtered_results))
sample = filtered_results[sample_idx]
print(f"\nExample sample {sample_idx}:")
print(f"  Ground truth bboxes: {sample['gt_bboxes']}")
print(f"  Number of detections: {len(sample['detections'])}")