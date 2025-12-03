# import torch
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import os

# # Load results
# results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth')

# def visualize_sample(sample_idx, results, dataset_root='/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train'):
#     """
#     Visualize a sample with ground truth and predicted bounding boxes
#     """
#     sample = results[sample_idx]
    
#     # Extract information
#     if 'image_path' in sample:
#         # If full path is stored
#         img_path = sample['image_path']
#     elif 'gt_synsets' in sample and 'image_id' in sample:
#         # Reconstruct path from synset and image_id
#         synset = sample['gt_synsets'][0]
#         img_id = sample['image_id']
#         img_path = os.path.join(dataset_root, synset, f"{img_id}.JPEG")
#     elif 'gt_synsets' in sample:
#         # Try to find the image in the synset folder
#         synset = sample['gt_synsets'][0]
#         synset_folder = os.path.join(dataset_root, synset)
#         # List all images and take the one at sample_idx position (approximate)
#         images = sorted([f for f in os.listdir(synset_folder) if f.endswith('.JPEG')])
#         if images:
#             img_path = os.path.join(synset_folder, images[0])
#     else:
#         print("Cannot determine image path from sample data")
#         print("Available keys:", sample.keys())
#         return
    
#     # Load image
#     try:
#         img = Image.open(img_path).convert('RGB')
#     except:
#         print(f"Could not load image from {img_path}")
#         return
    
#     # Create figure
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))
#     ax.imshow(img)
    
#     # Draw ground truth bbox (in red)
#     gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
#     # Check if bbox is in xywh or xyxy format
#     if gt_bbox.shape[0] == 4:
#         # Assume xyxy format: [x1, y1, x2, y2]
#         x1, y1, x2, y2 = gt_bbox
#         width = x2 - x1
#         height = y2 - y1
#     else:
#         print(f"Unexpected bbox format: {gt_bbox}")
#         return
    
#     rect = patches.Rectangle((x1, y1), width, height, 
#                              linewidth=3, edgecolor='red', 
#                              facecolor='none', label='Ground Truth')
#     ax.add_patch(rect)
    
#     # Draw predicted bboxes (in green)
#     for i, det in enumerate(sample['detections']):
#         pred_bbox = det['bbox_xyxy'].cpu().numpy()
#         x1, y1, x2, y2 = pred_bbox
#         width = x2 - x1
#         height = y2 - y1
#         objectness = det['objectness_score']
        
#         rect = patches.Rectangle((x1, y1), width, height, 
#                                  linewidth=2, edgecolor='green', 
#                                  facecolor='none', 
#                                  label=f'Prediction {i+1} (obj={objectness:.2f})')
#         ax.add_patch(rect)
    
#     # Add legend and title
#     ax.legend(loc='upper right')
#     class_info = sample.get('gt_synsets', ['unknown'])[0]
#     ax.set_title(f'Sample {sample_idx} - Class: {class_info}')
#     ax.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(f'/home/20204130/Falcon/FALcon-main/results/vis/visualization_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
#     plt.show()
    
#     print(f"Saved visualization to: visualization_sample_{sample_idx}.png")


# # First, let's check what information is available in a sample
# print("Sample 0 keys:", results[0].keys())
# print("\nSample 0 structure:")
# for key, value in results[0].items():
#     if isinstance(value, torch.Tensor):
#         print(f"  {key}: {value.shape} - {value.dtype}")
#     elif isinstance(value, list):
#         print(f"  {key}: list of length {len(value)}")
#     else:
#         print(f"  {key}: {type(value)}")

# # Try to visualize a few samples
# print("\n" + "="*60)
# print("Attempting to visualize samples...")
# print("="*60)

# # Visualize a few different samples
# for sample_idx in [0, 3, 10, 100]:
#     if sample_idx in results:
#         print(f"\nVisualizing sample {sample_idx}...")
#         visualize_sample(sample_idx, results)

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

# Load results
results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth')

# First, let's inspect what's actually in the sample
print("=" * 60)
print("INSPECTING SAMPLE STRUCTURE")
print("=" * 60)
sample_0 = results[0]
print("\nSample 0 keys:", list(sample_0.keys()))
print("\nDetailed structure:")
for key, value in sample_0.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: tensor {value.shape}, dtype={value.dtype}")
        if value.numel() <= 10:  # Print small tensors
            print(f"    value: {value}")
    elif isinstance(value, list):
        print(f"  {key}: list of length {len(value)}")
        if len(value) > 0 and len(value) <= 3:
            print(f"    first item type: {type(value[0])}")
            print(f"    content: {value}")
    else:
        print(f"  {key}: {type(value).__name__}")
        print(f"    value: {value}")

print("\n" + "=" * 60)
print("If you see 'image_path' or 'image_id' above, we can use it!")
print("=" * 60)


def visualize_sample_v2(sample_idx, results, dataset_root='/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train'):
    """
    Visualize a sample - handles multiple possible data formats
    """
    sample = results[sample_idx]
    
    img_path = None
    
    # Method 1: Direct image path
    if 'image_path' in sample:
        img_path = sample['image_path']
        print(f"Found image_path: {img_path}")
    
    # Method 2: Using image_id (common format)
    elif 'image_id' in sample:
        img_id = sample['image_id']
        print(f"Found image_id: {img_id}")
        
        # ImageNet IDs are typically like "n01440764_10026"
        # Split to get synset and image number
        if isinstance(img_id, str) and '_' in img_id:
            synset = img_id.split('_')[0]
            img_path = os.path.join(dataset_root, synset, f"{img_id}.JPEG")
        else:
            print(f"Unexpected image_id format: {img_id}")
    
    # Method 3: Using metadata if available
    elif 'metadata' in sample and 'image_path' in sample['metadata']:
        img_path = sample['metadata']['image_path']
    
    # Method 4: Try to reconstruct from index (last resort)
    else:
        print("Could not find image path information in sample")
        print("Available keys:", list(sample.keys()))
        
        # Try to list dataset structure to help
        if os.path.exists(dataset_root):
            synsets = sorted([d for d in os.listdir(dataset_root) if d.startswith('n')])
            print(f"\nFound {len(synsets)} synset folders in dataset")
            print(f"First few synsets: {synsets[:5]}")
            
            # User needs to provide mapping
            print("\nYou may need to provide the original dataset or index mapping")
        return None
    
    # Try to load the image
    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            print(f"Successfully loaded image: {img_path}")
            print(f"Image size: {img.size}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    else:
        print(f"Image path does not exist: {img_path}")
        return None
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    # Draw ground truth bbox (red)
    gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
    
    # Handle both xywh and xyxy formats
    if len(gt_bbox) == 4:
        # Check if it looks like xywh (x, y, w, h) or xyxy (x1, y1, x2, y2)
        # Usually if values are all positive and bbox[2] < image_width, it's xyxy
        x1, y1, x2, y2 = gt_bbox
        if x2 > img.size[0] or y2 > img.size[1]:  # Likely xywh format
            x, y, w, h = gt_bbox
            x1, y1, x2, y2 = x, y, x+w, y+h
        width = x2 - x1
        height = y2 - y1
    
    rect = patches.Rectangle((x1, y1), width, height, 
                             linewidth=3, edgecolor='red', 
                             facecolor='none', label='Ground Truth')
    ax.add_patch(rect)
    
    # Draw predicted bboxes (green)
    for i, det in enumerate(sample['detections']):
        pred_bbox = det['bbox_xyxy'].cpu().numpy()
        x1, y1, x2, y2 = pred_bbox
        width = x2 - x1
        height = y2 - y1
        objectness = det['objectness_score']
        
        rect = patches.Rectangle((x1, y1), width, height, 
                                 linewidth=2, edgecolor='green', 
                                 facecolor='none', 
                                 label=f'Pred (obj={objectness:.3f})')
        ax.add_patch(rect)
    
    # Add title with class info
    class_info = sample.get('gt_synsets', sample.get('gt_labels', ['unknown']))[0]
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Sample {sample_idx} - Class: {class_info}', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = f'/home/20204130/Falcon/FALcon-main/results/vis/visualization_sample_{sample_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved visualization to: {save_path}\n")
    return img_path


# Try to visualize
print("\n" + "=" * 60)
print("ATTEMPTING VISUALIZATION")
print("=" * 60)

for sample_idx in [0, 3]:
    if sample_idx in results:
        print(f"\n--- Sample {sample_idx} ---")
        visualize_sample_v2(sample_idx, results)