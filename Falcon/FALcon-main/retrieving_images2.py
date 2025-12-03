import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import glob

# Create results folder
results_vis_folder = 'results_vis'
os.makedirs(results_vis_folder, exist_ok=True)
print(f"Created/using folder: {results_vis_folder}")

# Load results
results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth')

dataset_root = '/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train'

# Step 1: Build a mapping from sample index to image path
print("\nBuilding index to image path mapping...")
print("=" * 60)

# Get all synsets (class folders)
synsets = sorted([d for d in os.listdir(dataset_root) if d.startswith('n')])
print(f"Found {len(synsets)} synset folders: {synsets}")

# Create mapping: class_id -> synset_name
class_to_synset = {i: synset for i, synset in enumerate(synsets)}
print("\nClass to Synset mapping:")
for class_id, synset in class_to_synset.items():
    print(f"  Class {class_id} -> {synset}")

# Build complete list of all images in order
all_images = []
for synset in synsets:
    synset_folder = os.path.join(dataset_root, synset)
    images = sorted(glob.glob(os.path.join(synset_folder, '*.JPEG')))
    for img_path in images:
        all_images.append(img_path)

print(f"\nTotal images found: {len(all_images)}")
print(f"First few images:\n  " + "\n  ".join(all_images[:3]))

# Create index to image path mapping
index_to_image = {i: img_path for i, img_path in enumerate(all_images)}


def visualize_sample_with_mapping(sample_idx, results, index_to_image, output_folder):
    """
    Visualize a sample using the index to image mapping
    """
    if sample_idx not in results:
        print(f"Sample {sample_idx} not in results")
        return
    
    if sample_idx not in index_to_image:
        print(f"Sample {sample_idx} not in image mapping (only have {len(index_to_image)} images)")
        return
    
    sample = results[sample_idx]
    img_path = index_to_image[sample_idx]
    
    # Load image
    try:
        img = Image.open(img_path).convert('RGB')
        print(f"Loaded: {img_path}")
        print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    
    # Ground truth bbox (RED)
    gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
    # The bbox appears to be in [x, y, w, h] format based on the values
    x, y, w, h = gt_bbox
    
    rect = patches.Rectangle((x, y), w, h, 
                             linewidth=3, edgecolor='red', 
                             facecolor='none', label='Ground Truth', linestyle='-')
    ax.add_patch(rect)
    
    # Predicted bboxes (GREEN) - now handling multiple detections
    num_detections = len(sample['detections'])
    colors = ['green', 'cyan', 'yellow', 'magenta']  # Different colors for multiple detections
    
    for i, det in enumerate(sample['detections']):
        pred_bbox = det['bbox_xyxy'].cpu().numpy()
        x1, y1, x2, y2 = pred_bbox
        width = x2 - x1
        height = y2 - y1
        objectness = det['objectness_score']
        
        color = colors[i % len(colors)] if num_detections > 1 else 'green'
        
        rect = patches.Rectangle((x1, y1), width, height, 
                                 linewidth=2, edgecolor=color, 
                                 facecolor='none', 
                                 label=f'Prediction {i+1} (obj={objectness:.3f})', 
                                 linestyle='-')
        ax.add_patch(rect)
    
    # Calculate IoU for best detection
    gt_x1, gt_y1, gt_x2, gt_y2 = x, y, x+w, y+h
    if len(sample['detections']) > 0:
        # Use highest objectness detection
        best_det = max(sample['detections'], key=lambda d: d['objectness_score'])
        pred_bbox = best_det['bbox_xyxy'].cpu().numpy()
        px1, py1, px2, py2 = pred_bbox
        
        # IoU calculation
        xi1 = max(gt_x1, px1)
        yi1 = max(gt_y1, py1)
        xi2 = min(gt_x2, px2)
        yi2 = min(gt_y2, py2)
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        pred_area = (px2 - px1) * (py2 - py1)
        union = gt_area + pred_area - intersection
        iou = intersection / union if union > 0 else 0
    else:
        iou = 0.0
    
    # Title with info
    class_id = sample['gt_synsets'][0]
    synset = class_to_synset.get(class_id, 'unknown')
    img_name = os.path.basename(img_path)
    
    ax.legend(loc='upper right', fontsize=10)
    title = f'Sample {sample_idx} | Class {class_id} ({synset}) | {num_detections} detection(s) | Best IoU: {iou:.3f}\n{img_name}'
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, f'sample_{sample_idx}_class_{class_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}\n")


# Find samples with multiple detections
print("\n" + "=" * 60)
print("FINDING SAMPLES WITH MULTIPLE DETECTIONS")
print("=" * 60)

multi_detection_samples = []
detection_counts = {}

for sample_idx, sample in results.items():
    num_detections = len(sample['detections'])
    if num_detections not in detection_counts:
        detection_counts[num_detections] = 0
    detection_counts[num_detections] += 1
    
    if num_detections > 1:
        multi_detection_samples.append((sample_idx, num_detections))

print(f"Detection count distribution:")
for num_dets, count in sorted(detection_counts.items()):
    print(f"  {num_dets} detection(s): {count} samples ({100*count/len(results):.1f}%)")

print(f"\nFound {len(multi_detection_samples)} samples with multiple detections")
if multi_detection_samples:
    print(f"Samples with most detections:")
    # Sort by number of detections
    multi_detection_samples.sort(key=lambda x: x[1], reverse=True)
    for sample_idx, num_dets in multi_detection_samples[:5]:
        print(f"  Sample {sample_idx}: {num_dets} detections")


# Visualize samples with multiple detections
print("\n" + "=" * 60)
print("VISUALIZING SAMPLES WITH MULTIPLE DETECTIONS")
print("=" * 60)

# Create a subfolder for multi-detection samples
multi_det_folder = os.path.join(results_vis_folder, 'multiple_detections')
os.makedirs(multi_det_folder, exist_ok=True)

if multi_detection_samples:
    # Visualize top 10 samples with most detections
    for sample_idx, num_dets in multi_detection_samples[:10]:
        print(f"\n--- Sample {sample_idx} with {num_dets} detections ---")
        if sample_idx in index_to_image:
            sample = results[sample_idx]
            img_path = index_to_image[sample_idx]
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                ax.imshow(img)
                
                # GT bbox (red)
                gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
                x, y, w, h = gt_bbox
                rect = patches.Rectangle((x, y), w, h, 
                                         linewidth=4, edgecolor='red', 
                                         facecolor='none', label='Ground Truth', linestyle='-')
                ax.add_patch(rect)
                
                # Multiple predicted bboxes with different colors
                colors = ['green', 'cyan', 'yellow', 'magenta', 'orange', 'purple']
                
                for i, det in enumerate(sample['detections']):
                    pred_bbox = det['bbox_xyxy'].cpu().numpy()
                    x1, y1, x2, y2 = pred_bbox
                    objectness = det['objectness_score']
                    color = colors[i % len(colors)]
                    
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                             linewidth=3, edgecolor=color, 
                                             facecolor='none', 
                                             label=f'Detection {i+1} (obj={objectness:.3f})')
                    ax.add_patch(rect)
                
                class_id = sample['gt_synsets'][0]
                synset = class_to_synset.get(class_id, 'unknown')
                img_name = os.path.basename(img_path)
                
                ax.legend(loc='upper right', fontsize=9)
                ax.set_title(f'Sample {sample_idx} | Class {class_id} ({synset}) | {num_dets} Detections\n{img_name}', 
                             fontsize=12, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(multi_det_folder, f'multi_det_sample_{sample_idx}_{num_dets}dets.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
else:
    print("No samples with multiple detections found")


# Visualize individual samples
print("\n" + "=" * 60)
print("VISUALIZING INDIVIDUAL SAMPLES")
print("=" * 60)

for sample_idx in [0, 3, 10, 50, 100]:
    if sample_idx in results:
        print(f"\n--- Sample {sample_idx} ---")
        visualize_sample_with_mapping(sample_idx, results, index_to_image, results_vis_folder)


# Create grid visualization for a specific class
def visualize_class_grid(results, index_to_image, class_to_synset, output_folder, target_class=0, num_samples=9):
    """
    Create a grid of visualizations for a specific class
    """
    # Find samples from target class
    class_samples = [(idx, sample) for idx, sample in results.items() 
                     if sample['gt_synsets'][0] == target_class]
    
    print(f"\nFound {len(class_samples)} samples from Class {target_class}")
    
    # Take first num_samples
    class_samples = class_samples[:num_samples]
    
    rows = 3
    cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    axes = axes.flatten()
    
    for idx, (sample_idx, sample) in enumerate(class_samples):
        if sample_idx not in index_to_image:
            continue
            
        img_path = index_to_image[sample_idx]
        img = Image.open(img_path).convert('RGB')
        
        ax = axes[idx]
        ax.imshow(img)
        
        # GT bbox (red)
        gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
        x, y, w, h = gt_bbox
        rect = patches.Rectangle((x, y), w, h, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Pred bbox (green for first, other colors for additional)
        num_detections = len(sample['detections'])
        colors = ['green', 'cyan', 'yellow']
        
        if num_detections > 0:
            for i, det in enumerate(sample['detections']):
                pred_bbox = det['bbox_xyxy'].cpu().numpy()
                x1, y1, x2, y2 = pred_bbox
                color = colors[i % len(colors)]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
            
            # Use best detection for metrics
            best_det = max(sample['detections'], key=lambda d: d['objectness_score'])
            objectness = best_det['objectness_score']
            
            # Calculate IoU
            pred_bbox = best_det['bbox_xyxy'].cpu().numpy()
            x1, y1, x2, y2 = pred_bbox
            gt_x1, gt_y1 = x, y
            gt_x2, gt_y2 = x + w, y + h
            xi1 = max(gt_x1, x1)
            yi1 = max(gt_y1, y1)
            xi2 = min(gt_x2, x2)
            yi2 = min(gt_y2, y2)
            intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            gt_area = w * h
            pred_area = (x2 - x1) * (y2 - y1)
            union = gt_area + pred_area - intersection
            iou = intersection / union if union > 0 else 0
            
            title = f'Sample {sample_idx}\n{num_detections} det(s) | Obj: {objectness:.3f} | IoU: {iou:.3f}'
            ax.set_title(title, fontsize=8)
        else:
            ax.set_title(f'Sample {sample_idx}\nNo detection', fontsize=9)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(class_samples), len(axes)):
        axes[idx].axis('off')
    
    synset = class_to_synset.get(target_class, 'unknown')
    plt.suptitle(f'Class {target_class} ({synset}) - Red: GT, Green/Cyan/Yellow: Predictions', 
                 fontsize=14, y=0.995)
    plt.tight_layout()
    save_path = os.path.join(output_folder, f'class_{target_class}_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved grid visualization: {save_path}")


# Visualize all classes as grids
print("\n" + "=" * 60)
print("CREATING CLASS GRID VISUALIZATIONS")
print("=" * 60)

for class_id in range(5):  # Classes 0-4
    print(f"\n--- Class {class_id} ---")
    visualize_class_grid(results, index_to_image, class_to_synset, 
                        results_vis_folder, target_class=class_id, num_samples=9)


# Create a summary visualization comparing all classes
def create_class_comparison(results, index_to_image, class_to_synset, output_folder):
    """
    Create a comparison showing one sample from each class
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    for class_id in range(5):
        # Find first sample from this class
        class_samples = [(idx, sample) for idx, sample in results.items() 
                         if sample['gt_synsets'][0] == class_id]
        
        if not class_samples:
            continue
        
        sample_idx, sample = class_samples[0]
        
        if sample_idx not in index_to_image:
            continue
        
        img_path = index_to_image[sample_idx]
        img = Image.open(img_path).convert('RGB')
        
        ax = axes[class_id]
        ax.imshow(img)
        
        # GT bbox (red)
        gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
        x, y, w, h = gt_bbox
        rect = patches.Rectangle((x, y), w, h, 
                                 linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Pred bbox (green)
        if len(sample['detections']) > 0:
            det = sample['detections'][0]
            pred_bbox = det['bbox_xyxy'].cpu().numpy()
            x1, y1, x2, y2 = pred_bbox
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=3, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        synset = class_to_synset.get(class_id, 'unknown')
        ax.set_title(f'Class {class_id}\n{synset}', fontsize=12)
        ax.axis('off')
    
    plt.suptitle('Class Comparison - Red: GT, Green: Prediction', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(output_folder, 'class_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved class comparison: {save_path}")


print("\n" + "=" * 60)
print("CREATING CLASS COMPARISON")
print("=" * 60)
create_class_comparison(results, index_to_image, class_to_synset, results_vis_folder)

print("\n" + "=" * 60)
print(f"ALL VISUALIZATIONS SAVED TO: {results_vis_folder}/")
print(f"Multiple detection samples saved to: {multi_det_folder}/")
print("=" * 60)