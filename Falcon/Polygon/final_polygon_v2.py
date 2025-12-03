# generate_training_data.py
"""
Complete pipeline: Generate polygons + compute importance scores + save training data
All in one efficient pass through the data
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import numpy as np
import json
from shapely.geometry import box, mapping
from shapely.ops import unary_union
import seaborn as sns


class TrainingDataGenerator:
    """
    Generates training data with importance scores from detection results
    Grid size of 14x14 to match 224px images with 16px patches
    """
    
    def __init__(self, grid_size=(14, 14)):
        """
        Args:
            grid_size: (rows, cols) - use (14, 14) for 224px images with 16px patches
        """
        self.grid_size = grid_size
        self.num_tokens = grid_size[0] * grid_size[1]
        self.grid_width = 1.0 / grid_size[1]
        self.grid_height = 1.0 / grid_size[0]
    
    def create_grid_cells(self):
        """Create grid cell polygons in normalized coordinates"""
        grid_cells = []
        
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                x_min = col * self.grid_width
                y_min = row * self.grid_height
                x_max = (col + 1) * self.grid_width
                y_max = (row + 1) * self.grid_height
                
                cell_polygon = box(x_min, y_min, x_max, y_max)
                grid_cells.append({
                    'polygon': cell_polygon,
                    'row': row,
                    'col': col,
                    'token_idx': row * self.grid_size[1] + col
                })
        
        return grid_cells
    
    def bbox_to_polygon(self, bbox_xyxy, image_size):
        """Convert bbox from pixels to normalized polygon"""
        if hasattr(bbox_xyxy, 'cpu'):
            bbox = bbox_xyxy.cpu().numpy()
        else:
            bbox = np.array(bbox_xyxy)
        
        x1, y1, x2, y2 = bbox
        img_width, img_height = image_size
        
        x1_norm = float(x1 / img_width)
        y1_norm = float(y1 / img_height)
        x2_norm = float(x2 / img_width)
        y2_norm = float(y2 / img_height)
        
        return box(x1_norm, y1_norm, x2_norm, y2_norm)
    
    def process_single_image(self, sample, img_path):
        """
        Process one image: create polygons AND compute importance scores
        
        Returns:
            dict with all data needed for training
        """
        # Load image
        img = Image.open(img_path)
        image_size = img.size
        
        # Create grid cells
        grid_cells = self.create_grid_cells()
        
        # Get detection polygons
        detection_polygons = []
        detection_info = []
        
        for i, det in enumerate(sample['detections']):
            bbox_xyxy = det['bbox_xyxy']
            poly = self.bbox_to_polygon(bbox_xyxy, image_size)
            objectness = float(det['objectness_score'])
            
            detection_polygons.append(poly)
            detection_info.append({
                'detection_id': i,
                'bbox_xyxy_pixels': bbox_xyxy.cpu().tolist() if hasattr(bbox_xyxy, 'cpu') else bbox_xyxy.tolist(),
                'objectness_score': objectness,
                'polygon_geojson': mapping(poly)
            })
        
        # Create unified polygon
        if len(detection_polygons) == 1:
            unified_polygon = detection_polygons[0]
        else:
            unified_polygon = unary_union(detection_polygons)
        
        # Compute importance scores (overlap counts per grid cell)
        importance_scores = []
        for cell in grid_cells:
            cell_poly = cell['polygon']
            overlap_count = sum(1 for det_poly in detection_polygons 
                              if cell_poly.intersects(det_poly))
            importance_scores.append(overlap_count)
        
        # Get ground truth
        gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
        
        # Return complete data
        return {
            'image_path': img_path,
            'image_name': os.path.basename(img_path),
            'image_size': list(image_size),
            'class_id': int(sample['gt_synsets'][0]),
            'num_detections': len(detection_polygons),
            
            # Importance scores (for training)
            'importance_scores': importance_scores,
            'max_importance': max(importance_scores),
            'mean_importance': float(np.mean(importance_scores)),
            'total_overlaps': sum(importance_scores),
            
            # Polygon data (for reference/visualization)
            'unified_polygon': mapping(unified_polygon),
            'individual_detections': detection_info,
            'ground_truth_bbox_pixels': gt_bbox.tolist()
        }
    
    def process_all_images(self, results, index_to_image):
        """Process all images in one pass"""
        all_data = {}
        
        print(f"\nProcessing {len(results)} images...")
        
        processed = 0
        for sample_idx, sample in results.items():
            if sample_idx not in index_to_image:
                continue
            
            img_path = index_to_image[sample_idx]
            img_name = os.path.basename(img_path)
            
            try:
                result = self.process_single_image(sample, img_path)
                all_data[img_name] = result
                
                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{len(results)} images")
                    
            except Exception as e:
                print(f"  Error processing {img_name}: {e}")
                continue
        
        print(f"\nSuccessfully processed {processed} images")
        return all_data
    
    def save_training_data(self, all_data, output_folder):
        """
        Save training data in the format needed by the dataset loader
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 1. Training data (minimal - what the model needs)
        training_data = {}
        for img_name, data in all_data.items():
            training_data[img_name] = {
                'image_path': data['image_path'],
                'class_id': data['class_id'],
                'num_detections': data['num_detections'],
                'importance_scores': data['importance_scores'],
                'max_importance': data['max_importance'],
                'mean_importance': data['mean_importance']
            }
        
        training_path = os.path.join(output_folder, 'training_data.json')
        with open(training_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\n✓ Saved training data to: {training_path}")
        
        # 2. Complete data (with polygons for reference)
        complete_path = os.path.join(output_folder, 'complete_data.json')
        with open(complete_path, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"✓ Saved complete data to: {complete_path}")
        
        return training_path, complete_path
    
    def analyze_importance_distribution(self, all_data):
        """Analyze importance score distribution"""
        print("\n" + "="*60)
        print("IMPORTANCE SCORE ANALYSIS")
        print("="*60)
        
        all_scores = []
        all_max_scores = []
        
        for data in all_data.values():
            all_scores.extend(data['importance_scores'])
            all_max_scores.append(data['max_importance'])
        
        print(f"\nOverall statistics (all {len(all_scores)} grid cells):")
        print(f"  Mean: {np.mean(all_scores):.3f}")
        print(f"  Median: {np.median(all_scores):.3f}")
        print(f"  Std: {np.std(all_scores):.3f}")
        print(f"  Min: {int(np.min(all_scores))}")
        print(f"  Max: {int(np.max(all_scores))}")
        
        print(f"\nPer-image statistics:")
        print(f"  Mean max importance: {np.mean(all_max_scores):.2f}")
        
        # Distribution
        unique, counts = np.unique(all_scores, return_counts=True)
        print(f"\nImportance value distribution:")
        for val, count in zip(unique, counts):
            percentage = 100 * count / len(all_scores)
            print(f"  {int(val)}: {count:6d} cells ({percentage:5.1f}%)")
        
        # Multi-detection analysis
        multi_det_images = {name: data for name, data in all_data.items() 
                           if data['num_detections'] > 1}
        
        print(f"\nImages with multiple detections: {len(multi_det_images)} ({100*len(multi_det_images)/len(all_data):.1f}%)")
        
        if multi_det_images:
            # Show detection count distribution
            det_counts = {}
            for data in multi_det_images.values():
                n = data['num_detections']
                det_counts[n] = det_counts.get(n, 0) + 1
            
            print("\nMultiple detection distribution:")
            for n_dets, count in sorted(det_counts.items()):
                print(f"  {n_dets} detections: {count} images")
        
        return multi_det_images
    
    def print_multi_detection_examples(self, all_data, num_examples=5):
        """Print detailed examples of images with multiple detections"""
        # Find images with multiple detections
        multi_det_images = [(name, data) for name, data in all_data.items() 
                           if data['num_detections'] > 1]
        
        if not multi_det_images:
            print("\nNo images with multiple detections found.")
            return []
        
        # Sort by number of detections (descending)
        multi_det_images.sort(key=lambda x: x[1]['num_detections'], reverse=True)
        
        print("\n" + "="*60)
        print("MULTI-DETECTION EXAMPLES")
        print("="*60)
        
        examples = multi_det_images[:num_examples]
        
        for i, (img_name, data) in enumerate(examples, 1):
            print(f"\n--- Example {i}: {img_name} ---")
            print(f"  Class ID: {data['class_id']}")
            print(f"  Number of detections: {data['num_detections']}")
            print(f"  Max importance: {data['max_importance']}")
            print(f"  Mean importance: {data['mean_importance']:.3f}")
            print(f"  Total overlaps: {data['total_overlaps']}")
            
            print(f"  Individual detections:")
            for det in data['individual_detections']:
                bbox = det['bbox_xyxy_pixels']
                print(f"    Detection {det['detection_id']}: "
                      f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}], "
                      f"objectness={det['objectness_score']:.3f}")
            
            # Show grid with overlap counts
            grid = np.array(data['importance_scores']).reshape(self.grid_size)
            cells_with_2plus = np.sum(grid >= 2)
            cells_with_1 = np.sum(grid == 1)
            cells_with_0 = np.sum(grid == 0)
            
            print(f"  Grid overlap statistics:")
            print(f"    Cells with 2+ overlaps: {cells_with_2plus}")
            print(f"    Cells with 1 overlap: {cells_with_1}")
            print(f"    Cells with 0 overlaps: {cells_with_0}")
            
            # Show a few high-importance cells
            high_importance_cells = [(idx, score) for idx, score in enumerate(data['importance_scores']) 
                                    if score >= 2]
            if high_importance_cells:
                print(f"  High importance cells (score ≥ 2):")
                for idx, score in high_importance_cells[:5]:
                    row = idx // self.grid_size[1]
                    col = idx % self.grid_size[1]
                    print(f"    Cell (row={row}, col={col}): importance={score}")
        
        return examples
    
    def visualize_examples(self, all_data, output_folder, num_examples=5):
        """Visualize importance scores for example images (single detections)"""
        # Get single detection examples
        single_det_images = [(name, data) for name, data in all_data.items() 
                            if data['num_detections'] == 1]
        
        if single_det_images:
            example_images = single_det_images[:num_examples]
            
            fig, axes = plt.subplots(1, min(num_examples, len(example_images)), 
                                    figsize=(4*min(num_examples, len(example_images)), 4))
            if num_examples == 1:
                axes = [axes]
            
            for i, (img_name, data) in enumerate(example_images):
                grid = np.array(data['importance_scores']).reshape(self.grid_size)
                
                ax = axes[i]
                sns.heatmap(grid, annot=True, fmt='d', cmap='YlOrRd',
                           cbar=False, ax=ax, linewidths=0.5)
                
                short_name = img_name[:20] + '...' if len(img_name) > 20 else img_name
                ax.set_title(f"{short_name}\n1 detection, max={data['max_importance']}", 
                            fontsize=9)
                ax.set_xlabel('Column')
                ax.set_ylabel('Row')
            
            plt.tight_layout()
            vis_path = os.path.join(output_folder, 'single_detection_examples.png')
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved single detection visualizations to: {vis_path}")
    
    def visualize_multi_detection_examples(self, multi_det_examples, output_folder):
        """Visualize importance scores for multi-detection examples"""
        if not multi_det_examples:
            return
        
        num_examples = min(5, len(multi_det_examples))
        
        fig, axes = plt.subplots(1, num_examples, figsize=(4*num_examples, 4))
        if num_examples == 1:
            axes = [axes]
        
        for i, (img_name, data) in enumerate(multi_det_examples[:num_examples]):
            grid = np.array(data['importance_scores']).reshape(self.grid_size)
            
            ax = axes[i]
            sns.heatmap(grid, annot=True, fmt='d', cmap='YlOrRd',
                       cbar=False, ax=ax, linewidths=0.5, vmin=0, vmax=max(2, grid.max()))
            
            short_name = img_name[:15] + '...' if len(img_name) > 15 else img_name
            ax.set_title(f"{short_name}\n{data['num_detections']} det(s), max={data['max_importance']}", 
                        fontsize=9)
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
        
        plt.tight_layout()
        vis_path = os.path.join(output_folder, 'multi_detection_examples.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved multi-detection visualizations to: {vis_path}")


def main():
    """Main execution"""
    print("="*60)
    print("GENERATING TRAINING DATA")
    print("="*60)
    
    # Paths
    results_path = '/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth'
    dataset_root = '/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train'
    output_folder = './results_vis/training_data_output'
    
    # Load results
    print("\nLoading detection results...")
    results = torch.load(results_path)
    print(f"✓ Loaded {len(results)} samples")
    
    # Build image mapping
    print("\nBuilding image index...")
    synsets = sorted([d for d in os.listdir(dataset_root) if d.startswith('n')])
    
    all_images = []
    for synset in synsets:
        synset_folder = os.path.join(dataset_root, synset)
        images = sorted(glob.glob(os.path.join(synset_folder, '*.JPEG')))
        all_images.extend(images)
    
    index_to_image = {i: img_path for i, img_path in enumerate(all_images)}
    print(f"✓ Found {len(all_images)} images")
    
    # Initialize generator (14x14 grid for 224px images with 16px patches)
    generator = TrainingDataGenerator(grid_size=(14, 14))
    
    # Process all images
    print("\n" + "="*60)
    print("PROCESSING IMAGES")
    print("="*60)
    all_data = generator.process_all_images(results, index_to_image)
    
    if not all_data:
        print("\n✗ Error: No images processed!")
        return
    
    # Save training data
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    training_path, complete_path = generator.save_training_data(all_data, output_folder)
    
    # Analyze distribution
    multi_det_images = generator.analyze_importance_distribution(all_data)
    
    # Print multi-detection examples
    multi_det_examples = generator.print_multi_detection_examples(all_data, num_examples=5)
    
    # Visualize examples
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    generator.visualize_examples(all_data, output_folder, num_examples=5)
    generator.visualize_multi_detection_examples(multi_det_examples, output_folder)
    
    # Show example (single detection)
    print("\n" + "="*60)
    print("SINGLE DETECTION EXAMPLE")
    print("="*60)
    single_det = [data for data in all_data.values() if data['num_detections'] == 1]
    if single_det:
        example_data = single_det[0]
        print(f"\nImage: {example_data['image_name']}")
        print(f"  Path: {example_data['image_path']}")
        print(f"  Class ID: {example_data['class_id']}")
        print(f"  Num detections: {example_data['num_detections']}")
        print(f"  Importance scores (first 20): {example_data['importance_scores'][:20]}")
        print(f"  Max importance: {example_data['max_importance']}")
        print(f"  Mean importance: {example_data['mean_importance']:.3f}")
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  - Training data: {training_path}")
    print(f"  - Complete data: {complete_path}")
    print(f"  - Single detection viz: {output_folder}/single_detection_examples.png")
    print(f"  - Multi detection viz: {output_folder}/multi_detection_examples.png")
    print(f"\nUse this file for training:")
    print(f"  {training_path}")
    print(f"\nNext step:")
    print(f"  bash run_pretrain_importance.sh")


if __name__ == "__main__":
    main()
