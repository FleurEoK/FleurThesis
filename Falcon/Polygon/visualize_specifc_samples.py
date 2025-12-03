#!/usr/bin/env python3
"""
Visualize polygons and importance scores for specific samples
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

# Samples to visualize
SAMPLE_IDS = [
    'n01440764_4409.JPEG',
    'n01440764_5456.JPEG', 
    'n01440764_7160.JPEG',
    'n01443537_596.JPEG'
]

def load_data(complete_data_path):
    """Load the complete data JSON"""
    print(f"Loading data from: {complete_data_path}")
    with open(complete_data_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} images\n")
    return data

def visualize_sample(img_name, data, save_path):
    """
    Create comprehensive visualization for one sample:
    - Grid with importance scores (heatmap)
    - Detection polygons overlaid on grid
    - Individual detection polygons separate
    """
    fig = plt.figure(figsize=(18, 6))
    
    # Extract data
    importance_scores = np.array(data['importance_scores']).reshape(14, 14)
    num_detections = data['num_detections']
    max_importance = data['max_importance']
    mean_importance = data['mean_importance']
    class_id = data['class_id']
    
    # Main title
    fig.suptitle(f'{img_name} | Class: {class_id} | Detections: {num_detections} | Max Score: {max_importance}',
                 fontsize=14, fontweight='bold')
    
    # ===== Plot 1: Importance Score Heatmap =====
    ax1 = plt.subplot(131)
    
    # Create heatmap
    sns.heatmap(importance_scores, annot=True, fmt='d', cmap='YlOrRd',
                cbar=True, ax=ax1, linewidths=0.5, linecolor='gray',
                vmin=0, vmax=max(importance_scores.max(), 2),
                cbar_kws={'label': 'Overlap Count'})
    
    ax1.set_title(f'Importance Scores (14×14 Grid)\nMax={max_importance}, Mean={mean_importance:.2f}',
                  fontweight='bold', fontsize=10)
    ax1.set_xlabel('Column (Patch X)', fontsize=9)
    ax1.set_ylabel('Row (Patch Y)', fontsize=9)
    
    # ===== Plot 2: Polygons on Grid =====
    ax2 = plt.subplot(132)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.set_title('Detection Polygons on Grid', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Normalized X', fontsize=9)
    ax2.set_ylabel('Normalized Y', fontsize=9)
    
    # Draw 14x14 grid
    grid_size = 14
    cell_size = 1.0 / grid_size
    
    for i in range(grid_size + 1):
        ax2.axhline(y=i * cell_size, color='gray', linewidth=0.5, alpha=0.4)
        ax2.axvline(x=i * cell_size, color='gray', linewidth=0.5, alpha=0.4)
    
    # Color cells by importance score
    for row in range(14):
        for col in range(14):
            score = importance_scores[row, col]
            if score > 0:
                x_min = col * cell_size
                y_min = row * cell_size
                
                if score >= 2:
                    color = 'red'
                    alpha = 0.3
                elif score == 1:
                    color = 'orange'
                    alpha = 0.2
                else:
                    continue
                
                rect = plt.Rectangle((x_min, y_min), cell_size, cell_size,
                                    facecolor=color, alpha=alpha, edgecolor='none')
                ax2.add_patch(rect)
    
    # Draw detection polygons
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, detection in enumerate(data['individual_detections']):
        det_id = detection['detection_id']
        objectness = detection['objectness_score']
        poly_geom = detection['polygon_geojson']
        
        color = colors[idx % len(colors)]
        
        if poly_geom['type'] == 'Polygon':
            coords = np.array(poly_geom['coordinates'][0])
            polygon_patch = mpatches.Polygon(
                coords, closed=True,
                edgecolor=color, facecolor='none',
                linewidth=2.5, linestyle='-',
                label=f'Det{det_id} (obj={objectness:.2f})'
            )
            ax2.add_patch(polygon_patch)
    
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # ===== Plot 3: Individual Detection Polygons =====
    ax3 = plt.subplot(133)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.invert_yaxis()
    ax3.set_title(f'Individual Detections (n={num_detections})', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Normalized X', fontsize=9)
    ax3.set_ylabel('Normalized Y', fontsize=9)
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    
    # Draw individual detection polygons with fill
    for idx, detection in enumerate(data['individual_detections']):
        det_id = detection['detection_id']
        objectness = detection['objectness_score']
        poly_geom = detection['polygon_geojson']
        bbox = detection['bbox_xyxy_pixels']
        
        color = colors[idx % len(colors)]
        
        if poly_geom['type'] == 'Polygon':
            coords = np.array(poly_geom['coordinates'][0])
            polygon_patch = mpatches.Polygon(
                coords, closed=True,
                edgecolor=color, facecolor=color,
                alpha=0.35, linewidth=2,
                label=f'Det{det_id}: obj={objectness:.3f}\n  bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]'
            )
            ax3.add_patch(polygon_patch)
    
    ax3.legend(loc='upper right', fontsize=7, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def print_sample_details(img_name, data):
    """Print detailed statistics for a sample"""
    print(f"\n{'='*60}")
    print(f"Sample: {img_name}")
    print(f"{'='*60}")
    
    print(f"\nBasic Info:")
    print(f"  Path: {data['image_path']}")
    print(f"  Image size: {data['image_size']}")
    print(f"  Class ID: {data['class_id']}")
    print(f"  Number of detections: {data['num_detections']}")
    
    print(f"\nImportance Scores:")
    print(f"  Max: {data['max_importance']}")
    print(f"  Mean: {data['mean_importance']:.3f}")
    print(f"  Total overlaps: {data['total_overlaps']}")
    
    # Grid statistics
    importance_array = np.array(data['importance_scores'])
    print(f"\nGrid Cell Statistics:")
    for score in range(int(importance_array.max()) + 1):
        count = np.sum(importance_array == score)
        percentage = 100 * count / len(importance_array)
        print(f"  Cells with score {score}: {count} ({percentage:.1f}%)")
    
    print(f"\nDetection Details:")
    for det in data['individual_detections']:
        bbox = det['bbox_xyxy_pixels']
        print(f"  Detection {det['detection_id']}:")
        print(f"    Objectness: {det['objectness_score']:.4f}")
        print(f"    BBox (pixels): [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        
        # Calculate bbox size
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        print(f"    Size: {width:.0f}×{height:.0f} px (area={area:.0f} px²)")
    
    # High importance cells
    if data['max_importance'] >= 2:
        print(f"\nHigh Importance Cells (score ≥ 2):")
        for idx, score in enumerate(data['importance_scores']):
            if score >= 2:
                row = idx // 14
                col = idx % 14
                print(f"  Cell[{row:2d},{col:2d}] (token {idx:3d}): score={score}")

def main():
    """Main execution"""
    print("="*60)
    print("VISUALIZING SPECIFIC SAMPLES")
    print("="*60)
    
    # Paths - update if needed
    complete_data_path = './results_vis/training_data_output/complete_data.json'
    output_dir = './results_vis/specific_samples'
    
    # Alternative paths to try
    alt_paths = [
        '/home/20204130/Falcon/FALcon-main/results_vis/training_data_output/complete_data.json',
        'complete_data.json',
        'results_vis/training_data_output/complete_data.json'
    ]
    
    # Try to load data
    data = None
    for path in [complete_data_path] + alt_paths:
        try:
            data = load_data(path)
            complete_data_path = path
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        print("\n✗ ERROR: Could not find complete_data.json")
        print("\nSearched in:")
        for path in [complete_data_path] + alt_paths:
            print(f"  - {path}")
        print("\nPlease update the path in the script or run from the correct directory.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Process each sample
    found_samples = []
    missing_samples = []
    
    for sample_id in SAMPLE_IDS:
        # Try with and without .JPEG extension
        sample_keys = [sample_id, sample_id.replace('.JPEG', ''), sample_id + '.JPEG']
        
        found = False
        for key in sample_keys:
            if key in data:
                found_samples.append(key)
                found = True
                
                # Print details
                print_sample_details(key, data[key])
                
                # Create visualization
                save_path = f"{output_dir}/{key.replace('.JPEG', '')}_visualization.png"
                visualize_sample(key, data[key], save_path)
                
                break
        
        if not found:
            print(f"\n⚠ WARNING: {sample_id} not found in data")
            missing_samples.append(sample_id)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✓ Found and visualized: {len(found_samples)} samples")
    for sample in found_samples:
        print(f"  ✓ {sample}")
    
    if missing_samples:
        print(f"\n✗ Missing: {len(missing_samples)} samples")
        for sample in missing_samples:
            print(f"  ✗ {sample}")
        
        print(f"\nAvailable samples (first 10):")
        for i, key in enumerate(list(data.keys())[:10]):
            print(f"  {key}")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"\nVisualization files:")
    for sample in found_samples:
        filename = sample.replace('.JPEG', '') + '_visualization.png'
        print(f"  - {filename}")

if __name__ == '__main__':
    main()