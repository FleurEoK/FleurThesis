import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import glob
import numpy as np
import json
import geopandas as gpd
from shapely.geometry import box, Polygon, mapping, shape
from shapely.ops import unary_union
import pandas as pd
import seaborn as sns

# Create results folder
results_vis_folder = 'results_vis'
os.makedirs(results_vis_folder, exist_ok=True)
print(f"Created/using folder: {results_vis_folder}")

# Load results
results = torch.load('/home/20204130/Falcon/FALcon-main/results/imagenet/wsol_method_PSOL/trained_on_train_split/arch_vgg16_pretrained_init_normalization_none_seed_16/imagenet_localization_only_from1to6500_filtered.pth')

dataset_root = '/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train'

# Build image mapping (same as before)
print("\nBuilding index to image path mapping...")
print("=" * 60)

synsets = sorted([d for d in os.listdir(dataset_root) if d.startswith('n')])
print(f"Found {len(synsets)} synset folders: {synsets}")

class_to_synset = {i: synset for i, synset in enumerate(synsets)}
print("\nClass to Synset mapping:")
for class_id, synset in class_to_synset.items():
    print(f"  Class {class_id} -> {synset}")

all_images = []
for synset in synsets:
    synset_folder = os.path.join(dataset_root, synset)
    images = sorted(glob.glob(os.path.join(synset_folder, '*.JPEG')))
    for img_path in images:
        all_images.append(img_path)

print(f"\nTotal images found: {len(all_images)}")
index_to_image = {i: img_path for i, img_path in enumerate(all_images)}


class PolygonGridAnalyzer:
    """
    Analyzer that creates polygons for each image and stores them in JSON
    """
    def __init__(self, image_size=(512, 512), grid_size=(5, 5)):
        self.image_size = image_size
        self.grid_size = grid_size
        self.grid_width = 1.0 / grid_size[0]
        self.grid_height = 1.0 / grid_size[1]
    
    def create_grid_geodataframe(self):
        """
        Create a GeoDataFrame with grid cells
        
        Returns:
            GeoDataFrame with grid cell polygons and identifiers
        """
        cells = []
        
        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                # Calculate cell boundaries in normalized coordinates (0-1)
                x_min = col * self.grid_width
                y_min = row * self.grid_height
                x_max = (col + 1) * self.grid_width
                y_max = (row + 1) * self.grid_height
                
                # Create grid cell polygon
                cell_polygon = box(x_min, y_min, x_max, y_max)
                
                cells.append({
                    'row': row,
                    'col': col,
                    'cell_id': f'r{row}_c{col}',
                    'geometry': cell_polygon
                })
        
        gdf = gpd.GeoDataFrame(cells, crs='EPSG:4326')
        return gdf
    
    def bbox_to_polygon(self, bbox_xyxy, image_size):
        """
        Convert bounding box from pixel coordinates to normalized polygon
        
        Args:
            bbox_xyxy: tensor or array [x1, y1, x2, y2] in pixels
            image_size: tuple (width, height)
            
        Returns:
            Shapely Polygon in normalized coordinates
        """
        if hasattr(bbox_xyxy, 'cpu'):
            bbox = bbox_xyxy.cpu().numpy()
        else:
            bbox = np.array(bbox_xyxy)
        
        x1, y1, x2, y2 = bbox
        img_width, img_height = image_size
        
        # Normalize to 0-1 range
        x1_norm = float(x1 / img_width)
        y1_norm = float(y1 / img_height)
        x2_norm = float(x2 / img_width)
        y2_norm = float(y2 / img_height)
        
        return box(x1_norm, y1_norm, x2_norm, y2_norm)
    
    def create_unified_polygon(self, sample, img_path):
        """
        Create a single unified polygon from all detections in an image
        
        Args:
            sample: sample data from results
            img_path: path to the image
            
        Returns:
            tuple: (unified Shapely polygon, list of individual polygons, metadata)
        """
        # Load image to get dimensions
        img = Image.open(img_path)
        image_size = img.size
        
        # Get all detection polygons
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
        
        # Create unified polygon (union of all detections)
        if len(detection_polygons) == 1:
            unified_polygon = detection_polygons[0]
        else:
            unified_polygon = unary_union(detection_polygons)
        
        # Get ground truth info
        gt_bbox = sample['gt_bboxes'][0].cpu().numpy()
        gt_polygon = box(
            float(gt_bbox[0] / image_size[0]),
            float(gt_bbox[1] / image_size[1]),
            float((gt_bbox[0] + gt_bbox[2]) / image_size[0]),
            float((gt_bbox[1] + gt_bbox[3]) / image_size[1])
        )
        
        metadata = {
            'image_path': img_path,
            'image_name': os.path.basename(img_path),
            'image_size': list(image_size),
            'num_detections': len(detection_polygons),
            'class_id': int(sample['gt_synsets'][0]),
            'unified_polygon_area': float(unified_polygon.area),
            'ground_truth_polygon': mapping(gt_polygon),
            'ground_truth_bbox_pixels': gt_bbox.tolist()
        }
        
        return unified_polygon, detection_polygons, detection_info, metadata
    
    def save_polygons_to_json(self, results, index_to_image, output_file):
        """
        Save all image polygons to a single JSON file
        
        Args:
            results: dictionary from .pth file
            index_to_image: mapping from sample index to image path
            output_file: path to output JSON file
            
        Returns:
            Dictionary with all polygon data
        """
        all_polygons_data = {}
        
        print(f"\nCreating polygons for {len(results)} images...")
        
        processed = 0
        for sample_idx, sample in results.items():
            if sample_idx not in index_to_image:
                continue
            
            img_path = index_to_image[sample_idx]
            
            try:
                unified_poly, individual_polys, detection_info, metadata = \
                    self.create_unified_polygon(sample, img_path)
                
                # Convert to GeoJSON format
                polygon_data = {
                    'metadata': metadata,
                    'unified_polygon': mapping(unified_poly),
                    'individual_detections': detection_info
                }
                
                # Use image name as key
                img_name = os.path.basename(img_path)
                all_polygons_data[img_name] = polygon_data
                
                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{len(results)} images")
                    
            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                continue
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(all_polygons_data, f, indent=2)
        
        print(f"\nSaved {processed} polygons to {output_file}")
        return all_polygons_data
    
    def save_polygons_by_class(self, results, index_to_image, class_to_synset, output_folder):
        """
        Save polygons grouped by class in separate JSON files
        
        Args:
            results: dictionary from .pth file
            index_to_image: mapping from sample index to image path
            class_to_synset: mapping from class ID to synset name
            output_folder: folder to save class-specific JSON files
            
        Returns:
            Dictionary mapping class_id to polygon data
        """
        os.makedirs(output_folder, exist_ok=True)
        
        class_polygons = {}
        
        print(f"\nCreating polygons grouped by class...")
        
        for sample_idx, sample in results.items():
            if sample_idx not in index_to_image:
                continue
            
            img_path = index_to_image[sample_idx]
            class_id = int(sample['gt_synsets'][0])
            
            try:
                unified_poly, individual_polys, detection_info, metadata = \
                    self.create_unified_polygon(sample, img_path)
                
                # Convert to GeoJSON format
                polygon_data = {
                    'metadata': metadata,
                    'unified_polygon': mapping(unified_poly),
                    'individual_detections': detection_info
                }
                
                # Group by class
                if class_id not in class_polygons:
                    class_polygons[class_id] = {}
                
                img_name = os.path.basename(img_path)
                class_polygons[class_id][img_name] = polygon_data
                
            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                continue
        
        # Save each class to separate JSON file
        for class_id, polygons in class_polygons.items():
            synset = class_to_synset.get(class_id, 'unknown')
            output_file = os.path.join(output_folder, f'class_{class_id}_{synset}_polygons.json')
            
            with open(output_file, 'w') as f:
                json.dump(polygons, f, indent=2)
            
            print(f"  Saved Class {class_id} ({synset}): {len(polygons)} images to {output_file}")
        
        return class_polygons
    
    def save_individual_polygon_files(self, results, index_to_image, output_folder, max_files=10):
        """
        Save individual JSON files (one per image) - for first N images as examples
        
        Args:
            results: dictionary from .pth file
            index_to_image: mapping from sample index to image path
            output_folder: folder to save individual JSON files
            max_files: maximum number of individual files to save
        """
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\nSaving individual polygon files (first {max_files} images)...")
        
        saved = 0
        for sample_idx, sample in results.items():
            if saved >= max_files:
                break
            
            if sample_idx not in index_to_image:
                continue
            
            img_path = index_to_image[sample_idx]
            
            try:
                unified_poly, individual_polys, detection_info, metadata = \
                    self.create_unified_polygon(sample, img_path)
                
                # Convert to GeoJSON format
                polygon_data = {
                    'metadata': metadata,
                    'unified_polygon': mapping(unified_poly),
                    'individual_detections': detection_info
                }
                
                # Create safe filename
                img_name = os.path.basename(img_path).replace('.JPEG', '')
                safe_name = "".join(c for c in img_name if c.isalnum() or c in ('-', '_'))
                
                output_file = os.path.join(output_folder, f'{safe_name}_polygon.json')
                
                with open(output_file, 'w') as f:
                    json.dump(polygon_data, f, indent=2)
                
                print(f"  Saved: {output_file}")
                saved += 1
                
            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                continue
    
    def process_single_image_with_grid(self, sample, img_path, grid_gdf):
        """
        Process a single image and calculate overlap counts for each grid cell
        
        Args:
            sample: sample data from results
            img_path: path to the image
            grid_gdf: GeoDataFrame with grid cells
            
        Returns:
            GeoDataFrame with overlap counts for this image
        """
        # Load image to get dimensions
        img = Image.open(img_path)
        image_size = img.size
        
        # Get all detection polygons
        detection_polygons = []
        for det in sample['detections']:
            bbox_xyxy = det['bbox_xyxy']
            poly = self.bbox_to_polygon(bbox_xyxy, image_size)
            detection_polygons.append(poly)
        
        # Calculate overlap count for each grid cell
        overlap_counts = []
        
        for idx, cell_row in grid_gdf.iterrows():
            cell_poly = cell_row['geometry']
            
            # Count how many detection polygons intersect this cell
            overlap_count = 0
            for det_poly in detection_polygons:
                if cell_poly.intersects(det_poly):
                    overlap_count += 1
            
            overlap_counts.append(overlap_count)
        
        # Create result GeoDataFrame
        result_gdf = grid_gdf.copy()
        result_gdf['overlap_count'] = overlap_counts
        result_gdf['image_path'] = img_path
        result_gdf['num_detections'] = len(detection_polygons)
        
        return result_gdf
    
    def process_all_images(self, results, index_to_image):
        """
        Process all images and create cumulative GeoDataFrame
        
        Args:
            results: dictionary from .pth file
            index_to_image: mapping from sample index to image path
            
        Returns:
            tuple: (list of individual GeoDataFrames, cumulative GeoDataFrame)
        """
        # Create base grid
        grid_gdf = self.create_grid_geodataframe()
        
        individual_gdfs = []
        cumulative_counts = np.zeros(len(grid_gdf), dtype=int)
        
        print(f"\nProcessing {len(results)} images for grid overlap analysis...")
        
        processed = 0
        for sample_idx, sample in results.items():
            if sample_idx not in index_to_image:
                continue
            
            img_path = index_to_image[sample_idx]
            
            try:
                # Process single image
                img_gdf = self.process_single_image_with_grid(sample, img_path, grid_gdf)
                individual_gdfs.append(img_gdf)
                
                # Add to cumulative counts
                cumulative_counts += img_gdf['overlap_count'].values
                
                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{len(results)} images")
                    
            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                continue
        
        # Create cumulative GeoDataFrame
        cumulative_gdf = grid_gdf.copy()
        cumulative_gdf['cumulative_overlap_count'] = cumulative_counts
        cumulative_gdf['total_images'] = processed
        
        print(f"\nSuccessfully processed {processed} images")
        
        return individual_gdfs, cumulative_gdf
    
    def process_by_class(self, results, index_to_image, class_to_synset):
        """
        Process images grouped by class
        
        Args:
            results: dictionary from .pth file
            index_to_image: mapping from sample index to image path
            class_to_synset: mapping from class ID to synset name
            
        Returns:
            Dictionary mapping class_id to cumulative GeoDataFrame
        """
        grid_gdf = self.create_grid_geodataframe()
        
        class_gdfs = {}
        class_counts = {}
        class_image_counts = {}
        
        print(f"\nProcessing images by class for grid analysis...")
        
        for sample_idx, sample in results.items():
            if sample_idx not in index_to_image:
                continue
            
            img_path = index_to_image[sample_idx]
            class_id = sample['gt_synsets'][0]
            
            try:
                # Process single image
                img_gdf = self.process_single_image_with_grid(sample, img_path, grid_gdf)
                
                # Initialize class data if needed
                if class_id not in class_counts:
                    class_counts[class_id] = np.zeros(len(grid_gdf), dtype=int)
                    class_image_counts[class_id] = 0
                
                # Add to class cumulative counts
                class_counts[class_id] += img_gdf['overlap_count'].values
                class_image_counts[class_id] += 1
                
            except Exception as e:
                print(f"  Error processing sample {sample_idx}: {e}")
                continue
        
        # Create GeoDataFrames for each class
        for class_id, counts in class_counts.items():
            class_gdf = grid_gdf.copy()
            class_gdf['cumulative_overlap_count'] = counts
            class_gdf['class_id'] = class_id
            class_gdf['synset'] = class_to_synset.get(class_id, 'unknown')
            class_gdf['num_images'] = class_image_counts[class_id]
            
            class_gdfs[class_id] = class_gdf
            
            print(f"  Class {class_id} ({class_to_synset.get(class_id, 'unknown')}): "
                  f"{class_image_counts[class_id]} images")
        
        return class_gdfs
    
    def save_geodataframe(self, gdf, output_path, format='geojson'):
        """
        Save GeoDataFrame to file
        
        Args:
            gdf: GeoDataFrame to save
            output_path: path for output file (without extension)
            format: 'geojson', 'shapefile', or 'geoparquet'
        """
        if format == 'geojson':
            gdf.to_file(f"{output_path}.geojson", driver='GeoJSON')
        elif format == 'shapefile':
            gdf.to_file(f"{output_path}.shp")
        elif format == 'geoparquet':
            gdf.to_parquet(f"{output_path}.parquet")
        
        print(f"Saved GeoDataFrame to {output_path}.{format}")
    
    def visualize_geodataframe(self, gdf, count_column='overlap_count', 
                              title='Grid Overlap Counts', output_path=None):
        """
        Visualize GeoDataFrame with overlap counts
        
        Args:
            gdf: GeoDataFrame to visualize
            count_column: name of column with counts
            title: plot title
            output_path: if provided, save plot to this path
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot grid cells colored by overlap count
        gdf.plot(column=count_column, ax=ax, legend=True,
                legend_kwds={'label': 'Overlap Count'},
                cmap='YlOrRd', edgecolor='black', linewidth=0.5)
        
        # Add cell annotations
        for idx, row in gdf.iterrows():
            centroid = row['geometry'].centroid
            count = row[count_column]
            ax.text(centroid.x, centroid.y, str(count),
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {output_path}")
        else:
            plt.show()
    
    def create_grid_matrix(self, gdf, count_column='overlap_count'):
        """
        Convert GeoDataFrame to grid matrix for heatmap visualization
        
        Args:
            gdf: GeoDataFrame with grid cells
            count_column: column name with counts
            
        Returns:
            2D numpy array
        """
        grid_matrix = np.zeros(self.grid_size, dtype=int)
        
        for idx, row in gdf.iterrows():
            r = row['row']
            c = row['col']
            grid_matrix[r, c] = row[count_column]
        
        return grid_matrix


# Initialize the analyzer
print("\n" + "=" * 60)
print("POLYGON EXTRACTION AND GRID ANALYSIS")
print("=" * 60)

analyzer = PolygonGridAnalyzer(image_size=(512, 512), grid_size=(5, 5))

# Create output folders
polygon_output_folder = os.path.join(results_vis_folder, 'polygons')
os.makedirs(polygon_output_folder, exist_ok=True)

# 1. SAVE ALL POLYGONS TO SINGLE JSON FILE
print("\n" + "=" * 60)
print("STEP 1: SAVING ALL POLYGONS TO SINGLE JSON")
print("=" * 60)

all_polygons_json = os.path.join(polygon_output_folder, 'all_image_polygons.json')
all_polygons_data = analyzer.save_polygons_to_json(results, index_to_image, all_polygons_json)

print(f"\nSaved all polygons to: {all_polygons_json}")
print(f"Total images: {len(all_polygons_data)}")

# Show example of first polygon
first_img_name = list(all_polygons_data.keys())[0]
first_polygon_data = all_polygons_data[first_img_name]
print(f"\nExample polygon data for {first_img_name}:")
print(f"  Metadata: {first_polygon_data['metadata']}")
print(f"  Unified polygon type: {first_polygon_data['unified_polygon']['type']}")
print(f"  Number of individual detections: {len(first_polygon_data['individual_detections'])}")


# 2. SAVE POLYGONS GROUPED BY CLASS
print("\n" + "=" * 60)
print("STEP 2: SAVING POLYGONS BY CLASS")
print("=" * 60)

class_polygons_folder = os.path.join(polygon_output_folder, 'by_class')
class_polygons = analyzer.save_polygons_by_class(results, index_to_image, 
                                                  class_to_synset, class_polygons_folder)


# 3. SAVE INDIVIDUAL POLYGON FILES (FIRST 10 AS EXAMPLES)
print("\n" + "=" * 60)
print("STEP 3: SAVING INDIVIDUAL POLYGON FILES (EXAMPLES)")
print("=" * 60)

individual_polygons_folder = os.path.join(polygon_output_folder, 'individual_examples')
analyzer.save_individual_polygon_files(results, index_to_image, 
                                       individual_polygons_folder, max_files=10)


# 4. GRID ANALYSIS (GEODATAFRAMES)
print("\n" + "=" * 60)
print("STEP 4: GRID ANALYSIS WITH GEODATAFRAMES")
print("=" * 60)

geo_output_folder = os.path.join(results_vis_folder, 'geopandas_grids')
os.makedirs(geo_output_folder, exist_ok=True)

# Process all images for grid analysis
individual_gdfs, cumulative_gdf = analyzer.process_all_images(results, index_to_image)

# Save cumulative GeoDataFrame
cumulative_output = os.path.join(geo_output_folder, 'cumulative_grid')
analyzer.save_geodataframe(cumulative_gdf, cumulative_output, format='geojson')
analyzer.save_geodataframe(cumulative_gdf, cumulative_output, format='geoparquet')

# Save as CSV for easy viewing
cumulative_csv = cumulative_gdf.drop(columns=['geometry'])
cumulative_csv.to_csv(f"{cumulative_output}.csv", index=False)

# Visualize cumulative grid
print("\nCumulative Grid Statistics:")
print(cumulative_gdf[['cell_id', 'row', 'col', 'cumulative_overlap_count']])
print(f"\nTotal overlap count: {cumulative_gdf['cumulative_overlap_count'].sum()}")
print(f"Max overlap in a cell: {cumulative_gdf['cumulative_overlap_count'].max()}")
print(f"Mean overlap per cell: {cumulative_gdf['cumulative_overlap_count'].mean():.2f}")

# Create visualizations
vis_path = os.path.join(geo_output_folder, 'cumulative_grid_visualization.png')
analyzer.visualize_geodataframe(cumulative_gdf, 
                               count_column='cumulative_overlap_count',
                               title='Cumulative Grid - All Images',
                               output_path=vis_path)

# Create heatmap version
grid_matrix = analyzer.create_grid_matrix(cumulative_gdf, 'cumulative_overlap_count')
plt.figure(figsize=(8, 6))

sns.heatmap(grid_matrix, annot=True, fmt='d', cmap='YlOrRd',
           cbar_kws={'label': 'Cumulative Overlap Count'})
plt.title('Cumulative Grid Heatmap - All Images')
plt.xlabel('Column')
plt.ylabel('Row')
plt.tight_layout()
heatmap_path = os.path.join(geo_output_folder, 'cumulative_grid_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved heatmap to {heatmap_path}")


# Process by class
print("\n" + "=" * 60)
print("STEP 5: CLASS-SPECIFIC GRID ANALYSIS")
print("=" * 60)

class_gdfs = analyzer.process_by_class(results, index_to_image, class_to_synset)

# Save class-specific GeoDataFrames
class_folder = os.path.join(geo_output_folder, 'by_class')
os.makedirs(class_folder, exist_ok=True)

for class_id, class_gdf in class_gdfs.items():
    synset = class_to_synset.get(class_id, 'unknown')
    
    output_base = os.path.join(class_folder, f'class_{class_id}_{synset}')
    
    # Save as GeoJSON and GeoParquet
    analyzer.save_geodataframe(class_gdf, output_base, format='geojson')
    analyzer.save_geodataframe(class_gdf, output_base, format='geoparquet')
    
    # Save as CSV
    class_csv = class_gdf.drop(columns=['geometry'])
    class_csv.to_csv(f"{output_base}.csv", index=False)
    
    # Visualize
    vis_path = f"{output_base}_visualization.png"
    num_images = class_gdf['num_images'].iloc[0]
    analyzer.visualize_geodataframe(class_gdf,
                                   count_column='cumulative_overlap_count',
                                   title=f'Class {class_id} ({synset})\n{num_images} images',
                                   output_path=vis_path)
    
    print(f"\nClass {class_id} ({synset}) statistics:")
    print(f"  Total images: {num_images}")
    print(f"  Total overlap count: {class_gdf['cumulative_overlap_count'].sum()}")
    print(f"  Max overlap in a cell: {class_gdf['cumulative_overlap_count'].max()}")


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nPolygon JSON files saved to: {polygon_output_folder}/")
print("  - all_image_polygons.json (all images in one file)")
print(f"  - by_class/ (separate JSON per class)")
print(f"  - individual_examples/ (10 individual image JSONs)")
print(f"\nGeoDataFrame grid analysis saved to: {geo_output_folder}/")
print("  - cumulative_grid.geojson/geoparquet/csv")
print("  - by_class/ (per-class grid GeoDataFrames)")
print("=" * 60)