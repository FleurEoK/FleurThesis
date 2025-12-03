import json
import os

pseudo_bbox_dir = '/home/20204130/Falcon/FALcon-main/PSOL/results/ImageNet_train_subset/'
output_dir = '/home/20204130/Falcon/FALcon-main/PSOL/results/processed_bboxes/'
os.makedirs(output_dir, exist_ok=True)

classes = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']

for cls_name in classes:
    json_file = os.path.join(pseudo_bbox_dir, f"{cls_name}_bbox.json")
    
    with open(json_file, 'r') as f:
        original_data = json.load(f)
    
    # Create new structure: {image_name: bbox_data}
    simplified_data = {}
    
    for full_path, bbox in original_data.items():
        # Extract just the filename without extension
        # e.g., "/long/path/n01440764_10026.JPEG" -> "n01440764_10026"
        filename = os.path.basename(full_path)
        image_name = os.path.splitext(filename)[0]
        
        simplified_data[image_name] = bbox
    
    # Save the simplified version
    output_file = os.path.join(output_dir, f"{cls_name}_bbox_simplified.json")
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)
    
    print(f"Processed {cls_name}: {len(simplified_data)} images")