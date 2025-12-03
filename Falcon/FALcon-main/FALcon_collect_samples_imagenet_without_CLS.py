#!/usr/bin/env python3
"""
Simplified FALcon script for localization only (no classification)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import copy
import argparse
import xml.etree.ElementTree as ET
import json

from FALcon_config_test_as_WSOL import FALcon_config
from FALcon_models_vgg import customizable_VGG as custom_vgg
from utils.utils_dataloaders import get_dataloaders
from AVS_functions import extract_and_resize_glimpses_for_batch, get_grid
from torchvision.ops import nms

# Parse arguments
parser = argparse.ArgumentParser(description='FALcon localization only')
parser.add_argument('--start', default=1, type=int, help='First sample index')
parser.add_argument('--end', default=-1, type=int, help='Last sample index')
args = parser.parse_args()

# Setup
config_3 = FALcon_config
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set seeds
SEED = config_3.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataloader
train_loader = get_dataloaders(config_3, loader_type=config_3.loader_type)
train_loader.dataset.fetch_one_bbox = False

# FALcon model only (no classification model needed)
model_3 = custom_vgg(config_3).to(device)
for p in model_3.parameters():
    p.requires_grad_(False)
model_3.eval()
print("FALcon (localization) model loaded successfully")

# Create save directory
if not os.path.exists(config_3.save_dir): 
    os.makedirs(config_3.save_dir)

# def has_valid_annotation(image_path, pseudo_bbox_dir):
#     """Check if an image has a valid XML annotation"""
#     # Get the image filename and class from the path
#     # Assuming path structure: .../class_name/image_name.JPEG
#     path_parts = image_path.split(os.sep)
#     class_name = path_parts[-2]
#     image_filename = path_parts[-1]
#     image_basename = os.path.splitext(image_filename)[0]
    
#     # Construct XML path
#     xml_file = os.path.join(pseudo_bbox_dir, class_name, f"{image_basename}.xml")
    
#     if not os.path.isfile(xml_file):
#         return False
    
#     try:
#         # Try to parse the XML to make sure it's valid
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         # Check if there are any object annotations
#         objects = root.findall('object')
#         return len(objects) > 0
#     except (ET.ParseError, OSError):
#         return False

# Load all pseudo bbox annotations

# Load simplified pseudo bbox annotations
pseudo_bbox_dir = '/home/20204130/Falcon/FALcon-main/PSOL/results/processed_bboxes/'
classes = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']

print(f"Loading simplified pseudo bbox annotations for {len(classes)} classes...")
pseudo_bboxes = {}
for cls_name in classes:
    json_file = os.path.join(pseudo_bbox_dir, f"{cls_name}_bbox_simplified.json")
    with open(json_file, 'r') as f:
        pseudo_bboxes[cls_name] = json.load(f)
    print(f"Loaded {len(pseudo_bboxes[cls_name])} annotations for {cls_name}")

#creating index to image mapping for easier access
print("\nLoading index to image name mapping...")
index_to_info = {}
train_cls_file = '/home/20204130/imagenet-subset/ILSVRC/ImageSets/CLS-LOC/train_cls.txt'
with open(train_cls_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        path = parts[0]  # e.g., "n01440764/n01440764_10026"
        idx = int(parts[1]) - 1  # Convert to 0-indexed
        class_name, image_name = path.split('/')
        index_to_info[idx] = {'class': class_name, 'image': image_name}
print(f"Loaded mapping for {len(index_to_info)} images")


def has_pseudo_bbox(image_identifier, class_name, pseudo_bboxes_dict):
    """
    Check if an image has pseudo bbox
    image_identifier: e.g., "n01440764_10026" 
    class_name: e.g., "n01440764"
    """
    if class_name not in pseudo_bboxes_dict:
        return False
    return image_identifier in pseudo_bboxes_dict[class_name]

def FALcon_from_init_glimpse_loc(config, locmodel, input_image, init_glimpse_loc, switch_location_th):
    """Run FALcon foveation from initial glimpse location"""
    foveation_progress_per_glimpse_loc = []
    
    glimpses_locs_dims = torch.zeros((input_image.shape[0], 4), dtype=torch.int).to(input_image.device)
    glimpses_locs_dims[:, 0] = init_glimpse_loc[0] + 0.5 - (config.glimpse_size_grid[0]/2.0)
    glimpses_locs_dims[:, 1] = init_glimpse_loc[1] + 0.5 - (config.glimpse_size_grid[1]/2.0)
    glimpses_locs_dims[:, 2] = config.glimpse_size_init[0]
    glimpses_locs_dims[:, 3] = config.glimpse_size_init[1]
    foveation_progress_per_glimpse_loc.append(glimpses_locs_dims.clone().detach())
    
    for g in range(config.num_glimpses):
        glimpses_extracted_resized = extract_and_resize_glimpses_for_batch(
            input_image, glimpses_locs_dims,
            config.glimpse_size_fixed[1], config.glimpse_size_fixed[0])
            
        glimpses_change_predictions, switch_location_predictions = locmodel(glimpses_extracted_resized)

        switch_location_probability = torch.sigmoid(switch_location_predictions.clone().detach()).squeeze(1)  
        switch_location_actions = (switch_location_probability >= switch_location_th).item()
        if switch_location_actions:
            break

        # Update glimpse based on predictions
        x_min_current = glimpses_locs_dims[:, 0].clone().detach()
        x_max_current = (glimpses_locs_dims[:, 0] + glimpses_locs_dims[:, 2]).clone().detach()
        y_min_current = glimpses_locs_dims[:, 1].clone().detach()
        y_max_current = (glimpses_locs_dims[:, 1] + glimpses_locs_dims[:, 3]).clone().detach()

        glimpses_change_probability = torch.sigmoid(glimpses_change_predictions.clone().detach())
        glimpses_change_actions = (glimpses_change_probability >= config.glimpse_change_th)
        
        # Update bounds
        x_min_new = torch.clamp(x_min_current - glimpses_change_actions[:, 0]*config.glimpse_size_step[0], min=0)
        x_max_new = torch.clamp(x_max_current + glimpses_change_actions[:, 1]*config.glimpse_size_step[0], max=input_image.shape[-1])
        y_min_new = torch.clamp(y_min_current - glimpses_change_actions[:, 2]*config.glimpse_size_step[1], min=0)
        y_max_new = torch.clamp(y_max_current + glimpses_change_actions[:, 3]*config.glimpse_size_step[1], max=input_image.shape[-2])
            
        glimpses_locs_dims[:, 0] = x_min_new.clone().detach()
        glimpses_locs_dims[:, 1] = y_min_new.clone().detach()
        glimpses_locs_dims[:, 2] = x_max_new.clone().detach() - glimpses_locs_dims[:, 0]
        glimpses_locs_dims[:, 3] = y_max_new.clone().detach() - glimpses_locs_dims[:, 1]
        foveation_progress_per_glimpse_loc.append(glimpses_locs_dims.clone().detach())

    foveation_results = {}
    foveation_results["final_glimpse_switch_probability"] = switch_location_probability.item()
    foveation_results["final_glimpse_objectness"] = 1.0 - switch_location_probability.item()
    foveation_results["final_glimpse_loc_and_dim"] = copy.deepcopy(glimpses_locs_dims)
    foveation_results["foveation_progress"] = copy.deepcopy(foveation_progress_per_glimpse_loc)
    return foveation_results

# Main processing loop
args.end = len(train_loader.dataset) if args.end == -1 else args.end
collected_samples = {}
skipped_count = 0
processed_count = 0

with torch.no_grad():
    for i in range(args.start-1, args.end, 1):
        # try:
        #     # Load ImageNet data
        #     if config_3.dataset == 'imagenet2013-det' or config_3.dataset == 'imagenet':
        #         # Load the sample
        #         image, target_class, target_bbox = train_loader.dataset[i]
                
        #         # Check if bbox is empty (no annotations)
        #         if isinstance(target_bbox, list) and len(target_bbox) == 0:
        #             skipped_count += 1
        #             if skipped_count % 100 == 0:
        #                 print(f"Skipped {skipped_count} images without annotations so far...")
        #             continue
        #         elif isinstance(target_bbox, torch.Tensor) and target_bbox.numel() == 0:
        #             skipped_count += 1
        #             if skipped_count % 100 == 0:
        #                 print(f"Skipped {skipped_count} images without annotations so far...")
        #             continue
                
        #         image = image.unsqueeze(0).to(device)
        #     else:
        #         raise ValueError(f"Unsupported dataset: {config_3.dataset}. This script supports 'imagenet' and 'imagenet2013-det' only.")
        # except IndexError as e:
        #     print(f"Skipping sample {i} due to bounding box format issue: {e}")
        #     skipped_count += 1
        #     continue


        try:
            # Check if index exists in mapping
            if i not in index_to_info:
                skipped_count += 1
                continue
            
            # Get image identifier and class name
            class_name = index_to_info[i]['class']
            image_name = index_to_info[i]['image']
            
            # Check if this image has a pseudo bbox annotation
            if not has_pseudo_bbox(image_name, class_name, pseudo_bboxes):
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"Skipped {skipped_count} images without pseudo bbox...")
                continue
            
            image, target_class, target_bbox = train_loader.dataset[i]
            image = image.unsqueeze(0).to(device)
            print('target class: ', target_class)
            print('image: ', image)
            print('target_bbox: ', target_bbox)
            
        except Exception as e:
            print(f"Skipping sample {i} due to error: {e}")
            skipped_count += 1
            continue

        # Get all grid cell centers
        all_grid_cells_centers = get_grid(
            (config_3.full_res_img_size[1], config_3.full_res_img_size[0]),
            config_3.glimpse_size_grid, grid_center_coords=True).to(device)

        
        # Foveate from every grid cell
        switch_location_th = config_3.switch_location_th
        all_potential_locations = []
        
        for grid_cell in all_grid_cells_centers:
            foveation_results = FALcon_from_init_glimpse_loc(
                config=config_3, locmodel=model_3, input_image=image, 
                init_glimpse_loc=grid_cell, switch_location_th=switch_location_th)
            
            # Store high objectness detections
            if foveation_results["final_glimpse_switch_probability"] < switch_location_th:
                foveation_results["xywh_box"] = copy.deepcopy(foveation_results["final_glimpse_loc_and_dim"][0])
                foveation_results["xyxy_box"] = copy.deepcopy(foveation_results["final_glimpse_loc_and_dim"][0])
                foveation_results["xyxy_box"][2] += foveation_results["xyxy_box"][0]
                foveation_results["xyxy_box"][3] += foveation_results["xyxy_box"][1]
                all_potential_locations.append(foveation_results)

        # Filter using NMS based on objectness (no classification needed)
        final_detections = []
        if len(all_potential_locations) > 0:
            xyxy_boxes = []
            obj_scores = []
            for potential_location in all_potential_locations:
                xyxy_boxes.append(potential_location["xyxy_box"])
                obj_scores.append(potential_location["final_glimpse_objectness"])
            
            xyxy_boxes = torch.stack(xyxy_boxes, dim=0).float()
            obj_scores = torch.tensor(obj_scores).to(xyxy_boxes.device)
            nms_filtered_idx = nms(xyxy_boxes, obj_scores, config_3.objectness_based_nms_th)
            
            for idx in nms_filtered_idx:
                detection = all_potential_locations[idx.item()]
                # Only keep localization info, no classification
                final_detections.append({
                    "bbox_xywh": detection["xywh_box"],
                    "bbox_xyxy": detection["xyxy_box"], 
                    "objectness_score": detection["final_glimpse_objectness"],
                    "foveation_progress": detection["foveation_progress"]
                })
        else:
            # No detections found
            final_detections = []

        # Store results
        sample_stats = {}
        sample_stats["gt_synsets"] = [target_class]
        sample_stats["gt_bboxes"] = copy.deepcopy(target_bbox)
        #sample_stats["gt_synsets"] = []
        #sample_stats["gt_bboxes"] = torch.zeros((0, 4))
            
        sample_stats["detections"] = final_detections
        collected_samples[i] = sample_stats
        
        processed_count += 1
        
        # Save periodically
        torch.save(collected_samples, 
                  config_3.save_dir + f"{config_3.dataset}_localization_only_from{args.start}to{args.end}.pth")
        
        if processed_count % 100 == 0:
            print(f"Processed: {processed_count}, Skipped: {skipped_count}, Total seen: {i+1}")

print(f"\nLocalization complete!")
print(f"Total processed: {processed_count}")
print(f"Total skipped (no annotations): {skipped_count}")
print(f"Results saved.")