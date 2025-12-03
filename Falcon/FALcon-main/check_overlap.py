#!/usr/bin/env python3
"""
Check overlap between training images and their annotations
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths from your config
dataset_dir = '/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/'
pseudo_bbox_dir = '/home/20204130/imagenet-subset/ILSVRC/Annotations/CLS-LOC/train'

# Get all training images organized by class
train_dir = os.path.join(dataset_dir, 'train')
classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

print(f"Found {len(classes)} classes in training directory")
print(f"Classes: {classes}\n")

total_images = 0
total_images_with_annotations = 0
total_images_without_annotations = 0
missing_xml_files = []
empty_annotation_files = []

class_stats = {}

for cls_idx, cls_name in enumerate(classes):
    class_dir = os.path.join(train_dir, cls_name)
    
    # Get all images in this class
    image_files = [f for f in os.listdir(class_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    num_images = len(image_files)
    total_images += num_images
    
    # Check for XML annotations for each image
    annotated_images = set()
    for img_file in image_files:
        # Get image basename without extension
        img_basename = os.path.splitext(img_file)[0]
        
        # Check if corresponding XML file exists
        xml_file = os.path.join(pseudo_bbox_dir, cls_name, f"{img_basename}.xml")
        
        if os.path.isfile(xml_file):
            try:
                # Try to parse the XML to make sure it's valid
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # Check if there are any object annotations
                objects = root.findall('object')
                if len(objects) > 0:
                    annotated_images.add(img_file)
            except (ET.ParseError, OSError) as e:
                # XML file exists but is corrupted or empty
                pass
    
    num_annotated = len(annotated_images)
    num_missing = num_images - num_annotated
    coverage = (num_annotated / num_images * 100) if num_images > 0 else 0
    
    total_images_with_annotations += num_annotated
    total_images_without_annotations += num_missing
    
    class_stats[cls_name] = {
        'num_images': num_images,
        'num_annotated': num_annotated,
        'num_missing': num_missing,
        'coverage': coverage
    }
    
    if num_annotated == 0:
        empty_annotation_files.append(cls_name)
    
    print(f"Class {cls_name}:")
    print(f"  Images in directory: {num_images}")
    print(f"  Images with annotations: {num_annotated}")
    print(f"  Coverage: {coverage:.1f}%")
    if num_missing > 0 and num_missing <= 5:
        missing = set(image_files) - annotated_images
        print(f"  Missing annotations for: {list(missing)[:5]}")
    elif num_missing > 5:
        print(f"  Missing annotations for {num_missing} images")
    print()

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print(f"Total classes: {len(classes)}")
print(f"Total training images: {total_images}")
print(f"Images with annotations: {total_images_with_annotations}")
print(f"Images without annotations: {total_images_without_annotations}")
print(f"Overall coverage: {(total_images_with_annotations/total_images*100):.1f}%")
print()

if empty_annotation_files:
    print(f"Classes with NO annotations ({len(empty_annotation_files)}):")
    for cls in empty_annotation_files:
        print(f"  - {cls}")
    print()

# Classes with low coverage
low_coverage = {k: v for k, v in class_stats.items() if v['coverage'] < 90}
if low_coverage:
    print(f"Classes with <90% annotation coverage ({len(low_coverage)}):")
    for cls, stats in sorted(low_coverage.items(), key=lambda x: x[1]['coverage']):
        print(f"  - {cls}: {stats['coverage']:.1f}% ({stats['num_annotated']}/{stats['num_images']})")