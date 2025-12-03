# datasets/datasets_importance.py
"""
Dataset classes with importance awareness
Extended from original SemAIM dataset

NOTE: This file is ALREADY compatible with single GPU training.
No distributed training code present - works as-is.
"""

import os
import json
import torch
import PIL
import torchvision.transforms
import torchvision
from PIL import Image

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from io import BytesIO as Bytes2Data


class ImageListFolder(datasets.ImageFolder):
    """Original ImageFolder with annotation file support"""
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        # ann file is a txt file, each line is a sample
        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        # for loop that reads each line of the txt file and creates a tuple with the path and the target
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')


class ImageListFolderWithImportance(Dataset):
    """
    ImageFolder with importance scores from JSON
    Compatible with original ImageListFolder interface
    Works with single GPU or distributed training
    """
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, importance_json_path=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000
        
        # Load annotation file
        assert ann_file is not None
        print('load info from', ann_file)
        
        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()
        
        print(f'loaded {len(self.samples)} samples from annotation file')
        
        # Load importance scores if provided
        self.use_importance = importance_json_path is not None and os.path.exists(importance_json_path)
        self.importance_data = {}
        
        if self.use_importance:
            print(f'loading importance data from: {importance_json_path}')
            with open(importance_json_path, 'r') as f:
                importance_json = json.load(f)
            
            # Build mapping from image path to importance scores
            for img_name, data in importance_json.items():
                # Get importance scores
                importance_scores = data.get('importance_scores', [])
                if not importance_scores:
                    importance_scores = [1.0] * 196  # default uniform scores
                
                self.importance_data[img_name] = {
                    'importance_scores': torch.tensor(importance_scores, dtype=torch.float32),
                    'num_detections': data.get('num_detections', 0)
                }
            
            print(f'loaded importance data for {len(self.importance_data)} images')
        else:
            print('no importance data provided, using standard dataset')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        
        # Load image
        try:
            img = self.loader(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Return with or without importance
        if self.use_importance:
            img_name = os.path.basename(path)
            
            if img_name in self.importance_data:
                importance_scores = self.importance_data[img_name]['importance_scores']
                num_detections = self.importance_data[img_name]['num_detections']
            else:
                # Default uniform importance if not found
                importance_scores = torch.ones(196, dtype=torch.float32)
                num_detections = 0
            
            # Return dict format for importance-aware training
            return {
                'image': img,
                'target': target,
                'importance_scores': importance_scores,
                'image_name': img_name,
                'num_detections': num_detections
            }
        else:
            # Return tuple format for standard training
            return img, target


class ImageNetWithImportance(Dataset):
    """
    Simple ImageNet dataset with importance scores
    For use when you don't have annotation files
    Works with single GPU or distributed training
    """
    def __init__(self, root, importance_json_path, transform=None, split='train'):
        self.root = root
        self.transform = transform
        
        # Load importance scores
        print(f"Loading importance data from: {importance_json_path}")
        with open(importance_json_path, 'r') as f:
            self.importance_data = json.load(f)
        
        # Build sample list
        self.samples = []
        missing_files = []
        
        for img_name, data in self.importance_data.items():
            # Get image path
            if 'image_path' in data and data['image_path']:
                img_path = data['image_path']
            else:
                print(f"Warning: No image_path for {img_name}, skipping")
                continue
            
            # Check if file exists
            if not os.path.exists(img_path):
                missing_files.append(img_path)
                continue
            
            # Get importance scores
            importance_scores = data.get('importance_scores', [])
            if not importance_scores:
                print(f"Warning: No importance_scores for {img_name}, using uniform")
                importance_scores = [1.0] * 196
            
            self.samples.append({
                'path': img_path,
                'name': img_name,
                'class_id': data['class_id'],
                'num_detections': data.get('num_detections', 0),
                'importance_scores': torch.tensor(importance_scores, dtype=torch.float32)
            })
        
        if missing_files:
            print(f"\nWarning: {len(missing_files)} image files not found")
            if len(missing_files) <= 10:
                for f in missing_files:
                    print(f"  Missing: {f}")
        
        print(f"Successfully loaded {len(self.samples)} images with importance scores\n")
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found! Check JSON format and image paths.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            'image': img,
            'target': sample['class_id'],
            'importance_scores': sample['importance_scores'],
            'class_id': sample['class_id'],
            'image_name': sample['name'],
            'num_detections': sample['num_detections']
        }


def build_dataset(is_train, args):
    """
    Build dataset with optional importance awareness
    Works with both single GPU and distributed training
    """
    transform = build_transform(is_train, args)

    folder = os.path.join(args.data_path, 'train' if is_train else 'val')
    ann_file = os.path.join(args.data_path, 'train.txt' if is_train else 'val.txt')
    
    # Check if we should use importance-aware dataset
    use_importance = getattr(args, 'use_importance_dataset', False)
    importance_json_path = getattr(args, 'importance_json_path', None)
    
    if use_importance and importance_json_path and os.path.exists(importance_json_path):
        print("Using importance-aware dataset")
        
        # Check if annotation file exists
        if os.path.exists(ann_file):
            # Use ImageListFolderWithImportance (with annotation file)
            dataset = ImageListFolderWithImportance(
                folder, 
                transform=transform, 
                ann_file=ann_file,
                importance_json_path=importance_json_path
            )
        else:
            # Use ImageNetWithImportance (without annotation file)
            print(f"Annotation file {ann_file} not found, using ImageNetWithImportance")
            dataset = ImageNetWithImportance(
                root=folder,
                importance_json_path=importance_json_path,
                transform=transform,
                split='train' if is_train else 'val'
            )
    else:
        print("Using standard dataset (no importance)")
        # Use original ImageListFolder
        dataset = ImageListFolder(folder, transform=transform, ann_file=ann_file)

    print(dataset)
    return dataset


def build_transform(is_train, args):
    """
    Build transforms (works for both single GPU and distributed)
    
    Note: Requires these attributes in args:
    - input_size
    - color_jitter (for training)
    - aa (auto augment, for training)
    - reprob (random erasing prob, for training)
    - remode (random erasing mode, for training)
    - recount (random erasing count, for training)
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)