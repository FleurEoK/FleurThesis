import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageNetWithImportanceFinetune(Dataset):
    """
    ImageNet dataset with importance scores for fine-tuning.
    
    This dataset loads images along with their precomputed importance scores
    from FALcon detections, enabling importance-aware fine-tuning.
    """
    
    def __init__(self, root, ann_file, importance_file, transform=None):
        """
        Args:
            root (str): Root directory of images.
            ann_file (str): Path to annotation file (e.g., train.txt).
                           Format: "path/to/image.JPEG class_id"
            importance_file (str): Path to JSON file containing importance scores.
                                  Format: {"image_name": [196-dim importance vector]}
            transform: Image transformations to apply.
        """
        self.root = root
        self.transform = transform
        
        # Load image paths and labels from annotation file
        self.samples = []
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_path = parts[0]
                label = int(parts[1])
                self.samples.append((img_path, label))
        
        # Load importance scores
        print(f"Loading importance scores from {importance_file}...")
        with open(importance_file, 'r') as f:
            self.importance_data = json.load(f)
        
        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"Importance scores available for {len(self.importance_data)} images")
        
    def __len__(self):
        return len(self.samples)
    
    def _get_image_identifier(self, img_path):
        """
        Extract image identifier from path.
        e.g., "n01440764/n01440764_10026.JPEG" -> "n01440764_10026"
        """
        basename = os.path.basename(img_path)
        image_name = os.path.splitext(basename)[0]
        return image_name
    
    def __getitem__(self, index):
        """
        Returns:
            dict: {
                'image': transformed image tensor (C, H, W),
                'label': class label (int),
                'importance_scores': importance score vector (196,),
                'image_path': original image path,
                'image_name': image identifier
            }
        """
        img_path, label = self.samples[index]
        
        # Load image
        full_path = os.path.join(self.root, img_path)
        image = Image.open(full_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get image identifier and importance scores
        image_name = self._get_image_identifier(img_path)
        
        if image_name in self.importance_data:
            importance_scores = torch.tensor(
                self.importance_data[image_name], 
                dtype=torch.float32
            )
        else:
            # Default: uniform importance (all ones)
            importance_scores = torch.ones(196, dtype=torch.float32)
        
        return {
            'image': image,
            'label': label,
            'importance_scores': importance_scores,
            'image_path': img_path,
            'image_name': image_name
        }


class ImageNetStandardFinetune(Dataset):
    """
    Standard ImageNet dataset for baseline fine-tuning (no importance).
    Returns (image, label) tuples for backward compatibility.
    """
    
    def __init__(self, root, ann_file, transform=None):
        """
        Args:
            root (str): Root directory of images.
            ann_file (str): Path to annotation file.
            transform: Image transformations to apply.
        """
        self.root = root
        self.transform = transform
        
        # Load image paths and labels
        self.samples = []
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                img_path = parts[0]
                label = int(parts[1])
                self.samples.append((img_path, label))
        
        print(f"Standard dataset initialized with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[index]
        
        # Load image
        full_path = os.path.join(self.root, img_path)
        image = Image.open(full_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label