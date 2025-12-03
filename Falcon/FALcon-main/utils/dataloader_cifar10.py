from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torch
import numpy as np
import math
import copy
import random
from PIL import Image
import torchvision.transforms.functional as F

class ResizedImgAndBBox:
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bbox):
        w, h = img.size
        new_w, new_h = self.size[::-1]
        rate_w, rate_h = new_w / w, new_h / h
        bbox = bbox.copy()
        bbox[:, [0, 2]] *= rate_w
        bbox[:, [1, 3]] *= rate_h
        img = F.resize(img, self.size, self.interpolation)
        return img, bbox

class RandomHorizontalFlipImgAndBBox:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bbox):
        if random.random() < self.p:
            w, _ = img.size
            bbox = bbox.copy()
            bbox[:, [0, 2]] = w - bbox[:, [2, 0]]
            img = F.hflip(img)
        return img, bbox

class CIFAR10WithBBox(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 bbox_dict=None, input_size=(128, 128), fetch_one_bbox=True):
        self.dataset = CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.target_transform = target_transform
        self.bbox_dict = bbox_dict
        self.input_size = input_size
        self.fetch_one_bbox = fetch_one_bbox

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        w, h = img.size

        if self.bbox_dict and index in self.bbox_dict:
            bbox = np.array(self.bbox_dict[index], dtype='float32')
        else:
            bbox = np.array([[w * 0.25, h * 0.25, w * 0.75, h * 0.75]], dtype='float32')  # Fake box

        img, bbox = ResizedImgAndBBox(self.input_size)(img, bbox)
        img, bbox = RandomHorizontalFlipImgAndBBox()(img, bbox)

        bbox[:, 2] -= bbox[:, 0]
        bbox[:, 3] -= bbox[:, 1]
        bbox = torch.tensor(bbox)

        if self.fetch_one_bbox:
            bbox = bbox[bbox[:, 2]*bbox[:, 3].argmax().item()]

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target, bbox
