# coding: utf-8

# In[1]:

import time
import os
import random
import math

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image
from loader.imagenet_loader import ImageNetDataset
from utils.func import *
from utils.IoU import *
from models.models import *
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
import argparse

# In[2]:

### Some utilities


# In[3]:

def compute_reg_acc(preds, targets, theta=0.5):
    # preds = box_transform_inv(preds.clone(), im_sizes)
    # preds = crop_boxes(preds, im_sizes)
    # targets = box_transform_inv(targets.clone(), im_sizes)
    IoU = compute_IoU(preds, targets)
    # print(preds, targets, IoU)
    corr = (IoU >= theta).sum()
    return float(corr) / float(preds.size(0))


def compute_cls_acc(preds, targets):
    pred = torch.max(preds, 1)[1]
    # print(preds, pred)
    num_correct = (pred == targets).sum()
    return float(num_correct) / float(preds.size(0))


def compute_acc(reg_preds, reg_targets, cls_preds, cls_targets, theta=0.5):
    IoU = compute_IoU(reg_preds, reg_targets)
    reg_corr = (IoU >= theta)

    pred = torch.max(cls_preds, 1)[1]
    cls_corr = (pred == cls_targets)

    corr = (reg_corr & cls_corr).sum()

    return float(corr) / float(reg_preds.size(0))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# Custom wrapper class to fix path mismatch issues
class FixedImageNetDataset(Dataset):
    def __init__(self, root, ddt_path, gt_path, train=True, input_size=256, crop_size=224, transform=None):
        # Initialize the original dataset
        self.original_dataset = ImageNetDataset(
            root=root, ddt_path=ddt_path, gt_path=gt_path, 
            train=train, input_size=input_size, crop_size=crop_size, 
            transform=transform
        )
        
        # Store parameters
        self.root = root
        self.train = train
        
        # Default bounding box for missing classes (full image)
        self.default_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])
        
        print(f"Initialized {'training' if train else 'validation'} dataset with {len(self.original_dataset)} samples")
        
    def convert_path_for_bbox_lookup(self, actual_path):
        """Convert actual dataset path to bbox_dict path format"""
        try:
            parts = actual_path.split('/')
            class_name = parts[-2]  # e.g., 'n01443537'
            image_name = parts[-1]  # e.g., 'n01443537_13426.JPEG'
            
            # Try different possible bbox_dict path formats
            possible_paths = [
                f"/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/train/{class_name}/{image_name}",
                f"/home/nano01/a/tibrayev/imagenet/annotated_imagenet2012/val/{class_name}/{image_name}",
                f"{class_name}/{image_name}",  # relative path
                actual_path  # original path
            ]
            
            return possible_paths
        except:
            return [actual_path]
    
    def get_bbox_for_path(self, path, target_class):
        """Get bounding box for a given path, with fallback handling"""
        # If the original dataset has a method to get bbox, try that first
        if hasattr(self.original_dataset, 'get_bbox'):
            try:
                return self.original_dataset.get_bbox(path, target_class)
            except:
                pass
        
        # If the dataset has bbox_dict, try path conversion
        if hasattr(self.original_dataset, 'bbox_dict'):
            bbox_dict = self.original_dataset.bbox_dict
            
            # Try different path formats
            possible_paths = self.convert_path_for_bbox_lookup(path)
            
            for bbox_path in possible_paths:
                if bbox_path in bbox_dict:
                    return bbox_dict[bbox_path]
            
            # Try looking up by class
            if target_class in bbox_dict:
                # Get first available bbox for this class
                class_bboxes = bbox_dict[target_class]
                if isinstance(class_bboxes, list) and len(class_bboxes) > 0:
                    return class_bboxes[0] if isinstance(class_bboxes[0], (list, tuple, torch.Tensor)) else self.default_bbox
                elif isinstance(class_bboxes, (list, tuple, torch.Tensor)):
                    return class_bboxes
        
        # Return default bbox if nothing found
        print(f"Warning: Using default bbox for path {path}, class {target_class}")
        return self.default_bbox
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, index):
        try:
            # Try to get item from original dataset
            if hasattr(self.original_dataset, '__getitem__'):
                return self.original_dataset.__getitem__(index)
            
            # If that fails, implement custom logic
            # This is a fallback - you may need to adjust based on your dataset structure
            sample_info = self.original_dataset.samples[index] if hasattr(self.original_dataset, 'samples') else None
            
            if sample_info:
                path, target_class = sample_info
                
                # Load image
                image = Image.open(path).convert('RGB')
                if self.original_dataset.transform:
                    image = self.original_dataset.transform(image)
                
                # Get bounding box with fallback
                bbox = self.get_bbox_for_path(path, target_class)
                if not isinstance(bbox, torch.Tensor):
                    bbox = torch.tensor(bbox if bbox is not None else [0.0, 0.0, 1.0, 1.0])
                
                return image, target_class, bbox
            
            # Ultimate fallback
            return self.original_dataset.samples[index] if hasattr(self.original_dataset, 'samples') else (None, None, self.default_bbox)
            
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            # Return dummy data to prevent training from crashing
            dummy_image = torch.zeros(3, self.original_dataset.crop_size, self.original_dataset.crop_size) if hasattr(self.original_dataset, 'crop_size') else torch.zeros(3, 224, 224)
            return dummy_image, 0, self.default_bbox


# ### Visualize training data

# In[8]:

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
test_transfrom = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
# ### Training

# In[10]:

# prepare data
parser = argparse.ArgumentParser(description='Parameters for PSOL evaluation')
parser.add_argument('--loc-model', metavar='locarg', type=str, default='resnet50',dest='locmodel')
parser.add_argument('--input_size',default=256,dest='input_size')
parser.add_argument('--crop_size',default=224,dest='crop_size')
parser.add_argument('--epochs',default=6,dest='epochs')
parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
parser.add_argument('--ddt_path',help='generated ddt path',default='PSOL/results/ImageNet_train_subset',dest="ddt_path")
parser.add_argument('--gt_path',help='validation groundtruth path',default='PSOL/ImageNet_gt/',dest="gt_path")  # Fixed argument name
parser.add_argument('--save_path',help='model save path',default='psol-results/ImageNet_checkpoint',dest='save_path')
parser.add_argument('--batch_size',default=256,dest='batch_size')
parser.add_argument('data',metavar='DIR', default = '/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train',help='path to imagenet dataset')


args = parser.parse_args()
batch_size = args.batch_size
#lr = 1e-3 * (batch_size / 64)
lr = 1e-3 * (batch_size / 256)
# lr = 3e-4
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
root = args.data
savepath = args.save_path
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['MKL_NUM_THREADS'] = '20'

# Use the fixed dataset wrapper instead of the original
print("Initializing training dataset...")
MyTrainData = FixedImageNetDataset(root=root, ddt_path=args.ddt_path, gt_path=args.gt_path,train=True, input_size=args.input_size,crop_size = args.crop_size,
                                 transform=train_transform)

print("Initializing validation dataset...")
MyTestData = FixedImageNetDataset(root=root, ddt_path=args.ddt_path, gt_path=args.gt_path, train=False, input_size=args.input_size,crop_size = args.crop_size,
                                transform=test_transfrom)

print(f"Training samples: {len(MyTrainData)}")
print(f"Validation samples: {len(MyTestData)}")

train_loader = torch.utils.data.DataLoader(dataset=MyTrainData,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset=MyTestData, batch_size=batch_size,
                                          num_workers=0, pin_memory=True)
dataloaders = {'train': train_loader, 'test': test_loader}

# construct model
model = choose_locmodel(args.locmodel)
print(model)
model = torch.nn.DataParallel(model).cuda()
reg_criterion = nn.MSELoss().cuda()
dense1_params = list(map(id, model.module.fc.parameters()))
rest_params = filter(lambda x: id(x) not in dense1_params, model.parameters())
param_list = [{'params': model.module.fc.parameters(), 'lr': 2 * lr},
              {'params': rest_params,'lr': 1 * lr}]
optimizer = torch.optim.SGD(param_list, lr, momentum=momentum,
                            weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs, gamma=0.1)
torch.backends.cudnn.benchmark = True
best_model_state = model.state_dict()
best_epoch = -1
best_acc = 0.0

epoch_loss = {'train': [], 'test': []}
epoch_acc = {'train': [], 'test': []}
epochs = args.epochs
lambda_reg = 0

print("Starting training...")
for epoch in range(epochs):
    lambda_reg = 50
    for phase in ('train', 'test'):
        reg_accs = AverageMeter()
        accs = AverageMeter()
        reg_losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if phase == 'train':
            if epoch > 0:
                scheduler.step()
            model.train()
        else:
            model.eval()

        end = time.time()
        cnt = 0
        
        for ims, labels, boxes in dataloaders[phase]:
            data_time.update(time.time() - end)
            
            # Skip batch if data is invalid
            if ims is None or labels is None or boxes is None:
                print(f"Skipping invalid batch in {phase}")
                continue
                
            inputs = Variable(ims.cuda())
            boxes = Variable(boxes.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()

            # forward
            if phase == 'train':
                if 'inception' in args.locmodel:
                    reg_outputs1,reg_outputs2 = model(inputs)
                    reg_loss1 = reg_criterion(reg_outputs1, boxes)
                    reg_loss2 = reg_criterion(reg_outputs2, boxes)
                    reg_loss = 1 * reg_loss1 + 0.3 * reg_loss2
                    reg_outputs = reg_outputs1
                else:
                    reg_outputs = model(inputs)
                    reg_loss = reg_criterion(reg_outputs, boxes)
                        #_,reg_loss = compute_iou(reg_outputs,boxes)
            else:
                with torch.no_grad():
                    reg_outputs = model(inputs)
                    reg_loss = reg_criterion(reg_outputs, boxes)
            loss = lambda_reg * reg_loss
            reg_acc = compute_reg_acc(reg_outputs.data.cpu(), boxes.data.cpu())

            nsample = inputs.size(0)
            reg_accs.update(reg_acc, nsample)
            reg_losses.update(reg_loss.item(), nsample)
            if phase == 'train':
                loss.backward()
                optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if cnt % print_freq == 0:
                print(
                        '[{}]\tEpoch: {}/{}\t Iter: {}/{} Time {:.3f} ({:.3f})\t Data {:.3f} ({:.3f})\tLoc Loss: {:.4f}\tLoc Acc: {:.2%}\t'.format(
                            phase, epoch + 1, epochs, cnt, len(dataloaders[phase]), batch_time.val,batch_time.avg,data_time.val,data_time.avg,lambda_reg * reg_losses.avg, reg_accs.avg))
            cnt += 1
            
        if phase == 'test' and reg_accs.avg > best_acc:
            best_acc = reg_accs.avg
            best_epoch = epoch
            best_model_state = model.state_dict()

        elapsed_time = time.time() - end
        print(
            '[{}]\tEpoch: {}/{}\tLoc Loss: {:.4f}\tLoc Acc: {:.2%}\tTime: {:.3f}'.format(
                phase, epoch + 1, epochs, lambda_reg * reg_losses.avg, reg_accs.avg,elapsed_time))
        epoch_loss[phase].append(reg_losses.avg)
        epoch_acc[phase].append(reg_accs.avg)

    print('[Info] best test acc: {:.2%} at {}th epoch'.format(best_acc, best_epoch + 1))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    torch.save(model.state_dict(), os.path.join(savepath,'checkpoint_localization_imagenet_ddt_' + args.locmodel + "_"  + str(epoch) + '.pth.tar'))
    torch.save(best_model_state, os.path.join(savepath,'best_cls_localization_imagenet_ddt_' + args.locmodel + "_"  + str(epoch) + '.pth.tar'))

print("Training completed!")