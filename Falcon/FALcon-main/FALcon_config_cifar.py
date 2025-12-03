import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define all hyperparameters for CIFAR-10
class FALcon_config(object):
    # SEED
    seed                    = 16
    loader_random_seed      = 1
    
    # dataset
    dataset                 = 'cifar10'
    dataset_dir             = './data'
    num_classes             = 10
    in_num_channels         = 3
    full_res_img_size       = (32, 32) #(height, width) as used in transforms.Resize
    valid_split_size        = 0.1 # should be in range [0, 1)
    
    # model_M3    
    model_name                  = 'vgg16'
    initialize                  = 'pretrained'
    
    assert initialize in ['pretrained', 'random', 'resume_from_pretrained', 'resume_from_random'], \
    "Specify which initialization method to choose. Options ('pretrained', 'random', 'resume_from_pretrained', 'resume_from_random')"
    if 'resume' in initialize:
        initialize, init_factual    = initialize.split("_from_")
    else:
        init_factual                = initialize
    
    if 'vgg' in model_name:
        downsampling            = 'M'
        fc1                     = 256
        fc2                     = 128
        dropout                 = 0.5
        norm                    = 'none'
        init_weights            = True
        adaptive_avg_pool_out   = (1, 1)
        saccade_fc1             = 256
        saccade_dropout         = False
        assert model_name in ['custom_vgg8_narrow_k2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'], \
        "Specify which VGG model to use for training. Options ('custom_vgg8_narrow_k2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn')"
        assert norm in ['none', 'batchnorm', 'evonorm'], \
        "Specify which normalization type to use for normalization layers. Options ('batchnorm', 'instancenorm', 'layernorm', 'evonorm')"
        
    elif 'resnet' in model_name:
        norm                    = 'batchnorm'
        init_weights            = True
        adaptive_avg_pool_out   = (1, 1)
        fc1                     = 512 # for RL head
        fc2                     = 256 # for RL head
        assert model_name in ['resnet50', 'resnet101'], \
        "Specify which ResNet model to use for training. Options ('resnet50', 'resnet101')"
        assert norm in ['batchnorm', 'instancenorm', 'layernorm', 'evonorm'], \
        "Specify which normalization type to use for normalization layers. Options ('batchnorm', 'instancenorm', 'layernorm', 'evonorm')"



    # training
    train_loader_type       = 'train'
    if train_loader_type == 'train_and_val':
        valid_loader_type   = 'train_and_val'
    elif train_loader_type == 'train':
        valid_loader_type   = 'test'
        print("Warning: selected training on entire ImageNet train split, hence validation is going to be performed on test (ImageNet val) split!")
    else:
        raise ValueError("Unrecognized type of split to train on: ({})".format(train_loader_type))

    experiment_name         = (dataset + '/wsol_method_{}'.format(wsol_method) +
                               '/trained_on_{}_split/'.format(train_loader_type) + 
                               'arch_{}_{}_init_normalization_{}_seed_{}/'.format(model_name, init_factual, norm, seed))
    save_dir                = './results/' + experiment_name
    batch_size_train        = 512
    batch_size_eval         = 512
    batch_size_inf          = 512
    epochs                  = 100
    lr_start                = 1e-2
    lr_min                  = 1e-5
    milestones              = [30, 60, 90]
    weight_decay            = 0.0001
    momentum                = 0.9
    
    # testing
    ckpt_dir                = save_dir + 'model.pth'
    attr_detection_th       = 0.0
    
    # AVS-specific parameters
    num_glimpses            = 10
    fovea_control_neurons   = 4
    
    glimpse_size_grid       = (20, 20) #(width, height) of each grid when initially dividing image into grid cells
    glimpse_size_init       = (20, 20) #(width, height) size of initial foveation glimpse at the selected grid cell (usually, the same as above)
    glimpse_size_fixed      = (96, 96) #(width, height) size of foveated glimpse as perceived by the network
    glimpse_size_step       = (20, 20) #step size of foveation in (x, y) direction at each action in each (+dx, -dx, +dy, -dy) directions
    glimpse_change_th       = 0.5      #threshold, deciding whether or not to take the action based on post-sigmoid logit value 
    iou_th                  = 0.5
    # switching cell behavior
    ratio_wrong_init_glimpses   = 0.5 # ratio of the incorrect initial glimpses to the total glimpses in the batch
    switch_location_th          = 0.5