"""
Pre-training with Importance-Aware SemAIM (Single GPU Version)
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import timm
import timm.optim.optim_factory as optim_factory
from timm.utils import ModelEma
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import models_semaim_importance
from engines.engine_pretrain_importance import train_one_epoch
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from datasets.datasets_complete import ImageNetWithImportance


def get_args_parser():
    parser = argparse.ArgumentParser('SemAIM pre-training with Importance (Single GPU)')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='aim_base_importance', type=str, metavar='MODEL',
                        help='Name of model to train: aim_base_importance, aim_large_importance, aim_huge_importance')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--query_depth', default=12, type=int,
                        help='decoder depth')
    parser.add_argument('--share_weight', action='store_true',
                        help='Share weight between encoder and decoder')

    parser.add_argument('--prediction_head_type', default='MLP', type=str,
                        help='the type of prediction head: MLP or LINEAR')
    parser.add_argument('--gaussian_kernel_size', default=None, type=int,
                        help='Use gaussian blur to smooth the target image')
    parser.add_argument('--gaussian_sigma', default=None, type=int,
                        help='standard deviation of gaussian blur')
    parser.add_argument('--loss_type', default='L2', type=str,
                        help='Calculate loss between prediction and target: L1 or L2')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (patched) normalized pixels as targets for computing loss')

    # Importance parameters
    parser.add_argument('--use_importance_bias', action='store_true',
                        help='Use importance scores as attention bias')
    parser.add_argument('--use_importance_pe', action='store_true',
                        help='Use importance-weighted position encoding')
    parser.add_argument('--importance_json_path', 
                        default='/home/20204130/Falcon/Polygon/results_vis/training_data_output/training_data.json',
                        type=str, help='Path to importance scores JSON')
    
    # Set defaults for importance
    parser.set_defaults(use_importance_bias=True)
    parser.set_defaults(use_importance_pe=True)
    
    # semaim
    parser.add_argument('--permutation_type', default='spatial_importance', type=str,
                        help='Permutation type: spatial_importance, center2out, stochastic, etc.')
    parser.add_argument('--use_ema_model', action='store_true', help='Use ema features')
    parser.set_defaults(use_ema_model=False)
    parser.add_argument('--predict_feature', default='none', type=str, 
                        help='Use features as targets: none, inference, ema, dino, clip')
    parser.add_argument('--attention_type', default='cls', type=str, 
                        help='Attention type: gap, cls and self')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-4, metavar='LR',
                        help='base learning rate: absolute lr = base lr * total_batch size/256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--not_use_fp16', action='store_true', help='whether to use fp16')
    parser.set_defaults(not_use_fp16=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train', type=str, help='dataset path')
    parser.add_argument('--use_importance_dataset', action='store_true',
                        help='Use dataset with importance scores')
    parser.set_defaults(use_importance_dataset=True)

    parser.add_argument('--output_dir', default='./pretrain/aim_base_importance',
                        help='path where to save')
    parser.add_argument('--log_dir', default='./output_dir', help='tensorboard log path')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint frequency')
    parser.add_argument('--experiment', default='exp_importance', type=str, 
                        help='experiment name')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Create dataset with importance scores
    if args.use_importance_dataset and os.path.exists(args.importance_json_path):
        print(f"Using importance dataset with scores from: {args.importance_json_path}")
        dataset_train = ImageNetWithImportance(
            root=args.data_path,
            importance_json_path=args.importance_json_path,
            transform=transform_train,
            split='train'
        )
    else:
        print("Using standard ImageFolder dataset (no importance scores)")
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), 
                                            transform=transform_train)
    
    print(dataset_train)

    # Simple random sampler for single GPU
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Using RandomSampler for single GPU training")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model with importance awareness
    out_dim = 512
    model = models_semaim_importance.__dict__[args.model](
        img_size=args.input_size,
        permutation_type=args.permutation_type,
        attention_type=args.attention_type,
        query_depth=args.query_depth, 
        share_weight=args.share_weight,
        out_dim=out_dim,
        prediction_head_type=args.prediction_head_type,
        gaussian_kernel_size=args.gaussian_kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        loss_type=args.loss_type, 
        predict_feature=args.predict_feature,
        norm_pix_loss=args.norm_pix_loss,
        use_importance_bias=args.use_importance_bias,
        use_importance_pe=args.use_importance_pe,
        importance_json_path=args.importance_json_path if args.use_importance_dataset else None
    )

    model.to(device)

    print("Model = %s" % str(model))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Importance bias enabled: {args.use_importance_bias}")
    print(f"Importance PE enabled: {args.use_importance_pe}")

    # define ema model
    model_ema = None
    teacher_model = None
    if args.use_ema_model:
        model_ema = ModelEma(model, decay=0.999, device=args.device, resume='')
    elif args.predict_feature == 'dino':
        teacher_model = timm.models.vit_base_patch16_224(num_classes=0)
        state_dict = torch.load('/path_to_dino_model/dino_vitbase16_pretrain.pth')
        msg = teacher_model.load_state_dict(state_dict, strict=False)
        print("loaded dino model with msg:", msg)
        teacher_model.to(device)
        teacher_model.eval()
    elif args.predict_feature == 'clip':
        from models.models_clip import build_model
        state_dict = torch.load('/path_to_clip_model/clip_vitbase16_pretrain.pth', map_location='cpu')
        model_clip = build_model(state_dict)
        msg = model_clip.load_state_dict(state_dict, strict=False)
        print("loaded clip model with msg:", msg)
        model_clip.float()
        teacher_model = model_clip.visual
        teacher_model.to(device)
        teacher_model.eval()

    # Effective batch size calculation for single GPU
    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 32

    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # No DDP wrapping for single GPU
    model_without_ddp = model
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    
    if args.not_use_fp16:
        loss_scaler = None
    else:
        loss_scaler = NativeScaler()

    # Load checkpoint if exists
    ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not found in {}, train from random initialization".format(ckpt_path))
    else:
        print("Found checkpoint at {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Load model state
        model_without_ddp.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer state
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load loss scaler state
        if loss_scaler is not None and 'loss_scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        
        # Load EMA model if used
        if model_ema is not None and 'ema_state_dict' in checkpoint:
            model_ema.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        # Resume from saved epoch
        args.start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {args.start_epoch}")

    # Setup tensorboard logging
    log_dir = os.path.join(args.log_dir, f"{args.model}.{args.experiment}")
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler, args.clip_grad,
            log_writer=log_writer,
            args=args, 
            model_ema=model_ema,
            teacher_model=teacher_model,
            use_importance=args.use_importance_dataset
        )

        # Save checkpoint
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model": args.model,
            "args": vars(args),  # Save args for reproducibility
        }
        if loss_scaler is not None:
            save_dict['loss_scaler'] = loss_scaler.state_dict()
        if model_ema is not None:
            save_dict['ema_state_dict'] = model_ema.ema.state_dict()

        # Save temporary checkpoint
        ckpt_path = os.path.join(args.output_dir, f"{args.model}.{args.experiment}.temp.pth")
        torch.save(save_dict, ckpt_path)
        print(f"model_path: {ckpt_path}")

        # Save periodic checkpoint
        if args.output_dir and ((epoch + 1) % args.saveckp_freq == 0 or epoch + 1 == args.epochs):
            ckpt_path = os.path.join(args.output_dir, 
                                    "{}.{}.{:04d}.pth".format(args.model, args.experiment, epoch+1))
            torch.save(save_dict, ckpt_path)

        # Log statistics
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if log_writer is not None:
            log_writer.flush()
        
        with open(os.path.join(args.output_dir,
                              "{}.{}.log.txt".format(args.model, args.experiment)), 
                 mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)