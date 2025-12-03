"""
Diagnostic script to identify the illegal construction error
"""

import torch
import sys
import inspect  # Import at the top!
sys.path.append('/home/20204130/SemAIM')

# Test 1: Check if model accepts img_size parameter
print("=" * 60)
print("TEST 1: Model constructor parameters")
print("=" * 60)

model_fn = None  # Initialize to avoid undefined error

#Check model availability
print("CHECK MODEL AVAILABILITY IN IMPORTANCE FILE")
# First, let's see what's in the module
from models import models_semaim_importance
print("Available models:", dir(models_semaim_importance))
print("\nOr check __all__:", getattr(models_semaim_importance, '__all__', 'Not defined'))


try:
    from models import models_semaim_importance
    
    # Get the model function
    model_fn = models_semaim_importance.__dict__['aim_base_importance']
    
    # Get signature
    sig = inspect.signature(model_fn)
    print(f"Model parameters: {list(sig.parameters.keys())}")
    
    # Try creating model WITHOUT img_size
    print("\nTrying to create model WITHOUT img_size...")
    model1 = model_fn(
        permutation_type='spatial_importance',
        attention_type='cls',
        query_depth=12,
        share_weight=False,
        out_dim=512,
        prediction_head_type='MLP',
        gaussian_kernel_size=None,
        gaussian_sigma=None,
        loss_type='L2',
        predict_feature='none',
        norm_pix_loss=False,
        use_importance_bias=True,
        use_importance_pe=True,
        importance_json_path=None
    )
    print("✓ Model created successfully WITHOUT img_size")
    
except Exception as e:
    print(f"✗ Error WITHOUT img_size: {e}")
    import traceback
    traceback.print_exc()

if model_fn is not None:
    try:
        # Try creating model WITH img_size
        print("\nTrying to create model WITH img_size...")
        model2 = model_fn(
            img_size=224,  # ← This might be the problem
            permutation_type='spatial_importance',
            attention_type='cls',
            query_depth=12,
            share_weight=False,
            out_dim=512,
            prediction_head_type='MLP',
            gaussian_kernel_size=None,
            gaussian_sigma=None,
            loss_type='L2',
            predict_feature='none',
            norm_pix_loss=False,
            use_importance_bias=True,
            use_importance_pe=True,
            importance_json_path=None
        )
        print("✓ Model created successfully WITH img_size")
        
    except Exception as e:
        print(f"✗ Error WITH img_size: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Skipping WITH img_size test (model_fn not defined)")

# Test 2: Check dataset format
print("\n" + "=" * 60)
print("TEST 2: Dataset return format")
print("=" * 60)

try:
    from datasets.datasets_complete import ImageNetWithImportance
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageNetWithImportance(
        root='/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC/train',
        importance_json_path='/home/20204130/Falcon/Polygon/results_vis/training_data_output/training_data.json',
        transform=transform,
        split='train'
    )
    
    # Get one sample
    sample = dataset[0]
    print(f"Dataset returns type: {type(sample)}")
    
    if isinstance(sample, dict):
        print(f"Dictionary keys: {sample.keys()}")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")
    elif isinstance(sample, tuple):
        print(f"Tuple length: {len(sample)}")
        for i, item in enumerate(sample):
            if hasattr(item, 'shape'):
                print(f"  Item {i}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  Item {i}: {type(item)}")
    
except Exception as e:
    print(f"✗ Dataset error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check training engine compatibility
print("\n" + "=" * 60)
print("TEST 3: Training engine compatibility")
print("=" * 60)

try:
    from engines.engine_pretrain_importance import train_one_epoch
    sig = inspect.signature(train_one_epoch)
    print(f"train_one_epoch parameters: {list(sig.parameters.keys())}")
    
except Exception as e:
    print(f"✗ Engine import error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)