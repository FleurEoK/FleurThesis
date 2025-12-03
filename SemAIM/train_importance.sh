#!/bin/bash
# run_complete_pipeline.sh

source $HOME/SemAIM/env.sh

set -e

echo ""
echo "=========================================="
echo "STARTING TRAINING"
echo "=========================================="

# Set paths
DATA_PATH="/home/20204130/imagenet-subset/ILSVRC/Data/CLS-LOC"
IMPORTANCE_JSON="/home/20204130/Falcon/Polygon/results_vis/training_data_output/training_data.json"
OUTPUT_DIR="./pretrain/aim_base_importance"
LOG_DIR="./logs"

# Check if training data exists
if [ ! -f "$IMPORTANCE_JSON" ]; then
    echo "Error: Training data not found at $IMPORTANCE_JSON"
    exit 1
fi

echo "Using training data: $IMPORTANCE_JSON"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Set number of GPUs
NUM_GPUS=8

# Run training
OMP_NUM_THREADS=1 

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS \
    SemAIM-master/main_pretrain_importance.py \
    --model aim_base_importance \
    --input_size 224 \
    --data_path $DATA_PATH \
    --importance_json_path $IMPORTANCE_JSON \
    --use_importance_dataset \
    --use_importance_bias \
    --use_importance_pe \
    --permutation_type spatial_importance \
    --query_depth 12 \
    --prediction_head_type MLP \
    --loss_type L2 \
    --norm_pix_loss \
    --blr 2e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 30 \
    --batch_size 64 \
    --epochs 80 \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --experiment exp_importance_v1 \
    --saveckp_freq 20 \
    --num_workers 8 \
    --pin_mem