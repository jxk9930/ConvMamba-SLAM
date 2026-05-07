#!/bin/bash
#SBATCH -J cmamba_bounded
#SBATCH -o /home1/11460/jiminkim/logs/cmamba_bounded_%j.out
#SBATCH -e /home1/11460/jiminkim/logs/cmamba_bounded_%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A IRI26016
set -ex
# ── Config ──
NAME="ConvMamba_slam_bounded_v1"
TARGET_STEPS=250000
GPUS=2
PER_GPU_BATCH=2
WORK_DIR=$WORK/convmamba_slam_bounded
LATEST_CKPT="$WORK_DIR/checkpoints/${NAME}_latest.pth"
LOG_DIR=$HOME/logs
mkdir -p $LOG_DIR
# ── Environment ──
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate droid_convmamba
cd $WORK_DIR
echo "============================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Latest ckpt:   $LATEST_CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "============================================================"
# ── Resume logic ──
if [ -f "$LATEST_CKPT" ]; then
    CKPT_ARG="--ckpt $LATEST_CKPT"
    echo "[Resume] Found latest checkpoint, resuming."
else
    CKPT_ARG=""
    echo "[Fresh]  No checkpoint found, starting from step 0."
fi
# ── Pre-check: are we already done? ──
if [ -f "$LATEST_CKPT" ]; then
    CURRENT_STEP=$(python -c "
import torch
ckpt = torch.load('$LATEST_CKPT', map_location='cpu')
print(ckpt.get('total_steps', 0))
" 2>/dev/null || echo "0")
    echo "[Check]  Current step: $CURRENT_STEP / $TARGET_STEPS"
    if [ "$CURRENT_STEP" -ge "$TARGET_STEPS" ]; then
        echo "[Done]   Target reached. Exiting."
        exit 0
    fi
fi
# ── Launch training ──
python train_convmamba_v3.py \
    --name $NAME \
    --gpus $GPUS \
    --batch $PER_GPU_BATCH \
    --steps $TARGET_STEPS \
    --total_steps $TARGET_STEPS \
    --datapath datasets/TartanAir \
    $CKPT_ARG
echo "============================================================"
echo "Job $SLURM_JOB_ID finished at $(date)"
echo "============================================================"
