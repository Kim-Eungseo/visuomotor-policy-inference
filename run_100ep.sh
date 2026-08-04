#!/usr/bin/env bash
# Retrain BYOL (mask + nomask) on the 100-ep teleop dataset, holdout 20 ep for eval.
set -e
cd /home/ngseo/visuomotor-policy-inference
export VPI_TELEOP_ROOT=/home/ngseo/remove_hook/lerobot/data/lerobot/hyundai_uiwang_left_teleop
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== BYOL mask (100ep, holdout 20, 30ep) ====="
$CONDA run --no-capture-output -n lr python teleop_byol.py \
  --use_mask 1 --epochs 30 --save_every 10 --holdout 20 --out byol100_mask.pt

echo "===== BYOL nomask (100ep, holdout 20, 30ep) ====="
$CONDA run --no-capture-output -n lr python teleop_byol.py \
  --use_mask 0 --epochs 30 --save_every 10 --holdout 20 --out byol100_nomask.pt

echo "===== ALL TRAINING DONE ====="
