#!/usr/bin/env bash
# A/B: BYOL with (front <-> front*mask) background-invariance positive on the MAIN left set.
# Same recipe/holdout as run_left_byol.sh -> directly comparable to nomask e6 (k=32 0.0651).
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== BYOL left-main MASK variant (6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_byol.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 \
  --use_mask 1 --mask_npz masks_left_stride3.npz --out byol_left_mask.pt

for E in 2 4 6; do
  echo "===== EVAL: byol_left_mask_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt byol_left_mask_e$E.pt
done

echo "===== ALL DONE ====="
