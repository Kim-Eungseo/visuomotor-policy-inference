#!/usr/bin/env bash
# Train BYOL on the MAIN left dataset (110k), then eval frozen vs checkpoints (holdout 26 ep).
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== BYOL left-main (stride3, 240x320 preload, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_byol.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 --out byol_left.pt

echo "===== EVAL: frozen ====="
$CONDA run --no-capture-output -n lr python -u left_knn_eval.py

for E in 2 4 6; do
  echo "===== EVAL: byol_left_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt byol_left_e$E.pt
done

echo "===== ALL DONE ====="
