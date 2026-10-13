#!/usr/bin/env bash
# Crop-only ablation: front crop (bottom 30%, sides 15%) with PLAIN InfoNCE — no bg-swap/edge augs.
# Isolates the crop effect vs the notcn baseline (0.0634 @k=32).
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main CROP-PLAIN (cb=0.3 cs=0.15, no mask/edge, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 --tcn_p 0 \
  --crop_bottom 0.3 --crop_side 0.15 --out infonce_cropplain_left.pt

for E in 2 4 6; do
  echo "===== EVAL: infonce_cropplain_left_e$E (cropped) ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py \
    --ckpt infonce_cropplain_left_e$E.pt --crop_bottom 0.3 --crop_side 0.15
done
echo "===== ALL DONE ====="
