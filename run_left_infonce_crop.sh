#!/usr/bin/env bash
# Task-region crop A/B: champion recipe (bg-swap + edge clutter, tcn off) + front crop
# (cut bottom 30%, sides 15% each -> keep top 70% x center 70%). Eval uses the same crop.
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main CROP (cb=0.3 cs=0.15, mask+edge, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 --tcn_p 0 \
  --use_mask 1 --bg_p 0.5 --edge_p 0.5 --crop_bottom 0.3 --crop_side 0.15 \
  --out infonce_crop_left.pt

for E in 2 4 6; do
  echo "===== EVAL: infonce_crop_left_e$E (cropped) ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py \
    --ckpt infonce_crop_left_e$E.pt --crop_bottom 0.3 --crop_side 0.15
done

echo "===== ALL DONE ====="
