#!/usr/bin/env bash
# InfoNCE + background-swap augmentation (hand+hook mask; NOT the BYOL invariance positive).
# TCN off (proven no-op at optimum), false-neg masking kept.
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main MASK-AUG (tcn_p=0, bg_p=0.5, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 --tcn_p 0 \
  --use_mask 1 --bg_p 0.5 --out infonce_mask_left.pt

for E in 2 4 6; do
  echo "===== EVAL: infonce_mask_left_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt infonce_mask_left_e$E.pt
done

echo "===== ALL DONE ====="
