#!/usr/bin/env bash
# Ablation: InfoNCE with TCN mining OFF (--tcn_p 0, aug-only positives; false-neg masking kept).
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main NO-TCN (tcn_p=0, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 --tcn_p 0 --out infonce_notcn_left.pt

for E in 2 4 6; do
  echo "===== EVAL: infonce_notcn_left_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt infonce_notcn_left_e$E.pt
done

echo "===== ALL DONE ====="
