#!/usr/bin/env bash
# 12-epoch InfoNCE: bg-swap (mask) + border-band clutter patches, both p=0.5. TCN off.
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main MASK+EDGE (tcn_p=0, bg_p=0.5, edge_p=0.5, 12 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --epochs 12 --save_every 2 --holdout 26 --stride 3 --tcn_p 0 \
  --use_mask 1 --bg_p 0.5 --edge_p 0.5 --out infonce_mask12_left.pt

for E in 4 6 8 10 12; do
  echo "===== EVAL: infonce_mask12_left_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt infonce_mask12_left_e$E.pt
done

echo "===== ALL DONE ====="
