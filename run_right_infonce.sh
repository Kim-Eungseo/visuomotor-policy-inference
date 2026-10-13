#!/usr/bin/env bash
# RIGHT-arm VINN encoder: plain InfoNCE + edge-clutter aug (no bg-swap — the hand+hook seg model
# is unvalidated on the INSPIRE right hand). Holdout 24/118 (~20%), same protocol as left.
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE right-main (edge_p=0.5, tcn off, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --side right --epochs 6 --save_every 2 --holdout 24 --stride 3 --tcn_p 0 \
  --edge_p 0.5 --out infonce_right.pt

echo "===== EVAL: frozen (right) ====="
$CONDA run --no-capture-output -n lr python -u left_knn_eval.py --side right --holdout 24

for E in 2 4 6; do
  echo "===== EVAL: infonce_right_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py \
    --side right --holdout 24 --ckpt infonce_right_e$E.pt
done
echo "===== ALL DONE ====="
