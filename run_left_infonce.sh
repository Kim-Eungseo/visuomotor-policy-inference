#!/usr/bin/env bash
# Train InfoNCE (TCN-mined positives + false-neg masking) on the MAIN left dataset,
# then eval checkpoints vs frozen/BYOL (holdout 26 ep). Same eval as run_left_byol.sh.
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main (stride3, 240x320 preload, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --epochs 6 --save_every 2 --holdout 26 --stride 3 --out infonce_left.pt

for E in 2 4 6; do
  echo "===== EVAL: infonce_left_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt infonce_left_e$E.pt
done

echo "===== ALL DONE ====="
