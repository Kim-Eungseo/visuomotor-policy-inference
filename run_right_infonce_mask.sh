#!/usr/bin/env bash
# RIGHT-arm champion candidate: InfoNCE + bg-swap (motion masks) + edge clutter, tcn off.
# A/B partner of run_right_infonce.sh (edge-only baseline, e2 = 0.0303 @k=32).
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE right-main MASK+EDGE (bg_p=0.5, edge_p=0.5, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --side right --epochs 6 --save_every 2 --holdout 24 --stride 3 --tcn_p 0 \
  --use_mask 1 --mask_npz masks_right_stride3.npz --bg_p 0.5 --edge_p 0.5 \
  --out infonce_right_mask.pt

for E in 2 4 6; do
  echo "===== EVAL: infonce_right_mask_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py \
    --side right --holdout 24 --ckpt infonce_right_mask_e$E.pt
done
echo "===== ALL DONE ====="
