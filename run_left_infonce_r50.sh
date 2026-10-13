#!/usr/bin/env bash
# CNN encoder A/B: ImageNet ResNet-50 (unfreeze layer4) with the same plain InfoNCE recipe,
# no crop, no bg-swap/edge. Also evals the frozen IN-pretrained R50 as reference.
set -e
cd /home/ngseo/visuomotor-policy-inference
CONDA=/home/ngseo/miniconda3/bin/conda

echo "===== InfoNCE left-main R50 (plain, 6 epochs) ====="
$CONDA run --no-capture-output -n lr python -u left_infonce.py \
  --arch resnet50 --epochs 6 --save_every 2 --holdout 26 --stride 3 --tcn_p 0 \
  --out infonce_r50_left.pt

echo "===== EVAL: frozen resnet50 ====="
$CONDA run --no-capture-output -n lr python -u left_knn_eval.py --arch resnet50

for E in 2 4 6; do
  echo "===== EVAL: infonce_r50_left_e$E ====="
  $CONDA run --no-capture-output -n lr python -u left_knn_eval.py --ckpt infonce_r50_left_e$E.pt
done
echo "===== ALL DONE ====="
