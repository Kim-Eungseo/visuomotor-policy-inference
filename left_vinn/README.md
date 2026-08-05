# VINN artifacts — MAIN left set (InfoNCE era)

Non-parametric VINN policy artifacts for the left arm, built from the full 110,568-frame /
131-episode dataset (30fps). Drop-in for `app/knn_runner.py`. Published (current champion +
maskcat bank) at **hf.co/Ngseo/hyundai-uiwang-left-vinn**; this dir keeps local copies and the
legacy variants. Weights/banks are gitignored — only configs/READMEs are in git.

## Contents
```
left_vinn/
  encoders/
    vits16_infonce_mask12_e6/  # CHAMPION: InfoNCE + bg-swap + edge-clutter augs, epoch 6
    vits16_infonce_e2/         # InfoNCE, aug-only positives (first InfoNCE release)
    vits16_byol_nomask/        # legacy BYOL winner
    vits16_byol_mask/          # legacy BYOL + invariance-positive (lost its A/B)
  banks/
    left_infonce_mask12.npz    # emb(110568,768) act(110568,26) epi — by mask12_e6  <- SERVING
    left_infonce.npz           # by infonce_e2
    left_maskcat_stride3.npz   # cf/cw/mp(fp16)+act/epi/fi, stride-3 all eps — maskcat experiment
    left_nomask.npz, left_mask.npz  # legacy BYOL banks
```
Rebuild any of these with `python build_left_vinn.py <name-filter>` (needs the /tmp eval caches).

## Results (26-ep holdout; full history in the HF model card)
- Clean kNN L1 @k=32: frozen 0.0730 → BYOL 0.0651 → **InfoNCE family 0.0633–0.0634** (ceiling;
  ±TCN, ±mask, ±edge, 6 vs 12 epochs all converge there).
- Perturbed-holdout (`left_robust_eval.py`): under front clutter, nomask degrades +4.9% vs
  **mask12 +0.8%** — the offline signature of the real-robot mask advantage.
- Mask-latent concat retrieval (`left_maskcat_eval.py`): null — α≤0.5 flat, α≥1 worse.

## Serve
```bash
export VPI_POLICY=knn
export VPI_KNN_DINO=left_vinn/encoders/vits16_infonce_mask12_e6
export VPI_KNN_BANK=left_vinn/banks/left_infonce_mask12.npz
# k=32, softmin-L2 weighting (VINN paper III-B) by default; VPI_KNN_WEIGHT=softmax_cos for legacy
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```
