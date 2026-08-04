# VINN artifacts (BYOL-finetuned DINOv3-ViTS16 + kNN banks) — MAIN left set

Non-parametric VINN policy artifacts built from **this dataset's full 110,568 frames (131 episodes,
30fps)**. Two BYOL-finetuned DINOv3-ViT-S/16 encoders + their full-dataset kNN banks, drop-in for the
`visuomotor-policy-inference` kNN runner (`app/knn_runner.py`).

## Contents
```
vinn/
  encoders/
    vits16_byol_nomask/   # plain 2-aug BYOL (front,wrist)          <- RECOMMENDED (A/B winner)
    vits16_byol_mask/     # + (front <-> front*hand+hook-mask) background-invariance positive
  banks/
    left_nomask.npz       # emb(110568,768) act(110568,26) epi -- encoded by vits16_byol_nomask
    left_mask.npz         # encoded by vits16_byol_mask
```
Encoders: BYOL epoch-6, trained on stride-3 frames of the first 105 episodes (last 26 held out for the
eval below), last-2-blocks unfrozen. Banks: L2-normalized `concat(front_CLS, wrist_CLS)` for **all 131
episodes** + 26-d action + episode index. Masks came from `Ngseo/hyundai-uiwang-hook-segmentation`.

## Held-out kNN L1 (105-ep bank -> 26-ep query, lower is better)
| encoder            | k=1    | k=8    | k=32   |
|--------------------|--------|--------|--------|
| frozen DINOv3      | 0.0913 | 0.0773 | 0.0730 |
| **byol_nomask e6** | 0.0806 | 0.0694 | **0.0651** |
| byol_mask e6       | 0.0832 | 0.0725 | 0.0686 |

Plain BYOL cuts L1 ~11% vs frozen. The hand+hook background-invariance positive is consistently ~0.004
WORSE at every epoch (hand+hook is only ~2.4% of the frame — forcing invariance to the other 97.6%
discards retrieval-relevant context). **Use `vits16_byol_nomask` + `left_nomask.npz`.**

## Serve (chunk + temporal-ensembling VINN)
```bash
export VPI_POLICY=knn
export VPI_KNN_DINO=vinn/encoders/vits16_byol_nomask
export VPI_KNN_BANK=vinn/banks/left_nomask.npz
# runner defaults: k=32, chunk=8, temporal-ensemble -> single 26-d action per tick
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```
