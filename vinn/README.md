# VINN artifacts (BYOL-finetuned DINOv3-ViTS16 + kNN banks)

Non-parametric VINN policy artifacts built **from this dataset's 4640 frames** (100 episodes, 2fps).
Two BYOL-finetuned DINOv3-ViT-S/16 encoders + their kNN embedding banks, ready to drop into the
`visuomotor-policy-inference` kNN runner (`app/knn_runner.py`).

## Contents
```
vinn/
  encoders/
    vits16_byol_mask/     # BYOL finetune WITH (front <-> front_masked) background-invariance positive
    vits16_byol_nomask/   # BYOL finetune with plain front/wrist augmentation only (no mask)
  banks/
    teleop_mask.npz       # emb(4640,768) act(4640,26) epi(4640,)  -- encoded by vits16_byol_mask
    teleop_nomask.npz     # encoded by vits16_byol_nomask
```
Each encoder is a self-contained HF model dir (`AutoModel.from_pretrained` loads it). Banks store
L2-normalized `concat(front_CLS, wrist_CLS)` (768-d) + the 26-d action + episode index per frame.
Encoders are the epoch-30 checkpoints (best; still improving at 30 ep on this data).

## Held-out kNN L1 (80-ep bank -> 20-ep query), lower is better
| encoder        | k=1    | k=8    | k=32   |
|----------------|--------|--------|--------|
| frozen DINOv3  | 0.0135 | 0.0159 | 0.0207 |
| **byol_mask**  | 0.0080 | 0.0076 | 0.0091 |
| **byol_nomask**| 0.0073 | 0.0073 | 0.0093 |

BYOL cuts held-out L1 ~50% vs frozen. mask ≈ nomask (nomask marginally better at low k, mask at k=32):
the background-invariance positive adds no measurable benefit on this homogeneous set (same scene across
all episodes) — the gain is general BYOL domain adaptation. **nomask recommended for deployment.**

## Serve (chunk + temporal-ensembling VINN)
```bash
# after downloading this folder locally (huggingface-cli download ... --repo-type dataset)
export VPI_POLICY=knn
export VPI_KNN_DINO=vinn/encoders/vits16_byol_nomask      # or .../vits16_byol_mask
export VPI_KNN_BANK=vinn/banks/teleop_nomask.npz          # match the encoder
# runner defaults: k=32, chunk=8, temporal-ensemble -> single 26-d action per tick
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```
