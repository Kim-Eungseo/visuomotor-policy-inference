"""t-SNE of frozen-DINOv3 image embeddings, colored by within-episode progress (start=red -> end=purple).

Encodes EVERY frame's (front, wrist) CLS with a frozen DINOv3-ViT-S/16 (concat, L2-norm) -- the same
embedding the VINN bank uses -- then t-SNEs to 2D and scatters with a red->purple trajectory-progress
colormap so trajectory structure/similarity is visible.

  conda run -n lr python tsne_traj.py --repo <id> --root <path> --out <png> --title "<t>"
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def pick_keys(cam_keys):
    front = next(k for k in cam_keys if "front" in k and "mask" not in k)
    wrist = next(k for k in cam_keys if "wrist" in k)
    return front, wrist


@torch.no_grad()
def enc(model, x):
    x = F.interpolate(x.to(DEV), size=(224, 224), mode="bilinear", align_corners=False)
    x = (x.clamp(0, 1) - MEAN) / STD
    return F.normalize(model(pixel_values=x).last_hidden_state[:, 0], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--max_tsne", type=int, default=30000, help="stratified subsample cap for t-SNE/plot")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo, root=args.root, revision="main")
    fkey, wkey = pick_keys(ds.meta.camera_keys)
    print(f"{args.repo}: {len(ds)} frames, {ds.num_episodes} eps | keys: {fkey}, {wkey}")

    model = AutoModel.from_pretrained(DINO).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad = False

    dl = DataLoader(ds, batch_size=128, num_workers=4, shuffle=False)
    embs, epis, fidx = [], [], []
    done = 0
    for nb, b in enumerate(dl):
        e = F.normalize(torch.cat([enc(model, b[fkey]), enc(model, b[wkey])], dim=-1), dim=-1)
        embs.append(e.cpu().numpy().astype(np.float16))
        epis.append(np.asarray(b["episode_index"]).reshape(-1))
        fidx.append(np.asarray(b["frame_index"]).reshape(-1))
        done += e.shape[0]
        if nb % 5 == 0:
            print(f"  encoded {done}/{len(ds)}", flush=True)
    emb = np.concatenate(embs).astype(np.float32)
    epi = np.concatenate(epis)
    fi = np.concatenate(fidx)
    del model
    torch.cuda.empty_cache()   # release CUDA before sklearn (avoid joblib fork+CUDA deadlock)

    # within-episode progress in [0,1]
    prog = np.zeros(len(fi), np.float32)
    for e in np.unique(epi):
        m = epi == e
        mx = fi[m].max()
        prog[m] = fi[m] / mx if mx > 0 else 0.0

    # stratified subsample for legibility/speed (encoding stays on ALL frames)
    idx = np.arange(len(emb))
    if len(emb) > args.max_tsne:
        step = len(emb) / args.max_tsne
        idx = np.unique((np.arange(args.max_tsne) * step).astype(int))
        print(f"  t-SNE on stratified {len(idx)}/{len(emb)} frames (every ~{step:.1f}th)")

    X = PCA(n_components=50, random_state=0).fit_transform(emb[idx])
    Y = TSNE(n_components=2, init="pca", perplexity=30, max_iter=1000,
             random_state=0, n_jobs=1).fit_transform(X)

    plt.figure(figsize=(11, 9))
    s = 10 if len(idx) < 8000 else 4
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=prog[idx], cmap="rainbow_r", s=s,
                     alpha=0.6 if len(idx) < 8000 else 0.4, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label("trajectory progress  (start → end)", fontsize=12)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["start (red)", "end (purple)"])
    plt.title(f"{args.title}\nfrozen DINOv3-ViTS16 (front+wrist) · t-SNE · {len(idx)} frames / {ds.num_episodes} eps",
              fontsize=12)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
