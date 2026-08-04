"""Step B: load an emb npz (from tsne_encode.py) -> PCA -> t-SNE -> scatter colored by within-episode
progress (start=red -> end=purple). NO torch import (runs in a CUDA-free process, so sklearn/joblib
can't deadlock on fork+CUDA)."""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--max_tsne", type=int, default=30000)
    args = ap.parse_args()

    z = np.load(args.npz)
    emb = z["emb"].astype(np.float32)
    epi, fi = z["epi"], z["fi"]
    n_ep = int(z["n_ep"]) if "n_ep" in z else len(np.unique(epi))
    print(f"loaded {emb.shape} from {args.npz} | {n_ep} eps", flush=True)

    prog = np.zeros(len(fi), np.float32)
    for e in np.unique(epi):
        m = epi == e
        mx = fi[m].max()
        prog[m] = fi[m] / mx if mx > 0 else 0.0

    idx = np.arange(len(emb))
    if len(emb) > args.max_tsne:
        step = len(emb) / args.max_tsne
        idx = np.unique((np.arange(args.max_tsne) * step).astype(int))
        print(f"  t-SNE on stratified {len(idx)}/{len(emb)} frames", flush=True)

    X = PCA(n_components=50, random_state=0).fit_transform(emb[idx])
    print("  PCA done, running t-SNE...", flush=True)
    Y = TSNE(n_components=2, init="pca", perplexity=30, max_iter=1000,
             random_state=0, n_jobs=1).fit_transform(X)
    print("  t-SNE done, plotting...", flush=True)

    small = len(idx) < 8000
    plt.figure(figsize=(11, 9))
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=prog[idx], cmap="rainbow_r",
                     s=10 if small else 4, alpha=0.6 if small else 0.4, linewidths=0)
    cb = plt.colorbar(sc)
    cb.set_label("trajectory progress (start → end)", fontsize=12)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["start (red)", "end (purple)"])
    plt.title(f"{args.title}\nfrozen DINOv3-ViTS16 (front+wrist) · t-SNE · {len(idx)} frames / {n_ep} eps", fontsize=12)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
