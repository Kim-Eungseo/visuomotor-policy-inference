"""Deployable VINN-style kNN policy runner (image-only) with chunk + temporal ensembling.

The kNN policy serves two artifacts (no trained action head):
  1. a frozen DINOv3 image encoder, and
  2. an embedding memory bank (.npz: emb (N,D) unit-norm, act (N,26), epi (N,) episode index).

Inference (per control tick): encode front+wrist -> cosine kNN over the bank -> locally-
weighted average of the neighbors' *future N-step action chunks* (built at load time from
act+epi). A small ring buffer of recent chunk predictions is temporally ensembled so the
action for the current frame is the (recency-weighted) average of every overlapping chunk
that targets it — the retrieval-native equivalent of RTC's smooth chunk stitching.
Set VPI_KNN_CHUNK=1 to fall back to plain single-step VINN.

Env: VPI_KNN_BANK, VPI_KNN_DINO, VPI_KNN_K (32), VPI_KNN_TEMP (0.05),
     VPI_KNN_CHUNK (chunk length N, default 8), VPI_KNN_ENS_M (ensemble decay, default 0.1).
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


class KNNPolicyRunner:
    def __init__(self, bank_path: str | None = None, dino: str | None = None,
                 device: str | None = None, k: int | None = None, temp: float | None = None):
        # Default to the lighter DINOv3-ViT-S/16: ~same kNN L1 as ViT-B/16 but ~2x faster.
        self.dino = dino or os.environ.get("VPI_KNN_DINO", "facebook/dinov3-vits16-pretrain-lvd1689m")
        dev = device or os.environ.get("VPI_DEVICE", "cuda")
        self.device = dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu"
        self.k = int(k or os.environ.get("VPI_KNN_K", "32"))
        self.temp = float(temp or os.environ.get("VPI_KNN_TEMP", "0.05"))
        self.N = max(1, int(os.environ.get("VPI_KNN_CHUNK", "8")))       # chunk length (1 = single-step)
        self.ens_m = float(os.environ.get("VPI_KNN_ENS_M", "0.1"))       # temporal-ensemble recency decay
        # which camera(s) to encode into the query embedding — must match how the bank was built.
        # e.g. VPI_KNN_CAMS=wrist  -> wrist-only 768-d ; default front,wrist -> concatenated 1536-d.
        self.cams = [c.strip() for c in os.environ.get("VPI_KNN_CAMS", "front,wrist").split(",") if c.strip()]

        bank_path = bank_path or os.environ.get("VPI_KNN_BANK", "knn_bank_left_vits16.npz")
        z = np.load(bank_path)
        self.bank = torch.tensor(z["emb"], device=self.device)          # (M, D) L2-normalized
        act = np.asarray(z["act"], dtype=np.float32)                    # (M, 26)
        epi = np.asarray(z["epi"]).reshape(-1) if "epi" in z else np.zeros(len(act), np.int64)
        self.action_dim = int(act.shape[1])
        self.n_bank = int(self.bank.shape[0])
        # future N-step action chunk per bank frame (within the same episode; pad by hold-last)
        self.bank_fut = torch.tensor(self._build_future(act, epi, self.N), device=self.device)  # (M,N,26)

        self.model = AutoModel.from_pretrained(self.dino).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._wd = np.exp(-self.ens_m * np.arange(self.N)).astype(np.float32)  # ensemble weights by offset
        self._hist: deque = deque(maxlen=self.N)   # recent per-query chunk predictions (N,26) each

    @staticmethod
    def _build_future(act, epi, N):
        M, A = act.shape
        fut = np.repeat(act[:, None, :], N, axis=1)   # start = hold-last fallback
        for d in range(1, N):
            idx = np.minimum(np.arange(M) + d, M - 1)
            same = (np.arange(M) + d < M) & (epi[idx] == epi)
            fut[:, d, :] = np.where(same[:, None], act[idx], fut[:, d - 1, :])
        return fut

    def reset(self) -> None:
        self._hist.clear()   # new episode -> no cross-episode ensembling

    @torch.no_grad()
    def _embed(self, img_uint8_hwc: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(img_uint8_hwc)).float().div(255).permute(2, 0, 1).unsqueeze(0)
        x = F.interpolate(x.to(self.device), size=(224, 224), mode="bilinear", align_corners=False)
        x = (x.clamp(0.0, 1.0) - self._mean) / self._std
        return F.normalize(self.model(pixel_values=x).last_hidden_state[:, 0], dim=-1)

    @torch.no_grad()
    def predict(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray,
                state=None, gripper_sensor=None, wrist_ft_sensor=None, extra_inputs=None) -> np.ndarray:
        # image-only: state/gripper/ft accepted (server drop-in) but ignored.
        imgs = {"front": front_rgb, "wrist": wrist_rgb}
        q = F.normalize(torch.cat([self._embed(imgs[c]) for c in self.cams], dim=-1), dim=-1)
        vals, idx = torch.topk(q @ self.bank.T, self.k, dim=1)      # (1,k)
        w = torch.softmax(vals / self.temp, dim=1)                  # (1,k)
        chunk = (w[0][:, None, None] * self.bank_fut[idx[0]]).sum(0).float().cpu().numpy()  # (N,26)

        if self.N == 1:
            return chunk[0]
        self._hist.append(chunk)
        # temporal ensemble: current frame = recency-weighted avg over d of hist[-1-d][d]
        num = np.zeros(self.action_dim, np.float32)
        den = 0.0
        for d, pc in enumerate(reversed(self._hist)):   # d=0 -> newest query (step 0 targets now)
            num += self._wd[d] * pc[d]
            den += self._wd[d]
        return num / max(den, 1e-8)

    def info(self) -> dict:
        return {
            "model_id": f"knn-vinn:{self.dino.split('/')[-1]}",
            "device": self.device,
            "cameras": [f"observation.images.{c}_rgb" for c in self.cams],
            "state_dim": 0,
            "action_dim": self.action_dim,
            "scheduler": "knn-vinn",
            "num_inference_steps": None,
            "knn": {"encoder": self.dino, "bank_size": self.n_bank, "k": self.k, "temp": self.temp,
                    "chunk": self.N, "ensemble_decay": self.ens_m,
                    "mode": "single-step" if self.N == 1 else "chunk+temporal-ensembling"},
        }
