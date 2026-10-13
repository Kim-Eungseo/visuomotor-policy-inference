"""Deployable VINN-style kNN policy runner (image-only) with chunk + temporal ensembling.

The kNN policy serves two artifacts (no trained action head):
  1. a frozen DINOv3 image encoder, and
  2. an embedding memory bank (.npz: emb (N,D) unit-norm, act (N,26), epi (N,) episode index).

Inference (per control tick): encode front+wrist -> kNN over the bank (embeddings are unit-norm,
so cosine topk == L2-distance topk) -> locally-weighted average of the neighbors' *future N-step
action chunks* (built at load time from act+epi). A small ring buffer of recent chunk predictions
is temporally ensembled so the action for the current frame is the (recency-weighted) average of
every overlapping chunk that targets it — the retrieval-native equivalent of RTC's smooth chunk
stitching. Set VPI_KNN_CHUNK=1 to fall back to plain single-step VINN.

Neighbor weighting (VPI_KNN_WEIGHT):
  softmin_l2 (default) — VINN paper III-B / Atkeson LWR: w_i = softmax(-||e - e_i||_2 / bw).
      Paper-exact is bw=1 (no bandwidth term). NOTE: with unit-norm embeddings d in [0,2], so
      exp(-d) weight ratios are <= e^2 (~7.4x) — much flatter than the legacy softmax.
      VPI_KNN_BW tunes the bandwidth the paper leaves implicit.
  softmax_cos — legacy: w_i = softmax(cos_sim / VPI_KNN_TEMP), temp 0.05 (sharply peaked).

Neighbor SELECTION (VPI_KNN_SELECT) — the RL v1 hook (see VINN_RL_PLAN.md):
  softmin (default) — execute the weighted AVERAGE of the top-k chunks (previous behavior).
  top1 / thompson — pick ONE candidate among the top-k and execute its chunk:
      score_i = exp(-||e-e_i||/bw) * v_i * prog_i
      v_i = posterior mean (top1) or Beta sample (thompson) from the per-entry value sidecar
      (VPI_KNN_VALUES npz: alpha, beta aligned to bank rows; missing file -> uniform prior 3/1),
      prog_i = VPI_KNN_PROG_GAMMA^backstep vs the last selected (episode, frame) — the learned-
      value generalization of the loop-break heuristic. If every candidate's posterior mean is
      below VPI_KNN_VMIN the tick falls back to softmin (safety net).
  Per-tick JSONL logs (candidates/distances/selection) go to VPI_KNN_LOG_DIR when set, one file
  per episode (rotated on reset()); update_values.py turns logs + episode labels into the sidecar.

Env: VPI_KNN_BANK, VPI_KNN_DINO, VPI_KNN_K (32), VPI_KNN_WEIGHT (softmin_l2), VPI_KNN_BW (1.0),
     VPI_KNN_TEMP (0.05, softmax_cos only), VPI_KNN_CHUNK (chunk length N, default 8),
     VPI_KNN_ENS_M (ensemble decay, default 0.1), VPI_KNN_SELECT (softmin), VPI_KNN_VALUES,
     VPI_KNN_VMIN (0.2), VPI_KNN_PROG_GAMMA (0.98), VPI_KNN_LOG_DIR.
"""
from __future__ import annotations

import json
import os
import time
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
        self.weight = os.environ.get("VPI_KNN_WEIGHT", "softmin_l2")      # paper III-B | "softmax_cos" legacy
        self.bw = float(os.environ.get("VPI_KNN_BW", "1.0"))              # softmin bandwidth; 1.0 = paper-exact
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
        # per-bank-frame provenance for retrieval visualization: episode id + position within episode.
        self.epi = epi.astype(np.int64)
        self.frame_in_epi = np.zeros(len(epi), np.int64)               # 0-based index of the frame in its episode
        _seen: dict[int, int] = {}
        for _i, _e in enumerate(self.epi.tolist()):
            _seen[_e] = _seen.get(_e, -1) + 1
            self.frame_in_epi[_i] = _seen[_e]
        self.last_retrieval: list[dict] | None = None                 # top-k neighbors of the most recent predict

        # --- RL v1: single-candidate selection + per-entry value sidecar (VINN_RL_PLAN.md) -----
        self.select = os.environ.get("VPI_KNN_SELECT", "softmin")     # softmin | top1 | thompson
        self.vmin = float(os.environ.get("VPI_KNN_VMIN", "0.2"))
        self.prog_gamma = float(os.environ.get("VPI_KNN_PROG_GAMMA", "0.98"))
        vpath = os.environ.get("VPI_KNN_VALUES", "")
        if vpath and os.path.exists(vpath):
            vz = np.load(vpath)
            self.val_a = np.asarray(vz["alpha"], np.float64)
            self.val_b = np.asarray(vz["beta"], np.float64)
            assert len(self.val_a) == self.n_bank, "value sidecar / bank size mismatch"
            print(f"[knn] value sidecar {vpath}: mean v={float((self.val_a/(self.val_a+self.val_b)).mean()):.3f}")
        else:                                                          # uniform demo prior
            self.val_a = np.full(self.n_bank, 3.0)
            self.val_b = np.ones(self.n_bank)
        self._last_sel: tuple[int, int] | None = None                  # (episode, frame) of last selection
        self.last_selected: int | None = None                          # global bank idx picked this tick
        self.log_dir = os.environ.get("VPI_KNN_LOG_DIR", "")
        self._log_f = None
        self._tick = 0
        # future N-step action chunk per bank frame (within the same episode; pad by hold-last)
        self.bank_fut = torch.tensor(self._build_future(act, epi, self.N), device=self.device)  # (M,N,26)

        # --- Version A: active loop-break (opt-in via VPI_KNN_LOOPBREAK) ------------------------
        # After grasping, the query aliases onto the dense grasp-HOLD frames and the top-1 match stops
        # advancing -> the robot idles in place. When the top-1 (episode, timestep) stays in a tiny band
        # for LB_WIN steps, we FORCE-ADVANCE: emit the future chunk from LB_ADV*escape timesteps later
        # in that same episode, pushing the robot along the demo until the observation changes and normal
        # retrieval resumes. Frames are contiguous per episode, so gidx(episode,frame)=start[e]+frame.
        self.loopbreak = os.environ.get("VPI_KNN_LOOPBREAK", "0").lower() in ("1", "true", "yes")
        self.lb_win = int(os.environ.get("VPI_KNN_LB_WIN", "6"))    # stuck if no advance over this many steps
        self.lb_adv = int(os.environ.get("VPI_KNN_LB_ADV", "4"))    # timesteps to jump forward per stuck step
        self.lb_tol = int(os.environ.get("VPI_KNN_LB_TOL", "3"))    # frame band width counted as "not advancing"
        self.epi_start = {int(e): int(np.where(self.epi == e)[0][0]) for e in np.unique(self.epi)}
        self.epi_len = {int(e): int((self.epi == e).sum()) for e in np.unique(self.epi)}
        self._recent: deque = deque(maxlen=self.lb_win)   # recent top-1 (epi, frame) matches
        self._escape = 0                                  # current forward advance (timesteps)
        self.last_escape: dict | None = None              # provenance of the last forced advance (or None)

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
        self._recent.clear(); self._escape = 0; self.last_escape = None   # reset loop-break state
        self._last_sel = None
        if self._log_f:                                   # rotate the per-episode selection log
            self._log_f.close(); self._log_f = None
        self._tick = 0

    def _log_tick(self, rec: dict) -> None:
        if not self.log_dir:
            return
        if self._log_f is None:
            os.makedirs(self.log_dir, exist_ok=True)
            self._log_f = open(os.path.join(self.log_dir, f"ep_{int(time.time() * 1000)}.jsonl"), "a")
        self._log_f.write(json.dumps(rec) + "\n"); self._log_f.flush()

    def _loopbreak(self, chunk, top_gidx):
        """If the top-1 match has been stuck in a tiny (episode, timestep) band, override `chunk` with
        the future chunk from further along that same episode (force progress). Returns the chunk to use."""
        e = int(self.epi[top_gidx]); f = int(self.frame_in_epi[top_gidx])
        self._recent.append((e, f))
        stuck = False
        if len(self._recent) == self._recent.maxlen:
            es = {p[0] for p in self._recent}; fs = [p[1] for p in self._recent]
            stuck = (len(es) == 1) and (max(fs) - min(fs) <= self.lb_tol)   # same episode, not advancing
        if not stuck:
            self._escape = 0; self.last_escape = None
            return chunk
        self._escape += self.lb_adv
        tgt = min(f + self._escape, self.epi_len[e] - 1)                    # advance along the demo
        gfwd = self.epi_start[e] + tgt
        self.last_escape = {"epi": e, "from": f, "to": tgt, "advance": self._escape}
        return self.bank_fut[gfwd].float().cpu().numpy()

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
        vals, idx = torch.topk(q @ self.bank.T, self.k, dim=1)      # (1,k) cosine; same ranking as L2
        if self.weight == "softmax_cos":                            # legacy sharp weighting
            w = torch.softmax(vals / self.temp, dim=1)              # (1,k)
        else:                                                       # VINN III-B: softmin of L2 distances
            d = (2.0 - 2.0 * vals).clamp_min(0).sqrt()              # unit-norm: ||e-e_i|| = sqrt(2-2cos)
            w = torch.softmax(-d / self.bw, dim=1)                  # (1,k); bw=1 -> exp(-d)/sum exp(-d)

        _ii = idx[0].detach().cpu().numpy(); _ww = w[0].detach().cpu().numpy()
        dnp = np.sqrt(np.clip(2.0 - 2.0 * vals[0].detach().cpu().numpy(), 0, None))
        self.last_selected, fallback = None, False
        if self.select != "softmin":                # RL v1: pick ONE candidate by dist x value x progress
            pm = self.val_a[_ii] / (self.val_a[_ii] + self.val_b[_ii])
            if pm.max() < self.vmin:                # every candidate has a bad record -> softmin safety net
                fallback = True
            else:
                v = np.random.beta(self.val_a[_ii], self.val_b[_ii]) if self.select == "thompson" else pm
                prog = np.ones(len(_ii))
                if self._last_sel is not None:      # penalize going backwards along the last-followed demo
                    le, lf = self._last_sel
                    back = np.maximum(0, lf - self.frame_in_epi[_ii])
                    prog = np.where(self.epi[_ii] == le, self.prog_gamma ** back, 1.0)
                score = np.exp(-dnp / self.bw) * v * prog
                j = int(score.argmax())
                self.last_selected = int(_ii[j])
                self._last_sel = (int(self.epi[_ii[j]]), int(self.frame_in_epi[_ii[j]]))
        if self.last_selected is not None:
            chunk = self.bank_fut[self.last_selected].float().cpu().numpy()            # (N,26) one entry
        else:
            chunk = (w[0][:, None, None] * self.bank_fut[idx[0]]).sum(0).float().cpu().numpy()

        # record which bank frames were retrieved this step (for the loop-visualization graph):
        # top-k neighbors ordered by weight, each tagged with its source episode + frame-in-episode.
        self.last_retrieval = [
            {"index": int(_ii[j]), "weight": float(_ww[j]),
             "epi": int(self.epi[_ii[j]]), "frame": int(self.frame_in_epi[_ii[j]]),
             "selected": self.last_selected == int(_ii[j])}
            for j in range(len(_ii))
        ]
        self._tick += 1
        self._log_tick({"t": self._tick, "mode": self.select, "sel": self.last_selected,
                        "fallback": fallback, "cand": [int(x) for x in _ii],
                        "dist": [round(float(x), 4) for x in dnp]})

        if self.loopbreak:                          # Version A: force-advance out of a detected idle-loop
            chunk = self._loopbreak(chunk, self.last_selected if self.last_selected is not None else int(_ii[0]))

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
            "knn": {"encoder": self.dino, "bank_size": self.n_bank, "k": self.k,
                    "select": self.select,
                    **({"prog_gamma": self.prog_gamma, "vmin": self.vmin,
                        "values_mean": float((self.val_a / (self.val_a + self.val_b)).mean())}
                       if self.select != "softmin" else {}),
                    "weighting": self.weight,
                    **({"bandwidth": self.bw} if self.weight != "softmax_cos" else {"temp": self.temp}),
                    "chunk": self.N, "ensemble_decay": self.ens_m,
                    "mode": "single-step" if self.N == 1 else "chunk+temporal-ensembling",
                    "loopbreak": ({"win": self.lb_win, "adv": self.lb_adv, "tol": self.lb_tol}
                                  if self.loopbreak else False)},
        }
