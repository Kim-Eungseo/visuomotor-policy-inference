#!/usr/bin/env python3
"""RF-DETR 2-class (hand + hook) segmentation server (GPU). Loads RFDETRSeg2XLarge(num_classes=2)
once on CUDA and serves:

    POST /segment  {image: base64 jpeg}  ->  {overlay: base64 png, counts:{hand,hook}, n_masks, ms}
    GET  /health                          ->  {ok, device, classes}

Classes: 0 = hand (green overlay), 1 = hook (magenta). Run inside the inference image (CUDA torch);
rfdetr is pip-installed at container start. See ../seg_server.sh. The GUI polls this.
"""
from __future__ import annotations

import base64
import io
import os
import time

import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

WEIGHTS = os.environ.get("SEG_WEIGHTS", "/weights/best_2cls_ver1.pth")
NUM_CLASSES = int(os.environ.get("SEG_NUM_CLASSES", "2"))
THRESHOLD = float(os.environ.get("SEG_THRESHOLD", "0.30"))
CLASS_NAMES = {0: "hand", 1: "hook"}
CLASS_COLORS = {0: np.array([0, 255, 0], np.float32),      # hand = green
                1: np.array([200, 50, 255], np.float32)}   # hook = magenta

app = FastAPI(title="RF-DETR 2-class segmentation", version="2.0.0")
_state = {"model": None, "device": "cpu"}


@app.on_event("startup")
def _load():
    import torch
    from rfdetr import RFDETRSeg2XLarge
    _state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    model = RFDETRSeg2XLarge(num_classes=NUM_CLASSES, pretrain_weights=WEIGHTS)
    # NOTE: skipping model.optimize_for_inference() — its TorchScript trace adds ~2 min to startup and
    # only speeds up the ~148 ms inference, which is NOT the bottleneck (image encoding was). Not worth it.
    # GPU warm-up (first predict compiles CUDA kernels, ~3 s)
    warm = Image.fromarray(np.zeros((480, 640, 3), np.uint8))
    try:
        with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
            model.predict(warm, threshold=THRESHOLD)
    except Exception:
        pass
    _state["model"] = model
    print(f"[seg] 2-class model loaded on {_state['device']} (classes={CLASS_NAMES})", flush=True)


class SegRequest(BaseModel):
    image: str                                        # base64 JPEG/PNG (RGB)


def _decode(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def _encode_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _encode_jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    # JPEG is ~16x faster to encode than PNG on a full-res frame (236ms->14ms) and 4x smaller —
    # the live overlay's bottleneck was PNG encoding, not the GPU. Fine for a display overlay.
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"ok": _state["model"] is not None, "device": _state["device"],
            "classes": CLASS_NAMES, "threshold": THRESHOLD}


def _predict_masks(rgb: np.ndarray):
    """Run the model -> (per-class boolean masks {cid: HxW}, union boolean mask HxW, counts)."""
    import torch
    H, W = rgb.shape[:2]
    with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
        r = _state["model"].predict(Image.fromarray(rgb), threshold=THRESHOLD)
    class_masks: dict[int, np.ndarray] = {}
    counts: dict[str, int] = {}
    union = np.zeros((H, W), bool)
    masks = getattr(r, "mask", None)
    if masks is not None and len(masks) > 0:
        masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        cls = getattr(r, "class_id", None)
        cls = (cls.cpu().numpy() if hasattr(cls, "cpu") else np.asarray(cls)) if cls is not None \
            else np.zeros(len(masks_np))
        for cid in np.unique(cls).astype(int):
            idx = cls == cid
            m = np.any(masks_np[idx], axis=0)
            if m.shape[:2] != (H, W):                 # RF-DETR masks may be low-res -> nearest upsample
                m = np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize((W, H), Image.NEAREST)) > 127
            class_masks[int(cid)] = m
            union |= m
            counts[CLASS_NAMES.get(int(cid), f"class{cid}")] = int(idx.sum())
    return class_masks, union, counts


@app.post("/segment")
def segment(req: SegRequest):
    if _state["model"] is None:
        return {"overlay": None, "counts": {}, "n_masks": 0, "ms": 0.0, "error": "model not loaded"}
    rgb = _decode(req.image)
    t0 = time.time()
    class_masks, _union, counts = _predict_masks(rgb)
    overlay = rgb.copy()
    for cid, m in class_masks.items():
        color = CLASS_COLORS.get(int(cid), np.array([250, 250, 80], np.float32))
        overlay[m] = (0.5 * overlay[m] + 0.5 * color).astype(np.uint8)
    return {"overlay": _encode_jpeg(overlay), "counts": counts,   # JPEG: PNG encode was the bottleneck
            "n_masks": int(sum(counts.values())), "ms": (time.time() - t0) * 1000.0}


@app.post("/segment_masked")
def segment_masked(req: SegRequest):
    """Keep only the hand+hook pixels (black out the rest). If nothing is detected, the ORIGINAL frame
    is returned unchanged (so a missed detection never destroys a frame). Used by the GUI post-processor."""
    if _state["model"] is None:
        return {"masked": None, "counts": {}, "n_masks": 0, "ms": 0.0, "error": "model not loaded"}
    rgb = _decode(req.image)
    t0 = time.time()
    _cm, union, counts = _predict_masks(rgb)
    if union.any():
        masked = np.zeros_like(rgb); masked[union] = rgb[union]
    else:
        masked = rgb                                   # no detection -> keep original frame
    return {"masked": _encode_png(masked), "counts": counts, "detected": bool(union.any()),
            "n_masks": int(sum(counts.values())), "ms": (time.time() - t0) * 1000.0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("SEG_PORT", "8100")))
