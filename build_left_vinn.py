"""Package the MAIN-left VINN artifacts: export encoders (BYOL e6 pair + InfoNCE e2 winner) as
self-contained HF model dirs, and convert their cached 110k-frame embedding npz
(from left_knn_eval.py) into serving banks (emb fp32 / act / epi — the knn_runner format).

  conda run -n lr python build_left_vinn.py            # all configs
  conda run -n lr python build_left_vinn.py infonce    # only configs whose ckpt matches
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModel

DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"
STAGE = "left_vinn"
CONFIGS = [  # (ckpt, cache npz from eval, encoder out, bank out)
    ("byol_left_e6.pt",      "/tmp/emb_left_byol_left_e6.npz",
     f"{STAGE}/encoders/vits16_byol_nomask", f"{STAGE}/banks/left_nomask.npz"),
    ("byol_left_mask_e6.pt", "/tmp/emb_left_byol_left_mask_e6.npz",
     f"{STAGE}/encoders/vits16_byol_mask",   f"{STAGE}/banks/left_mask.npz"),
    ("infonce_left_e2.pt",   "/tmp/emb_left_infonce_left_e2.npz",
     f"{STAGE}/encoders/vits16_infonce_e2",  f"{STAGE}/banks/left_infonce.npz"),
    ("infonce_mask12_left_e6.pt", "/tmp/emb_left_infonce_mask12_left_e6.npz",
     f"{STAGE}/encoders/vits16_infonce_mask12_e6", f"{STAGE}/banks/left_infonce_mask12.npz"),
]
if len(sys.argv) > 1:
    CONFIGS = [c for c in CONFIGS if sys.argv[1] in c[0]]

for ckpt, cache, enc_out, bank_out in CONFIGS:
    print(f"=== {ckpt} ===")
    model = AutoModel.from_pretrained(DINO)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("backbone", sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"state_dict mismatch: {len(miss)}/{len(unexp)}"
    os.makedirs(enc_out, exist_ok=True)
    model.save_pretrained(enc_out)
    print(f"  encoder -> {enc_out}")

    z = np.load(cache)
    emb = z["emb"].astype(np.float32)           # fp16 cache -> fp32 (runner matmuls in fp32)
    act, epi = z["act"].astype(np.float32), z["epi"].astype(np.int64)
    assert emb.shape[0] == act.shape[0] == epi.shape[0] == 110568, emb.shape
    os.makedirs(os.path.dirname(bank_out), exist_ok=True)
    np.savez(bank_out, emb=emb, act=act, epi=epi)
    print(f"  bank -> {bank_out}: emb{emb.shape} act{act.shape} eps={len(np.unique(epi))}")

print("BUILD DONE")
