#!/usr/bin/env bash
# (Re)deploy the VINN (kNN) inference server on :8000 with a BYOL-finetuned DINOv3 encoder + bank.
# The knn runner encodes concat(front_CLS, wrist_CLS) (768-d) and retrieves 26-d action chunks.
#
#   ./vinn_server.sh              # default: BYOL mask encoder + teleop_mask.npz
#   ./vinn_server.sh nomask       # BYOL nomask encoder + teleop_nomask.npz
#
# Artifacts live under ~/.cache/huggingface/vinn (mounted at /hf/vinn). Rebuild a bank with
# knn_build_bank.py. This replaces any running vpi_inference container.
set -uo pipefail

VARIANT="${1:-mask}"    # mask | nomask | pruned | loopbreak
# mask/nomask now point at the FULL-dataset artifacts (110,568 frames / 131 ep, ep28 removed) from
# Ngseo/hyundai-uiwang-left:vinn/. (Old 4640-frame teleop_*.npz + vits16_byol_* still in vinn/ if needed.)
# Two INDEPENDENT anti-idle-loop experiments (test one at a time, A/B against `mask`):
#   pruned    : same encoder, grasp/hold-pruned bank (left_mask_pruned.npz). loop-break OFF.
#   loopbreak : same (ep28-removed) bank, active loop-break ON (force-advance on detected idle-loop).
LOOPBREAK=0
case "$VARIANT" in
  mask)      ENC=left_mask;          BANK=left_mask.npz;;
  nomask)    ENC=vits16_byol_nomask; BANK=teleop_nomask.npz;;
  pruned)    ENC=left_mask;          BANK=left_mask_pruned.npz;;
  loopbreak) ENC=left_mask;          BANK=left_mask.npz; LOOPBREAK=1;;
  *) echo "usage: $0 {mask|nomask|pruned|loopbreak}"; exit 1;;
esac

HF="$HOME/.cache/huggingface"
K="${VPI_KNN_K:-32}"                    # full-dataset banks -> README recommends k=32
CHUNK="${VPI_KNN_CHUNK:-8}"   # 8 = chunk + temporal-ensembling (README default); 1 = single-step VINN
LOOPBREAK="${VPI_KNN_LOOPBREAK:-$LOOPBREAK}"   # 1 = Version A active loop-break (only the loopbreak variant)

echo "[vinn] deploying knn server: encoder=$ENC bank=$BANK k=$K chunk=$CHUNK loopbreak=$LOOPBREAK"
# The forked lerobot diffusion policy (use_robot_state / image-only support) needed to load the
# Ngseo *-imgonly diffusion checkpoints. Overlaid onto the image's pip lerobot so /load works.
LR_FORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/system_Teleop/src/System_/lerobot_system/lerobot/src/lerobot/policies/diffusion"
LR_PKG="/usr/local/lib/python3.11/site-packages/lerobot/policies/diffusion"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker rm -f vpi_inference >/dev/null 2>&1 || true
docker run -d --name vpi_inference \
  --network host --gpus all --restart unless-stopped \
  -v "$HF:/hf" \
  -v "$REPO/app:/srv/app:ro" \
  --workdir /srv \
  -v "$LR_FORK/configuration_diffusion.py:$LR_PKG/configuration_diffusion.py:ro" \
  -v "$LR_FORK/modeling_diffusion.py:$LR_PKG/modeling_diffusion.py:ro" \
  -e VPI_DEVICE=cuda \
  -e VPI_POLICY=knn \
  -e VPI_KNN_DINO="/hf/vinn/encoders/$ENC" \
  -e VPI_KNN_BANK="/hf/vinn/banks/$BANK" \
  -e VPI_KNN_K="$K" \
  -e VPI_KNN_CHUNK="$CHUNK" \
  -e VPI_KNN_LOOPBREAK="$LOOPBREAK" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  visuomotor-policy-inference:latest \
  uvicorn app.server:app --host 0.0.0.0 --port 8000 >/dev/null

echo -n "[vinn] waiting for model load "
for i in $(seq 1 40); do
  curl -s --max-time 3 localhost:8000/info 2>/dev/null | grep -q knn-vinn && { echo " ready"; break; }
  echo -n "."; sleep 3; [ "$i" -eq 40 ] && { echo " NOT ready (docker logs vpi_inference)"; exit 1; }
done
curl -s --max-time 4 localhost:8000/info 2>/dev/null | python3 -m json.tool
