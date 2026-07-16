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

VARIANT="${1:-mask}"    # mask | nomask
case "$VARIANT" in
  mask)   ENC=vits16_byol_mask;   BANK=teleop_mask.npz;;
  nomask) ENC=vits16_byol_nomask; BANK=teleop_nomask.npz;;
  *) echo "usage: $0 {mask|nomask}"; exit 1;;
esac

HF="$HOME/.cache/huggingface"
K="${VPI_KNN_K:-4}"
CHUNK="${VPI_KNN_CHUNK:-1}"   # 1 = single-step VINN (predict only the current step, no chunk/temporal-ensemble)

echo "[vinn] deploying knn server: encoder=$ENC bank=$BANK k=$K chunk=$CHUNK"
docker rm -f vpi_inference >/dev/null 2>&1 || true
docker run -d --name vpi_inference \
  --network host --gpus all --restart unless-stopped \
  -v "$HF:/hf" \
  -e VPI_DEVICE=cuda \
  -e VPI_POLICY=knn \
  -e VPI_KNN_DINO="/hf/vinn/encoders/$ENC" \
  -e VPI_KNN_BANK="/hf/vinn/banks/$BANK" \
  -e VPI_KNN_K="$K" \
  -e VPI_KNN_CHUNK="$CHUNK" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  visuomotor-policy-inference:latest \
  uvicorn app.server:app --host 0.0.0.0 --port 8000 >/dev/null

echo -n "[vinn] waiting for model load "
for i in $(seq 1 40); do
  curl -s --max-time 3 localhost:8000/info 2>/dev/null | grep -q knn-vinn && { echo " ready"; break; }
  echo -n "."; sleep 3; [ "$i" -eq 40 ] && { echo " NOT ready (docker logs vpi_inference)"; exit 1; }
done
curl -s --max-time 4 localhost:8000/info 2>/dev/null | python3 -m json.tool
