#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0
cd /mnt/d/Projects/yolo-seg-hailo
# shellcheck disable=SC1091
source .wsl-hailo/bin/activate
python - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
cd dataset-yolo
yolo export \
  model=../runs/segment/yolov8n-seg-hailo8/weights/best.pt \
  format=hailo \
  name=hailo8 \
  imgsz=640 \
  data=data.yaml \
  2>&1 | tee ../runs/segment/yolov8n-seg-hailo8/weights/hailo_export.log
echo "EXIT:${PIPESTATUS[0]}"
