#!/usr/bin/env bash
set -euo pipefail

# Run this from the diffusion-pipe-stereo directory, or pass it explicitly:
#   WORKSPACE=/path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo bash scripts/setup_remote_env.sh

WORKSPACE="${WORKSPACE:-$(pwd)}"
cd "$WORKSPACE"

python -m py_compile \
  train.py \
  utils/dataset.py \
  models/wan_stereo.py \
  tools/wan_stereo_dataset_qa.py \
  tools/prepare_stereo4d_hf_smoke.py \
  tools/export_wan_stereo_lora.py

python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
PY

python - <<'PY'
from models.wan_stereo import WanStereoPipeline
print('pipeline', WanStereoPipeline.name)
PY
