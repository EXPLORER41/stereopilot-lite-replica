#!/usr/bin/env bash
set -euo pipefail

fail=0
check_dir() {
  local path="$1"
  if [ -d "$path" ]; then
    echo "ok dir $path"
  else
    echo "ERROR missing directory: $path" >&2
    fail=1
  fi
}
check_file() {
  local path="$1"
  if [ -f "$path" ]; then
    echo "ok file $path"
  else
    echo "ERROR missing file: $path" >&2
    fail=1
  fi
}

for d in   StereoPilot StereoPilot/asset StereoPilot/ckpt StereoPilot/ckpt/Wan2.1-T2V-1.3B StereoPilot/output StereoPilot/stereo_output   StereoPilot/models StereoPilot/sample StereoPilot/submodules/Wan2_1 StereoPilot/toml StereoPilot/utils StereoPilot/StereoPilot_Dataprocess   diffusion-pipe-stereo docs scripts models models/Wan2.1-T2V-1.3B data datasets runs stereopilot_exports; do
  check_dir "$d"
done

for f in   StereoPilot/sample.py   StereoPilot/models/StereoPilot.py   StereoPilot/asset/StereoPilot_logo.png   StereoPilot/sample/sample1.mp4   StereoPilot/ckpt/README.md   diffusion-pipe-stereo/train.py   diffusion-pipe-stereo/models/wan_stereo.py   diffusion-pipe-stereo/tools/prepare_stereo4d_hf_smoke.py   diffusion-pipe-stereo/tools/wan_stereo_dataset_qa.py   diffusion-pipe-stereo/tools/export_wan_stereo_lora.py   docs/workspace_layout.md; do
  check_file "$f"
done

echo "[expected external artifacts]"
for f in   StereoPilot/ckpt/StereoPilot.safetensors   StereoPilot/ckpt/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors; do
  if [ -e "$f" ]; then
    echo "present external artifact: $f"
  else
    echo "missing external artifact, expected after manual download if needed: $f"
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "workspace layout check failed" >&2
  exit 1
fi

echo "workspace layout skeleton passed"
