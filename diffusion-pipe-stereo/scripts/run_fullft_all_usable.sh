#!/usr/bin/env bash
set -eo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate StereoPilot

REPO=/path/to/workspace/diffusion-pipe-stereo
CHECK_DIR=/path/to/workspace/runs/fullft_checks
EXPORT_DIR=/path/to/workspace/stereopilot_exports/fullft
DATA_ROOT=/path/to/workspace/data/stereo4d_hf_all_usable
CONFIG=examples/wan_stereo_fullft_all_usable.toml
RUN_ROOT=/path/to/workspace/runs/wan_stereo_fullft_all_usable

mkdir -p "$CHECK_DIR" "$EXPORT_DIR" "$RUN_ROOT"
cd "$REPO"

echo "=== $(date '+%F %T') QA all usable data ==="
python tools/wan_stereo_dataset_qa.py \
  --right "$DATA_ROOT/right" \
  --left "$DATA_ROOT/left" \
  --min-frames 81 \
  --min-width 512 \
  --min-height 288 \
  --json "$CHECK_DIR/stereo4d_hf_all_usable_fullft_qa.json"

echo "=== $(date '+%F %T') cache latents/text ==="
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$CONFIG" \
  --cache_only --regenerate_cache

echo "=== $(date '+%F %T') full-parameter training ==="
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  deepspeed --num_gpus=1 train.py --deepspeed \
  --config "$CONFIG" \
  --trust_cache

echo "=== $(date '+%F %T') locate latest run ==="
RUN_DIR=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "$RUN_DIR" || ! -f "$RUN_DIR/epoch1/model.safetensors" ]]; then
  echo "Could not locate finished fullft all-usable model.safetensors under $RUN_ROOT" >&2
  exit 1
fi
echo "RUN_DIR=$RUN_DIR"
ls -lh "$RUN_DIR/epoch1/model.safetensors"

echo "=== $(date '+%F %T') export official StereoPilot safetensors ==="
python tools/export_wan_stereo_full_model.py \
  --input "$RUN_DIR/epoch1" \
  --output "$EXPORT_DIR/StereoPilot_fullft_all_usable.safetensors" \
  --reference /path/to/workspace/StereoPilot/ckpt/StereoPilot.safetensors \
  --save-dtype reference \
  --overwrite

cat > "$EXPORT_DIR/infer_fullft_all_usable.toml" <<'TOML'
# Inference configuration file

[model]
type = 'stereopilot'
ckpt_path = '/path/to/workspace/StereoPilot/ckpt/Wan2.1-T2V-1.3B'
transformer_path = '/path/to/workspace/stereopilot_exports/fullft/StereoPilot_fullft_all_usable.safetensors'
pretrained_path = '/path/to/workspace/stereopilot_exports/fullft/StereoPilot_fullft_all_usable.safetensors'
dtype = 'bfloat16'
transformer_dtype = 'float8'
TOML

echo "=== $(date '+%F %T') official StereoPilot inference smoke ==="
mkdir -p "$EXPORT_DIR/fullft_all_usable_output"
cd /path/to/workspace/StereoPilot
CUDA_VISIBLE_DEVICES=0 python sample.py \
  --config "$EXPORT_DIR/infer_fullft_all_usable.toml" \
  --input ./sample/sample1.mp4 \
  --output_folder "$EXPORT_DIR/fullft_all_usable_output" \
  --device cuda:0

echo "=== $(date '+%F %T') DONE ==="
find "$EXPORT_DIR" -maxdepth 2 -type f \( -name 'StereoPilot_fullft_all_usable.safetensors' -o -name 'infer_fullft_all_usable.toml' -o -path '*/fullft_all_usable_output/*.mp4' \) -printf '%p %s bytes\n'
