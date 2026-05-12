#!/usr/bin/env bash
set -euo pipefail

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate stereo4d_data

ROOT=/path/to/workspace/datasets/stereo4d_hf_raw
mkdir -p "$ROOT/download_left" "$ROOT/download_right" "$ROOT/extract_left" "$ROOT/extract_right" "$ROOT/logs"

python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path("/path/to/workspace/datasets/stereo4d_hf_raw")
items = [
    ("KevinMathew/stereo4d-lefteye-perspective", root / "download_left"),
    ("KevinMathew/stereo4d-righteye-perspective", root / "download_right"),
]
for repo_id, local_dir in items:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id}", flush=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="test_mp4s.tar",
        local_dir=local_dir,
    )
    print(f"[downloaded] {path}", flush=True)
PY

echo "[extract] left"
tar -xf "$ROOT/download_left/test_mp4s.tar" -C "$ROOT/extract_left"

echo "[extract] right"
tar -xf "$ROOT/download_right/test_mp4s.tar" -C "$ROOT/extract_right"

echo "[summary]"
find "$ROOT/extract_left" -type f -name '*.mp4' | wc -l
find "$ROOT/extract_right" -type f -name '*.mp4' | wc -l
du -sh "$ROOT"

echo "[done]"
