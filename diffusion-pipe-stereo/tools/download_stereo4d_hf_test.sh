#!/usr/bin/env bash
set -euo pipefail

# Optional overrides:
#   CONDA_SH=/path/to/miniconda/etc/profile.d/conda.sh \
#   STEREO4D_HF_RAW_ROOT=/path/to/datasets/stereo4d_hf_raw \
#   bash tools/download_stereo4d_hf_test.sh

CONDA_SH="${CONDA_SH:-/opt/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate stereo4d_data
fi

ROOT="${STEREO4D_HF_RAW_ROOT:-/path/to/datasets/stereo4d_hf_raw}"
mkdir -p "$ROOT/download_left" "$ROOT/download_right" "$ROOT/extract_left" "$ROOT/extract_right" "$ROOT/logs"

ROOT="$ROOT" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(os.environ["ROOT"])
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
