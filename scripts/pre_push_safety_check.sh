#!/usr/bin/env bash
set -euo pipefail

fail=0

echo "[check] forbidden tracked paths and file types"
forbidden_regex='(^|/)(ckpt|data|datasets|runs|stereopilot_exports|output|stereo_output|cache|\.cache)/|\.(safetensors|pt|pth|ckpt|tar|zip|mp4|mov|mkv|avi|webm)$'
if git ls-files | grep -E "$forbidden_regex"; then
  echo "ERROR: forbidden tracked files or directories found" >&2
  fail=1
fi

echo "[check] tracked files larger than 95 MiB"
while IFS= read -r file; do
  [ -f "$file" ] || continue
  size=$(stat -c '%s' "$file" 2>/dev/null || stat -f '%z' "$file")
  if [ "$size" -gt 99614720 ]; then
    echo "ERROR: tracked file exceeds 95 MiB: $file ($size bytes)" >&2
    fail=1
  fi
done < <(git ls-files)

echo "[check] public docs/scripts for private machine metadata"
sensitive_regex='100\.111\.220\.101|/home/[A-Za-z0-9._-]+|C:\\Users|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}'
if git grep -n -E "$sensitive_regex" -- README.md LICENSE_NOTES.md docs scripts 2>/dev/null; then
  echo "ERROR: sensitive-looking machine metadata or token pattern found in public-facing files" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "pre-push safety check failed" >&2
  exit 1
fi

echo "pre-push safety check passed"
