#!/usr/bin/env bash
set -euo pipefail

fail=0

echo "[check] forbidden tracked paths and file types"
while IFS= read -r file; do
  case "$file" in
    StereoPilot/sample/*.mp4|StereoPilot/asset/*.png|StereoPilot/asset/*.jpg|StereoPilot/asset/*.jpeg)
      ;;
    *.safetensors|*.pt|*.pth|*.ckpt|*.tar|*.zip|*.7z|*.rar|*.gz|*.xz|*.mov|*.mkv|*.avi|*.webm|*.mp4)
      echo "ERROR: forbidden tracked artifact: $file" >&2
      fail=1
      ;;
  esac

  case "$file" in
    data/*|datasets/*|runs/*|stereopilot_exports/*|models/*|StereoPilot/ckpt/*|StereoPilot/output/*|StereoPilot/stereo_output/*)
      case "$file" in
        */README.md|*/.gitkeep)
          ;;
        *)
          echo "ERROR: runtime/artifact directory may only track README.md or .gitkeep: $file" >&2
          fail=1
          ;;
      esac
      ;;
  esac

  case "$file" in
    */global_step*/*|*/checkpoint*/*|*/epoch*/*)
      echo "ERROR: training checkpoint path is tracked: $file" >&2
      fail=1
      ;;
  esac
done < <(git ls-files)

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
sensitive_regex='100\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|/home/[A-Za-z0-9._-]+|[A-Za-z]:\\Users|ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}'
if git grep -n -E "$sensitive_regex" -- README.md LICENSE_NOTES.md docs scripts 2>/dev/null; then
  echo "ERROR: sensitive-looking machine metadata or token pattern found in public-facing files" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "pre-push safety check failed" >&2
  exit 1
fi

echo "pre-push safety check passed"
