# StereoPilot-lite Replica

Research prototype, not official StereoPilot training code.

This repository is a code-only monorepo snapshot for an engineering reproduction of a StereoPilot-style stereo-video training and inference workflow. It combines:

- `StereoPilot/`: a snapshot of the upstream StereoPilot inference code.
- `diffusion-pipe-stereo/`: a diffusion-pipe-based training replica with a `wan_stereo` pipeline.

The goal is to prove an end-to-end path from paired left/right stereo clips to a LoRA adapter, then to a merged `StereoPilot.safetensors`-style checkpoint that can be loaded by the StereoPilot inference code.

## What Is Not Included

This public repository intentionally does not include large or licensed artifacts:

- model weights or checkpoints
- Wan2.1 base weights
- official StereoPilot weights
- LoRA adapters or merged safetensors
- datasets, extracted videos, generated clips, or evaluation videos
- training caches, DeepSpeed checkpoints, run logs, or experiment outputs

Before pushing changes, run:

```bash
bash scripts/pre_push_safety_check.sh
```

## Repository Layout

```text
stereopilot-lite-replica/
  StereoPilot/                       # Inference-side StereoPilot snapshot, no ckpt/output files
  diffusion-pipe-stereo/             # Training-side diffusion-pipe replica
    models/wan_stereo.py             # Stereo left-latent -> right-latent training pipeline
    tools/prepare_stereo4d_hf_smoke.py
    tools/wan_stereo_dataset_qa.py
    tools/export_wan_stereo_lora.py
    examples/wan_stereo_*.toml
  docs/
    reproduction_runbook.md
    stereo4d_data_notes.md
  scripts/
    setup_remote_env.sh
    pre_push_safety_check.sh
  LICENSE_NOTES.md
```

## Proven Environment

The chain was smoke-tested on a single RTX 5090 32GB Linux machine with:

- Ubuntu/Linux with NVIDIA driver supporting CUDA 12.8+
- Conda environment for training/inference: Python 3.12
- PyTorch `2.11.0+cu128`
- DeepSpeed `0.18.4`
- PEFT `0.19.1`
- bitsandbytes `0.49.2`
- a separate data-preparation Conda environment: Python 3.11 with `huggingface_hub`, `ffmpeg`, and `ffprobe`

A smaller GPU can inspect and edit the code, but real Wan video training should be run on the larger GPU.

## Required External Assets

Place these outside Git:

```text
/path/to/workspace/
  StereoPilot/
    ckpt/
      Wan2.1-T2V-1.3B/
      StereoPilot.safetensors          # optional reference/inference checkpoint
  data/
  datasets/
  runs/
  stereopilot_exports/
```

Required for training:

- Wan2.1 T2V 1.3B base checkpoint at a path referenced by the training TOML.
- Paired stereo clips: right eye is the training target, left eye is the conditioning/control video.

Required for official-style inference:

- a merged full transformer safetensors file produced by `tools/export_wan_stereo_lora.py`, or the official StereoPilot checkpoint placed manually outside Git.

## Data Format

The `wan_stereo` dataset expects:

```text
/path/to/data/stereo_dataset/
  left/
    clip_000001.mp4
    clip_000002.mp4
  right/
    clip_000001.mp4
    clip_000001.txt
    clip_000002.mp4
    clip_000002.txt
```

Requirements:

- true stereo pairs, not ordinary 2D videos
- matching left/right filenames
- matching fps, frame count, and resolution
- no hard cuts inside a training clip
- no subtitles, watermarks, large black borders, or severe vertical disparity
- initial smoke format: `16fps`, at least `81` frames, `512x288`
- recommended clip export for smoke: `84` frames, because some mp4 duration estimates otherwise undercount frames

Use `domain_label = "parallel"` for parallel/VR180/rendered stereo and `domain_label = "converged"` for converged 3D movie style.

## Prepare Stereo4D-Derived Smoke Data

The repository includes helpers for HuggingFace pre-rectified Stereo4D-derived left/right perspective mp4s. Treat this data as research-only unless you have separate rights.

```bash
conda create -n stereo4d_data python=3.11 -y
conda activate stereo4d_data
python -m pip install -U pip wheel huggingface_hub datasets ffmpeg-python tqdm
```

Download and extract the test tar files outside Git:

```bash
mkdir -p /path/to/datasets/stereo4d_hf_raw
cd /path/to/datasets/stereo4d_hf_raw

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/KevinMathew/stereo4d-lefteye-perspective left_repo
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/KevinMathew/stereo4d-righteye-perspective right_repo

cd left_repo && git lfs pull --include="test_mp4s.tar" --exclude=""
cd ../right_repo && git lfs pull --include="test_mp4s.tar" --exclude=""

mkdir -p ../extract_left ../extract_right
tar -xf ../left_repo/test_mp4s.tar -C ../extract_left
tar -xf ../right_repo/test_mp4s.tar -C ../extract_right
```

Prepare 10, 50, or all usable clips:

```bash
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo
conda activate stereo4d_data

python tools/prepare_stereo4d_hf_smoke.py \
  --left-root /path/to/datasets/stereo4d_hf_raw/extract_left \
  --right-root /path/to/datasets/stereo4d_hf_raw/extract_right \
  --output-root /path/to/data/stereo4d_hf_50 \
  --num-clips 50 \
  --fps 16 \
  --frames 84 \
  --width 512 \
  --height 288 \
  --overwrite
```

For all usable clips, use a high value such as `--num-clips 999999`; the script will stop when eligible pairs are exhausted.

## QA The Dataset

```bash
conda activate StereoPilot
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo

python tools/wan_stereo_dataset_qa.py \
  --right /path/to/data/stereo4d_hf_50/right \
  --left /path/to/data/stereo4d_hf_50/left \
  --min-frames 81 \
  --min-width 512 \
  --min-height 288 \
  --json /path/to/runs/stereo4d_hf_50_qa.json
```

The expected smoke result is matched pairs with zero errors. Warnings should be investigated before full training.

## Train Wan Stereo LoRA

Edit the selected TOML first:

```text
diffusion-pipe-stereo/examples/wan_stereo_stereo4d_hf_50.toml
diffusion-pipe-stereo/examples/wan_stereo_dataset.stereo4d_hf_50.toml
```

Set absolute paths for:

- `ckpt_path`
- `output_dir`
- dataset `path`
- dataset `control_path`

Cache latents/text first:

```bash
conda activate StereoPilot
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/wan_stereo_stereo4d_hf_50.toml \
  --cache_only --regenerate_cache
```

Run training:

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/wan_stereo_stereo4d_hf_50.toml \
  --trust_cache
```

The smoke acceptance criterion is that DeepSpeed reaches at least one training step and writes an adapter checkpoint under `output_dir`.

## Merge LoRA To StereoPilot Format

The training checkpoint is a PEFT LoRA adapter. The StereoPilot inference code expects a full transformer `state_dict`. Use the exporter to merge:

```bash
python tools/export_wan_stereo_lora.py \
  --base /path/to/weights/Wan2.1-T2V-1.3B \
  --adapter /path/to/runs/wan_stereo_1_3b_stereo4d_hf_50/YOUR_RUN/epoch1 \
  --output /path/to/stereopilot_exports/StereoPilot_merged_stereo4d_hf_50.safetensors \
  --reference /path/to/weights/StereoPilot.safetensors \
  --save-dtype reference \
  --overwrite
```

The exporter removes LoRA-specific keys and writes `parall_embedding` and `converge_embedding` keys compatible with the StereoPilot loader.

## Inference Smoke Test

Create a separate inference TOML outside the tracked default config, pointing to the merged safetensors:

```toml
[model]
path = "/path/to/stereopilot_exports/StereoPilot_merged_stereo4d_hf_50.safetensors"
```

Run the inference snapshot:

```bash
cd /path/to/workspace/stereopilot-lite-replica/StereoPilot
conda activate StereoPilot

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --config /path/to/stereopilot_exports/infer_merged_stereo4d_hf_50.toml \
  --input ./sample/sample1.mp4 \
  --output_folder /path/to/stereopilot_exports/stereo4d_hf_50_output \
  --device cuda:0
```

## Proven Smoke Chain

The end-to-end chain has been validated with real Stereo4D-derived left/right clips:

- paired clip preparation
- dataset QA
- latent/text cache generation
- DeepSpeed training smoke
- LoRA adapter export into full StereoPilot-style safetensors
- official StereoPilot inference loader smoke test

This validates code compatibility and workflow wiring. It does not claim the learned model quality matches the official StereoPilot release.

## License And Data Notice

See `LICENSE_NOTES.md`. In short:

- Upstream StereoPilot code is MIT.
- Upstream diffusion-pipe is GPL-3.0; modifications in `diffusion-pipe-stereo/` should be treated as GPL-derived.
- Wan model weights have their own license and are not included.
- Stereo4D-derived HuggingFace data is not included and should be treated as non-commercial research data unless you have separate rights.
