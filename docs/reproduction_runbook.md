# Reproduction Runbook

This runbook uses placeholder paths. Replace them with your own machine paths and keep datasets, weights, caches, and outputs outside Git.

## 1. Workspace Layout

```text
/path/to/workspace/
  stereopilot-lite-replica/
    StereoPilot/
    diffusion-pipe-stereo/
  weights/
    Wan2.1-T2V-1.3B/
    StereoPilot.safetensors
  datasets/
    stereo4d_hf_raw/
  data/
    stereo4d_hf_50/
  runs/
  stereopilot_exports/
```

## 2. Training Environment

```bash
conda create -n StereoPilot python=3.12 -y
conda activate StereoPilot
python -m pip install -U pip wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo
pip install -r requirements.txt
```

Smoke-tested versions included PyTorch `2.11.0+cu128`, DeepSpeed `0.18.4`, PEFT `0.19.1`, and bitsandbytes `0.49.2` on an RTX 5090 32GB Linux machine.

## 3. Data Preparation Environment

```bash
conda create -n stereo4d_data python=3.11 -y
conda activate stereo4d_data
python -m pip install -U pip wheel huggingface_hub datasets ffmpeg-python tqdm
```

Make sure `ffmpeg` and `ffprobe` are on `PATH`.

## 4. Download Stereo4D-Derived Test Tar Files

```bash
mkdir -p /path/to/workspace/datasets/stereo4d_hf_raw
cd /path/to/workspace/datasets/stereo4d_hf_raw

git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/KevinMathew/stereo4d-lefteye-perspective left_repo
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/KevinMathew/stereo4d-righteye-perspective right_repo

cd left_repo && git lfs pull --include="test_mp4s.tar" --exclude=""
cd ../right_repo && git lfs pull --include="test_mp4s.tar" --exclude=""

mkdir -p ../extract_left ../extract_right
tar -xf ../left_repo/test_mp4s.tar -C ../extract_left
tar -xf ../right_repo/test_mp4s.tar -C ../extract_right
```

These tar files are not part of this repository.

## 5. Prepare Clips

```bash
conda activate stereo4d_data
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo

python tools/prepare_stereo4d_hf_smoke.py \
  --left-root /path/to/workspace/datasets/stereo4d_hf_raw/extract_left \
  --right-root /path/to/workspace/datasets/stereo4d_hf_raw/extract_right \
  --output-root /path/to/workspace/data/stereo4d_hf_50 \
  --num-clips 50 \
  --fps 16 \
  --frames 84 \
  --width 512 \
  --height 288 \
  --overwrite
```

The script writes:

```text
/path/to/workspace/data/stereo4d_hf_50/
  left/clip_000001.mp4
  right/clip_000001.mp4
  right/clip_000001.txt
  manifest.jsonl
```

## 6. QA Dataset

```bash
conda activate StereoPilot
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo

python tools/wan_stereo_dataset_qa.py \
  --right /path/to/workspace/data/stereo4d_hf_50/right \
  --left /path/to/workspace/data/stereo4d_hf_50/left \
  --min-frames 81 \
  --min-width 512 \
  --min-height 288 \
  --json /path/to/workspace/runs/stereo4d_hf_50_qa.json
```

Expected smoke output: matched pairs, no missing files, no errors.

## 7. Configure Training

Use one of the included configs, for example:

```text
diffusion-pipe-stereo/examples/wan_stereo_stereo4d_hf_50.toml
diffusion-pipe-stereo/examples/wan_stereo_dataset.stereo4d_hf_50.toml
```

Set absolute paths for:

```toml
ckpt_path = "/path/to/workspace/weights/Wan2.1-T2V-1.3B"
output_dir = "/path/to/workspace/runs/wan_stereo_1_3b_stereo4d_hf_50"
dataset = "examples/wan_stereo_dataset.stereo4d_hf_50.toml"
```

And in the dataset config:

```toml
path = "/path/to/workspace/data/stereo4d_hf_50/right"
control_path = "/path/to/workspace/data/stereo4d_hf_50/left"
size_buckets = [[512, 288, 81]]
domain_label = "parallel"
```

## 8. Cache Latents And Text

```bash
conda activate StereoPilot
cd /path/to/workspace/stereopilot-lite-replica/diffusion-pipe-stereo

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/wan_stereo_stereo4d_hf_50.toml \
  --cache_only --regenerate_cache
```

## 9. Train

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
deepspeed --num_gpus=1 train.py --deepspeed \
  --config examples/wan_stereo_stereo4d_hf_50.toml \
  --trust_cache
```

The training output should include a PEFT adapter directory such as:

```text
/path/to/workspace/runs/wan_stereo_1_3b_stereo4d_hf_50/YOUR_RUN/epoch1/adapter_model.safetensors
```

## 10. Export To StereoPilot-Compatible Weights

```bash
python tools/export_wan_stereo_lora.py \
  --base /path/to/workspace/weights/Wan2.1-T2V-1.3B \
  --adapter /path/to/workspace/runs/wan_stereo_1_3b_stereo4d_hf_50/YOUR_RUN/epoch1 \
  --output /path/to/workspace/stereopilot_exports/StereoPilot_merged_stereo4d_hf_50.safetensors \
  --reference /path/to/workspace/weights/StereoPilot.safetensors \
  --save-dtype reference \
  --overwrite
```

This produces a full `state_dict` that follows the key style expected by `StereoPilot/models/StereoPilot.py`.

## 11. Inference Smoke Test

Create an inference TOML outside Git, pointing to the merged safetensors. Then run:

```bash
cd /path/to/workspace/stereopilot-lite-replica/StereoPilot
conda activate StereoPilot

CUDA_VISIBLE_DEVICES=0 python sample.py \
  --config /path/to/workspace/stereopilot_exports/infer_merged_stereo4d_hf_50.toml \
  --input ./sample/sample1.mp4 \
  --output_folder /path/to/workspace/stereopilot_exports/stereo4d_hf_50_output \
  --device cuda:0
```

## 12. What A Successful Smoke Run Means

A successful run proves that:

- the paired stereo dataset format is accepted
- the VAE/text cache path works
- DeepSpeed can execute training
- the adapter can be merged into a full StereoPilot-style checkpoint
- the official inference loader can load the merged checkpoint

It does not prove official-quality results. That requires much larger, cleaner, properly licensed stereo data and longer experiments.
