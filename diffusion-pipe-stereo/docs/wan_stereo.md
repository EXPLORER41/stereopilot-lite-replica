# Wan StereoPilot-lite

This repository now includes a minimal StereoPilot-style training prototype named
`wan_stereo`. It is not the official StereoPilot training code. The goal is to
test whether a Wan T2V backbone can learn a single-forward left-eye to right-eye
mapping from paired stereo clips.

## What It Trains

- Base model: Wan2.1 T2V, with Wan2.1-T2V-1.3B recommended for first runs.
- Input: left-eye video encoded by the Wan VAE.
- Target: matching right-eye video encoded by the same VAE.
- Loss: latent-space MSE through diffusion-pipe's normal loss wrapper.
- Domain switch: `domain_label = 0` for parallel/Stereo4D style and
  `domain_label = 1` for converged/3DMovie style.

Unlike normal Wan training, `wan_stereo` does not sample a diffusion timestep
and does not train a noise-prediction target. It feeds the left-eye latent to the
transformer and supervises the raw output against the right-eye latent. This
matches the public StereoPilot inference shape more closely than a normal Wan
LoRA run, but it is still only a practical prototype.

## Dataset Layout

Use diffusion-pipe's existing `control_path` pairing:

```text
/data/stereo/right/clip_000001.mp4
/data/stereo/right/clip_000001.txt
/data/stereo/left/clip_000001.mp4
```

The right-eye directory is `path`; the left-eye directory is `control_path`.
File stems must match exactly. Captions are read from the right-eye directory.

Recommended clip preparation:

- true stereo pairs only; no anaglyph and no monocular pseudo-pairs
- 16 fps
- 81 frames
- synchronized left/right views
- no subtitles, watermarks, credits, shot changes, or black borders
- start with `512x288x81`; scale to `640x368x81` or `832x480x81` after sanity checks

## Run

Edit the paths in:

- `examples/wan_stereo_1_3b.toml`
- `examples/wan_stereo_dataset.toml`

Check the paired folders before caching:

```bash
python tools/wan_stereo_dataset_qa.py --right /data/stereo/right --left /data/stereo/left --min-frames 81 --min-width 512 --min-height 288 --json /tmp/wan_stereo_qa.json
```

Then cache and train:

```bash
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_stereo_1_3b.toml --cache_only
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config examples/wan_stereo_1_3b.toml
```

On a 32GB RTX 5090, begin with LoRA rank 32, `blocks_to_swap = 18`, and
`size_buckets = [[512, 288, 81]]`. If the first run is stable, increase spatial
resolution before increasing LoRA rank.

An 8GB development GPU is enough for editing and syntax checks, but it should
not be expected to load Wan2.1-T2V-1.3B plus the Wan text encoder for caching or
training. Use the 5090 machine for `--cache_only` and the actual training run.

## Sanity Checks

Before any large run:

- overfit 50-100 paired clips and confirm the decoded right-eye output changes
  from the left-eye input instead of merely copying it
- run one directory with `domain_label = 'converged'` and one with
  `domain_label = 'parallel'` to confirm the domain embeddings are trainable
- regenerate caches after changing `domain_label`, `control_path`, frame buckets,
  or source videos
