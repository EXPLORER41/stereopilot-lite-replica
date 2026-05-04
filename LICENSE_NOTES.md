# License And Attribution Notes

This repository is a code-only research prototype. It is not an official StereoPilot release and it does not include model weights, datasets, LoRA adapters, merged checkpoints, generated videos, or training caches.

## Upstream Projects

- `StereoPilot/` is a snapshot of the upstream StereoPilot inference project. The upstream repository identifies its code license as MIT. Keep the original upstream notices when redistributing that code.
- `diffusion-pipe-stereo/` is derived from `tdrussell/diffusion-pipe`. diffusion-pipe is GPL-3.0, so modifications in this training-side tree should be treated as GPL-derived.
- Wan2.1 model weights are not included. Download and use them only under the applicable Wan model license and terms.
- Official StereoPilot weights are not included. Download and use them only under the applicable upstream terms.

## Data Notice

No datasets are included in this repository.

The documented Stereo4D-derived HuggingFace pre-rectified mp4 workflow is for research prototyping. Treat those assets as non-commercial research data unless you have separate permission that covers your use case. Do not commit downloaded tar files, extracted videos, generated left/right clips, captions produced from licensed videos, caches, training runs, or evaluation outputs.

For commercial training, use data with explicit rights for AI/ML training or generative-model fine-tuning.

## Public Safety Rules

Before every push, run:

```bash
bash scripts/pre_push_safety_check.sh
```

Do not commit:

- `ckpt/`, `data/`, `datasets/`, `runs/`, `stereopilot_exports/`, `output/`, `stereo_output/`, `cache/`, `.cache/`
- `*.safetensors`, `*.pt`, `*.pth`, `*.ckpt`, `*.tar`, `*.zip`, `*.mp4`, `*.mov`, `*.mkv`, `*.avi`, `*.webm`
- real machine usernames, private IPs, passwords, tokens, or local absolute paths
