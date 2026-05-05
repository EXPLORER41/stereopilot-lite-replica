# External Checkpoints

This directory mirrors the runtime `StereoPilot/ckpt/` location on the training machine.

Do not commit real weights here. Place external checkpoint files manually:

| Path | Required | Purpose | Notes |
| --- | --- | --- | --- |
| `StereoPilot/ckpt/StereoPilot.safetensors` | Optional for official inference/reference export | Official StereoPilot transformer checkpoint | Large external artifact, not included. |
| `StereoPilot/ckpt/Wan2.1-T2V-1.3B/` | Required for training/export | Wan2.1 T2V 1.3B base checkpoint directory | Large external artifact, not included. |

The training configs in `diffusion-pipe-stereo/examples/` should point `ckpt_path` to the Wan2.1 directory above or to an equivalent absolute path.
