# Wan Stereo Full-Parameter Feasibility Mode

This directory is an isolated copy of the working LoRA-based `wan_stereo` pipeline. It is used to test whether Wan2.1-T2V-1.3B full-parameter StereoPilot-lite training can run on a single RTX 5090 32GB.

Key differences from the LoRA configs:

- no `[adapter]` block, so `train.py` uses `is_adapter = False`
- saved model artifact is `model.safetensors`, not `adapter_model.safetensors`
- `blocks_to_swap = 0`, because the current training script only supports block swapping for LoRA
- `transformer_dtype = 'bfloat16'`, avoiding trainable float8 transformer weights
- output directories are under `/path/to/workspace/runs/wan_stereo_fullft_*`

This mode is for feasibility testing first. If it OOMs, keep the logs as evidence that full-parameter training at this clip format does not fit the available single-GPU setup.
