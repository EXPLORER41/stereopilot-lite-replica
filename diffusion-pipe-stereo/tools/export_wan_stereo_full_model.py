#!/usr/bin/env python3
"""Validate/export a full-parameter wan_stereo checkpoint for StereoPilot inference.

The full-parameter training path saves `model.safetensors`. This tool checks that
it is not a LoRA adapter, optionally aligns dtype/key style to a reference
StereoPilot checkpoint, and writes a StereoPilot-compatible safetensors file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import safetensors.torch


def strip_prefix(key: str) -> str:
    for prefix in ("diffusion_model.", "model.diffusion_model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to fullft model.safetensors or containing directory")
    parser.add_argument("--output", required=True, help="Output StereoPilot-compatible safetensors path")
    parser.add_argument("--reference", default=None, help="Optional official StereoPilot.safetensors for key/shape/dtype checks")
    parser.add_argument("--save-dtype", default="input", choices=["input", "reference", "bfloat16", "float16", "float32"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    if in_path.is_dir():
        in_path = in_path / "model.safetensors"
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sd = safetensors.torch.load_file(in_path)
    lora_keys = [k for k in sd if "lora" in k.lower()]
    if lora_keys:
        raise RuntimeError(f"Input looks like a LoRA adapter; found {len(lora_keys)} LoRA keys, e.g. {lora_keys[:5]}")

    sd = {strip_prefix(k): v.detach().cpu() for k, v in sd.items()}
    ref = None
    if args.reference:
        ref = safetensors.torch.load_file(args.reference)
        missing = sorted(set(ref) - set(sd))
        extra = sorted(set(sd) - set(ref))
        shape_mismatch = sorted(k for k in set(ref) & set(sd) if tuple(ref[k].shape) != tuple(sd[k].shape))
        print(f"reference_keys={len(ref)} input_keys={len(sd)} missing={len(missing)} extra={len(extra)} shape_mismatch={len(shape_mismatch)}")
        if missing:
            print("missing examples:", missing[:20])
        if extra:
            print("extra examples:", extra[:20])
        if shape_mismatch:
            print("shape mismatch examples:", [(k, tuple(sd[k].shape), tuple(ref[k].shape)) for k in shape_mismatch[:10]])
        if missing or extra or shape_mismatch:
            raise RuntimeError("Full model keys/shapes do not match the reference StereoPilot checkpoint")

    if args.save_dtype == "reference":
        if ref is None:
            raise ValueError("--save-dtype reference requires --reference")
        sd = {k: v.to(ref[k].dtype) for k, v in sd.items()}
    elif args.save_dtype != "input":
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.save_dtype]
        sd = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in sd.items()}

    safetensors.torch.save_file(sd, out_path, metadata={"format": "pt"})
    print(f"wrote {out_path}")
    print(f"keys={len(sd)} lora_keys=0")


if __name__ == "__main__":
    main()
