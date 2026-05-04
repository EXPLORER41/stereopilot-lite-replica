import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
PREFIX = "diffusion_model."
DOMAIN_KEYS = ("parall_embedding", "converge_embedding")


def resolve_adapter_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Adapter safetensors not found: {path}")
    return path


def resolve_adapter_config(adapter_path, explicit_config):
    if explicit_config:
        config_path = Path(explicit_config)
    else:
        config_path = adapter_path.parent / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Adapter config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f), config_path


def resolve_base_path(path):
    path = Path(path)
    if path.is_dir():
        path = path / "diffusion_pytorch_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Base transformer safetensors not found: {path}")
    return path


def strip_prefix(key):
    return key[len(PREFIX):] if key.startswith(PREFIX) else key


def lora_scale(config, target_key, rank):
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    r = int(rank_pattern.get(target_key, config.get("r", rank)))
    alpha = float(alpha_pattern.get(target_key, config.get("lora_alpha", r)))
    if config.get("use_rslora", False):
        return alpha / (r ** 0.5)
    return alpha / r


def merge_lora(base_state, adapter_state, adapter_config):
    merged = {key: value.detach().cpu().clone() for key, value in base_state.items()}
    a_keys = sorted(key for key in adapter_state if key.endswith(LORA_A_SUFFIX))
    if not a_keys:
        raise ValueError("No LoRA A tensors found in adapter.")

    fan_in_fan_out = bool(adapter_config.get("fan_in_fan_out", False))
    merged_count = 0
    for a_key in a_keys:
        b_key = a_key[:-len(LORA_A_SUFFIX)] + LORA_B_SUFFIX
        if b_key not in adapter_state:
            raise KeyError(f"Missing LoRA B tensor for {a_key}: {b_key}")

        target_key = strip_prefix(a_key[:-len(LORA_A_SUFFIX)])
        weight_key = f"{target_key}.weight"
        if weight_key not in merged:
            raise KeyError(f"Base weight for LoRA tensor not found: {weight_key}")

        lora_a = adapter_state[a_key].detach().cpu()
        lora_b = adapter_state[b_key].detach().cpu()
        base_weight = merged[weight_key]
        scale = lora_scale(adapter_config, target_key, lora_a.shape[0])
        delta = torch.matmul(lora_b.float(), lora_a.float()) * scale
        if fan_in_fan_out:
            delta = delta.T
        if delta.shape != base_weight.shape:
            raise ValueError(
                f"Delta shape mismatch for {weight_key}: "
                f"delta={tuple(delta.shape)} base={tuple(base_weight.shape)}"
            )
        merged[weight_key] = (base_weight.float() + delta).to(base_weight.dtype).contiguous()
        merged_count += 1

    for domain_key in DOMAIN_KEYS:
        adapter_key = f"{PREFIX}{domain_key}"
        if adapter_key not in adapter_state:
            raise KeyError(f"Missing trained domain embedding in adapter: {adapter_key}")
        merged[domain_key] = adapter_state[adapter_key].detach().cpu().contiguous()

    unexpected = [
        key for key in adapter_state
        if not key.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX))
        and strip_prefix(key) not in DOMAIN_KEYS
    ]
    if unexpected:
        raise ValueError(f"Unexpected non-LoRA adapter tensors: {unexpected[:20]}")

    return merged, merged_count


def verify_against_reference(merged, reference_path):
    reference = load_file(str(reference_path), device="cpu")
    merged_keys = set(merged)
    reference_keys = set(reference)
    missing = sorted(reference_keys - merged_keys)
    extra = sorted(merged_keys - reference_keys)
    shape_mismatch = []
    for key in sorted(merged_keys & reference_keys):
        if tuple(merged[key].shape) != tuple(reference[key].shape):
            shape_mismatch.append((key, tuple(merged[key].shape), tuple(reference[key].shape)))
    return missing, extra, shape_mismatch


def cast_to_reference_dtypes(state, reference_path):
    reference = load_file(str(reference_path), device="cpu")
    result = {}
    for key, tensor in state.items():
        if key not in reference:
            result[key] = tensor.contiguous()
            continue
        result[key] = tensor.to(reference[key].dtype).contiguous()
    return result


def cast_to_single_dtype(state, dtype_name):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_name]
    return {key: tensor.to(dtype).contiguous() for key, tensor in state.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Merge a diffusion-pipe wan_stereo LoRA adapter into a full StereoPilot-compatible transformer safetensors."
    )
    parser.add_argument("--base", required=True, help="Wan2.1-T2V-1.3B folder or diffusion_pytorch_model.safetensors")
    parser.add_argument("--adapter", required=True, help="Adapter folder or adapter_model.safetensors")
    parser.add_argument("--output", required=True, help="Output merged safetensors path")
    parser.add_argument("--adapter-config", default=None, help="Optional explicit adapter_config.json")
    parser.add_argument("--reference", default=None, help="Optional official StereoPilot.safetensors for key/shape verification")
    parser.add_argument(
        "--save-dtype",
        choices=("source", "float32", "float16", "bfloat16", "reference"),
        default="source",
        help="Output tensor dtype policy. Use 'reference' with --reference to mimic official StereoPilot dtype layout.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    args = parser.parse_args()

    base_path = resolve_base_path(args.base)
    adapter_path = resolve_adapter_path(args.adapter)
    adapter_config, adapter_config_path = resolve_adapter_config(adapter_path, args.adapter_config)
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading base: {base_path}")
    base_state = load_file(str(base_path), device="cpu")
    print(f"Loading adapter: {adapter_path}")
    adapter_state = load_file(str(adapter_path), device="cpu")
    print(f"Loading adapter config: {adapter_config_path}")

    merged, merged_count = merge_lora(base_state, adapter_state, adapter_config)
    lora_keys = [key for key in merged if "lora_" in key.lower()]
    if lora_keys:
        raise RuntimeError(f"Merged state still contains LoRA keys: {lora_keys[:20]}")

    if args.save_dtype == "reference":
        if not args.reference:
            raise ValueError("--save-dtype reference requires --reference")
        print(f"Casting output tensors to reference dtypes: {args.reference}")
        merged = cast_to_reference_dtypes(merged, args.reference)
    elif args.save_dtype != "source":
        print(f"Casting all output tensors to {args.save_dtype}")
        merged = cast_to_single_dtype(merged, args.save_dtype)

    metadata = {
        "format": "pt",
        "merged_from_base": str(base_path),
        "merged_from_adapter": str(adapter_path),
        "adapter_config": str(adapter_config_path),
        "merge_type": "wan_stereo_lora_to_stereopilot_full",
    }

    print(f"Saving merged state: {output_path}")
    save_file(merged, str(output_path), metadata=metadata)
    print(f"Merged LoRA modules: {merged_count}")
    print(f"Output tensors: {len(merged)}")
    print(f"Domain embeddings: {[key for key in DOMAIN_KEYS if key in merged]}")

    if args.reference:
        reference_path = Path(args.reference)
        missing, extra, shape_mismatch = verify_against_reference(merged, reference_path)
        print(f"Reference: {reference_path}")
        print(f"Missing keys vs reference: {len(missing)}")
        print(f"Extra keys vs reference: {len(extra)}")
        print(f"Shape mismatches vs reference: {len(shape_mismatch)}")
        if missing or extra or shape_mismatch:
            if missing:
                print("Missing:", missing[:20])
            if extra:
                print("Extra:", extra[:20])
            if shape_mismatch:
                print("Shape mismatch:", shape_mismatch[:20])
            raise SystemExit(1)


if __name__ == "__main__":
    main()
