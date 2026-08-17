#!/usr/bin/env python3
"""Compare native comfy-kitchen W4A8 layouts on real FLUX.2 Klein weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from comfy_kitchen.tensor import QuantizedTensor
from safetensors import safe_open


DEFAULT_LAYERS = (
    "double_blocks.0.img_attn.qkv",
    "double_blocks.4.img_mlp.0",
    "single_blocks.0.linear1",
    "single_blocks.23.linear2",
)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


@torch.no_grad()
def main():
    args = _args()
    source_path = Path(args.bf16).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    variants = (
        ("symmetric_codebook_fp8", True, True, torch.float8_e4m3fn),
        ("symmetric_uniform_fp8", True, False, torch.float8_e4m3fn),
        ("asymmetric_uniform_fp8", False, False, torch.float8_e4m3fn),
        ("symmetric_codebook_fp32", True, True, torch.float32),
    )
    report = {"schema_version": 1, "source": str(source_path), "layers": []}
    with safe_open(str(source_path), framework="pt", device="cpu") as source:
        for layer in args.layers:
            key = f"{layer}.weight"
            weight = source.get_tensor(key).to(device=device).contiguous()
            energy = float(weight.float().square().sum())
            row = {"name": layer, "shape": list(weight.shape), "variants": []}
            for group_size in args.group_sizes:
                for name, symmetric, codebook, scale_dtype in variants:
                    quantized = QuantizedTensor.from_float(
                        weight,
                        "AsymW4A8Int8Layout",
                        group_size=group_size,
                        convrot_groupsize=256,
                        symmetric=symmetric,
                        scale_dtype=scale_dtype,
                        codebook=codebook,
                    )
                    restored = quantized.dequantize()
                    error_sq = float((restored.float() - weight.float()).square().sum())
                    state = quantized.state_dict("weight")
                    row["variants"].append(
                        {
                            "name": name,
                            "group_size": group_size,
                            "relative_l2_error_pct": 100.0 * math.sqrt(error_sq / max(energy, 1e-30)),
                            "serialized_bytes": sum(_nbytes(value) for value in state.values()),
                            "state": {
                                state_key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                for state_key, value in state.items()
                            },
                        }
                    )
                    print(
                        f"{layer} gs{group_size:<2} {name:<27} "
                        f"err={row['variants'][-1]['relative_l2_error_pct']:.4f}%",
                        flush=True,
                    )
                    del quantized, restored, state
            report["layers"].append(row)
            del weight
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
