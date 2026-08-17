#!/usr/bin/env python3
"""Profile FLUX.2 Klein quantization on real ComfyUI denoising activations.

Run this file with *ComfyUI's* Python environment.  It loads BF16, W8 and W4
checkpoints together, feeds identical BF16 activations through corresponding
quantized projections/blocks, and records isolated response error.  The
compiler itself remains in Musubi's environment; this worker intentionally
uses ComfyUI's runtime so measurements exercise the kernels used for images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_PROMPTS = [
    {
        "id": "photoreal_text",
        "seed": 424242,
        "text": (
            "A cinematic close-up photograph of a weathered astronaut standing in a "
            "rain-soaked neon market at night, intricate suit materials, realistic skin, "
            "reflections, readable sign saying ORBITAL MARKET, volumetric light, shallow depth of field"
        ),
    },
    {
        "id": "product_typography",
        "seed": 195487,
        "text": (
            "Premium studio product photograph of a cobalt blue perfume bottle on black stone, "
            "gold foil label with perfectly readable words NORTH STAR No. 7, droplets, rim light, "
            "accurate glass reflections, luxury advertising layout"
        ),
    },
    {
        "id": "spatial_scene",
        "seed": 731995,
        "text": (
            "Wide editorial photograph of a red fox sitting left of a small green tent beside an alpine lake, "
            "two yellow kayaks on the right, snow mountains reflected in water, sunrise mist, detailed natural textures"
        ),
    },
    {
        "id": "graphic_detail",
        "seed": 880031,
        "text": (
            "Intricate dark fantasy book cover, silver mechanical owl centered above an ancient clock, "
            "symmetrical filigree, title THE LAST HOUR in crisp serif lettering, teal and amber ink, print-ready detail"
        ),
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, help="ComfyUI application directory containing nodes.py")
    parser.add_argument("--bf16", required=True, help="BF16 diffusion model filename")
    parser.add_argument("--int8", required=True, help="All-W8 ConvRot diffusion model filename")
    parser.add_argument("--int4", required=True, help="All-W4 ConvRot diffusion model filename")
    parser.add_argument("--clip", default="qwen_3_8b.safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", help="Optional JSON list replacing the built-in calibration prompts")
    parser.add_argument("--limit-prompts", type=int, help="Use only the first N prompts (smoke tests)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="flux2")
    parser.add_argument("--skip-blocks", action="store_true", help="Profile projections only")
    return parser.parse_args()


def _load_prompts(path: str | None):
    if path is None:
        return DEFAULT_PROMPTS
    with open(path, encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt file must contain a non-empty JSON list")
    for index, item in enumerate(prompts):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"Prompt {index} requires a string 'text'")
        item.setdefault("id", f"prompt_{index + 1}")
        item.setdefault("seed", 424242 + index)
    return prompts


def _eligible_projection(name: str) -> bool:
    if name.startswith("double_blocks."):
        return name.endswith(
            (
                ".img_attn.qkv",
                ".img_attn.proj",
                ".img_mlp.0",
                ".img_mlp.2",
                ".txt_attn.qkv",
                ".txt_attn.proj",
                ".txt_mlp.0",
                ".txt_mlp.2",
            )
        )
    return name.startswith("single_blocks.") and name.endswith((".linear1", ".linear2"))


def _eligible_block(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 2 and parts[0] in {"double_blocks", "single_blocks"} and parts[1].isdigit()


def _flatten_tensors(value: Any) -> Iterable[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _flatten_tensors(item)


class Metrics:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.data: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    def update(self, name: str, variant: str, reference: Any, candidate: Any) -> None:
        torch = self.torch
        refs = list(_flatten_tensors(reference))
        cands = list(_flatten_tensors(candidate))
        if len(refs) != len(cands):
            raise RuntimeError(f"Output structure mismatch for {name}: {len(refs)} != {len(cands)}")
        metric = self.data[name].get(variant)
        if metric is None:
            device = refs[0].device
            zero = torch.zeros((), device=device, dtype=torch.float64)
            metric = {
                "calls": 0,
                "elements": 0,
                "reference_sq": zero.clone(),
                "candidate_sq": zero.clone(),
                "error_sq": zero.clone(),
                "dot": zero.clone(),
                "max_abs_error": zero.clone(),
            }
            self.data[name][variant] = metric
        metric["calls"] += 1
        for ref, cand in zip(refs, cands):
            ref32 = ref.float()
            cand32 = cand.float()
            error = cand32 - ref32
            metric["elements"] += ref.numel()
            metric["reference_sq"] += (ref32 * ref32).sum(dtype=torch.float64)
            metric["candidate_sq"] += (cand32 * cand32).sum(dtype=torch.float64)
            metric["error_sq"] += (error * error).sum(dtype=torch.float64)
            metric["dot"] += (ref32 * cand32).sum(dtype=torch.float64)
            metric["max_abs_error"] = torch.maximum(metric["max_abs_error"], error.abs().max().double())

    def finish(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        output = {}
        for name, variants in self.data.items():
            output[name] = {}
            for variant, metric in variants.items():
                ref_sq = float(metric["reference_sq"].item())
                cand_sq = float(metric["candidate_sq"].item())
                error_sq = float(metric["error_sq"].item())
                output[name][variant] = {
                    "calls": metric["calls"],
                    "elements": metric["elements"],
                    "relative_rmse_pct": 100.0 * math.sqrt(error_sq / max(ref_sq, 1e-30)),
                    "cosine": float(metric["dot"].item()) / math.sqrt(max(ref_sq * cand_sq, 1e-30)),
                    "reference_rms": math.sqrt(ref_sq / max(metric["elements"], 1)),
                    "error_rms": math.sqrt(error_sq / max(metric["elements"], 1)),
                    "max_abs_error": float(metric["max_abs_error"].item()),
                }
        return output


def _clone_block_call(args, kwargs):
    """Clone residual streams because FLUX blocks update them in place."""
    cloned_args = list(args)
    if cloned_args:
        cloned_args[0] = cloned_args[0].clone()
    cloned_kwargs = dict(kwargs)
    for key in ("img", "txt"):
        if key in cloned_kwargs:
            cloned_kwargs[key] = cloned_kwargs[key].clone()
    return tuple(cloned_args), cloned_kwargs


def main() -> int:
    args = _parse_args()
    prompts = _load_prompts(args.prompts)
    if args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise ValueError("--limit-prompts must be at least 1")
        prompts = prompts[: args.limit_prompts]
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    if not (comfy_root / "nodes.py").is_file():
        raise FileNotFoundError(f"Not a ComfyUI application directory: {comfy_root}")

    # ComfyUI has its own argument parser at import time.  Keep only runtime
    # flags it owns after this worker has parsed its arguments.
    sys.argv = [sys.argv[0], "--highvram", "--disable-auto-launch", "--disable-metadata"]
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import torch
    import comfy.model_management
    import comfy.utils
    import comfyui_version
    import nodes

    comfy.utils.PROGRESS_BAR_ENABLED = False

    if not torch.cuda.is_available():
        raise RuntimeError("Sensitivity profiling requires CUDA")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    loader = nodes.UNETLoader()
    bf16 = loader.load_unet(args.bf16, "default")[0]
    int8 = loader.load_unet(args.int8, "default")[0]
    int4 = loader.load_unet(args.int4, "default")[0]
    clip = nodes.CLIPLoader().load_clip(args.clip, "flux2", "default")[0]

    bf_modules = dict(bf16.model.diffusion_model.named_modules())
    int8_modules = dict(int8.model.diffusion_model.named_modules())
    int4_modules = dict(int4.model.diffusion_model.named_modules())
    projection_names = sorted(name for name in bf_modules if _eligible_projection(name))
    block_names = sorted(name for name in bf_modules if _eligible_block(name))
    if len(projection_names) != 112 or len(block_names) != 32:
        raise RuntimeError(
            f"Expected 112 projections and 32 blocks, found {len(projection_names)} and {len(block_names)}"
        )

    projection_metrics = Metrics(torch)
    block_metrics = Metrics(torch)
    handles = []

    def make_projection_hook(name):
        def hook(_module, call_args, reference):
            activation = call_args[0]
            with torch.inference_mode():
                candidate = int8_modules[name](activation)
                projection_metrics.update(name, "int8", reference, candidate)
                del candidate
                candidate = int4_modules[name](activation)
                projection_metrics.update(name, "int4", reference, candidate)
                del candidate

        return hook

    for name in projection_names:
        handles.append(bf_modules[name].register_forward_hook(make_projection_hook(name)))

    pending_calls = {}
    if not args.skip_blocks:
        def make_pre_hook(name):
            def pre_hook(_module, call_args, call_kwargs):
                pending_calls[name] = _clone_block_call(call_args, call_kwargs)
            return pre_hook

        def make_block_hook(name):
            def block_hook(_module, _call_args, _call_kwargs, reference):
                base_args, base_kwargs = pending_calls.pop(name)
                int8_args, int8_kwargs = _clone_block_call(base_args, base_kwargs)
                int4_args, int4_kwargs = _clone_block_call(base_args, base_kwargs)
                with torch.inference_mode():
                    candidate = int8_modules[name](*int8_args, **int8_kwargs)
                    block_metrics.update(name, "int8", reference, candidate)
                    del candidate
                    candidate = int4_modules[name](*int4_args, **int4_kwargs)
                    block_metrics.update(name, "int4", reference, candidate)
                    del candidate
            return block_hook

        for name in block_names:
            handles.append(
                bf_modules[name].register_forward_pre_hook(make_pre_hook(name), with_kwargs=True)
            )
            handles.append(
                bf_modules[name].register_forward_hook(make_block_hook(name), with_kwargs=True)
            )

    bf16.set_additional_models("flux_klein_sensitivity", [int8, int4])
    prompt_runs = []
    try:
        for prompt in prompts:
            positive = nodes.CLIPTextEncode().encode(clip, prompt["text"])[0]
            negative = nodes.CLIPTextEncode().encode(clip, "")[0]
            latent = {
                "samples": torch.zeros(
                    [1, 128, args.height // 16, args.width // 16],
                    device=comfy.model_management.intermediate_device(),
                )
            }
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            run_started = time.perf_counter()
            result = nodes.KSampler().sample(
                bf16,
                int(prompt["seed"]),
                args.steps,
                args.cfg,
                args.sampler,
                args.scheduler,
                positive,
                negative,
                latent,
                1.0,
            )[0]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - run_started
            latent_rms = float(result["samples"].float().square().mean().sqrt().item())
            run = {
                "id": prompt["id"],
                "seed": int(prompt["seed"]),
                "text": prompt["text"],
                "seconds": elapsed,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "latent_rms": latent_rms,
            }
            prompt_runs.append(run)
            print(
                f"PROFILE {prompt['id']} {elapsed:.2f}s "
                f"peak={run['peak_cuda_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    report = {
        "schema_version": 1,
        "method": (
            "isolated quantized projection and block response on identical BF16 denoising activations"
        ),
        "models": {"bf16": args.bf16, "int8": args.int8, "int4": args.int4},
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "comfyui": getattr(comfyui_version, "__version__", "unknown"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "settings": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
        },
        "prompt_runs": prompt_runs,
        "projection_metrics": projection_metrics.finish(),
        "block_metrics": {} if args.skip_blocks else block_metrics.finish(),
        "total_seconds": time.time() - started,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {output_path}", flush=True)
    comfy.model_management.unload_all_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
