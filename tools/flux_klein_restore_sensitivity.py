#!/usr/bin/env python3
"""Measure end-to-end gain from upgrading each FLUX.2 Klein layer/block.

Execute with ComfyUI's venv.  For each candidate this worker starts from the
fully quantized ConvRot checkpoint, substitutes exactly one BF16 or INT8 module, runs
the same deterministic denoising trajectory, and measures final latent error
against BF16.  This complements local activation error with downstream impact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from flux_klein_sensitivity import DEFAULT_PROMPTS, _eligible_block, _eligible_projection


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--int4", required=True)
    parser.add_argument("--clip", default="qwen_3_8b.safetensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="flux2")
    parser.add_argument("--variant", choices=["int8", "int4", "both"], default="int4")
    parser.add_argument(
        "--restore-mode",
        choices=["bf16", "int8"],
        default="bf16",
        help="Precision substituted into each candidate (INT8 is useful for W4-majority search).",
    )
    parser.add_argument("--scope", choices=["blocks", "layers", "both"], default="both")
    parser.add_argument("--limit-candidates", type=int, help="Smoke-test only the first N candidates")
    parser.add_argument(
        "--start-candidate",
        type=int,
        default=0,
        help="Skip this many lexicographically sorted candidates in each scope.",
    )
    return parser.parse_args()


def _prompts(path, limit):
    if path:
        with open(path, encoding="utf-8") as handle:
            values = json.load(handle)
    else:
        values = [dict(item) for item in DEFAULT_PROMPTS]
    if not isinstance(values, list) or not values:
        raise ValueError("Prompts must be a non-empty JSON list")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit-prompts must be at least 1")
        values = values[:limit]
    return values


def _metric(reference, candidate):
    ref = reference.float()
    cand = candidate.float()
    error = cand - ref
    return {
        "reference_sq": float((ref * ref).sum().item()),
        "candidate_sq": float((cand * cand).sum().item()),
        "error_sq": float((error * error).sum().item()),
        "dot": float((ref * cand).sum().item()),
        "elements": ref.numel(),
    }


def _merge_metrics(metrics):
    totals = {
        key: sum(metric[key] for metric in metrics)
        for key in ("reference_sq", "candidate_sq", "error_sq", "dot", "elements")
    }
    totals["relative_rmse_pct"] = 100.0 * math.sqrt(
        totals["error_sq"] / max(totals["reference_sq"], 1e-30)
    )
    totals["cosine"] = totals["dot"] / math.sqrt(
        max(totals["reference_sq"] * totals["candidate_sq"], 1e-30)
    )
    return totals


def main():
    args = _parse_args()
    prompts = _prompts(args.prompts, args.limit_prompts)
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    if not (comfy_root / "nodes.py").is_file():
        raise FileNotFoundError(comfy_root)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.argv = [sys.argv[0], "--highvram", "--disable-auto-launch", "--disable-metadata"]
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import torch
    import comfy.model_management
    import comfy.utils
    import comfyui_version
    import nodes

    comfy.utils.PROGRESS_BAR_ENABLED = False

    loader = nodes.UNETLoader()
    bf16 = loader.load_unet(args.bf16, "default")[0]
    variants = {}
    int8_model = None
    if args.variant in {"int8", "both"} or args.restore_mode == "int8":
        int8_model = loader.load_unet(args.int8, "default")[0]
    if args.variant in {"int8", "both"}:
        variants["int8"] = int8_model
    if args.variant in {"int4", "both"}:
        variants["int4"] = loader.load_unet(args.int4, "default")[0]
    clip = nodes.CLIPLoader().load_clip(args.clip, "flux2", "default")[0]

    bf_modules = dict(bf16.model.diffusion_model.named_modules())
    restore_model = bf16 if args.restore_mode == "bf16" else int8_model
    restore_modules = dict(restore_model.model.diffusion_model.named_modules())
    scopes = {}
    if args.scope in {"blocks", "both"}:
        scopes["blocks"] = sorted(name for name in bf_modules if _eligible_block(name))
    if args.scope in {"layers", "both"}:
        scopes["layers"] = sorted(name for name in bf_modules if _eligible_projection(name))
    if len(scopes.get("blocks", [])) not in {0, 32}:
        raise RuntimeError(f"Expected 32 blocks, found {len(scopes['blocks'])}")
    if len(scopes.get("layers", [])) not in {0, 112}:
        raise RuntimeError(f"Expected 112 layers, found {len(scopes['layers'])}")
    if args.start_candidate < 0:
        raise ValueError("--start-candidate must be non-negative")
    if args.start_candidate:
        scopes = {key: values[args.start_candidate :] for key, values in scopes.items()}
    if args.limit_candidates is not None:
        if args.limit_candidates < 1:
            raise ValueError("--limit-candidates must be at least 1")
        scopes = {key: values[: args.limit_candidates] for key, values in scopes.items()}

    conditions = []
    for prompt in prompts:
        conditions.append(
            {
                "prompt": prompt,
                "positive": nodes.CLIPTextEncode().encode(clip, prompt["text"])[0],
                "negative": nodes.CLIPTextEncode().encode(clip, "")[0],
            }
        )

    def sample(model, condition):
        latent = {
            "samples": torch.zeros(
                [1, 128, args.height // 16, args.width // 16],
                device=comfy.model_management.intermediate_device(),
            )
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = nodes.KSampler().sample(
                model,
                int(condition["prompt"]["seed"]),
                args.steps,
                args.cfg,
                args.sampler,
                args.scheduler,
                condition["positive"],
                condition["negative"],
                latent,
                1.0,
            )[0]["samples"].detach().float().cpu()
        torch.cuda.synchronize()
        return output, time.perf_counter() - started

    report = {
        "schema_version": 1,
        "method": (
            f"one-at-a-time {args.restore_mode.upper()} upgrade from a fully quantized checkpoint"
        ),
        "models": {"bf16": args.bf16, **{name: {"int8": args.int8, "int4": args.int4}[name] for name in variants}},
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
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
            "prompts": [item["prompt"] for item in conditions],
        },
        "variants": {},
    }

    references = []
    reference_times = []
    for condition in conditions:
        latent, elapsed = sample(bf16, condition)
        references.append(latent)
        reference_times.append(elapsed)
        print(f"REFERENCE {condition['prompt']['id']} {elapsed:.2f}s", flush=True)
    report["bf16_seconds"] = reference_times

    for variant_name, quant_model in variants.items():
        baseline_latents = []
        baseline_metrics = []
        baseline_times = []
        for condition, reference in zip(conditions, references):
            latent, elapsed = sample(quant_model, condition)
            baseline_latents.append(latent)
            baseline_metrics.append(_metric(reference, latent))
            baseline_times.append(elapsed)
        baseline = _merge_metrics(baseline_metrics)
        variant_report = {
            "baseline": baseline,
            "baseline_seconds": baseline_times,
            "scopes": {},
        }
        report["variants"][variant_name] = variant_report
        print(
            f"BASELINE {variant_name} rRMSE={baseline['relative_rmse_pct']:.4f}% "
            f"cos={baseline['cosine']:.7f}",
            flush=True,
        )

        for scope_name, candidates in scopes.items():
            rows = []
            for index, name in enumerate(candidates, start=1):
                patched = quant_model.clone()
                patched.add_object_patch(f"diffusion_model.{name}", restore_modules[name])
                patched.set_additional_models("flux_klein_precision_restore", [restore_model])
                metrics = []
                elapsed_total = 0.0
                per_prompt = []
                for condition, reference in zip(conditions, references):
                    latent, elapsed = sample(patched, condition)
                    metric = _metric(reference, latent)
                    metrics.append(metric)
                    elapsed_total += elapsed
                    per_prompt.append(
                        {
                            "id": condition["prompt"]["id"],
                            "relative_rmse_pct": 100.0 * math.sqrt(
                                metric["error_sq"] / max(metric["reference_sq"], 1e-30)
                            ),
                            "seconds": elapsed,
                        }
                    )
                merged = _merge_metrics(metrics)
                error_reduction = 100.0 * (
                    baseline["error_sq"] - merged["error_sq"]
                ) / max(baseline["error_sq"], 1e-30)
                row = {
                    "name": name,
                    "weight_elements": sum(
                        parameter.numel()
                        for parameter in bf_modules[name].parameters(recurse=True)
                        if parameter.ndim >= 2
                    ),
                    **merged,
                    "error_energy_reduction_pct": error_reduction,
                    "relative_rmse_reduction_pct": 100.0 * (
                        baseline["relative_rmse_pct"] - merged["relative_rmse_pct"]
                    ) / max(baseline["relative_rmse_pct"], 1e-30),
                    "seconds": elapsed_total,
                    "prompts": per_prompt,
                }
                rows.append(row)
                print(
                    f"RESTORE {variant_name} {scope_name} {index:3}/{len(candidates)} "
                    f"gain={error_reduction:+8.3f}% rRMSE={merged['relative_rmse_pct']:.4f}% {name}",
                    flush=True,
                )
                # Persist completed candidates so a long calibration retains evidence.
                variant_report["scopes"][scope_name] = rows
                with open(output_path, "w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2)
                    handle.write("\n")
            variant_report["scopes"][scope_name] = rows

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {output_path}", flush=True)
    comfy.model_management.unload_all_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
