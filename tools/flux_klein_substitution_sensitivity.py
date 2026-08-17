#!/usr/bin/env python3
"""Measure the damage from substituting INT4 into an INT8 FLUX.2 Klein model.

Run this worker with ComfyUI's virtual environment.  Every candidate starts
from the same all-INT8 ConvRot checkpoint, replaces exactly one projection or
transformer block with its all-INT4 counterpart, and follows an identical
deterministic denoising trajectory.  Ranking forward substitutions avoids the
strong interaction bias produced by restoring layers from an all-INT4 model.
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
    parser.add_argument("--scope", choices=["blocks", "layers", "both"], default="both")
    parser.add_argument("--limit-candidates", type=int)
    return parser.parse_args()


def _load_prompts(path, limit):
    if path:
        with open(path, encoding="utf-8") as handle:
            prompts = json.load(handle)
    else:
        prompts = [dict(item) for item in DEFAULT_PROMPTS]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompts must be a non-empty JSON list")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit-prompts must be at least 1")
        prompts = prompts[:limit]
    return prompts


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


def _merge(metrics):
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
    prompts = _load_prompts(args.prompts, args.limit_prompts)
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not (comfy_root / "nodes.py").is_file():
        raise FileNotFoundError(comfy_root)
    output.parent.mkdir(parents=True, exist_ok=True)

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
    int8 = loader.load_unet(args.int8, "default")[0]
    int4 = loader.load_unet(args.int4, "default")[0]
    clip = nodes.CLIPLoader().load_clip(args.clip, "flux2", "default")[0]

    modules = {
        "bf16": dict(bf16.model.diffusion_model.named_modules()),
        "int8": dict(int8.model.diffusion_model.named_modules()),
        "int4": dict(int4.model.diffusion_model.named_modules()),
    }
    scopes = {}
    if args.scope in {"blocks", "both"}:
        scopes["blocks"] = sorted(name for name in modules["bf16"] if _eligible_block(name))
    if args.scope in {"layers", "both"}:
        scopes["layers"] = sorted(name for name in modules["bf16"] if _eligible_projection(name))
    if len(scopes.get("blocks", [])) not in {0, 32}:
        raise RuntimeError(f"Expected 32 blocks, found {len(scopes['blocks'])}")
    if len(scopes.get("layers", [])) not in {0, 112}:
        raise RuntimeError(f"Expected 112 projections, found {len(scopes['layers'])}")
    if args.limit_candidates is not None:
        if args.limit_candidates < 1:
            raise ValueError("--limit-candidates must be at least 1")
        scopes = {name: values[: args.limit_candidates] for name, values in scopes.items()}

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
            latent = nodes.KSampler().sample(
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
        return latent, time.perf_counter() - started

    report = {
        "schema_version": 1,
        "method": "one-at-a-time INT4 forward substitution into all-INT8 ConvRot",
        "models": {"bf16": args.bf16, "int8": args.int8, "int4": args.int4},
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
            "prompts": [condition["prompt"] for condition in conditions],
        },
        "scopes": {},
    }

    references = []
    for condition in conditions:
        latent, elapsed = sample(bf16, condition)
        references.append(latent)
        print(f"REFERENCE {condition['prompt']['id']} {elapsed:.2f}s", flush=True)

    baseline_metrics = []
    baseline_times = []
    for condition, reference in zip(conditions, references):
        latent, elapsed = sample(int8, condition)
        baseline_metrics.append(_metric(reference, latent))
        baseline_times.append(elapsed)
    baseline = _merge(baseline_metrics)
    report["baseline"] = {**baseline, "seconds": baseline_times}
    print(
        f"BASELINE int8 rRMSE={baseline['relative_rmse_pct']:.4f}% "
        f"cos={baseline['cosine']:.7f}",
        flush=True,
    )

    for scope_name, candidates in scopes.items():
        rows = []
        for index, name in enumerate(candidates, start=1):
            model = int8.clone()
            model.add_object_patch(f"diffusion_model.{name}", modules["int4"][name])
            model.set_additional_models("flux_klein_int4_substitution_source", [int4])
            values = []
            times = []
            per_prompt = []
            for condition, reference in zip(conditions, references):
                latent, elapsed = sample(model, condition)
                metric = _metric(reference, latent)
                values.append(metric)
                times.append(elapsed)
                per_prompt.append(
                    {
                        "id": condition["prompt"]["id"],
                        "relative_rmse_pct": 100.0 * math.sqrt(
                            metric["error_sq"] / max(metric["reference_sq"], 1e-30)
                        ),
                    }
                )
            merged = _merge(values)
            degradation = 100.0 * (
                merged["error_sq"] - baseline["error_sq"]
            ) / max(baseline["error_sq"], 1e-30)
            bf_module = modules["bf16"][name]
            weight_elements = sum(
                parameter.numel()
                for parameter in bf_module.parameters(recurse=True)
                if parameter.ndim >= 2
            )
            row = {
                "name": name,
                "weight_elements": weight_elements,
                **merged,
                "error_energy_increase_pct": degradation,
                "relative_rmse_increase_pct": 100.0 * (
                    merged["relative_rmse_pct"] - baseline["relative_rmse_pct"]
                ) / max(baseline["relative_rmse_pct"], 1e-30),
                "seconds": times,
                "prompts": per_prompt,
            }
            rows.append(row)
            report["scopes"][scope_name] = rows
            print(
                f"SUBSTITUTE {scope_name} {index:3}/{len(candidates)} "
                f"damage={degradation:+9.3f}% rRMSE={merged['relative_rmse_pct']:.4f}% {name}",
                flush=True,
            )
            with open(output, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {output}", flush=True)
    comfy.model_management.unload_all_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
