#!/usr/bin/env python3
"""Ablate each W4 route in a mixed FLUX.2 Klein policy at a target resolution."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from flux_klein_sensitivity import DEFAULT_PROMPTS, _eligible_projection


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--bf16", required=True)
    parser.add_argument("--int8", required=True)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip", default="qwen_3_8b.safetensors")
    parser.add_argument("--prompt-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="flux2")
    return parser.parse_args()


def _metric(reference, candidate):
    ref = reference.float()
    cand = candidate.float()
    error = cand - ref
    reference_sq = float((ref * ref).sum().item())
    candidate_sq = float((cand * cand).sum().item())
    error_sq = float((error * error).sum().item())
    dot = float((ref * cand).sum().item())
    return {
        "reference_sq": reference_sq,
        "candidate_sq": candidate_sq,
        "error_sq": error_sq,
        "dot": dot,
        "elements": ref.numel(),
        "relative_rmse_pct": 100.0 * math.sqrt(error_sq / max(reference_sq, 1e-30)),
        "cosine": dot / math.sqrt(max(reference_sq * candidate_sq, 1e-30)),
    }


def main():
    args = _args()
    if not 0 <= args.prompt_index < len(DEFAULT_PROMPTS):
        raise ValueError(f"--prompt-index must be in [0, {len(DEFAULT_PROMPTS) - 1}]")
    prompt = DEFAULT_PROMPTS[args.prompt_index]
    with open(Path(args.policy).expanduser().resolve(), encoding="utf-8") as handle:
        policy = json.load(handle)
    if policy.get("schema_version") != 1:
        raise ValueError("Policy must use schema_version 1")
    w4_layers = sorted(name for name, mode in policy.get("layers", {}).items() if mode == "int4")
    if not w4_layers:
        raise ValueError("Policy contains no explicit INT4 layers")

    comfy_root = Path(args.comfy_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [sys.argv[0], "--highvram", "--disable-auto-launch", "--disable-metadata"]
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import torch
    import comfy.model_management
    import comfy.utils
    import nodes

    comfy.utils.PROGRESS_BAR_ENABLED = False
    loader = nodes.UNETLoader()
    bf16 = loader.load_unet(args.bf16, "default")[0]
    int8 = loader.load_unet(args.int8, "default")[0]
    mixed = loader.load_unet(args.mixed, "default")[0]
    clip = nodes.CLIPLoader().load_clip(args.clip, "flux2", "default")[0]
    int8_modules = dict(int8.model.diffusion_model.named_modules())
    eligible = {
        name for name in dict(bf16.model.diffusion_model.named_modules()) if _eligible_projection(name)
    }
    unknown = set(w4_layers) - eligible
    if unknown:
        raise ValueError(f"Unknown policy layers: {sorted(unknown)}")
    positive = nodes.CLIPTextEncode().encode(clip, prompt["text"])[0]
    negative = nodes.CLIPTextEncode().encode(clip, "")[0]

    def sample(model):
        latent = {
            "samples": torch.zeros(
                [1, 128, args.height // 16, args.width // 16],
                device=comfy.model_management.intermediate_device(),
            )
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = nodes.KSampler().sample(
                model,
                int(prompt["seed"]),
                args.steps,
                args.cfg,
                args.sampler,
                args.scheduler,
                positive,
                negative,
                latent,
                1.0,
            )[0]["samples"].detach().float().cpu()
        torch.cuda.synchronize()
        return result, time.perf_counter() - started

    reference, reference_seconds = sample(bf16)
    baseline_latent, baseline_seconds = sample(mixed)
    baseline = _metric(reference, baseline_latent)
    report = {
        "schema_version": 1,
        "method": "restore one selective-W4 projection to the compiled INT8 module",
        "models": {"bf16": args.bf16, "int8": args.int8, "mixed": args.mixed},
        "policy": str(Path(args.policy).resolve()),
        "settings": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "prompt": prompt,
        },
        "reference_seconds": reference_seconds,
        "baseline": {**baseline, "seconds": baseline_seconds},
        "ablations": [],
    }
    print(
        f"BASELINE {baseline['relative_rmse_pct']:.4f}% cosine={baseline['cosine']:.7f}",
        flush=True,
    )
    for index, name in enumerate(w4_layers, start=1):
        model = mixed.clone()
        model.add_object_patch(f"diffusion_model.{name}", int8_modules[name])
        model.set_additional_models("flux_klein_ablation_int8_source", [int8])
        latent, seconds = sample(model)
        metric = _metric(reference, latent)
        reduction = 100.0 * (baseline["error_sq"] - metric["error_sq"]) / max(
            baseline["error_sq"], 1e-30
        )
        row = {
            "restored_layer": name,
            **metric,
            "error_energy_reduction_pct": reduction,
            "seconds": seconds,
        }
        report["ablations"].append(row)
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(
            f"RESTORE {index}/{len(w4_layers)} gain={reduction:+8.3f}% "
            f"rRMSE={metric['relative_rmse_pct']:.4f}% {name}",
            flush=True,
        )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
