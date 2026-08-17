#!/usr/bin/env python3
"""Evaluate complete mixed FLUX.2 Klein policies in memory with ComfyUI."""

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
    parser.add_argument("--int4", required=True)
    parser.add_argument("--fp8")
    parser.add_argument("--nvfp4")
    parser.add_argument("--clip", default="qwen_3_8b.safetensors")
    parser.add_argument("--policies", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="flux2")
    return parser.parse_args()


def _load_prompts(path, limit):
    if path:
        with open(path, encoding="utf-8") as handle:
            prompts = json.load(handle)
    else:
        prompts = [dict(item) for item in DEFAULT_PROMPTS]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def _load_policy(path, eligible):
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported policy schema in {path}")
    default = raw.get("default_mode", "int4")
    layers = raw.get("layers", {})
    routes = {name: layers.get(name, default) for name in eligible}
    invalid = set(routes.values()) - {"bf16", "int8", "int4"}
    unknown = set(layers) - set(eligible)
    if invalid or unknown:
        raise ValueError(f"Invalid policy {path}: modes={invalid}, unknown={sorted(unknown)[:3]}")
    return raw, routes


def _metric(reference, candidate):
    ref = reference.float()
    cand = candidate.float()
    error = cand - ref
    return {
        "reference_sq": float((ref * ref).sum()),
        "candidate_sq": float((cand * cand).sum()),
        "error_sq": float((error * error).sum()),
        "dot": float((ref * cand).sum()),
        "elements": ref.numel(),
    }


def _merge(values):
    out = {key: sum(item[key] for item in values) for key in values[0]}
    out["relative_rmse_pct"] = 100 * math.sqrt(out["error_sq"] / max(out["reference_sq"], 1e-30))
    out["cosine"] = out["dot"] / math.sqrt(max(out["reference_sq"] * out["candidate_sq"], 1e-30))
    return out


def main():
    args = _args()
    prompts = _load_prompts(args.prompts, args.limit_prompts)
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
    int4 = loader.load_unet(args.int4, "default")[0]
    fp8 = loader.load_unet(args.fp8, "default")[0] if args.fp8 else None
    nvfp4 = loader.load_unet(args.nvfp4, "default")[0] if args.nvfp4 else None
    clip = nodes.CLIPLoader().load_clip(args.clip, "flux2", "default")[0]
    modules = {
        "bf16": dict(bf16.model.diffusion_model.named_modules()),
        "int8": dict(int8.model.diffusion_model.named_modules()),
        "int4": dict(int4.model.diffusion_model.named_modules()),
    }
    eligible = sorted(name for name in modules["bf16"] if _eligible_projection(name))
    if len(eligible) != 112:
        raise RuntimeError(f"Expected 112 projections, found {len(eligible)}")

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
            result = nodes.KSampler().sample(
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
        return result, time.perf_counter() - started

    references = []
    for condition in conditions:
        latent, elapsed = sample(bf16, condition)
        references.append(latent)
        print(f"REFERENCE {condition['prompt']['id']} {elapsed:.2f}s", flush=True)

    report = {
        "schema_version": 1,
        "settings": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "prompts": prompts,
        },
        "controls": {},
        "policies": [],
    }

    controls = {"all_int4": int4, "all_int8": int8}
    if fp8 is not None:
        controls["fp8_scaled"] = fp8
    if nvfp4 is not None:
        controls["nvfp4"] = nvfp4
    for name, model in controls.items():
        values = []
        times = []
        for condition, reference in zip(conditions, references):
            latent, elapsed = sample(model, condition)
            values.append(_metric(reference, latent))
            times.append(elapsed)
        report["controls"][name] = {**_merge(values), "seconds": times}
        print(
            f"CONTROL {name} rRMSE={report['controls'][name]['relative_rmse_pct']:.4f}% "
            f"cos={report['controls'][name]['cosine']:.7f}",
            flush=True,
        )

    for path in args.policies:
        raw, routes = _load_policy(path, eligible)
        model = int4.clone()
        counts = {mode: 0 for mode in ("bf16", "int8", "int4")}
        for name, mode in routes.items():
            counts[mode] += 1
            if mode != "int4":
                model.add_object_patch(f"diffusion_model.{name}", modules[mode][name])
        model.set_additional_models("flux_klein_policy_sources", [bf16, int8])
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
                    "relative_rmse_pct": 100 * math.sqrt(metric["error_sq"] / max(metric["reference_sq"], 1e-30)),
                }
            )
        result = {
            "name": raw["name"],
            "path": str(Path(path).resolve()),
            "counts": counts,
            **_merge(values),
            "seconds": times,
            "prompts": per_prompt,
        }
        report["policies"].append(result)
        print(
            f"POLICY {raw['name']} {counts} rRMSE={result['relative_rmse_pct']:.4f}% "
            f"cos={result['cosine']:.7f}",
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
