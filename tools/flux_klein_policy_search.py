#!/usr/bin/env python3
"""Interaction-aware search for a compact mixed INT4/INT8 FLUX.2 Klein policy.

The search keeps the INT4 weight budget effectively constant by swapping
same-shaped projection groups, then scores every complete policy against BF16
on deterministic ComfyUI denoising trajectories.  A seeded simulated annealing
walk avoids treating layer errors as independent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    parser.add_argument("--clip", default="qwen_3_8b.safetensors")
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--initial-policy", help="Optional schema-v1 mixed policy used as the starting state")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--prompts")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="flux2")
    parser.add_argument("--initial-temperature", type=float, default=0.12)
    parser.add_argument("--final-temperature", type=float, default=0.002)
    parser.add_argument(
        "--restart-interval",
        type=int,
        default=30,
        help="Reset the walk to the best state after this many non-improving proposals (0 disables).",
    )
    return parser.parse_args()


def _load_prompts(path, limit):
    if path:
        with open(path, encoding="utf-8") as handle:
            prompts = json.load(handle)
    else:
        prompts = [dict(item) for item in DEFAULT_PROMPTS]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError("At least one prompt is required")
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
    totals["objective"] = totals["error_sq"] / max(totals["reference_sq"], 1e-30)
    totals["relative_rmse_pct"] = 100.0 * math.sqrt(totals["objective"])
    totals["cosine"] = totals["dot"] / math.sqrt(
        max(totals["reference_sq"] * totals["candidate_sq"], 1e-30)
    )
    return totals


def main():
    args = _args()
    if not 0.0 < args.target <= 1.0:
        raise ValueError("--target must be in (0, 1]")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    rng = random.Random(args.seed)
    prompts = _load_prompts(args.prompts, args.limit_prompts)
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.sensitivity).expanduser().resolve(), encoding="utf-8") as handle:
        sensitivity = json.load(handle)
    measured = sensitivity.get("scopes", {}).get("layers", [])
    if len(measured) != 112:
        raise ValueError(f"Expected 112 sensitivity rows, found {len(measured)}")

    weights = {row["name"]: int(row["weight_elements"]) for row in measured}
    total_weight = sum(weights.values())
    ranked = sorted(
        measured,
        key=lambda row: (
            float(row["error_energy_increase_pct"]) / max(int(row["weight_elements"]), 1),
            row["name"],
        ),
    )
    selected = set()
    selected_weight = 0
    for row in ranked:
        selected.add(row["name"])
        selected_weight += weights[row["name"]]
        if selected_weight >= args.target * total_weight:
            break
    if args.initial_policy:
        with open(Path(args.initial_policy).expanduser().resolve(), encoding="utf-8") as handle:
            initial_policy = json.load(handle)
        if initial_policy.get("schema_version") != 1:
            raise ValueError("Initial policy must use schema_version 1")
        default_mode = initial_policy.get("default_mode", "int8")
        routes = {
            name: initial_policy.get("layers", {}).get(name, default_mode)
            for name in weights
        }
        if set(routes.values()) - {"int8", "int4"}:
            raise ValueError("Policy search only accepts INT8/INT4 starting routes")
        unknown = set(initial_policy.get("layers", {})) - set(weights)
        if unknown:
            raise ValueError(f"Unknown initial-policy layers: {sorted(unknown)[:3]}")
        selected = {name for name, mode in routes.items() if mode == "int4"}
        selected_weight = sum(weights[name] for name in selected)
        if abs(selected_weight / total_weight - args.target) > 0.02:
            raise ValueError(
                "Initial policy weight fraction differs from --target by more than two percentage points"
            )

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
    int4_modules = dict(int4.model.diffusion_model.named_modules())
    eligible = sorted(
        name for name in dict(bf16.model.diffusion_model.named_modules()) if _eligible_projection(name)
    )
    if set(eligible) != set(weights):
        raise RuntimeError("Sensitivity/model layer sets do not match")

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
        with torch.inference_mode():
            return nodes.KSampler().sample(
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

    references = []
    for condition in conditions:
        torch.cuda.synchronize()
        started = time.perf_counter()
        references.append(sample(bf16, condition))
        torch.cuda.synchronize()
        print(f"REFERENCE {condition['prompt']['id']} {time.perf_counter() - started:.2f}s", flush=True)

    cache = {}

    def evaluate(layer_names):
        key = tuple(sorted(layer_names))
        if key in cache:
            return cache[key]
        model = int8.clone()
        for name in key:
            model.add_object_patch(f"diffusion_model.{name}", int4_modules[name])
        model.set_additional_models("flux_klein_int4_search_source", [int4])
        values = []
        per_prompt = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        for condition, reference in zip(conditions, references):
            latent = sample(model, condition)
            metric = _metric(reference, latent)
            values.append(metric)
            per_prompt.append(
                {
                    "id": condition["prompt"]["id"],
                    "relative_rmse_pct": 100.0 * math.sqrt(
                        metric["error_sq"] / max(metric["reference_sq"], 1e-30)
                    ),
                }
            )
        torch.cuda.synchronize()
        result = {**_merge(values), "seconds": time.perf_counter() - started, "prompts": per_prompt}
        cache[key] = result
        return result

    current = set(selected)
    current_result = evaluate(current)
    best = set(current)
    best_result = current_result
    history = []
    non_improving = 0
    # Same-sized swaps hold the serialized W4 budget exactly constant.  The
    # small set of projection shapes makes this both strict and well connected.
    groups = {}
    for name, count in weights.items():
        groups.setdefault(count, []).append(name)

    for iteration in range(1, args.iterations + 1):
        progress = (iteration - 1) / max(args.iterations - 1, 1)
        temperature = args.initial_temperature * (
            args.final_temperature / args.initial_temperature
        ) ** progress
        proposal = set(current)
        swaps = 2 if rng.random() < 0.18 else 1
        changed = False
        for _ in range(swaps):
            removable = [name for name in proposal if any(x not in proposal for x in groups[weights[name]])]
            if not removable:
                break
            old = rng.choice(removable)
            replacements = [name for name in groups[weights[old]] if name not in proposal]
            new = rng.choice(replacements)
            proposal.remove(old)
            proposal.add(new)
            changed = True
        if not changed:
            continue
        proposal_result = evaluate(proposal)
        delta = proposal_result["objective"] - current_result["objective"]
        accept = delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12))
        if accept:
            current = proposal
            current_result = proposal_result
        improved = proposal_result["objective"] < best_result["objective"]
        if improved:
            best = set(proposal)
            best_result = proposal_result
            non_improving = 0
        else:
            non_improving += 1
        history.append(
            {
                "iteration": iteration,
                "temperature": temperature,
                "accepted": accept,
                "improved": improved,
                "proposal_relative_rmse_pct": proposal_result["relative_rmse_pct"],
                "current_relative_rmse_pct": current_result["relative_rmse_pct"],
                "best_relative_rmse_pct": best_result["relative_rmse_pct"],
            }
        )
        print(
            f"SEARCH {iteration:4}/{args.iterations} proposal={proposal_result['relative_rmse_pct']:8.4f}% "
            f"current={current_result['relative_rmse_pct']:8.4f}% best={best_result['relative_rmse_pct']:8.4f}% "
            f"{'accept' if accept else 'reject'}{' BEST' if improved else ''}",
            flush=True,
        )
        if args.restart_interval and non_improving >= args.restart_interval:
            current = set(best)
            current_result = best_result
            non_improving = 0
            print(f"RESTART best={best_result['relative_rmse_pct']:.4f}%", flush=True)
        if improved or iteration % 10 == 0:
            report = {
                "schema_version": 1,
                "method": "seeded same-shape simulated annealing over complete mixed policies",
                "runtime": {
                    "python": sys.version,
                    "executable": sys.executable,
                    "torch": torch.__version__,
                    "comfyui": getattr(comfyui_version, "__version__", "unknown"),
                    "gpu": torch.cuda.get_device_name(),
                },
                "settings": {
                    "target": args.target,
                    "iterations": args.iterations,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "prompts": prompts,
                },
                "int4_weight_fraction": sum(weights[name] for name in best) / total_weight,
                "best": best_result,
                "policy": {
                    "schema_version": 1,
                    "name": f"search_{round(args.target * 100):02d}_seed_{args.seed}",
                    "default_mode": "int8",
                    "layers": {name: "int4" for name in sorted(best)},
                },
                "history": history,
            }
            with open(output, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

    report = {
        "schema_version": 1,
        "method": "seeded same-shape simulated annealing over complete mixed policies",
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "comfyui": getattr(comfyui_version, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
        },
        "settings": {
            "target": args.target,
            "iterations": args.iterations,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg": args.cfg,
            "prompts": prompts,
        },
        "int4_weight_fraction": sum(weights[name] for name in best) / total_weight,
        "best": best_result,
        "policy": {
            "schema_version": 1,
            "name": f"search_{round(args.target * 100):02d}_seed_{args.seed}",
            "default_mode": "int8",
            "layers": {name: "int4" for name in sorted(best)},
        },
        "history": history,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    policy_output = output.with_suffix(".policy.json")
    with open(policy_output, "w", encoding="utf-8") as handle:
        json.dump(report["policy"], handle, indent=2)
        handle.write("\n")
    print(
        f"BEST rRMSE={best_result['relative_rmse_pct']:.4f}% "
        f"W4={sum(weights[name] for name in best) / total_weight * 100:.2f}% "
        f"REPORT {output} POLICY {policy_output}",
        flush=True,
    )
    comfy.model_management.unload_all_models()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
