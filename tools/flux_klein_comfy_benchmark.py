#!/usr/bin/env python3
"""Generate FLUX.2 Klein comparison images and benchmark native ComfyUI runtime.

Run one isolated process per diffusion checkpoint with ComfyUI's virtual
environment.  Sampling, VAE decode, PNG save, peak-VRAM accounting, and timing
all happen through the installed ComfyUI runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from flux_klein_sensitivity import DEFAULT_PROMPTS


CASES = [
    (1024, DEFAULT_PROMPTS[0]),
    (1280, DEFAULT_PROMPTS[1]),
    (1536, DEFAULT_PROMPTS[2]),
    (2048, DEFAULT_PROMPTS[3]),
]


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model", required=True, help="ComfyUI diffusion-model filename")
    parser.add_argument("--label", required=True)
    parser.add_argument("--clip", default="qwen_3_8b.safetensors")
    parser.add_argument("--vae", default="Flux/flux2-vae.safetensors")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="flux2")
    parser.add_argument("--timed-runs", type=int, default=2)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1024, 1280, 1536, 2048])
    return parser.parse_args()


def main():
    args = _args()
    if args.timed_runs < 1:
        raise ValueError("--timed-runs must be positive")
    selected = set(args.resolutions)
    cases = [(size, prompt) for size, prompt in CASES if size in selected]
    if len(cases) != len(selected):
        raise ValueError(f"Supported resolutions are {[size for size, _ in CASES]}")
    comfy_root = Path(args.comfy_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    sys.argv = [sys.argv[0], "--highvram", "--disable-auto-launch"]
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import torch
    import comfy.model_management
    import comfy.utils
    import comfyui_version
    import folder_paths
    import nodes

    comfy.utils.PROGRESS_BAR_ENABLED = False
    folder_paths.set_output_directory(str(output_dir))
    model_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", args.model))

    load_started = time.perf_counter()
    model = nodes.UNETLoader().load_unet(args.model, "default")[0]
    model_load_seconds = time.perf_counter() - load_started
    clip = nodes.CLIPLoader().load_clip(args.clip, "flux2", "default")[0]
    vae = nodes.VAELoader().load_vae(args.vae)[0]

    conditions = {}
    for size, prompt in cases:
        conditions[size] = {
            "prompt": prompt,
            "positive": nodes.CLIPTextEncode().encode(clip, prompt["text"])[0],
            "negative": nodes.CLIPTextEncode().encode(clip, "")[0],
        }
    comfy.model_management.unload_all_models()
    torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "method": "isolated native ComfyUI warm-run benchmark",
        "label": args.label,
        "model": args.model,
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_load_seconds": model_load_seconds,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "comfyui": getattr(comfyui_version, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "native_nvfp4_compute": bool(comfy.model_management.supports_nvfp4_compute()),
        },
        "settings": {
            "steps": args.steps,
            "cfg": args.cfg,
            "sampler": args.sampler,
            "scheduler": args.scheduler,
            "timed_runs": args.timed_runs,
        },
        "cases": [],
    }

    def sample(size, condition):
        latent = {
            "samples": torch.zeros(
                [1, 128, size // 16, size // 16],
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
            )[0]

    for size, prompt in cases:
        condition = conditions[size]
        # Shape-specific warmup loads the checkpoint and primes installed kernels.
        warmup = sample(size, condition)
        torch.cuda.synchronize()
        del warmup
        torch.cuda.empty_cache()

        times = []
        peaks = []
        saved_latent = None
        for run_index in range(args.timed_runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            latent = sample(size, condition)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            peaks.append(torch.cuda.max_memory_allocated())
            if saved_latent is None:
                saved_latent = latent
            else:
                del latent
            print(
                f"SAMPLE {args.label} {size}px run={run_index + 1}/{args.timed_runs} "
                f"{elapsed:.4f}s peak={peaks[-1] / 2**30:.3f}GiB",
                flush=True,
            )

        comfy.model_management.unload_all_models()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        with torch.inference_mode():
            image = nodes.VAEDecode().decode(vae, saved_latent)[0]
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        decode_peak = torch.cuda.max_memory_allocated()
        prefix = f"{size}px/{args.label}_{size}px_{prompt['id']}"
        saved = nodes.SaveImage().save_images(
            image,
            filename_prefix=prefix,
            prompt={
                "model": args.model,
                "label": args.label,
                "text": prompt["text"],
                "seed": prompt["seed"],
                **report["settings"],
            },
            extra_pnginfo={"benchmark": "FLUX.2 Klein quantization comparison"},
        )["ui"]["images"][0]
        image_path = output_dir / saved["subfolder"] / saved["filename"]
        del saved_latent, image
        comfy.model_management.unload_all_models()
        torch.cuda.empty_cache()

        median_seconds = statistics.median(times)
        row = {
            "resolution": size,
            "width": size,
            "height": size,
            "prompt": prompt,
            "sampling_seconds": times,
            "median_sampling_seconds": median_seconds,
            "steps_per_second": args.steps / median_seconds,
            "diffusion_peak_allocated_bytes": max(peaks),
            "diffusion_peak_allocated_gib": max(peaks) / 2**30,
            "decode_seconds": decode_seconds,
            "decode_peak_allocated_bytes": decode_peak,
            "decode_peak_allocated_gib": decode_peak / 2**30,
            "image": str(image_path),
        }
        report["cases"].append(row)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"SAVED {image_path}", flush=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"REPORT {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
