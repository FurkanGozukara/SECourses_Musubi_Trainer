"""Run a controlled legacy-vs-automatic SDPA training benchmark pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import toml


ROOT = Path(__file__).resolve().parents[2]
TRAINER_ROOT = ROOT / "musubi-tuner"
VENV_ROOT = ROOT / "venv"


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=JSON_VALUE: {item}")
        key, raw_value = item.split("=", 1)
        overrides[key] = _parse_value(raw_value)
    return overrides


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: list[str], *, timeout: int = 30) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return (result.stdout or result.stderr).strip()


def _gpu_metadata(gpu_index: int) -> dict[str, Any]:
    query = "name,uuid,driver_version,memory.total"
    output = _run_text(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [value.strip() for value in output.split(",")]
    return {
        "index": gpu_index,
        "name": values[0] if values else "unknown",
        "uuid": values[1] if len(values) > 1 else "unknown",
        "driver_version": values[2] if len(values) > 2 else "unknown",
        "memory_total_mib": float(values[3]) if len(values) > 3 else None,
    }


def _attention_probe(gpu_index: int) -> dict[str, Any]:
    python = VENV_ROOT / "Scripts" / "python.exe"
    probe_code = """
import json
import torch
from musubi_tuner.modules.attention import resolve_sdpa_backend, should_use_external_flash_for_sdpa
try:
    native = bool(torch.backends.cuda.is_flash_attention_available())
except Exception:
    native = False
print(json.dumps({
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    "native_flash_available": native,
    "external_flash_training_probe": should_use_external_flash_for_sdpa(),
    "automatic_backend": resolve_sdpa_backend(),
    "legacy_backend": resolve_sdpa_backend(True),
}))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    result = subprocess.run(
        [str(python), "-c", probe_code],
        cwd=TRAINER_ROOT,
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "status": "failed",
        "return_code": result.returncode,
        "stderr": result.stderr[-1000:],
    }


def _gpu_memory_used(gpu_index: int) -> float | None:
    output = _run_text(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    try:
        return float(output.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def _parse_step_rate(log_text: str) -> tuple[float | None, int | None]:
    seconds_per_step = [
        float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)s/it", log_text)
    ]
    completed_steps = [
        int(value)
        for value in re.findall(r"steps:\s+\d+%[^\r\n]*?\|\s*(\d+)/\d+", log_text)
    ]
    if seconds_per_step:
        return seconds_per_step[-1], completed_steps[-1] if completed_steps else None

    iterations_per_second = [
        float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)it/s", log_text)
    ]
    if iterations_per_second and iterations_per_second[-1] > 0:
        return 1.0 / iterations_per_second[-1], completed_steps[
            -1
        ] if completed_steps else None
    return None, completed_steps[-1] if completed_steps else None


def _prepare_config(
    base_config: Path,
    output_root: Path,
    name: str,
    mode: str,
    steps: int,
    overrides: dict[str, Any],
    drop_keys: set[str],
) -> Path:
    values = toml.load(base_config)
    for key in {
        "max_train_epochs",
        "save_every_n_epochs",
        "save_every_n_steps",
        "sample_every_n_epochs",
        "sample_every_n_steps",
        "compile",
        "compile_fullgraph",
        "block_swap_h2d_only",
        "use_pinned_memory_for_block_swap",
    } | drop_keys:
        values.pop(key, None)

    values.update(overrides)
    values.update(
        {
            "blocks_to_swap": 0,
            "max_train_steps": steps,
            "output_dir": str((output_root / mode / "weights").resolve()),
            "output_name": f"benchmark_{name}_{mode}",
            "sdpa": True,
            "use_legacy_sdpa": mode == "legacy",
        }
    )
    for key in ("flash_attn", "flash3", "sage_attn", "xformers"):
        values.pop(key, None)

    config_path = output_root / mode / "training.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml.dumps(values), encoding="utf-8")
    return config_path


def _cleanup_weights(output_root: Path) -> list[str]:
    removed: list[str] = []
    weights_root = (output_root / "weights").resolve()
    if output_root.resolve() not in weights_root.parents:
        raise RuntimeError(f"Refusing to clean unexpected output path: {weights_root}")
    if not weights_root.exists():
        return removed
    for path in weights_root.rglob("*.safetensors"):
        removed.append(path.name)
        path.unlink()
    return removed


def _run_mode(
    *,
    name: str,
    mode: str,
    config_path: Path,
    train_script: Path,
    output_root: Path,
    gpu_index: int,
    cleanup_weights: bool,
) -> dict[str, Any]:
    accelerate = VENV_ROOT / "Scripts" / "accelerate.exe"
    command = [
        str(accelerate),
        "launch",
        "--num_processes",
        "1",
        "--num_machines",
        "1",
        "--num_cpu_threads_per_process",
        "1",
        str(train_script),
        "--config_file",
        str(config_path),
    ]
    log_path = output_root / mode / "training.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )

    baseline_memory = _gpu_memory_used(gpu_index)
    peak_memory = baseline_memory
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"# command: {subprocess.list2cmdline(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=TRAINER_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            used = _gpu_memory_used(gpu_index)
            if used is not None:
                peak_memory = used if peak_memory is None else max(peak_memory, used)
            time.sleep(0.5)
        return_code = process.wait()
    duration = time.time() - started

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    seconds_per_step, completed_steps = _parse_step_rate(log_text)
    removed_weights = _cleanup_weights(output_root / mode) if cleanup_weights else []
    return {
        "name": name,
        "mode": mode,
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "duration_seconds": round(duration, 3),
        "seconds_per_step": seconds_per_step,
        "completed_steps": completed_steps,
        "peak_memory_mib": peak_memory,
        "baseline_memory_mib": baseline_memory,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "log": str(log_path),
        "command": command,
        "removed_weight_files": removed_weights,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).with_name("results")
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=JSON_VALUE"
    )
    parser.add_argument("--drop", dest="drop_keys", action="append", default=[])
    parser.add_argument(
        "--order", choices=["legacy-first", "automatic-first"], default="legacy-first"
    )
    parser.add_argument("--keep-weights", action="store_true")
    args = parser.parse_args()

    base_config = args.base_config.resolve()
    train_script = args.train_script.resolve()
    if not base_config.is_file():
        parser.error(f"Base config does not exist: {base_config}")
    if not train_script.is_file():
        parser.error(f"Training script does not exist: {train_script}")
    if args.steps < 2:
        parser.error("--steps must be at least 2")

    output_root = (args.output_root / args.name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    overrides = _parse_overrides(args.overrides)
    modes = (
        ["legacy", "automatic"]
        if args.order == "legacy-first"
        else ["automatic", "legacy"]
    )
    results: list[dict[str, Any]] = []
    attention_probe = _attention_probe(args.gpu_index)
    for index, mode in enumerate(modes):
        config_path = _prepare_config(
            base_config,
            output_root,
            args.name,
            mode,
            args.steps,
            overrides,
            set(args.drop_keys),
        )
        result = _run_mode(
            name=args.name,
            mode=mode,
            config_path=config_path,
            train_script=train_script,
            output_root=output_root,
            gpu_index=args.gpu_index,
            cleanup_weights=not args.keep_weights,
        )
        results.append(result)
        result_by_mode = {item["mode"]: item for item in results}
        legacy_rate = result_by_mode.get("legacy", {}).get("seconds_per_step")
        automatic_rate = result_by_mode.get("automatic", {}).get("seconds_per_step")
        speedup_percent = None
        if legacy_rate and automatic_rate:
            speedup_percent = round(
                (legacy_rate - automatic_rate) / legacy_rate * 100.0, 2
            )
        (output_root / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at_unix": time.time(),
                    "gpu": _gpu_metadata(args.gpu_index),
                    "attention_probe": attention_probe,
                    "base_config": str(base_config),
                    "train_script": str(train_script),
                    "steps": args.steps,
                    "blocks_to_swap": 0,
                    "automatic_speedup_percent": speedup_percent,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if result["return_code"] != 0:
            print(f"{args.name} {mode} failed; see {result['log']}")
        if index + 1 < len(modes):
            time.sleep(3)

    for result in results:
        print(
            f"{result['mode']}: status={result['status']} "
            f"seconds_per_step={result['seconds_per_step']} peak_memory_mib={result['peak_memory_mib']}"
        )
    return 0 if all(result["return_code"] == 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
