import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.hunyuan_model.models import load_transformer
from musubi_tuner.networks import lora
from musubi_tuner.qwen_image import qwen_image_model
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner_gui.custom_logging import setup_logging

log = setup_logging()


def _cleanup_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(choice: str) -> torch.device:
    if not choice:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalized = choice.strip().lower()
    if "auto" in normalized:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA requested but unavailable. Falling back to CPU.")
    return torch.device("cpu")


def _resolve_output_dtype(choice: Optional[str]) -> Optional[torch.dtype]:
    if not choice:
        return None
    normalized = choice.strip().lower()
    if normalized.startswith("auto"):
        return None
    mapping = {
        "fp16": torch.float16,
        "bf16": getattr(torch, "bfloat16", None),
        "float32": torch.float32,
    }
    dtype = mapping.get(normalized)
    if dtype is None:
        log.warning("Unsupported LoRA dtype '%s'. Using per-tensor dtype.", choice)
    return dtype


def _load_base_model(
    dit_path: str,
    model_type: str,
    dit_in_channels: int,
    device: torch.device,
):
    if model_type == "qwen":
        log.info("Loading Qwen Image transformer for merge")
        transformer = qwen_image_model.load_qwen_image_model(
            device=device,
            dit_path=dit_path,
            attn_mode="torch",
            split_attn=False,
            loading_device=device,
            dit_weight_dtype=torch.bfloat16,
            fp8_scaled=False,
            num_layers=None,
            disable_numpy_memmap=True,
        )
        transformer.to(device)
        transformer.eval()
        return transformer

    log.info("Loading DiT transformer for merge")
    transformer = load_transformer(
        dit_path,
        "torch",
        False,
        "cpu",
        torch.bfloat16,
        in_channels=dit_in_channels,
    )
    transformer.eval()
    return transformer


def _merge_loras_into_transformer(
    transformer,
    lora_paths: List[str],
    multipliers: List[float],
    device: torch.device,
) -> Tuple[bool, str]:
    try:
        for lora_path, multiplier in zip(lora_paths, multipliers):
            log.info("Merging LoRA %s with multiplier %s", lora_path, multiplier)
            weights_sd = load_file(lora_path)
            network = lora.create_arch_network_from_weights(
                multiplier,
                weights_sd,
                unet=transformer,
                for_inference=True,
            )
            network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)
            del weights_sd, network
            _cleanup_memory()
    except Exception as exc:
        log.exception("Failed merging LoRA weights: %s", exc)
        return False, str(exc)

    return True, "Merged successfully"


def _merge_and_save(
    dit_path: str,
    dit_in_channels: int,
    lora_paths: List[str],
    multipliers: List[float],
    device_choice: str,
    output_path: str,
    overwrite_existing: bool,
    model_type: str,
) -> Tuple[str, str]:
    if not dit_path or not os.path.isfile(dit_path):
        return "error", f"Base DiT checkpoint not found: {dit_path}"

    if not lora_paths:
        return "error", "At least one LoRA weight file is required."

    missing = [path for path in lora_paths if not os.path.isfile(path)]
    if missing:
        return "error", f"LoRA file(s) not found: {', '.join(missing)}"

    if not output_path:
        return "error", "Output path must be specified."

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            return "error", f"Unable to create output directory '{output_dir}': {exc}"

    if os.path.exists(output_path) and not overwrite_existing:
        return "skip", f"Output already exists: {output_path}"

    device = _resolve_device(device_choice)
    transformer = None
    try:
        transformer = _load_base_model(dit_path, model_type, dit_in_channels, device)
        success, merge_message = _merge_loras_into_transformer(
            transformer,
            lora_paths,
            multipliers,
            device,
        )

        if not success:
            return "error", f"Merge failed: {merge_message}"

        mem_eff_save_file(transformer.state_dict(), output_path)
    except Exception as exc:
        log.exception("Failed to merge and save LoRA: %s", exc)
        return "error", f"Error during merge: {exc}"
    finally:
        if transformer is not None:
            del transformer
        _cleanup_memory()

    summary = (
        "✅ Merge complete!\n"
        f"- Base DiT: {dit_path}\n"
        f"- LoRAs: {', '.join(lora_paths)}\n"
        f"- Multipliers: {', '.join(f'{m:.3f}' for m in multipliers)}\n"
        f"- Device: {device}\n"
        f"- Output: {output_path}"
    )
    return "success", summary


def _merge_loras_only(
    lora_paths: List[str],
    multipliers: List[float],
    output_path: str,
    overwrite_existing: bool,
    target_dtype_choice: Optional[str],
) -> Tuple[str, str]:
    if len(lora_paths) < 2:
        return "error", "At least two LoRA files are required to create a merged LoRA."

    missing = [path for path in lora_paths if not os.path.isfile(path)]
    if missing:
        return "error", f"LoRA file(s) not found: {', '.join(missing)}"

    if not output_path:
        return "error", "Output path must be specified."

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as exc:
            return "error", f"Unable to create output directory '{output_dir}': {exc}"

    if os.path.exists(output_path) and not overwrite_existing:
        return "skip", f"Output already exists: {output_path}"

    accumulated: Dict[str, torch.Tensor] = {}
    dtype_map: Dict[str, torch.dtype] = {}
    target_dtype = _resolve_output_dtype(target_dtype_choice)
    merged_metadata: Optional[Dict[str, str]] = None
    merged_from: List[str] = []

    try:
        for lora_path, multiplier in zip(lora_paths, multipliers):
            log.info("Accumulating LoRA %s with multiplier %s", lora_path, multiplier)
            merged_from.append(os.path.basename(lora_path))

            if merged_metadata is None:
                try:
                    with safe_open(lora_path, framework="pt", device="cpu") as f:
                        file_metadata = f.metadata() or {}
                    merged_metadata = dict(file_metadata)
                except Exception as meta_exc:
                    log.warning("Unable to read metadata from '%s': %s", lora_path, meta_exc)
                    merged_metadata = {}

            weights_sd = load_file(lora_path)
            for key, tensor in weights_sd.items():
                original_dtype = tensor.dtype
                if key not in dtype_map:
                    dtype_map[key] = original_dtype
                elif dtype_map[key] != original_dtype:
                    log.warning(
                        "LoRA tensor '%s' has mixed dtypes (%s vs %s). Using float32 accumulation.",
                        key,
                        dtype_map[key],
                        original_dtype,
                    )
                tensor = tensor.to(torch.float32) * float(multiplier)
                if key in accumulated:
                    accumulated[key] += tensor
                else:
                    accumulated[key] = tensor.clone()
            del weights_sd
            _cleanup_memory()

        if not accumulated:
            return "error", "No tensors were found to merge. Verify the LoRA files contain trainable weights."

        merged_state: Dict[str, torch.Tensor] = {}
        for key, tensor in accumulated.items():
            desired_dtype = target_dtype or dtype_map.get(key) or torch.float32
            merged_state[key] = tensor.to(dtype=desired_dtype, device="cpu")

        if merged_metadata is None:
            merged_metadata = {}

        merged_metadata["merged_from"] = ", ".join(merged_from)
        if target_dtype_choice and target_dtype_choice.lower() != "auto":
            merged_metadata["merged_dtype"] = target_dtype_choice.upper()
        else:
            merged_metadata["merged_dtype"] = "AUTO"

        mem_eff_save_file(merged_state, output_path, metadata=merged_metadata)
    except Exception as exc:
        log.exception("Failed to merge LoRAs: %s", exc)
        return "error", f"Error while merging LoRAs: {exc}"
    finally:
        accumulated.clear()
        _cleanup_memory()

    dtype_label = (
        target_dtype_choice
        if target_dtype_choice and target_dtype_choice.lower() != "auto"
        else "Auto (match tensors)"
    )
    summary = (
        "✅ LoRA merge complete!\n"
        f"- LoRAs: {', '.join(lora_paths)}\n"
        f"- Multipliers: {', '.join(f'{m:.3f}' for m in multipliers)}\n"
        f"- Output dtype: {dtype_label}\n"
        f"- Output: {output_path}"
    )
    return "success", summary


def merge_single(payload: Dict[str, object]) -> Tuple[str, str]:
    merge_mode = payload.get("merge_mode", "lora_to_lora")
    lora_paths = [path for path in payload.get("lora_paths", []) if path]
    multipliers = [float(value) for value in payload.get("multipliers", [])]
    dit_path = payload.get("dit_path", "")
    output_path = payload.get("output_path", "")
    overwrite_existing = bool(payload.get("overwrite_existing", False))
    merged_lora_dtype_choice = payload.get("merged_lora_dtype_choice")

    if not lora_paths:
        return "error", "At least one LoRA file must be specified."

    if merge_mode == "lora_to_lora":
        return _merge_loras_only(
            lora_paths=lora_paths,
            multipliers=multipliers,
            output_path=output_path,
            overwrite_existing=overwrite_existing,
            target_dtype_choice=merged_lora_dtype_choice,
        )

    return _merge_and_save(
        dit_path=dit_path,
        dit_in_channels=int(payload.get("dit_in_channels", 16)),
        lora_paths=lora_paths,
        multipliers=multipliers,
        device_choice=str(payload.get("device_choice", "auto")),
        output_path=output_path,
        overwrite_existing=overwrite_existing,
        model_type=str(payload.get("model_type", "qwen")),
    )


def batch_merge(payload: Dict[str, object]) -> Tuple[str, str]:
    import glob

    dit_path = payload.get("dit_path", "")
    lora_folder = payload.get("lora_folder", "")
    output_folder = payload.get("output_folder", "")
    output_suffix = payload.get("output_suffix", "_merged")
    output_extension = payload.get("output_extension", ".safetensors")
    dit_in_channels = int(payload.get("dit_in_channels", 16))
    model_type = str(payload.get("model_type", "qwen"))
    device_choice = str(payload.get("device_choice", "auto"))
    multiplier = float(payload.get("multiplier", 1.0))
    recursive = bool(payload.get("recursive", True))
    overwrite_existing = bool(payload.get("overwrite_existing", False))
    lora_extensions = payload.get("lora_extensions", ".safetensors")

    if not dit_path or not os.path.isfile(dit_path):
        return "error", f"❌ Base DiT checkpoint not found: {dit_path}"

    if not lora_folder or not os.path.isdir(lora_folder):
        return "error", f"❌ LoRA folder not found: {lora_folder}"

    output_extension = output_extension.strip() or ".safetensors"
    if not output_extension.startswith("."):
        output_extension = "." + output_extension

    extensions = [
        ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
        for ext in lora_extensions.split(",")
        if ext.strip()
    ]
    if not extensions:
        extensions = [".safetensors"]

    search_pattern = "**/*" if recursive else "*"
    lora_files: List[str] = []
    for ext in extensions:
        pattern = os.path.join(lora_folder, f"{search_pattern}{ext}")
        lora_files.extend(glob.glob(pattern, recursive=recursive))

    lora_files = sorted(set(lora_files))

    if not lora_files:
        return "success", (
            f"⚠️ No LoRA files found in {lora_folder} with extensions {', '.join(extensions)}"
        )

    base_output_folder = output_folder.strip() or lora_folder

    successes = 0
    skips = 0
    failures = 0
    logs: List[str] = []

    for lora_path in lora_files:
        rel_path = os.path.relpath(lora_path, lora_folder)
        rel_dir = os.path.dirname(rel_path)
        base_name, _ = os.path.splitext(os.path.basename(lora_path))

        target_dir = (
            os.path.join(base_output_folder, rel_dir)
            if rel_dir not in ("", ".")
            else base_output_folder
        )
        os.makedirs(target_dir, exist_ok=True)

        output_name = f"{base_name}{output_suffix}{output_extension}"
        output_path = os.path.join(target_dir, output_name)

        status, message = _merge_and_save(
            dit_path=dit_path,
            dit_in_channels=dit_in_channels,
            lora_paths=[lora_path],
            multipliers=[multiplier],
            device_choice=device_choice,
            output_path=output_path,
            overwrite_existing=overwrite_existing,
            model_type=model_type,
        )

        if status == "success":
            successes += 1
            logs.append(f"✓ {rel_path} → {os.path.relpath(output_path, base_output_folder)}")
        elif status == "skip":
            skips += 1
            logs.append(f"⊘ {rel_path} (skipped: {message})")
        else:
            failures += 1
            logs.append(f"✗ {rel_path} (error: {message})")

    summary = (
        "Batch merge completed!\n"
        f"- Success: {successes}\n"
        f"- Skipped: {skips}\n"
        f"- Failed: {failures}\n\n"
        + "\n".join(logs)
    )
    return "success", summary


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_result(path: str, status: str, message: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"status": status, "message": message}, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA merge worker process")
    parser.add_argument("--mode", choices=["single", "batch"], required=True)
    parser.add_argument("--input", required=True, help="Path to JSON payload")
    parser.add_argument("--output", required=True, help="Path to JSON result file")
    args = parser.parse_args()

    try:
        payload = _read_json(args.input)
        if args.mode == "single":
            status, message = merge_single(payload)
        else:
            status, message = batch_merge(payload)
        _write_result(args.output, status, message)
        return 0 if status != "error" else 1
    except Exception as exc:
        log.exception("LoRA merge worker crashed: %s", exc)
        try:
            _write_result(args.output, "error", str(exc))
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

