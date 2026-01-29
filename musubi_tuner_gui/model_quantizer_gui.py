import gradio as gr
import os
import sys
import subprocess
import re
from typing import Dict, List, Optional, Tuple
import psutil
import toml

from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig
from .common_gui import (
    get_file_path,
    get_folder_path,
    get_saveasfilename_path,
    get_file_path_or_save_as,
    setup_environment,
    save_executed_script,
    generate_script_content,
)
from .custom_logging import setup_logging

log = setup_logging()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WORKFLOW_QUANTIZE = "Quantize (FP8/INT8/NVFP4/MXFP8)"
WORKFLOW_CONVERT_FP8 = "Convert FP8 scaled → comfy_quant"
WORKFLOW_CONVERT_INT8 = "Convert INT8 scaled → comfy_quant"
WORKFLOW_LEGACY_INPUT = "Legacy: Add input_scale"
WORKFLOW_CLEANUP_FP8 = "Legacy: Cleanup FP8 scaled"
WORKFLOW_ACTCAL = "Activation calibration (actcal)"
WORKFLOW_EDIT_QUANT = "Edit comfy_quant metadata"
WORKFLOW_HYBRID_MXFP8 = "Hybrid MXFP8 (tensorwise fallback)"
WORKFLOW_DRY_RUN = "Dry Run (Analyze / Create Template)"

QUANT_FORMAT_FP8 = "FP8 (E4M3)"
QUANT_FORMAT_INT8 = "INT8"
QUANT_FORMAT_NVFP4 = "NVFP4 (FP4 E2M1)"
QUANT_FORMAT_MXFP8 = "MXFP8 (Microscaling)"

PRESET_CUSTOM = "Custom (manual)"
PRESET_FAST = "Fast (Simple Quantization)"
PRESET_NORMAL = "Normal (Balanced)"
PRESET_HIGH = "High Quality (Slow)"
PRESET_INT8_FAST = "INT8 Fast (Block 128)"
PRESET_INT8_TENSOR = "INT8 Tensorwise (Fastest)"
PRESET_MXFP8_BALANCED = "MXFP8 Balanced (Blackwell)"
PRESET_NVFP4_BALANCED = "NVFP4 Balanced (Blackwell)"
PRESET_NVFP4_Z = "NVFP4 Z-Image (Best Quality)"

MODEL_PRESET_NONE = "None (manual)"

OUTPUT_MODE_FULL = "Full (all logs)"
OUTPUT_MODE_COMPACT = "Compact (hide progress bars)"
OUTPUT_MODE_SUMMARY = "Summary only (warnings/errors)"
OUTPUT_MODE_CHOICES = [OUTPUT_MODE_FULL, OUTPUT_MODE_COMPACT, OUTPUT_MODE_SUMMARY]


def _load_model_filters() -> Dict[str, Dict[str, object]]:
    try:
        from convert_to_quant.constants import MODEL_FILTERS

        return MODEL_FILTERS
    except Exception as exc:
        log.warning(f"Could not import convert_to_quant MODEL_FILTERS: {exc}")
        return {
            "t5xxl": {"help": "T5-XXL text encoder", "category": "text"},
            "mistral": {"help": "Mistral text encoder", "category": "text"},
            "visual": {"help": "Visual encoder", "category": "text"},
            "flux2": {"help": "Flux.2 diffusion", "category": "diffusion"},
            "distillation_large": {"help": "Chroma distilled (large)", "category": "diffusion"},
            "distillation_small": {"help": "Chroma distilled (small)", "category": "diffusion"},
            "nerf_large": {"help": "NeRF (large)", "category": "diffusion"},
            "nerf_small": {"help": "NeRF (small)", "category": "diffusion"},
            "radiance": {"help": "Radiance diffusion", "category": "diffusion"},
            "wan": {"help": "WAN video model", "category": "video"},
            "hunyuan": {"help": "Hunyuan video model", "category": "video"},
            "qwen": {"help": "Qwen Image", "category": "image"},
            "zimage": {"help": "Z-Image", "category": "image"},
            "zimage_refiner": {"help": "Z-Image Refiner", "category": "image"},
        }


MODEL_FILTERS = _load_model_filters()
MODEL_CATEGORY_LABELS = {
    "text": "Text Encoders",
    "diffusion": "Diffusion Models",
    "video": "Video Models",
    "image": "Image Models",
}


def _ordered_filter_names() -> List[str]:
    def _sort_key(item):
        name, cfg = item
        cat = cfg.get("category", "")
        return (cat, name)

    return [name for name, _ in sorted(MODEL_FILTERS.items(), key=_sort_key)]


MODEL_PRESET_CHOICES = [MODEL_PRESET_NONE] + _ordered_filter_names()

MODEL_PRESET_OVERRIDES = {
    "t5xxl": {"include_input_scale": True},
}

MODEL_PRESET_SETTINGS = {
    name: {
        "preset": PRESET_NORMAL,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    }
    for name in MODEL_FILTERS.keys()
}
MODEL_PRESET_SETTINGS.update({
    "t5xxl": {
        "preset": PRESET_NORMAL,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
        "include_input_scale": True,
    },
    "zimage": {
        "preset": PRESET_NVFP4_Z,
        "quant_format": QUANT_FORMAT_NVFP4,
        "scaling_mode": "tensor",
    },
    "zimage_refiner": {
        "preset": PRESET_NVFP4_BALANCED,
        "quant_format": QUANT_FORMAT_NVFP4,
        "scaling_mode": "tensor",
    },
})

PRESET_OVERRIDES = {
    PRESET_FAST: {
        "simple": True,
        "skip_inefficient_layers": True,
        "num_iter": 200,
        "calib_samples": 1024,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.02,
        "min_k": 16,
        "max_k": 64,
        "full_matrix": False,
        "full_precision_matrix_mult": False,
    },
    PRESET_NORMAL: {
        "simple": False,
        "skip_inefficient_layers": True,
        "num_iter": 400,
        "calib_samples": 1024,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.02,
        "min_k": 16,
        "max_k": 64,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_HIGH: {
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 3000,
        "calib_samples": 4096,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.1,
        "min_k": 64,
        "max_k": 256,
        "full_matrix": True,
        "full_precision_matrix_mult": True,
    },
    PRESET_INT8_FAST: {
        "quant_format": QUANT_FORMAT_INT8,
        "comfy_quant": True,
        "scaling_mode": "block",
        "block_size": 128,
        "simple": True,
        "skip_inefficient_layers": True,
        "num_iter": 200,
        "calib_samples": 1024,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.02,
        "min_k": 16,
        "max_k": 64,
        "full_matrix": False,
        "full_precision_matrix_mult": False,
    },
    PRESET_INT8_TENSOR: {
        "quant_format": QUANT_FORMAT_INT8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": True,
        "skip_inefficient_layers": True,
        "num_iter": 120,
        "calib_samples": 512,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.02,
        "min_k": 16,
        "max_k": 64,
        "full_matrix": False,
        "full_precision_matrix_mult": False,
    },
    PRESET_MXFP8_BALANCED: {
        "quant_format": QUANT_FORMAT_MXFP8,
        "comfy_quant": True,
        "simple": False,
        "skip_inefficient_layers": True,
        "num_iter": 800,
        "calib_samples": 2048,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.2,
        "min_k": 64,
        "max_k": 1024,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_NVFP4_BALANCED: {
        "quant_format": QUANT_FORMAT_NVFP4,
        "comfy_quant": True,
        "simple": False,
        "skip_inefficient_layers": True,
        "num_iter": 5000,
        "calib_samples": 4096,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 8.077300000003e-3,
        "top_p": 0.2,
        "min_k": 64,
        "max_k": 1024,
        "scale_optimization": "iterative",
        "scale_refinement_rounds": 1,
        "full_precision_matrix_mult": True,
    },
    PRESET_NVFP4_Z: {
        "quant_format": QUANT_FORMAT_NVFP4,
        "comfy_quant": True,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 90000,
        "calib_samples": 8192,
        "optimizer": "original",
        "lr_schedule": "adaptive",
        "lr": 1.60773,
        "top_p": 0.2,
        "min_k": 32,
        "max_k": 2048,
        "scale_optimization": "iterative",
        "scale_refinement_rounds": 1,
        "manual_seed": 42,
        "verbose": "NORMAL",
    },
}


def _to_int(value, default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _flatten_dict(data: Dict[str, object], prefix: str = "") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in data.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, new_key))
        else:
            flat[new_key] = value
    return flat


class ModelQuantizer:
    def __init__(self, headless: bool, config: Optional[GUIConfig]) -> None:
        self.headless = headless
        self.config = config
        self.single_process: Optional[subprocess.Popen] = None
        self.batch_process: Optional[subprocess.Popen] = None
        self.single_cancel_requested = False
        self.batch_cancel_requested = False

    def _resolve_python(self) -> List[str]:
        venv_python = os.path.join(REPO_ROOT, "venv", "Scripts", "python.exe")
        if os.path.isfile(venv_python):
            return [venv_python]
        return [sys.executable]

    def _base_cmd(self) -> List[str]:
        return self._resolve_python() + ["-m", "convert_to_quant.cli.main"]

    def _tail_text(self, text: str, max_chars: int = 40000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:2000] + "\n\n... (output truncated) ...\n\n" + text[-max_chars:]

    def _is_progress_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        progress_tokens = ("it/s", "s/it", "%|", "|#", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█")
        if any(token in stripped for token in progress_tokens):
            return True
        if "Optimizing" in stripped and ("%" in stripped or "|" in stripped):
            return True
        if "Loading tensors" in stripped and ("%" in stripped or "|" in stripped):
            return True
        if "Processing layers" in stripped and ("%" in stripped or "|" in stripped):
            return True
        return False

    def _progress_summary(self, line: str) -> Optional[str]:
        if ":" not in line:
            return None
        summary = line.split(":", 1)[0].strip()
        if not summary:
            return None
        return f"{summary}..."

    def _compact_progress_line(self, line: str) -> str:
        stripped = line.strip()
        match = re.match(r"^(.*?):\s*([0-9]+%)\|.*?\|\s*([0-9]+/[0-9]+)", stripped)
        if match:
            desc = match.group(1).strip()
            percent = match.group(2)
            count = match.group(3)
            return f"{desc}: {percent} ({count})"
        return stripped

    def _is_summary_line(self, line: str) -> bool:
        lowered = line.lower()
        keywords = (
            "error",
            "fatal",
            "warning",
            "failed",
            "complete",
            "completed",
            "summary",
            "saved",
            "output:",
            "cancelled",
            "canceled",
            "done:",
            "skipped",
        )
        return any(keyword in lowered for keyword in keywords)

    def _is_running(self, process: Optional[subprocess.Popen]) -> bool:
        return process is not None and process.poll() is None

    def _terminate_process(self, process: Optional[subprocess.Popen]) -> bool:
        if not process or process.poll() is not None:
            return False
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            parent.kill()
            return True
        except psutil.Error:
            return False

    def _run_command(
        self,
        cmd: List[str],
        process_attr: str,
        label: str,
        output_mode: str = OUTPUT_MODE_FULL,
    ) -> Tuple[str, int]:
        log.info("Executing: %s", " ".join(cmd))
        script_content = generate_script_content(cmd, label)
        save_executed_script(script_content=script_content, config_name=None, script_type="model_quantizer")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=setup_environment(),
            cwd=REPO_ROOT,
        )
        setattr(self, process_attr, process)
        output_lines: List[str] = []
        progress_summaries: set[str] = set()
        progress_active = False
        progress_len = 0
        try:
            if process.stdout:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if self._is_progress_line(line):
                        if output_mode != OUTPUT_MODE_SUMMARY:
                            compact_line = self._compact_progress_line(line)
                            if compact_line:
                                pad = ""
                                if progress_len > len(compact_line):
                                    pad = " " * (progress_len - len(compact_line))
                                sys.stdout.write("\r" + compact_line + pad)
                                sys.stdout.flush()
                                progress_active = True
                                progress_len = len(compact_line)

                        if output_mode == OUTPUT_MODE_FULL:
                            output_lines.append(line)
                        elif output_mode == OUTPUT_MODE_COMPACT:
                            summary = self._progress_summary(line)
                            if summary and summary not in progress_summaries:
                                progress_summaries.add(summary)
                                output_lines.append(summary)
                        continue

                    if progress_active:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        progress_active = False
                        progress_len = 0

                    if output_mode == OUTPUT_MODE_SUMMARY and not self._is_summary_line(line):
                        continue

                    output_lines.append(line)
                    if line:
                        log.info("[Model Quantizer] %s", line)
            process.wait()
            if progress_active:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return "\n".join(output_lines), int(process.returncode or 0)
        finally:
            setattr(self, process_attr, None)

    def cancel_single(self) -> str:
        if not self._is_running(self.single_process):
            return "No single-file conversion is running."
        self.single_cancel_requested = True
        if self._terminate_process(self.single_process):
            return "Cancellation requested. Stopping conversion..."
        return "Unable to cancel conversion. It may have already finished."

    def cancel_batch(self) -> str:
        if not self._is_running(self.batch_process):
            return "No batch conversion is running."
        self.batch_cancel_requested = True
        if self._terminate_process(self.batch_process):
            return "Cancellation requested. Stopping batch conversion..."
        return "Unable to cancel batch conversion. It may have already finished."

    def _build_command(
        self,
        input_path: str,
        output_path: Optional[str],
        params: Dict[str, object],
    ) -> List[str]:
        cmd = self._base_cmd()
        cmd += ["-i", input_path]
        if output_path:
            cmd += ["-o", output_path]

        if params.get("verbose"):
            cmd += ["--verbose", str(params.get("verbose"))]

        workflow = params.get("workflow")
        quant_format = params.get("quant_format")

        if workflow == WORKFLOW_CONVERT_FP8:
            cmd.append("--convert-fp8-scaled")
            if params.get("hp_filter"):
                cmd += ["--hp-filter", str(params.get("hp_filter"))]
            if params.get("full_precision_mm"):
                cmd.append("--full-precision-mm")
            if params.get("include_input_scale"):
                cmd.append("--input_scale")
            return cmd

        if workflow == WORKFLOW_CONVERT_INT8:
            cmd.append("--convert-int8-scaled")
            if params.get("block_size") is not None:
                cmd += ["--block_size", str(params.get("block_size"))]
            if params.get("include_input_scale"):
                cmd.append("--input_scale")
            if params.get("save_quant_metadata"):
                cmd.append("--save-quant-metadata")
            return cmd

        if workflow == WORKFLOW_LEGACY_INPUT:
            cmd.append("--legacy_input_add")
            return cmd

        if workflow == WORKFLOW_CLEANUP_FP8:
            cmd.append("--cleanup-fp8-scaled")
            if params.get("scaled_fp8_marker") is not None:
                cmd += ["--scaled-fp8-marker", str(params.get("scaled_fp8_marker"))]
            if params.get("include_input_scale"):
                cmd.append("--input_scale")
            return cmd

        if workflow == WORKFLOW_ACTCAL:
            cmd.append("--actcal")
            if params.get("actcal_samples") is not None:
                cmd += ["--actcal-samples", str(params.get("actcal_samples"))]
            if params.get("actcal_percentile") is not None:
                cmd += ["--actcal-percentile", str(params.get("actcal_percentile"))]
            if params.get("actcal_lora"):
                cmd += ["--actcal-lora", str(params.get("actcal_lora"))]
            if params.get("actcal_seed") is not None:
                cmd += ["--actcal-seed", str(params.get("actcal_seed"))]
            if params.get("actcal_device"):
                cmd += ["--actcal-device", str(params.get("actcal_device"))]
            return cmd

        if workflow == WORKFLOW_EDIT_QUANT:
            cmd.append("--edit-quant")
            if params.get("remove_keys"):
                cmd += ["--remove-keys", str(params.get("remove_keys"))]
            if params.get("add_keys"):
                cmd += ["--add-keys", str(params.get("add_keys"))]
            if params.get("quant_filter"):
                cmd += ["--quant-filter", str(params.get("quant_filter"))]
            if params.get("save_quant_metadata"):
                cmd.append("--save-quant-metadata")
            return cmd

        if workflow == WORKFLOW_HYBRID_MXFP8:
            cmd.append("--make-hybrid-mxfp8")
            if params.get("tensor_scales_path"):
                cmd += ["--tensor-scales", str(params.get("tensor_scales_path"))]
            return cmd

        if workflow == WORKFLOW_DRY_RUN:
            dry_run = params.get("dry_run") or "analyze"
            cmd += ["--dry-run", str(dry_run)]

        if params.get("comfy_quant"):
            cmd.append("--comfy_quant")

        if quant_format == QUANT_FORMAT_INT8:
            cmd.append("--int8")
        elif quant_format == QUANT_FORMAT_NVFP4:
            cmd.append("--nvfp4")
        elif quant_format == QUANT_FORMAT_MXFP8:
            cmd.append("--mxfp8")

        if params.get("simple"):
            cmd.append("--simple")
        if params.get("skip_inefficient_layers"):
            cmd.append("--heur")
        if params.get("full_precision_matrix_mult"):
            cmd.append("--full_precision_matrix_mult")
        if params.get("include_input_scale"):
            cmd.append("--input_scale")

        if params.get("scaling_mode"):
            cmd += ["--scaling_mode", str(params.get("scaling_mode"))]
        if params.get("block_size") is not None:
            cmd += ["--block_size", str(params.get("block_size"))]

        if params.get("custom_layers"):
            cmd += ["--custom-layers", str(params.get("custom_layers"))]
        if params.get("exclude_layers"):
            cmd += ["--exclude-layers", str(params.get("exclude_layers"))]
        if params.get("custom_type"):
            cmd += ["--custom-type", str(params.get("custom_type"))]
        if params.get("custom_block_size") is not None:
            cmd += ["--custom-block-size", str(params.get("custom_block_size"))]
        if params.get("custom_scaling_mode"):
            cmd += ["--custom-scaling-mode", str(params.get("custom_scaling_mode"))]
        if params.get("custom_simple"):
            cmd.append("--custom-simple")
        if params.get("custom_heur"):
            cmd.append("--custom-heur")

        if params.get("fallback_type"):
            cmd += ["--fallback", str(params.get("fallback_type"))]
        if params.get("fallback_block_size") is not None:
            cmd += ["--fallback-block-size", str(params.get("fallback_block_size"))]
        if params.get("fallback_simple"):
            cmd.append("--fallback-simple")

        if params.get("full_matrix"):
            cmd.append("--full_matrix")

        if params.get("calib_samples") is not None:
            cmd += ["--calib_samples", str(params.get("calib_samples"))]
        if params.get("manual_seed") is not None:
            cmd += ["--manual_seed", str(params.get("manual_seed"))]
        if params.get("optimizer"):
            cmd += ["--optimizer", str(params.get("optimizer"))]
        if params.get("num_iter") is not None:
            cmd += ["--num_iter", str(params.get("num_iter"))]
        if params.get("lr") is not None:
            cmd += ["--lr", str(params.get("lr"))]
        if params.get("lr_schedule"):
            cmd += ["--lr_schedule", str(params.get("lr_schedule"))]
        if params.get("lr_gamma") is not None:
            cmd += ["--lr_gamma", str(params.get("lr_gamma"))]
        if params.get("lr_patience") is not None:
            cmd += ["--lr_patience", str(params.get("lr_patience"))]
        if params.get("lr_factor") is not None:
            cmd += ["--lr_factor", str(params.get("lr_factor"))]
        if params.get("lr_min") is not None:
            cmd += ["--lr_min", str(params.get("lr_min"))]
        if params.get("lr_cooldown") is not None:
            cmd += ["--lr_cooldown", str(params.get("lr_cooldown"))]
        if params.get("lr_threshold") is not None:
            cmd += ["--lr_threshold", str(params.get("lr_threshold"))]
        if params.get("lr_adaptive_mode"):
            cmd += ["--lr_adaptive_mode", str(params.get("lr_adaptive_mode"))]
        if params.get("lr_shape_influence") is not None:
            cmd += ["--lr-shape-influence", str(params.get("lr_shape_influence"))]
        if params.get("lr_threshold_mode"):
            cmd += ["--lr-threshold-mode", str(params.get("lr_threshold_mode"))]

        if params.get("early_stop_loss") is not None:
            cmd += ["--early-stop-loss", str(params.get("early_stop_loss"))]
        if params.get("early_stop_lr") is not None:
            cmd += ["--early-stop-lr", str(params.get("early_stop_lr"))]
        if params.get("early_stop_stall") is not None:
            cmd += ["--early-stop-stall", str(params.get("early_stop_stall"))]

        if params.get("scale_refinement_rounds") is not None:
            cmd += ["--scale-refinement", str(params.get("scale_refinement_rounds"))]
        if params.get("scale_optimization"):
            cmd += ["--scale-optimization", str(params.get("scale_optimization"))]

        if params.get("top_p") is not None:
            cmd += ["--top_p", str(params.get("top_p"))]
        if params.get("min_k") is not None:
            cmd += ["--min_k", str(params.get("min_k"))]
        if params.get("max_k") is not None:
            cmd += ["--max_k", str(params.get("max_k"))]

        if params.get("save_quant_metadata"):
            cmd.append("--save-quant-metadata")
        if params.get("no_normalize_scales"):
            cmd.append("--no-normalize-scales")

        if params.get("input_scales_path"):
            cmd += ["--input-scales", str(params.get("input_scales_path"))]

        if params.get("layer_config_path"):
            cmd += ["--layer-config", str(params.get("layer_config_path"))]
            if params.get("layer_config_fullmatch"):
                cmd.append("--fullmatch")

        if params.get("verbose_pinned"):
            cmd.append("--verbose-pinned")
        if params.get("low_memory"):
            cmd.append("--low-memory")

        for name, enabled in params.get("model_filters", {}).items():
            if enabled:
                cmd.append(f"--{name}")

        return cmd

    def _default_output_name(self, input_path: str, params: Dict[str, object]) -> str:
        base, _ = os.path.splitext(input_path)
        workflow = params.get("workflow")
        quant_format = params.get("quant_format")
        simple = bool(params.get("simple"))
        scaling_mode = params.get("scaling_mode") or "tensor"
        block_size = params.get("block_size")
        has_filters = any(params.get("model_filters", {}).values())
        has_custom = bool(params.get("custom_layers"))
        mixed_suffix = "mixed" if (has_filters or has_custom) else ""

        if workflow == WORKFLOW_CONVERT_FP8:
            return f"{base}_fp8mixed.safetensors"
        if workflow == WORKFLOW_CONVERT_INT8:
            return f"{base}_int8_comfy.safetensors"
        if workflow == WORKFLOW_LEGACY_INPUT:
            return f"{base}_with_input_scale.safetensors"
        if workflow == WORKFLOW_CLEANUP_FP8:
            return f"{base}_cleaned.safetensors"
        if workflow == WORKFLOW_ACTCAL:
            return f"{base}_calibrated.safetensors"
        if workflow == WORKFLOW_EDIT_QUANT:
            return f"{base}_edited.safetensors"
        if workflow == WORKFLOW_HYBRID_MXFP8:
            return f"{base}_hybrid.safetensors"

        prefix = "simple_" if simple else "learned_"

        if quant_format == QUANT_FORMAT_NVFP4 and (params.get("custom_type") or params.get("fallback_type")):
            format_str = "fp8"
            scaling_str = f"_{scaling_mode}"
        elif quant_format == QUANT_FORMAT_MXFP8 and (params.get("custom_type") or params.get("fallback_type")):
            format_str = "fp8"
            scaling_str = f"_{scaling_mode}"
        elif quant_format == QUANT_FORMAT_NVFP4:
            format_str = "nvfp4"
            scaling_str = ""
        elif quant_format == QUANT_FORMAT_MXFP8:
            format_str = "mxfp8"
            scaling_str = ""
        elif quant_format == QUANT_FORMAT_INT8:
            format_str = "int8"
            if scaling_mode == "tensor":
                scaling_str = "_tensorwise"
            else:
                scaling_str = f"_bs{block_size or 128}"
        else:
            format_str = "fp8"
            scaling_str = f"_{scaling_mode}"

        return f"{base}_{prefix}{format_str}{mixed_suffix}{scaling_str}.safetensors"

    def run_single(
        self,
        input_file: str,
        output_file: str,
        delete_original: bool,
        params: Dict[str, object],
    ) -> str:
        if self._is_running(self.single_process):
            return "A single-file conversion is already running. Cancel it before starting another."
        if not input_file:
            return "Input file is required."
        if not os.path.isfile(input_file):
            return f"Input file not found: {input_file}"

        output_path = output_file or self._default_output_name(input_file, params)
        if output_path and os.path.abspath(output_path) == os.path.abspath(input_file):
            return "Output file cannot be the same as input file."

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cmd = self._build_command(input_file, output_path, params)
        self.single_cancel_requested = False
        output_mode = params.get("output_mode", OUTPUT_MODE_FULL)
        output_text, return_code = self._run_command(
            cmd,
            "single_process",
            "Model quantizer",
            output_mode=output_mode,
        )
        was_cancelled = self.single_cancel_requested
        self.single_cancel_requested = False

        if was_cancelled:
            return "Conversion cancelled."
        if return_code != 0:
            return self._tail_text(f"Conversion failed (exit code {return_code}).\n\n{output_text}")

        if delete_original and os.path.isfile(output_path):
            try:
                os.remove(input_file)
            except Exception as exc:
                return self._tail_text(f"Conversion completed but could not delete input: {exc}\n\n{output_text}")

        return self._tail_text(f"Conversion completed successfully.\nOutput: {output_path}\n\n{output_text}")

    def run_batch(
        self,
        input_folder: str,
        output_folder: str,
        extensions: str,
        recursive: bool,
        overwrite_existing: bool,
        delete_original: bool,
        params: Dict[str, object],
    ) -> str:
        if self._is_running(self.batch_process):
            return "A batch conversion is already running. Cancel it before starting another."
        if not input_folder:
            return "Input folder is required."
        if not os.path.isdir(input_folder):
            return f"Input folder not found: {input_folder}"

        ext_list = [ext.strip().lower() for ext in (extensions or ".safetensors").split(",") if ext.strip()]
        if not ext_list:
            ext_list = [".safetensors"]

        files: List[str] = []
        if recursive:
            for root, _, filenames in os.walk(input_folder):
                for fname in filenames:
                    if os.path.splitext(fname)[1].lower() in ext_list:
                        files.append(os.path.join(root, fname))
        else:
            for fname in os.listdir(input_folder):
                full = os.path.join(input_folder, fname)
                if os.path.isfile(full) and os.path.splitext(fname)[1].lower() in ext_list:
                    files.append(full)

        if not files:
            return "No matching files found for batch conversion."

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)

        log_lines: List[str] = []
        self.batch_cancel_requested = False
        for idx, file_path in enumerate(sorted(files), start=1):
            if self.batch_cancel_requested:
                log_lines.append("Batch conversion cancelled by user.")
                break

            output_path = self._default_output_name(file_path, params)
            if output_folder:
                output_path = os.path.join(output_folder, os.path.basename(output_path))

            if os.path.isfile(output_path) and not overwrite_existing:
                log_lines.append(f"[{idx}/{len(files)}] Skipped (output exists): {output_path}")
                continue

            if output_path and os.path.abspath(output_path) == os.path.abspath(file_path):
                log_lines.append(f"[{idx}/{len(files)}] Skipped (output matches input): {file_path}")
                continue

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            log_lines.append(f"[{idx}/{len(files)}] Converting: {file_path}")
            cmd = self._build_command(file_path, output_path, params)
            output_mode = params.get("output_mode", OUTPUT_MODE_FULL)
            output_text, return_code = self._run_command(
                cmd,
                "batch_process",
                "Model quantizer batch",
                output_mode=output_mode,
            )
            if self.batch_cancel_requested:
                log_lines.append("Batch conversion cancelled by user.")
                break
            if return_code != 0:
                log_lines.append(f"[{idx}/{len(files)}] Failed (exit code {return_code})")
                log_lines.append(self._tail_text(output_text))
                continue

            if delete_original and os.path.isfile(output_path):
                try:
                    os.remove(file_path)
                except Exception as exc:
                    log_lines.append(f"[{idx}/{len(files)}] Converted but could not delete input: {exc}")
            log_lines.append(f"[{idx}/{len(files)}] Done: {output_path}")

        return self._tail_text("\n".join(log_lines))


def model_quantizer_tab_legacy(headless: bool, config: GUIConfig) -> None:
    quantizer = ModelQuantizer(headless=headless, config=config)

    gr.Markdown("# Model Quantizer")
    gr.Markdown(
        "Quantize safetensors checkpoints with **convert_to_quant**. Supports FP8, INT8, NVFP4, MXFP8, "
        "metadata conversions, calibration, and batch processing."
    )

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    with gr.Accordion("Configuration file Settings", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    with gr.Accordion("Presets", open=True):
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                label="Quality Preset",
                choices=[
                    PRESET_CUSTOM,
                    PRESET_FAST,
                    PRESET_NORMAL,
                    PRESET_HIGH,
                    PRESET_INT8_FAST,
                    PRESET_INT8_TENSOR,
                    PRESET_MXFP8_BALANCED,
                    PRESET_NVFP4_BALANCED,
                    PRESET_NVFP4_Z,
                ],
                value=config.get("model_quantizer.preset", PRESET_CUSTOM),
                info="Quickly apply recommended optimization settings.",
            )
            model_preset_dropdown = gr.Dropdown(
                label="Model Preset (Recommended Filters)",
                choices=MODEL_PRESET_CHOICES,
                value=config.get("model_quantizer.model_preset", MODEL_PRESET_NONE),
                info="Select a model to apply its recommended exclusion filters.",
            )

    with gr.Accordion("Workflow", open=True):
        workflow = gr.Dropdown(
            label="Workflow",
            choices=[
                WORKFLOW_QUANTIZE,
                WORKFLOW_CONVERT_FP8,
                WORKFLOW_CONVERT_INT8,
                WORKFLOW_LEGACY_INPUT,
                WORKFLOW_CLEANUP_FP8,
                WORKFLOW_ACTCAL,
                WORKFLOW_EDIT_QUANT,
                WORKFLOW_HYBRID_MXFP8,
                WORKFLOW_DRY_RUN,
            ],
            value=config.get("model_quantizer.workflow", WORKFLOW_QUANTIZE),
        )

    with gr.Accordion("Quantization Format", open=True) as quant_format_group:
        with gr.Row():
            quant_format = gr.Dropdown(
                label="Primary Format",
                choices=[QUANT_FORMAT_FP8, QUANT_FORMAT_INT8, QUANT_FORMAT_NVFP4, QUANT_FORMAT_MXFP8],
                value=config.get("model_quantizer.quant_format", QUANT_FORMAT_FP8),
            )
            comfy_quant = gr.Checkbox(
                label="Write ComfyUI Metadata (.comfy_quant)",
                value=config.get("model_quantizer.comfy_quant", True),
            )
            full_precision_matrix_mult = gr.Checkbox(
                label="comfy_quant: Full precision matrix mult",
                value=config.get("model_quantizer.full_precision_matrix_mult", False),
            )

        with gr.Row():
            scaling_mode = gr.Dropdown(
                label="Scaling Mode",
                choices=["tensor", "row", "block", "block3d", "block2d"],
                value=config.get("model_quantizer.scaling_mode", "tensor"),
            )
            block_size = gr.Number(
                label="Block Size",
                value=config.get("model_quantizer.block_size", 64),
                step=1,
                interactive=True,
            )
            include_input_scale = gr.Checkbox(
                label="Include input_scale tensors",
                value=config.get("model_quantizer.include_input_scale", False),
            )

    with gr.Accordion("Model Filters", open=False) as model_filter_group:
        filter_checkboxes: Dict[str, gr.Checkbox] = {}
        for category_key, category_label in MODEL_CATEGORY_LABELS.items():
            filters_in_category = [
                (name, cfg) for name, cfg in MODEL_FILTERS.items()
                if cfg.get("category") == category_key
            ]
            if not filters_in_category:
                continue
            with gr.Group():
                gr.Markdown(f"### {category_label}")
                for name, cfg in sorted(filters_in_category, key=lambda item: item[0]):
                    label = f"{name}: {cfg.get('help', '')}".strip(": ")
                    filter_checkboxes[name] = gr.Checkbox(
                        label=label,
                        value=bool(config.get(f"model_quantizer.filter.{name}", False)),
                    )

    with gr.Accordion("Layer Mixing & Exclusions", open=False) as layer_mixing_group:
        with gr.Row():
            custom_layers = gr.Textbox(
                label="Custom Layers (Regex)",
                placeholder="e.g. attn|mlp",
                value=config.get("model_quantizer.custom_layers", ""),
            )
            exclude_layers = gr.Textbox(
                label="Exclude Layers (Regex)",
                placeholder="e.g. norm|bias",
                value=config.get("model_quantizer.exclude_layers", ""),
            )
        with gr.Row():
            custom_type = gr.Dropdown(
                label="Custom Type",
                choices=[None, "fp8", "int8", "mxfp8", "nvfp4"],
                value=config.get("model_quantizer.custom_type", None),
                allow_custom_value=False,
            )
            custom_block_size = gr.Number(
                label="Custom Block Size",
                value=config.get("model_quantizer.custom_block_size", None),
                step=1,
            )
            custom_scaling_mode = gr.Dropdown(
                label="Custom Scaling Mode",
                choices=[None, "tensor", "row", "block", "block3d", "block2d"],
                value=config.get("model_quantizer.custom_scaling_mode", None),
            )
        with gr.Row():
            custom_simple = gr.Checkbox(
                label="Custom: Simple quantization",
                value=config.get("model_quantizer.custom_simple", False),
            )
            custom_heur = gr.Checkbox(
                label="Custom: Heuristics",
                value=config.get("model_quantizer.custom_heur", False),
            )

        with gr.Row():
            fallback_type = gr.Dropdown(
                label="Fallback Type",
                choices=[None, "fp8", "int8", "mxfp8", "nvfp4"],
                value=config.get("model_quantizer.fallback_type", None),
            )
            fallback_block_size = gr.Number(
                label="Fallback Block Size",
                value=config.get("model_quantizer.fallback_block_size", None),
                step=1,
            )
            fallback_simple = gr.Checkbox(
                label="Fallback: Simple quantization",
                value=config.get("model_quantizer.fallback_simple", False),
            )

    with gr.Accordion("Optimization & Quality", open=False) as optimization_group:
        with gr.Row():
            simple = gr.Checkbox(
                label="Simple quantization (skip SVD)",
                value=config.get("model_quantizer.simple", False),
            )
            skip_inefficient_layers = gr.Checkbox(
                label="Skip inefficient layers (heuristics)",
                value=config.get("model_quantizer.skip_inefficient_layers", False),
            )
            full_matrix = gr.Checkbox(
                label="Use full SVD matrix",
                value=config.get("model_quantizer.full_matrix", False),
            )
        with gr.Row():
            calib_samples = gr.Number(
                label="Calibration Samples",
                value=config.get("model_quantizer.calib_samples", 6144),
                step=1,
            )
            manual_seed = gr.Number(
                label="Manual Seed (-1=random)",
                value=config.get("model_quantizer.manual_seed", -1),
                step=1,
            )
            optimizer = gr.Dropdown(
                label="Optimizer",
                choices=["original", "adamw", "radam"],
                value=config.get("model_quantizer.optimizer", "original"),
            )
        with gr.Row():
            num_iter = gr.Number(
                label="Iterations",
                value=config.get("model_quantizer.num_iter", 1000),
                step=1,
            )
            lr = gr.Number(
                label="Learning Rate",
                value=config.get("model_quantizer.lr", 8.077300000003e-3),
            )
            lr_schedule = gr.Dropdown(
                label="LR Schedule",
                choices=["adaptive", "exponential", "plateau"],
                value=config.get("model_quantizer.lr_schedule", "adaptive"),
            )
        with gr.Row():
            top_p = gr.Number(
                label="Top P",
                value=config.get("model_quantizer.top_p", 0.2),
            )
            min_k = gr.Number(
                label="Min K",
                value=config.get("model_quantizer.min_k", 64),
                step=1,
            )
            max_k = gr.Number(
                label="Max K",
                value=config.get("model_quantizer.max_k", 1024),
                step=1,
            )

    with gr.Accordion("Advanced LR & Early Stopping", open=False) as advanced_lr_group:
        with gr.Row():
            lr_gamma = gr.Number(
                label="LR Gamma",
                value=config.get("model_quantizer.lr_gamma", 0.99),
            )
            lr_patience = gr.Number(
                label="LR Patience",
                value=config.get("model_quantizer.lr_patience", 9),
                step=1,
            )
            lr_factor = gr.Number(
                label="LR Factor",
                value=config.get("model_quantizer.lr_factor", 0.95),
            )
        with gr.Row():
            lr_min = gr.Number(
                label="LR Min",
                value=config.get("model_quantizer.lr_min", 1e-10),
            )
            lr_cooldown = gr.Number(
                label="LR Cooldown",
                value=config.get("model_quantizer.lr_cooldown", 6),
                step=1,
            )
            lr_threshold = gr.Number(
                label="LR Threshold",
                value=config.get("model_quantizer.lr_threshold", 0.0),
            )
        with gr.Row():
            lr_adaptive_mode = gr.Dropdown(
                label="LR Adaptive Mode",
                choices=["simple-reset", "no-reset"],
                value=config.get("model_quantizer.lr_adaptive_mode", "simple-reset"),
            )
            lr_shape_influence = gr.Number(
                label="LR Shape Influence",
                value=config.get("model_quantizer.lr_shape_influence", 1.0),
            )
            lr_threshold_mode = gr.Dropdown(
                label="LR Threshold Mode",
                choices=["rel", "abs"],
                value=config.get("model_quantizer.lr_threshold_mode", "rel"),
            )
        with gr.Row():
            early_stop_loss = gr.Number(
                label="Early Stop Loss",
                value=config.get("model_quantizer.early_stop_loss", 1e-8),
            )
            early_stop_lr = gr.Number(
                label="Early Stop LR",
                value=config.get("model_quantizer.early_stop_lr", 1e-10),
            )
            early_stop_stall = gr.Number(
                label="Early Stop Stall",
                value=config.get("model_quantizer.early_stop_stall", 1000),
                step=1,
            )

    with gr.Accordion("NVFP4 / MXFP8 Options", open=False) as nvfp4_group:
        with gr.Row():
            scale_optimization = gr.Dropdown(
                label="Scale Optimization (NVFP4)",
                choices=["fixed", "iterative", "joint"],
                value=config.get("model_quantizer.scale_optimization", "fixed"),
            )
            scale_refinement_rounds = gr.Number(
                label="Scale Refinement Rounds",
                value=config.get("model_quantizer.scale_refinement_rounds", 1),
                step=1,
            )
        with gr.Row():
            input_scales_path = gr.Textbox(
                label="Input Scales (.json/.safetensors)",
                value=config.get("model_quantizer.input_scales_path", ""),
                placeholder="Optional input scales for NVFP4",
            )
            input_scales_button = gr.Button("Browse File", size="lg")
        with gr.Row():
            tensor_scales_path = gr.Textbox(
                label="Tensor Scales for Hybrid MXFP8",
                value=config.get("model_quantizer.tensor_scales_path", ""),
                placeholder="Optional tensorwise FP8 model for hybrid conversion",
            )
            tensor_scales_button = gr.Button("Browse File", size="lg")

    with gr.Accordion("Layer Config & Dry Run", open=False) as layer_config_group:
        with gr.Row():
            layer_config_path = gr.Textbox(
                label="Layer Config JSON",
                value=config.get("model_quantizer.layer_config_path", ""),
                placeholder="Path to layer-config JSON",
            )
            layer_config_button = gr.Button("Browse File", size="lg")
        layer_config_fullmatch = gr.Checkbox(
            label="Layer Config Fullmatch",
            value=config.get("model_quantizer.layer_config_fullmatch", False),
        )
        dry_run = gr.Dropdown(
            label="Dry Run Mode",
            choices=[None, "analyze", "create-template"],
            value=config.get("model_quantizer.dry_run", None),
        )

    with gr.Accordion("Activation Calibration (actcal)", open=False):
        with gr.Row():
            actcal_samples = gr.Number(
                label="Calibration Samples",
                value=config.get("model_quantizer.actcal_samples", 64),
                step=1,
            )
            actcal_percentile = gr.Number(
                label="Percentile",
                value=config.get("model_quantizer.actcal_percentile", 99.9),
            )
            actcal_seed = gr.Number(
                label="Calibration Seed",
                value=config.get("model_quantizer.actcal_seed", 42),
                step=1,
            )
        with gr.Row():
            actcal_lora = gr.Textbox(
                label="LoRA for Calibration (optional)",
                value=config.get("model_quantizer.actcal_lora", ""),
                placeholder="Optional LoRA file for informed calibration",
            )
            actcal_lora_button = gr.Button("Browse File", size="lg")
        actcal_device = gr.Textbox(
            label="Calibration Device",
            value=config.get("model_quantizer.actcal_device", ""),
            placeholder="cpu, cuda, cuda:0",
        )

    with gr.Accordion("Legacy & Metadata Tools", open=False) as legacy_group:
        with gr.Row():
            save_quant_metadata = gr.Checkbox(
                label="Save Quantization Metadata",
                value=config.get("model_quantizer.save_quant_metadata", False),
            )
            no_normalize_scales = gr.Checkbox(
                label="Disable Scale Normalization",
                value=config.get("model_quantizer.no_normalize_scales", False),
            )
            scaled_fp8_marker = gr.Dropdown(
                label="scaled_fp8 Marker Size",
                choices=[0, 2],
                value=config.get("model_quantizer.scaled_fp8_marker", 0),
            )
        with gr.Row():
            hp_filter = gr.Textbox(
                label="HP Filter (convert fp8 scaled)",
                value=config.get("model_quantizer.hp_filter", ""),
                placeholder="Regex for high-precision validation",
            )
            full_precision_mm = gr.Checkbox(
                label="Full precision matmul (convert fp8 scaled)",
                value=config.get("model_quantizer.full_precision_mm", False),
            )
        with gr.Row():
            remove_keys = gr.Textbox(
                label="Remove Keys (edit comfy_quant)",
                value=config.get("model_quantizer.remove_keys", ""),
                placeholder="Comma-separated keys",
            )
            add_keys = gr.Textbox(
                label="Add/Update Keys (edit comfy_quant)",
                value=config.get("model_quantizer.add_keys", ""),
                placeholder="'full_precision_matrix_mult': true, 'group_size': 64",
            )
        quant_filter = gr.Textbox(
            label="Quant Filter (edit comfy_quant)",
            value=config.get("model_quantizer.quant_filter", ""),
            placeholder="Regex for layers to edit",
        )

    with gr.Accordion("Runtime & Output", open=True):
        with gr.Row():
            verbose = gr.Dropdown(
                label="Verbosity",
                choices=["DEBUG", "VERBOSE", "NORMAL", "MINIMAL"],
                value=config.get("model_quantizer.verbose", "MINIMAL"),
            )
            verbose_pinned = gr.Checkbox(
                label="Verbose pinned memory transfers",
                value=config.get("model_quantizer.verbose_pinned", False),
            )
            low_memory = gr.Checkbox(
                label="Low memory mode",
                value=config.get("model_quantizer.low_memory", False),
            )

    with gr.Tabs():
        with gr.Tab("Single File"):
            with gr.Row():
                single_input_file = gr.Textbox(
                    label="Input Safetensors",
                    value=config.get("model_quantizer.single_input_file", ""),
                    placeholder="Path to model .safetensors",
                )
                single_input_button = gr.Button("Browse File", size="lg")
            with gr.Row():
                single_output_file = gr.Textbox(
                    label="Output File (optional)",
                    value=config.get("model_quantizer.single_output_file", ""),
                    placeholder="Leave empty for auto naming",
                )
                single_output_button = gr.Button("Save As", size="lg")
            with gr.Row():
                single_delete_original = gr.Checkbox(
                    label="Delete original after success",
                    value=config.get("model_quantizer.single_delete_original", False),
                )
            single_run_button = gr.Button("Start Conversion", variant="primary")
            single_cancel_button = gr.Button("Cancel", variant="secondary")
            single_status = gr.Textbox(
                label="Single Conversion Log",
                lines=16,
                max_lines=60,
                interactive=False,
            )

        with gr.Tab("Batch Folder"):
            with gr.Row():
                batch_input_folder = gr.Textbox(
                    label="Input Folder",
                    value=config.get("model_quantizer.batch_input_folder", ""),
                    placeholder="Folder with model files",
                )
                batch_input_button = gr.Button("Browse Folder", size="lg")
            with gr.Row():
                batch_output_folder = gr.Textbox(
                    label="Output Folder (optional)",
                    value=config.get("model_quantizer.batch_output_folder", ""),
                    placeholder="Leave empty to use input folder",
                )
                batch_output_button = gr.Button("Browse Folder", size="lg")
            with gr.Row():
                batch_extensions = gr.Textbox(
                    label="File Extensions",
                    value=config.get("model_quantizer.batch_extensions", ".safetensors"),
                    placeholder=".safetensors,.pt",
                )
                batch_recursive = gr.Checkbox(
                    label="Recursive",
                    value=config.get("model_quantizer.batch_recursive", True),
                )
                batch_overwrite = gr.Checkbox(
                    label="Overwrite Existing Outputs",
                    value=config.get("model_quantizer.batch_overwrite", False),
                )
            with gr.Row():
                batch_delete_original = gr.Checkbox(
                    label="Delete originals after success",
                    value=config.get("model_quantizer.batch_delete_original", False),
                )
            batch_run_button = gr.Button("Start Batch Conversion", variant="primary")
            batch_cancel_button = gr.Button("Cancel Batch", variant="secondary")
            batch_status = gr.Textbox(
                label="Batch Conversion Log",
                lines=18,
                max_lines=80,
                interactive=False,
            )

    def _build_params_dict(
        workflow_value,
        quant_format_value,
        comfy_quant_value,
        full_precision_matrix_mult_value,
        scaling_mode_value,
        block_size_value,
        include_input_scale_value,
        custom_layers_value,
        exclude_layers_value,
        custom_type_value,
        custom_block_size_value,
        custom_scaling_mode_value,
        custom_simple_value,
        custom_heur_value,
        fallback_type_value,
        fallback_block_size_value,
        fallback_simple_value,
        simple_value,
        skip_inefficient_layers_value,
        full_matrix_value,
        calib_samples_value,
        manual_seed_value,
        optimizer_value,
        num_iter_value,
        lr_value,
        lr_schedule_value,
        top_p_value,
        min_k_value,
        max_k_value,
        lr_gamma_value,
        lr_patience_value,
        lr_factor_value,
        lr_min_value,
        lr_cooldown_value,
        lr_threshold_value,
        lr_adaptive_mode_value,
        lr_shape_influence_value,
        lr_threshold_mode_value,
        early_stop_loss_value,
        early_stop_lr_value,
        early_stop_stall_value,
        scale_optimization_value,
        scale_refinement_rounds_value,
        input_scales_path_value,
        tensor_scales_path_value,
        layer_config_path_value,
        layer_config_fullmatch_value,
        dry_run_value,
        verbose_value,
        output_mode_value,
        verbose_pinned_value,
        low_memory_value,
        save_quant_metadata_value,
        no_normalize_scales_value,
        scaled_fp8_marker_value,
        hp_filter_value,
        full_precision_mm_value,
        remove_keys_value,
        add_keys_value,
        quant_filter_value,
        actcal_samples_value,
        actcal_percentile_value,
        actcal_lora_value,
        actcal_seed_value,
        actcal_device_value,
        filter_values: Dict[str, bool],
    ) -> Dict[str, object]:
        return {
            "workflow": workflow_value,
            "quant_format": quant_format_value,
            "comfy_quant": bool(comfy_quant_value),
            "full_precision_matrix_mult": bool(full_precision_matrix_mult_value),
            "scaling_mode": scaling_mode_value,
            "block_size": _to_int(block_size_value, None),
            "include_input_scale": bool(include_input_scale_value),
            "custom_layers": custom_layers_value.strip() if isinstance(custom_layers_value, str) else custom_layers_value,
            "exclude_layers": exclude_layers_value.strip() if isinstance(exclude_layers_value, str) else exclude_layers_value,
            "custom_type": custom_type_value,
            "custom_block_size": _to_int(custom_block_size_value, None),
            "custom_scaling_mode": custom_scaling_mode_value,
            "custom_simple": bool(custom_simple_value),
            "custom_heur": bool(custom_heur_value),
            "fallback_type": fallback_type_value,
            "fallback_block_size": _to_int(fallback_block_size_value, None),
            "fallback_simple": bool(fallback_simple_value),
            "simple": bool(simple_value),
            "skip_inefficient_layers": bool(skip_inefficient_layers_value),
            "full_matrix": bool(full_matrix_value),
            "calib_samples": _to_int(calib_samples_value, None),
            "manual_seed": _to_int(manual_seed_value, None),
            "optimizer": optimizer_value,
            "num_iter": _to_int(num_iter_value, None),
            "lr": _to_float(lr_value, None),
            "lr_schedule": lr_schedule_value,
            "top_p": _to_float(top_p_value, None),
            "min_k": _to_int(min_k_value, None),
            "max_k": _to_int(max_k_value, None),
            "lr_gamma": _to_float(lr_gamma_value, None),
            "lr_patience": _to_int(lr_patience_value, None),
            "lr_factor": _to_float(lr_factor_value, None),
            "lr_min": _to_float(lr_min_value, None),
            "lr_cooldown": _to_int(lr_cooldown_value, None),
            "lr_threshold": _to_float(lr_threshold_value, None),
            "lr_adaptive_mode": lr_adaptive_mode_value,
            "lr_shape_influence": _to_float(lr_shape_influence_value, None),
            "lr_threshold_mode": lr_threshold_mode_value,
            "early_stop_loss": _to_float(early_stop_loss_value, None),
            "early_stop_lr": _to_float(early_stop_lr_value, None),
            "early_stop_stall": _to_int(early_stop_stall_value, None),
            "scale_optimization": scale_optimization_value,
            "scale_refinement_rounds": _to_int(scale_refinement_rounds_value, None),
            "input_scales_path": input_scales_path_value.strip() if isinstance(input_scales_path_value, str) else input_scales_path_value,
            "tensor_scales_path": tensor_scales_path_value.strip() if isinstance(tensor_scales_path_value, str) else tensor_scales_path_value,
            "layer_config_path": layer_config_path_value.strip() if isinstance(layer_config_path_value, str) else layer_config_path_value,
            "layer_config_fullmatch": bool(layer_config_fullmatch_value),
            "dry_run": dry_run_value,
            "verbose": verbose_value,
            "output_mode": output_mode_value,
            "verbose_pinned": bool(verbose_pinned_value),
            "low_memory": bool(low_memory_value),
            "save_quant_metadata": bool(save_quant_metadata_value),
            "no_normalize_scales": bool(no_normalize_scales_value),
            "scaled_fp8_marker": _to_int(scaled_fp8_marker_value, None),
            "hp_filter": hp_filter_value.strip() if isinstance(hp_filter_value, str) else hp_filter_value,
            "full_precision_mm": bool(full_precision_mm_value),
            "remove_keys": remove_keys_value.strip() if isinstance(remove_keys_value, str) else remove_keys_value,
            "add_keys": add_keys_value.strip() if isinstance(add_keys_value, str) else add_keys_value,
            "quant_filter": quant_filter_value.strip() if isinstance(quant_filter_value, str) else quant_filter_value,
            "actcal_samples": _to_int(actcal_samples_value, None),
            "actcal_percentile": _to_float(actcal_percentile_value, None),
            "actcal_lora": actcal_lora_value.strip() if isinstance(actcal_lora_value, str) else actcal_lora_value,
            "actcal_seed": _to_int(actcal_seed_value, None),
            "actcal_device": actcal_device_value.strip() if isinstance(actcal_device_value, str) else actcal_device_value,
            "model_filters": filter_values,
        }

    def _collect_filter_values(*values) -> Dict[str, bool]:
        return {name: bool(val) for name, val in zip(filter_checkboxes.keys(), values)}

    def _apply_preset(preset_name: str):
        overrides = PRESET_OVERRIDES.get(preset_name, {})
        field_names = [
            "quant_format",
            "comfy_quant",
            "full_precision_matrix_mult",
            "scaling_mode",
            "block_size",
            "simple",
            "skip_inefficient_layers",
            "num_iter",
            "calib_samples",
            "optimizer",
            "lr_schedule",
            "lr",
            "top_p",
            "min_k",
            "max_k",
            "full_matrix",
            "scale_optimization",
            "scale_refinement_rounds",
            "manual_seed",
            "verbose",
        ]
        updates = []
        for name in field_names:
            if name in overrides:
                updates.append(gr.update(value=overrides[name]))
            else:
                updates.append(gr.update())
        return updates

    preset_dropdown.change(
        fn=_apply_preset,
        inputs=[preset_dropdown],
        outputs=[
            quant_format,
            comfy_quant,
            full_precision_matrix_mult,
            scaling_mode,
            block_size,
            simple,
            skip_inefficient_layers,
            num_iter,
            calib_samples,
            optimizer,
            lr_schedule,
            lr,
            top_p,
            min_k,
            max_k,
            full_matrix,
            scale_optimization,
            scale_refinement_rounds,
            manual_seed,
            verbose,
        ],
        show_progress=False,
    )

    def _apply_model_preset(selected: str):
        updates = []
        overrides = MODEL_PRESET_OVERRIDES.get(selected, {})
        for name in filter_checkboxes.keys():
            updates.append(gr.update(value=(name == selected)))
        include_update = overrides.get("include_input_scale")
        if include_update is None:
            include_update = gr.update()
        else:
            include_update = gr.update(value=include_update)
        return updates + [include_update]

    model_preset_dropdown.change(
        fn=_apply_model_preset,
        inputs=[model_preset_dropdown],
        outputs=list(filter_checkboxes.values()) + [include_input_scale],
        show_progress=False,
    )

    def _update_workflow_visibility(selected: str):
        is_quantize = selected == WORKFLOW_QUANTIZE or selected == WORKFLOW_DRY_RUN
        return (
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
        )

    workflow.change(
        fn=_update_workflow_visibility,
        inputs=[workflow],
        outputs=[
            quant_format_group,
            model_filter_group,
            layer_mixing_group,
            optimization_group,
            advanced_lr_group,
            nvfp4_group,
            layer_config_group,
        ],
        show_progress=False,
    )

    single_input_button.click(
        fn=lambda current: get_file_path(current, ".safetensors", "Model files"),
        inputs=[single_input_file],
        outputs=[single_input_file],
        show_progress=False,
    )
    single_output_button.click(
        fn=lambda current: get_saveasfilename_path(
            current, extensions="*.safetensors", extension_name="Safetensors files"
        ),
        inputs=[single_output_file],
        outputs=[single_output_file],
        show_progress=False,
    )
    batch_input_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[batch_input_folder],
        outputs=[batch_input_folder],
        show_progress=False,
    )
    batch_output_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[batch_output_folder],
        outputs=[batch_output_folder],
        show_progress=False,
    )
    layer_config_button.click(
        fn=lambda current: get_file_path(current, ".json", "JSON files"),
        inputs=[layer_config_path],
        outputs=[layer_config_path],
        show_progress=False,
    )
    input_scales_button.click(
        fn=lambda current: get_file_path(current, ".json", "JSON or Safetensors"),
        inputs=[input_scales_path],
        outputs=[input_scales_path],
        show_progress=False,
    )
    tensor_scales_button.click(
        fn=lambda current: get_file_path(current, ".safetensors", "Safetensors files"),
        inputs=[tensor_scales_path],
        outputs=[tensor_scales_path],
        show_progress=False,
    )
    actcal_lora_button.click(
        fn=lambda current: get_file_path(current, ".safetensors", "Safetensors files"),
        inputs=[actcal_lora],
        outputs=[actcal_lora],
        show_progress=False,
    )

    single_inputs = [
        workflow,
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        include_input_scale,
        custom_layers,
        exclude_layers,
        custom_type,
        custom_block_size,
        custom_scaling_mode,
        custom_simple,
        custom_heur,
        fallback_type,
        fallback_block_size,
        fallback_simple,
        simple,
        skip_inefficient_layers,
        full_matrix,
        calib_samples,
        manual_seed,
        optimizer,
        num_iter,
        lr,
        lr_schedule,
        top_p,
        min_k,
        max_k,
        lr_gamma,
        lr_patience,
        lr_factor,
        lr_min,
        lr_cooldown,
        lr_threshold,
        lr_adaptive_mode,
        lr_shape_influence,
        lr_threshold_mode,
        early_stop_loss,
        early_stop_lr,
        early_stop_stall,
        scale_optimization,
        scale_refinement_rounds,
        input_scales_path,
        tensor_scales_path,
        layer_config_path,
        layer_config_fullmatch,
        dry_run,
        verbose,
        output_mode,
        verbose_pinned,
        low_memory,
        save_quant_metadata,
        no_normalize_scales,
        scaled_fp8_marker,
        hp_filter,
        full_precision_mm,
        remove_keys,
        add_keys,
        quant_filter,
        actcal_samples,
        actcal_percentile,
        actcal_lora,
        actcal_seed,
        actcal_device,
    ] + list(filter_checkboxes.values())

    def _run_single(
        single_input_file_value,
        single_output_file_value,
        single_delete_original_value,
        *values,
    ):
        filter_count = len(filter_checkboxes)
        if filter_count:
            filter_values = _collect_filter_values(*values[-filter_count:])
            base_values = values[:-filter_count]
        else:
            filter_values = {}
            base_values = values
        params = _build_params_dict(
            *base_values,
            filter_values=filter_values,
        )
        return quantizer.run_single(
            input_file=single_input_file_value,
            output_file=single_output_file_value,
            delete_original=bool(single_delete_original_value),
            params=params,
        )

    single_run_button.click(
        fn=_run_single,
        inputs=[single_input_file, single_output_file, single_delete_original] + single_inputs,
        outputs=[single_status],
        show_progress=True,
    )

    single_cancel_button.click(
        fn=quantizer.cancel_single,
        inputs=[],
        outputs=[single_status],
        show_progress=False,
    )

    def _run_batch(
        batch_input_folder_value,
        batch_output_folder_value,
        batch_extensions_value,
        batch_recursive_value,
        batch_overwrite_value,
        batch_delete_original_value,
        *values,
    ):
        filter_count = len(filter_checkboxes)
        if filter_count:
            filter_values = _collect_filter_values(*values[-filter_count:])
            base_values = values[:-filter_count]
        else:
            filter_values = {}
            base_values = values
        params = _build_params_dict(
            *base_values,
            filter_values=filter_values,
        )
        return quantizer.run_batch(
            input_folder=batch_input_folder_value,
            output_folder=batch_output_folder_value,
            extensions=batch_extensions_value,
            recursive=bool(batch_recursive_value),
            overwrite_existing=bool(batch_overwrite_value),
            delete_original=bool(batch_delete_original_value),
            params=params,
        )

    batch_run_button.click(
        fn=_run_batch,
        inputs=[
            batch_input_folder,
            batch_output_folder,
            batch_extensions,
            batch_recursive,
            batch_overwrite,
            batch_delete_original,
        ] + single_inputs,
        outputs=[batch_status],
        show_progress=True,
    )

    batch_cancel_button.click(
        fn=quantizer.cancel_batch,
        inputs=[],
        outputs=[batch_status],
        show_progress=False,
    )

    settings_names = [
        "workflow",
        "quant_format",
        "comfy_quant",
        "full_precision_matrix_mult",
        "scaling_mode",
        "block_size",
        "include_input_scale",
        "custom_layers",
        "exclude_layers",
        "custom_type",
        "custom_block_size",
        "custom_scaling_mode",
        "custom_simple",
        "custom_heur",
        "fallback_type",
        "fallback_block_size",
        "fallback_simple",
        "simple",
        "skip_inefficient_layers",
        "full_matrix",
        "calib_samples",
        "manual_seed",
        "optimizer",
        "num_iter",
        "lr",
        "lr_schedule",
        "top_p",
        "min_k",
        "max_k",
        "lr_gamma",
        "lr_patience",
        "lr_factor",
        "lr_min",
        "lr_cooldown",
        "lr_threshold",
        "lr_adaptive_mode",
        "lr_shape_influence",
        "lr_threshold_mode",
        "early_stop_loss",
        "early_stop_lr",
        "early_stop_stall",
        "scale_optimization",
        "scale_refinement_rounds",
        "input_scales_path",
        "tensor_scales_path",
        "layer_config_path",
        "layer_config_fullmatch",
        "dry_run",
        "verbose",
        "output_mode",
        "verbose_pinned",
        "low_memory",
        "save_quant_metadata",
        "no_normalize_scales",
        "scaled_fp8_marker",
        "hp_filter",
        "full_precision_mm",
        "remove_keys",
        "add_keys",
        "quant_filter",
        "actcal_samples",
        "actcal_percentile",
        "actcal_lora",
        "actcal_seed",
        "actcal_device",
        "single_input_file",
        "single_output_file",
        "single_delete_original",
        "batch_input_folder",
        "batch_output_folder",
        "batch_extensions",
        "batch_recursive",
        "batch_overwrite",
        "batch_delete_original",
        "preset",
        "model_preset",
    ] + [f"filter.{name}" for name in filter_checkboxes.keys()]

    settings_components = [
        workflow,
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        include_input_scale,
        custom_layers,
        exclude_layers,
        custom_type,
        custom_block_size,
        custom_scaling_mode,
        custom_simple,
        custom_heur,
        fallback_type,
        fallback_block_size,
        fallback_simple,
        simple,
        skip_inefficient_layers,
        full_matrix,
        calib_samples,
        manual_seed,
        optimizer,
        num_iter,
        lr,
        lr_schedule,
        top_p,
        min_k,
        max_k,
        lr_gamma,
        lr_patience,
        lr_factor,
        lr_min,
        lr_cooldown,
        lr_threshold,
        lr_adaptive_mode,
        lr_shape_influence,
        lr_threshold_mode,
        early_stop_loss,
        early_stop_lr,
        early_stop_stall,
        scale_optimization,
        scale_refinement_rounds,
        input_scales_path,
        tensor_scales_path,
        layer_config_path,
        layer_config_fullmatch,
        dry_run,
        verbose,
        output_mode,
        verbose_pinned,
        low_memory,
        save_quant_metadata,
        no_normalize_scales,
        scaled_fp8_marker,
        hp_filter,
        full_precision_mm,
        remove_keys,
        add_keys,
        quant_filter,
        actcal_samples,
        actcal_percentile,
        actcal_lora,
        actcal_seed,
        actcal_device,
        single_input_file,
        single_output_file,
        single_delete_original,
        batch_input_folder,
        batch_output_folder,
        batch_extensions,
        batch_recursive,
        batch_overwrite,
        batch_delete_original,
        preset_dropdown,
        model_preset_dropdown,
    ] + list(filter_checkboxes.values())

    def _save_configuration(action, save_as_bool, file_path, headless_value, print_only, *values):
        parameters = list(zip(settings_names, values))
        original_file_path = file_path
        if save_as_bool or not file_path:
            file_path = get_saveasfilename_path(
                file_path, extensions="*.toml", extension_name="TOML files"
            )
        if not file_path:
            return original_file_path, "Save cancelled"
        destination_directory = os.path.dirname(file_path)
        if destination_directory:
            os.makedirs(destination_directory, exist_ok=True)
        from .common_gui import SaveConfigFile
        SaveConfigFile(parameters=parameters, file_path=file_path, exclusion=["file_path", "save_as", "save_as_bool"])
        config_name = os.path.basename(file_path)
        return file_path, f"Saved: {config_name}"

    def _open_configuration(action, ask_for_file, file_path, headless_value, print_only, *values):
        original_file_path = file_path
        if ask_for_file:
            file_path = get_file_path_or_save_as(
                file_path, default_extension=".toml", extension_name="TOML files"
            )
        if not file_path:
            return [original_file_path, "Load cancelled"] + [gr.update() for _ in settings_components]
        if not os.path.isfile(file_path) and ask_for_file:
            return [file_path, f"New config: {os.path.basename(file_path)}"] + [gr.update() for _ in settings_components]
        if not os.path.isfile(file_path):
            return [original_file_path, "Config file not found"] + [gr.update() for _ in settings_components]
        try:
            data = toml.load(file_path)
        except Exception as exc:
            return [original_file_path, f"Failed to load: {exc}"] + [gr.update() for _ in settings_components]
        flat = _flatten_dict(data)
        values_out = []
        for name, current in zip(settings_names, values):
            value = flat.get(name)
            if value is None:
                value = flat.get(f"model_quantizer.{name}")
            if value is None:
                values_out.append(current)
            else:
                values_out.append(value)
        return [file_path, f"Loaded: {os.path.basename(file_path)}"] + values_out

    configuration.button_open_config.click(
        fn=_open_configuration,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_components,
        show_progress=False,
    )

    configuration.button_load_config.click(
        fn=_open_configuration,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_components,
        show_progress=False,
        queue=False,
    )

    configuration.button_save_config.click(
        fn=_save_configuration,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )

    configuration.config_file_name.change(
        fn=lambda config_name, *args: (
            _open_configuration("open_configuration", False, config_name, dummy_headless.value, False, *args)
            if config_name and config_name.endswith(".toml")
            else ([config_name, ""] + [gr.update() for _ in settings_components])
        ),
        inputs=[configuration.config_file_name] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_components,
        show_progress=False,
        queue=False,
    )


def model_quantizer_tab(headless: bool, config: GUIConfig) -> None:
    quantizer = ModelQuantizer(headless=headless, config=config)

    gr.Markdown("# Model Quantizer")
    gr.Markdown(
        "Quantize safetensors checkpoints with **convert_to_quant**. Supports FP8, INT8, NVFP4, MXFP8, "
        "metadata conversions, calibration, and batch processing."
    )

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    with gr.Accordion("Configuration file Settings", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Presets", open=True):
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        label="Quality Preset",
                        choices=[
                            PRESET_CUSTOM,
                            PRESET_FAST,
                            PRESET_NORMAL,
                            PRESET_HIGH,
                            PRESET_INT8_FAST,
                            PRESET_INT8_TENSOR,
                            PRESET_MXFP8_BALANCED,
                            PRESET_NVFP4_BALANCED,
                            PRESET_NVFP4_Z,
                        ],
                        value=config.get("model_quantizer.preset", PRESET_CUSTOM),
                        info="Quickly apply recommended optimization settings.",
                    )
                    model_preset_dropdown = gr.Dropdown(
                        label="Model Preset (Recommended Filters)",
                        choices=MODEL_PRESET_CHOICES,
                        value=config.get("model_quantizer.model_preset", MODEL_PRESET_NONE),
                        info="Select a model to apply its recommended exclusion filters and defaults.",
                    )

            with gr.Accordion("Workflow", open=True):
                workflow = gr.Dropdown(
                    label="Workflow",
                    choices=[
                        WORKFLOW_QUANTIZE,
                        WORKFLOW_CONVERT_FP8,
                        WORKFLOW_CONVERT_INT8,
                        WORKFLOW_LEGACY_INPUT,
                        WORKFLOW_CLEANUP_FP8,
                        WORKFLOW_ACTCAL,
                        WORKFLOW_EDIT_QUANT,
                        WORKFLOW_HYBRID_MXFP8,
                        WORKFLOW_DRY_RUN,
                    ],
                    value=config.get("model_quantizer.workflow", WORKFLOW_QUANTIZE),
                )

            with gr.Accordion("Quantization Format", open=True) as quant_format_group:
                with gr.Row():
                    quant_format = gr.Dropdown(
                        label="Primary Format",
                        choices=[QUANT_FORMAT_FP8, QUANT_FORMAT_INT8, QUANT_FORMAT_NVFP4, QUANT_FORMAT_MXFP8],
                        value=config.get("model_quantizer.quant_format", QUANT_FORMAT_FP8),
                    )
                    comfy_quant = gr.Checkbox(
                        label="Write ComfyUI Metadata (.comfy_quant)",
                        value=config.get("model_quantizer.comfy_quant", True),
                    )
                    full_precision_matrix_mult = gr.Checkbox(
                        label="comfy_quant: Full precision matrix mult",
                        value=config.get("model_quantizer.full_precision_matrix_mult", False),
                    )

                with gr.Row():
                    scaling_mode = gr.Dropdown(
                        label="Scaling Mode",
                        choices=["tensor", "row", "block", "block3d", "block2d"],
                        value=config.get("model_quantizer.scaling_mode", "tensor"),
                    )
                    block_size = gr.Number(
                        label="Block Size",
                        value=config.get("model_quantizer.block_size", 64),
                        step=1,
                        interactive=True,
                    )
                    include_input_scale = gr.Checkbox(
                        label="Include input_scale tensors",
                        value=config.get("model_quantizer.include_input_scale", False),
                    )

            with gr.Accordion("Model Filters", open=False) as model_filter_group:
                filter_checkboxes: Dict[str, gr.Checkbox] = {}
                for category_key, category_label in MODEL_CATEGORY_LABELS.items():
                    filters_in_category = [
                        (name, cfg) for name, cfg in MODEL_FILTERS.items()
                        if cfg.get("category") == category_key
                    ]
                    if not filters_in_category:
                        continue
                    with gr.Group():
                        gr.Markdown(f"### {category_label}")
                        for name, cfg in sorted(filters_in_category, key=lambda item: item[0]):
                            label = f"{name}: {cfg.get('help', '')}".strip(": ")
                            filter_checkboxes[name] = gr.Checkbox(
                                label=label,
                                value=bool(config.get(f"model_quantizer.filter.{name}", False)),
                            )

            with gr.Accordion("Layer Mixing & Exclusions", open=False) as layer_mixing_group:
                with gr.Row():
                    custom_layers = gr.Textbox(
                        label="Custom Layers (Regex)",
                        placeholder="e.g. attn|mlp",
                        value=config.get("model_quantizer.custom_layers", ""),
                    )
                    exclude_layers = gr.Textbox(
                        label="Exclude Layers (Regex)",
                        placeholder="e.g. norm|bias",
                        value=config.get("model_quantizer.exclude_layers", ""),
                    )
                with gr.Row():
                    custom_type = gr.Dropdown(
                        label="Custom Type",
                        choices=[None, "fp8", "int8", "mxfp8", "nvfp4"],
                        value=config.get("model_quantizer.custom_type", None),
                        allow_custom_value=False,
                    )
                    custom_block_size = gr.Number(
                        label="Custom Block Size",
                        value=config.get("model_quantizer.custom_block_size", None),
                        step=1,
                    )
                    custom_scaling_mode = gr.Dropdown(
                        label="Custom Scaling Mode",
                        choices=[None, "tensor", "row", "block", "block3d", "block2d"],
                        value=config.get("model_quantizer.custom_scaling_mode", None),
                    )
                with gr.Row():
                    custom_simple = gr.Checkbox(
                        label="Custom: Simple quantization",
                        value=config.get("model_quantizer.custom_simple", False),
                    )
                    custom_heur = gr.Checkbox(
                        label="Custom: Heuristics",
                        value=config.get("model_quantizer.custom_heur", False),
                    )

                with gr.Row():
                    fallback_type = gr.Dropdown(
                        label="Fallback Type",
                        choices=[None, "fp8", "int8", "mxfp8", "nvfp4"],
                        value=config.get("model_quantizer.fallback_type", None),
                    )
                    fallback_block_size = gr.Number(
                        label="Fallback Block Size",
                        value=config.get("model_quantizer.fallback_block_size", None),
                        step=1,
                    )
                    fallback_simple = gr.Checkbox(
                        label="Fallback: Simple quantization",
                        value=config.get("model_quantizer.fallback_simple", False),
                    )

        with gr.Column(scale=1):
            with gr.Accordion("Optimization & Quality", open=False) as optimization_group:
                with gr.Row():
                    simple = gr.Checkbox(
                        label="Simple quantization (skip SVD)",
                        value=config.get("model_quantizer.simple", False),
                    )
                    skip_inefficient_layers = gr.Checkbox(
                        label="Skip inefficient layers (heuristics)",
                        value=config.get("model_quantizer.skip_inefficient_layers", False),
                    )
                    full_matrix = gr.Checkbox(
                        label="Use full SVD matrix",
                        value=config.get("model_quantizer.full_matrix", False),
                    )
                with gr.Row():
                    calib_samples = gr.Number(
                        label="Calibration Samples",
                        value=config.get("model_quantizer.calib_samples", 6144),
                        step=1,
                    )
                    manual_seed = gr.Number(
                        label="Manual Seed (-1=random)",
                        value=config.get("model_quantizer.manual_seed", -1),
                        step=1,
                    )
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=["original", "adamw", "radam"],
                        value=config.get("model_quantizer.optimizer", "original"),
                    )
                with gr.Row():
                    num_iter = gr.Number(
                        label="Iterations",
                        value=config.get("model_quantizer.num_iter", 1000),
                        step=1,
                    )
                    lr = gr.Number(
                        label="Learning Rate",
                        value=config.get("model_quantizer.lr", 8.077300000003e-3),
                    )
                    lr_schedule = gr.Dropdown(
                        label="LR Schedule",
                        choices=["adaptive", "exponential", "plateau"],
                        value=config.get("model_quantizer.lr_schedule", "adaptive"),
                    )
                with gr.Row():
                    top_p = gr.Number(
                        label="Top P",
                        value=config.get("model_quantizer.top_p", 0.2),
                    )
                    min_k = gr.Number(
                        label="Min K",
                        value=config.get("model_quantizer.min_k", 64),
                        step=1,
                    )
                    max_k = gr.Number(
                        label="Max K",
                        value=config.get("model_quantizer.max_k", 1024),
                        step=1,
                    )

            with gr.Accordion("Advanced LR & Early Stopping", open=False) as advanced_lr_group:
                with gr.Row():
                    lr_gamma = gr.Number(
                        label="LR Gamma",
                        value=config.get("model_quantizer.lr_gamma", 0.99),
                    )
                    lr_patience = gr.Number(
                        label="LR Patience",
                        value=config.get("model_quantizer.lr_patience", 9),
                        step=1,
                    )
                    lr_factor = gr.Number(
                        label="LR Factor",
                        value=config.get("model_quantizer.lr_factor", 0.95),
                    )
                with gr.Row():
                    lr_min = gr.Number(
                        label="LR Min",
                        value=config.get("model_quantizer.lr_min", 1e-10),
                    )
                    lr_cooldown = gr.Number(
                        label="LR Cooldown",
                        value=config.get("model_quantizer.lr_cooldown", 6),
                        step=1,
                    )
                    lr_threshold = gr.Number(
                        label="LR Threshold",
                        value=config.get("model_quantizer.lr_threshold", 0.0),
                    )
                with gr.Row():
                    lr_adaptive_mode = gr.Dropdown(
                        label="LR Adaptive Mode",
                        choices=["simple-reset", "no-reset"],
                        value=config.get("model_quantizer.lr_adaptive_mode", "simple-reset"),
                    )
                    lr_shape_influence = gr.Number(
                        label="LR Shape Influence",
                        value=config.get("model_quantizer.lr_shape_influence", 1.0),
                    )
                    lr_threshold_mode = gr.Dropdown(
                        label="LR Threshold Mode",
                        choices=["rel", "abs"],
                        value=config.get("model_quantizer.lr_threshold_mode", "rel"),
                    )
                with gr.Row():
                    early_stop_loss = gr.Number(
                        label="Early Stop Loss",
                        value=config.get("model_quantizer.early_stop_loss", 1e-8),
                    )
                    early_stop_lr = gr.Number(
                        label="Early Stop LR",
                        value=config.get("model_quantizer.early_stop_lr", 1e-10),
                    )
                    early_stop_stall = gr.Number(
                        label="Early Stop Stall",
                        value=config.get("model_quantizer.early_stop_stall", 1000),
                        step=1,
                    )

            with gr.Accordion("NVFP4 / MXFP8 Options", open=False) as nvfp4_group:
                with gr.Row():
                    scale_optimization = gr.Dropdown(
                        label="Scale Optimization (NVFP4)",
                        choices=["fixed", "iterative", "joint"],
                        value=config.get("model_quantizer.scale_optimization", "fixed"),
                    )
                    scale_refinement_rounds = gr.Number(
                        label="Scale Refinement Rounds",
                        value=config.get("model_quantizer.scale_refinement_rounds", 1),
                        step=1,
                    )
                with gr.Row():
                    input_scales_path = gr.Textbox(
                        label="Input Scales (.json/.safetensors)",
                        value=config.get("model_quantizer.input_scales_path", ""),
                        placeholder="Optional input scales for NVFP4",
                    )
                    input_scales_button = gr.Button("Browse File", size="lg")
                with gr.Row():
                    tensor_scales_path = gr.Textbox(
                        label="Tensor Scales for Hybrid MXFP8",
                        value=config.get("model_quantizer.tensor_scales_path", ""),
                        placeholder="Optional tensorwise FP8 model for hybrid conversion",
                    )
                    tensor_scales_button = gr.Button("Browse File", size="lg")

            with gr.Accordion("Layer Config & Dry Run", open=False) as layer_config_group:
                with gr.Row():
                    layer_config_path = gr.Textbox(
                        label="Layer Config JSON",
                        value=config.get("model_quantizer.layer_config_path", ""),
                        placeholder="Path to layer-config JSON",
                    )
                    layer_config_button = gr.Button("Browse File", size="lg")
                layer_config_fullmatch = gr.Checkbox(
                    label="Layer Config Fullmatch",
                    value=config.get("model_quantizer.layer_config_fullmatch", False),
                )
                dry_run = gr.Dropdown(
                    label="Dry Run Mode",
                    choices=[None, "analyze", "create-template"],
                    value=config.get("model_quantizer.dry_run", None),
                )

            with gr.Accordion("Activation Calibration (actcal)", open=False):
                with gr.Row():
                    actcal_samples = gr.Number(
                        label="Calibration Samples",
                        value=config.get("model_quantizer.actcal_samples", 64),
                        step=1,
                    )
                    actcal_percentile = gr.Number(
                        label="Percentile",
                        value=config.get("model_quantizer.actcal_percentile", 99.9),
                    )
                    actcal_seed = gr.Number(
                        label="Calibration Seed",
                        value=config.get("model_quantizer.actcal_seed", 42),
                        step=1,
                    )
                with gr.Row():
                    actcal_lora = gr.Textbox(
                        label="LoRA for Calibration (optional)",
                        value=config.get("model_quantizer.actcal_lora", ""),
                        placeholder="Optional LoRA file for informed calibration",
                    )
                    actcal_lora_button = gr.Button("Browse File", size="lg")
                actcal_device = gr.Textbox(
                    label="Calibration Device",
                    value=config.get("model_quantizer.actcal_device", ""),
                    placeholder="cpu, cuda, cuda:0",
                )

            with gr.Accordion("Legacy & Metadata Tools", open=False) as legacy_group:
                with gr.Row():
                    save_quant_metadata = gr.Checkbox(
                        label="Save Quantization Metadata",
                        value=config.get("model_quantizer.save_quant_metadata", False),
                    )
                    no_normalize_scales = gr.Checkbox(
                        label="Disable Scale Normalization",
                        value=config.get("model_quantizer.no_normalize_scales", False),
                    )
                    scaled_fp8_marker = gr.Dropdown(
                        label="scaled_fp8 Marker Size",
                        choices=[0, 2],
                        value=config.get("model_quantizer.scaled_fp8_marker", 0),
                    )
                with gr.Row():
                    hp_filter = gr.Textbox(
                        label="HP Filter (convert fp8 scaled)",
                        value=config.get("model_quantizer.hp_filter", ""),
                        placeholder="Regex for high-precision validation",
                    )
                    full_precision_mm = gr.Checkbox(
                        label="Full precision matmul (convert fp8 scaled)",
                        value=config.get("model_quantizer.full_precision_mm", False),
                    )
                with gr.Row():
                    remove_keys = gr.Textbox(
                        label="Remove Keys (edit comfy_quant)",
                        value=config.get("model_quantizer.remove_keys", ""),
                        placeholder="Comma-separated keys",
                    )
                    add_keys = gr.Textbox(
                        label="Add/Update Keys (edit comfy_quant)",
                        value=config.get("model_quantizer.add_keys", ""),
                        placeholder="'full_precision_matrix_mult': true, 'group_size': 64",
                    )
                quant_filter = gr.Textbox(
                    label="Quant Filter (edit comfy_quant)",
                    value=config.get("model_quantizer.quant_filter", ""),
                    placeholder="Regex for layers to edit",
                )

            with gr.Accordion("Runtime & Output", open=True):
                with gr.Row():
                    verbose = gr.Dropdown(
                        label="Verbosity",
                        choices=["DEBUG", "VERBOSE", "NORMAL", "MINIMAL"],
                        value=config.get("model_quantizer.verbose", "MINIMAL"),
                    )
                    verbose_pinned = gr.Checkbox(
                        label="Verbose pinned memory transfers",
                        value=config.get("model_quantizer.verbose_pinned", False),
                    )
                    low_memory = gr.Checkbox(
                        label="Low memory mode",
                        value=config.get("model_quantizer.low_memory", False),
                    )
                with gr.Row():
                    output_mode = gr.Dropdown(
                        label="Output Mode",
                        choices=OUTPUT_MODE_CHOICES,
                        value=config.get("model_quantizer.output_mode", OUTPUT_MODE_COMPACT),
                        info="Compact hides tqdm progress bars; Summary keeps warnings/errors.",
                    )

    with gr.Tabs():
        with gr.Tab("Single File"):
            with gr.Row():
                single_input_file = gr.Textbox(
                    label="Input Safetensors",
                    value=config.get("model_quantizer.single_input_file", ""),
                    placeholder="Path to model .safetensors",
                )
                single_input_button = gr.Button("Browse File", size="lg")
            with gr.Row():
                single_output_file = gr.Textbox(
                    label="Output File (optional)",
                    value=config.get("model_quantizer.single_output_file", ""),
                    placeholder="Leave empty for auto naming",
                )
                single_output_button = gr.Button("Save As", size="lg")
            with gr.Row():
                single_delete_original = gr.Checkbox(
                    label="Delete original after success",
                    value=config.get("model_quantizer.single_delete_original", False),
                )
            single_run_button = gr.Button("Start Conversion", variant="primary")
            single_cancel_button = gr.Button("Cancel", variant="secondary")
            single_status = gr.Textbox(
                label="Single Conversion Log",
                lines=16,
                max_lines=60,
                interactive=False,
            )

        with gr.Tab("Batch Folder"):
            with gr.Row():
                batch_input_folder = gr.Textbox(
                    label="Input Folder",
                    value=config.get("model_quantizer.batch_input_folder", ""),
                    placeholder="Folder with model files",
                )
                batch_input_button = gr.Button("Browse Folder", size="lg")
            with gr.Row():
                batch_output_folder = gr.Textbox(
                    label="Output Folder (optional)",
                    value=config.get("model_quantizer.batch_output_folder", ""),
                    placeholder="Leave empty to use input folder",
                )
                batch_output_button = gr.Button("Browse Folder", size="lg")
            with gr.Row():
                batch_extensions = gr.Textbox(
                    label="File Extensions",
                    value=config.get("model_quantizer.batch_extensions", ".safetensors"),
                    placeholder=".safetensors,.pt",
                )
                batch_recursive = gr.Checkbox(
                    label="Recursive",
                    value=config.get("model_quantizer.batch_recursive", True),
                )
                batch_overwrite = gr.Checkbox(
                    label="Overwrite Existing Outputs",
                    value=config.get("model_quantizer.batch_overwrite", False),
                )
            with gr.Row():
                batch_delete_original = gr.Checkbox(
                    label="Delete originals after success",
                    value=config.get("model_quantizer.batch_delete_original", False),
                )
            batch_run_button = gr.Button("Start Batch Conversion", variant="primary")
            batch_cancel_button = gr.Button("Cancel Batch", variant="secondary")
            batch_status = gr.Textbox(
                label="Batch Conversion Log",
                lines=18,
                max_lines=80,
                interactive=False,
            )

    def _build_params_dict(
        workflow_value,
        quant_format_value,
        comfy_quant_value,
        full_precision_matrix_mult_value,
        scaling_mode_value,
        block_size_value,
        include_input_scale_value,
        custom_layers_value,
        exclude_layers_value,
        custom_type_value,
        custom_block_size_value,
        custom_scaling_mode_value,
        custom_simple_value,
        custom_heur_value,
        fallback_type_value,
        fallback_block_size_value,
        fallback_simple_value,
        simple_value,
        skip_inefficient_layers_value,
        full_matrix_value,
        calib_samples_value,
        manual_seed_value,
        optimizer_value,
        num_iter_value,
        lr_value,
        lr_schedule_value,
        top_p_value,
        min_k_value,
        max_k_value,
        lr_gamma_value,
        lr_patience_value,
        lr_factor_value,
        lr_min_value,
        lr_cooldown_value,
        lr_threshold_value,
        lr_adaptive_mode_value,
        lr_shape_influence_value,
        lr_threshold_mode_value,
        early_stop_loss_value,
        early_stop_lr_value,
        early_stop_stall_value,
        scale_optimization_value,
        scale_refinement_rounds_value,
        input_scales_path_value,
        tensor_scales_path_value,
        layer_config_path_value,
        layer_config_fullmatch_value,
        dry_run_value,
        verbose_value,
        verbose_pinned_value,
        low_memory_value,
        save_quant_metadata_value,
        no_normalize_scales_value,
        scaled_fp8_marker_value,
        hp_filter_value,
        full_precision_mm_value,
        remove_keys_value,
        add_keys_value,
        quant_filter_value,
        actcal_samples_value,
        actcal_percentile_value,
        actcal_lora_value,
        actcal_seed_value,
        actcal_device_value,
        filter_values: Dict[str, bool],
    ) -> Dict[str, object]:
        return {
            "workflow": workflow_value,
            "quant_format": quant_format_value,
            "comfy_quant": bool(comfy_quant_value),
            "full_precision_matrix_mult": bool(full_precision_matrix_mult_value),
            "scaling_mode": scaling_mode_value,
            "block_size": _to_int(block_size_value, None),
            "include_input_scale": bool(include_input_scale_value),
            "custom_layers": custom_layers_value.strip() if isinstance(custom_layers_value, str) else custom_layers_value,
            "exclude_layers": exclude_layers_value.strip() if isinstance(exclude_layers_value, str) else exclude_layers_value,
            "custom_type": custom_type_value,
            "custom_block_size": _to_int(custom_block_size_value, None),
            "custom_scaling_mode": custom_scaling_mode_value,
            "custom_simple": bool(custom_simple_value),
            "custom_heur": bool(custom_heur_value),
            "fallback_type": fallback_type_value,
            "fallback_block_size": _to_int(fallback_block_size_value, None),
            "fallback_simple": bool(fallback_simple_value),
            "simple": bool(simple_value),
            "skip_inefficient_layers": bool(skip_inefficient_layers_value),
            "full_matrix": bool(full_matrix_value),
            "calib_samples": _to_int(calib_samples_value, None),
            "manual_seed": _to_int(manual_seed_value, None),
            "optimizer": optimizer_value,
            "num_iter": _to_int(num_iter_value, None),
            "lr": _to_float(lr_value, None),
            "lr_schedule": lr_schedule_value,
            "top_p": _to_float(top_p_value, None),
            "min_k": _to_int(min_k_value, None),
            "max_k": _to_int(max_k_value, None),
            "lr_gamma": _to_float(lr_gamma_value, None),
            "lr_patience": _to_int(lr_patience_value, None),
            "lr_factor": _to_float(lr_factor_value, None),
            "lr_min": _to_float(lr_min_value, None),
            "lr_cooldown": _to_int(lr_cooldown_value, None),
            "lr_threshold": _to_float(lr_threshold_value, None),
            "lr_adaptive_mode": lr_adaptive_mode_value,
            "lr_shape_influence": _to_float(lr_shape_influence_value, None),
            "lr_threshold_mode": lr_threshold_mode_value,
            "early_stop_loss": _to_float(early_stop_loss_value, None),
            "early_stop_lr": _to_float(early_stop_lr_value, None),
            "early_stop_stall": _to_int(early_stop_stall_value, None),
            "scale_optimization": scale_optimization_value,
            "scale_refinement_rounds": _to_int(scale_refinement_rounds_value, None),
            "input_scales_path": input_scales_path_value.strip() if isinstance(input_scales_path_value, str) else input_scales_path_value,
            "tensor_scales_path": tensor_scales_path_value.strip() if isinstance(tensor_scales_path_value, str) else tensor_scales_path_value,
            "layer_config_path": layer_config_path_value.strip() if isinstance(layer_config_path_value, str) else layer_config_path_value,
            "layer_config_fullmatch": bool(layer_config_fullmatch_value),
            "dry_run": dry_run_value,
            "verbose": verbose_value,
            "verbose_pinned": bool(verbose_pinned_value),
            "low_memory": bool(low_memory_value),
            "save_quant_metadata": bool(save_quant_metadata_value),
            "no_normalize_scales": bool(no_normalize_scales_value),
            "scaled_fp8_marker": _to_int(scaled_fp8_marker_value, None),
            "hp_filter": hp_filter_value.strip() if isinstance(hp_filter_value, str) else hp_filter_value,
            "full_precision_mm": bool(full_precision_mm_value),
            "remove_keys": remove_keys_value.strip() if isinstance(remove_keys_value, str) else remove_keys_value,
            "add_keys": add_keys_value.strip() if isinstance(add_keys_value, str) else add_keys_value,
            "quant_filter": quant_filter_value.strip() if isinstance(quant_filter_value, str) else quant_filter_value,
            "actcal_samples": _to_int(actcal_samples_value, None),
            "actcal_percentile": _to_float(actcal_percentile_value, None),
            "actcal_lora": actcal_lora_value.strip() if isinstance(actcal_lora_value, str) else actcal_lora_value,
            "actcal_seed": _to_int(actcal_seed_value, None),
            "actcal_device": actcal_device_value.strip() if isinstance(actcal_device_value, str) else actcal_device_value,
            "model_filters": filter_values,
        }

    def _collect_filter_values(*values) -> Dict[str, bool]:
        return {name: bool(val) for name, val in zip(filter_checkboxes.keys(), values)}

    preset_field_names = [
        "quant_format",
        "comfy_quant",
        "full_precision_matrix_mult",
        "scaling_mode",
        "block_size",
        "simple",
        "skip_inefficient_layers",
        "num_iter",
        "calib_samples",
        "optimizer",
        "lr_schedule",
        "lr",
        "top_p",
        "min_k",
        "max_k",
        "full_matrix",
        "scale_optimization",
        "scale_refinement_rounds",
        "manual_seed",
        "verbose",
    ]

    preset_field_components = [
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        simple,
        skip_inefficient_layers,
        num_iter,
        calib_samples,
        optimizer,
        lr_schedule,
        lr,
        top_p,
        min_k,
        max_k,
        full_matrix,
        scale_optimization,
        scale_refinement_rounds,
        manual_seed,
        verbose,
    ]

    def _apply_preset(preset_name: str):
        overrides = PRESET_OVERRIDES.get(preset_name, {})
        updates = []
        for name in preset_field_names:
            if name in overrides:
                updates.append(gr.update(value=overrides[name]))
            else:
                updates.append(gr.update())
        return updates

    preset_dropdown.change(
        fn=_apply_preset,
        inputs=[preset_dropdown],
        outputs=preset_field_components,
        show_progress=False,
    )

    def _apply_model_preset(selected: str):
        if selected == MODEL_PRESET_NONE:
            return (
                [gr.update() for _ in filter_checkboxes.values()]
                + [gr.update()]
                + [gr.update() for _ in preset_field_components]
                + [gr.update()]
            )

        model_settings = MODEL_PRESET_SETTINGS.get(selected, {})
        preset_name = model_settings.get("preset", PRESET_NORMAL)
        combined: Dict[str, object] = {}
        combined.update(PRESET_OVERRIDES.get(preset_name, {}))
        combined.update(model_settings)

        filter_updates = [
            gr.update(value=(name == selected))
            for name in filter_checkboxes.keys()
        ]
        preset_update = gr.update(value=preset_name)

        field_updates = []
        for name in preset_field_names:
            if name in combined:
                field_updates.append(gr.update(value=combined[name]))
            else:
                field_updates.append(gr.update())

        include_update = (
            gr.update(value=combined["include_input_scale"])
            if "include_input_scale" in combined
            else gr.update()
        )
        return filter_updates + [preset_update] + field_updates + [include_update]

    model_preset_dropdown.change(
        fn=_apply_model_preset,
        inputs=[model_preset_dropdown],
        outputs=list(filter_checkboxes.values()) + [preset_dropdown] + preset_field_components + [include_input_scale],
        show_progress=False,
    )

    def _update_workflow_visibility(selected: str):
        is_quantize = selected == WORKFLOW_QUANTIZE or selected == WORKFLOW_DRY_RUN
        return (
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
            gr.update(visible=is_quantize),
        )

    workflow.change(
        fn=_update_workflow_visibility,
        inputs=[workflow],
        outputs=[
            quant_format_group,
            model_filter_group,
            layer_mixing_group,
            optimization_group,
            advanced_lr_group,
            nvfp4_group,
            layer_config_group,
        ],
        show_progress=False,
    )

    single_input_button.click(
        fn=lambda current: get_file_path(current, ".safetensors", "Model files"),
        inputs=[single_input_file],
        outputs=[single_input_file],
        show_progress=False,
    )
    single_output_button.click(
        fn=lambda current: get_saveasfilename_path(
            current, extensions="*.safetensors", extension_name="Safetensors files"
        ),
        inputs=[single_output_file],
        outputs=[single_output_file],
        show_progress=False,
    )
    batch_input_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[batch_input_folder],
        outputs=[batch_input_folder],
        show_progress=False,
    )
    batch_output_button.click(
        fn=lambda current: get_folder_path(current),
        inputs=[batch_output_folder],
        outputs=[batch_output_folder],
        show_progress=False,
    )
    layer_config_button.click(
        fn=lambda current: get_file_path(current, ".json", "JSON files"),
        inputs=[layer_config_path],
        outputs=[layer_config_path],
        show_progress=False,
    )
    input_scales_button.click(
        fn=lambda current: get_file_path(current, ".json", "JSON or Safetensors"),
        inputs=[input_scales_path],
        outputs=[input_scales_path],
        show_progress=False,
    )
    tensor_scales_button.click(
        fn=lambda current: get_file_path(current, ".safetensors", "Safetensors files"),
        inputs=[tensor_scales_path],
        outputs=[tensor_scales_path],
        show_progress=False,
    )
    actcal_lora_button.click(
        fn=lambda current: get_file_path(current, ".safetensors", "Safetensors files"),
        inputs=[actcal_lora],
        outputs=[actcal_lora],
        show_progress=False,
    )

    single_inputs = [
        workflow,
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        include_input_scale,
        custom_layers,
        exclude_layers,
        custom_type,
        custom_block_size,
        custom_scaling_mode,
        custom_simple,
        custom_heur,
        fallback_type,
        fallback_block_size,
        fallback_simple,
        simple,
        skip_inefficient_layers,
        full_matrix,
        calib_samples,
        manual_seed,
        optimizer,
        num_iter,
        lr,
        lr_schedule,
        top_p,
        min_k,
        max_k,
        lr_gamma,
        lr_patience,
        lr_factor,
        lr_min,
        lr_cooldown,
        lr_threshold,
        lr_adaptive_mode,
        lr_shape_influence,
        lr_threshold_mode,
        early_stop_loss,
        early_stop_lr,
        early_stop_stall,
        scale_optimization,
        scale_refinement_rounds,
        input_scales_path,
        tensor_scales_path,
        layer_config_path,
        layer_config_fullmatch,
        dry_run,
        verbose,
        verbose_pinned,
        low_memory,
        save_quant_metadata,
        no_normalize_scales,
        scaled_fp8_marker,
        hp_filter,
        full_precision_mm,
        remove_keys,
        add_keys,
        quant_filter,
        actcal_samples,
        actcal_percentile,
        actcal_lora,
        actcal_seed,
        actcal_device,
    ] + list(filter_checkboxes.values())

    def _run_single(
        single_input_file_value,
        single_output_file_value,
        single_delete_original_value,
        *values,
    ):
        filter_count = len(filter_checkboxes)
        if filter_count:
            filter_values = _collect_filter_values(*values[-filter_count:])
            base_values = values[:-filter_count]
        else:
            filter_values = {}
            base_values = values
        params = _build_params_dict(
            *base_values,
            filter_values=filter_values,
        )
        return quantizer.run_single(
            input_file=single_input_file_value,
            output_file=single_output_file_value,
            delete_original=bool(single_delete_original_value),
            params=params,
        )

    single_run_button.click(
        fn=_run_single,
        inputs=[single_input_file, single_output_file, single_delete_original] + single_inputs,
        outputs=[single_status],
        show_progress=True,
    )

    single_cancel_button.click(
        fn=quantizer.cancel_single,
        inputs=[],
        outputs=[single_status],
        show_progress=False,
    )

    def _run_batch(
        batch_input_folder_value,
        batch_output_folder_value,
        batch_extensions_value,
        batch_recursive_value,
        batch_overwrite_value,
        batch_delete_original_value,
        *values,
    ):
        filter_count = len(filter_checkboxes)
        if filter_count:
            filter_values = _collect_filter_values(*values[-filter_count:])
            base_values = values[:-filter_count]
        else:
            filter_values = {}
            base_values = values
        params = _build_params_dict(
            *base_values,
            filter_values=filter_values,
        )
        return quantizer.run_batch(
            input_folder=batch_input_folder_value,
            output_folder=batch_output_folder_value,
            extensions=batch_extensions_value,
            recursive=bool(batch_recursive_value),
            overwrite_existing=bool(batch_overwrite_value),
            delete_original=bool(batch_delete_original_value),
            params=params,
        )

    batch_run_button.click(
        fn=_run_batch,
        inputs=[
            batch_input_folder,
            batch_output_folder,
            batch_extensions,
            batch_recursive,
            batch_overwrite,
            batch_delete_original,
        ] + single_inputs,
        outputs=[batch_status],
        show_progress=True,
    )

    batch_cancel_button.click(
        fn=quantizer.cancel_batch,
        inputs=[],
        outputs=[batch_status],
        show_progress=False,
    )

    settings_names = [
        "workflow",
        "quant_format",
        "comfy_quant",
        "full_precision_matrix_mult",
        "scaling_mode",
        "block_size",
        "include_input_scale",
        "custom_layers",
        "exclude_layers",
        "custom_type",
        "custom_block_size",
        "custom_scaling_mode",
        "custom_simple",
        "custom_heur",
        "fallback_type",
        "fallback_block_size",
        "fallback_simple",
        "simple",
        "skip_inefficient_layers",
        "full_matrix",
        "calib_samples",
        "manual_seed",
        "optimizer",
        "num_iter",
        "lr",
        "lr_schedule",
        "top_p",
        "min_k",
        "max_k",
        "lr_gamma",
        "lr_patience",
        "lr_factor",
        "lr_min",
        "lr_cooldown",
        "lr_threshold",
        "lr_adaptive_mode",
        "lr_shape_influence",
        "lr_threshold_mode",
        "early_stop_loss",
        "early_stop_lr",
        "early_stop_stall",
        "scale_optimization",
        "scale_refinement_rounds",
        "input_scales_path",
        "tensor_scales_path",
        "layer_config_path",
        "layer_config_fullmatch",
        "dry_run",
        "verbose",
        "verbose_pinned",
        "low_memory",
        "save_quant_metadata",
        "no_normalize_scales",
        "scaled_fp8_marker",
        "hp_filter",
        "full_precision_mm",
        "remove_keys",
        "add_keys",
        "quant_filter",
        "actcal_samples",
        "actcal_percentile",
        "actcal_lora",
        "actcal_seed",
        "actcal_device",
        "single_input_file",
        "single_output_file",
        "single_delete_original",
        "batch_input_folder",
        "batch_output_folder",
        "batch_extensions",
        "batch_recursive",
        "batch_overwrite",
        "batch_delete_original",
        "preset",
        "model_preset",
    ] + [f"filter.{name}" for name in filter_checkboxes.keys()]

    settings_components = [
        workflow,
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        include_input_scale,
        custom_layers,
        exclude_layers,
        custom_type,
        custom_block_size,
        custom_scaling_mode,
        custom_simple,
        custom_heur,
        fallback_type,
        fallback_block_size,
        fallback_simple,
        simple,
        skip_inefficient_layers,
        full_matrix,
        calib_samples,
        manual_seed,
        optimizer,
        num_iter,
        lr,
        lr_schedule,
        top_p,
        min_k,
        max_k,
        lr_gamma,
        lr_patience,
        lr_factor,
        lr_min,
        lr_cooldown,
        lr_threshold,
        lr_adaptive_mode,
        lr_shape_influence,
        lr_threshold_mode,
        early_stop_loss,
        early_stop_lr,
        early_stop_stall,
        scale_optimization,
        scale_refinement_rounds,
        input_scales_path,
        tensor_scales_path,
        layer_config_path,
        layer_config_fullmatch,
        dry_run,
        verbose,
        verbose_pinned,
        low_memory,
        save_quant_metadata,
        no_normalize_scales,
        scaled_fp8_marker,
        hp_filter,
        full_precision_mm,
        remove_keys,
        add_keys,
        quant_filter,
        actcal_samples,
        actcal_percentile,
        actcal_lora,
        actcal_seed,
        actcal_device,
        single_input_file,
        single_output_file,
        single_delete_original,
        batch_input_folder,
        batch_output_folder,
        batch_extensions,
        batch_recursive,
        batch_overwrite,
        batch_delete_original,
        preset_dropdown,
        model_preset_dropdown,
    ] + list(filter_checkboxes.values())

    def _save_configuration(action, save_as_bool, file_path, headless_value, print_only, *values):
        parameters = list(zip(settings_names, values))
        original_file_path = file_path
        if save_as_bool or not file_path:
            file_path = get_saveasfilename_path(
                file_path, extensions="*.toml", extension_name="TOML files"
            )
        if not file_path:
            return original_file_path, "Save cancelled"
        destination_directory = os.path.dirname(file_path)
        if destination_directory:
            os.makedirs(destination_directory, exist_ok=True)
        from .common_gui import SaveConfigFile
        SaveConfigFile(parameters=parameters, file_path=file_path, exclusion=["file_path", "save_as", "save_as_bool"])
        config_name = os.path.basename(file_path)
        return file_path, f"Saved: {config_name}"

    def _open_configuration(action, ask_for_file, file_path, headless_value, print_only, *values):
        original_file_path = file_path
        if ask_for_file:
            file_path = get_file_path_or_save_as(
                file_path, default_extension=".toml", extension_name="TOML files"
            )
        if not file_path:
            return [original_file_path, "Load cancelled"] + [gr.update() for _ in settings_components]
        if not os.path.isfile(file_path) and ask_for_file:
            return [file_path, f"New config: {os.path.basename(file_path)}"] + [gr.update() for _ in settings_components]
        if not os.path.isfile(file_path):
            return [original_file_path, "Config file not found"] + [gr.update() for _ in settings_components]
        try:
            data = toml.load(file_path)
        except Exception as exc:
            return [original_file_path, f"Failed to load: {exc}"] + [gr.update() for _ in settings_components]
        flat = _flatten_dict(data)
        values_out = []
        for name, current in zip(settings_names, values):
            value = flat.get(name)
            if value is None:
                value = flat.get(f"model_quantizer.{name}")
            if value is None:
                values_out.append(current)
            else:
                values_out.append(value)
        return [file_path, f"Loaded: {os.path.basename(file_path)}"] + values_out

    configuration.button_open_config.click(
        fn=_open_configuration,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_components,
        show_progress=False,
    )

    configuration.button_load_config.click(
        fn=_open_configuration,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_components,
        show_progress=False,
        queue=False,
    )

    configuration.button_save_config.click(
        fn=_save_configuration,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )

    configuration.config_file_name.change(
        fn=lambda config_name, *args: (
            _open_configuration("open_configuration", False, config_name, dummy_headless.value, False, *args)
            if config_name and config_name.endswith(".toml")
            else ([config_name, ""] + [gr.update() for _ in settings_components])
        ),
        inputs=[configuration.config_file_name] + settings_components,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_components,
        show_progress=False,
        queue=False,
    )
