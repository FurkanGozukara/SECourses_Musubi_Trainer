import gradio as gr
import os
import sys
import subprocess
import re
import json
import importlib.util
import copy
import threading
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
WORKFLOW_HYBRID_MXFP8 = "Hybrid MXFP8 (requires HybridMXFP8Layout)"
WORKFLOW_DRY_RUN = "Dry Run (Analyze / Create Template)"

QUANT_FORMAT_FP8 = "FP8 (E4M3)"
QUANT_FORMAT_INT8 = "INT8"
QUANT_FORMAT_NVFP4 = "NVFP4 (FP4 E2M1)"
QUANT_FORMAT_MXFP8 = "MXFP8 (Microscaling)"

PRESET_CUSTOM = "Custom (manual)"
PRESET_FAST = "Fast (Simple Quantization)"
PRESET_NORMAL = "Normal (Balanced)"
PRESET_HIGH = "High Quality (Slow)"
PRESET_EXTREME = "Extreme Quality (Very Slow)"
PRESET_INT8_FAST = "INT8 Blockwise (QuantOps / Experimental)"
PRESET_INT8_TENSOR = "INT8 Tensorwise (QuantOps custom / RTX 30xx+)"
PRESET_INT8_CONVROT = "INT8 ConvRot Rowwise (QuantOps custom / Best INT8)"
PRESET_MXFP8_BALANCED = "MXFP8 Balanced (QuantOps / Blackwell)"
PRESET_NVFP4_BALANCED = "NVFP4 Balanced (QuantOps / Blackwell)"
PRESET_NVFP4_Z = "NVFP4 Z-Image (Aggressive / Expert)"
PRESET_FP8_SCALED = "FP8 Scaled (Tensorwise)"
PRESET_FP8_MIXED = "FP8 Compatibility (Tensorwise)"

MODEL_PRESET_NONE = "None (manual)"
ERNIE_IMAGE_EXCLUDE_LAYERS = (
    r"(time_embedding|adaLN_modulation|final_linear|final_norm|x_embedder|"
    r"layers[.]0[.]self_attention|layers[.]0[.]mlp.gate_proj|"
    r"layers[.]0[.]mlp[.]up_proj|text_proj)"
)
BOOGU_EXCLUDE_LAYERS = r"(image_index_embedding|ref_image_patch_embedder)"
KREA2_EXCLUDE_LAYERS = (
    r"^(first|last|tmlp|tproj|txtmlp|img_in|final_layer|time_embed|time_mod_proj)([.]|$)|"
    r"^(txtfusion|text_fusion)[.]projector([.]|$)"
)
KREA2_LAYER_CONFIG_PATH = os.path.join(REPO_ROOT, "model_quantizer_presets", "krea2_fp8_layer_config.json")
KREA2_GENERATED_LAYER_CONFIG_DIR = os.path.join(REPO_ROOT, "model_quantizer_presets", "generated")
KREA2_LAYER_CONFIG_PATTERNS = (
    (
        r"(^|[.])attn[.](gate|wo|to_gate|to_out[.]0)$",
        {"full_precision_matrix_mult": True},
    ),
    (
        r"(^|[.])(mlp|ff)[.]down$",
        {"full_precision_matrix_mult": True},
    ),
    (
        r"(^|[.])attn[.](wq|wk|wv|to_q|to_k|to_v)$",
        {},
    ),
    (
        r"(^|[.])(mlp|ff)[.](gate|up)$",
        {},
    ),
)
FP8_ONLY_LAYER_CONFIG_PATHS = {
    os.path.normcase(os.path.abspath(KREA2_LAYER_CONFIG_PATH)),
}
NON_FP8_QUANT_FORMATS = {QUANT_FORMAT_INT8, QUANT_FORMAT_NVFP4, QUANT_FORMAT_MXFP8}


def _is_fp8_only_layer_config(path: object) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    try:
        normalized = os.path.normcase(os.path.abspath(os.path.expanduser(path.strip())))
    except (OSError, TypeError, ValueError):
        return False
    return normalized in FP8_ONLY_LAYER_CONFIG_PATHS


def _is_krea2_managed_layer_config(path: object) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    try:
        normalized = os.path.normcase(os.path.abspath(os.path.expanduser(path.strip())))
        generated_root = os.path.normcase(os.path.abspath(KREA2_GENERATED_LAYER_CONFIG_DIR))
    except (OSError, TypeError, ValueError):
        return False
    return normalized in FP8_ONLY_LAYER_CONFIG_PATHS or normalized.startswith(generated_root + os.sep)


def _krea2_fp8_format_for_scaling(scaling_mode: str) -> str:
    if scaling_mode == "row":
        return "float8_e4m3fn_rowwise"
    if scaling_mode in ("block", "block2d"):
        return "float8_e4m3fn_blockwise"
    if scaling_mode == "block3d":
        return "float8_e4m3fn_block3d"
    return "float8_e4m3fn"


def _krea2_int8_format_for_scaling(scaling_mode: str) -> str:
    if scaling_mode == "block":
        return "int8_blockwise"
    return "int8_tensorwise"


def _krea2_layer_config_settings(params: Dict[str, object]) -> Tuple[Dict[str, object], str]:
    quant_format = params.get("quant_format")
    scaling_mode = _coerce_scaling_mode_for_format(
        str(quant_format),
        params.get("scaling_mode", "tensor"),
    )
    block_size = _coerce_block_size_for_format(
        str(quant_format),
        scaling_mode,
        params.get("block_size"),
    )

    if quant_format == QUANT_FORMAT_INT8:
        fmt = _krea2_int8_format_for_scaling(scaling_mode)
        suffix = f"int8_{scaling_mode}"
    elif quant_format == QUANT_FORMAT_MXFP8:
        fmt = "mxfp8"
        suffix = "mxfp8"
    elif quant_format == QUANT_FORMAT_NVFP4:
        fmt = "nvfp4"
        suffix = "nvfp4"
    else:
        fmt = _krea2_fp8_format_for_scaling(scaling_mode)
        suffix = f"fp8_{scaling_mode}"

    settings: Dict[str, object] = {"format": fmt}
    if quant_format in (QUANT_FORMAT_FP8, QUANT_FORMAT_INT8) and scaling_mode:
        settings["scaling_mode"] = scaling_mode
    if block_size is not None and fmt in {"float8_e4m3fn_blockwise", "int8_blockwise"}:
        settings["block_size"] = int(block_size)

    return settings, suffix


def _write_krea2_layer_config_for_params(params: Dict[str, object]) -> str:
    base_settings, suffix = _krea2_layer_config_settings(params)
    config: Dict[str, Dict[str, object]] = {}
    use_precision_flags = params.get("quant_format") in {
        QUANT_FORMAT_FP8,
        QUANT_FORMAT_NVFP4,
        QUANT_FORMAT_MXFP8,
    }

    for pattern, extra_settings in KREA2_LAYER_CONFIG_PATTERNS:
        settings = dict(base_settings)
        if use_precision_flags:
            settings.update(extra_settings)
        config[pattern] = settings

    os.makedirs(KREA2_GENERATED_LAYER_CONFIG_DIR, exist_ok=True)
    path = os.path.join(KREA2_GENERATED_LAYER_CONFIG_DIR, f"krea2_{suffix}_layer_config.json")
    existing = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None
    if existing != config:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
    return path


def _clear_incompatible_model_settings(selected_model: str, settings: Dict[str, object]) -> None:
    if settings.get("quant_format") == QUANT_FORMAT_FP8:
        return
    if _is_fp8_only_layer_config(settings.get("layer_config_path")):
        settings["layer_config_path"] = ""
        settings["layer_config_fullmatch"] = False

OUTPUT_MODE_FULL = "Full (all logs)"
OUTPUT_MODE_COMPACT = "Compact (hide progress bars)"
OUTPUT_MODE_SUMMARY = "Summary only (warnings/errors)"
OUTPUT_MODE_CHOICES = [OUTPUT_MODE_FULL, OUTPUT_MODE_COMPACT, OUTPUT_MODE_SUMMARY]

OPTIMIZER_CHOICES = ["prodigy", "original", "adamw", "radam"]
OPTIMIZER_INFO = "Prodigy is convert_to_quant's current upstream default and requires prodigy-plus-schedule-free."
SAFE_RUNTIME_DEFAULTS = {
    "low_memory": True,
    "save_quant_metadata": True,
    "comfy_quant": True,
}
FORMAT_DEFAULTS = {
    QUANT_FORMAT_FP8: {
        "comfy_quant": True,
        "full_precision_matrix_mult": False,
        "scaling_mode": "tensor",
        "block_size": None,
        "convrot": False,
        "convrot_group_size": 256,
        "low_memory": True,
        "save_quant_metadata": True,
    },
    QUANT_FORMAT_INT8: {
        "comfy_quant": True,
        "full_precision_matrix_mult": False,
        "scaling_mode": "block",
        "block_size": 128,
        "convrot": False,
        "convrot_group_size": 256,
        "low_memory": True,
        "save_quant_metadata": True,
    },
    QUANT_FORMAT_MXFP8: {
        "comfy_quant": True,
        "full_precision_matrix_mult": False,
        "scaling_mode": "tensor",
        "block_size": None,
        "convrot": False,
        "convrot_group_size": 256,
        "low_memory": True,
        "save_quant_metadata": True,
    },
    QUANT_FORMAT_NVFP4: {
        "comfy_quant": True,
        "full_precision_matrix_mult": False,
        "scaling_mode": "tensor",
        "block_size": None,
        "convrot": False,
        "convrot_group_size": 256,
        "low_memory": True,
        "save_quant_metadata": True,
    },
}
MANUAL_QUANT_DEFAULTS = {
    "calib_samples": 3072,
    "optimizer": "prodigy",
    "num_iter": 4000,
    "lr": 1.0,
    "lr_schedule": "plateau",
    "top_p": 0.2,
    "min_k": 256,
    "max_k": 1280,
    "lr_gamma": 0.99,
    "lr_patience": 1,
    "lr_factor": 0.95,
    "lr_min": 1e-8,
    "lr_cooldown": 0,
    "lr_threshold": 0.0,
    "lr_adaptive_mode": "simple-reset",
    "lr_shape_influence": 1.0,
    "lr_threshold_mode": "rel",
    "early_stop_loss": 5e-9,
    "early_stop_lr": 1.01e-8,
    "early_stop_stall": 2000,
}


def _has_prodigy_optimizer() -> bool:
    return importlib.util.find_spec("prodigyplus") is not None


def _load_model_filters() -> Dict[str, Dict[str, object]]:
    import_errors = []
    for import_path in (
        "convert_to_quant.constants",
        "convert_to_quant.convert_to_quant.constants",
    ):
        try:
            module = __import__(import_path, fromlist=["MODEL_FILTERS"])
            return getattr(module, "MODEL_FILTERS")
        except Exception as exc:
            import_errors.append(f"{import_path}: {exc}")

    log.warning("Could not import convert_to_quant MODEL_FILTERS: %s", " | ".join(import_errors))
    return {
        "qwen35": {"help": "Qwen2.5 text/multimodal model", "category": "text"},
        "t5xxl": {"help": "T5-XXL text encoder", "category": "text"},
        "mistral": {"help": "Mistral text encoder", "category": "text"},
        "visual": {"help": "Visual encoder", "category": "text"},
        "generic_text": {"help": "Generic text encoder", "category": "text"},
        "anima": {"help": "Anima diffusion model", "category": "diffusion"},
        "lens": {"help": "LENS diffusion model", "category": "diffusion"},
        "flux1": {"help": "Flux.1 diffusion", "category": "diffusion"},
        "flux2": {"help": "Flux.2 diffusion", "category": "diffusion"},
        "flux_klein": {"help": "FLUX.2 Klein diffusion", "category": "diffusion"},
        "distillation_large": {"help": "Chroma distilled (large)", "category": "diffusion"},
        "distillation_small": {"help": "Chroma distilled (small)", "category": "diffusion"},
        "nerf_large": {"help": "NeRF (large)", "category": "diffusion"},
        "nerf_small": {"help": "NeRF (small)", "category": "diffusion"},
        "radiance": {"help": "Radiance diffusion", "category": "diffusion"},
        "wan": {"help": "WAN video model", "category": "video"},
        "hunyuan": {"help": "Hunyuan video model", "category": "video"},
        "ltx2": {"help": "LTX v2 / v2.3 video model", "category": "video"},
        "ltx2_3": {"help": "LTX 2.3 video model", "category": "video"},
        "ltxv2": {"help": "LTXv2 video model", "category": "video"},
        "qwen": {"help": "Qwen Image", "category": "image"},
        "ernie_image": {"help": "ERNIE Image diffusion transformer", "category": "image"},
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

MODEL_PRESET_DISPLAY_NAMES = {
    "anima": "Anima (Base/Turbo)",
    "boogu": "Boogu-Image",
    "ernie_image": "ERNIE Image",
    "flux1": "FLUX.1",
    "flux2": "FLUX.2",
    "flux_klein": "FLUX 2 Klein Models",
    "generic_text": "Generic Text Encoder",
    "hunyuan": "Hunyuan Video 1.5",
    "krea2": "Krea 2 (Raw/Turbo)",
    "lens": "Microsoft LENS",
    "ltxv2": "LTX (2 / 2.3)",
    "ltx2": "LTX (2 / 2.3)",
    "ltx2_3": "LTX (2 / 2.3)",
    "qwen": "Qwen Image / Edit (2509, 2511, 2512)",
    "qwen35": "Qwen2.5 Text/Multimodal",
    "t5xxl": "T5-XXL",
    "wan": "WAN (2.1 / 2.2)",
    "zimage": "Z-Image",
    "zimage_refiner": "Z-Image Refiner",
}

MODEL_PRESET_LEGACY_ALIASES = {
    "flux2_klein_9b": "flux_klein",
    "flux2_klein_9b_kv": "flux_klein",
    "flux2_klein_4b": "flux_klein",
}
MODEL_PRESET_LEGACY_LABELS = {
    "Anima": "anima",
    "Boogu": "boogu",
    "FLUX.2 Klein": "flux_klein",
    "FLUX.2-klein-9B": "flux_klein",
    "FLUX.2-klein-9b-kv": "flux_klein",
    "FLUX.2-klein-4B": "flux_klein",
    "LENS": "lens",
    "LTX_2_and_2.3": "ltxv2",
    "Qwen Image / Edit": "qwen",
    "WAN Video": "wan",
}
REGEX_ONLY_MODEL_PRESETS = {"boogu", "ernie_image", "krea2"}
GUI_ONLY_MODEL_PRESETS = {"flux1", "flux_klein"}
PRIMARY_MODEL_PRESET_VALUES = {
    "anima",
    "boogu",
    "ernie_image",
    "flux1",
    "flux2",
    "hunyuan",
    "krea2",
    "lens",
    "ltxv2",
    "qwen",
    "wan",
    "zimage",
    "zimage_refiner",
}
MODEL_PRESET_FILTER_ALIASES = {}
MODEL_PRESET_EXTRA_FILTERS = {
    # Keep combined filter behavior visible in Gradio instead of relying on CLI-side aliases.
    "qwen35": {"generic_text"},
}

FLUX_KLEIN_MODEL_SETTINGS = {
    "preset": PRESET_NORMAL,
    "quant_format": QUANT_FORMAT_FP8,
    "comfy_quant": True,
    "scaling_mode": "tensor",
}

SCALING_MODE_INFO = (
    "Quality/compatibility tradeoff. Tensor uses one scale for the whole weight tensor: lowest scale overhead and "
    "usually the simplest, fastest, most compatible path, but least adaptive when channels or regions have very "
    "different ranges. Row uses one scale per output row: usually better than tensor for uneven per-channel ranges, "
    "with modest scale overhead; for INT8 row this is the ConvRot route and needs matching runtime support. "
    "Block uses one scale per 2D block: usually the best quality of these modes on uneven weights, especially for "
    "INT8 blockwise, but it adds scale metadata and requires dimensions divisible by the block size. Some models "
    "are layer-sensitive; for FP8 runs, the Krea 2 preset keeps its main attention and MLP projections on FP8 tensor scaling. "
    "NVFP4 and MXFP8 use fixed internal block microscaling, so their generic Scaling Mode control is locked to tensor."
)

BLOCK_SIZE_INFO = (
    "Only used for block-wise quantization. Smaller blocks mean more local scales and usually better accuracy, "
    "but more scale overhead and sometimes more memory/runtime cost. Larger blocks reduce scale overhead, but usually "
    "reduce quality. Common starting points: FP8 block-wise 64, INT8 block-wise 128. INT8 block-wise layers must "
    "also be divisible by the chosen block size."
)

SCALING_MODE_CHOICES = ["tensor", "row", "block"]
CUSTOM_SCALING_MODE_CHOICES = [None] + SCALING_MODE_CHOICES
FIXED_SCALING_QUANT_FORMATS = {QUANT_FORMAT_NVFP4, QUANT_FORMAT_MXFP8}
FORMAT_SCALING_MODE_CHOICES = {
    QUANT_FORMAT_FP8: SCALING_MODE_CHOICES,
    QUANT_FORMAT_INT8: SCALING_MODE_CHOICES,
    QUANT_FORMAT_NVFP4: ["tensor"],
    QUANT_FORMAT_MXFP8: ["tensor"],
}
QUANT_TYPE_TO_FORMAT = {
    "fp8": QUANT_FORMAT_FP8,
    "int8": QUANT_FORMAT_INT8,
    "nvfp4": QUANT_FORMAT_NVFP4,
    "mxfp8": QUANT_FORMAT_MXFP8,
}


def _normalize_scaling_mode(value, default="tensor"):
    if value in ("block2d", "block3d"):
        return "block"
    if value in SCALING_MODE_CHOICES:
        return value
    return default


def _scaling_mode_choices_for_format(selected_format: str) -> List[str]:
    return FORMAT_SCALING_MODE_CHOICES.get(selected_format, SCALING_MODE_CHOICES)


def _coerce_scaling_mode_for_format(selected_format: str, value, default: str = "tensor") -> str:
    choices = _scaling_mode_choices_for_format(selected_format)
    normalized = _normalize_scaling_mode(value, default)
    return normalized if normalized in choices else choices[0]


def _uses_block_size(selected_format: str, selected_scaling: str) -> bool:
    if selected_format in FIXED_SCALING_QUANT_FORMATS:
        return False
    return selected_scaling == "block" or (
        selected_format == QUANT_FORMAT_INT8 and selected_scaling == "row"
    )


def _coerce_block_size_for_format(selected_format: str, selected_scaling: str, value):
    if not _uses_block_size(selected_format, selected_scaling):
        return None
    return _to_int(value, None)


def _scaling_mode_choices_for_quant_type(quant_type: Optional[str]) -> List[object]:
    selected_format = QUANT_TYPE_TO_FORMAT.get(quant_type)
    if selected_format:
        return _scaling_mode_choices_for_format(selected_format)
    return [None]


def _coerce_optional_scaling_mode_for_quant_type(quant_type: Optional[str], value):
    selected_format = QUANT_TYPE_TO_FORMAT.get(quant_type)
    if not selected_format:
        return None
    return _coerce_scaling_mode_for_format(selected_format, value or "tensor")


def _normalize_optional_scaling_mode(value):
    if value is None or value == "":
        return None
    return _normalize_scaling_mode(value, None)

CALIB_SAMPLES_INFO = (
    "Used here for bias-correction calibration, not dataset calibration. The tool generates random inputs and uses "
    "them to estimate output bias shift after quantization. More samples usually make the correction more stable, "
    "but cost more time and memory. Diminishing returns usually start after a few thousand samples."
)

ACTCAL_SAMPLES_INFO = (
    "Used only by activation calibration (actcal) to estimate input_scale values. More samples can improve stability, "
    "but increase calibration time and memory use."
)

INPUT_SCALE_INFO = (
    "Writes per-layer input_scale tensors to the output model. This is mostly for loader compatibility; for many "
    "FP8/INT8 exports it is just 1.0, but some text-encoder and calibrated paths can use real non-default values."
)

FULL_MATRIX_INFO = (
    "Uses full SVD instead of the faster low-rank SVD approximation. This can improve quality or stability on some "
    "sensitive layers, but it is much slower and uses much more memory."
)

BIAS_CORRECTION_PANEL_MD = """
### How bias correction works

Quantization changes the weights, so a layer's output can shift a little away from the original BF16/FP16 behavior.
Bias correction tries to cancel that average output shift.

The quantizer does this:

1. Generate synthetic calibration inputs `X`
2. Run the original layer and the quantized-dequantized layer
3. Measure the output difference per channel
4. Average that difference across samples
5. Add that average correction back into the bias

In simplified form:

```text
Y_original = X @ W_original^T
Y_quant    = X @ W_dequantized^T
delta      = mean(Y_original - Y_quant, dim=0)
b_new      = b_original + delta
```

What this helps with:
- Reduces average output offset after quantization
- Makes the quantized layer behave closer to the original layer

What this does not do:
- It does not restore the original weights
- It does not fix all quantization error
- It mainly fixes the average per-channel drift

Practical note:
- More `Calibration Samples` usually makes the correction estimate more stable, but also increases time and memory use.
"""

QUALITY_GUIDANCE_MD = """
### Default route

The default preset is tuned to avoid OOM first: simple FP8, low-memory streaming, and quant metadata enabled.
This is the safest starting point for very large checkpoints or smaller GPUs.

### Higher quality route

Use these changes when you can spend more time and memory for better fidelity:

1. Pick the closest Model Preset first so sensitive layers stay high precision.
2. Change Quality Preset from Fast to FP8 Scaled, FP8 Compatibility, Normal, High Quality, or Extreme Quality.
3. Disable Simple quantization to enable learned rounding.
4. Keep Save Quantization Metadata and Low memory mode enabled unless you have a reason to turn them off.
5. Raise Calibration Samples gradually, for example 2048, 3072, then 4096.
6. Use Extreme Quality only for difficult layers or final high-quality runs, because Full SVD is much heavier.
7. For unsupported models, run Dry Run / Create Template and add regex exclusions for remaining 2D BF16-sensitive layers.

Model-specific notes from upstream issues:

- Qwen Image / Edit (2509, 2511, 2512): use FP8 Scaled or Compatibility, keep full precision matrix multiplication enabled, and do not use Simple for best quality.
- FLUX.1: use the FLUX.1 preset for the modern equivalent of Comfy's scaled-FP8 reference: FP8 E4M3 tensor scaling, `.comfy_quant`, quant metadata, low-memory, full precision matrix multiplication, and input_scale tensors. The old public Comfy checkpoint uses legacy `scaled_fp8` tensors without metadata; new exports should keep metadata enabled.
- Z-Image and Anima (Base/Turbo): avoid NVFP4 as the default route; issue reports showed noisy output. Use FP8 Compatibility first.
- Boogu-Image: use the Boogu-Image preset, which applies the known exclusion regex for image/reference embedding layers.
- ERNIE Image: use the ERNIE preset, which applies the tested exclusion regex.
- Krea 2 Raw/Turbo: the Krea 2 preset starts on Normal (Balanced): learned FP8 tensor scaling, Krea-specific projection/time/final-layer exclusions, metadata, and low-memory. Local SwarmUI tests showed Krea 2 FP8 tensor output stayed good while FP8 blockwise and INT8 blockwise degraded when Q/K/V and MLP gate/up were block-quantized. The GUI applies the same Krea 2 attention/MLP regex scope for FP8, INT8, MXFP8, and NVFP4 by generating a matching layer config at run time; FP8 keeps full-precision matrix multiplication only on gate/output/down projections. Switch Quality Preset to Fast only when you specifically want the simpler official-style FP8 route.
- ComfyUI-QuantOps: load these exports with QuantOps quantized loader nodes or a patched QuantOps stock-loader auto integration. Keep metadata enabled so QuantOps can identify tensor, row, block, MXFP8, and NVFP4 layouts instead of guessing from scales.
- INT8 Blockwise requires QuantOps runtime support. Stock ComfyUI `Load Diffusion Model` and SwarmUI's normal model dropdown need a QuantOps stock-loader auto patch; otherwise use the QuantOps quantized loader nodes.
- INT8 Tensorwise and ConvRot require the custom comfy-kitchen INT8 build plus `--enable-triton-backend`; without that, expect fallback behavior or load failure.
- QuantOps loader nodes have their own `low_memory` and `disable_dynamic` toggles. For quantized text encoders, upstream issue comments recommend enabling both when the pipeline clogs or memory spikes.
- INT8 text encoders, especially T5, have reported Triton NaNs on short prompts. If that happens, use the QuantOps loader with the PyTorch backend or stay with FP8 for text encoders.
- RTX 30xx and 40xx should prefer FP8/INT8 routes. MXFP8 and NVFP4 are Blackwell-oriented expert presets; treat them as compatibility/quality experiments unless your ComfyUI runtime has the required comfy-kitchen layouts.
- On older cards, enabling Triton for FP8 may trip unsupported FP8 dtype compilation errors in unrelated non-QuantOps workflows. Disable the Triton backend for those workflows if that happens.
- Hybrid MXFP8 needs `HybridMXFP8Layout` from comfy-kitchen. If that layout is not registered in ComfyUI, use MXFP8 Balanced or FP8 instead.
- Blockwise FP8/INT8 require layer dimensions compatible with their block size. Tensor/row scaling is safer for broad model coverage and lower risk of unsupported layers.
- Published third-party Krea 2 quantized checkpoints exist, but their visual quality is not independently validated here.
"""

MODEL_PRESET_VALUE_BY_LABEL = {
    label: key for key, label in MODEL_PRESET_DISPLAY_NAMES.items()
}
MODEL_PRESET_VALUE_BY_LABEL.update(MODEL_PRESET_LEGACY_LABELS)
MODEL_PRESET_VALUE_BY_LABEL["LTX (2 / 2.3)"] = "ltxv2"
MODEL_PRESET_VALUE_BY_LABEL["LTX_2_and_2.3"] = "ltxv2"

MODEL_PRESET_FIELD = "model_preset"
MODEL_PRESET_PRIMARY_FIELD = "model_preset_primary"
MODEL_PRESET_OTHER_FIELD = "model_preset_other"


def _model_preset_label(value: str) -> str:
    value = MODEL_PRESET_VALUE_BY_LABEL.get(value, value)
    value = MODEL_PRESET_LEGACY_ALIASES.get(value, value)
    return MODEL_PRESET_DISPLAY_NAMES.get(value, value)


def _model_preset_value(value: str) -> str:
    value = MODEL_PRESET_VALUE_BY_LABEL.get(value, value)
    return MODEL_PRESET_LEGACY_ALIASES.get(value, value)


def _model_preset_filter(value: str) -> str:
    if value in REGEX_ONLY_MODEL_PRESETS:
        return ""
    return MODEL_PRESET_FILTER_ALIASES.get(value, value)


def _model_preset_filters(value: str) -> set:
    selected_value = _model_preset_value(value)
    filters = set()
    primary_filter = _model_preset_filter(selected_value)
    if primary_filter:
        filters.add(primary_filter)
    filters.update(MODEL_PRESET_EXTRA_FILTERS.get(selected_value, set()))
    return {name for name in filters if name in MODEL_FILTERS}


def _model_preset_sort_key(label: str) -> str:
    return label.casefold()


def _ordered_model_preset_choices(values) -> List[str]:
    labels = {
        _model_preset_label(value)
        for value in values
        if _model_preset_label(value) != MODEL_PRESET_NONE
    }
    labels.add(MODEL_PRESET_NONE)
    return sorted(labels, key=_model_preset_sort_key)


def _all_model_preset_values():
    return set(MODEL_FILTERS.keys()) | REGEX_ONLY_MODEL_PRESETS | GUI_ONLY_MODEL_PRESETS


MODEL_PRESET_PRIMARY_CHOICES = _ordered_model_preset_choices(PRIMARY_MODEL_PRESET_VALUES)
MODEL_PRESET_OTHER_CHOICES = _ordered_model_preset_choices(
    value
    for value in _all_model_preset_values()
    if _model_preset_label(value) not in MODEL_PRESET_PRIMARY_CHOICES
)
MODEL_PRESET_CHOICES = _ordered_model_preset_choices(_all_model_preset_values())


def _split_model_preset_selection(value: str):
    selected_value = _model_preset_value(value)
    if selected_value == MODEL_PRESET_NONE:
        return MODEL_PRESET_NONE, MODEL_PRESET_NONE, MODEL_PRESET_NONE

    label = _model_preset_label(selected_value)
    if label in MODEL_PRESET_PRIMARY_CHOICES:
        return label, MODEL_PRESET_NONE, selected_value
    if label in MODEL_PRESET_OTHER_CHOICES:
        return MODEL_PRESET_NONE, label, selected_value
    return MODEL_PRESET_NONE, MODEL_PRESET_NONE, MODEL_PRESET_NONE


def _model_preset_component_value(field_name: str, raw_value: str, current_value):
    primary_value, other_value, effective_value = _split_model_preset_selection(raw_value)
    if field_name == MODEL_PRESET_FIELD:
        return effective_value
    if field_name == MODEL_PRESET_PRIMARY_FIELD:
        return primary_value
    if field_name == MODEL_PRESET_OTHER_FIELD:
        return other_value
    return current_value


def _visible_model_preset_value(primary_value: str, other_value: str) -> str:
    primary_value = _model_preset_value(primary_value)
    other_value = _model_preset_value(other_value)
    if primary_value != MODEL_PRESET_NONE:
        return primary_value
    if other_value != MODEL_PRESET_NONE:
        return other_value
    return MODEL_PRESET_NONE

MODEL_PRESET_SETTINGS = {
    name: {
        "preset": PRESET_NORMAL,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
        "exclude_layers": "",
    }
    for name in MODEL_FILTERS.keys()
}
MODEL_PRESET_SETTINGS.update({
    "flux1": {
        "preset": PRESET_FP8_SCALED,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "full_precision_matrix_mult": True,
        "include_input_scale": True,
        "low_memory": True,
        "save_quant_metadata": True,
    },
    "flux_klein": FLUX_KLEIN_MODEL_SETTINGS.copy(),
    "t5xxl": {
        "preset": PRESET_NORMAL,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
        "include_input_scale": True,
    },
    "wan": {
        "preset": PRESET_FP8_SCALED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
    "qwen": {
        "preset": PRESET_FP8_SCALED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
        "simple": False,
        "full_precision_matrix_mult": True,
    },
    "boogu": {
        "preset": PRESET_INT8_FAST,
        "quant_format": QUANT_FORMAT_INT8,
        "comfy_quant": True,
        "scaling_mode": "block",
        "block_size": 128,
        "exclude_layers": BOOGU_EXCLUDE_LAYERS,
        "custom_type": None,
        "custom_block_size": None,
        "custom_scaling_mode": None,
        "custom_simple": False,
        "custom_heur": False,
        "fallback_type": None,
        "fallback_block_size": None,
        "fallback_simple": False,
    },
    "krea2": {
        "preset": PRESET_NORMAL,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "full_precision_matrix_mult": False,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "exclude_layers": KREA2_EXCLUDE_LAYERS,
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "convrot": False,
        "convrot_group_size": 256,
        "low_memory": True,
        "save_quant_metadata": True,
    },
    "ernie_image": {
        "preset": PRESET_FP8_SCALED,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "exclude_layers": ERNIE_IMAGE_EXCLUDE_LAYERS,
    },
    "anima": {
        "preset": PRESET_FP8_MIXED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
    "zimage": {
        "preset": PRESET_FP8_MIXED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
    "zimage_refiner": {
        "preset": PRESET_FP8_MIXED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
    "ltxv2": {
        "preset": PRESET_FP8_MIXED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
    "ltx2": {
        "preset": PRESET_FP8_MIXED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
    "ltx2_3": {
        "preset": PRESET_FP8_MIXED,
        "quant_format": QUANT_FORMAT_FP8,
        "scaling_mode": "tensor",
    },
})

PRESET_OVERRIDES = {
    PRESET_FAST: {
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": True,
        "skip_inefficient_layers": False,
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
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 2000,
        "calib_samples": 2048,
        "top_p": 0.12,
        "min_k": 128,
        "max_k": 896,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_HIGH: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 6000,
        "calib_samples": 4096,
        "top_p": 0.2,
        "min_k": 256,
        "max_k": 1536,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_EXTREME: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 6000,
        "calib_samples": 4096,
        "top_p": 0.2,
        "min_k": 256,
        "max_k": 1536,
        "full_matrix": True,
        "full_precision_matrix_mult": True,
    },
    PRESET_FP8_SCALED: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 3000,
        "calib_samples": 3072,
        "top_p": 0.16,
        "min_k": 128,
        "max_k": 1024,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_FP8_MIXED: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_FP8,
        "comfy_quant": True,
        "fallback_type": None,
        "fallback_block_size": None,
        "fallback_simple": False,
        "scaling_mode": "tensor",
        "block_size": None,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 2500,
        "calib_samples": 3072,
        "top_p": 0.14,
        "min_k": 128,
        "max_k": 1024,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_INT8_FAST: {
        "quant_format": QUANT_FORMAT_INT8,
        "comfy_quant": True,
        "scaling_mode": "block",
        "block_size": 128,
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "simple": True,
        "skip_inefficient_layers": False,
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
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "simple": True,
        "skip_inefficient_layers": False,
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
    PRESET_INT8_CONVROT: {
        "quant_format": QUANT_FORMAT_INT8,
        "comfy_quant": True,
        "scaling_mode": "row",
        "block_size": 128,
        "convrot": True,
        "convrot_group_size": 256,
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "simple": True,
        "skip_inefficient_layers": False,
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
    PRESET_MXFP8_BALANCED: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_MXFP8,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 3000,
        "calib_samples": 3072,
        "top_p": 0.2,
        "min_k": 128,
        "max_k": 1280,
        "full_matrix": False,
        "full_precision_matrix_mult": True,
    },
    PRESET_NVFP4_BALANCED: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_NVFP4,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "simple": False,
        "skip_inefficient_layers": True,
        "num_iter": 4000,
        "calib_samples": 4096,
        "top_p": 0.2,
        "min_k": 128,
        "max_k": 1280,
        "scale_optimization": "iterative",
        "scale_refinement_rounds": 1,
        "full_precision_matrix_mult": True,
    },
    PRESET_NVFP4_Z: {
        **MANUAL_QUANT_DEFAULTS,
        "quant_format": QUANT_FORMAT_NVFP4,
        "comfy_quant": True,
        "scaling_mode": "tensor",
        "block_size": None,
        "layer_config_path": "",
        "layer_config_fullmatch": False,
        "simple": False,
        "skip_inefficient_layers": False,
        "num_iter": 12000,
        "calib_samples": 8192,
        "top_p": 0.2,
        "min_k": 256,
        "max_k": 2048,
        "scale_optimization": "iterative",
        "scale_refinement_rounds": 1,
        "manual_seed": 42,
        "verbose": "NORMAL",
        "full_precision_matrix_mult": True,
    },
}

PRESET_BASE_SETTINGS = {
    **MANUAL_QUANT_DEFAULTS,
    "quant_format": QUANT_FORMAT_FP8,
    "comfy_quant": SAFE_RUNTIME_DEFAULTS["comfy_quant"],
    "full_precision_matrix_mult": False,
    "scaling_mode": "tensor",
    "block_size": None,
    "convrot": False,
    "convrot_group_size": 256,
    "exclude_layers": "",
    "custom_type": None,
    "custom_block_size": None,
    "custom_scaling_mode": None,
    "custom_simple": False,
    "custom_heur": False,
    "fallback_type": None,
    "fallback_block_size": None,
    "fallback_simple": False,
    "simple": False,
    "skip_inefficient_layers": False,
    "full_matrix": False,
    "scale_optimization": "fixed",
    "scale_refinement_rounds": 1,
    "layer_config_path": "",
    "layer_config_fullmatch": False,
    "manual_seed": -1,
    "verbose": "NORMAL",
    "low_memory": SAFE_RUNTIME_DEFAULTS["low_memory"],
    "save_quant_metadata": SAFE_RUNTIME_DEFAULTS["save_quant_metadata"],
}

for _preset_settings in PRESET_OVERRIDES.values():
    _preset_settings.setdefault("low_memory", SAFE_RUNTIME_DEFAULTS["low_memory"])
    _preset_settings.setdefault("save_quant_metadata", SAFE_RUNTIME_DEFAULTS["save_quant_metadata"])
    _preset_settings.setdefault("comfy_quant", SAFE_RUNTIME_DEFAULTS["comfy_quant"])
    _preset_settings.setdefault("convrot", False)
    _preset_settings.setdefault("convrot_group_size", 256)


def _combined_preset_settings(
    model_preset_value: str,
    preset_name: Optional[str],
    *,
    use_model_default_preset: bool = False,
) -> Tuple[str, Dict[str, object]]:
    selected_model = _model_preset_value(model_preset_value)
    model_settings = MODEL_PRESET_SETTINGS.get(selected_model, {})
    if use_model_default_preset and model_settings.get("preset"):
        effective_preset = str(model_settings["preset"])
    else:
        effective_preset = preset_name or model_settings.get("preset") or PRESET_NORMAL

    if effective_preset == PRESET_CUSTOM and not use_model_default_preset:
        return effective_preset, {}

    preset_overrides = PRESET_OVERRIDES.get(effective_preset, {})
    model_overrides = {name: value for name, value in model_settings.items() if name != "preset"}

    combined: Dict[str, object] = dict(PRESET_BASE_SETTINGS)
    if preset_overrides.get("quant_format") in NON_FP8_QUANT_FORMATS:
        combined.update(model_overrides)
        combined.update(preset_overrides)
    else:
        combined.update(preset_overrides)
        combined.update(model_overrides)

    _clear_incompatible_model_settings(selected_model, combined)
    return effective_preset, combined


def _to_int(value, default: Optional[int]) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _to_optional_positive_int(value, default: Optional[int] = None) -> Optional[int]:
    parsed = _to_int(value, default)
    if parsed is None:
        return default
    return parsed if parsed > 0 else default


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
        self.single_queue_lock = threading.RLock()
        self.single_queue: List[Dict[str, object]] = []
        self.single_current_job: Optional[Dict[str, object]] = None
        self.single_worker_thread: Optional[threading.Thread] = None
        self.single_job_counter = 0
        self.single_status_text = "Ready."

    def _format_single_job(self, job: Dict[str, object]) -> str:
        input_file = str(job.get("input_file") or "")
        output_path = str(job.get("output_path") or "")
        input_name = os.path.basename(input_file) or input_file
        output_name = os.path.basename(output_path) or output_path
        if output_name:
            return f"#{job.get('id')} {input_name} -> {output_name}"
        return f"#{job.get('id')} {input_name}"

    def _single_queue_text_locked(self) -> str:
        lines: List[str] = []
        if self.single_current_job is not None:
            lines.append(f"Running: {self._format_single_job(self.single_current_job)}")
        else:
            lines.append("Running: none")

        if self.single_queue:
            lines.append("")
            lines.append(f"Queued: {len(self.single_queue)}")
            for index, job in enumerate(self.single_queue, start=1):
                lines.append(f"{index}. {self._format_single_job(job)}")
        else:
            lines.append("")
            lines.append("Queued: none")
        return "\n".join(lines)

    def single_queue_text(self) -> str:
        with self.single_queue_lock:
            return self._single_queue_text_locked()

    def single_queue_and_status(self) -> Tuple[str, str]:
        with self.single_queue_lock:
            return self._single_queue_text_locked(), self.single_status_text

    def _set_single_status(self, text: str) -> None:
        with self.single_queue_lock:
            self.single_status_text = self._tail_text(text)

    def _append_single_status(self, text: str) -> None:
        with self.single_queue_lock:
            if self.single_status_text and self.single_status_text != "Ready.":
                combined = f"{self.single_status_text}\n{text}"
            else:
                combined = text
            self.single_status_text = self._tail_text(combined)

    def _resolve_python(self) -> List[str]:
        venv_python = os.path.join(REPO_ROOT, "venv", "Scripts", "python.exe")
        if os.path.isfile(venv_python):
            return [venv_python]
        return [sys.executable]

    def _base_cmd(self) -> List[str]:
        return self._resolve_python() + [
            "-m",
            "convert_to_quant.convert_to_quant.cli.main",
        ]

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

    def cancel_single(self) -> Tuple[str, str]:
        with self.single_queue_lock:
            queued_count = len(self.single_queue)
            self.single_queue.clear()
            running = self._is_running(self.single_process)
            self.single_cancel_requested = True

        if running:
            if self._terminate_process(self.single_process):
                message = f"Cancellation requested. Cleared {queued_count} queued conversion(s) and stopped the running conversion."
            else:
                message = f"Cleared {queued_count} queued conversion(s), but the running conversion may have already finished."
        elif queued_count:
            message = f"Cancelled {queued_count} queued conversion(s)."
        else:
            message = "No single-file conversion is running or queued."

        self._set_single_status(message)
        return self.single_queue_and_status()

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
        params = copy.deepcopy(params)
        self._apply_managed_layer_config(params)
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
        if params.get("convrot"):
            cmd.append("--convrot")
            if params.get("convrot_group_size") is not None:
                cmd += ["--convrot-group-size", str(params.get("convrot_group_size"))]

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
        if isinstance(params.get("custom_block_size"), int) and params.get("custom_block_size") > 0:
            cmd += ["--custom-block-size", str(params.get("custom_block_size"))]
        if params.get("custom_scaling_mode"):
            cmd += ["--custom-scaling-mode", str(params.get("custom_scaling_mode"))]
        if params.get("custom_simple"):
            cmd.append("--custom-simple")
        if params.get("custom_heur"):
            cmd.append("--custom-heur")

        if params.get("fallback_type"):
            cmd += ["--fallback", str(params.get("fallback_type"))]
        if isinstance(params.get("fallback_block_size"), int) and params.get("fallback_block_size") > 0:
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

    def _apply_managed_layer_config(self, params: Dict[str, object]) -> None:
        if params.get("workflow") != WORKFLOW_QUANTIZE:
            return

        model_preset = _model_preset_value(str(params.get(MODEL_PRESET_FIELD) or MODEL_PRESET_NONE))
        layer_config_path = params.get("layer_config_path")

        if model_preset == "krea2":
            if not layer_config_path or _is_krea2_managed_layer_config(layer_config_path):
                params["layer_config_path"] = _write_krea2_layer_config_for_params(params)
                params["layer_config_fullmatch"] = False
            return

        if params.get("quant_format") != QUANT_FORMAT_FP8 and _is_fp8_only_layer_config(layer_config_path):
            params["layer_config_path"] = ""
            params["layer_config_fullmatch"] = False

    def _validate_quantization_params(self, params: Dict[str, object]) -> Optional[str]:
        if params.get("workflow") != WORKFLOW_QUANTIZE:
            return None
        model_preset = _model_preset_value(str(params.get(MODEL_PRESET_FIELD) or MODEL_PRESET_NONE))
        layer_config_path = params.get("layer_config_path")
        if (
            params.get("quant_format") != QUANT_FORMAT_FP8
            and _is_fp8_only_layer_config(layer_config_path)
            and model_preset != "krea2"
        ):
            return (
                "The selected layer config is FP8-only, but the selected quantization format is not FP8. "
                "Clear Layer Config JSON or select the Krea 2 model preset so the GUI can generate a matching config."
            )
        if params.get("simple"):
            return None
        if params.get("optimizer") != "prodigy":
            return None
        if _has_prodigy_optimizer():
            return None
        return (
            "Optimizer 'prodigy' requires the optional package 'prodigy-plus-schedule-free'. "
            "Install it in the active environment, or switch the optimizer to 'original', 'adamw', or 'radam'."
        )

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
            elif scaling_mode == "row":
                scaling_str = "_rowwise"
            else:
                scaling_str = f"_bs{block_size or 128}"
        else:
            format_str = "fp8"
            scaling_str = f"_{scaling_mode}"

        return f"{base}_{prefix}{format_str}{mixed_suffix}{scaling_str}.safetensors"

    def _start_single_worker_locked(self) -> None:
        if self.single_worker_thread and self.single_worker_thread.is_alive():
            return
        self.single_worker_thread = threading.Thread(
            target=self._single_worker_loop,
            name="model-quantizer-single-queue",
            daemon=True,
        )
        self.single_worker_thread.start()

    def enqueue_single(
        self,
        input_file: str,
        output_file: str,
        delete_original: bool,
        params: Dict[str, object],
    ) -> Tuple[str, str]:
        if not input_file:
            self._set_single_status("Input file is required.")
            return self.single_queue_and_status()
        if not os.path.isfile(input_file):
            self._set_single_status(f"Input file not found: {input_file}")
            return self.single_queue_and_status()
        validation_error = self._validate_quantization_params(params)
        if validation_error:
            self._set_single_status(validation_error)
            return self.single_queue_and_status()

        output_path = output_file or self._default_output_name(input_file, params)
        if output_path and os.path.abspath(output_path) == os.path.abspath(input_file):
            self._set_single_status("Output file cannot be the same as input file.")
            return self.single_queue_and_status()

        with self.single_queue_lock:
            self.single_job_counter += 1
            job = {
                "id": self.single_job_counter,
                "input_file": input_file,
                "output_path": output_path,
                "delete_original": bool(delete_original),
                "params": copy.deepcopy(params),
            }
            self.single_queue.append(job)
            self.single_status_text = f"Queued single-file conversion: {self._format_single_job(job)}"
            self._start_single_worker_locked()
            return self._single_queue_text_locked(), self.single_status_text

    def _single_worker_loop(self) -> None:
        while True:
            with self.single_queue_lock:
                if not self.single_queue:
                    self.single_current_job = None
                    self.single_worker_thread = None
                    return
                job = self.single_queue.pop(0)
                self.single_current_job = job
                self.single_cancel_requested = False
                self.single_status_text = f"Starting conversion: {self._format_single_job(job)}"

            result = self.run_single(
                input_file=str(job.get("input_file") or ""),
                output_file=str(job.get("output_path") or ""),
                delete_original=bool(job.get("delete_original")),
                params=copy.deepcopy(job.get("params") or {}),
            )

            with self.single_queue_lock:
                finished_label = self._format_single_job(job)
                self.single_current_job = None
                self.single_status_text = self._tail_text(f"{finished_label}\n\n{result}")

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
        validation_error = self._validate_quantization_params(params)
        if validation_error:
            return validation_error

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
        validation_error = self._validate_quantization_params(params)
        if validation_error:
            return validation_error

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
    initial_model_preset_primary, initial_model_preset_other, _ = _split_model_preset_selection(
        config.get("model_quantizer.model_preset", "krea2")
    )
    initial_quant_format = config.get("model_quantizer.quant_format", QUANT_FORMAT_FP8)
    initial_scaling_mode = _coerce_scaling_mode_for_format(
        initial_quant_format,
        config.get("model_quantizer.scaling_mode", "tensor"),
    )
    initial_block_size = _coerce_block_size_for_format(
        initial_quant_format,
        initial_scaling_mode,
        config.get("model_quantizer.block_size", 64),
    )
    initial_custom_type = config.get("model_quantizer.custom_type", None)
    initial_custom_scaling_mode = _coerce_optional_scaling_mode_for_quant_type(
        initial_custom_type,
        config.get("model_quantizer.custom_scaling_mode", None),
    )
    initial_custom_block_size = (
        None
        if not _uses_block_size(QUANT_TYPE_TO_FORMAT.get(initial_custom_type), initial_custom_scaling_mode)
        else config.get("model_quantizer.custom_block_size", None)
    )
    initial_fallback_type = config.get("model_quantizer.fallback_type", None)
    initial_fallback_block_size = (
        None
        if not initial_fallback_type or QUANT_TYPE_TO_FORMAT.get(initial_fallback_type) in FIXED_SCALING_QUANT_FORMATS
        else config.get(
            "model_quantizer.fallback_block_size",
            128 if initial_fallback_type == "int8" else 64,
        )
    )

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
                            PRESET_EXTREME,
                            PRESET_FP8_SCALED,
                            PRESET_FP8_MIXED,
                            PRESET_INT8_FAST,
                            PRESET_INT8_TENSOR,
                            PRESET_INT8_CONVROT,
                            PRESET_MXFP8_BALANCED,
                            PRESET_NVFP4_BALANCED,
                            PRESET_NVFP4_Z,
                        ],
                        value=config.get("model_quantizer.preset", PRESET_NORMAL),
                        info="Quickly apply recommended optimization settings.",
                    )
                    model_preset_primary_dropdown = gr.Dropdown(
                        label="Model Preset (Recommended Filters)",
                        choices=MODEL_PRESET_PRIMARY_CHOICES,
                        value=initial_model_preset_primary,
                        info="Select a model to apply its recommended exclusion filters and defaults.",
                    )
                    model_preset_other_dropdown = gr.Dropdown(
                        label="Model Preset (Other Filters)",
                        choices=MODEL_PRESET_OTHER_CHOICES,
                        value=initial_model_preset_other,
                        info="Additional presets and upstream filters.",
                    )
            with gr.Accordion("Quality Upgrade Notes", open=False):
                gr.Markdown(QUALITY_GUIDANCE_MD)

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
                        value=initial_quant_format,
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
                        choices=_scaling_mode_choices_for_format(initial_quant_format),
                        value=initial_scaling_mode,
                        info=SCALING_MODE_INFO,
                    )
                    block_size = gr.Number(
                        label="Block Size",
                        value=initial_block_size,
                        step=1,
                        interactive=initial_quant_format not in FIXED_SCALING_QUANT_FORMATS and initial_scaling_mode != "tensor",
                        info=BLOCK_SIZE_INFO,
                    )
                    include_input_scale = gr.Checkbox(
                        label="Include input_scale tensors",
                        value=config.get("model_quantizer.include_input_scale", False),
                        info=INPUT_SCALE_INFO,
                    )
                with gr.Row():
                    convrot = gr.Checkbox(
                        label="ConvRot (INT8 Rowwise)",
                        value=config.get("model_quantizer.convrot", False),
                        info="Applies Hadamard rotation before INT8 row-wise quantization. Useful for Krea2 INT8 quality; ignored unless INT8 + row scaling is used.",
                    )
                    convrot_group_size = gr.Number(
                        label="ConvRot Group Size",
                        value=config.get("model_quantizer.convrot_group_size", 256),
                        step=1,
                        info="Power-of-4 group size for ConvRot. Upstream default is 256.",
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
                        value=initial_custom_type,
                        allow_custom_value=False,
                    )
                    custom_block_size = gr.Number(
                        label="Custom Block Size",
                        value=initial_custom_block_size,
                        step=1,
                        interactive=(
                            bool(initial_custom_type)
                            and _uses_block_size(QUANT_TYPE_TO_FORMAT.get(initial_custom_type), initial_custom_scaling_mode)
                        ),
                        info=BLOCK_SIZE_INFO,
                    )
                    custom_scaling_mode = gr.Dropdown(
                        label="Custom Scaling Mode",
                        choices=_scaling_mode_choices_for_quant_type(initial_custom_type),
                        value=initial_custom_scaling_mode,
                        info=SCALING_MODE_INFO,
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
                        value=initial_fallback_type,
                    )
                    fallback_block_size = gr.Number(
                        label="Fallback Block Size",
                        value=initial_fallback_block_size,
                        step=1,
                        interactive=(
                            bool(initial_fallback_type)
                            and QUANT_TYPE_TO_FORMAT.get(initial_fallback_type) not in FIXED_SCALING_QUANT_FORMATS
                        ),
                        info=BLOCK_SIZE_INFO,
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
                        value=config.get("model_quantizer.simple", True),
                    )
                    skip_inefficient_layers = gr.Checkbox(
                        label="Skip inefficient layers (heuristics)",
                        value=config.get("model_quantizer.skip_inefficient_layers", False),
                    )
                    full_matrix = gr.Checkbox(
                        label="Use full SVD matrix",
                        value=config.get("model_quantizer.full_matrix", False),
                        info=FULL_MATRIX_INFO,
                    )
                with gr.Row():
                    calib_samples = gr.Number(
                        label="Calibration Samples",
                        value=config.get("model_quantizer.calib_samples", 1024),
                        step=1,
                        info=CALIB_SAMPLES_INFO,
                    )
                    manual_seed = gr.Number(
                        label="Manual Seed (-1=random)",
                        value=config.get("model_quantizer.manual_seed", -1),
                        step=1,
                    )
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=OPTIMIZER_CHOICES,
                        value=config.get("model_quantizer.optimizer", "original"),
                        info=OPTIMIZER_INFO,
                    )
                with gr.Row():
                    num_iter = gr.Number(
                        label="Iterations",
                        value=config.get("model_quantizer.num_iter", 200),
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
                        value=config.get("model_quantizer.top_p", 0.02),
                    )
                    min_k = gr.Number(
                        label="Min K",
                        value=config.get("model_quantizer.min_k", 16),
                        step=1,
                    )
                    max_k = gr.Number(
                        label="Max K",
                        value=config.get("model_quantizer.max_k", 64),
                        step=1,
                    )

            with gr.Accordion("Advanced LR & Early Stopping", open=False) as advanced_lr_group:
                with gr.Row():
                    lr_gamma = gr.Number(
                        label="LR Gamma",
                        value=config.get("model_quantizer.lr_gamma", MANUAL_QUANT_DEFAULTS["lr_gamma"]),
                    )
                    lr_patience = gr.Number(
                        label="LR Patience",
                        value=config.get("model_quantizer.lr_patience", MANUAL_QUANT_DEFAULTS["lr_patience"]),
                        step=1,
                    )
                    lr_factor = gr.Number(
                        label="LR Factor",
                        value=config.get("model_quantizer.lr_factor", MANUAL_QUANT_DEFAULTS["lr_factor"]),
                    )
                with gr.Row():
                    lr_min = gr.Number(
                        label="LR Min",
                        value=config.get("model_quantizer.lr_min", MANUAL_QUANT_DEFAULTS["lr_min"]),
                    )
                    lr_cooldown = gr.Number(
                        label="LR Cooldown",
                        value=config.get("model_quantizer.lr_cooldown", MANUAL_QUANT_DEFAULTS["lr_cooldown"]),
                        step=1,
                    )
                    lr_threshold = gr.Number(
                        label="LR Threshold",
                        value=config.get("model_quantizer.lr_threshold", MANUAL_QUANT_DEFAULTS["lr_threshold"]),
                    )
                with gr.Row():
                    lr_adaptive_mode = gr.Dropdown(
                        label="LR Adaptive Mode",
                        choices=["simple-reset", "no-reset"],
                        value=config.get("model_quantizer.lr_adaptive_mode", MANUAL_QUANT_DEFAULTS["lr_adaptive_mode"]),
                    )
                    lr_shape_influence = gr.Number(
                        label="LR Shape Influence",
                        value=config.get("model_quantizer.lr_shape_influence", MANUAL_QUANT_DEFAULTS["lr_shape_influence"]),
                    )
                    lr_threshold_mode = gr.Dropdown(
                        label="LR Threshold Mode",
                        choices=["rel", "abs"],
                        value=config.get("model_quantizer.lr_threshold_mode", MANUAL_QUANT_DEFAULTS["lr_threshold_mode"]),
                    )
                with gr.Row():
                    early_stop_loss = gr.Number(
                        label="Early Stop Loss",
                        value=config.get("model_quantizer.early_stop_loss", MANUAL_QUANT_DEFAULTS["early_stop_loss"]),
                    )
                    early_stop_lr = gr.Number(
                        label="Early Stop LR",
                        value=config.get("model_quantizer.early_stop_lr", MANUAL_QUANT_DEFAULTS["early_stop_lr"]),
                    )
                    early_stop_stall = gr.Number(
                        label="Early Stop Stall",
                        value=config.get("model_quantizer.early_stop_stall", MANUAL_QUANT_DEFAULTS["early_stop_stall"]),
                        step=1,
                    )

            with gr.Accordion("NVFP4 / MXFP8 Options", open=False) as nvfp4_group:
                with gr.Row():
                    scale_optimization = gr.Dropdown(
                        label="Scale Optimization (NVFP4)",
                        choices=["fixed", "iterative", "joint", "dualround"],
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
                        label="Activation Calibration Samples",
                        value=config.get("model_quantizer.actcal_samples", 64),
                        step=1,
                        info=ACTCAL_SAMPLES_INFO,
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
                        value=config.get("model_quantizer.save_quant_metadata", True),
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
                        value=config.get("model_quantizer.verbose", "NORMAL"),
                    )
                    verbose_pinned = gr.Checkbox(
                        label="Verbose pinned memory transfers",
                        value=config.get("model_quantizer.verbose_pinned", False),
                    )
                    low_memory = gr.Checkbox(
                        label="Low memory mode",
                        value=config.get("model_quantizer.low_memory", True),
                    )
            with gr.Accordion("How bias correction works", open=False):
                gr.Markdown(BIAS_CORRECTION_PANEL_MD)
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
            single_queue_status = gr.Textbox(
                label="Single Conversion Queue",
                value=quantizer.single_queue_text(),
                lines=6,
                max_lines=18,
                interactive=False,
            )
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
        convrot_value,
        convrot_group_size_value,
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
        model_preset_primary_value,
        model_preset_other_value,
        filter_values: Dict[str, bool],
    ) -> Dict[str, object]:
        normalized_scaling_mode = _coerce_scaling_mode_for_format(
            quant_format_value,
            scaling_mode_value,
        )
        normalized_block_size = _coerce_block_size_for_format(
            quant_format_value,
            normalized_scaling_mode,
            block_size_value,
        )
        custom_scaling_mode_normalized = _coerce_optional_scaling_mode_for_quant_type(
            custom_type_value,
            custom_scaling_mode_value,
        )
        custom_block_size_normalized = (
            None
            if (
                QUANT_TYPE_TO_FORMAT.get(custom_type_value) in FIXED_SCALING_QUANT_FORMATS
                or not _uses_block_size(QUANT_TYPE_TO_FORMAT.get(custom_type_value), custom_scaling_mode_normalized)
            )
            else _to_optional_positive_int(custom_block_size_value, None)
        )
        fallback_block_size_normalized = (
            None
            if (
                not fallback_type_value
                or QUANT_TYPE_TO_FORMAT.get(fallback_type_value) in FIXED_SCALING_QUANT_FORMATS
            )
            else _to_optional_positive_int(fallback_block_size_value, None)
        )
        return {
            "workflow": workflow_value,
            "quant_format": quant_format_value,
            "comfy_quant": bool(comfy_quant_value),
            "full_precision_matrix_mult": bool(full_precision_matrix_mult_value),
            "scaling_mode": normalized_scaling_mode,
            "block_size": normalized_block_size,
            "include_input_scale": bool(include_input_scale_value),
            "convrot": bool(convrot_value),
            "convrot_group_size": _to_int(convrot_group_size_value, 256),
            "custom_layers": custom_layers_value.strip() if isinstance(custom_layers_value, str) else custom_layers_value,
            "exclude_layers": exclude_layers_value.strip() if isinstance(exclude_layers_value, str) else exclude_layers_value,
            "custom_type": custom_type_value,
            "custom_block_size": custom_block_size_normalized,
            "custom_scaling_mode": custom_scaling_mode_normalized,
            "custom_simple": bool(custom_simple_value),
            "custom_heur": bool(custom_heur_value),
            "fallback_type": fallback_type_value,
            "fallback_block_size": fallback_block_size_normalized,
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
            MODEL_PRESET_FIELD: _visible_model_preset_value(model_preset_primary_value, model_preset_other_value),
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
        "convrot",
        "convrot_group_size",
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
        "num_iter",
        "calib_samples",
        "optimizer",
        "lr_schedule",
        "lr",
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
        "full_matrix",
        "scale_optimization",
        "scale_refinement_rounds",
        "layer_config_path",
        "layer_config_fullmatch",
        "manual_seed",
        "verbose",
        "low_memory",
        "save_quant_metadata",
    ]

    preset_field_components = [
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        convrot,
        convrot_group_size,
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
        num_iter,
        calib_samples,
        optimizer,
        lr_schedule,
        lr,
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
        full_matrix,
        scale_optimization,
        scale_refinement_rounds,
        layer_config_path,
        layer_config_fullmatch,
        manual_seed,
        verbose,
        low_memory,
        save_quant_metadata,
    ]

    def _preset_field_updates(overrides: Dict[str, object]):
        updates = []
        selected_format = overrides.get("quant_format")
        selected_scaling = _coerce_scaling_mode_for_format(
            selected_format,
            overrides.get("scaling_mode", "tensor"),
        ) if selected_format else None
        selected_custom_type = overrides.get("custom_type")
        selected_fallback_type = overrides.get("fallback_type")
        for name in preset_field_names:
            if name == "scaling_mode" and selected_format:
                updates.append(
                    gr.update(
                        value=selected_scaling,
                        choices=_scaling_mode_choices_for_format(selected_format),
                    )
                )
            elif name == "block_size" and not _uses_block_size(selected_format, selected_scaling):
                updates.append(gr.update(value=None, interactive=False))
            elif name == "block_size" and selected_format:
                updates.append(gr.update(value=overrides.get(name), interactive=True))
            elif name == "custom_scaling_mode" and "custom_type" in overrides:
                updates.append(
                    gr.update(
                        value=_coerce_optional_scaling_mode_for_quant_type(
                            selected_custom_type,
                            overrides.get(name),
                        ),
                        choices=_scaling_mode_choices_for_quant_type(selected_custom_type),
                    )
                )
            elif name == "custom_block_size" and not _uses_block_size(
                QUANT_TYPE_TO_FORMAT.get(selected_custom_type),
                _coerce_optional_scaling_mode_for_quant_type(
                    selected_custom_type,
                    overrides.get("custom_scaling_mode"),
                ),
            ):
                updates.append(gr.update(value=None, interactive=False))
            elif name == "custom_block_size" and "custom_type" in overrides:
                updates.append(gr.update(value=overrides.get(name), interactive=True))
            elif name == "fallback_block_size" and (
                not selected_fallback_type
                or QUANT_TYPE_TO_FORMAT.get(selected_fallback_type) in FIXED_SCALING_QUANT_FORMATS
            ):
                updates.append(gr.update(value=None, interactive=False))
            elif name == "fallback_block_size" and "fallback_type" in overrides:
                default_block_size = 128 if selected_fallback_type == "int8" else 64
                updates.append(gr.update(value=overrides.get(name) or default_block_size, interactive=True))
            elif name in overrides:
                updates.append(gr.update(value=overrides[name]))
            else:
                updates.append(gr.update())
        return updates

    def _apply_preset(preset_name: str, primary_model_preset: str, other_model_preset: str):
        selected_model_preset = _visible_model_preset_value(primary_model_preset, other_model_preset)
        _, overrides = _combined_preset_settings(selected_model_preset, preset_name)
        return _preset_field_updates(overrides)

    preset_dropdown.change(
        fn=_apply_preset,
        inputs=[preset_dropdown, model_preset_primary_dropdown, model_preset_other_dropdown],
        outputs=preset_field_components,
        show_progress=False,
    )

    format_default_field_names = [
        "comfy_quant",
        "full_precision_matrix_mult",
        "scaling_mode",
        "block_size",
        "convrot",
        "convrot_group_size",
        "low_memory",
        "save_quant_metadata",
    ]
    format_default_components = [
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        convrot,
        convrot_group_size,
        low_memory,
        save_quant_metadata,
    ]

    def _apply_quant_format_defaults(selected_format: str):
        defaults = FORMAT_DEFAULTS.get(selected_format, {})
        selected_scaling = _coerce_scaling_mode_for_format(
            selected_format,
            defaults.get("scaling_mode", "tensor"),
        )
        updates = []
        for name in format_default_field_names:
            if name == "scaling_mode":
                updates.append(
                    gr.update(
                        value=selected_scaling,
                        choices=_scaling_mode_choices_for_format(selected_format),
                    )
                )
            elif name == "block_size" and not _uses_block_size(selected_format, selected_scaling):
                updates.append(gr.update(value=None, interactive=False))
            elif name in defaults:
                updates.append(gr.update(value=defaults[name], interactive=True) if name == "block_size" else gr.update(value=defaults[name]))
            else:
                updates.append(gr.update())
        return updates

    quant_format.input(
        fn=_apply_quant_format_defaults,
        inputs=[quant_format],
        outputs=format_default_components,
        show_progress=False,
    )

    def _apply_scaling_mode_defaults(selected_scaling: str, selected_format: str):
        selected_scaling = _coerce_scaling_mode_for_format(selected_format, selected_scaling)
        if not _uses_block_size(selected_format, selected_scaling):
            return gr.update(value=None, interactive=False), gr.update(value=False)
        if selected_scaling == "block":
            block_default = 128 if selected_format == QUANT_FORMAT_INT8 else 64
            return gr.update(value=block_default, interactive=True), gr.update(value=False)
        if selected_scaling == "row":
            return gr.update(value=128, interactive=True), gr.update(value=False)
        return gr.update(value=None, interactive=False), gr.update(value=False)

    scaling_mode.input(
        fn=_apply_scaling_mode_defaults,
        inputs=[scaling_mode, quant_format],
        outputs=[block_size, convrot],
        show_progress=False,
    )

    def _apply_custom_type_defaults(selected_type: str, current_scaling: str):
        selected_format = QUANT_TYPE_TO_FORMAT.get(selected_type)
        selected_scaling = _coerce_optional_scaling_mode_for_quant_type(
            selected_type,
            current_scaling,
        )
        scaling_update = gr.update(
            value=selected_scaling,
            choices=_scaling_mode_choices_for_quant_type(selected_type),
        )
        if not selected_type or not _uses_block_size(selected_format, selected_scaling):
            return gr.update(value=None, interactive=False), scaling_update
        default_block_size = 128 if selected_type == "int8" else 64
        return gr.update(value=default_block_size, interactive=True), scaling_update

    custom_type.input(
        fn=_apply_custom_type_defaults,
        inputs=[custom_type, custom_scaling_mode],
        outputs=[custom_block_size, custom_scaling_mode],
        show_progress=False,
    )

    def _apply_custom_scaling_mode_defaults(selected_scaling: str, selected_type: str):
        selected_format = QUANT_TYPE_TO_FORMAT.get(selected_type)
        selected_scaling = _coerce_optional_scaling_mode_for_quant_type(
            selected_type,
            selected_scaling,
        )
        if not _uses_block_size(selected_format, selected_scaling):
            return gr.update(value=None, interactive=False)
        default_block_size = 128 if selected_type == "int8" else 64
        return gr.update(value=default_block_size, interactive=True)

    custom_scaling_mode.input(
        fn=_apply_custom_scaling_mode_defaults,
        inputs=[custom_scaling_mode, custom_type],
        outputs=[custom_block_size],
        show_progress=False,
    )
    custom_scaling_mode.change(
        fn=_apply_custom_scaling_mode_defaults,
        inputs=[custom_scaling_mode, custom_type],
        outputs=[custom_block_size],
        show_progress=False,
    )

    def _apply_fallback_type_defaults(selected_type: str):
        selected_format = QUANT_TYPE_TO_FORMAT.get(selected_type)
        if not selected_type or selected_format in FIXED_SCALING_QUANT_FORMATS:
            return gr.update(value=None, interactive=False)
        default_block_size = 128 if selected_type == "int8" else 64
        return gr.update(value=default_block_size, interactive=True)

    fallback_type.input(
        fn=_apply_fallback_type_defaults,
        inputs=[fallback_type],
        outputs=[fallback_block_size],
        show_progress=False,
    )
    fallback_type.change(
        fn=_apply_fallback_type_defaults,
        inputs=[fallback_type],
        outputs=[fallback_block_size],
        show_progress=False,
    )

    def _apply_model_preset(selected: str, selected_preset: str):
        selected_value = _model_preset_value(selected)
        selected_filters = _model_preset_filters(selected_value)
        reset_other_update = (
            gr.update(value=MODEL_PRESET_NONE)
            if selected_value != MODEL_PRESET_NONE
            else gr.update()
        )

        if selected_value == MODEL_PRESET_NONE:
            return (
                [reset_other_update]
                + [gr.update() for _ in filter_checkboxes.values()]
                + [gr.update()]
                + [gr.update() for _ in preset_field_components]
                + [gr.update()]
            )

        effective_preset, combined = _combined_preset_settings(
            selected_value,
            selected_preset,
            use_model_default_preset=True,
        )

        filter_updates = [
            gr.update(value=(name in selected_filters))
            for name in filter_checkboxes.keys()
        ]

        include_update = (
            gr.update(value=combined["include_input_scale"])
            if "include_input_scale" in combined
            else gr.update()
        )
        return (
            [reset_other_update]
            + filter_updates
            + [gr.update(value=effective_preset)]
            + _preset_field_updates(combined)
            + [include_update]
        )

    model_preset_primary_dropdown.change(
        fn=_apply_model_preset,
        inputs=[model_preset_primary_dropdown, preset_dropdown],
        outputs=[model_preset_other_dropdown] + list(filter_checkboxes.values()) + [preset_dropdown] + preset_field_components + [include_input_scale],
        show_progress=False,
    )

    model_preset_other_dropdown.change(
        fn=_apply_model_preset,
        inputs=[model_preset_other_dropdown, preset_dropdown],
        outputs=[model_preset_primary_dropdown] + list(filter_checkboxes.values()) + [preset_dropdown] + preset_field_components + [include_input_scale],
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

    workflow.input(
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
        convrot,
        convrot_group_size,
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
        model_preset_primary_dropdown,
        model_preset_other_dropdown,
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
        return quantizer.enqueue_single(
            input_file=single_input_file_value,
            output_file=single_output_file_value,
            delete_original=bool(single_delete_original_value),
            params=params,
        )

    single_run_button.click(
        fn=_run_single,
        inputs=[single_input_file, single_output_file, single_delete_original] + single_inputs,
        outputs=[single_queue_status, single_status],
        show_progress=False,
        queue=False,
    )

    single_cancel_button.click(
        fn=quantizer.cancel_single,
        inputs=[],
        outputs=[single_queue_status, single_status],
        show_progress=False,
        queue=False,
    )

    single_queue_timer = gr.Timer(2.0)
    single_queue_timer.tick(
        fn=quantizer.single_queue_and_status,
        inputs=[],
        outputs=[single_queue_status, single_status],
        show_progress=False,
        queue=False,
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
        "convrot",
        "convrot_group_size",
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
        MODEL_PRESET_PRIMARY_FIELD,
        MODEL_PRESET_OTHER_FIELD,
    ] + [f"filter.{name}" for name in filter_checkboxes.keys()]

    settings_components = [
        workflow,
        quant_format,
        comfy_quant,
        full_precision_matrix_mult,
        scaling_mode,
        block_size,
        include_input_scale,
        convrot,
        convrot_group_size,
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
        model_preset_primary_dropdown,
        model_preset_other_dropdown,
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
        values_by_name = dict(parameters)
        visible_model_preset = _visible_model_preset_value(
            values_by_name.get(MODEL_PRESET_PRIMARY_FIELD, MODEL_PRESET_NONE),
            values_by_name.get(MODEL_PRESET_OTHER_FIELD, MODEL_PRESET_NONE),
        )
        parameters.append((MODEL_PRESET_FIELD, visible_model_preset))
        from .common_gui import SaveConfigFile
        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "save_as_bool"],
        )
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
        loaded_model_preset = flat.get(MODEL_PRESET_FIELD)
        if loaded_model_preset is None:
            loaded_model_preset = flat.get(f"model_quantizer.{MODEL_PRESET_FIELD}")
        loaded_model_preset_primary = flat.get(MODEL_PRESET_PRIMARY_FIELD)
        if loaded_model_preset_primary is None:
            loaded_model_preset_primary = flat.get(f"model_quantizer.{MODEL_PRESET_PRIMARY_FIELD}")
        loaded_model_preset_other = flat.get(MODEL_PRESET_OTHER_FIELD)
        if loaded_model_preset_other is None:
            loaded_model_preset_other = flat.get(f"model_quantizer.{MODEL_PRESET_OTHER_FIELD}")
        loaded_quant_format = flat.get("quant_format")
        if loaded_quant_format is None:
            loaded_quant_format = flat.get("model_quantizer.quant_format")
        loaded_scaling_mode = flat.get("scaling_mode")
        if loaded_scaling_mode is None:
            loaded_scaling_mode = flat.get("model_quantizer.scaling_mode")
        loaded_custom_type = flat.get("custom_type")
        if loaded_custom_type is None:
            loaded_custom_type = flat.get("model_quantizer.custom_type")
        loaded_custom_scaling_mode = flat.get("custom_scaling_mode")
        if loaded_custom_scaling_mode is None:
            loaded_custom_scaling_mode = flat.get("model_quantizer.custom_scaling_mode")
        loaded_fallback_type = flat.get("fallback_type")
        if loaded_fallback_type is None:
            loaded_fallback_type = flat.get("model_quantizer.fallback_type")
        values_out = []
        for name, current in zip(settings_names, values):
            if name in (MODEL_PRESET_PRIMARY_FIELD, MODEL_PRESET_OTHER_FIELD):
                loaded_visible_model_preset = (
                    loaded_model_preset_primary
                    if name == MODEL_PRESET_PRIMARY_FIELD
                    else loaded_model_preset_other
                )
                if loaded_visible_model_preset is None:
                    loaded_visible_model_preset = loaded_model_preset
                if loaded_visible_model_preset is None:
                    values_out.append(current)
                else:
                    values_out.append(
                        _model_preset_component_value(name, loaded_visible_model_preset, current)
                    )
                continue
            value = flat.get(name)
            if value is None:
                value = flat.get(f"model_quantizer.{name}")
            if value is None:
                values_out.append(current)
            else:
                if name == "scaling_mode":
                    selected_format = loaded_quant_format or config.get("model_quantizer.quant_format", QUANT_FORMAT_FP8)
                    value = _coerce_scaling_mode_for_format(selected_format, value)
                    values_out.append(
                        gr.update(
                            value=value,
                            choices=_scaling_mode_choices_for_format(selected_format),
                        )
                    )
                    continue
                elif name == "block_size":
                    selected_format = loaded_quant_format or config.get("model_quantizer.quant_format", QUANT_FORMAT_FP8)
                    selected_scaling = _coerce_scaling_mode_for_format(
                        selected_format,
                        loaded_scaling_mode or "tensor",
                    )
                    value = _coerce_block_size_for_format(selected_format, selected_scaling, value)
                    values_out.append(
                        gr.update(
                            value=value,
                            interactive=_uses_block_size(selected_format, selected_scaling),
                        )
                    )
                    continue
                elif name == "custom_block_size":
                    loaded_custom_scaling = _coerce_optional_scaling_mode_for_quant_type(
                        loaded_custom_type,
                        loaded_custom_scaling_mode,
                    )
                    value = (
                        None
                        if (
                            not _uses_block_size(QUANT_TYPE_TO_FORMAT.get(loaded_custom_type), loaded_custom_scaling)
                        )
                        else _to_optional_positive_int(value, None)
                    )
                    values_out.append(
                        gr.update(
                            value=value,
                            interactive=(
                                bool(loaded_custom_type)
                                and _uses_block_size(QUANT_TYPE_TO_FORMAT.get(loaded_custom_type), loaded_custom_scaling)
                            ),
                        )
                    )
                    continue
                elif name == "custom_scaling_mode":
                    value = _coerce_optional_scaling_mode_for_quant_type(loaded_custom_type, value)
                    values_out.append(
                        gr.update(
                            value=value,
                            choices=_scaling_mode_choices_for_quant_type(loaded_custom_type),
                        )
                    )
                    continue
                elif name == "fallback_block_size":
                    value = (
                        None
                        if (
                            not loaded_fallback_type
                            or QUANT_TYPE_TO_FORMAT.get(loaded_fallback_type) in FIXED_SCALING_QUANT_FORMATS
                        )
                        else (
                            _to_optional_positive_int(value, None)
                            or (128 if loaded_fallback_type == "int8" else 64)
                        )
                    )
                    values_out.append(
                        gr.update(
                            value=value,
                            interactive=(
                                bool(loaded_fallback_type)
                                and QUANT_TYPE_TO_FORMAT.get(loaded_fallback_type) not in FIXED_SCALING_QUANT_FORMATS
                            ),
                        )
                    )
                    continue
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
