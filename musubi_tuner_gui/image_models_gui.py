"""Spec-driven trainer tabs for the newer image models.

One module drives Qwen-Image 2.1, ERNIE-Image, Mage-Flow / Mage-Flow-Edit and HiDream-O1.
Every model-specific difference (scripts, model files, cache flags, extra options, defaults,
validation) lives in an ``ImageArchSpec`` instead of if/else branches, so adding another
image architecture means adding one spec. The proven Ideogram 4 / Krea 2 module
(``modern_image_lora_gui``) is reused for its tested helpers and stays unchanged.

Guardrails honoured here (see AI_TRAINER_GUARDRAILS.md):
1. Logging: debug-mode logging is mapped to the ``log_with`` field before the runtime TOML is
   written, so ``SaveConfigFileToRun`` resolves ``logging_dir`` to ``output_dir/logs``.
2. Training mode: ``training_mode`` round-trips through the runtime TOML and is inferred on
   load with ``infer_training_mode_from_loaded_config``; panel visibility is re-synced with a
   visibility-only ``.then()``.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable

import gradio as gr
import toml

from .class_accelerate_launch import AccelerateLaunch
from .class_advanced_training import AdvancedTraining
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .class_network import Network
from .class_optimizer_and_scheduler import OptimizerAndScheduler
from .class_save_load import SaveLoadSettings
from .class_training import TrainingSettings
from .common_gui import (
    text_for_list,
    strip_path_parameters,
    strip_path_text,
    SaveConfigFile,
    SaveConfigFileToRun,
    get_file_path,
    get_file_path_or_save_as,
    get_folder_path,
    print_command_and_toml,
    requires_native_compile_toolchain,
    resolve_portable_model_value,
    run_cmd_advanced_training,
    save_executed_script,
    scriptdir,
    setup_environment,
    utf8_subprocess_options,
    validate_block_swap_options,
)
from .custom_logging import setup_logging
from .dataset_config_generator import (
    dataset_toml_matches,
    generate_dataset_config_from_folders,
    generate_kandinsky6_dataset_config_from_folders,
    round_frames_to_4n_plus_1,
    save_dataset_config,
    validate_dataset_config,
)
from .full_finetune_gui import (
    FULL_FINE_TUNING_MODE,
    LORA_TRAINING_MODE,
    TRAINING_MODE_CHOICES,
    infer_training_mode_from_loaded_config,
    is_full_fine_tuning,
    normalize_image_training_parameters,
    normalize_training_mode,
    training_mode_runtime_exclusions,
)
from .modern_image_lora_gui import (
    _build_workflow_script as _modern_build_workflow_script,
    _coerce_cli_list,
    _coerce_float_list,
    _find_accelerate_launch,
    _replace_parameter,
    _resolve_device,
)
from .optimizer_catalog import validate_automagic_configuration
from .preset_library import PresetLibrary, is_shipped_preset

log = setup_logging()

DEBUG_LOGGING_MODES = {
    "Enable Logging (TensorBoard)": "tensorboard",
    "Enable Logging (WandB)": "wandb",
    "Enable Logging (All)": "all",
}
DEBUG_CLI_FLAGS = {
    "Show Timesteps (Image)": "--show_timesteps image",
    "Show Timesteps (Console)": "--show_timesteps console",
}
TIMESTEP_CHOICES = [
    "sigmoid",
    "shift",
    "uniform",
    "uniform_shift",
    "qwen_shift",
    "flux_shift",
    "flux2_shift",
    "sigma",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
]
WEIGHTING_CHOICES = ["none", "mode", "logit_normal", "cosmap", "sigma_sqrt"]


# ----------------------------------------------------------------------------------------------
# Spec types
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPath:
    """A model file (or folder) field shown in the Model panel."""

    key: str
    label: str
    filename: str
    info: str
    required_for: frozenset = frozenset({"train"})  # subset of {"latent", "text", "train", "sample"}
    extension: str = ".safetensors"
    is_dir: bool = False


@dataclass(frozen=True)
class ExtraOption:
    """A model-specific setting. ``runtime`` keys are written to the training TOML."""

    key: str
    kind: str  # "checkbox" | "dropdown" | "number" | "textbox"
    label: str
    default: object
    panel: str  # "model" | "schedule" | "sample" | "caching" | "dataset" | "advanced" | "full" (Full Fine-Tuning panel)
    info: str = ""
    choices: tuple = ()
    runtime: bool = True
    lora_only: bool = False
    full_only: bool = False
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None


@dataclass(frozen=True)
class ImageArchSpec:
    key: str
    display_name: str
    slug: str
    network_module: str
    train_script: str
    full_train_script: str | None
    latent_cache_script: str
    text_cache_script: str
    model_folder: str
    model_paths: tuple
    max_blocks_to_swap: int
    max_blocks_to_swap_full: int
    preset_folders: tuple
    supports_fp8_scaled: bool = True
    supports_h2d_block_swap: bool = True
    supported_attention: frozenset = frozenset({"sdpa", "flash_attn", "xformers"})
    timestep_choices: tuple = tuple(TIMESTEP_CHOICES)
    weighting_choices: tuple = tuple(WEIGHTING_CHOICES)
    resolution_step: int = 16
    supports_control_images: bool = False
    control_label: str = "Use control / reference images (editing)"
    extras: tuple = ()
    sample_flags: tuple = ("w", "h", "s", "d", "l", "n")
    defaults: dict = field(default_factory=dict)
    lora_defaults: dict = field(default_factory=dict)
    full_defaults: dict = field(default_factory=dict)
    gui_only_keys: frozenset = frozenset()
    notes: str = ""
    build_latent_cache_args: Callable[[dict], list[str]] | None = None
    build_text_cache_args: Callable[[dict], list[str]] | None = None
    validate: Callable[[dict, bool], None] | None = None
    dataset_kind: str = "image"  # "image" | "video" (video datasets also accept image folders)
    default_frames: int = 121
    download_bundles: tuple = ()  # Download_Train_Models.py --model keys

    @property
    def supports_full_finetune(self) -> bool:
        return bool(self.full_train_script)

    def model_path(self, key: str) -> ModelPath | None:
        return next((item for item in self.model_paths if item.key == key), None)


# ----------------------------------------------------------------------------------------------
# Per-model cache argument builders and validation hooks
# ----------------------------------------------------------------------------------------------


def _qwen21_latent_args(params: dict) -> list[str]:
    args = ["--vae", str(params["vae"])]
    if params.get("vae_tiling"):
        args.append("--vae_tiling")
    return args


def _qwen21_text_args(params: dict) -> list[str]:
    args = ["--text_encoder", str(params["text_encoder"])]
    dtype = str(params.get("caching_teo_text_encoder_dtype") or "bfloat16")
    args.extend(["--text_encoder_dtype", dtype])
    if params.get("caching_teo_fp8_vl"):
        args.append("--fp8_vl")
    return args


def _qwen21_validate(params: dict, full_finetune: bool) -> None:
    if full_finetune and params.get("caching_teo_fp8_vl") and str(params.get("caching_teo_device")) == "cpu":
        raise ValueError("FP8 text-encoder caching needs a CUDA device; set the Text Cache Device to cuda.")


def _ernie_latent_args(params: dict) -> list[str]:
    return ["--vae", str(params["vae"])]


def _ernie_text_args(params: dict) -> list[str]:
    return ["--text_encoder", str(params["text_encoder"])]


def _mage_latent_args(params: dict) -> list[str]:
    args = ["--vae", str(params["vae"]), "--vae_dtype", str(params.get("vae_dtype") or "bfloat16")]
    if params.get("is_edit"):
        args.append("--is_edit")
    seed = params.get("seed")
    if seed not in (None, ""):
        args.extend(["--seed", str(int(seed))])
    return args


def _mage_text_args(params: dict) -> list[str]:
    args = [
        "--text_encoder",
        str(params["text_encoder"]),
        "--text_encoder_dtype",
        str(params.get("caching_teo_text_encoder_dtype") or "bfloat16"),
    ]
    if params.get("is_edit"):
        args.append("--is_edit")
    return args


def _mage_validate(params: dict, full_finetune: bool) -> None:
    if params.get("compile") and params.get("compile_fullgraph"):
        raise ValueError("Mage-Flow rejects torch.compile fullgraph mode (packed segment lengths are data dependent).")
    if params.get("fp8_base") and not params.get("fp8_scaled"):
        raise ValueError("Mage-Flow rejects plain FP8 Base; enable Scaled FP8 together with FP8 Base.")


def _hidream_latent_args(params: dict) -> list[str]:
    return []  # pixel-space model: the pixel cache needs no VAE


def _hidream_text_args(params: dict) -> list[str]:
    args = ["--model_type", str(params.get("model_type") or "full")]
    if params.get("caching_teo_fp8_te"):
        args.extend(["--dit", str(params["dit"]), "--fp8_te"])
    return args


def _hidream_validate(params: dict, full_finetune: bool) -> None:
    task = str(params.get("task") or "t2i")
    blocks = int(params.get("blocks_to_swap") or 0)
    batch_size = int(params.get("dataset_batch_size") or 1)
    if blocks > 0 and batch_size > 1 and (task == "i2i" or params.get("flash_attn")):
        raise ValueError(
            "HiDream-O1 block swap with the i2i task or FlashAttention requires a dataset batch size of 1."
        )
    if task == "i2i" and not params.get("use_control_images"):
        raise ValueError("HiDream-O1 task i2i needs control/reference images: enable 'Use control images' in the dataset panel.")
    if task == "t2i" and params.get("use_control_images"):
        raise ValueError("HiDream-O1 control/reference datasets need Task = i2i.")
    if float(params.get("dino_loss_weight") or 0) > 0:
        try:
            import sensecraft  # noqa: F401
        except ImportError as exc:
            raise ValueError("DINO perceptual loss needs the optional 'sensecraft[dinov3]' package.") from exc


# ----------------------------------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------------------------------

COMMON_LORA_DEFAULTS = {
    "network_dim": 128,
    "network_alpha": 128,
    "optimizer_type": "AdaFactor",
    "optimizer_args": ["scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01"],
    "learning_rate": 6e-5,
    "lr_scheduler": "constant",
    "max_grad_norm": 0.0,
}
COMMON_FULL_DEFAULTS = {
    "optimizer_type": "AdaFactor",
    "optimizer_args": ["scale_parameter=False", "relative_step=False", "warmup_init=False", "weight_decay=0.01"],
    "learning_rate": 1e-5,
    "lr_scheduler": "constant",
    "max_grad_norm": 0.0,
    "full_bf16": True,
    "fused_backward_pass": True,
    "mem_eff_save": True,
    "save_precision": "bf16",
}

QWEN_IMAGE_21 = ImageArchSpec(
    key="qwen_image_21",
    display_name="Qwen-Image 2.1",
    slug="qwen_image_21",
    network_module="networks.lora_qwen_image_21",
    train_script="qwen_image_21_train_network.py",
    full_train_script="qwen_image_21_train.py",
    latent_cache_script="qwen_image_21_cache_latents.py",
    text_cache_script="qwen_image_21_cache_text_encoder_outputs.py",
    model_folder="Training_Models_Qwen_Image_2_1",
    model_paths=(
        ModelPath("dit", "Qwen-Image 2.1 DiT", "qwen_image_2.1_bf16.safetensors", "BF16 DiT (ComfyUI or Diffusers single file).", frozenset({"train"})),
        ModelPath("vae", "Qwen-Image 2.1 RGBA VAE", "qwen_image_2.1_vae.safetensors", "Diffusers-key VAE file. Needed for latent caching and samples.", frozenset({"latent", "sample"})),
        ModelPath("text_encoder", "Qwen3-VL 8B Text Encoder", "qwen3vl_8b_bf16.safetensors", "BF16/FP32 Qwen3-VL-8B-Instruct. Needed for text caching and samples.", frozenset({"text", "sample"})),
    ),
    max_blocks_to_swap=30,  # 32 blocks; backward needs 2 resident blocks (31 fails in full fine-tuning)
    max_blocks_to_swap_full=30,
    preset_folders=("Qwen_Image_2.1_Training_Configs",),
    download_bundles=("qwen_image_21",),
    resolution_step=32,
    supports_control_images=True,
    control_label="Use control images (Qwen-Image 2.1 editing)",
    extras=(
        ExtraOption("vae_tiling", "checkbox", "VAE Tiling", False, "model", "Tile the VAE for large images (latent caching and samples)."),
        ExtraOption("fp8_vl", "checkbox", "FP8 Text Encoder for Samples", False, "sample", "Store Qwen3-VL language blocks in FP8 while generating training samples."),
        ExtraOption("caching_teo_fp8_vl", "checkbox", "FP8 Text Encoder for Caching", False, "caching", "Quantize Qwen3-VL language blocks to FP8 while caching (~8 GB less VRAM).", runtime=False),
        ExtraOption("caching_teo_text_encoder_dtype", "dropdown", "Text Encoder Dtype", "bfloat16", "caching", "Use float32 only for CPU caching.", choices=("bfloat16", "float32"), runtime=False),
    ),
    defaults={
        "dataset_resolution_width": 1328,
        "dataset_resolution_height": 1328,
        "timestep_sampling": "sigmoid",
        "weighting_scheme": "mode",
        "mode_scale": 1.0,  # researched Qwen-Image recipe (sigmoid + mode 1.0, shift 2.2)
        "discrete_flow_shift": 2.2,
        "sample_width": 1328,
        "sample_height": 1328,
        "sample_steps": 30,
        "sample_cfg_scale": 4.0,
        "sample_negative_prompt": " ",
        "seed": 99,
        "sample_seed": 99,
        "max_train_epochs": 200,
        "save_every_n_epochs": 25,
    },
    lora_defaults=COMMON_LORA_DEFAULTS,
    full_defaults={**COMMON_FULL_DEFAULTS, "learning_rate": 9e-6},
    gui_only_keys=frozenset({"caching_teo_fp8_vl", "caching_teo_text_encoder_dtype"}),
    notes=(
        "Qwen-Image 2.1 (Sept 2026) has its own DiT, RGBA VAE and Qwen3-VL 8B text encoder - Qwen-Image 1.x "
        "models and caches cannot be reused. Image sides are aligned to multiples of 32. Enable control images "
        "to train editing LoRAs (1+ ordered references per target)."
    ),
    build_latent_cache_args=_qwen21_latent_args,
    build_text_cache_args=_qwen21_text_args,
    validate=_qwen21_validate,
)

ERNIE_IMAGE = ImageArchSpec(
    key="ernie_image",
    display_name="ERNIE-Image",
    slug="ernie_image",
    network_module="networks.lora_ernie_image",
    train_script="ernie_image_train_network.py",
    full_train_script=None,
    latent_cache_script="ernie_image_cache_latents.py",
    text_cache_script="ernie_image_cache_text_encoder_outputs.py",
    model_folder="Training_Models_ERNIE_Image",
    model_paths=(
        ModelPath("dit", "ERNIE-Image DiT", "ernie-image.safetensors", "BF16 DiT (ComfyUI single file or first Diffusers shard).", frozenset({"train"})),
        ModelPath("vae", "FLUX.2 VAE", "ernie_image_flux2_vae.safetensors", "ERNIE-Image uses the FLUX.2 VAE. Required by the trainer.", frozenset({"latent", "train"})),
        ModelPath("text_encoder", "Ministral 3 3B Text Encoder", "ministral-3-3b.safetensors", "Required by the trainer and the text cache.", frozenset({"text", "train"})),
    ),
    max_blocks_to_swap=34,
    max_blocks_to_swap_full=34,
    preset_folders=("ERNIE_Image_Training_Configs",),
    download_bundles=("ernie_image",),
    extras=(),
    defaults={
        "timestep_sampling": "shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 4.0,
        "sample_steps": 50,
        "sample_cfg_scale": 4.0,
        "sample_negative_prompt": "low quality, blurry",
        "max_train_epochs": 200,
        "save_every_n_epochs": 25,
    },
    lora_defaults=COMMON_LORA_DEFAULTS,
    notes=(
        "Baidu ERNIE-Image (8B DiT, Ministral 3 text encoder, FLUX.2 VAE). LoRA training only. The official "
        "scheduler uses flow shift 4.0, which is the recommended starting point."
    ),
    build_latent_cache_args=_ernie_latent_args,
    build_text_cache_args=_ernie_text_args,
)

MAGE_FLOW = ImageArchSpec(
    key="mage_flow",
    display_name="Mage-Flow",
    slug="mage_flow",
    network_module="musubi_tuner.networks.lora_mage_flow",  # the Mage-Flow trainer checks this exact module path
    train_script="mage_flow_train_network.py",
    full_train_script=None,
    latent_cache_script="mage_flow_cache_latents.py",
    text_cache_script="mage_flow_cache_text_encoder_outputs.py",
    model_folder="Training_Models_Mage_Flow",
    model_paths=(
        ModelPath("dit", "Mage-Flow DiT", "mage_flow_base_bf16.safetensors", "Base T2I DiT (or the Edit Base DiT when Edit mode is on).", frozenset({"train"})),
        ModelPath("vae", "MageVAE", "mage_flow_vae_bf16.safetensors", "Needed for latent caching and samples.", frozenset({"latent", "sample"})),
        ModelPath("text_encoder", "Qwen3-VL 4B Text Encoder", "mage_flow_qwen3vl_4b_bf16.safetensors", "Needed for text caching and samples.", frozenset({"text", "sample"})),
    ),
    max_blocks_to_swap=10,
    max_blocks_to_swap_full=10,
    preset_folders=("Mage_Flow_Training_Configs", "Mage_Flow_Edit_Training_Configs"),
    download_bundles=("mage_flow", "mage_flow_edit"),
    supported_attention=frozenset({"sdpa", "flash_attn"}),
    supports_control_images=True,
    control_label="Use reference images (required for Mage-Flow-Edit, 1-3 per target)",
    extras=(
        ExtraOption("is_edit", "checkbox", "Mage-Flow-Edit Mode", False, "model", "Train the Edit model with 1-3 ordered reference images per target."),
        ExtraOption("caching_teo_text_encoder_dtype", "dropdown", "Text Encoder Dtype", "bfloat16", "caching", "", choices=("bfloat16", "float16", "float32"), runtime=False),
    ),
    defaults={
        "timestep_sampling": "uniform_shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 6.0,
        "sample_steps": 30,
        "sample_cfg_scale": 5.0,
        "sample_negative_prompt": " ",
        "vae_dtype": "bfloat16",
        "max_train_epochs": 200,
        "save_every_n_epochs": 25,
    },
    lora_defaults=COMMON_LORA_DEFAULTS,
    gui_only_keys=frozenset({"caching_teo_text_encoder_dtype"}),
    notes=(
        "Microsoft Mage-Flow (4B, MIT): native-resolution T2I and instruction editing. LoRA only. The trainer "
        "uses uniform_shift timesteps with static shift 6, matching the released inference scheduler."
    ),
    build_latent_cache_args=_mage_latent_args,
    build_text_cache_args=_mage_text_args,
    validate=_mage_validate,
)

HIDREAM_O1 = ImageArchSpec(
    key="hidream_o1",
    display_name="HiDream-O1",
    slug="hidream_o1",
    network_module="networks.lora_hidream_o1",
    train_script="hidream_o1_train_network.py",
    full_train_script="hidream_o1_train.py",
    latent_cache_script="hidream_o1_cache_pixel.py",
    text_cache_script="hidream_o1_cache_text_encoder_outputs.py",
    model_folder="Training_Models_HiDream_O1",
    model_paths=(
        ModelPath("dit", "HiDream-O1 Checkpoint", "hidream_o1_image_bf16.safetensors", "Single BF16 checkpoint (full or dev). No separate VAE or text encoder.", frozenset({"train"})),
    ),
    max_blocks_to_swap=34,  # 36 decoder layers; keep 2 resident for backward
    max_blocks_to_swap_full=34,
    preset_folders=("HiDream_O1_Training_Configs",),
    download_bundles=("hidream_o1",),
    supports_h2d_block_swap=True,
    supported_attention=frozenset({"sdpa", "flash_attn"}),
    resolution_step=32,
    supports_control_images=True,
    control_label="Use control/reference images (Task must be i2i)",
    extras=(
        ExtraOption("model_type", "dropdown", "Model Type", "full", "model", "full = base model; dev = distilled dev model.", choices=("full", "dev")),
        ExtraOption("task", "dropdown", "Task", "t2i", "model", "t2i = text-to-image; i2i = control/reference conditioned (also trains the visual encoder).", choices=("t2i", "i2i")),
        ExtraOption("noise_scale_start", "number", "Noise Scale Start", 8.0, "schedule", "8.0 for full, 7.5 for dev.", step=0.1),
        ExtraOption("noise_scale_end", "number", "Noise Scale End", 8.0, "schedule", "8.0 for full, 7.5 for dev.", step=0.1),
        ExtraOption("noise_clip_std", "number", "Noise Clip Std", 0.0, "schedule", "0.0 for full, 2.5 for dev.", step=0.1),
        ExtraOption("skip_t2i_visual_dummy", "checkbox", "Skip T2I Visual Dummy Forward", True, "advanced", "Faster single-GPU t2i training. Disable only for multi-GPU FSDP."),
        ExtraOption("dino_loss_weight", "number", "DINOv3 Perceptual Loss Weight", 0.0, "advanced", "0 disables. Needs the optional sensecraft[dinov3] package.", step=0.01),
        ExtraOption("dino_loss_every_n_steps", "number", "DINO Loss Every N Steps", 1, "advanced", "", minimum=1, step=1),
        ExtraOption("caching_teo_fp8_te", "checkbox", "Cache FP8 Text Embeddings", False, "caching", "Caches initial text token embeddings in FP8 (needs the checkpoint).", runtime=False),
        ExtraOption(
            "freeze_text_embeddings",
            "checkbox",
            "Freeze Text Token Embeddings",
            False,
            "full",
            "Keeps the 0.6B-parameter token-embedding table frozen (like not training a text encoder); every transformer "
            "layer still trains. Its gradient and optimizer update otherwise need several GB at once, so this lets full "
            "fine-tuning fit smaller GPUs.",
            full_only=True,
        ),
    ),
    sample_flags=("w", "h", "d", "n"),
    defaults={
        "dataset_resolution_width": 1024,
        "dataset_resolution_height": 1024,
        "timestep_sampling": "uniform",
        "weighting_scheme": "none",
        "discrete_flow_shift": 3.0,
        "sample_width": 1024,
        "sample_height": 1024,
        "sample_steps": 0,
        "sample_cfg_scale": 5.0,
        "max_train_epochs": 200,
        "save_every_n_epochs": 25,
    },
    lora_defaults={**COMMON_LORA_DEFAULTS, "learning_rate": 4e-5},
    full_defaults={**COMMON_FULL_DEFAULTS, "learning_rate": 5e-6},
    gui_only_keys=frozenset({"caching_teo_fp8_te"}),
    notes=(
        "HiDream-O1-Image is a pixel-space Qwen3-VL based model: one checkpoint, no VAE and no separate text "
        "encoder. The pixel cache stores 32x32 patch tokens, so image sides must be multiples of 32. Uniform "
        "timesteps match the official SFT recipe."
    ),
    build_latent_cache_args=_hidream_latent_args,
    build_text_cache_args=_hidream_text_args,
    validate=_hidream_validate,
)

def _k6_latent_args(params: dict) -> list[str]:
    args = [
        "--vae",
        str(params["vae"]),
        "--audio_vae",
        str(params["audio_vae"]),
        "--audio_vae_scaling_factor",
        str(float(params.get("audio_vae_scaling_factor") or 0.5302)),
        "--skip_broken",
    ]
    vae_memory_gb = float(params.get("caching_latent_vae_memory_gb") or 0)
    if vae_memory_gb > 0:
        args.extend(["--vae_tiling_memory_gb", f"{vae_memory_gb:g}"])
    return args


def _k6_text_args(params: dict) -> list[str]:
    args = [
        "--text_encoder_qwen",
        str(params["text_encoder_qwen"]),
        "--text_encoder_clip",
        str(params["text_encoder_clip"]),
        "--max_length",
        str(int(params.get("max_length") or 1024)),
    ]
    if params.get("caching_teo_quantized_qwen"):
        args.append("--quantized_qwen")
    return args


def _k6_validate(params: dict, full_finetune: bool) -> None:
    if params.get("convrot_int8") and (params.get("fp8_base") or params.get("fp8_scaled")):
        raise ValueError("Kandinsky 6: ConvRot INT8 and FP8 Base are mutually exclusive; choose one base quantization.")
    variant = str(params.get("model_variant") or "auto")
    dit = os.path.basename(str(params.get("dit") or "")).lower() + str(params.get("dit") or "").lower()
    lite = variant == "lite" or (variant == "auto" and "lite" in dit)
    if lite and int(params.get("blocks_to_swap") or 0) > 30:
        raise ValueError("Kandinsky 6 Lite allows at most 30 swapped blocks (Pro allows 58).")
    frames = int(params.get("num_frames") or 121)
    if round_frames_to_4n_plus_1(frames) != frames:
        raise ValueError(f"Kandinsky 6 frame counts must be 4*n+1 (e.g. 81, 97, 121); got {frames}.")
    if str(params.get("task") or "t2av") == "ti2av" and str(params.get("dataset_config_mode") or "") != "Use TOML File":
        raise ValueError(
            "Kandinsky 6 ti2av needs a JSONL dataset with a control_path (first-frame image) per clip. "
            "Use 'Use TOML File' with a dataset TOML that points to that JSONL."
        )
    if int(params.get("dataset_batch_size") or 1) != 1:
        raise ValueError("Kandinsky 6 trains with batch size 1; raise gradient accumulation instead.")


K6_LITE = "Kandinsky-6.0-Lite-5s-Diffusers"
KANDINSKY_6 = ImageArchSpec(
    key="kandinsky6",
    display_name="Kandinsky 6",
    slug="kandinsky6",
    network_module="networks.lora_kandinsky6",
    train_script="kandinsky6_train_network.py",
    full_train_script=None,
    latent_cache_script="kandinsky6_cache_latents.py",
    text_cache_script="kandinsky6_cache_text_encoder_outputs.py",
    model_folder="Training_Models_Kandinsky_6",
    model_paths=(
        ModelPath("dit", "Kandinsky 6 DiT (transformer)", f"{K6_LITE}/transformer/diffusion_pytorch_model.safetensors", "Regular Lite or Pro transformer (or a ConvRot INT8 prequantized file). Distilled checkpoints cannot be trained.", frozenset({"train"})),
        ModelPath("vae", "Video VAE Folder", f"{K6_LITE}/vae", "The snapshot's vae folder.", frozenset({"latent", "sample"}), is_dir=True),
        ModelPath("audio_vae", "Audio VAE Folder", f"{K6_LITE}/audio_vae", "The snapshot's audio_vae folder.", frozenset({"latent", "sample"}), is_dir=True),
        ModelPath("vocoder", "Vocoder Folder", f"{K6_LITE}/vocoder", "BigVGAN vocoder, needed only for training samples.", frozenset({"sample"}), is_dir=True),
        ModelPath("text_encoder_qwen", "Qwen2.5-VL Text Encoder Folder", f"{K6_LITE}/text_encoder", "Needed for text caching and samples.", frozenset({"text", "sample"}), is_dir=True),
        ModelPath("text_encoder_clip", "CLIP Text Encoder Folder", f"{K6_LITE}/text_encoder_2", "Needed for text caching and samples.", frozenset({"text", "sample"}), is_dir=True),
    ),
    max_blocks_to_swap=58,
    max_blocks_to_swap_full=58,
    preset_folders=("Kandinsky_6_Training_Configs",),
    download_bundles=("kandinsky6_lite",),
    supported_attention=frozenset({"sdpa"}),
    timestep_choices=("uniform", "shift", "sigmoid", "logsnr"),
    weighting_choices=("none",),
    resolution_step=16,
    dataset_kind="video",
    default_frames=121,
    extras=(
        ExtraOption("model_variant", "dropdown", "Model Variant", "auto", "model", "auto detects Lite/Pro from the DiT; lite allows 30 swapped blocks, pro 58.", choices=("auto", "lite", "pro")),
        ExtraOption("task", "dropdown", "Task", "t2av", "model", "t2av = text to video+audio; ti2av = first-frame conditioned (needs a JSONL dataset with control_path).", choices=("t2av", "ti2av")),
        ExtraOption("convrot_int8", "checkbox", "ConvRot INT8 Base (LoRA)", False, "model", "Quantize the frozen DiT to ConvRot INT8 while loading (needs Triton). Not combinable with FP8."),
        ExtraOption("convrot_int8_bwd", "dropdown", "ConvRot INT8 Backward", "bf16", "model", "bf16 = most accurate; int8 = faster gradients.", choices=("bf16", "int8")),
        ExtraOption("scheduler_scale", "number", "Scheduler Scale", 1.0, "schedule", "Shifted-uniform schedule scale applied to video and audio times.", step=0.1),
        ExtraOption("independent_time", "checkbox", "Independent Video/Audio Times", True, "schedule", "Sample separate diffusion times for video and audio (official default)."),
        ExtraOption("audio_loss_weight", "number", "Audio Loss Weight", 1.0, "schedule", "Scales the audio loss for clips cached with real audio.", step=0.05),
        ExtraOption("video_only", "checkbox", "Video Only (No Audio Supervision)", False, "schedule", "Disable the audio loss entirely."),
        ExtraOption("audio_vae_scaling_factor", "number", "Audio VAE Scaling Factor", 0.5302, "caching", "0.5302 for the regular Lite and Pro checkpoints.", step=0.0001),
        ExtraOption("max_length", "number", "Text Max Length", 1024, "caching", "Qwen2.5-VL token budget for captions.", minimum=64, step=1),
        ExtraOption("caching_teo_quantized_qwen", "checkbox", "NF4 Qwen for Text Caching", False, "caching", "bitsandbytes NF4 text encoder while caching (saves VRAM).", runtime=False),
        ExtraOption("caching_latent_vae_memory_gb", "number", "VAE Tiling Memory (GB)", 0, "caching", "Plan the video VAE's encode tiles for at most this much GPU memory; 0 = all free memory. Lower values cache latents with less VRAM (tiles are blended).", runtime=False, minimum=0, step=0.5),
    ),
    sample_flags=("w", "h", "f", "s", "d", "g", "n"),
    defaults={
        "dataset_resolution_width": 864,
        "dataset_resolution_height": 480,
        "num_frames": 121,
        "frame_extraction": "head",
        "frame_stride": 1,
        "frame_sample": 1,
        "max_frames": 241,
        "timestep_sampling": "uniform",
        "weighting_scheme": "none",
        "discrete_flow_shift": 1.0,
        "sample_width": 864,
        "sample_height": 480,
        "sample_num_frames": 121,
        "sample_steps": 30,
        "sample_cfg_scale": 5.0,
        "max_train_epochs": 100,
        "save_every_n_epochs": 10,
        "gradient_checkpointing": True,
    },
    lora_defaults={
        "network_dim": 32,
        "network_alpha": 32,
        "optimizer_type": "AdamW8bit",
        "optimizer_args": ["weight_decay=0.01"],
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "max_grad_norm": 1.0,
    },
    gui_only_keys=frozenset({"caching_teo_quantized_qwen"}),
    notes=(
        "Kandinsky 6 Lite/Pro (Oct 2026) jointly generates video with audio at 24 fps. Train t2av LoRAs from "
        "clips with sound, or appearance LoRAs from images (cached as silent one-frame videos). Frame counts "
        "must be 4n+1; 121 frames is about 5 seconds."
    ),
    build_latent_cache_args=_k6_latent_args,
    build_text_cache_args=_k6_text_args,
    validate=_k6_validate,
)

ARCHITECTURES: dict[str, ImageArchSpec] = {
    spec.key: spec for spec in (QWEN_IMAGE_21, ERNIE_IMAGE, MAGE_FLOW, HIDREAM_O1, KANDINSKY_6)
}


def get_spec(key: str) -> ImageArchSpec:
    try:
        return ARCHITECTURES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown image architecture: {key}") from exc


# ----------------------------------------------------------------------------------------------
# Parameter keys
# ----------------------------------------------------------------------------------------------

ACCELERATE_KEYS = [
    "mixed_precision",
    "num_cpu_threads_per_process",
    "num_processes",
    "num_machines",
    "multi_gpu",
    "gpu_ids",
    "main_process_port",
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_use_fullgraph",
    "dynamo_use_dynamic",
    "extra_accelerate_launch_args",
]
DATASET_KEYS = [
    "dataset_config_mode",
    "dataset_config",
    "parent_folder_path",
    "dataset_resolution_width",
    "dataset_resolution_height",
    "dataset_caption_extension",
    "create_missing_captions",
    "caption_strategy",
    "dataset_batch_size",
    "dataset_enable_bucket",
    "dataset_bucket_no_upscale",
    "dataset_cache_directory",
    "generated_toml_path",
]
CONTROL_KEYS = ["use_control_images", "control_directory_name"]
VIDEO_DATASET_KEYS = ["num_frames", "frame_extraction", "frame_stride", "frame_sample", "max_frames"]
MEMORY_KEYS = [
    "dit_dtype",
    "vae_dtype",
    "fp8_base",
    "fp8_scaled",
    "disable_numpy_memmap",
    "blocks_to_swap",
    "use_pinned_memory_for_block_swap",
    "block_swap_h2d_only",
    "block_swap_ring_size",
]
COMPILE_KEYS = ["compile", "compile_backend", "compile_mode", "compile_dynamic", "compile_fullgraph", "compile_cache_size_limit"]
SCHEDULE_KEYS = ["timestep_sampling", "weighting_scheme", "discrete_flow_shift", "sigmoid_scale", "mode_scale", "min_timestep", "max_timestep"]
TRAINING_KEYS = [
    "sdpa",
    "flash_attn",
    "sage_attn",
    "xformers",
    "split_attn",
    "use_legacy_sdpa",
    "max_train_steps",
    "max_train_epochs",
    "max_data_loader_n_workers",
    "persistent_data_loader_workers",
    "seed",
    "gradient_checkpointing",
    "gradient_checkpointing_cpu_offload",
    "gradient_accumulation_steps",
    "full_bf16",
    "full_fp16",
    "fused_backward_pass",
    "block_swap_optimizer_patch_params",
    "logging_dir",
    "log_with",
    "log_prefix",
    "log_tracker_name",
    "wandb_run_name",
    "log_tracker_config",
    "wandb_api_key",
    "log_config",
    "ddp_timeout",
    "ddp_gradient_as_bucket_view",
    "ddp_static_graph",
]
SAMPLE_KEYS = [
    "sample_every_n_steps",
    "sample_every_n_epochs",
    "sample_at_first",
    "sample_prompts",
    "sample_output_dir",
    "disable_prompt_enhancement",
    "sample_width",
    "sample_height",
    "sample_steps",
    "sample_seed",
    "sample_negative_prompt",
    "sample_cfg_scale",
]
CACHE_KEYS = [
    "cache_latents",
    "caching_latent_device",
    "caching_latent_batch_size",
    "caching_latent_num_workers",
    "caching_latent_skip_existing",
    "caching_latent_keep_cache",
    "cache_text_encoder_outputs",
    "caching_teo_device",
    "caching_teo_batch_size",
    "caching_teo_num_workers",
    "caching_teo_skip_existing",
    "caching_teo_keep_cache",
]
OPTIMIZER_KEYS = [
    "optimizer_type",
    "optimizer_args",
    "learning_rate",
    "max_grad_norm",
    "lr_scheduler",
    "lr_warmup_steps",
    "lr_decay_steps",
    "lr_scheduler_num_cycles",
    "lr_scheduler_power",
    "lr_scheduler_timescale",
    "lr_scheduler_min_lr_ratio",
    "lr_scheduler_type",
    "lr_scheduler_args",
]
NETWORK_KEYS = [
    "no_metadata",
    "network_weights",
    "network_module",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "training_comment",
    "dim_from_weights",
    "scale_weight_norms",
    "base_weights",
    "base_weights_multiplier",
]
SAVE_KEYS = [
    "output_dir",
    "output_name",
    "resume",
    "save_precision",
    "save_every_n_epochs",
    "save_last_n_epochs",
    "save_every_n_steps",
    "save_last_n_steps",
    "save_last_n_epochs_state",
    "save_last_n_steps_state",
    "save_state",
    "save_state_on_train_end",
    "mem_eff_save",
]
METADATA_KEYS = [
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "metadata_reso",
    "metadata_arch",
]
HUGGINGFACE_KEYS = [
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "save_state_to_huggingface",
    "resume_from_huggingface",
    "async_upload",
]

GUI_ONLY_KEYS = {
    "additional_parameters",
    "debug_mode",
    "num_cpu_threads_per_process",
    "num_processes",
    "num_machines",
    "multi_gpu",
    "gpu_ids",
    "main_process_port",
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_use_fullgraph",
    "dynamo_use_dynamic",
    "extra_accelerate_launch_args",
    "dataset_config_mode",
    "parent_folder_path",
    "dataset_resolution_width",
    "dataset_resolution_height",
    "dataset_caption_extension",
    "create_missing_captions",
    "caption_strategy",
    "dataset_batch_size",
    "dataset_enable_bucket",
    "dataset_bucket_no_upscale",
    "dataset_cache_directory",
    "generated_toml_path",
    "use_control_images",
    "control_directory_name",
    "sample_output_dir",
    "disable_prompt_enhancement",
    "sample_width",
    "sample_height",
    "sample_steps",
    "sample_seed",
    "sample_negative_prompt",
    "sample_cfg_scale",
    "sample_num_frames",
    "num_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
}


def param_keys(spec: ImageArchSpec) -> list[str]:
    keys = list(ACCELERATE_KEYS)
    keys += ["additional_parameters", "debug_mode", "training_mode"]
    keys += DATASET_KEYS
    if spec.dataset_kind == "video":
        keys += VIDEO_DATASET_KEYS
    if spec.supports_control_images:
        keys += CONTROL_KEYS
    keys += [item.key for item in spec.model_paths]
    keys += MEMORY_KEYS
    keys += COMPILE_KEYS
    keys += SCHEDULE_KEYS
    keys += TRAINING_KEYS
    keys += SAMPLE_KEYS
    if spec.dataset_kind == "video":
        keys.append("sample_num_frames")
    keys += CACHE_KEYS
    keys += [extra.key for extra in spec.extras]
    keys += OPTIMIZER_KEYS + NETWORK_KEYS + SAVE_KEYS + METADATA_KEYS + HUGGINGFACE_KEYS
    duplicates = {key for key in keys if keys.count(key) > 1}
    if duplicates:
        raise AssertionError(f"{spec.display_name}: duplicate parameter keys {sorted(duplicates)}")
    return keys


# ----------------------------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------------------------


def _default_model_paths(spec: ImageArchSpec) -> dict[str, str]:
    model_dir = os.path.normpath(os.path.join(scriptdir, "..", spec.model_folder))
    defaults = {}
    for item in spec.model_paths:
        candidate = os.path.join(model_dir, item.filename)
        defaults[item.key] = candidate if os.path.exists(candidate) else ""
    return defaults


def architecture_defaults(spec: ImageArchSpec, full_finetune: bool = False) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mixed_precision": "bf16",
        "num_cpu_threads_per_process": 1,
        "num_processes": 1,
        "num_machines": 1,
        "multi_gpu": False,
        "gpu_ids": "0",
        "main_process_port": 0,
        "dynamo_backend": "no",
        "dynamo_mode": "",
        "dynamo_use_fullgraph": False,
        "dynamo_use_dynamic": False,
        "extra_accelerate_launch_args": "",
        "additional_parameters": "",
        "debug_mode": "None",
        "training_mode": FULL_FINE_TUNING_MODE if full_finetune else LORA_TRAINING_MODE,
        "dataset_config_mode": "Generate from Folder Structure",
        "dataset_config": "",
        "parent_folder_path": "",
        "dataset_resolution_width": 1024,
        "dataset_resolution_height": 1024,
        "dataset_caption_extension": ".txt",
        "create_missing_captions": True,
        "caption_strategy": "folder_name",
        "dataset_batch_size": 1,
        "dataset_enable_bucket": True,
        "dataset_bucket_no_upscale": False,
        "dataset_cache_directory": "cache_dir",
        "generated_toml_path": "",
        "use_control_images": False,
        "control_directory_name": "control_images",
        "num_frames": spec.default_frames,
        "frame_extraction": "head",
        "frame_stride": 1,
        "frame_sample": 1,
        "max_frames": max(spec.default_frames, 241),
        "sample_num_frames": spec.default_frames,
        "dit_dtype": "bfloat16",
        "vae_dtype": "bfloat16",
        "fp8_base": False,
        "fp8_scaled": False,
        "disable_numpy_memmap": False,
        "blocks_to_swap": 0,
        "use_pinned_memory_for_block_swap": False,
        "block_swap_h2d_only": False,
        "block_swap_ring_size": 2,
        "compile": False,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_dynamic": "auto",
        "compile_fullgraph": False,
        "compile_cache_size_limit": 0,
        "timestep_sampling": "shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 3.0,
        "sigmoid_scale": 1.0,
        "mode_scale": 1.29,
        "min_timestep": 0,
        "max_timestep": 1000,
        "sdpa": True,
        "flash_attn": False,
        "sage_attn": False,
        "xformers": False,
        "split_attn": False,
        "use_legacy_sdpa": False,
        "max_train_steps": 80000,
        "max_train_epochs": 200,
        "max_data_loader_n_workers": 2,
        "persistent_data_loader_workers": True,
        "seed": 42,
        "gradient_checkpointing": True,
        "gradient_checkpointing_cpu_offload": False,
        "gradient_accumulation_steps": 1,
        "full_bf16": False,
        "full_fp16": False,
        "fused_backward_pass": False,
        "block_swap_optimizer_patch_params": False,
        "logging_dir": "",
        "log_with": "",
        "sample_every_n_steps": 0,
        "sample_every_n_epochs": 0,
        "sample_at_first": False,
        "sample_prompts": "",
        "sample_output_dir": "",
        "disable_prompt_enhancement": False,
        "sample_width": 1024,
        "sample_height": 1024,
        "sample_steps": 28,
        "sample_seed": 42,
        "sample_negative_prompt": "",
        "sample_cfg_scale": 4.0,
        "cache_latents": True,
        "caching_latent_device": "cuda",
        "caching_latent_batch_size": 1,
        "caching_latent_num_workers": 2,
        "caching_latent_skip_existing": True,
        "caching_latent_keep_cache": True,
        "cache_text_encoder_outputs": True,
        "caching_teo_device": "cuda",
        "caching_teo_batch_size": 1,
        "caching_teo_num_workers": 2,
        "caching_teo_skip_existing": True,
        "caching_teo_keep_cache": True,
        "lr_warmup_steps": 0,
        "network_module": spec.network_module,
        "network_dropout": 0,
        "output_dir": os.path.join(scriptdir, "outputs", spec.slug),
        "output_name": f"my_{spec.slug}_{'fine_tune' if full_finetune else 'lora'}",
        "save_precision": "bf16",
        "save_every_n_epochs": 10,
        "save_state": False,
        "save_state_on_train_end": False,
        "mem_eff_save": False,
    }
    for extra in spec.extras:
        defaults[extra.key] = extra.default
    defaults.update(spec.defaults)
    defaults.update(spec.full_defaults if full_finetune else spec.lora_defaults)
    defaults.update(_default_model_paths(spec))
    return defaults


def _prepare_config(spec: ImageArchSpec, config) -> GUIConfig:
    defaults = architecture_defaults(spec)
    if isinstance(config, dict):
        data = dict(config)
        for key, value in defaults.items():
            data.setdefault(key, value)
        return type("GUIConfigAdapter", (), {"config": data, "get": data.get})()
    if not hasattr(config, "config"):
        config = GUIConfig()
    if not getattr(config, "config", None):
        config.config = defaults.copy()
    else:
        for key, value in defaults.items():
            config.config.setdefault(key, value)
    return config


# ----------------------------------------------------------------------------------------------
# Normalization and workflow
# ----------------------------------------------------------------------------------------------


@dataclass
class ImageWorkflow:
    parameters: list[tuple[str, object]]
    config_path: str
    latent_cache_command: list[str] | None
    text_cache_command: list[str] | None
    train_command: list[str]

    @property
    def commands(self) -> list[list[str]]:
        commands = [command for command in (self.latent_cache_command, self.text_cache_command) if command]
        commands.append(self.train_command)
        return commands


executors: dict[str, CommandExecutor] = {}
train_state_values = {key: time.time() for key in ARCHITECTURES}


def _trainer_script_path(filename: str) -> str:
    return os.path.normpath(os.path.join(scriptdir, "musubi-tuner", "src", "musubi_tuner", filename))


def _set(param_dict: dict, parameters: list, key: str, value):
    param_dict[key] = value
    return _replace_parameter(parameters, key, value)


def _require_path(param_dict: dict, key: str, label: str) -> str:
    value = str(param_dict.get(key) or "").strip()
    if not value:
        raise ValueError(f"{label} is required.")
    if not os.path.exists(value):
        raise ValueError(f"{label} does not exist: {value}")
    return os.path.abspath(value)


def normalize_parameters(
    spec: ImageArchSpec,
    parameters: list[tuple[str, object]],
    *,
    validate_paths: bool = True,
) -> tuple[dict, list[tuple[str, object]]]:
    param_dict = dict(parameters)
    if not spec.supports_full_finetune and is_full_fine_tuning(param_dict.get("training_mode")):
        raise ValueError(f"{spec.display_name} supports LoRA training only in this backend.")
    param_dict, parameters, full_finetune = normalize_image_training_parameters(parameters)

    if full_finetune:
        parameters = _set(param_dict, parameters, "network_module", "")
    else:
        parameters = _set(param_dict, parameters, "network_module", spec.network_module)
        parameters = _set(param_dict, parameters, "dit_dtype", "bfloat16")

    save_precision = str(param_dict.get("save_precision") or "bf16").lower()
    save_precision = "fp32" if save_precision == "float" else save_precision
    if save_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("save_precision must be bf16, fp16, or fp32.")
    parameters = _set(param_dict, parameters, "save_precision", save_precision)

    mixed_precision = str(param_dict.get("mixed_precision") or "bf16")
    if param_dict.get("full_bf16") and mixed_precision != "bf16":
        raise ValueError("Full BF16 training requires mixed_precision='bf16'.")
    if param_dict.get("full_fp16") and mixed_precision != "fp16":
        raise ValueError("Full FP16 training requires mixed_precision='fp16'.")

    attention = [name for name in ("sdpa", "flash_attn", "sage_attn", "xformers") if bool(param_dict.get(name))]
    unsupported = [name for name in attention if name not in spec.supported_attention]
    if unsupported:
        raise ValueError(f"{spec.display_name} training does not support: {', '.join(unsupported)}.")
    if not attention:
        parameters = _set(param_dict, parameters, "sdpa", True)
    elif len(attention) > 1:
        raise ValueError("Select exactly one attention backend.")

    try:
        blocks_to_swap = int(param_dict.get("blocks_to_swap", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("blocks_to_swap must be an integer.") from exc
    maximum = spec.max_blocks_to_swap_full if full_finetune else spec.max_blocks_to_swap
    if not 0 <= blocks_to_swap <= maximum:
        raise ValueError(f"blocks_to_swap for {spec.display_name} must be between 0 and {maximum}.")
    parameters = _set(param_dict, parameters, "blocks_to_swap", blocks_to_swap)
    if not spec.supports_h2d_block_swap and param_dict.get("block_swap_h2d_only"):
        parameters = _set(param_dict, parameters, "block_swap_h2d_only", False)
    validate_block_swap_options(param_dict, lora_training=not full_finetune)

    if spec.supports_fp8_scaled and not full_finetune:
        if bool(param_dict.get("fp8_scaled")) and not bool(param_dict.get("fp8_base")):
            parameters = _set(param_dict, parameters, "fp8_base", True)
    else:
        parameters = _set(param_dict, parameters, "fp8_base", False)
        parameters = _set(param_dict, parameters, "fp8_scaled", False)

    for key in ("caching_latent_batch_size", "caching_latent_num_workers", "caching_teo_batch_size", "caching_teo_num_workers"):
        value = param_dict.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"{key} must be at least 1.")

    for extra in spec.extras:
        if extra.kind == "number" and param_dict.get(extra.key) is not None:
            value = float(param_dict[extra.key])
            parameters = _set(param_dict, parameters, extra.key, int(value) if float(value).is_integer() and (extra.step or 1) >= 1 else value)

    base_weights = _coerce_cli_list(param_dict.get("base_weights"))
    multipliers = _coerce_float_list(param_dict.get("base_weights_multiplier"))
    parameters = _set(param_dict, parameters, "base_weights", base_weights or None)
    parameters = _set(param_dict, parameters, "base_weights_multiplier", multipliers or None)

    debug_mode = str(param_dict.get("debug_mode") or "")
    if debug_mode in DEBUG_LOGGING_MODES and not str(param_dict.get("log_with") or "").strip():
        # Guardrail 1: route debug logging through log_with so the runtime TOML gets a
        # logging_dir under output_dir/logs (SaveConfigFileToRun) instead of a bare CLI flag.
        parameters = _set(param_dict, parameters, "log_with", DEBUG_LOGGING_MODES[debug_mode])

    if spec.validate:
        spec.validate(param_dict, full_finetune)

    if not validate_paths:
        return param_dict, parameters

    dataset_config = str(param_dict.get("dataset_config") or "").strip()
    if not dataset_config or not os.path.isfile(dataset_config):
        raise ValueError(f"Dataset config does not exist: {dataset_config or '(empty)'}")
    parameters = _set(param_dict, parameters, "dataset_config", os.path.abspath(dataset_config))

    sampling = bool(str(param_dict.get("sample_prompts") or "").strip()) and (
        int(param_dict.get("sample_every_n_steps") or 0) > 0
        or int(param_dict.get("sample_every_n_epochs") or 0) > 0
        or bool(param_dict.get("sample_at_first"))
    )
    needed = {"train"}
    if param_dict.get("cache_latents", True):
        needed.add("latent")
    if param_dict.get("cache_text_encoder_outputs", True):
        needed.add("text")
    if sampling:
        needed.add("sample")
    for item in spec.model_paths:
        value = str(param_dict.get(item.key) or "").strip()
        if item.required_for & needed:
            parameters = _set(param_dict, parameters, item.key, _require_path(param_dict, item.key, item.label))
        elif value:
            if not os.path.exists(value):
                raise ValueError(f"{item.label} does not exist: {value}")
            parameters = _set(param_dict, parameters, item.key, os.path.abspath(value))

    sample_prompts = str(param_dict.get("sample_prompts") or "").strip()
    if sample_prompts:
        if not os.path.isfile(sample_prompts):
            raise ValueError(f"Sample prompt file does not exist: {sample_prompts}")
        parameters = _set(param_dict, parameters, "sample_prompts", os.path.abspath(sample_prompts))

    output_dir = str(param_dict.get("output_dir") or "").strip()
    if not output_dir:
        raise ValueError("Output directory is required.")
    if not str(param_dict.get("output_name") or "").strip():
        raise ValueError("Output name is required.")
    parameters = _set(param_dict, parameters, "output_dir", os.path.abspath(output_dir))
    return param_dict, parameters


def _cache_common(spec: ImageArchSpec, param_dict: dict, prefix: str) -> list[str]:
    args = []
    for suffix, flag in (("batch_size", "--batch_size"), ("num_workers", "--num_workers")):
        value = param_dict.get(f"caching_{prefix}_{suffix}")
        if value is not None:
            args.extend([flag, str(int(value))])
    if param_dict.get(f"caching_{prefix}_skip_existing"):
        args.append("--skip_existing")
    if param_dict.get(f"caching_{prefix}_keep_cache"):
        args.append("--keep_cache")
    return args


def build_cache_commands(
    spec: ImageArchSpec,
    param_dict: dict,
    *,
    python_cmd: str | None = None,
) -> tuple[list[str] | None, list[str] | None]:
    python_cmd = python_cmd or sys.executable
    dataset_config = str(param_dict["dataset_config"])
    gpu_ids = param_dict.get("gpu_ids")
    latent_command = None
    if bool(param_dict.get("cache_latents", True)):
        latent_command = [
            python_cmd,
            _trainer_script_path(spec.latent_cache_script),
            "--dataset_config",
            dataset_config,
            "--device",
            _resolve_device(param_dict.get("caching_latent_device"), gpu_ids),
        ]
        latent_command += _cache_common(spec, param_dict, "latent")
        if spec.build_latent_cache_args:
            latent_command += spec.build_latent_cache_args(param_dict)
    text_command = None
    if bool(param_dict.get("cache_text_encoder_outputs", True)):
        text_command = [
            python_cmd,
            _trainer_script_path(spec.text_cache_script),
            "--dataset_config",
            dataset_config,
            "--device",
            _resolve_device(param_dict.get("caching_teo_device"), gpu_ids),
        ]
        text_command += _cache_common(spec, param_dict, "teo")
        if spec.build_text_cache_args:
            text_command += spec.build_text_cache_args(param_dict)
    return latent_command, text_command


def run_config_exclusions(spec: ImageArchSpec, parameters: list[tuple[str, object]]) -> list[str]:
    param_dict = dict(parameters)
    full_finetune = is_full_fine_tuning(param_dict.get("training_mode"))
    exclusions = set(GUI_ONLY_KEYS)
    exclusions.update(spec.gui_only_keys)
    exclusions.update(training_mode_runtime_exclusions(param_dict.get("training_mode")))
    exclusions.update(key for key, _ in parameters if key.startswith("caching_") or key in {"cache_latents", "cache_text_encoder_outputs"})
    for extra in spec.extras:
        if not extra.runtime or (extra.lora_only and full_finetune) or (extra.full_only and not full_finetune):
            exclusions.add(extra.key)
    if not full_finetune:
        exclusions.add("dit_dtype")  # LoRA trainers keep the DiT in BF16
    if not spec.supports_fp8_scaled or full_finetune:
        exclusions.update({"fp8_base", "fp8_scaled"})
    if not spec.model_path("vae"):
        exclusions.add("vae_dtype")
    if str(param_dict.get("timestep_sampling") or "") not in {"sigmoid", "shift", "qinglong_flux", "qinglong_qwen"}:
        exclusions.add("sigmoid_scale")
    if str(param_dict.get("weighting_scheme") or "") != "mode":
        exclusions.add("mode_scale")
    return sorted(exclusions)


def _debug_cli_flags(debug_mode: object) -> str:
    return DEBUG_CLI_FLAGS.get(str(debug_mode or ""), "")


def prepare_workflow(
    spec_key: str,
    parameters: list[tuple[str, object]],
    config_path: str,
    *,
    validate_paths: bool = True,
    python_cmd: str | None = None,
) -> ImageWorkflow:
    parameters = strip_path_parameters(parameters)  # quoted / padded paths (Explorer's "Copy as path")
    spec = get_spec(spec_key)
    param_dict, parameters = normalize_parameters(spec, parameters, validate_paths=validate_paths)
    python_cmd = python_cmd or sys.executable
    if validate_paths:
        os.makedirs(str(param_dict["output_dir"]), exist_ok=True)

    full_finetune = is_full_fine_tuning(param_dict.get("training_mode"))
    mandatory = ["dataset_config", "dit", "output_dir", "output_name"]
    if not full_finetune:
        mandatory.append("network_module")
    SaveConfigFileToRun(
        parameters=parameters,
        file_path=config_path,
        exclusion=run_config_exclusions(spec, parameters),
        mandatory_keys=mandatory,
    )
    latent_command, text_command = build_cache_commands(spec, param_dict, python_cmd=python_cmd)

    train_command = AccelerateLaunch.run_cmd(
        run_cmd=_find_accelerate_launch(python_cmd),
        dynamo_backend=param_dict.get("dynamo_backend"),
        dynamo_mode=param_dict.get("dynamo_mode"),
        dynamo_use_fullgraph=param_dict.get("dynamo_use_fullgraph"),
        dynamo_use_dynamic=param_dict.get("dynamo_use_dynamic"),
        num_processes=param_dict.get("num_processes"),
        num_machines=param_dict.get("num_machines"),
        multi_gpu=param_dict.get("multi_gpu"),
        gpu_ids=param_dict.get("gpu_ids"),
        main_process_port=param_dict.get("main_process_port"),
        num_cpu_threads_per_process=param_dict.get("num_cpu_threads_per_process"),
        mixed_precision=param_dict.get("mixed_precision"),
        extra_accelerate_launch_args=param_dict.get("extra_accelerate_launch_args"),
    )
    script = spec.full_train_script if full_finetune else spec.train_script
    train_command.extend([_trainer_script_path(script), "--config_file", config_path])
    additional = str(param_dict.get("additional_parameters") or "").strip()
    debug = _debug_cli_flags(param_dict.get("debug_mode"))
    if debug:
        additional = f"{additional} {debug}".strip()
    train_command = run_cmd_advanced_training(run_cmd=train_command, additional_parameters=additional)
    return ImageWorkflow(parameters, config_path, latent_command, text_command, train_command)


def create_dataset_config(spec: ImageArchSpec, param_dict: dict) -> tuple[str, list[str]]:
    parent = strip_path_text(param_dict.get("parent_folder_path"))
    if not parent or not os.path.isdir(parent):
        raise ValueError(f"Dataset parent folder does not exist: {parent or '(empty)'}")
    save_dir = str(param_dict.get("output_dir") or "").strip() or parent
    os.makedirs(save_dir, exist_ok=True)
    step = spec.resolution_step
    width = int(param_dict.get("dataset_resolution_width") or 1024)
    height = int(param_dict.get("dataset_resolution_height") or 1024)
    width, height = max(step, width - width % step), max(step, height - height % step)
    use_controls = bool(param_dict.get("use_control_images")) and spec.supports_control_images
    control_dir = str(param_dict.get("control_directory_name") or "control_images").strip() if use_controls else f"__{spec.slug}_no_control__"
    if spec.dataset_kind == "video":
        config, messages = generate_kandinsky6_dataset_config_from_folders(
            parent_folder=parent,
            resolution=(width, height),
            caption_extension=str(param_dict.get("dataset_caption_extension") or ".txt"),
            create_missing_captions=bool(param_dict.get("create_missing_captions", True)),
            caption_strategy=str(param_dict.get("caption_strategy") or "folder_name"),
            enable_bucket=bool(param_dict.get("dataset_enable_bucket", True)),
            bucket_no_upscale=bool(param_dict.get("dataset_bucket_no_upscale", False)),
            cache_directory_name=str(param_dict.get("dataset_cache_directory") or "cache_dir"),
            num_frames=int(param_dict.get("num_frames") or spec.default_frames),
            frame_extraction=str(param_dict.get("frame_extraction") or "head"),
            frame_stride=int(param_dict.get("frame_stride") or 1),
            frame_sample=int(param_dict.get("frame_sample") or 1),
            max_frames=int(param_dict.get("max_frames") or 241),
        )
        path = os.path.join(save_dir, f"{spec.slug}_dataset_{datetime.now().strftime('%Y%m%d-%H%M%S')}.toml")
        save_dataset_config(config, path)
        valid, validation_messages = validate_dataset_config(path)
        messages.extend(validation_messages)
        if not valid:
            raise ValueError("\n".join(messages))
        messages.append(f"Saved dataset TOML: {path}")
        return os.path.abspath(path), messages
    config, messages = generate_dataset_config_from_folders(
        parent_folder=parent,
        resolution=(width, height),
        caption_extension=str(param_dict.get("dataset_caption_extension") or ".txt"),
        create_missing_captions=bool(param_dict.get("create_missing_captions", True)),
        caption_strategy=str(param_dict.get("caption_strategy") or "folder_name"),
        batch_size=int(param_dict.get("dataset_batch_size") or 1),
        enable_bucket=bool(param_dict.get("dataset_enable_bucket", True)),
        bucket_no_upscale=bool(param_dict.get("dataset_bucket_no_upscale", False)),
        cache_directory_name=str(param_dict.get("dataset_cache_directory") or "cache_dir"),
        control_directory_name=control_dir,
    )
    if use_controls:
        # an edit / i2i run without reference images would only fail later, while caching or training
        missing = [entry.get("image_directory", "?") for entry in config.get("datasets", []) if not entry.get("control_directory")]
        if missing:
            raise ValueError(
                f"Reference images are enabled ('{spec.control_label}') but these dataset folders have no "
                f"'{control_dir}' sub-folder: {', '.join(missing)}. Put the reference images in "
                f"<dataset folder>/{control_dir}, named like their target image (image_0.png, image_1.png ...), "
                "or turn reference images off."
            )
    path = os.path.join(save_dir, f"{spec.slug}_dataset_{datetime.now().strftime('%Y%m%d-%H%M%S')}.toml")
    save_dataset_config(config, path)
    valid, validation_messages = validate_dataset_config(path)
    messages.extend(validation_messages)
    if not valid:
        raise ValueError("\n".join(messages))
    if use_controls:
        messages.append(f"Control/reference images: '{control_dir}' sub-folders are paired with the targets.")
    messages.append(f"Saved dataset TOML: {path}")
    return os.path.abspath(path), messages


def generate_dataset_toml(spec_key: str, *values):
    spec = get_spec(spec_key)
    keys = [
        "parent_folder_path",
        "dataset_resolution_width",
        "dataset_resolution_height",
        "dataset_caption_extension",
        "create_missing_captions",
        "caption_strategy",
        "dataset_batch_size",
        "dataset_enable_bucket",
        "dataset_bucket_no_upscale",
        "dataset_cache_directory",
        "output_dir",
        "use_control_images",
        "control_directory_name",
        "num_frames",
        "frame_extraction",
        "frame_stride",
        "frame_sample",
        "max_frames",
    ]
    param_dict = dict(zip(keys, values))
    try:
        path, messages = create_dataset_config(spec, param_dict)
        return path, path, "\n".join(messages)
    except Exception as exc:
        message = f"Failed to generate dataset TOML: {exc}"
        log.error(message)
        return gr.update(), gr.update(), message


def enhance_sample_prompts(spec: ImageArchSpec, param_dict: dict, parameters: list) -> tuple[dict, list]:
    prompt_path = str(param_dict.get("sample_prompts") or "").strip()
    if not prompt_path or bool(param_dict.get("disable_prompt_enhancement")) or not os.path.isfile(prompt_path):
        return param_dict, parameters
    values = {
        "w": int(param_dict.get("sample_width") or 1024),
        "h": int(param_dict.get("sample_height") or 1024),
        "s": int(param_dict.get("sample_steps") or 0),
        "d": int(param_dict.get("sample_seed") if param_dict.get("sample_seed") is not None else 42),
        "l": float(param_dict.get("sample_cfg_scale") or 0),
        "g": float(param_dict.get("sample_cfg_scale") or 0),
        "f": round_frames_to_4n_plus_1(int(param_dict.get("sample_num_frames") or spec.default_frames)),
        "n": str(param_dict.get("sample_negative_prompt") or ""),
    }

    def has_flag(line: str, flag: str) -> bool:
        return re.search(rf"(?<!\S)--{re.escape(flag)}(?:\s+|=)", line, re.IGNORECASE) is not None

    lines = []
    with open(prompt_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lines.append(line)
                continue
            enhanced = re.sub(r"(?<!\S)-(?P<flag>w|h|f|s|d|l|g|n)(?=\s|=)", r"--\g<flag>", stripped, flags=re.IGNORECASE)
            for flag in spec.sample_flags:
                value = values[flag]
                if flag in ("s", "l", "g") and not value:
                    continue
                if flag == "n" and (value == "" or values["l"] <= 1.0 and ("l" in spec.sample_flags or "g" in spec.sample_flags)):
                    continue
                if not has_flag(enhanced, flag):
                    enhanced += f" --{flag} {value}"
            lines.append(enhanced)
    output_dir = str(param_dict.get("sample_output_dir") or "").strip() or str(param_dict.get("output_dir") or "").strip()
    if not output_dir:
        return param_dict, parameters
    os.makedirs(output_dir, exist_ok=True)
    name = str(param_dict.get("output_name") or spec.slug)
    path = os.path.abspath(os.path.join(output_dir, f"{name}_enhanced_prompts_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Generated by SECourses Musubi Trainer\n")
        handle.writelines(f"{line}\n" for line in lines)
    parameters = _set(param_dict, parameters, "sample_prompts", path)
    return param_dict, parameters


# ----------------------------------------------------------------------------------------------
# Save / load / train actions
# ----------------------------------------------------------------------------------------------


def save_configuration(spec_key: str, save_as: bool, file_path: str, parameters: list):
    spec = get_spec(spec_key)
    original = file_path
    if save_as or not file_path:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")
    if not file_path:
        return original, gr.update(value="No file selected.", visible=True)
    if not file_path.lower().endswith(".toml"):
        file_path = os.path.splitext(file_path)[0] + ".toml"
    if is_shipped_preset(file_path):
        message = "Shipped GPU presets are read-only. Type a new file name (for example in your output folder) and click Save."
        gr.Warning(message)
        return original, gr.update(value=message, visible=True)
    try:
        _, parameters = normalize_parameters(spec, parameters, validate_paths=False)
        SaveConfigFile(parameters=parameters, file_path=file_path, exclusion=["file_path", "save_as", "headless", "print_only"])
        message = f"Configuration saved: {os.path.basename(file_path)}"
        gr.Info(message)
        return file_path, gr.update(value=message, visible=True)
    except Exception as exc:
        message = f"Failed to save configuration: {exc}"
        log.exception(message)
        gr.Warning(message)
        return original, gr.update(value=message, visible=True)


def config_value_for_component(spec: ImageArchSpec, key: str, value, default, data: dict | None = None):
    if key == "training_mode":
        if not spec.supports_full_finetune:
            return LORA_TRAINING_MODE
        if data is not None:
            return infer_training_mode_from_loaded_config(data)
        return normalize_training_mode(value)
    if key == "network_module":
        return spec.network_module
    if key == "save_precision" and value in (None, ""):
        return "bf16"
    if key == "blocks_to_swap":
        try:
            maximum = max(spec.max_blocks_to_swap, spec.max_blocks_to_swap_full)
            return max(0, min(int(value or 0), maximum))
        except (TypeError, ValueError):
            return default
    if isinstance(value, list):
        if key in {"optimizer_args", "lr_scheduler_args", "network_args", "base_weights", "base_weights_multiplier"}:
            return text_for_list(key, value)
        if not isinstance(default, list):
            return value[0] if value else default
    if value is None and isinstance(default, (int, float, bool)):
        return default
    return value


def open_configuration(spec_key: str, ask_for_file: bool, file_path: str, parameters: list):
    spec = get_spec(spec_key)
    original = file_path
    current = [value for _, value in parameters]
    if ask_for_file:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")
    if not file_path:
        return tuple([original, gr.update(value="", visible=False)] + current)
    if not os.path.isfile(file_path):
        message = (
            f"New configuration will be created: {os.path.basename(file_path)}"
            if ask_for_file
            else f"Configuration does not exist: {file_path}"
        )
        if not ask_for_file:
            gr.Warning(message)
        return tuple([file_path if ask_for_file else original, gr.update(value=message, visible=True)] + current)
    try:
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            data = toml.load(handle)
        values = []
        for key, default in parameters:
            value = resolve_portable_model_value(key, data.get(key, default))
            values.append(config_value_for_component(spec, key, value, default, data))
        message = f"Configuration loaded: {os.path.basename(file_path)}"
        gr.Info(message)
        return tuple([file_path, gr.update(value=message, visible=True)] + values)
    except Exception as exc:
        message = f"Failed to load configuration: {exc}"
        log.exception(message)
        gr.Warning(message)
        return tuple([original, gr.update(value=message, visible=True)] + current)


def train_model(spec_key: str, headless: bool, print_only: bool, parameters: list):
    parameters = strip_path_parameters(parameters)  # quoted / padded paths (Explorer's "Copy as path")
    spec = get_spec(spec_key)
    param_dict = dict(parameters)
    validate_automagic_configuration(param_dict, warning_callback=None if headless else gr.Warning)

    if str(param_dict.get("dataset_config_mode") or "") == "Generate from Folder Structure":
        generated = str(param_dict.get("generated_toml_path") or "").strip()
        # reuse the generated TOML only while it still describes the chosen folder and resolution
        if not generated or not os.path.isfile(generated) or not dataset_toml_matches(
            generated, param_dict.get("parent_folder_path"),
            (param_dict.get("dataset_resolution_width"), param_dict.get("dataset_resolution_height")),
        ):
            generated, _ = create_dataset_config(spec, param_dict)
        parameters = _set(param_dict, parameters, "dataset_config", generated)
        parameters = _set(param_dict, parameters, "generated_toml_path", generated)

    param_dict, parameters = enhance_sample_prompts(spec, param_dict, parameters)
    output_dir = str(param_dict.get("output_dir") or "").strip()
    if not output_dir:
        raise ValueError("Output directory is required.")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_name = str(param_dict.get("output_name") or spec.slug)
    config_path = os.path.join(output_dir, f"{output_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.toml")
    workflow = prepare_workflow(spec_key, parameters, config_path)

    if print_only:
        for command in workflow.commands[:-1]:
            print_command_and_toml(command, "")
        print_command_and_toml(workflow.train_command, workflow.config_path)
        gr.Info(f"{spec.display_name}: workflow printed to the console (nothing was started).")
        return None

    executor = executors.get(spec_key)
    if executor is None:
        raise RuntimeError(f"{spec.display_name} command executor is not initialized.")
    script_path, script_content = _modern_build_workflow_script(spec, workflow)
    save_executed_script(script_content=script_content, config_name=output_name, script_type=spec.slug)
    env = setup_environment(compile_requested=requires_native_compile_toolchain(workflow.parameters))
    command = [script_path] if platform.system() == "Windows" else ["bash", script_path]
    executor.execute_command(run_cmd=command, env=env, shell=platform.system() == "Windows")
    train_state_values[spec_key] = time.time()
    return (
        gr.Button(visible=False or headless),
        gr.Row(visible=True),
        gr.Button(interactive=True),
        gr.Textbox(value=f"{spec.display_name} workflow in progress..."),
        gr.Textbox(value=train_state_values[spec_key]),
    )


def gui_actions(spec_key: str, action: str, ask_for_file: bool, config_file_name: str, headless: bool, print_only: bool, *args):
    spec = get_spec(spec_key)
    parameters = list(zip(param_keys(spec), args))
    if action == "open_configuration":
        return open_configuration(spec_key, ask_for_file, config_file_name, parameters)
    if action == "save_configuration":
        return save_configuration(spec_key, ask_for_file, config_file_name, parameters)
    if action == "train_model":
        try:
            return train_model(spec_key, headless, print_only, parameters)
        except gr.Error:
            raise
        except Exception as exc:
            log.exception("Failed to start %s training", spec.display_name)
            raise gr.Error(
                f"{type(exc).__name__}: {exc}",
                title=f"{spec.display_name} training could not start",
                duration=None,
                print_exception=False,
            ) from exc
    raise ValueError(f"Unknown GUI action: {action}")


def download_model_files(spec_key: str, *current_paths):
    """Run Download_Train_Models.py for the spec's bundles, streaming its output into the tab."""
    spec = get_spec(spec_key)
    root = os.path.abspath(os.path.join(scriptdir, ".."))
    script = os.path.join(root, "Download_Train_Models.py")
    keep = [gr.update() for _ in spec.model_paths]
    if not os.path.isfile(script):
        yield [gr.update(value=f"Downloader not found: {script}", visible=True)] + keep + [gr.update()]
        return
    lines: list[str] = []
    for bundle in spec.download_bundles:
        lines.append(f"=== Downloading bundle '{bundle}' into {root} ===")
        yield [gr.update(value="\n".join(lines[-60:]), visible=True)] + keep + [gr.update()]
        process = subprocess.Popen(
            [sys.executable, "-u", script, "--model", bundle],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=0x08000000 if os.name == "nt" else 0,
            **utf8_subprocess_options(),
        )
        last_yield = 0.0
        for raw in process.stdout:
            for piece in raw.replace("\r", "\n").splitlines():
                piece = piece.rstrip()
                if piece:
                    lines.append(piece)
            if time.time() - last_yield > 0.5:
                last_yield = time.time()
                yield [gr.update(value="\n".join(lines[-60:]), visible=True)] + keep + [gr.update()]
        code = process.wait()
        lines.append(f"=== '{bundle}' finished with exit code {code} ===")
        if code != 0:
            break
    resolved = paths_after_download(spec, current_paths)
    updates = [
        gr.update(value=resolved[item.key]) if resolved[item.key] != str(current or "").strip() else gr.update()
        for item, current in zip(spec.model_paths, current_paths)
    ]
    yield [gr.update(value="\n".join(lines[-60:]), visible=True)] + updates + [_model_status_markdown(spec, resolved)]


def paths_after_download(spec: ImageArchSpec, current_paths) -> dict[str, str]:
    """Model paths to show once the downloader finished: existing paths stay, missing ones are relocated."""
    defaults = _default_model_paths(spec)
    resolved = {}
    for item, current in zip(spec.model_paths, current_paths):
        current = str(current or "").strip()
        if current and os.path.exists(current):
            resolved[item.key] = current
            continue
        # a preset applied before downloading names its own file (e.g. the Mage-Flow-Edit DiT): find
        # that file in the local model folders first, the spec default is only the fallback
        relocated = resolve_portable_model_value(item.key, current) if current else current
        if relocated and relocated != current and os.path.exists(relocated):
            resolved[item.key] = relocated
        else:
            resolved[item.key] = defaults.get(item.key) or current
    return resolved


# ----------------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------------


def training_mode_switch_updates(spec: ImageArchSpec, mode, optimizer_type, learning_rate):
    """Recipe values after the user switches Training Mode (``.input`` only: a loaded preset keeps its values).

    Updates for optimizer type / args, learning rate, max grad norm, full BF16, fused backward, memory-efficient
    saving and H2D-only block swap. Full fine-tuning gets the spec's fused-Adafactor recipe; switching back to LoRA
    restores a LoRA learning rate. Values the user already changed in the right direction are kept.
    """
    try:
        full = is_full_fine_tuning(mode) and spec.supports_full_finetune
    except ValueError:
        full = False
    recipe = spec.full_defaults if full else spec.lora_defaults
    try:
        current_lr = float(learning_rate)
    except (TypeError, ValueError):
        current_lr = None
    keep = gr.update()
    if full:
        same_optimizer = str(optimizer_type or "").casefold() == str(recipe["optimizer_type"]).casefold()
        return (
            gr.update(value=recipe["optimizer_type"]),
            keep if same_optimizer else gr.update(value=" ".join(recipe["optimizer_args"])),
            gr.update(value=recipe["learning_rate"]) if current_lr is None or current_lr > 3e-5 else keep,
            gr.update(value=recipe.get("max_grad_norm", 0.0)),
            gr.update(value=True),
            gr.update(value=True),
            gr.update(value=True),
            gr.update(value=False),  # H2D-only streaming of a frozen base does not apply to full fine-tuning
        )
    return (
        keep,
        keep,
        gr.update(value=recipe["learning_rate"]) if current_lr is None or current_lr < 3e-5 else keep,
        keep,
        keep,
        gr.update(value=False),  # the fused Adafactor backward is a full fine-tuning option
        keep,
        keep,
    )


def _model_status_markdown(spec: ImageArchSpec, values: dict) -> str:
    rows = []
    for item in spec.model_paths:
        path = str(values.get(item.key) or "").strip()
        ok = bool(path) and os.path.exists(path)
        rows.append(f"{'✅' if ok else '⚠️'} **{item.label}** — {'found' if ok else 'not set / missing'}")
    hint = ""
    if not all("✅" in row for row in rows):
        hint = (
            f"\n\nDownload the files with **Windows_Download_Training_Model_Files.bat** (or "
            f"`python Download_Train_Models.py`) → *{spec.display_name}* bundle → `{spec.model_folder}`."
        )
    return "  \n".join(rows) + hint


def image_model_tab(spec_key: str, headless: bool = False, config: GUIConfig | dict = {}):
    spec = get_spec(spec_key)
    keys = param_keys(spec)
    config = _prepare_config(spec, config)
    initial_mode = LORA_TRAINING_MODE
    if spec.supports_full_finetune:
        try:
            initial_mode = normalize_training_mode(config.get("training_mode", LORA_TRAINING_MODE))
        except ValueError:
            initial_mode = LORA_TRAINING_MODE
    initial_full = is_full_fine_tuning(initial_mode)
    components: dict[str, gr.components.Component] = {}

    def reg(key: str, component):
        if key in components:
            raise AssertionError(f"{spec.display_name}: component registered twice: {key}")
        components[key] = component
        return component

    def value(key: str, fallback=None):
        result = config.get(key, fallback)
        return fallback if result is None else result

    def file_row(key: str, label: str, info: str, extension: str = ".safetensors", color: str = "blue", folder: bool = False):
        with gr.Row():
            box = reg(key, gr.Textbox(label=label, value=value(key, ""), info=info, scale=8))
            button = gr.Button("📂" if folder else "📁", size="lg", scale=0, min_width=60, elem_classes=["mbtn", f"mbtn-{color}"], visible=not headless)
            if folder:
                button.click(fn=lambda: get_folder_path(folder_path=""), outputs=[box], show_progress=False)
            else:
                button.click(
                    fn=lambda: get_file_path(file_path="", default_extension=extension, extension_name="Model files" if extension == ".safetensors" else "Files"),
                    outputs=[box],
                    show_progress=False,
                )
        return box

    def extra_component(extra: ExtraOption):
        current = value(extra.key, extra.default)
        if extra.kind == "checkbox":
            return reg(extra.key, gr.Checkbox(label=extra.label, value=bool(current), info=extra.info or None))
        if extra.kind == "dropdown":
            return reg(
                extra.key,
                gr.Dropdown(label=extra.label, choices=list(extra.choices), value=current if current in extra.choices else extra.default, info=extra.info or None, allow_custom_value=False),
            )
        if extra.kind == "number":
            kwargs = {"step": extra.step} if extra.step is not None else {}
            if extra.minimum is not None:
                kwargs["minimum"] = extra.minimum
            if extra.maximum is not None:
                kwargs["maximum"] = extra.maximum
            return reg(extra.key, gr.Number(label=extra.label, value=current, info=extra.info or None, **kwargs))
        return reg(extra.key, gr.Textbox(label=extra.label, value=str(current or ""), info=extra.info or None))

    def extras_for(panel: str):
        items = [extra for extra in spec.extras if extra.panel == panel]
        if items and panel_keywords:
            # extras render inside the panel created last; the settings search finds them by label and key
            panel_keywords[-1] += " " + " ".join(f"{extra.label} {extra.key.replace('_', ' ')}".lower() for extra in items)
        for start in range(0, len(items), 4):
            with gr.Row():
                for extra in items[start : start + 4]:
                    extra_component(extra)

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    # ------------------------------------------------------------------ header
    quick_start = (
        "**Quick start:** ① *Model Files* panel → ⬇️ Download Model Files (once) · ② pick your GPU preset below → ⚡ Apply Preset · "
        f"③ *{spec.display_name} Training Dataset* panel → your dataset folder · ④ Start training."
    )
    gr.Markdown(f"### {spec.display_name}\n{spec.notes}\n\n{quick_start}", elem_classes=["model-notes"])
    with gr.Accordion("Configuration & GPU Presets", open=True, elem_classes=["config-presets"]):
        configuration = ConfigurationFile(headless=headless, config=config)
        presets = PresetLibrary(spec.preset_folders, model_name=spec.display_name, headless=headless)

    with gr.Row(equal_height=True):
        training_mode = reg(
            "training_mode",
            gr.Radio(
                label="Training Mode",
                choices=TRAINING_MODE_CHOICES if spec.supports_full_finetune else [LORA_TRAINING_MODE],
                value=initial_mode,
                interactive=spec.supports_full_finetune,
                info=(
                    "LoRA trains a small adapter; Full Fine-Tuning updates the whole DiT and needs much more RAM/VRAM."
                    if spec.supports_full_finetune
                    else f"{spec.display_name} supports LoRA training only."
                ),
                scale=3,
                # LoRA-only models keep the (registered) radio hidden: one choice is not a choice
                visible=spec.supports_full_finetune,
            ),
        )
        presets.follow_training_mode(training_mode)
        model_status = gr.Markdown(_model_status_markdown(spec, config.config if hasattr(config, "config") else {}), elem_classes=["model-status"])

    with gr.Row():
        search_input = gr.Textbox(label="Search Settings", placeholder=f"Search {spec.display_name} settings (e.g. rank, block swap, fp8)", lines=1, scale=6)
        toggle_all_button = gr.Button("Open All Panels", variant="secondary", elem_classes=["mbtn", "mbtn-indigo"], scale=1)
        panels_state = gr.State(value="closed")
    search_results = gr.HTML(visible=False)

    accordions: list[gr.Accordion] = []
    panel_names: list[str] = []
    panel_keywords: list[str] = []

    def panel(name: str, keywords: str, classes: str = "preset_background", visible: bool = True):
        accordion = gr.Accordion(name, open=False, elem_classes=classes, visible=visible)
        accordions.append(accordion)
        panel_names.append(name)
        panel_keywords.append(keywords)
        return accordion

    # ------------------------------------------------------------------ dataset
    with panel(f"{spec.display_name} Training Dataset", "dataset captions resolution buckets cache folder toml control reference edit", "samples_background"):
        dataset_config_mode = reg(
            "dataset_config_mode",
            gr.Radio(label="Dataset Configuration Method", choices=["Use TOML File", "Generate from Folder Structure"], value=value("dataset_config_mode", "Generate from Folder Structure")),
        )
        with gr.Row(visible=value("dataset_config_mode") == "Use TOML File") as toml_mode_row:
            dataset_config = reg("dataset_config", gr.Textbox(label="Dataset Config TOML", value=value("dataset_config", ""), info="Used by both cache stages and training.", scale=8))
            dataset_config_button = gr.Button("📁", size="lg", scale=0, min_width=60, elem_classes=["mbtn", "mbtn-gold"], visible=not headless)
            dataset_config_button.click(fn=lambda: get_file_path(file_path="", default_extension=".toml", extension_name="TOML files"), outputs=[dataset_config], show_progress=False)
        with gr.Column(visible=value("dataset_config_mode") != "Use TOML File") as folder_mode_column:
            with gr.Row():
                parent_folder_path = reg(
                    "parent_folder_path",
                    gr.Textbox(label="Parent Dataset Folder", value=value("parent_folder_path", ""), info="Contains repeat-prefixed sub-folders such as 10_ohwx.", scale=8),
                )
                parent_button = gr.Button("📂", size="lg", scale=0, min_width=60, elem_classes=["mbtn", "mbtn-pink"], visible=not headless)
                parent_button.click(fn=lambda: get_folder_path(folder_path=""), outputs=[parent_folder_path], show_progress=False)
            with gr.Row():
                reg("dataset_resolution_width", gr.Number(label="Width", value=value("dataset_resolution_width", 1024), minimum=spec.resolution_step, step=spec.resolution_step, info=f"Multiple of {spec.resolution_step}."))
                reg("dataset_resolution_height", gr.Number(label="Height", value=value("dataset_resolution_height", 1024), minimum=spec.resolution_step, step=spec.resolution_step, info=f"Multiple of {spec.resolution_step}."))
                reg(
                    "dataset_batch_size",
                    gr.Number(
                        label="Batch Size",
                        value=1 if spec.dataset_kind == "video" else value("dataset_batch_size", 1),
                        minimum=1,
                        step=1,
                        interactive=spec.dataset_kind != "video",
                        info="Fixed to 1 for this model." if spec.dataset_kind == "video" else None,
                    ),
                )
            with gr.Row():
                reg("dataset_caption_extension", gr.Textbox(label="Caption Extension", value=value("dataset_caption_extension", ".txt")))
                reg("create_missing_captions", gr.Checkbox(label="Create Missing Captions", value=bool(value("create_missing_captions", True))))
                reg("caption_strategy", gr.Dropdown(label="Missing Caption Strategy", choices=["folder_name", "empty"], value=value("caption_strategy", "folder_name")))
            with gr.Row():
                reg("dataset_enable_bucket", gr.Checkbox(label="Enable Buckets", value=bool(value("dataset_enable_bucket", True))))
                reg("dataset_bucket_no_upscale", gr.Checkbox(label="Do Not Upscale Buckets", value=bool(value("dataset_bucket_no_upscale", False))))
                reg("dataset_cache_directory", gr.Textbox(label="Cache Directory Name", value=value("dataset_cache_directory", "cache_dir")))
            if spec.dataset_kind == "video":
                with gr.Row():
                    num_frames = reg(
                        "num_frames",
                        gr.Number(label="Target Frames", value=value("num_frames", spec.default_frames), minimum=1, step=4, info="Rounded down to 4n+1 (121 = 5 s at 24 fps). Image folders train as single frames."),
                    )
                    reg("frame_extraction", gr.Dropdown(label="Frame Extraction", choices=["head", "chunk", "slide", "uniform", "full"], value=value("frame_extraction", "head")))
                    reg("frame_stride", gr.Number(label="Frame Stride (slide)", value=value("frame_stride", 1), minimum=1, step=1))
                    reg("frame_sample", gr.Number(label="Frame Samples (uniform)", value=value("frame_sample", 1), minimum=1, step=1))
                    reg("max_frames", gr.Number(label="Max Frames (full)", value=value("max_frames", 241), minimum=1, step=4))
                num_frames.blur(lambda frames: round_frames_to_4n_plus_1(frames), inputs=[num_frames], outputs=[num_frames], show_progress=False)
            if spec.supports_control_images:
                with gr.Row():
                    reg("use_control_images", gr.Checkbox(label=spec.control_label, value=bool(value("use_control_images", False))))
                    reg(
                        "control_directory_name",
                        gr.Textbox(label="Control Images Sub-Folder Name", value=value("control_directory_name", "control_images"), info="Inside each dataset folder; files named like the target (image_0.png, image_1.png ...)."),
                    )
            generated_toml_path = reg("generated_toml_path", gr.Textbox(label="Generated Dataset TOML", value=value("generated_toml_path", ""), interactive=False))
            dataset_status = gr.Textbox(label="Dataset Status", lines=4, interactive=False)
            generate_dataset_button = gr.Button("Generate Dataset TOML", variant="primary", elem_classes=["mbtn", "mbtn-orange"])

        def toggle_dataset_mode(mode):
            return gr.Row(visible=mode == "Use TOML File"), gr.Column(visible=mode == "Generate from Folder Structure")

        dataset_config_mode.change(toggle_dataset_mode, inputs=[dataset_config_mode], outputs=[toml_mode_row, folder_mode_column], show_progress=False)

    # ------------------------------------------------------------------ model
    with panel(f"{spec.display_name} Model Files", "model dit vae text encoder checkpoint path", "preset_background"):
        colors = ["blue", "rose", "lime", "cyan"]
        for index, item in enumerate(spec.model_paths):
            file_row(item.key, item.label, item.info, item.extension, colors[index % len(colors)], folder=item.is_dir)
        extras_for("model")
        with gr.Row():
            refresh_status = gr.Button("🔍 Re-check Model Files", variant="secondary", elem_classes=["mbtn", "mbtn-teal"])
            download_button = gr.Button(
                "⬇️ Download Model Files",
                variant="secondary",
                elem_classes=["mbtn", "mbtn-sky"],
                visible=bool(spec.download_bundles),
            )
        download_log = gr.Textbox(label="Model Download Log", lines=8, max_lines=12, interactive=False, visible=False, autoscroll=True)

    # ------------------------------------------------------------------ memory
    full_max = spec.max_blocks_to_swap_full
    lora_max = spec.max_blocks_to_swap
    with panel("Memory & Speed (Block Swap, FP8, Precision)", "memory vram block swap fp8 scaled pinned h2d dtype speed", "preset_background"):
        with gr.Row():
            reg("dit_dtype", gr.Dropdown(label="DiT Dtype", choices=["bfloat16", "float32"], value="bfloat16", interactive=False, info="LoRA keeps the DiT in BF16; full fine-tuning follows Full BF16."))
            reg(
                "vae_dtype",
                gr.Dropdown(label="VAE Dtype", choices=["bfloat16", "float16", "float32"], value=value("vae_dtype", "bfloat16"), visible=spec.model_path("vae") is not None),
            )
            reg("fp8_base", gr.Checkbox(label="FP8 Base (LoRA)", value=bool(value("fp8_base", False)), visible=spec.supports_fp8_scaled, info="Store the frozen DiT in FP8. Enable together with Scaled FP8."))
            reg("fp8_scaled", gr.Checkbox(label="Scaled FP8 (LoRA)", value=bool(value("fp8_scaled", False)), visible=spec.supports_fp8_scaled, info="Recommended FP8 mode: per-block scaled weights, better quality."))
        with gr.Row():
            blocks_to_swap = reg(
                "blocks_to_swap",
                gr.Slider(
                    label="Blocks to Swap",
                    value=min(int(value("blocks_to_swap", 0) or 0), max(lora_max, full_max)),
                    minimum=0,
                    maximum=max(lora_max, full_max),
                    step=1,
                    info=f"Offload transformer blocks to RAM to save VRAM. Max {lora_max} (LoRA) / {full_max} (fine-tuning).",
                ),
            )
            reg("use_pinned_memory_for_block_swap", gr.Checkbox(label="Pinned Memory for Block Swap", value=bool(value("use_pinned_memory_for_block_swap", False)), info="Faster transfers; on Windows it also uses shared GPU memory."))
        with gr.Row():
            reg(
                "block_swap_h2d_only",
                gr.Checkbox(label="H2D-Only Block Swap (LoRA)", value=bool(value("block_swap_h2d_only", False)), visible=spec.supports_h2d_block_swap, info="Upload-only ring for frozen weights. Needs block swap + gradient checkpointing."),
            )
            reg("block_swap_ring_size", gr.Number(label="H2D Ring Size", value=value("block_swap_ring_size", 2), minimum=1, step=1, visible=spec.supports_h2d_block_swap, info="2 overlaps copy and compute; 1 minimizes VRAM."))
            reg("disable_numpy_memmap", gr.Checkbox(label="Disable NumPy Memmap", value=bool(value("disable_numpy_memmap", False)), info="Read weights normally instead of memory mapping."))

    # ------------------------------------------------------------------ compile
    with panel("Torch Compile", "compile torch inductor speed", "flux1_rank_layers_background"):
        with gr.Row():
            reg("compile", gr.Checkbox(label="Enable torch.compile", value=bool(value("compile", False)), info="Faster steps after a one-time warm-up. Needs a C++ toolchain on Windows."))
            reg("compile_backend", gr.Dropdown(label="Backend", choices=["inductor", "eager", "aot_eager", "cudagraphs"], value=value("compile_backend", "inductor")))
            reg("compile_mode", gr.Dropdown(label="Mode", choices=["default", "reduce-overhead", "max-autotune"], value=value("compile_mode", "default")))
        with gr.Row():
            reg("compile_dynamic", gr.Dropdown(label="Dynamic Shapes", choices=["auto", "true", "false"], value=value("compile_dynamic", "auto")))
            reg("compile_fullgraph", gr.Checkbox(label="Fullgraph", value=bool(value("compile_fullgraph", False))))
            reg("compile_cache_size_limit", gr.Number(label="Cache Size Limit", value=value("compile_cache_size_limit", 0), minimum=0, step=1))

    # ------------------------------------------------------------------ schedule
    with panel("Flow Matching & Timesteps", "timestep flow shift sigmoid weighting schedule noise", "preset_background"):
        with gr.Row():
            reg(
                "timestep_sampling",
                gr.Dropdown(label="Timestep Sampling", choices=list(spec.timestep_choices), value=value("timestep_sampling", spec.defaults.get("timestep_sampling", "shift")), allow_custom_value=True),
            )
            reg("weighting_scheme", gr.Dropdown(label="Weighting Scheme", choices=list(spec.weighting_choices), value=value("weighting_scheme", "none"), allow_custom_value=True))
            reg("discrete_flow_shift", gr.Number(label="Discrete Flow Shift", value=value("discrete_flow_shift", 3.0), step=0.1, info="Used by 'shift' and 'uniform_shift'."))
        with gr.Row():
            reg("sigmoid_scale", gr.Number(label="Sigmoid Scale", value=value("sigmoid_scale", 1.0), step=0.1))
            reg("mode_scale", gr.Number(label="Mode Scale", value=value("mode_scale", 1.29), step=0.01, info="Used by weighting scheme 'mode'."))
            reg("min_timestep", gr.Number(label="Min Timestep", value=value("min_timestep", 0), minimum=0, maximum=999, step=1))
            reg("max_timestep", gr.Number(label="Max Timestep", value=value("max_timestep", 1000), minimum=1, maximum=1000, step=1))
        extras_for("schedule")

    # ------------------------------------------------------------------ training
    with panel("Training Settings", "training attention steps epochs gradient checkpointing logging seed ddp", "preset_background"):
        training = TrainingSettings(headless=headless, config=config, supported_attention=set(spec.supported_attention))
        for key in TRAINING_KEYS:
            if hasattr(training, key):
                reg(key, getattr(training, key))

    # ------------------------------------------------------------------ optimizer
    with panel("Learning Rate, Optimizer & Scheduler", "optimizer learning rate scheduler adafactor adamw automagic prodigy warmup", "flux1_rank_layers_background"):
        optimizer = OptimizerAndScheduler(headless=headless, config=config)
        for key in OPTIMIZER_KEYS:
            reg(key, getattr(optimizer, key))

    # ------------------------------------------------------------------ lora / full
    network_accordion = panel("LoRA Settings", "lora network rank alpha dim dropout base weights", "flux1_background", visible=not initial_full)
    with network_accordion:
        network = Network(headless=headless, config=config)
        network.network_module.value = spec.network_module
        network.network_module.interactive = False
        for key in NETWORK_KEYS:
            reg(key, getattr(network, key))

    full_accordion = panel("Full Fine-Tuning Settings", "full fine tuning fused backward optimizer patch", "preset_background", visible=initial_full and spec.supports_full_finetune)
    with full_accordion:
        with gr.Row():
            reg("fused_backward_pass", gr.Checkbox(label="Fused Backward Pass", value=bool(value("fused_backward_pass", False)), info="Adafactor only; lowest memory. Needs accumulation steps = 1."))
            reg(
                "block_swap_optimizer_patch_params",
                gr.Checkbox(label="Patch Optimizer for Block Swap", value=bool(value("block_swap_optimizer_patch_params", False)), info="For AdamW/Adafactor without fused backward when blocks are swapped."),
            )
        extras_for("full")
        gr.Markdown("Full BF16 (Training Settings) halves weight and optimizer memory and is recommended. Memory-efficient saving is in the Save panel.")

    # ------------------------------------------------------------------ samples
    with panel("Sample Images During Training", "sample prompts cfg seed steps preview", "samples_background"):
        with gr.Row():
            reg("sample_every_n_steps", gr.Number(label="Sample Every N Steps", value=value("sample_every_n_steps", 0), minimum=0, step=1))
            reg("sample_every_n_epochs", gr.Number(label="Sample Every N Epochs", value=value("sample_every_n_epochs", 0), minimum=0, step=1))
            reg("sample_at_first", gr.Checkbox(label="Sample Before Training", value=bool(value("sample_at_first", False))))
        file_row("sample_prompts", "Sample Prompts File", "One prompt per line; supports --w --h --s --d --l --n --ci flags.", ".txt", "forest")
        with gr.Row():
            reg("sample_output_dir", gr.Textbox(label="Sample Output Directory", value=value("sample_output_dir", ""), info="Empty = the output folder."))
            reg("disable_prompt_enhancement", gr.Checkbox(label="Disable Prompt Defaults", value=bool(value("disable_prompt_enhancement", False)), info="Do not append the defaults below to each prompt line."))
        with gr.Row():
            reg("sample_width", gr.Number(label="Width", value=value("sample_width", 1024), minimum=spec.resolution_step, step=spec.resolution_step))
            reg("sample_height", gr.Number(label="Height", value=value("sample_height", 1024), minimum=spec.resolution_step, step=spec.resolution_step))
            reg("sample_steps", gr.Number(label="Steps", value=value("sample_steps", 28), minimum=0, step=1, info="0 = model default schedule." if "s" not in spec.sample_flags else None))
            if spec.dataset_kind == "video":
                reg("sample_num_frames", gr.Number(label="Frames", value=value("sample_num_frames", spec.default_frames), minimum=1, step=4, info="4n+1 pixel frames."))
            reg("sample_seed", gr.Number(label="Seed", value=value("sample_seed", 42), step=1))
        with gr.Row():
            reg("sample_negative_prompt", gr.Textbox(label="Negative Prompt", value=value("sample_negative_prompt", ""), info="Used when CFG > 1."))
            reg("sample_cfg_scale", gr.Number(label="CFG Scale", value=value("sample_cfg_scale", 4.0), step=0.1))
        extras_for("sample")

    # ------------------------------------------------------------------ caching
    with panel("Caching (Latents & Text Encoder)", "cache latent text encoder device batch fp8 skip existing", "samples_background"):
        with gr.Row():
            reg("cache_latents", gr.Checkbox(label="Run Latent / Pixel Caching", value=bool(value("cache_latents", True))))
            reg("caching_latent_device", gr.Dropdown(label="Latent Cache Device", choices=["cuda", "cpu"], value=value("caching_latent_device", "cuda"), allow_custom_value=True))
            reg("caching_latent_batch_size", gr.Number(label="Latent Batch Size", value=value("caching_latent_batch_size", 1), minimum=1, step=1))
            reg("caching_latent_num_workers", gr.Number(label="Latent Workers", value=value("caching_latent_num_workers", 2), minimum=1, step=1))
        with gr.Row():
            reg("caching_latent_skip_existing", gr.Checkbox(label="Skip Existing Latents", value=bool(value("caching_latent_skip_existing", True))))
            reg("caching_latent_keep_cache", gr.Checkbox(label="Keep Unreferenced Latents", value=bool(value("caching_latent_keep_cache", True))))
        with gr.Row():
            reg("cache_text_encoder_outputs", gr.Checkbox(label="Run Text Encoder Caching", value=bool(value("cache_text_encoder_outputs", True))))
            reg("caching_teo_device", gr.Dropdown(label="Text Cache Device", choices=["cuda", "cpu"], value=value("caching_teo_device", "cuda"), allow_custom_value=True))
            reg("caching_teo_batch_size", gr.Number(label="Text Batch Size", value=value("caching_teo_batch_size", 1), minimum=1, step=1))
            reg("caching_teo_num_workers", gr.Number(label="Text Workers", value=value("caching_teo_num_workers", 2), minimum=1, step=1))
        with gr.Row():
            reg("caching_teo_skip_existing", gr.Checkbox(label="Skip Existing Text Caches", value=bool(value("caching_teo_skip_existing", True))))
            reg("caching_teo_keep_cache", gr.Checkbox(label="Keep Unreferenced Text Caches", value=bool(value("caching_teo_keep_cache", True))))
        extras_for("caching")

    # ------------------------------------------------------------------ save
    with panel("Save Models & Resume Training", "save output checkpoint resume precision state", "samples_background"):
        save_load = SaveLoadSettings(headless=headless, config=config, show_mem_eff_save=True)
        for key in SAVE_KEYS:
            reg(key, getattr(save_load, key))

    # ------------------------------------------------------------------ accelerate / advanced / metadata / hf
    with panel("Accelerate Launch (GPUs & Precision)", "accelerate gpu ids precision multi gpu dynamo", "flux1_background"):
        accelerate = AccelerateLaunch(config=config)
        for key in ACCELERATE_KEYS:
            reg(key, getattr(accelerate, key))
    with panel("Advanced Settings", "advanced parameters debug extra", "samples_background"):
        advanced = AdvancedTraining(headless=headless, training_type="lora", config=config)
        reg("additional_parameters", advanced.additional_parameters)
        reg("debug_mode", advanced.debug_mode)
        extras_for("advanced")
    with panel("Metadata", "metadata author license tags title", "flux1_rank_layers_background"):
        metadata = MetaData(config=config)
        for key in METADATA_KEYS:
            reg(key, getattr(metadata, key))
    with panel("HuggingFace Upload", "huggingface upload repository token", "huggingface_background"):
        huggingface = HuggingFace(config=config)
        for key in HUGGINGFACE_KEYS:
            reg(key, getattr(huggingface, key))

    # ------------------------------------------------------------------ run controls
    executors[spec_key] = CommandExecutor(headless=headless)
    executor = executors[spec_key]
    run_state = gr.Textbox(value=train_state_values[spec_key], visible=False)
    with gr.Row():
        print_button = gr.Button("🧾 Print Training Workflow", elem_classes=["mbtn", "mbtn-slate"])
        toggle_all_button_bottom = gr.Button("Open All Panels", variant="secondary", elem_classes=["mbtn", "mbtn-purple"])

    missing = [key for key in keys if key not in components]
    extra_registered = [key for key in components if key not in keys]
    if missing or extra_registered:
        raise AssertionError(f"{spec.display_name}: missing components {missing}; unexpected {extra_registered}")
    settings_list = [components[key] for key in keys]

    # ------------------------------------------------------------------ events
    generate_dataset_button.click(
        partial(generate_dataset_toml, spec_key),
        inputs=[
            components["parent_folder_path"],
            components["dataset_resolution_width"],
            components["dataset_resolution_height"],
            components["dataset_caption_extension"],
            components["create_missing_captions"],
            components["caption_strategy"],
            components["dataset_batch_size"],
            components["dataset_enable_bucket"],
            components["dataset_bucket_no_upscale"],
            components["dataset_cache_directory"],
            components["output_dir"],
            components.get("use_control_images", dummy_false),
            components.get("control_directory_name", gr.Textbox(value="control_images", visible=False)),
            components.get("num_frames", gr.Number(value=spec.default_frames, visible=False)),
            components.get("frame_extraction", gr.Textbox(value="head", visible=False)),
            components.get("frame_stride", gr.Number(value=1, visible=False)),
            components.get("frame_sample", gr.Number(value=1, visible=False)),
            components.get("max_frames", gr.Number(value=241, visible=False)),
        ],
        outputs=[dataset_config, generated_toml_path, dataset_status],
        show_progress=False,
    )

    path_components = [components[item.key] for item in spec.model_paths]

    def recheck(*paths):
        return _model_status_markdown(spec, {item.key: path for item, path in zip(spec.model_paths, paths)})

    refresh_status.click(recheck, inputs=path_components, outputs=[model_status], show_progress=False)
    download_button.click(
        partial(download_model_files, spec_key),
        inputs=path_components,
        outputs=[download_log] + path_components + [model_status],
        show_progress="minimal",
    )
    for component in path_components:
        component.blur(recheck, inputs=path_components, outputs=[model_status], show_progress=False)

    if spec.key == "mage_flow":
        model_dir = os.path.normpath(os.path.join(scriptdir, "..", spec.model_folder))
        t2i_default = os.path.join(model_dir, "mage_flow_base_bf16.safetensors")
        edit_default = os.path.join(model_dir, "mage_flow_edit_base_bf16.safetensors")

        def switch_mage_variant(is_edit, current_dit):
            current = os.path.normcase(os.path.abspath(current_dit)) if current_dit else ""
            known = {os.path.normcase(t2i_default), os.path.normcase(edit_default), ""}
            update_dit = gr.update()
            if current in known:
                target = edit_default if is_edit else t2i_default
                update_dit = gr.update(value=target if os.path.exists(target) else current_dit)
            return update_dit, gr.update(value=bool(is_edit))

        components["is_edit"].change(
            switch_mage_variant,
            inputs=[components["is_edit"], components["dit"]],
            outputs=[components["dit"], components["use_control_images"]],
            show_progress=False,
        )

    if spec.key == "hidream_o1":
        model_dir = os.path.normpath(os.path.join(scriptdir, "..", spec.model_folder))

        def switch_hidream_type(model_type, current_dit):
            dev = model_type == "dev"
            noise = (7.5, 7.5, 2.5) if dev else (8.0, 8.0, 0.0)
            full_path = os.path.join(model_dir, "hidream_o1_image_bf16.safetensors")
            dev_path = os.path.join(model_dir, "hidream_o1_image_dev_bf16.safetensors")
            current = os.path.normcase(os.path.abspath(current_dit)) if current_dit else ""
            update_dit = gr.update()
            if current in {os.path.normcase(full_path), os.path.normcase(dev_path), ""}:
                target = dev_path if dev else full_path
                update_dit = gr.update(value=target if os.path.exists(target) else current_dit)
            return update_dit, noise[0], noise[1], noise[2], 0.0 if dev else 5.0

        components["model_type"].change(
            switch_hidream_type,
            inputs=[components["model_type"], components["dit"]],
            outputs=[components["dit"], components["noise_scale_start"], components["noise_scale_end"], components["noise_clip_std"], components["sample_cfg_scale"]],
            show_progress=False,
        )
        components["task"].change(lambda task: gr.update(value=task == "i2i"), inputs=[components["task"]], outputs=[components["use_control_images"]], show_progress=False)

    def search_settings(query):
        query = str(query or "").strip().lower()
        if not query:
            return gr.update(value="", visible=False)
        matches = [name for name, words in zip(panel_names, panel_keywords) if query in words or query in name.lower()]
        if not matches:
            return gr.update(value=f"<strong>No settings found for '{query}'.</strong>", visible=True)
        return gr.update(value="<strong>Matching panels</strong><ul>" + "".join(f"<li>{name}</li>" for name in matches) + "</ul>", visible=True)

    search_input.change(search_settings, inputs=[search_input], outputs=[search_results], show_progress=False)

    def toggle_panels(state):
        open_panels = state != "open"
        label = "Hide All Panels" if open_panels else "Open All Panels"
        return ["open" if open_panels else "closed", gr.Button(value=label), gr.Button(value=label)] + [gr.Accordion(open=open_panels) for _ in accordions]

    for button in (toggle_all_button, toggle_all_button_bottom):
        button.click(toggle_panels, inputs=[panels_state], outputs=[panels_state, toggle_all_button, toggle_all_button_bottom] + accordions, show_progress=False)

    def apply_training_mode(mode, dit_dtype_value, learning_rate_value):
        full = is_full_fine_tuning(mode)
        return (
            gr.Accordion(visible=not full),
            gr.Accordion(visible=full and spec.supports_full_finetune),
            gr.update(value="bfloat16" if not full else dit_dtype_value),
        )

    training_mode.change(
        apply_training_mode,
        inputs=[training_mode, components["dit_dtype"], components["learning_rate"]],
        outputs=[network_accordion, full_accordion, components["dit_dtype"]],
        show_progress=False,
    )
    mode_default_keys = ["optimizer_type", "optimizer_args", "learning_rate", "max_grad_norm", "full_bf16", "fused_backward_pass", "mem_eff_save", "block_swap_h2d_only"]
    training_mode.input(
        partial(training_mode_switch_updates, spec),
        inputs=[training_mode, components["optimizer_type"], components["learning_rate"]],
        outputs=[components[key] for key in mode_default_keys],
        show_progress=False,
    )

    def sync_training_mode_visibility(mode):
        # Guardrail 2: visibility-only sync after Load/Open (programmatic updates do not fire .change()).
        full = is_full_fine_tuning(mode)
        return gr.Accordion(visible=not full), gr.Accordion(visible=full and spec.supports_full_finetune)

    action = partial(gui_actions, spec_key)
    open_inputs = [gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list
    load_inputs = [gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list
    load_outputs = [configuration.config_file_name, configuration.config_status] + settings_list

    configuration.button_open_config.click(action, inputs=open_inputs, outputs=load_outputs, show_progress=False).then(
        sync_training_mode_visibility, inputs=[training_mode], outputs=[network_accordion, full_accordion], show_progress=False
    ).then(recheck, inputs=path_components, outputs=[model_status], show_progress=False)
    configuration.button_load_config.click(action, inputs=load_inputs, outputs=load_outputs, show_progress=False).then(
        sync_training_mode_visibility, inputs=[training_mode], outputs=[network_accordion, full_accordion], show_progress=False
    ).then(recheck, inputs=path_components, outputs=[model_status], show_progress=False)
    presets.bind(
        configuration.config_file_name,
        action,
        load_inputs,
        load_outputs,
        then_fn=sync_training_mode_visibility,
        then_inputs=[training_mode],
        then_outputs=[network_accordion, full_accordion],
    ).then(recheck, inputs=path_components, outputs=[model_status], show_progress=False)
    configuration.button_save_config.click(
        action,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
    )
    print_button.click(
        action,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )
    executor.button_run.click(
        action,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[executor.button_run, executor.stop_row, executor.button_stop_training, executor.training_status, run_state],
        show_progress=False,
    )
    executor.button_stop_training.click(
        executor.kill_command,
        inputs=[],
        outputs=[executor.button_run, executor.stop_row, executor.button_stop_training, executor.training_status],
        queue=False,
        show_progress=False,
    )
    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.stop_row, executor.button_stop_training, executor.training_status],
        show_progress=False,
    )
    return settings_list


def qwen_image_21_tab(headless: bool = False, config: GUIConfig | dict = {}):
    return image_model_tab("qwen_image_21", headless=headless, config=config)


def ernie_image_tab(headless: bool = False, config: GUIConfig | dict = {}):
    return image_model_tab("ernie_image", headless=headless, config=config)


def mage_flow_tab(headless: bool = False, config: GUIConfig | dict = {}):
    return image_model_tab("mage_flow", headless=headless, config=config)


def hidream_o1_tab(headless: bool = False, config: GUIConfig | dict = {}):
    return image_model_tab("hidream_o1", headless=headless, config=config)


def kandinsky6_tab(headless: bool = False, config: GUIConfig | dict = {}):
    return image_model_tab("kandinsky6", headless=headless, config=config)
