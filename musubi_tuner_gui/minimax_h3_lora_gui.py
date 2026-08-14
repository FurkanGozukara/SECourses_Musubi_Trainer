"""MiniMax H3 text/first-last-frame/reference-to-video-with-audio (T2VA / FL2VA / Ref2VA) LoRA training tab.

Standalone video+audio tab modeled on the LTX-2 tab architecture, with a single
canonical parameter list (MINIMAX_H3_PARAM_KEYS) shared by every save/load/
train action so parameters can never drift out of order.

Backend scripts (musubi-tuner):
    minimax_h3_cache_latents.py                - video VAE + audio VAE latent caching
    minimax_h3_cache_text_encoder_outputs.py   - Qwen3-VL 32B text embedding caching
    minimax_h3_train_network.py                - LoRA training (networks.lora_minimax_h3)

Model files (Comfy-Org/MiniMax-H3):
    DiT          minimax_h3_fl2va_*.safetensors (T2VA + FL2VA) or minimax_h3_ref2va_*  (Ref2VA)
                 full/pruned BF16 and full/pruned ConvRot INT8 are auto-detected
    Text encoder qwen3vl_32b_minimax_h3_{bf16,int8_convrot,nvfp4_awq}.safetensors (auto-detected)
    Video VAE    minimax_h3_video_vae_fp16.safetensors
    Audio VAE    minimax_h3_audio_vae_fp32.safetensors

Geometry contract: 24 fps, width/height multiples of 32, frame counts 17*n+5
(5, 22, ..., released range 124-345 = 5-15 s). Dataset batch_size is fixed to 1.
FP8 bases are rejected by the backend - ConvRot INT8 is the quantization path.
MiniMax-H3 is LoRA-only (no full fine-tuning script exists upstream).
"""

import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime

import gradio as gr
import toml

from .class_accelerate_launch import AccelerateLaunch
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig
from .common_gui import (
    SaveConfigFile,
    SaveConfigFileToRun,
    get_file_path,
    get_folder_path,
    get_file_path_or_save_as,
    print_command_and_toml,
    resolve_portable_model_value,
    run_cmd_advanced_training,
    save_executed_script,
    scriptdir,
    setup_environment,
)
from .custom_logging import setup_logging
from .dataset_config_generator import (
    generate_minimax_h3_dataset_config_from_folders,
    round_frames_to_minimax_h3,
    save_dataset_config,
    validate_dataset_config,
)
from .optimizer_catalog import add_automagic_optimizer_choices, optimizer_guidance, validate_automagic_configuration

log = setup_logging()

MINIMAX_H3_MODEL_FOLDER = "Training_Models_MiniMax_H3"
MINIMAX_H3_DIT_FILENAMES = [
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",  # preferred: lowest VRAM (~21 GB weights)
    "minimax_h3_fl2va_int8_convrot.safetensors",
    "minimax_h3_fl2va_pruned_bf16.safetensors",
    "minimax_h3_fl2va_bf16.safetensors",
]
MINIMAX_H3_TE_FILENAMES = [
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",  # preferred: lowest footprint (~15.7 GB)
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_bf16.safetensors",
]
MINIMAX_H3_VIDEO_VAE_FILENAME = "minimax_h3_video_vae_fp16.safetensors"
MINIMAX_H3_AUDIO_VAE_FILENAME = "minimax_h3_audio_vae_fp32.safetensors"
MINIMAX_H3_MAX_BLOCKS_TO_SWAP = 48  # 50 main DiT blocks, up to 48 swappable
MINIMAX_H3_RELEASED_MIN_FRAMES = 124  # 5 s at 24 fps
MINIMAX_H3_RELEASED_MAX_FRAMES = 345  # 15 s at 24 fps

# ---------------------------------------------------------------------------
# Canonical parameter list. Order MUST match the settings_list built in
# minimax_h3_lora_tab(); an assert there enforces it.
# ---------------------------------------------------------------------------
MINIMAX_H3_PARAM_KEYS = [
    # Accelerate launch
    "mixed_precision",
    "num_processes",
    "num_machines",
    "num_cpu_threads_per_process",
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_use_fullgraph",
    "dynamo_use_dynamic",
    "multi_gpu",
    "gpu_ids",
    "main_process_port",
    "extra_accelerate_launch_args",
    # Model
    "task",
    "dit",
    "video_vae",
    "audio_vae",
    "text_encoder",
    "disable_mmap",
    # Attention
    "sdpa",
    "flash_attn",
    "sage_attn",
    "xformers",
    # Quantization
    "convrot_int8",
    "convrot_int8_bwd",
    "prune_adaln",
    # Memory
    "blocks_to_swap",
    "use_pinned_memory_for_block_swap",
    "block_swap_h2d_only",
    "block_swap_ring_size",
    "gradient_checkpointing",
    "gradient_checkpointing_cpu_offload",
    # Dataset (GUI-only, consumed by the dataset TOML generator)
    "dataset_config_mode",
    "dataset_config",
    "parent_folder_path",
    "generated_toml_path",
    "dataset_resolution_width",
    "dataset_resolution_height",
    "dataset_enable_bucket",
    "dataset_bucket_no_upscale",
    "dataset_cache_directory",
    "dataset_caption_extension",
    "caption_strategy",
    "create_missing_captions",
    "dataset_num_frames",
    "dataset_frame_extraction",
    "dataset_frame_stride",
    "dataset_frame_sample",
    "dataset_max_frames",
    "allow_experimental_duration",
    # Audio supervision
    "video_only",
    "audio_loss_weight",
    # Caching (GUI-only, consumed by the cache command builders)
    "cache_latents",
    "caching_latent_device",
    "caching_latent_num_workers",
    "caching_latent_skip_existing",
    "caching_latent_keep_cache",
    "caching_latent_cache_seed",
    "cache_text_encoder_outputs",
    "caching_teo_device",
    "caching_teo_batch_size",
    "caching_teo_num_workers",
    "caching_teo_skip_existing",
    "caching_teo_keep_cache",
    "caching_teo_text_cache_dtype",
    # Text encoder runtime (shared by TE caching and training-time sampling)
    "text_encoder_blocks_to_swap",
    "text_encoder_attn_mode",
    "nvfp4_scaled_mm",
    # H3 losses / schedule
    "h3_guidance_loss_scale",
    "h3_guidance_loss_scale_audio",
    "h3_guidance_loss_sigma_min",
    "h3_guidance_loss_uncond_cache",
    "h3_teacher_matching",
    "h3_teacher_conditions",
    "h3_teacher_condition_sigma_max",
    "h3_teacher_loss_dc_weight",
    "h3_teacher_loss_mag_weight",
    "h3_teacher_preservation_weight",
    "h3_timestep_focus_prob",
    "h3_timestep_focus_min",
    "h3_timestep_focus_max",
    "h3_shift_video",
    "h3_shift_audio",
    "h3_visual_cond_clean",
    "h3_audio_cond_clean",
    "min_timestep",
    "max_timestep",
    # Network
    "network_module",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "network_weights",
    "dim_from_weights",
    "scale_weight_norms",
    # Optimizer / scheduler
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
    "gradient_accumulation_steps",
    # Training
    "max_train_steps",
    "max_train_epochs",
    "seed",
    "max_data_loader_n_workers",
    "persistent_data_loader_workers",
    # Saving
    "output_dir",
    "output_name",
    "save_every_n_epochs",
    "save_every_n_steps",
    "save_last_n_epochs",
    "save_last_n_steps",
    "save_last_n_epochs_state",
    "save_last_n_steps_state",
    "save_state",
    "save_state_on_train_end",
    "resume",
    # Sampling
    "sample_every_n_epochs",
    "sample_every_n_steps",
    "sample_at_first",
    "sample_prompts",
    "width",
    "height",
    "sample_num_frames",
    "sample_steps",
    "sample_seed",
    "h3_allow_experimental_sample_duration",
    "disable_prompt_enhancement",
    # Logging
    "logging_dir",
    "log_with",
    "log_prefix",
    "log_tracker_name",
    "log_tracker_config",
    "log_config",
    "wandb_api_key",
    "wandb_run_name",
    # Metadata
    "no_metadata",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "training_comment",
    # HuggingFace
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "save_state_to_huggingface",
    "resume_from_huggingface",
    "async_upload",
    # DDP
    "ddp_timeout",
    "ddp_gradient_as_bucket_view",
    "ddp_static_graph",
    # Misc
    "additional_parameters",
    "debug_mode",
]

# GUI-only keys that must never reach the training run TOML.
MINIMAX_H3_RUN_EXCLUSIONS = {
    "num_processes",
    "num_machines",
    "num_cpu_threads_per_process",
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_use_fullgraph",
    "dynamo_use_dynamic",
    "multi_gpu",
    "gpu_ids",
    "main_process_port",
    "extra_accelerate_launch_args",
    "dataset_config_mode",
    "parent_folder_path",
    "generated_toml_path",
    "dataset_resolution_width",
    "dataset_resolution_height",
    "dataset_enable_bucket",
    "dataset_bucket_no_upscale",
    "dataset_cache_directory",
    "dataset_caption_extension",
    "caption_strategy",
    "create_missing_captions",
    "dataset_num_frames",
    "dataset_frame_extraction",
    "dataset_frame_stride",
    "dataset_frame_sample",
    "dataset_max_frames",
    "allow_experimental_duration",  # latent-cache-only flag; the trainer does not accept it
    "cache_latents",
    "cache_text_encoder_outputs",
    "width",
    "height",
    "sample_num_frames",
    "sample_steps",
    "sample_seed",
    "disable_prompt_enhancement",
    "additional_parameters",
    "debug_mode",
}
# All caching_* keys are consumed by the cache command builders, never by the trainer.
MINIMAX_H3_RUN_EXCLUSIONS.update(key for key in MINIMAX_H3_PARAM_KEYS if key.startswith("caching_"))

# Optional string args: drop from the run TOML when empty so the backend
# argparse defaults (None) apply.
MINIMAX_H3_OPTIONAL_STRING_KEYS = {
    "video_vae",
    "audio_vae",
    "text_encoder",
    "text_encoder_attn_mode",
    "h3_guidance_loss_uncond_cache",
    "lr_scheduler_type",
    "logging_dir",
    "log_prefix",
    "log_tracker_name",
    "log_tracker_config",
    "wandb_api_key",
    "wandb_run_name",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "training_comment",
    "resume",
    "network_weights",
    "sample_prompts",
}

# Keys whose values must be integers in the run TOML (gr.Slider yields floats).
MINIMAX_H3_INT_KEYS = {
    "blocks_to_swap",
    "block_swap_ring_size",
    "network_dim",
    "network_alpha",
    "lr_warmup_steps",
    "lr_decay_steps",
    "lr_scheduler_num_cycles",
    "gradient_accumulation_steps",
    "max_train_steps",
    "max_train_epochs",
    "seed",
    "max_data_loader_n_workers",
    "min_timestep",
    "max_timestep",
    "save_every_n_epochs",
    "save_every_n_steps",
    "save_last_n_epochs",
    "save_last_n_steps",
    "save_last_n_epochs_state",
    "save_last_n_steps_state",
    "sample_every_n_epochs",
    "sample_every_n_steps",
    "width",
    "height",
    "sample_num_frames",
    "sample_steps",
    "sample_seed",
    "text_encoder_blocks_to_swap",
    "ddp_timeout",
    "main_process_port",
}

# Defaults are tuned for the lowest-VRAM best-quality configuration:
# pruned ConvRot INT8 FL2VA DiT (~21 GB weights, auto-detected, no flag needed),
# NVFP4+AWQ Qwen3-VL text encoder streamed from CPU (blocks_to_swap 50),
# H2D-only block swap of 48/50 DiT blocks, gradient checkpointing, AdamW8bit,
# rank 128 LoRA at LR 1e-4, and the guidance-distillation countermeasure
# (scale 4.0, sigma-min gate 0.15) recommended by the upstream docs.
MINIMAX_H3_DEFAULTS = {
    "mixed_precision": "bf16",
    "num_processes": 1,
    "num_machines": 1,
    "num_cpu_threads_per_process": 1,
    "dynamo_backend": "no",
    "dynamo_mode": "",
    "dynamo_use_fullgraph": False,
    "dynamo_use_dynamic": False,
    "multi_gpu": False,
    "gpu_ids": "0",
    "main_process_port": 0,
    "extra_accelerate_launch_args": "",
    "task": "t2va",
    "dit": "",
    "video_vae": "",
    "audio_vae": "",
    "text_encoder": "",
    "disable_mmap": False,
    "sdpa": True,
    "flash_attn": False,
    "sage_attn": False,
    "xformers": False,
    "convrot_int8": False,
    "convrot_int8_bwd": "bf16",
    "prune_adaln": False,
    "blocks_to_swap": 48,
    "use_pinned_memory_for_block_swap": False,
    "block_swap_h2d_only": True,
    "block_swap_ring_size": 2,
    "gradient_checkpointing": True,
    "gradient_checkpointing_cpu_offload": False,
    "dataset_config_mode": "Generate from Folder Structure",
    "dataset_config": "",
    "parent_folder_path": "",
    "generated_toml_path": "",
    "dataset_resolution_width": 512,
    "dataset_resolution_height": 512,
    "dataset_enable_bucket": True,
    "dataset_bucket_no_upscale": False,
    "dataset_cache_directory": "cache_dir",
    "dataset_caption_extension": ".txt",
    "caption_strategy": "folder_name",
    "create_missing_captions": True,
    "dataset_num_frames": 124,
    "dataset_frame_extraction": "head",
    "dataset_frame_stride": 1,
    "dataset_frame_sample": 1,
    "dataset_max_frames": 345,
    "allow_experimental_duration": False,
    "video_only": False,
    "audio_loss_weight": 1.0,
    "cache_latents": True,
    "caching_latent_device": "cuda",
    "caching_latent_num_workers": 2,
    "caching_latent_skip_existing": True,
    "caching_latent_keep_cache": True,
    "caching_latent_cache_seed": 42,
    "cache_text_encoder_outputs": True,
    "caching_teo_device": "cuda",
    "caching_teo_batch_size": 1,
    "caching_teo_num_workers": 1,
    "caching_teo_skip_existing": True,
    "caching_teo_keep_cache": True,
    "caching_teo_text_cache_dtype": "bf16",
    "text_encoder_blocks_to_swap": 50,
    "text_encoder_attn_mode": "",
    "nvfp4_scaled_mm": False,
    "h3_guidance_loss_scale": 4.0,
    "h3_guidance_loss_scale_audio": -1.0,
    "h3_guidance_loss_sigma_min": 0.15,
    "h3_guidance_loss_uncond_cache": "",
    "h3_teacher_matching": False,
    "h3_teacher_conditions": "first,last",
    "h3_teacher_condition_sigma_max": 0.75,
    "h3_teacher_loss_dc_weight": 1.0,
    "h3_teacher_loss_mag_weight": 1.0,
    "h3_teacher_preservation_weight": 1.0,
    "h3_timestep_focus_prob": 0.0,
    "h3_timestep_focus_min": 0.4,
    "h3_timestep_focus_max": 0.8,
    "h3_shift_video": 12.0,
    "h3_shift_audio": 3.0,
    "h3_visual_cond_clean": 0.999,
    "h3_audio_cond_clean": 1.0,
    "min_timestep": 0,
    "max_timestep": 1000,
    "network_module": "networks.lora_minimax_h3",
    "network_dim": 128,
    "network_alpha": 128,
    "network_dropout": 0.0,
    "network_args": "",
    "network_weights": "",
    "dim_from_weights": False,
    "scale_weight_norms": 0.0,
    "optimizer_type": "AdamW8bit",
    "optimizer_args": "",
    "learning_rate": 1e-4,
    "max_grad_norm": 1.0,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,
    "lr_decay_steps": 0,
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1.0,
    "lr_scheduler_timescale": 0,
    "lr_scheduler_min_lr_ratio": 0.0,
    "lr_scheduler_type": "",
    "lr_scheduler_args": "",
    "gradient_accumulation_steps": 1,
    "max_train_steps": 80000,
    "max_train_epochs": 100,
    "seed": 42,
    "max_data_loader_n_workers": 2,
    "persistent_data_loader_workers": True,
    "output_dir": "",
    "output_name": "my-minimax-h3-lora",
    "save_every_n_epochs": 1,
    "save_every_n_steps": 0,
    "save_last_n_epochs": 0,
    "save_last_n_steps": 0,
    "save_last_n_epochs_state": 0,
    "save_last_n_steps_state": 0,
    "save_state": False,
    "save_state_on_train_end": False,
    "resume": "",
    "sample_every_n_epochs": 0,
    "sample_every_n_steps": 0,
    "sample_at_first": False,
    "sample_prompts": "",
    "width": 512,
    "height": 512,
    "sample_num_frames": 124,
    "sample_steps": 30,
    "sample_seed": 42,
    "h3_allow_experimental_sample_duration": False,
    "disable_prompt_enhancement": False,
    "logging_dir": "",
    "log_with": "",
    "log_prefix": "",
    "log_tracker_name": "",
    "log_tracker_config": "",
    "log_config": False,
    "wandb_api_key": "",
    "wandb_run_name": "",
    "no_metadata": False,
    "metadata_author": "",
    "metadata_description": "",
    "metadata_license": "",
    "metadata_tags": "",
    "metadata_title": "",
    "training_comment": "",
    "huggingface_repo_id": "",
    "huggingface_token": "",
    "huggingface_repo_type": "model",
    "huggingface_repo_visibility": "private",
    "huggingface_path_in_repo": "",
    "save_state_to_huggingface": False,
    "resume_from_huggingface": False,
    "async_upload": False,
    "ddp_timeout": 0,
    "ddp_gradient_as_bucket_view": False,
    "ddp_static_graph": False,
    "additional_parameters": "",
    "debug_mode": "None",
}


@dataclass
class MinimaxH3Workflow:
    parameters: list
    config_path: str
    latent_cache_command: list | None
    text_cache_command: list | None
    train_command: list

    @property
    def commands(self) -> list:
        commands = []
        if self.latent_cache_command:
            commands.append(self.latent_cache_command)
        if self.text_cache_command:
            commands.append(self.text_cache_command)
        commands.append(self.train_command)
        return commands


executor: CommandExecutor | None = None
train_state_value = time.time()


def _trainer_script_path(filename: str) -> str:
    return os.path.normpath(os.path.join(scriptdir, "musubi-tuner", "src", "musubi_tuner", filename))


def _find_accelerate_launch(python_cmd: str) -> list:
    python_dir = os.path.dirname(python_cmd)
    fallback = os.path.join(python_dir, "accelerate.exe" if sys.platform == "win32" else "accelerate")
    if os.path.isfile(fallback):
        return [fallback, "launch"]
    accelerate_path = shutil.which("accelerate")
    if accelerate_path:
        return [accelerate_path, "launch"]
    return [python_cmd, "-m", "accelerate.commands.launch"]


def _resolve_device(value, gpu_ids) -> str:
    device = str(value or "cuda").strip()
    if device == "cuda":
        first_gpu = str(gpu_ids or "").split(",")[0].strip()
        return f"cuda:{first_gpu}" if first_gpu else "cuda:0"
    return device


def _replace_parameter(parameters, key, value):
    return [(name, value if name == key else current) for name, current in parameters]


def _to_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _default_model_paths() -> dict:
    """Pre-fill model paths when the standard download folder exists."""
    model_dir = os.path.normpath(os.path.join(scriptdir, "..", MINIMAX_H3_MODEL_FOLDER))
    defaults = {"dit": "", "text_encoder": "", "video_vae": "", "audio_vae": ""}
    for name in MINIMAX_H3_DIT_FILENAMES:
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            defaults["dit"] = candidate
            break
    for name in MINIMAX_H3_TE_FILENAMES:
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            defaults["text_encoder"] = candidate
            break
    video_vae = os.path.join(model_dir, MINIMAX_H3_VIDEO_VAE_FILENAME)
    if os.path.isfile(video_vae):
        defaults["video_vae"] = video_vae
    audio_vae = os.path.join(model_dir, MINIMAX_H3_AUDIO_VAE_FILENAME)
    if os.path.isfile(audio_vae):
        defaults["audio_vae"] = audio_vae
    return defaults


def _uncond_cache_path(param_dict: dict) -> str:
    """Default location of the guidance-loss uncond probe embedding cache."""
    explicit = str(param_dict.get("h3_guidance_loss_uncond_cache") or "").strip()
    if explicit:
        return explicit
    output_dir = str(param_dict.get("output_dir") or "").strip()
    output_name = str(param_dict.get("output_name") or "minimax_h3_lora")
    return os.path.join(output_dir, f"{output_name}_h3_uncond_space.safetensors") if output_dir else ""


def _guidance_loss_enabled(param_dict: dict) -> bool:
    return _to_float(param_dict.get("h3_guidance_loss_scale"), 0.0) > 0.0


def _validate_minimax_h3_parameters(param_dict: dict, *, validate_paths: bool = True) -> None:
    """GUI-side validation mirroring the backend's rules so errors show up before launch."""
    task = str(param_dict.get("task") or "t2va")
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"Unknown MiniMax H3 task: {task}")

    if validate_paths:
        dit = str(param_dict.get("dit") or "").strip()
        if not dit:
            raise ValueError(
                "MiniMax H3 DiT path is required (e.g. minimax_h3_fl2va_pruned_int8_convrot.safetensors from Comfy-Org/MiniMax-H3)."
            )
        if not os.path.isfile(dit):
            raise ValueError(f"MiniMax H3 DiT does not exist: {dit}")
        if task == "ref2va" and "fl2va" in os.path.basename(dit).lower():
            log.warning("Task ref2va usually needs a minimax_h3_ref2va_* transformer; an FL2VA file is selected.")
        if task in ("t2va", "fl2va") and "ref2va" in os.path.basename(dit).lower():
            log.warning("Tasks t2va/fl2va use the FL2VA transformer; a Ref2VA file is selected.")

        if bool(param_dict.get("cache_latents", True)):
            for key, label in (("video_vae", "Video VAE"), ("audio_vae", "Audio VAE")):
                value = str(param_dict.get(key) or "").strip()
                if not value:
                    raise ValueError(f"{label} path is required for latent caching (always required by MiniMax H3).")
                if not os.path.isfile(value):
                    raise ValueError(f"{label} does not exist: {value}")

        sampling_active = (
            _to_int(param_dict.get("sample_every_n_epochs")) > 0
            or _to_int(param_dict.get("sample_every_n_steps")) > 0
            or bool(param_dict.get("sample_at_first"))
        )
        needs_te = bool(param_dict.get("cache_text_encoder_outputs", True)) or sampling_active
        text_encoder = str(param_dict.get("text_encoder") or "").strip()
        if needs_te:
            if not text_encoder:
                raise ValueError(
                    "A Qwen3-VL 32B text encoder is required for text encoder caching / sampling "
                    "(qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors is the smallest)."
                )
            if not os.path.isfile(text_encoder):
                raise ValueError(f"Text encoder does not exist: {text_encoder}")
        if sampling_active:
            for key, label in (("video_vae", "Video VAE"), ("audio_vae", "Audio VAE")):
                value = str(param_dict.get(key) or "").strip()
                if not value or not os.path.isfile(value):
                    raise ValueError(f"{label} is required for training-time sampling.")

    fp8_flags = [flag for flag in ("fp8_base", "fp8_scaled") if param_dict.get(flag)]
    if fp8_flags:
        raise ValueError("MiniMax H3 does not support FP8 bases; use a ConvRot INT8 checkpoint (or convrot_int8) instead.")

    blocks_to_swap = _to_int(param_dict.get("blocks_to_swap"))
    if blocks_to_swap < 0 or blocks_to_swap > MINIMAX_H3_MAX_BLOCKS_TO_SWAP:
        raise ValueError(f"blocks_to_swap must be between 0 and {MINIMAX_H3_MAX_BLOCKS_TO_SWAP} for MiniMax H3.")
    if param_dict.get("block_swap_h2d_only"):
        if blocks_to_swap <= 0:
            raise ValueError("block_swap_h2d_only requires blocks_to_swap > 0.")
        if not param_dict.get("gradient_checkpointing"):
            raise ValueError("block_swap_h2d_only requires gradient_checkpointing.")
        if param_dict.get("gradient_checkpointing_cpu_offload"):
            raise ValueError("gradient_checkpointing_cpu_offload is incompatible with block_swap_h2d_only.")

    te_swap = _to_int(param_dict.get("text_encoder_blocks_to_swap"))
    if te_swap < 0 or te_swap > 50:
        raise ValueError("text_encoder_blocks_to_swap must be between 0 and 50 (50 Qwen3-VL decoder layers).")

    attention = [bool(param_dict.get(k)) for k in ("sdpa", "flash_attn", "sage_attn", "xformers")]
    if sum(attention) == 0:
        raise ValueError("Select one attention implementation (SDPA is recommended).")
    if sum(attention) > 1:
        raise ValueError("Select only one attention implementation (SDPA / FlashAttention / SageAttention / xformers).")

    teacher_matching = bool(param_dict.get("h3_teacher_matching"))
    guidance = _guidance_loss_enabled(param_dict)
    if teacher_matching and guidance:
        raise ValueError(
            "Teacher matching and the guidance loss are mutually exclusive: teacher targets already live in the "
            "distilled guided space. Set Guidance Loss Scale to 0 or disable Teacher Matching."
        )
    if teacher_matching and task != "t2va":
        raise ValueError("Teacher matching requires task t2va (an FL2VA teacher teaching a T2VA student).")

    if guidance and validate_paths:
        if not bool(param_dict.get("cache_text_encoder_outputs", True)):
            uncond = _uncond_cache_path(param_dict)
            if not uncond or not os.path.isfile(uncond):
                raise ValueError(
                    "The guidance loss needs the uncond probe embedding cache. Enable 'Cache Text Encoder Outputs' "
                    "(the GUI writes it automatically) or point 'Guidance Uncond Cache' at an existing file."
                )

    focus_min = _to_float(param_dict.get("h3_timestep_focus_min"), 0.4)
    focus_max = _to_float(param_dict.get("h3_timestep_focus_max"), 0.8)
    focus_prob = _to_float(param_dict.get("h3_timestep_focus_prob"), 0.0)
    if focus_prob > 0 and not (0.0 <= focus_min < focus_max <= 1.0):
        raise ValueError("Timestep focus band needs 0 <= min < max <= 1.")
    if focus_prob > 0 and (_to_int(param_dict.get("min_timestep")) > 0 or _to_int(param_dict.get("max_timestep"), 1000) < 1000):
        raise ValueError("h3_timestep_focus_prob does not compose with min_timestep/max_timestep; reset them to 0/1000.")

    if task == "ref2va" and str(param_dict.get("dataset_config_mode") or "") == "Generate from Folder Structure":
        raise ValueError(
            "Ref2VA training requires a video_jsonl_file dataset (ordered references). Switch Dataset Configuration Mode "
            "to 'Use TOML File' and supply a dataset TOML pointing at your Ref2VA JSONL."
        )

    if _to_int(param_dict.get("network_dim"), 128) <= 0:
        raise ValueError("Network dimension (rank) must be >= 1.")
    if _to_int(param_dict.get("max_train_steps")) <= 0 and _to_int(param_dict.get("max_train_epochs")) <= 0:
        raise ValueError("Set max_train_epochs or max_train_steps to a value greater than 0.")

    frames = _to_int(param_dict.get("dataset_num_frames"), 124)
    if not bool(param_dict.get("allow_experimental_duration")) and frames < MINIMAX_H3_RELEASED_MIN_FRAMES:
        raise ValueError(
            f"Target frames {frames} is below the released minimum of {MINIMAX_H3_RELEASED_MIN_FRAMES} (5 s at 24 fps). "
            "Enable 'Allow Experimental Duration' to train on shorter clips."
        )


def _normalize_minimax_h3_parameters(parameters, *, validate_paths: bool = True):
    param_dict = dict(parameters)

    # Force the fixed network module.
    param_dict["network_module"] = "networks.lora_minimax_h3"
    parameters = _replace_parameter(parameters, "network_module", "networks.lora_minimax_h3")

    task = str(param_dict.get("task") or "t2va").strip().lower()
    if task not in ("t2va", "fl2va", "ref2va"):
        task = "t2va"
    param_dict["task"] = task
    parameters = _replace_parameter(parameters, "task", task)

    # Integer coercion for slider/number widgets.
    for key in MINIMAX_H3_INT_KEYS:
        if key in param_dict and param_dict[key] is not None and not isinstance(param_dict[key], bool):
            coerced = _to_int(param_dict[key], 0)
            param_dict[key] = coerced
            parameters = _replace_parameter(parameters, key, coerced)

    _validate_minimax_h3_parameters(param_dict, validate_paths=validate_paths)
    return param_dict, parameters


def build_minimax_h3_cache_commands(param_dict: dict, *, python_cmd: str | None = None):
    python_cmd = python_cmd or sys.executable
    dataset_config = str(param_dict["dataset_config"])
    gpu_ids = param_dict.get("gpu_ids")
    task = str(param_dict.get("task") or "t2va")
    teacher_matching = bool(param_dict.get("h3_teacher_matching"))

    latent_command = None
    if bool(param_dict.get("cache_latents", True)):
        # The endpoint teacher (first,last) trains a T2VA student against FL2VA teacher
        # forwards, so its latent caches must include the first/last condition latents
        # (--task fl2va). The ref teacher uses the cached target latents themselves as the
        # reference, so plain t2va caches suffice (fl2va caches also work).
        teacher_conditions = str(param_dict.get("h3_teacher_conditions") or "first,last").strip() or "first,last"
        latent_task = "fl2va" if (teacher_matching and task == "t2va" and teacher_conditions != "ref") else task
        latent_command = [
            python_cmd,
            _trainer_script_path("minimax_h3_cache_latents.py"),
            "--dataset_config",
            dataset_config,
            "--task",
            latent_task,
            "--video_vae",
            str(param_dict["video_vae"]),
            "--audio_vae",
            str(param_dict["audio_vae"]),
            "--device",
            _resolve_device(param_dict.get("caching_latent_device"), gpu_ids),
            "--cache_seed",
            str(_to_int(param_dict.get("caching_latent_cache_seed"), 42)),
        ]
        num_workers = param_dict.get("caching_latent_num_workers")
        if num_workers is not None:
            latent_command.extend(["--num_workers", str(_to_int(num_workers, 2))])
        if param_dict.get("caching_latent_skip_existing"):
            latent_command.append("--skip_existing")
        if param_dict.get("caching_latent_keep_cache"):
            latent_command.append("--keep_cache")
        if param_dict.get("allow_experimental_duration"):
            latent_command.append("--allow_experimental_duration")
        if param_dict.get("disable_mmap"):
            latent_command.append("--disable_mmap")

    text_command = None
    if bool(param_dict.get("cache_text_encoder_outputs", True)):
        text_command = [
            python_cmd,
            _trainer_script_path("minimax_h3_cache_text_encoder_outputs.py"),
            "--dataset_config",
            dataset_config,
            "--task",
            task,
            "--text_encoder",
            str(param_dict["text_encoder"]),
            "--device",
            _resolve_device(param_dict.get("caching_teo_device"), gpu_ids),
            "--text_cache_dtype",
            str(param_dict.get("caching_teo_text_cache_dtype") or "bf16"),
        ]
        value = param_dict.get("caching_teo_batch_size")
        if value is not None:
            text_command.extend(["--batch_size", str(_to_int(value, 1))])
        num_workers = param_dict.get("caching_teo_num_workers")
        if num_workers is not None:
            text_command.extend(["--num_workers", str(_to_int(num_workers, 1))])
        if param_dict.get("caching_teo_skip_existing"):
            text_command.append("--skip_existing")
        if param_dict.get("caching_teo_keep_cache"):
            text_command.append("--keep_cache")
        te_swap = _to_int(param_dict.get("text_encoder_blocks_to_swap"))
        if te_swap > 0:
            text_command.extend(["--text_encoder_blocks_to_swap", str(te_swap)])
        attn_mode = str(param_dict.get("text_encoder_attn_mode") or "").strip()
        if attn_mode:
            text_command.extend(["--text_encoder_attn_mode", attn_mode])
        if param_dict.get("nvfp4_scaled_mm"):
            text_command.append("--nvfp4_scaled_mm")
        if param_dict.get("disable_mmap"):
            text_command.append("--disable_mmap")
        if teacher_matching and task == "t2va":
            teacher_conditions = str(param_dict.get("h3_teacher_conditions") or "first,last").strip() or "first,last"
            text_command.extend(["--teacher_conditions", teacher_conditions])
        if _guidance_loss_enabled(param_dict):
            uncond_path = _uncond_cache_path(param_dict)
            if uncond_path:
                text_command.extend(["--uncond_output", uncond_path])

    return latent_command, text_command


def _run_config_parameters(param_dict: dict, parameters):
    """Produce the parameter list for the training run TOML."""
    filtered = []
    for name, value in parameters:
        if name in MINIMAX_H3_OPTIONAL_STRING_KEYS and isinstance(value, str) and value.strip() == "":
            continue
        # -1 means "same as video scale" for the audio guidance scale (backend default None)
        if name == "h3_guidance_loss_scale_audio" and _to_float(value, -1.0) < 0:
            continue
        filtered.append((name, value))

    # Auto-wire the guidance-loss uncond cache written by the TE caching step.
    if _guidance_loss_enabled(param_dict):
        uncond_path = _uncond_cache_path(param_dict)
        if uncond_path:
            filtered = [(name, value) for name, value in filtered if name != "h3_guidance_loss_uncond_cache"]
            filtered.append(("h3_guidance_loss_uncond_cache", uncond_path))
    else:
        # Guidance disabled: never send scale/uncond so the backend stays on plain flow targets.
        filtered = [
            (name, value)
            for name, value in filtered
            if name not in ("h3_guidance_loss_scale", "h3_guidance_loss_scale_audio", "h3_guidance_loss_uncond_cache")
        ]
    return filtered


def _enhance_sample_prompts(param_dict: dict, parameters):
    """Append default --w/--h/--f/--s/--d flags to .txt prompt lines.

    MiniMax H3 sampling rejects negative prompts and CFG, so unlike other tabs no
    --l/--n flags are ever added. .json/.toml prompt files are passed through as-is.
    """
    prompt_path = str(param_dict.get("sample_prompts") or "").strip()
    if not prompt_path or bool(param_dict.get("disable_prompt_enhancement")):
        return param_dict, parameters
    if not os.path.isfile(prompt_path) or not prompt_path.lower().endswith(".txt"):
        return param_dict, parameters

    width = _to_int(param_dict.get("width"), 512)
    height = _to_int(param_dict.get("height"), 512)
    frames = round_frames_to_minimax_h3(_to_int(param_dict.get("sample_num_frames"), 124))
    steps = _to_int(param_dict.get("sample_steps"), 30)
    seed = _to_int(param_dict.get("sample_seed"), 42)

    def has_flag(line: str, flag: str) -> bool:
        return re.search(rf"(?<!\S)--{re.escape(flag)}(?:\s+|=)", line, re.IGNORECASE) is not None

    enhanced_lines = []
    with open(prompt_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                enhanced_lines.append(line)
                continue
            if not has_flag(stripped, "w"):
                stripped += f" --w {width}"
            if not has_flag(stripped, "h"):
                stripped += f" --h {height}"
            if not has_flag(stripped, "f"):
                stripped += f" --f {frames}"
            if not has_flag(stripped, "s"):
                stripped += f" --s {steps}"
            if not has_flag(stripped, "d"):
                stripped += f" --d {seed}"
            enhanced_lines.append(stripped)

    output_dir = str(param_dict.get("output_dir") or "").strip()
    if not output_dir:
        return param_dict, parameters
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = str(param_dict.get("output_name") or "minimax_h3_lora")
    enhanced_path = os.path.abspath(os.path.join(output_dir, f"{output_name}_enhanced_prompts_{timestamp}.txt"))
    with open(enhanced_path, "w", encoding="utf-8") as handle:
        handle.write("# Generated by SECourses Musubi Trainer (MiniMax H3)\n")
        for line in enhanced_lines:
            handle.write(f"{line}\n")

    param_dict["sample_prompts"] = enhanced_path
    parameters = _replace_parameter(parameters, "sample_prompts", enhanced_path)
    return param_dict, parameters


def prepare_minimax_h3_workflow(
    parameters, config_path: str, *, validate_paths: bool = True, python_cmd: str | None = None
) -> MinimaxH3Workflow:
    param_dict, parameters = _normalize_minimax_h3_parameters(parameters, validate_paths=validate_paths)
    python_cmd = python_cmd or sys.executable

    if validate_paths:
        os.makedirs(str(param_dict["output_dir"]), exist_ok=True)

    run_parameters = _run_config_parameters(param_dict, parameters)
    SaveConfigFileToRun(
        parameters=run_parameters,
        file_path=config_path,
        exclusion=sorted(MINIMAX_H3_RUN_EXCLUSIONS | {"file_path", "save_as", "headless", "print_only"}),
        mandatory_keys=["dataset_config", "dit", "output_dir", "output_name", "network_module", "task"],
    )

    latent_command, text_command = build_minimax_h3_cache_commands(param_dict, python_cmd=python_cmd)

    train_command = _find_accelerate_launch(python_cmd)
    train_command = AccelerateLaunch.run_cmd(
        run_cmd=train_command,
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
    train_command.extend([_trainer_script_path("minimax_h3_train_network.py"), "--config_file", config_path])

    additional = str(param_dict.get("additional_parameters") or "").strip()
    debug = {
        "Show Timesteps (Image)": "--show_timesteps image",
        "Show Timesteps (Console)": "--show_timesteps console",
    }.get(str(param_dict.get("debug_mode") or ""), "")
    if debug:
        additional = f"{additional} {debug}".strip()
    train_command = run_cmd_advanced_training(run_cmd=train_command, additional_parameters=additional)

    return MinimaxH3Workflow(
        parameters=parameters,
        config_path=config_path,
        latent_cache_command=latent_command,
        text_cache_command=text_command,
        train_command=train_command,
    )


def _command_line(command: list) -> str:
    if platform.system() == "Windows":
        return subprocess.list2cmdline([str(item) for item in command])
    return shlex.join([str(item) for item in command])


def _build_workflow_script(workflow: MinimaxH3Workflow) -> tuple:
    if platform.system() == "Windows":
        # UTF-8 console vars keep backend log lines (which include Japanese
        # help text) from crashing with cp1252 UnicodeEncodeError when the
        # script is re-run manually outside the GUI.
        lines = ["@echo off", "setlocal", "set PYTHONUTF8=1", "set PYTHONIOENCODING=utf-8:backslashreplace"]
        for index, command in enumerate(workflow.commands):
            label = "training" if index == len(workflow.commands) - 1 else "caching"
            lines.append(f"echo Starting MiniMax H3 {label}...")
            lines.append(_command_line(command))
            lines.append("if errorlevel 1 exit /b %errorlevel%")
        suffix = ".bat"
    else:
        lines = ["#!/bin/bash", "set -e", "export PYTHONUTF8=1", "export PYTHONIOENCODING=utf-8:backslashreplace"]
        for index, command in enumerate(workflow.commands):
            label = "training" if index == len(workflow.commands) - 1 else "caching"
            lines.append(f'echo "Starting MiniMax H3 {label}..."')
            lines.append(_command_line(command))
        suffix = ".sh"

    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as handle:
        handle.write(content)
        path = handle.name
    if platform.system() != "Windows":
        os.chmod(path, os.stat(path).st_mode | 0o100)
    return path, content


def _create_minimax_h3_dataset_config(param_dict: dict) -> tuple:
    parent_folder = str(param_dict.get("parent_folder_path") or "").strip()
    if not parent_folder:
        raise ValueError("Parent folder path is required to generate a dataset configuration.")

    resolution = (
        _to_int(param_dict.get("dataset_resolution_width"), 512),
        _to_int(param_dict.get("dataset_resolution_height"), 512),
    )
    config, messages = generate_minimax_h3_dataset_config_from_folders(
        parent_folder=parent_folder,
        resolution=resolution,
        caption_extension=str(param_dict.get("dataset_caption_extension") or ".txt"),
        create_missing_captions=bool(param_dict.get("create_missing_captions", True)),
        caption_strategy=str(param_dict.get("caption_strategy") or "folder_name"),
        enable_bucket=bool(param_dict.get("dataset_enable_bucket", True)),
        bucket_no_upscale=bool(param_dict.get("dataset_bucket_no_upscale", False)),
        cache_directory_name=str(param_dict.get("dataset_cache_directory") or "cache_dir"),
        num_frames=_to_int(param_dict.get("dataset_num_frames"), 124),
        frame_extraction=str(param_dict.get("dataset_frame_extraction") or "head"),
        frame_stride=_to_int(param_dict.get("dataset_frame_stride"), 1),
        frame_sample=_to_int(param_dict.get("dataset_frame_sample"), 1),
        max_frames=_to_int(param_dict.get("dataset_max_frames"), MINIMAX_H3_RELEASED_MAX_FRAMES),
        allow_experimental_duration=bool(param_dict.get("allow_experimental_duration", False)),
    )

    output_dir = str(param_dict.get("output_dir") or "").strip() or parent_folder
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_path = os.path.abspath(os.path.join(output_dir, f"minimax_h3_dataset_{timestamp}.toml"))
    save_dataset_config(config, toml_path)
    is_valid, validation_messages = validate_dataset_config(toml_path)
    messages.extend(validation_messages)
    if not is_valid:
        raise ValueError("Generated dataset config failed validation:\n" + "\n".join(validation_messages))
    return toml_path, messages


def generate_minimax_h3_dataset_toml(*args):
    parameters = list(zip(MINIMAX_H3_PARAM_KEYS, args))
    param_dict = dict(parameters)
    try:
        toml_path, messages = _create_minimax_h3_dataset_config(param_dict)
        status = "\n".join(messages + [f"[OK] Dataset config saved: {toml_path}"])
        gr.Info("MiniMax H3 dataset configuration generated.")
        return toml_path, toml_path, status
    except Exception as exc:
        log.exception("Failed to generate MiniMax H3 dataset config")
        raise gr.Error(f"{type(exc).__name__}: {exc}", print_exception=False) from exc


def train_minimax_h3_model(headless: bool, print_only: bool, parameters):
    global train_state_value
    param_dict = dict(parameters)
    validate_automagic_configuration(param_dict, warning_callback=None if headless else gr.Warning)

    if str(param_dict.get("dataset_config_mode") or "") == "Generate from Folder Structure":
        generated = str(param_dict.get("generated_toml_path") or "").strip()
        if not generated or not os.path.isfile(generated):
            generated, _ = _create_minimax_h3_dataset_config(param_dict)
        param_dict["dataset_config"] = generated
        parameters = _replace_parameter(parameters, "dataset_config", generated)
        parameters = _replace_parameter(parameters, "generated_toml_path", generated)
    else:
        dataset_config = str(param_dict.get("dataset_config") or "").strip()
        if not dataset_config or not os.path.isfile(dataset_config):
            raise ValueError("Dataset config TOML file is required (or switch to 'Generate from Folder Structure').")

    param_dict, parameters = _enhance_sample_prompts(param_dict, parameters)

    raw_output_dir = str(param_dict.get("output_dir") or "").strip()
    if not raw_output_dir:
        raise ValueError("Output directory is required.")
    output_dir = os.path.abspath(raw_output_dir)
    output_name = str(param_dict.get("output_name") or "minimax_h3_lora")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_path = os.path.join(output_dir, f"{output_name}_{timestamp}.toml")
    workflow = prepare_minimax_h3_workflow(parameters, config_path)

    if print_only:
        for command in workflow.commands[:-1]:
            print_command_and_toml(command, "")
        print_command_and_toml(workflow.train_command, workflow.config_path)
        return None

    if executor is None:
        raise RuntimeError("MiniMax H3 command executor is not initialized.")

    script_path, script_content = _build_workflow_script(workflow)
    save_executed_script(
        script_content=script_content,
        config_name=output_name,
        script_type="minimax_h3",
    )
    env = setup_environment()
    command = [script_path] if platform.system() == "Windows" else ["bash", script_path]
    executor.execute_command(
        run_cmd=command,
        env=env,
        shell=platform.system() == "Windows",
    )

    train_state_value = time.time()
    return (
        gr.Button(visible=False or headless),
        gr.Row(visible=True),
        gr.Button(interactive=True),
        gr.Textbox(value="MiniMax H3 workflow in progress (caching then training)..."),
        gr.Textbox(value=train_state_value),
    )


def save_minimax_h3_configuration(save_as: bool, file_path: str, parameters):
    original_path = file_path
    if save_as or not file_path:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")
    if not file_path:
        return original_path, gr.update(value="No file selected.", visible=True)
    if not file_path.lower().endswith(".toml"):
        file_path = os.path.splitext(file_path)[0] + ".toml"

    try:
        _, parameters = _normalize_minimax_h3_parameters(parameters, validate_paths=False)
        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "headless", "print_only"],
        )
        message = f"Configuration saved: {os.path.basename(file_path)}"
        gr.Info(message)
        return file_path, gr.update(value=message, visible=True)
    except Exception as exc:
        message = f"Failed to save configuration: {exc}"
        log.exception(message)
        gr.Error(message)
        return original_path, gr.update(value=message, visible=True)


def _config_value_for_component(key: str, value, default):
    if key == "network_module":
        return "networks.lora_minimax_h3"
    if key == "task":
        text = str(value or "t2va").strip().lower()
        return text if text in ("t2va", "fl2va", "ref2va") else "t2va"
    if key == "convrot_int8_bwd":
        text = str(value or "bf16").strip()
        return text if text in ("bf16", "int8") else "bf16"
    if key == "h3_teacher_conditions":
        text = str(value or "first,last").strip()
        return text if text in ("first,last", "ref") else "first,last"
    if isinstance(value, list):
        if key in {"optimizer_args", "lr_scheduler_args", "network_args"}:
            return " ".join(str(item) for item in value)
        if not isinstance(default, list):
            return value[0] if value else default
    return value


def open_minimax_h3_configuration(ask_for_file: bool, file_path: str, parameters):
    original_path = file_path
    if ask_for_file:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")
    if not file_path:
        return tuple([original_path, gr.update(value="", visible=False)] + [value for _, value in parameters])
    if ask_for_file and not os.path.isfile(file_path):
        message = f"New configuration will be created: {os.path.basename(file_path)}"
        return tuple([file_path, gr.update(value=message, visible=True)] + [value for _, value in parameters])
    if not os.path.isfile(file_path):
        message = f"Configuration does not exist: {file_path}"
        gr.Error(message)
        return tuple([original_path, gr.update(value=message, visible=True)] + [value for _, value in parameters])

    try:
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            data = toml.load(handle)
        values = []
        for key, default in parameters:
            value = resolve_portable_model_value(key, data.get(key, default))
            values.append(_config_value_for_component(key, value, default))
        message = f"Configuration loaded: {os.path.basename(file_path)}"
        gr.Info(message)
        return tuple([file_path, gr.update(value=message, visible=True)] + values)
    except Exception as exc:
        message = f"Failed to load configuration: {exc}"
        log.exception(message)
        gr.Error(message)
        return tuple([original_path, gr.update(value=message, visible=True)] + [value for _, value in parameters])


def minimax_h3_gui_actions(action: str, ask_for_file: bool, config_file_name: str, headless: bool, print_only: bool, *args):
    parameters = list(zip(MINIMAX_H3_PARAM_KEYS, args))
    if action == "open_configuration":
        return open_minimax_h3_configuration(ask_for_file, config_file_name, parameters)
    if action == "save_configuration":
        return save_minimax_h3_configuration(ask_for_file, config_file_name, parameters)
    if action == "train_model":
        try:
            return train_minimax_h3_model(headless, print_only, parameters)
        except gr.Error:
            raise
        except Exception as exc:
            log.exception("Failed to start MiniMax H3 training")
            raise gr.Error(
                f"{type(exc).__name__}: {exc}",
                title="MiniMax H3 training could not start",
                duration=None,
                print_exception=False,
            ) from exc
    raise ValueError(f"Unknown GUI action: {action}")


def minimax_h3_lora_tab(headless=False, config: GUIConfig = {}):
    global executor
    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    model_defaults = _default_model_paths()

    def get_value(key):
        default = MINIMAX_H3_DEFAULTS.get(key)
        if key in model_defaults:
            return config.get(key, None) or model_defaults[key] or default
        return config.get(key, default)

    registrations = []

    def reg(key, component):
        registrations.append((key, component))
        return component

    gr.Markdown(
        "Train LoRA models for **MiniMax H3** text-to-video-with-audio (T2VA), first/last-frame (FL2VA), and "
        "reference-to-video (Ref2VA). Video is 24 fps; width/height are 32 px steps; valid frame counts are 17n+5 "
        "(released range **124-345** frames = 5-15 s; shorter clips need 'Allow Experimental Duration'). "
        "Dataset batch size is fixed to 1 by the architecture - use gradient accumulation instead. "
        "Pruned ConvRot INT8 checkpoints (~21 GB) and the NVFP4 text encoder are auto-detected: no extra flags needed."
    )

    with gr.Accordion("Configuration File", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    accelerate_launch = AccelerateLaunch(config=config)

    with gr.Accordion("Model Settings", open=True):
        with gr.Row():
            task = reg(
                "task",
                gr.Dropdown(
                    label="Training Task",
                    choices=["t2va", "fl2va", "ref2va"],
                    value=get_value("task"),
                    info="t2va = text-to-video+audio, fl2va = first/last-frame conditioning, ref2va = ordered JSONL references. "
                    "t2va and fl2va use the FL2VA transformer; ref2va needs the Ref2VA transformer and a JSONL dataset.",
                ),
            )
        with gr.Row():
            dit = reg(
                "dit",
                gr.Textbox(
                    label="DiT (Transformer) Path",
                    placeholder="Path to minimax_h3_fl2va_pruned_int8_convrot.safetensors (recommended, ~21 GB)",
                    value=get_value("dit"),
                    info="Full/pruned BF16 or full/pruned ConvRot INT8 from Comfy-Org/MiniMax-H3 - format auto-detected. FP8 files are rejected.",
                ),
            )
            dit_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-blue"], visible=(not headless))
            dit_button.click(get_file_path, inputs=[dit], outputs=[dit], show_progress=False)
        with gr.Row():
            text_encoder = reg(
                "text_encoder",
                gr.Textbox(
                    label="Text Encoder (Qwen3-VL 32B) Path",
                    placeholder="Path to qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors (recommended, ~15.7 GB)",
                    value=get_value("text_encoder"),
                    info="BF16 (~51.5 GB), ConvRot INT8 (~27 GB) or NVFP4+AWQ (~15.7 GB) - auto-detected. "
                    "Used for text embedding caching and training-time sampling.",
                ),
            )
            text_encoder_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-lime"], visible=(not headless))
            text_encoder_button.click(get_file_path, inputs=[text_encoder], outputs=[text_encoder], show_progress=False)
        with gr.Row():
            video_vae = reg(
                "video_vae",
                gr.Textbox(
                    label="Video VAE Path",
                    placeholder="Path to minimax_h3_video_vae_fp16.safetensors (~5.2 GB)",
                    value=get_value("video_vae"),
                    info="Always required for latent caching (encoding is upcast to FP32 internally).",
                ),
            )
            video_vae_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-fuchsia"], visible=(not headless))
            video_vae_button.click(get_file_path, inputs=[video_vae], outputs=[video_vae], show_progress=False)
            audio_vae = reg(
                "audio_vae",
                gr.Textbox(
                    label="Audio VAE Path",
                    placeholder="Path to minimax_h3_audio_vae_fp32.safetensors (~0.6 GB)",
                    value=get_value("audio_vae"),
                    info="Always required - H3 caches audio latents even for silent videos.",
                ),
            )
            audio_vae_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-pink"], visible=(not headless))
            audio_vae_button.click(get_file_path, inputs=[audio_vae], outputs=[audio_vae], show_progress=False)
        with gr.Row():
            disable_mmap = reg(
                "disable_mmap",
                gr.Checkbox(
                    label="Disable memory-mapped loading",
                    value=get_value("disable_mmap"),
                    info="Disable mmap safetensors loading (slower, works around network-drive or antivirus issues).",
                ),
            )

    with gr.Accordion("Attention Settings", open=False):
        with gr.Row():
            sdpa = reg("sdpa", gr.Checkbox(label="SDPA", value=get_value("sdpa"), info="PyTorch scaled dot-product attention (recommended)."))
            flash_attn = reg(
                "flash_attn",
                gr.Checkbox(label="FlashAttention 2", value=get_value("flash_attn"), info="Requires flash-attn built for your CUDA/PyTorch."),
            )
            sage_attn = reg(
                "sage_attn",
                gr.Checkbox(label="SageAttention", value=get_value("sage_attn"), info="Requires sageattention package."),
            )
            xformers = reg(
                "xformers",
                gr.Checkbox(label="xformers", value=get_value("xformers"), info="Requires xformers package."),
            )

    with gr.Accordion("Quantization and Memory Settings", open=True):
        gr.Markdown(
            "**Pre-quantized ConvRot INT8 checkpoints are detected automatically - leave 'ConvRot INT8' unchecked for them.** "
            "Check it only to quantize a full/pruned **BF16** DiT at load time (bit-identical to the published INT8 files). "
            "FP8 is not supported by MiniMax H3. Block swap streams up to 48 of the 50 DiT blocks from CPU RAM; "
            "H2D-only swap roughly triples block-swap speed for frozen-base LoRA training."
        )
        with gr.Row():
            convrot_int8 = reg(
                "convrot_int8",
                gr.Checkbox(
                    label="ConvRot INT8 (quantize BF16 at load)",
                    value=get_value("convrot_int8"),
                    info="Quantize a BF16 DiT to ConvRot INT8 while loading (~66→34 GB or ~40→21 GB pruned). Not needed for pre-quantized files.",
                ),
            )
            convrot_int8_bwd = reg(
                "convrot_int8_bwd",
                gr.Dropdown(
                    label="ConvRot INT8 Backward",
                    choices=["bf16", "int8"],
                    value=get_value("convrot_int8_bwd"),
                    info="bf16 = accurate dequantized backward (default). int8 = faster fused GEMM backward (needs Triton + CUDA).",
                ),
            )
            prune_adaln = reg(
                "prune_adaln",
                gr.Checkbox(
                    label="Prune AdaLN at load",
                    value=get_value("prune_adaln"),
                    info="Prune a full BF16 DiT's ~26 GB AdaLN projections at load (rank-8 SVD, near-identical outputs). "
                    "No-op on already-pruned files; rejected for pre-quantized INT8 files.",
                ),
            )
        with gr.Row():
            blocks_to_swap = reg(
                "blocks_to_swap",
                gr.Slider(
                    label="Blocks to Swap",
                    minimum=0,
                    maximum=MINIMAX_H3_MAX_BLOCKS_TO_SWAP,
                    step=1,
                    value=get_value("blocks_to_swap"),
                    info=f"Stream N of the 50 DiT blocks from CPU RAM (0-{MINIMAX_H3_MAX_BLOCKS_TO_SWAP}). Higher = less VRAM, slower steps, more system RAM.",
                ),
            )
            block_swap_h2d_only = reg(
                "block_swap_h2d_only",
                gr.Checkbox(
                    label="H2D-only block swap",
                    value=get_value("block_swap_h2d_only"),
                    info="Frozen-base streaming (no device-to-host copies): measured ~3x faster block swap. Requires gradient checkpointing.",
                ),
            )
            block_swap_ring_size = reg(
                "block_swap_ring_size",
                gr.Slider(
                    label="H2D ring size",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=get_value("block_swap_ring_size"),
                    info="H2D-only GPU ring buffers. 2 = double buffering (default), 1 = lowest VRAM.",
                ),
            )
            use_pinned_memory_for_block_swap = reg(
                "use_pinned_memory_for_block_swap",
                gr.Checkbox(
                    label="Pinned memory for block swap",
                    value=get_value("use_pinned_memory_for_block_swap"),
                    info="Faster transfers when host RAM permits; leave off on Windows if 'shared GPU memory' fills up.",
                ),
            )
        with gr.Row():
            gradient_checkpointing = reg(
                "gradient_checkpointing",
                gr.Checkbox(label="Gradient Checkpointing", value=get_value("gradient_checkpointing"), info="Recompute activations during backward to save VRAM."),
            )
            gradient_checkpointing_cpu_offload = reg(
                "gradient_checkpointing_cpu_offload",
                gr.Checkbox(
                    label="Checkpointing CPU offload",
                    value=get_value("gradient_checkpointing_cpu_offload"),
                    info="Offload checkpointed activations to CPU (more RAM, less VRAM). Incompatible with H2D-only swap.",
                ),
            )
            text_encoder_blocks_to_swap = reg(
                "text_encoder_blocks_to_swap",
                gr.Slider(
                    label="Text Encoder Blocks to Swap",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=get_value("text_encoder_blocks_to_swap"),
                    info="Stream N of the 50 Qwen3-VL layers from CPU during TE caching / sampling (50 = minimum VRAM; per-layer ~0.3-0.9 GB).",
                ),
            )

    with gr.Accordion("Dataset Settings", open=True):
        with gr.Row():
            dataset_config_mode = reg(
                "dataset_config_mode",
                gr.Dropdown(
                    label="Dataset Configuration Mode",
                    choices=["Generate from Folder Structure", "Use TOML File"],
                    value=get_value("dataset_config_mode"),
                    info="Generate a dataset TOML from a parent folder of video subfolders, or supply your own TOML. Ref2VA requires a TOML with video_jsonl_file.",
                ),
            )
        with gr.Group() as dataset_generation_group:
            with gr.Row():
                parent_folder_path = reg(
                    "parent_folder_path",
                    gr.Textbox(
                        label="Parent Folder Path",
                        placeholder="Folder containing subfolders of videos (e.g. 1_mydataset)",
                        value=get_value("parent_folder_path"),
                        info="Each subfolder becomes a dataset. Prefix with repeats like '3_name'. Captions are .txt files next to the videos. "
                        "Audio comes from the video track, a same-stem .wav/.flac sidecar, or silence.",
                    ),
                )
                parent_folder_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-pink"], visible=(not headless))
                parent_folder_button.click(get_folder_path, inputs=[parent_folder_path], outputs=[parent_folder_path], show_progress=False)
            with gr.Row():
                dataset_resolution_width = reg(
                    "dataset_resolution_width",
                    gr.Number(label="Resolution Width", value=get_value("dataset_resolution_width"), precision=0, minimum=64, step=32,
                              info="Training width (32 px steps). Official example is 768x1344; 512x512 keeps VRAM/time low."),
                )
                dataset_resolution_height = reg(
                    "dataset_resolution_height",
                    gr.Number(label="Resolution Height", value=get_value("dataset_resolution_height"), precision=0, minimum=64, step=32,
                              info="Training height (32 px steps)."),
                )
                dataset_enable_bucket = reg(
                    "dataset_enable_bucket",
                    gr.Checkbox(label="Enable Bucketing", value=get_value("dataset_enable_bucket"),
                                info="Group media into resolution buckets (recommended for mixed sizes)."),
                )
                dataset_bucket_no_upscale = reg(
                    "dataset_bucket_no_upscale",
                    gr.Checkbox(label="Bucket No Upscale", value=get_value("dataset_bucket_no_upscale"),
                                info="Never upscale when bucketing (only downscale to fit)."),
                )
            with gr.Row():
                dataset_cache_directory = reg(
                    "dataset_cache_directory",
                    gr.Textbox(label="Cache Directory Name", value=get_value("dataset_cache_directory"),
                               info="Relative name creates a cache folder inside each dataset subfolder."),
                )
                dataset_caption_extension = reg(
                    "dataset_caption_extension",
                    gr.Textbox(label="Caption Extension", value=get_value("dataset_caption_extension"), info="Caption file extension (default .txt)."),
                )
                caption_strategy = reg(
                    "caption_strategy",
                    gr.Dropdown(
                        label="Caption Strategy",
                        choices=["folder_name", "empty"],
                        value=get_value("caption_strategy"),
                        info="Content for auto-created captions: the folder name (minus repeat prefix) or empty.",
                    ),
                )
                create_missing_captions = reg(
                    "create_missing_captions",
                    gr.Checkbox(label="Create Missing Captions", value=get_value("create_missing_captions"),
                                info="Create caption files for videos without one."),
                )
            gr.Markdown(
                "**🎬 Video Frame Settings** — MiniMax H3 frame counts must be 17n+5 (5, 22, 39, ..., 107, **124**, 141, ..., 345). "
                "Released training range is 124-345 frames (5-15 s at 24 fps); shorter needs 'Allow Experimental Duration'. "
                "All videos are normalized to 24 fps from their timestamps automatically."
            )
            with gr.Row():
                dataset_num_frames = reg(
                    "dataset_num_frames",
                    gr.Number(label="Target Frames", value=get_value("dataset_num_frames"), precision=0, minimum=5,
                              info="Frames per training sample. Rounded down to 17n+5 automatically. 124 = released minimum (5 s)."),
                )
                dataset_frame_extraction = reg(
                    "dataset_frame_extraction",
                    gr.Dropdown(
                        label="Frame Extraction",
                        choices=["head", "chunk", "slide", "uniform", "full"],
                        value=get_value("dataset_frame_extraction"),
                        info="head = first N frames (official default), chunk = consecutive chunks, slide = sliding window, uniform = evenly spaced, full = whole video.",
                    ),
                )
                dataset_frame_stride = reg(
                    "dataset_frame_stride",
                    gr.Number(label="Frame Stride (slide)", value=get_value("dataset_frame_stride"), precision=0, minimum=1,
                              info="Stride for 'slide' extraction."),
                )
                dataset_frame_sample = reg(
                    "dataset_frame_sample",
                    gr.Number(label="Frame Samples (uniform)", value=get_value("dataset_frame_sample"), precision=0, minimum=1,
                              info="Number of samples for 'uniform' extraction."),
                )
            with gr.Row():
                dataset_max_frames = reg(
                    "dataset_max_frames",
                    gr.Number(label="Max Frames", value=get_value("dataset_max_frames"), precision=0, minimum=5,
                              info="Upper bound on frames used from each video (345 = released maximum, 15 s)."),
                )
                allow_experimental_duration = reg(
                    "allow_experimental_duration",
                    gr.Checkbox(
                        label="Allow Experimental Duration",
                        value=get_value("allow_experimental_duration"),
                        info="Bypass the released 5-15 s duration check during latent caching (e.g. 39-frame smoke tests). Frame geometry (17n+5) still applies.",
                    ),
                )
                generate_toml_button = gr.Button("🛠️ Generate Dataset Config", variant="secondary", elem_classes=["mbtn", "mbtn-orange"])
        with gr.Row():
            dataset_config = reg(
                "dataset_config",
                gr.Textbox(
                    label="Dataset Config TOML",
                    placeholder="Path to dataset config .toml (auto-filled when generated)",
                    value=get_value("dataset_config"),
                    info="The dataset TOML used by caching and training. batch_size is forced to 1 by the architecture.",
                ),
            )
            dataset_config_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-gold"], visible=(not headless))
            dataset_config_button.click(get_file_path, inputs=[dataset_config], outputs=[dataset_config], show_progress=False)
            generated_toml_path = reg(
                "generated_toml_path",
                gr.Textbox(label="Generated TOML Path", value=get_value("generated_toml_path"), interactive=False,
                           info="Last generated dataset TOML (reused on Start Training when present)."),
            )
        dataset_status = gr.Textbox(label="Dataset Generation Status", lines=5, interactive=False, value="")

    with gr.Accordion("Audio Supervision", open=False):
        gr.Markdown(
            "H3 generates video **and audio** jointly. Videos with a real audio track (or a same-stem .wav sidecar) are "
            "supervised on audio too; silent videos never are. A pure video-only LoRA can degrade audio quality - "
            "the shared weights carry both streams."
        )
        with gr.Row():
            video_only = reg(
                "video_only",
                gr.Checkbox(
                    label="Video only (disable audio supervision)",
                    value=get_value("video_only"),
                    info="Audio loss weight 0 for all samples. Audio output of the trained LoRA becomes unconstrained.",
                ),
            )
            audio_loss_weight = reg(
                "audio_loss_weight",
                gr.Number(
                    label="Audio Loss Weight",
                    value=get_value("audio_loss_weight"),
                    minimum=0.0,
                    step=0.1,
                    info="Scale of the audio MSE term for samples cached with real audio (default 1.0).",
                ),
            )

    with gr.Accordion("Caching Settings", open=False):
        gr.Markdown(
            "Latent and text-encoder caches are (re)built automatically before each training start; existing cache files are skipped. "
            "TE caching streams the Qwen3-VL layers from CPU when 'Text Encoder Blocks to Swap' is set (50 = fits consumer GPUs)."
        )
        with gr.Row():
            cache_latents = reg(
                "cache_latents",
                gr.Checkbox(label="Cache Latents", value=get_value("cache_latents"), info="Run video+audio VAE latent caching before training."),
            )
            caching_latent_device = reg(
                "caching_latent_device",
                gr.Dropdown(label="Latent Device", choices=["cuda", "cpu"], value=get_value("caching_latent_device"), allow_custom_value=True,
                            info="Device for VAE encoding (first GPU from gpu_ids is used for 'cuda')."),
            )
            caching_latent_num_workers = reg(
                "caching_latent_num_workers",
                gr.Number(label="Latent Workers", value=get_value("caching_latent_num_workers"), precision=0, minimum=0),
            )
            caching_latent_cache_seed = reg(
                "caching_latent_cache_seed",
                gr.Number(label="Cache Seed", value=get_value("caching_latent_cache_seed"), precision=0, minimum=0,
                          info="Seed for reproducible target-video posterior samples."),
            )
            caching_latent_skip_existing = reg(
                "caching_latent_skip_existing",
                gr.Checkbox(label="Skip Existing", value=get_value("caching_latent_skip_existing"),
                            info="Skip up-to-date caches (stale caches are detected from file fingerprints and rebuilt)."),
            )
            caching_latent_keep_cache = reg(
                "caching_latent_keep_cache",
                gr.Checkbox(label="Keep Cache", value=get_value("caching_latent_keep_cache"), info="Keep cache files for removed/changed media."),
            )
        with gr.Row():
            cache_text_encoder_outputs = reg(
                "cache_text_encoder_outputs",
                gr.Checkbox(label="Cache Text Encoder Outputs", value=get_value("cache_text_encoder_outputs"),
                            info="Run Qwen3-VL text-embedding caching before training. Also writes the guidance-loss uncond cache when needed."),
            )
            caching_teo_device = reg(
                "caching_teo_device",
                gr.Dropdown(label="TE Device", choices=["cuda", "cpu"], value=get_value("caching_teo_device"), allow_custom_value=True),
            )
            caching_teo_batch_size = reg(
                "caching_teo_batch_size",
                gr.Number(label="TE Batch Size", value=get_value("caching_teo_batch_size"), precision=0, minimum=1),
            )
            caching_teo_num_workers = reg(
                "caching_teo_num_workers",
                gr.Number(label="TE Workers", value=get_value("caching_teo_num_workers"), precision=0, minimum=0),
            )
            caching_teo_skip_existing = reg(
                "caching_teo_skip_existing",
                gr.Checkbox(label="Skip Existing", value=get_value("caching_teo_skip_existing")),
            )
            caching_teo_keep_cache = reg(
                "caching_teo_keep_cache",
                gr.Checkbox(label="Keep Cache", value=get_value("caching_teo_keep_cache")),
            )
            caching_teo_text_cache_dtype = reg(
                "caching_teo_text_cache_dtype",
                gr.Dropdown(label="Text Cache Dtype", choices=["bf16", "float32"], value=get_value("caching_teo_text_cache_dtype"),
                            info="bf16 halves the text cache size (recommended)."),
            )
        with gr.Row():
            text_encoder_attn_mode = reg(
                "text_encoder_attn_mode",
                gr.Dropdown(
                    label="Text Encoder Attention",
                    choices=["", "sdpa", "flash_attention_2", "eager"],
                    value=get_value("text_encoder_attn_mode"),
                    info="Empty = transformers default (sdpa). Use flash_attention_2 for very long Ref2VA presentations.",
                ),
            )
            nvfp4_scaled_mm = reg(
                "nvfp4_scaled_mm",
                gr.Checkbox(
                    label="NVFP4 W4A4 matmuls",
                    value=get_value("nvfp4_scaled_mm"),
                    info="For the NVFP4 text encoder: faster W4A4 scaled_mm (needs PyTorch 2.10+ and a Blackwell GPU; slightly lower quality).",
                ),
            )

    with gr.Accordion("H3 Loss and Schedule Settings", open=True):
        gr.Markdown(
            "H3 checkpoints are CFG-distilled: plain flow-matching training slowly de-distills them (washed-out outputs). "
            "The **guidance loss** (scale 3-4; enabled by default at 4.0) re-anchors the target in the distilled space; "
            "its uncond probe cache is generated automatically during TE caching. Cost: ~+50% step time (less with the sigma-min gate). "
            "**Teacher matching** (t2va only) is the alternative: it matches a frozen privileged-condition teacher instead and replaces the guidance loss - "
            "either the FL2VA endpoint teacher (first,last) or the reference teacher (ref), which teaches from the clip itself and makes audio a real teaching target."
        )
        with gr.Row():
            h3_guidance_loss_scale = reg(
                "h3_guidance_loss_scale",
                gr.Number(
                    label="Guidance Loss Scale",
                    value=get_value("h3_guidance_loss_scale"),
                    minimum=0.0,
                    step=0.5,
                    info="0 = disabled (plain flow target). Field reports suggest 3-4; 4 is more reliable for longer runs.",
                ),
            )
            h3_guidance_loss_scale_audio = reg(
                "h3_guidance_loss_scale_audio",
                gr.Number(
                    label="Guidance Scale (Audio)",
                    value=get_value("h3_guidance_loss_scale_audio"),
                    step=0.5,
                    info="-1 = same as video scale. Separate scale for the audio target.",
                ),
            )
            h3_guidance_loss_sigma_min = reg(
                "h3_guidance_loss_sigma_min",
                gr.Number(
                    label="Guidance Sigma Min",
                    value=get_value("h3_guidance_loss_sigma_min"),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="Skip the extra uncond forward below this base sigma. Recommended 0.15 (saves ~15% of steps' extra cost).",
                ),
            )
            h3_guidance_loss_uncond_cache = reg(
                "h3_guidance_loss_uncond_cache",
                gr.Textbox(
                    label="Guidance Uncond Cache",
                    value=get_value("h3_guidance_loss_uncond_cache"),
                    placeholder="Auto: <output_dir>/<output_name>_h3_uncond_space.safetensors",
                    info="Uncond probe embedding (~10 KB). Leave empty - the GUI writes it during TE caching automatically.",
                ),
            )
        with gr.Row():
            h3_teacher_matching = reg(
                "h3_teacher_matching",
                gr.Checkbox(
                    label="Teacher Matching (t2va)",
                    value=get_value("h3_teacher_matching"),
                    info="Train the T2VA student against frozen FL2VA teacher predictions (real first/last frames). "
                    "Replaces the guidance loss; latent caches are built with --task fl2va automatically. "
                    "Audio becomes a base-preservation anchor in this mode. Generate with --lora_runtime_attach afterwards.",
                ),
            )
            h3_teacher_conditions = reg(
                "h3_teacher_conditions",
                gr.Dropdown(
                    label="Teacher Conditions",
                    choices=["first,last", "ref"],
                    value=get_value("h3_teacher_conditions"),
                    info="first,last = FL2VA endpoint teacher (latent caches auto-switch to fl2va; audio stays a preservation anchor). "
                    "ref = Ref2VA teacher on the training clip itself: complete information at every sigma, 3-5x lower teaching-band floor, "
                    "and audio becomes a real teaching target - slower/heavier teacher step.",
                ),
            )
            h3_teacher_condition_sigma_max = reg(
                "h3_teacher_condition_sigma_max",
                gr.Number(
                    label="Teacher Condition Sigma Max",
                    value=get_value("h3_teacher_condition_sigma_max"),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="Above this base sigma the teacher drops the endpoints (base-preservation anchor). Default 0.75; lower to 0.4-0.5 for low-diversity data.",
                ),
            )
            h3_teacher_loss_dc_weight = reg(
                "h3_teacher_loss_dc_weight",
                gr.Number(
                    label="Teacher DC Weight",
                    value=get_value("h3_teacher_loss_dc_weight"),
                    minimum=0.0,
                    step=0.1,
                    info="Weight of the video residual's color/tone (DC) component on teaching steps. 0.0-0.3 avoids absorbing the dataset palette; keep 1.0 for style LoRAs.",
                ),
            )
            h3_teacher_loss_mag_weight = reg(
                "h3_teacher_loss_mag_weight",
                gr.Number(
                    label="Teacher Magnitude Weight",
                    value=get_value("h3_teacher_loss_mag_weight"),
                    minimum=0.0,
                    step=0.1,
                    info="Magnitude term of the decomposed teacher loss (direction fixed at 1.0). 1.0 = plain-MSE-equal loss value.",
                ),
            )
            h3_teacher_preservation_weight = reg(
                "h3_teacher_preservation_weight",
                gr.Number(
                    label="Teacher Preservation Weight",
                    value=get_value("h3_teacher_preservation_weight"),
                    minimum=0.0,
                    step=0.1,
                    info="Loss weight of base-preservation anchor steps. Raise if anchor-band drift keeps growing.",
                ),
            )
        with gr.Row():
            h3_timestep_focus_prob = reg(
                "h3_timestep_focus_prob",
                gr.Number(
                    label="Timestep Focus Probability",
                    value=get_value("h3_timestep_focus_prob"),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="Probability of drawing the base sigma inside the focus band (0 = uniform). 0.5 with teacher matching converges the content band ~2x faster.",
                ),
            )
            h3_timestep_focus_min = reg(
                "h3_timestep_focus_min",
                gr.Number(label="Focus Band Min", value=get_value("h3_timestep_focus_min"), minimum=0.0, maximum=1.0, step=0.05,
                          info="Lower edge of the focus band (default 0.4)."),
            )
            h3_timestep_focus_max = reg(
                "h3_timestep_focus_max",
                gr.Number(label="Focus Band Max", value=get_value("h3_timestep_focus_max"), minimum=0.0, maximum=1.0, step=0.05,
                          info="Upper edge of the focus band (default 0.8)."),
            )
            min_timestep = reg(
                "min_timestep",
                gr.Number(label="Min Timestep", value=get_value("min_timestep"), precision=0, minimum=0, maximum=1000,
                          info="Clip of the shared base variable (0-1000, before the video/audio shifts). Does not compose with timestep focus."),
            )
            max_timestep = reg(
                "max_timestep",
                gr.Number(label="Max Timestep", value=get_value("max_timestep"), precision=0, minimum=0, maximum=1000,
                          info="1000 = pure noise end of the base range."),
            )
        with gr.Row():
            h3_shift_video = reg(
                "h3_shift_video",
                gr.Number(label="Video Flow Shift", value=get_value("h3_shift_video"), step=0.5,
                          info="Released video shift is 12.0 - change only for experiments."),
            )
            h3_shift_audio = reg(
                "h3_shift_audio",
                gr.Number(label="Audio Flow Shift", value=get_value("h3_shift_audio"), step=0.5,
                          info="Released audio shift is 3.0."),
            )
            h3_visual_cond_clean = reg(
                "h3_visual_cond_clean",
                gr.Number(label="Visual Condition Clean", value=get_value("h3_visual_cond_clean"), minimum=0.0, maximum=1.0, step=0.001,
                          info="Clean coefficient for FL2VA/Ref2VA visual condition augmentation (released: 0.999)."),
            )
            h3_audio_cond_clean = reg(
                "h3_audio_cond_clean",
                gr.Number(label="Audio Condition Clean", value=get_value("h3_audio_cond_clean"), minimum=0.0, maximum=1.0, step=0.001,
                          info="Clean coefficient for reference-audio conditions (released: 1.0)."),
            )

    with gr.Accordion("Network Settings (LoRA)", open=True):
        with gr.Row():
            network_module = reg(
                "network_module",
                gr.Textbox(label="Network Module", value="networks.lora_minimax_h3", interactive=False,
                           info="Fixed to networks.lora_minimax_h3. Targets attn.qkv_proj/out_proj and mlp.fc1/fc2 in the 50 DiT blocks."),
            )
            network_dim = reg(
                "network_dim",
                gr.Slider(label="Network Dimension (Rank)", minimum=1, maximum=256, step=1, value=get_value("network_dim"),
                          info="LoRA rank. 16 is the upstream doc default; 128 for maximum-capacity character/style training."),
            )
            network_alpha = reg(
                "network_alpha",
                gr.Slider(label="Network Alpha", minimum=1, maximum=256, step=1, value=get_value("network_alpha"),
                          info="LoRA alpha. Common practice: alpha = rank."),
            )
            network_dropout = reg(
                "network_dropout",
                gr.Number(label="Network Dropout", value=get_value("network_dropout"), minimum=0.0, maximum=1.0, step=0.05),
            )
        with gr.Row():
            network_weights = reg(
                "network_weights",
                gr.Textbox(label="Network Weights (warm start)", value=get_value("network_weights"),
                           placeholder="Optional path to existing LoRA weights to continue training from"),
            )
            dim_from_weights = reg(
                "dim_from_weights",
                gr.Checkbox(label="Dim from weights", value=get_value("dim_from_weights"),
                            info="Determine rank from the loaded network weights."),
            )
            network_args = reg(
                "network_args",
                gr.Textbox(label="Network Args", value=get_value("network_args"),
                           placeholder='Space separated, e.g. "exclude_patterns=[...]"',
                           info="Extra network arguments passed to networks.lora_minimax_h3."),
            )
            scale_weight_norms = reg(
                "scale_weight_norms",
                gr.Number(label="Scale Weight Norms", value=get_value("scale_weight_norms"), minimum=0.0, step=0.1,
                          info="0 = disabled. Scales weights when their norm exceeds this value."),
            )

    with gr.Accordion("Optimizer and Scheduler Settings", open=True):
        with gr.Row():
            optimizer_type = reg(
                "optimizer_type",
                gr.Dropdown(
                    label="Optimizer Type",
                    choices=add_automagic_optimizer_choices(["AdamW", "AdamW8bit", "AdaFactor", "came", "Lion", "Lion8bit", "prodigyplusscheduleFree"]),
                    allow_custom_value=True,
                    value=get_value("optimizer_type"),
                    info="AdamW8bit is the upstream MiniMax H3 recommendation.",
                ),
            )
            optimizer_args = reg(
                "optimizer_args",
                gr.Textbox(
                    label="Optimizer Arguments",
                    value=get_value("optimizer_args"),
                    placeholder='e.g. "weight_decay=0.01"',
                    info="Space-separated key=value pairs.",
                ),
            )
        optimizer_help = gr.Markdown(optimizer_guidance(get_value("optimizer_type")))
        optimizer_type.change(
            fn=lambda opt: optimizer_guidance(opt),
            inputs=[optimizer_type],
            outputs=[optimizer_help],
            show_progress=False,
        )
        with gr.Row():
            learning_rate = reg(
                "learning_rate",
                gr.Number(label="Learning Rate", value=get_value("learning_rate"), step=1e-6,
                          info="1e-4 is the community-validated H3 LoRA rate (2e-4 for fast low-rank runs)."),
            )
            max_grad_norm = reg(
                "max_grad_norm",
                gr.Number(label="Max Gradient Norm", value=get_value("max_grad_norm"), step=0.1, minimum=0,
                          info="Gradient clipping (0 = disabled)."),
            )
            gradient_accumulation_steps = reg(
                "gradient_accumulation_steps",
                gr.Number(label="Gradient Accumulation Steps", value=get_value("gradient_accumulation_steps"), precision=0, minimum=1,
                          info="H3 batch size is fixed to 1 - use accumulation for a larger effective batch."),
            )
        with gr.Row():
            lr_scheduler = reg(
                "lr_scheduler",
                gr.Dropdown(
                    label="LR Scheduler",
                    choices=["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear", "polynomial", "adafactor"],
                    value=get_value("lr_scheduler"),
                ),
            )
            lr_warmup_steps = reg(
                "lr_warmup_steps",
                gr.Number(label="Warmup Steps", value=get_value("lr_warmup_steps"), precision=0, minimum=0),
            )
            lr_decay_steps = reg(
                "lr_decay_steps",
                gr.Number(label="Decay Steps", value=get_value("lr_decay_steps"), precision=0, minimum=0),
            )
            lr_scheduler_num_cycles = reg(
                "lr_scheduler_num_cycles",
                gr.Number(label="Scheduler Cycles", value=get_value("lr_scheduler_num_cycles"), precision=0, minimum=1),
            )
        with gr.Row():
            lr_scheduler_power = reg(
                "lr_scheduler_power",
                gr.Number(label="Scheduler Power", value=get_value("lr_scheduler_power"), step=0.05),
            )
            lr_scheduler_timescale = reg(
                "lr_scheduler_timescale",
                gr.Number(label="Scheduler Timescale", value=get_value("lr_scheduler_timescale"), precision=0, minimum=0),
            )
            lr_scheduler_min_lr_ratio = reg(
                "lr_scheduler_min_lr_ratio",
                gr.Number(label="Min LR Ratio", value=get_value("lr_scheduler_min_lr_ratio"), step=0.01, minimum=0),
            )
            lr_scheduler_type = reg(
                "lr_scheduler_type",
                gr.Textbox(label="Custom Scheduler Type", value=get_value("lr_scheduler_type"), placeholder="Optional custom scheduler module"),
            )
            lr_scheduler_args = reg(
                "lr_scheduler_args",
                gr.Textbox(label="Scheduler Args", value=get_value("lr_scheduler_args"), placeholder='e.g. "T_max=100"'),
            )

    with gr.Accordion("Training Settings", open=True):
        with gr.Row():
            max_train_epochs = reg(
                "max_train_epochs",
                gr.Number(label="Max Train Epochs", value=get_value("max_train_epochs"), precision=0, minimum=0,
                          info="0 = use max train steps instead. Upstream doc example: 16 epochs."),
            )
            max_train_steps = reg(
                "max_train_steps",
                gr.Number(label="Max Train Steps", value=get_value("max_train_steps"), precision=0, minimum=0,
                          info="0 = derive from epochs. Both set: steps acts as a hard cap."),
            )
            seed = reg("seed", gr.Number(label="Seed", value=get_value("seed"), precision=0, minimum=0))
            max_data_loader_n_workers = reg(
                "max_data_loader_n_workers",
                gr.Number(label="DataLoader Workers", value=get_value("max_data_loader_n_workers"), precision=0, minimum=0),
            )
            persistent_data_loader_workers = reg(
                "persistent_data_loader_workers",
                gr.Checkbox(label="Persistent DataLoader Workers", value=get_value("persistent_data_loader_workers")),
            )

    with gr.Accordion("Saving Settings", open=True):
        with gr.Row():
            output_dir = reg(
                "output_dir",
                gr.Textbox(label="Output Directory", value=get_value("output_dir"), placeholder="Where checkpoints and run configs are written"),
            )
            output_dir_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-teal"], visible=(not headless))
            output_dir_button.click(get_folder_path, inputs=[output_dir], outputs=[output_dir], show_progress=False)
            output_name = reg(
                "output_name",
                gr.Textbox(label="Output Name", value=get_value("output_name"), info="Base filename for saved LoRA checkpoints."),
            )
        with gr.Row():
            save_every_n_epochs = reg(
                "save_every_n_epochs",
                gr.Number(label="Save Every N Epochs", value=get_value("save_every_n_epochs"), precision=0, minimum=0),
            )
            save_every_n_steps = reg(
                "save_every_n_steps",
                gr.Number(label="Save Every N Steps", value=get_value("save_every_n_steps"), precision=0, minimum=0),
            )
            save_last_n_epochs = reg(
                "save_last_n_epochs",
                gr.Number(label="Keep Last N Epochs", value=get_value("save_last_n_epochs"), precision=0, minimum=0, info="0 = keep all"),
            )
            save_last_n_steps = reg(
                "save_last_n_steps",
                gr.Number(label="Keep Last N Steps", value=get_value("save_last_n_steps"), precision=0, minimum=0, info="0 = keep all"),
            )
        with gr.Row():
            save_state = reg(
                "save_state",
                gr.Checkbox(label="Save Training State", value=get_value("save_state"), info="Save optimizer state for resuming."),
            )
            save_state_on_train_end = reg(
                "save_state_on_train_end",
                gr.Checkbox(label="Save State On Train End", value=get_value("save_state_on_train_end")),
            )
            save_last_n_epochs_state = reg(
                "save_last_n_epochs_state",
                gr.Number(label="Keep Last N Epoch States", value=get_value("save_last_n_epochs_state"), precision=0, minimum=0),
            )
            save_last_n_steps_state = reg(
                "save_last_n_steps_state",
                gr.Number(label="Keep Last N Step States", value=get_value("save_last_n_steps_state"), precision=0, minimum=0),
            )
            resume = reg(
                "resume",
                gr.Textbox(label="Resume From State", value=get_value("resume"), placeholder="Path to a saved state folder"),
            )

    with gr.Accordion("Sample Generation Settings", open=False):
        gr.Markdown(
            "Training-time samples decode joint video+audio MP4s under `output_dir/sample`. They load the text encoder on the GPU "
            "(NVFP4 ~15 GB / INT8 ~25 GB / BF16 ~48 GB resident, minus what 'Text Encoder Blocks to Swap' streams from CPU). "
            "Prompt files: .txt lines are auto-augmented with --w/--h/--f/--s/--d; .json files use "
            '[{"prompt", "width", "height", "frame_count", "sample_steps", "seed"}] and are passed through unchanged. '
            "H3 sampling has no negative prompts and no CFG."
        )
        with gr.Row():
            sample_every_n_epochs = reg(
                "sample_every_n_epochs",
                gr.Number(label="Sample Every N Epochs", value=get_value("sample_every_n_epochs"), precision=0, minimum=0),
            )
            sample_every_n_steps = reg(
                "sample_every_n_steps",
                gr.Number(label="Sample Every N Steps", value=get_value("sample_every_n_steps"), precision=0, minimum=0),
            )
            sample_at_first = reg(
                "sample_at_first",
                gr.Checkbox(label="Sample At First", value=get_value("sample_at_first"), info="Generate samples before training starts."),
            )
            h3_allow_experimental_sample_duration = reg(
                "h3_allow_experimental_sample_duration",
                gr.Checkbox(
                    label="Allow experimental sample duration",
                    value=get_value("h3_allow_experimental_sample_duration"),
                    info="Allow training samples shorter than the released 5-15 s range (e.g. 39-frame smoke samples).",
                ),
            )
        with gr.Row():
            sample_prompts = reg(
                "sample_prompts",
                gr.Textbox(label="Sample Prompts File", value=get_value("sample_prompts"),
                           placeholder="Path to a .txt (one prompt per line) or .json prompt file"),
            )
            sample_prompts_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-forest"], visible=(not headless))
            sample_prompts_button.click(get_file_path, inputs=[sample_prompts], outputs=[sample_prompts], show_progress=False)
        with gr.Row():
            width = reg("width", gr.Number(label="Sample Width", value=get_value("width"), precision=0, minimum=64, step=32))
            height = reg("height", gr.Number(label="Sample Height", value=get_value("height"), precision=0, minimum=64, step=32))
            sample_num_frames = reg(
                "sample_num_frames",
                gr.Number(label="Sample Frames", value=get_value("sample_num_frames"), precision=0, minimum=5,
                          info="17n+5 values (124 = 5 s). Rounded down automatically."),
            )
            sample_steps = reg(
                "sample_steps",
                gr.Number(label="Sample Steps", value=get_value("sample_steps"), precision=0, minimum=1,
                          info="30 is the doc default (official 50-step serving parity = 49)."),
            )
            sample_seed = reg("sample_seed", gr.Number(label="Sample Seed", value=get_value("sample_seed"), precision=0, minimum=0))
            disable_prompt_enhancement = reg(
                "disable_prompt_enhancement",
                gr.Checkbox(label="Disable prompt enhancement", value=get_value("disable_prompt_enhancement"),
                            info="Use the .txt prompt file exactly as written."),
            )

    with gr.Accordion("Logging / Metadata / HuggingFace", open=False):
        with gr.Row():
            logging_dir = reg("logging_dir", gr.Textbox(label="Logging Directory", value=get_value("logging_dir")))
            log_with = reg(
                "log_with",
                gr.Dropdown(label="Log With", choices=["", "tensorboard", "wandb", "all"], value=get_value("log_with")),
            )
            log_prefix = reg("log_prefix", gr.Textbox(label="Log Prefix", value=get_value("log_prefix")))
            log_tracker_name = reg("log_tracker_name", gr.Textbox(label="Tracker Name", value=get_value("log_tracker_name")))
            log_tracker_config = reg("log_tracker_config", gr.Textbox(label="Tracker Config Path", value=get_value("log_tracker_config")))
            log_config = reg("log_config", gr.Checkbox(label="Log Training Config", value=get_value("log_config")))
        with gr.Row():
            wandb_api_key = reg("wandb_api_key", gr.Textbox(label="WandB API Key", value=get_value("wandb_api_key")))
            wandb_run_name = reg("wandb_run_name", gr.Textbox(label="WandB Run Name", value=get_value("wandb_run_name")))
        with gr.Row():
            no_metadata = reg("no_metadata", gr.Checkbox(label="No Metadata", value=get_value("no_metadata")))
            metadata_author = reg("metadata_author", gr.Textbox(label="Author", value=get_value("metadata_author")))
            metadata_description = reg("metadata_description", gr.Textbox(label="Description", value=get_value("metadata_description")))
            metadata_license = reg("metadata_license", gr.Textbox(label="License", value=get_value("metadata_license")))
            metadata_tags = reg("metadata_tags", gr.Textbox(label="Tags", value=get_value("metadata_tags")))
            metadata_title = reg("metadata_title", gr.Textbox(label="Title", value=get_value("metadata_title")))
            training_comment = reg("training_comment", gr.Textbox(label="Training Comment", value=get_value("training_comment")))
        with gr.Row():
            huggingface_repo_id = reg("huggingface_repo_id", gr.Textbox(label="HF Repo ID", value=get_value("huggingface_repo_id")))
            huggingface_token = reg("huggingface_token", gr.Textbox(label="HF Token", value=get_value("huggingface_token")))
            huggingface_repo_type = reg(
                "huggingface_repo_type",
                gr.Dropdown(label="HF Repo Type", choices=["model", "dataset"], value=get_value("huggingface_repo_type")),
            )
            huggingface_repo_visibility = reg(
                "huggingface_repo_visibility",
                gr.Dropdown(label="HF Visibility", choices=["private", "public"], value=get_value("huggingface_repo_visibility")),
            )
            huggingface_path_in_repo = reg("huggingface_path_in_repo", gr.Textbox(label="HF Path In Repo", value=get_value("huggingface_path_in_repo")))
        with gr.Row():
            save_state_to_huggingface = reg("save_state_to_huggingface", gr.Checkbox(label="Save State To HF", value=get_value("save_state_to_huggingface")))
            resume_from_huggingface = reg("resume_from_huggingface", gr.Checkbox(label="Resume From HF", value=get_value("resume_from_huggingface")))
            async_upload = reg("async_upload", gr.Checkbox(label="Async Upload", value=get_value("async_upload")))
            ddp_timeout = reg("ddp_timeout", gr.Number(label="DDP Timeout (min)", value=get_value("ddp_timeout"), precision=0, minimum=0))
            ddp_gradient_as_bucket_view = reg("ddp_gradient_as_bucket_view", gr.Checkbox(label="DDP Grad Bucket View", value=get_value("ddp_gradient_as_bucket_view")))
            ddp_static_graph = reg("ddp_static_graph", gr.Checkbox(label="DDP Static Graph", value=get_value("ddp_static_graph")))

    with gr.Accordion("Additional Parameters", open=False):
        with gr.Row():
            additional_parameters = reg(
                "additional_parameters",
                gr.Textbox(
                    label="Additional CLI Parameters",
                    value=get_value("additional_parameters"),
                    placeholder='Appended verbatim to the training command, e.g. --h3_teacher_conditions first,last',
                    info="Escape hatch for any minimax_h3_train_network.py argument not exposed above.",
                ),
            )
            debug_mode = reg(
                "debug_mode",
                gr.Dropdown(
                    label="Debug Mode",
                    choices=["None", "Show Timesteps (Image)", "Show Timesteps (Console)"],
                    value=get_value("debug_mode"),
                ),
            )

    # Register AccelerateLaunch components under their canonical keys.
    accelerate_registrations = [
        ("mixed_precision", accelerate_launch.mixed_precision),
        ("num_processes", accelerate_launch.num_processes),
        ("num_machines", accelerate_launch.num_machines),
        ("num_cpu_threads_per_process", accelerate_launch.num_cpu_threads_per_process),
        ("dynamo_backend", accelerate_launch.dynamo_backend),
        ("dynamo_mode", accelerate_launch.dynamo_mode),
        ("dynamo_use_fullgraph", accelerate_launch.dynamo_use_fullgraph),
        ("dynamo_use_dynamic", accelerate_launch.dynamo_use_dynamic),
        ("multi_gpu", accelerate_launch.multi_gpu),
        ("gpu_ids", accelerate_launch.gpu_ids),
        ("main_process_port", accelerate_launch.main_process_port),
        ("extra_accelerate_launch_args", accelerate_launch.extra_accelerate_launch_args),
    ]
    registrations = accelerate_registrations + registrations

    # Order the registrations to match MINIMAX_H3_PARAM_KEYS exactly.
    registration_map = dict(registrations)
    missing = [key for key in MINIMAX_H3_PARAM_KEYS if key not in registration_map]
    extra = [key for key, _ in registrations if key not in set(MINIMAX_H3_PARAM_KEYS)]
    assert not missing and not extra, f"MiniMax H3 GUI parameter mismatch. Missing: {missing} Extra: {extra}"
    settings_list = [registration_map[key] for key in MINIMAX_H3_PARAM_KEYS]

    # Dataset config mode visibility toggle.
    dataset_config_mode.change(
        fn=lambda mode: gr.Group(visible=(mode == "Generate from Folder Structure")),
        inputs=[dataset_config_mode],
        outputs=[dataset_generation_group],
        show_progress=False,
    )

    # Round target frames to 17n+5 as the user edits.
    dataset_num_frames.blur(
        fn=lambda frames: round_frames_to_minimax_h3(frames),
        inputs=[dataset_num_frames],
        outputs=[dataset_num_frames],
        show_progress=False,
    )
    sample_num_frames.blur(
        fn=lambda frames: round_frames_to_minimax_h3(frames),
        inputs=[sample_num_frames],
        outputs=[sample_num_frames],
        show_progress=False,
    )

    generate_toml_button.click(
        generate_minimax_h3_dataset_toml,
        inputs=settings_list,
        outputs=[dataset_config, generated_toml_path, dataset_status],
        show_progress=True,
    )

    with gr.Column(), gr.Group():
        with gr.Row():
            print_button = gr.Button("Print Command", variant="secondary", elem_classes=["mbtn", "mbtn-slate"])
        executor = CommandExecutor(headless=headless)

    run_state = gr.Textbox(value=str(train_state_value), visible=False)

    configuration.button_open_config.click(
        minimax_h3_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )
    configuration.button_load_config.click(
        minimax_h3_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )
    configuration.button_save_config.click(
        minimax_h3_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )
    print_button.click(
        minimax_h3_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )
    executor.button_run.click(
        minimax_h3_gui_actions,
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
