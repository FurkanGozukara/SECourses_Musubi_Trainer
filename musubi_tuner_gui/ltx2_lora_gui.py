"""LTX-2 / LTX-2.3 video (and audio-video) LoRA training tab.

Standalone video-capable tab modeled on the WAN tab architecture, but with a
single canonical parameter list (LTX2_PARAM_KEYS) shared by every save/load/
train action so parameters can never drift out of order.

Backend scripts (musubi-tuner):
    ltx2_cache_latents.py                 - video/audio VAE latent caching
    ltx2_cache_text_encoder_outputs.py    - Gemma 3 12B text embedding caching
    ltx2_train_network.py                 - LoRA training (networks.lora_ltx2)

The single --ltx2_checkpoint file provides the DiT, video VAE, audio VAE,
vocoder and text-encoder connector weights. Gemma 3 12B is supplied either as
a HuggingFace directory (--gemma_root) or a single safetensors file
(--gemma_safetensors).
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
    generate_ltx2_dataset_config_from_folders,
    round_frames_to_ltx2,
    save_dataset_config,
    validate_dataset_config,
)
from .optimizer_catalog import add_automagic_optimizer_choices, optimizer_guidance, validate_automagic_configuration

log = setup_logging()

LTX2_MODEL_FOLDER = "Training_Models_LTX_2"
LTX2_CHECKPOINT_FILENAME = "ltx-2.3-22b-dev.safetensors"
LTX2_GEMMA_SAFETENSORS_FILENAME = "gemma_3_12B_it_fp8_e4m3fn.safetensors"
LTX2_GEMMA_ROOT_DIRNAME = "gemma-3-12b-it-qat-q4_0-unquantized"
LTX2_MAX_BLOCKS_TO_SWAP = 47  # 48-block LTX-2.3 22B transformer

# ---------------------------------------------------------------------------
# Canonical parameter list. Order MUST match the settings_list built in
# ltx2_lora_tab(); an assert there enforces it.
# ---------------------------------------------------------------------------
LTX2_PARAM_KEYS = [
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
    "ltx2_checkpoint",
    "gemma_root",
    "gemma_safetensors",
    "ltx_version",
    "ltx_version_check_mode",
    "ltx2_mode",
    "vae_dtype",
    "gemma_load_in_8bit",
    "gemma_load_in_4bit",
    "cpu_staged_checkpoint_loading",
    # Attention
    "sdpa",
    "flash_attn",
    "sage_attn",
    "xformers",
    "split_attn_target",
    "split_attn_mode",
    "split_attn_chunk_size",
    # Quantization
    "fp8_base",
    "fp8_scaled",
    "fp8_keep_blocks",
    "fp8_w8a8",
    "int8_convrot_dynamic",
    "int8_convrot_base",
    "int8_convrot_groupsize",
    "int8_convrot_no_mse_clip",
    "int8_convrot_write_quality_report",
    "nf4_base",
    "quantize_device",
    "w8a8_backend",
    # Memory
    "blocks_to_swap",
    "use_pinned_memory_for_block_swap",
    "block_swap_h2d_only",
    "block_swap_ring_size",
    "gradient_checkpointing",
    "gradient_checkpointing_cpu_offload",
    "blockwise_checkpointing",
    "blocks_to_checkpoint",
    "ltx2_low_ram_load",
    # Dataset (GUI-only, consumed by the dataset TOML generator)
    "dataset_config_mode",
    "dataset_config",
    "parent_folder_path",
    "generated_toml_path",
    "dataset_resolution_width",
    "dataset_resolution_height",
    "dataset_batch_size",
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
    "dataset_source_fps",
    "dataset_target_fps",
    # Audio / AV
    "ltx2_audio_source",
    "separate_audio_buckets",
    "ltx2_first_frame_conditioning_p",
    # Caching (GUI-only, consumed by the cache command builders)
    "cache_latents",
    "caching_latent_device",
    "caching_latent_batch_size",
    "caching_latent_num_workers",
    "caching_latent_skip_existing",
    "caching_latent_keep_cache",
    "caching_latent_atomic_writes",
    "caching_latent_vae_spatial_tile_size",
    "caching_latent_vae_spatial_tile_overlap",
    "caching_latent_vae_temporal_tile_size",
    "caching_latent_vae_temporal_tile_overlap",
    "caching_latent_vae_chunk_size",
    "cache_text_encoder_outputs",
    "caching_teo_device",
    "caching_teo_batch_size",
    "caching_teo_num_workers",
    "caching_teo_skip_existing",
    "caching_teo_keep_cache",
    "caching_teo_atomic_writes",
    "cache_before_connector",
    # Network
    "network_module",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "network_weights",
    "base_weights",
    "base_weights_multiplier",
    "dim_from_weights",
    "scale_weight_norms",
    "lora_target_preset",
    "train_connectors",
    "no_convert_to_comfy",
    "no_save_original_lora",
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
    "timestep_sampling",
    "discrete_flow_shift",
    "min_timestep",
    "max_timestep",
    "max_grad_norm_placeholder_removed",
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
    "sample_sampling_preset",
    "sample_sampler",
    "sample_sigma_schedule",
    "sample_with_offloading",
    "sample_tiled_vae",
    "sample_vae_tile_size",
    "sample_vae_tile_overlap",
    "sample_vae_temporal_tile_size",
    "sample_vae_temporal_tile_overlap",
    "sample_merge_audio",
    "sample_disable_audio",
    "sample_audio_only",
    "sample_steps",
    "sample_seed",
    "sample_cfg_scale",
    "sample_negative_prompt",
    "sample_output_dir",
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

# Remove a placeholder that documents where max_grad_norm used to live in an
# earlier draft; keeping the list literal stable is clearer than renumbering.
LTX2_PARAM_KEYS.remove("max_grad_norm_placeholder_removed")

# GUI-only keys that must never reach the training run TOML.
LTX2_RUN_EXCLUSIONS = {
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
    "dataset_batch_size",
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
    "dataset_source_fps",
    "dataset_target_fps",
    "ltx2_audio_source",
    "cache_latents",
    "cache_text_encoder_outputs",
    "cache_before_connector",
    "int8_convrot_write_quality_report",
    "sample_steps",
    "sample_seed",
    "sample_cfg_scale",
    "sample_negative_prompt",
    "sample_output_dir",
    "disable_prompt_enhancement",
    "additional_parameters",
    "debug_mode",
}
# All caching_* keys are consumed by the cache command builders, never by the trainer.
LTX2_RUN_EXCLUSIONS.update(key for key in LTX2_PARAM_KEYS if key.startswith("caching_"))

# Optional string args: drop from the run TOML when empty so the backend
# argparse defaults (None) apply.
LTX2_OPTIONAL_STRING_KEYS = {
    "fp8_keep_blocks",
    "quantize_device",
    "w8a8_backend",
    "gemma_root",
    "gemma_safetensors",
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
    "base_weights",
    "sample_prompts",
}

# Keys whose values must be integers in the run TOML (gr.Slider yields floats).
LTX2_INT_KEYS = {
    "split_attn_chunk_size",
    "blocks_to_swap",
    "block_swap_ring_size",
    "blocks_to_checkpoint",
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
    "sample_vae_tile_size",
    "sample_vae_tile_overlap",
    "sample_vae_temporal_tile_size",
    "sample_vae_temporal_tile_overlap",
    "ddp_timeout",
    "main_process_port",
}

LTX2_DEFAULTS = {
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
    "ltx2_checkpoint": "",
    "gemma_root": "",
    "gemma_safetensors": "",
    "ltx_version": "2.3",
    "ltx_version_check_mode": "warn",
    "ltx2_mode": "video",
    "vae_dtype": "bfloat16",
    "gemma_load_in_8bit": False,
    "gemma_load_in_4bit": False,
    "cpu_staged_checkpoint_loading": False,
    "sdpa": True,
    "flash_attn": False,
    "sage_attn": False,
    "xformers": False,
    "split_attn_target": "none",
    "split_attn_mode": "batch",
    "split_attn_chunk_size": 0,
    "fp8_base": False,
    "fp8_scaled": False,
    "fp8_keep_blocks": "",
    "fp8_w8a8": False,
    "int8_convrot_dynamic": False,
    "int8_convrot_base": False,
    "int8_convrot_groupsize": "auto",
    "int8_convrot_no_mse_clip": False,
    "int8_convrot_write_quality_report": False,
    "nf4_base": False,
    "quantize_device": "",
    "w8a8_backend": "",
    "blocks_to_swap": 0,
    "use_pinned_memory_for_block_swap": False,
    "block_swap_h2d_only": False,
    "block_swap_ring_size": 2,
    "gradient_checkpointing": True,
    "gradient_checkpointing_cpu_offload": False,
    "blockwise_checkpointing": False,
    "blocks_to_checkpoint": -1,
    "ltx2_low_ram_load": False,
    "dataset_config_mode": "Generate from Folder Structure",
    "dataset_config": "",
    "parent_folder_path": "",
    "generated_toml_path": "",
    "dataset_resolution_width": 768,
    "dataset_resolution_height": 512,
    "dataset_batch_size": 1,
    "dataset_enable_bucket": True,
    "dataset_bucket_no_upscale": False,
    "dataset_cache_directory": "cache_dir",
    "dataset_caption_extension": ".txt",
    "caption_strategy": "folder_name",
    "create_missing_captions": True,
    "dataset_num_frames": 33,
    "dataset_frame_extraction": "head",
    "dataset_frame_stride": 1,
    "dataset_frame_sample": 1,
    "dataset_max_frames": 129,
    "dataset_source_fps": 0,
    "dataset_target_fps": 25.0,
    "ltx2_audio_source": "video",
    "separate_audio_buckets": False,
    "ltx2_first_frame_conditioning_p": 0.1,
    "cache_latents": True,
    "caching_latent_device": "cuda",
    "caching_latent_batch_size": 1,
    "caching_latent_num_workers": 2,
    "caching_latent_skip_existing": True,
    "caching_latent_keep_cache": True,
    "caching_latent_atomic_writes": True,
    "caching_latent_vae_spatial_tile_size": 0,
    "caching_latent_vae_spatial_tile_overlap": 64,
    "caching_latent_vae_temporal_tile_size": 0,
    "caching_latent_vae_temporal_tile_overlap": 24,
    "caching_latent_vae_chunk_size": 0,
    "cache_text_encoder_outputs": True,
    "caching_teo_device": "cuda",
    "caching_teo_batch_size": 1,
    "caching_teo_num_workers": 1,
    "caching_teo_skip_existing": True,
    "caching_teo_keep_cache": True,
    "caching_teo_atomic_writes": True,
    "cache_before_connector": False,
    "network_module": "networks.lora_ltx2",
    "network_dim": 32,
    "network_alpha": 32,
    "network_dropout": 0.0,
    "network_args": "",
    "network_weights": "",
    "base_weights": "",
    "base_weights_multiplier": 1.0,
    "dim_from_weights": False,
    "scale_weight_norms": 0.0,
    "lora_target_preset": "t2v",
    "train_connectors": False,
    "no_convert_to_comfy": False,
    "no_save_original_lora": False,
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
    "timestep_sampling": "shifted_logit_normal",
    "discrete_flow_shift": 1.0,
    "min_timestep": 0,
    "max_timestep": 1000,
    "output_dir": "",
    "output_name": "my-ltx2-lora",
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
    "width": 768,
    "height": 512,
    "sample_num_frames": 49,
    "sample_sampling_preset": "defaults",
    "sample_sampler": "auto",
    "sample_sigma_schedule": "auto",
    "sample_with_offloading": True,
    "sample_tiled_vae": True,
    "sample_vae_tile_size": 512,
    "sample_vae_tile_overlap": 64,
    "sample_vae_temporal_tile_size": 48,
    "sample_vae_temporal_tile_overlap": 8,
    "sample_merge_audio": False,
    "sample_disable_audio": False,
    "sample_audio_only": False,
    "sample_steps": 30,
    "sample_seed": 42,
    "sample_cfg_scale": 4.0,
    "sample_negative_prompt": "",
    "sample_output_dir": "",
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
class Ltx2Workflow:
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
    model_dir = os.path.normpath(os.path.join(scriptdir, "..", LTX2_MODEL_FOLDER))
    checkpoint = os.path.join(model_dir, LTX2_CHECKPOINT_FILENAME)
    gemma_file = os.path.join(model_dir, LTX2_GEMMA_SAFETENSORS_FILENAME)
    gemma_root = os.path.join(model_dir, LTX2_GEMMA_ROOT_DIRNAME)
    return {
        "ltx2_checkpoint": checkpoint if os.path.isfile(checkpoint) else "",
        "gemma_safetensors": gemma_file if os.path.isfile(gemma_file) else "",
        "gemma_root": gemma_root if os.path.isdir(gemma_root) else "",
    }


def _validate_ltx2_parameters(param_dict: dict, *, validate_paths: bool = True) -> None:
    """GUI-side validation mirroring the backend's mutual-exclusion rules."""
    ltx2_checkpoint = str(param_dict.get("ltx2_checkpoint") or "").strip()
    if validate_paths:
        if not ltx2_checkpoint:
            raise ValueError("LTX-2 checkpoint path is required (ltx-2.3-22b-dev.safetensors or an INT8 ConvRot/FP8 variant).")
        if not os.path.isfile(ltx2_checkpoint):
            raise ValueError(f"LTX-2 checkpoint does not exist: {ltx2_checkpoint}")

    gemma_root = str(param_dict.get("gemma_root") or "").strip()
    gemma_safetensors = str(param_dict.get("gemma_safetensors") or "").strip()
    if gemma_root and gemma_safetensors:
        raise ValueError("Set either the Gemma HF directory (gemma_root) or the single-file Gemma safetensors, not both.")
    if validate_paths:
        needs_gemma = bool(param_dict.get("cache_text_encoder_outputs", True))
        sampling_active = (
            _to_int(param_dict.get("sample_every_n_epochs")) > 0
            or _to_int(param_dict.get("sample_every_n_steps")) > 0
            or bool(param_dict.get("sample_at_first"))
        )
        needs_gemma = needs_gemma or sampling_active
        if needs_gemma and not gemma_root and not gemma_safetensors:
            raise ValueError(
                "A Gemma 3 12B text encoder is required for text encoder caching / sampling: set gemma_root (HF folder) "
                "or gemma_safetensors (single file)."
            )
        if gemma_root and not os.path.isdir(gemma_root):
            raise ValueError(f"Gemma root directory does not exist: {gemma_root}")
        if gemma_safetensors and not os.path.isfile(gemma_safetensors):
            raise ValueError(f"Gemma safetensors file does not exist: {gemma_safetensors}")

    if param_dict.get("gemma_load_in_8bit") and param_dict.get("gemma_load_in_4bit"):
        raise ValueError("gemma_load_in_8bit and gemma_load_in_4bit are mutually exclusive.")
    if gemma_safetensors and (param_dict.get("gemma_load_in_8bit") or param_dict.get("gemma_load_in_4bit")):
        raise ValueError("Gemma 8-bit/4-bit loading cannot be combined with a single-file gemma_safetensors.")

    fp8 = bool(param_dict.get("fp8_base")) or bool(param_dict.get("fp8_scaled"))
    int8_dynamic = bool(param_dict.get("int8_convrot_dynamic"))
    int8_base = bool(param_dict.get("int8_convrot_base"))
    nf4 = bool(param_dict.get("nf4_base"))
    if int8_dynamic and int8_base:
        raise ValueError("Choose only one of INT8 ConvRot (dynamic) or INT8 ConvRot (pre-quantized base).")
    quant_modes = sum([fp8, int8_dynamic or int8_base, nf4])
    if quant_modes > 1:
        raise ValueError("Choose only one base quantization: FP8, INT8 ConvRot, or NF4.")
    if bool(param_dict.get("fp8_scaled")) and not bool(param_dict.get("fp8_base")):
        raise ValueError("fp8_scaled requires fp8_base to be enabled as well.")
    if bool(param_dict.get("fp8_w8a8")) and not (bool(param_dict.get("fp8_base")) and bool(param_dict.get("fp8_scaled"))):
        raise ValueError("fp8_w8a8 requires fp8_base and fp8_scaled.")

    blocks_to_swap = _to_int(param_dict.get("blocks_to_swap"))
    if blocks_to_swap < 0 or blocks_to_swap > LTX2_MAX_BLOCKS_TO_SWAP:
        raise ValueError(f"blocks_to_swap must be between 0 and {LTX2_MAX_BLOCKS_TO_SWAP}.")
    if param_dict.get("block_swap_h2d_only"):
        if blocks_to_swap <= 0:
            raise ValueError("block_swap_h2d_only requires blocks_to_swap > 0.")
        if not param_dict.get("gradient_checkpointing"):
            raise ValueError("block_swap_h2d_only requires gradient_checkpointing.")

    mode = str(param_dict.get("ltx2_mode") or "video")
    if mode == "audio" and str(param_dict.get("lora_target_preset") or "t2v") == "t2v":
        log.warning("Audio-only training usually needs an audio LoRA target preset (e.g. 'audio').")

    attention = [bool(param_dict.get(k)) for k in ("sdpa", "flash_attn", "sage_attn", "xformers")]
    if sum(attention) == 0:
        raise ValueError("Select one attention implementation (SDPA is recommended).")
    if sum(attention) > 1:
        raise ValueError("Select only one attention implementation (SDPA / FlashAttention / SageAttention / xformers).")

    if _to_int(param_dict.get("network_dim"), 32) <= 0:
        raise ValueError("Network dimension (rank) must be >= 1.")
    if _to_int(param_dict.get("max_train_steps")) <= 0 and _to_int(param_dict.get("max_train_epochs")) <= 0:
        raise ValueError("Set max_train_epochs or max_train_steps to a value greater than 0.")


def _normalize_ltx2_parameters(parameters, *, validate_paths: bool = True):
    param_dict = dict(parameters)

    # Force the fixed network module.
    param_dict["network_module"] = "networks.lora_ltx2"
    parameters = _replace_parameter(parameters, "network_module", "networks.lora_ltx2")

    # Normalize mode aliases.
    mode_map = {"v": "video", "a": "audio", "va": "av"}
    mode = str(param_dict.get("ltx2_mode") or "video").strip().lower()
    mode = mode_map.get(mode, mode)
    if mode not in ("video", "av", "audio"):
        mode = "video"
    param_dict["ltx2_mode"] = mode
    parameters = _replace_parameter(parameters, "ltx2_mode", mode)

    ltx_version = str(param_dict.get("ltx_version") or "2.3").strip()
    if ltx_version not in ("2.0", "2.3"):
        ltx_version = "2.3"
    param_dict["ltx_version"] = ltx_version
    parameters = _replace_parameter(parameters, "ltx_version", ltx_version)

    # Integer coercion for slider/number widgets.
    for key in LTX2_INT_KEYS:
        if key in param_dict and param_dict[key] is not None and not isinstance(param_dict[key], bool):
            coerced = _to_int(param_dict[key], 0)
            param_dict[key] = coerced
            parameters = _replace_parameter(parameters, key, coerced)

    _validate_ltx2_parameters(param_dict, validate_paths=validate_paths)
    return param_dict, parameters


def build_ltx2_cache_commands(param_dict: dict, *, python_cmd: str | None = None):
    python_cmd = python_cmd or sys.executable
    dataset_config = str(param_dict["dataset_config"])
    gpu_ids = param_dict.get("gpu_ids")
    mode = str(param_dict.get("ltx2_mode") or "video")

    latent_command = None
    if bool(param_dict.get("cache_latents", True)):
        latent_command = [
            python_cmd,
            _trainer_script_path("ltx2_cache_latents.py"),
            "--dataset_config",
            dataset_config,
            "--ltx2_checkpoint",
            str(param_dict["ltx2_checkpoint"]),
            "--device",
            _resolve_device(param_dict.get("caching_latent_device"), gpu_ids),
            "--ltx2_mode",
            mode,
        ]
        if param_dict.get("vae_dtype"):
            latent_command.extend(["--vae_dtype", str(param_dict["vae_dtype"])])
        if mode in ("av", "audio"):
            audio_source = str(param_dict.get("ltx2_audio_source") or "video")
            latent_command.extend(["--ltx2_audio_source", audio_source])
        for key, flag in (
            ("caching_latent_batch_size", "--batch_size"),
            ("caching_latent_num_workers", "--num_workers"),
        ):
            value = param_dict.get(key)
            if value is not None:
                latent_command.extend([flag, str(_to_int(value, 1))])
        if param_dict.get("caching_latent_skip_existing"):
            latent_command.append("--skip_existing")
        if param_dict.get("caching_latent_keep_cache"):
            latent_command.append("--keep_cache")
        if param_dict.get("caching_latent_atomic_writes"):
            latent_command.append("--atomic_cache_writes")
        if param_dict.get("cpu_staged_checkpoint_loading"):
            latent_command.append("--cpu_staged_checkpoint_loading")
        spatial_tile = _to_int(param_dict.get("caching_latent_vae_spatial_tile_size"))
        if spatial_tile > 0:
            latent_command.extend(["--vae_spatial_tile_size", str(spatial_tile)])
            latent_command.extend(
                ["--vae_spatial_tile_overlap", str(_to_int(param_dict.get("caching_latent_vae_spatial_tile_overlap"), 64))]
            )
        temporal_tile = _to_int(param_dict.get("caching_latent_vae_temporal_tile_size"))
        if temporal_tile > 0:
            latent_command.extend(["--vae_temporal_tile_size", str(temporal_tile)])
            latent_command.extend(
                ["--vae_temporal_tile_overlap", str(_to_int(param_dict.get("caching_latent_vae_temporal_tile_overlap"), 24))]
            )
        chunk_size = _to_int(param_dict.get("caching_latent_vae_chunk_size"))
        if chunk_size > 0:
            latent_command.extend(["--vae_chunk_size", str(chunk_size)])

    text_command = None
    if bool(param_dict.get("cache_text_encoder_outputs", True)):
        text_command = [
            python_cmd,
            _trainer_script_path("ltx2_cache_text_encoder_outputs.py"),
            "--dataset_config",
            dataset_config,
            "--ltx2_checkpoint",
            str(param_dict["ltx2_checkpoint"]),
            "--device",
            _resolve_device(param_dict.get("caching_teo_device"), gpu_ids),
            "--ltx2_mode",
            mode,
        ]
        gemma_root = str(param_dict.get("gemma_root") or "").strip()
        gemma_safetensors = str(param_dict.get("gemma_safetensors") or "").strip()
        if gemma_safetensors:
            text_command.extend(["--gemma_safetensors", gemma_safetensors])
        elif gemma_root:
            text_command.extend(["--gemma_root", gemma_root])
        mixed_precision = str(param_dict.get("mixed_precision") or "bf16")
        if mixed_precision in ("no", "fp16", "bf16"):
            text_command.extend(["--mixed_precision", mixed_precision])
        if param_dict.get("gemma_load_in_8bit"):
            text_command.append("--gemma_load_in_8bit")
        if param_dict.get("gemma_load_in_4bit"):
            text_command.append("--gemma_load_in_4bit")
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
        if param_dict.get("caching_teo_atomic_writes"):
            text_command.append("--atomic_cache_writes")
        if param_dict.get("cache_before_connector"):
            text_command.append("--cache_before_connector")
        if param_dict.get("cpu_staged_checkpoint_loading"):
            text_command.append("--cpu_staged_checkpoint_loading")

    return latent_command, text_command


def _run_config_parameters(param_dict: dict, parameters):
    """Produce the parameter list for the training run TOML."""
    filtered = []
    for name, value in parameters:
        if name in LTX2_OPTIONAL_STRING_KEYS and isinstance(value, str) and value.strip() == "":
            continue
        filtered.append((name, value))

    # Inject the ConvRot quality report path when requested.
    if bool(param_dict.get("int8_convrot_write_quality_report")) and bool(param_dict.get("int8_convrot_dynamic")):
        output_dir = str(param_dict.get("output_dir") or "").strip()
        output_name = str(param_dict.get("output_name") or "ltx2_lora")
        if output_dir:
            report_path = os.path.join(output_dir, f"{output_name}_int8_convrot_quality.json")
            filtered.append(("int8_convrot_quality_report", report_path))
    return filtered


def _enhance_sample_prompts(param_dict: dict, parameters):
    """Append default --w/--h/--f/--s/--l/--d/--n flags to prompt lines."""
    prompt_path = str(param_dict.get("sample_prompts") or "").strip()
    if not prompt_path or bool(param_dict.get("disable_prompt_enhancement")):
        return param_dict, parameters
    if not os.path.isfile(prompt_path):
        return param_dict, parameters

    width = _to_int(param_dict.get("width"), 768)
    height = _to_int(param_dict.get("height"), 512)
    frames = _to_int(param_dict.get("sample_num_frames"), 49)
    steps = _to_int(param_dict.get("sample_steps"), 30)
    seed = _to_int(param_dict.get("sample_seed"), 42)
    cfg_scale = _to_float(param_dict.get("sample_cfg_scale"), 4.0)
    negative = str(param_dict.get("sample_negative_prompt") or "").strip()

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
            if not has_flag(stripped, "l"):
                stripped += f" --l {cfg_scale}"
            if not has_flag(stripped, "d"):
                stripped += f" --d {seed}"
            if negative and not has_flag(stripped, "n"):
                stripped += f" --n {negative}"
            enhanced_lines.append(stripped)

    sample_dir = str(param_dict.get("sample_output_dir") or "").strip()
    output_dir = sample_dir or str(param_dict.get("output_dir") or "").strip()
    if not output_dir:
        return param_dict, parameters
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = str(param_dict.get("output_name") or "ltx2_lora")
    enhanced_path = os.path.abspath(os.path.join(output_dir, f"{output_name}_enhanced_prompts_{timestamp}.txt"))
    with open(enhanced_path, "w", encoding="utf-8") as handle:
        handle.write("# Generated by SECourses Musubi Trainer (LTX-2)\n")
        for line in enhanced_lines:
            handle.write(f"{line}\n")

    param_dict["sample_prompts"] = enhanced_path
    parameters = _replace_parameter(parameters, "sample_prompts", enhanced_path)
    return param_dict, parameters


def prepare_ltx2_workflow(parameters, config_path: str, *, validate_paths: bool = True, python_cmd: str | None = None) -> Ltx2Workflow:
    param_dict, parameters = _normalize_ltx2_parameters(parameters, validate_paths=validate_paths)
    python_cmd = python_cmd or sys.executable

    if validate_paths:
        os.makedirs(str(param_dict["output_dir"]), exist_ok=True)

    run_parameters = _run_config_parameters(param_dict, parameters)
    SaveConfigFileToRun(
        parameters=run_parameters,
        file_path=config_path,
        exclusion=sorted(LTX2_RUN_EXCLUSIONS | {"file_path", "save_as", "headless", "print_only"}),
        mandatory_keys=["dataset_config", "ltx2_checkpoint", "output_dir", "output_name", "network_module"],
    )

    latent_command, text_command = build_ltx2_cache_commands(param_dict, python_cmd=python_cmd)

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
    train_command.extend([_trainer_script_path("ltx2_train_network.py"), "--config_file", config_path])

    additional = str(param_dict.get("additional_parameters") or "").strip()
    debug = {
        "Show Timesteps (Image)": "--show_timesteps image",
        "Show Timesteps (Console)": "--show_timesteps console",
    }.get(str(param_dict.get("debug_mode") or ""), "")
    if debug:
        additional = f"{additional} {debug}".strip()
    train_command = run_cmd_advanced_training(run_cmd=train_command, additional_parameters=additional)

    return Ltx2Workflow(
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


def _build_workflow_script(workflow: Ltx2Workflow) -> tuple:
    if platform.system() == "Windows":
        # UTF-8 console vars keep backend log lines (which include Japanese
        # help text) from crashing with cp1252 UnicodeEncodeError when the
        # script is re-run manually outside the GUI.
        lines = ["@echo off", "setlocal", "set PYTHONUTF8=1", "set PYTHONIOENCODING=utf-8:backslashreplace"]
        for index, command in enumerate(workflow.commands):
            label = "training" if index == len(workflow.commands) - 1 else "caching"
            lines.append(f"echo Starting LTX-2 {label}...")
            lines.append(_command_line(command))
            lines.append("if errorlevel 1 exit /b %errorlevel%")
        suffix = ".bat"
    else:
        lines = ["#!/bin/bash", "set -e", "export PYTHONUTF8=1", "export PYTHONIOENCODING=utf-8:backslashreplace"]
        for index, command in enumerate(workflow.commands):
            label = "training" if index == len(workflow.commands) - 1 else "caching"
            lines.append(f'echo "Starting LTX-2 {label}..."')
            lines.append(_command_line(command))
        suffix = ".sh"

    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as handle:
        handle.write(content)
        path = handle.name
    if platform.system() != "Windows":
        os.chmod(path, os.stat(path).st_mode | 0o100)
    return path, content


def _create_ltx2_dataset_config(param_dict: dict) -> tuple:
    parent_folder = str(param_dict.get("parent_folder_path") or "").strip()
    if not parent_folder:
        raise ValueError("Parent folder path is required to generate a dataset configuration.")

    resolution = (
        _to_int(param_dict.get("dataset_resolution_width"), 768),
        _to_int(param_dict.get("dataset_resolution_height"), 512),
    )
    source_fps = _to_float(param_dict.get("dataset_source_fps"), 0)
    config, messages = generate_ltx2_dataset_config_from_folders(
        parent_folder=parent_folder,
        resolution=resolution,
        caption_extension=str(param_dict.get("dataset_caption_extension") or ".txt"),
        create_missing_captions=bool(param_dict.get("create_missing_captions", True)),
        caption_strategy=str(param_dict.get("caption_strategy") or "folder_name"),
        batch_size=_to_int(param_dict.get("dataset_batch_size"), 1),
        enable_bucket=bool(param_dict.get("dataset_enable_bucket", True)),
        bucket_no_upscale=bool(param_dict.get("dataset_bucket_no_upscale", False)),
        cache_directory_name=str(param_dict.get("dataset_cache_directory") or "cache_dir"),
        num_frames=_to_int(param_dict.get("dataset_num_frames"), 33),
        frame_extraction=str(param_dict.get("dataset_frame_extraction") or "head"),
        frame_stride=_to_int(param_dict.get("dataset_frame_stride"), 1),
        frame_sample=_to_int(param_dict.get("dataset_frame_sample"), 1),
        max_frames=_to_int(param_dict.get("dataset_max_frames"), 129),
        source_fps=source_fps if source_fps > 0 else None,
        target_fps=_to_float(param_dict.get("dataset_target_fps"), 25.0),
    )

    output_dir = str(param_dict.get("output_dir") or "").strip() or parent_folder
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_path = os.path.abspath(os.path.join(output_dir, f"ltx2_dataset_{timestamp}.toml"))
    save_dataset_config(config, toml_path)
    is_valid, validation_messages = validate_dataset_config(toml_path)
    messages.extend(validation_messages)
    if not is_valid:
        raise ValueError("Generated dataset config failed validation:\n" + "\n".join(validation_messages))
    return toml_path, messages


def generate_ltx2_dataset_toml(*args):
    parameters = list(zip(LTX2_PARAM_KEYS, args))
    param_dict = dict(parameters)
    try:
        toml_path, messages = _create_ltx2_dataset_config(param_dict)
        status = "\n".join(messages + [f"[OK] Dataset config saved: {toml_path}"])
        gr.Info("LTX-2 dataset configuration generated.")
        return toml_path, toml_path, status
    except Exception as exc:
        log.exception("Failed to generate LTX-2 dataset config")
        raise gr.Error(f"{type(exc).__name__}: {exc}", print_exception=False) from exc


def train_ltx2_model(headless: bool, print_only: bool, parameters):
    global train_state_value
    param_dict = dict(parameters)
    validate_automagic_configuration(param_dict, warning_callback=None if headless else gr.Warning)

    if str(param_dict.get("dataset_config_mode") or "") == "Generate from Folder Structure":
        generated = str(param_dict.get("generated_toml_path") or "").strip()
        if not generated or not os.path.isfile(generated):
            generated, _ = _create_ltx2_dataset_config(param_dict)
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
    output_name = str(param_dict.get("output_name") or "ltx2_lora")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_path = os.path.join(output_dir, f"{output_name}_{timestamp}.toml")
    workflow = prepare_ltx2_workflow(parameters, config_path)

    if print_only:
        for command in workflow.commands[:-1]:
            print_command_and_toml(command, "")
        print_command_and_toml(workflow.train_command, workflow.config_path)
        return None

    if executor is None:
        raise RuntimeError("LTX-2 command executor is not initialized.")

    script_path, script_content = _build_workflow_script(workflow)
    save_executed_script(
        script_content=script_content,
        config_name=output_name,
        script_type="ltx2",
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
        gr.Textbox(value="LTX-2 workflow in progress (caching then training)..."),
        gr.Textbox(value=train_state_value),
    )


def save_ltx2_configuration(save_as: bool, file_path: str, parameters):
    original_path = file_path
    if save_as or not file_path:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")
    if not file_path:
        return original_path, gr.update(value="No file selected.", visible=True)
    if not file_path.lower().endswith(".toml"):
        file_path = os.path.splitext(file_path)[0] + ".toml"

    try:
        _, parameters = _normalize_ltx2_parameters(parameters, validate_paths=False)
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
        return "networks.lora_ltx2"
    if key == "ltx_version":
        text = str(value).strip() if value is not None else "2.3"
        return text if text in ("2.0", "2.3") else "2.3"
    if key == "ltx2_mode":
        text = str(value or "video").strip().lower()
        return {"v": "video", "a": "audio", "va": "av"}.get(text, text if text in ("video", "av", "audio") else "video")
    if isinstance(value, list):
        if key in {"optimizer_args", "lr_scheduler_args", "network_args", "base_weights"}:
            return " ".join(str(item) for item in value)
        if key == "base_weights_multiplier":
            return value[0] if value else default
        if not isinstance(default, list):
            return value[0] if value else default
    return value


def open_ltx2_configuration(ask_for_file: bool, file_path: str, parameters):
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


def ltx2_gui_actions(action: str, ask_for_file: bool, config_file_name: str, headless: bool, print_only: bool, *args):
    parameters = list(zip(LTX2_PARAM_KEYS, args))
    if action == "open_configuration":
        return open_ltx2_configuration(ask_for_file, config_file_name, parameters)
    if action == "save_configuration":
        return save_ltx2_configuration(ask_for_file, config_file_name, parameters)
    if action == "train_model":
        try:
            return train_ltx2_model(headless, print_only, parameters)
        except gr.Error:
            raise
        except Exception as exc:
            log.exception("Failed to start LTX-2 training")
            raise gr.Error(
                f"{type(exc).__name__}: {exc}",
                title="LTX-2 training could not start",
                duration=None,
                print_exception=False,
            ) from exc
    raise ValueError(f"Unknown GUI action: {action}")


def ltx2_lora_tab(headless=False, config: GUIConfig = {}):
    global executor
    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    model_defaults = _default_model_paths()

    def get_value(key):
        default = LTX2_DEFAULTS.get(key)
        if key in model_defaults:
            return config.get(key, None) or model_defaults[key] or default
        return config.get(key, default)

    registrations = []

    def reg(key, component):
        registrations.append((key, component))
        return component

    gr.Markdown(
        "Train LoRA models for **LTX-2 (19B)** and **LTX-2.3 (22B)** text-to-video / audio-video models. "
        "The single LTX-2 checkpoint provides DiT + VAE + audio VAE + vocoder; Gemma 3 12B is the text encoder. "
        "Valid video frame counts are 8k+1 (1, 9, 17, 25, 33, 41, 49...) and training resolution steps are 32 px."
    )

    with gr.Accordion("Configuration File", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    with gr.Row():
        search_input = gr.Textbox(
            label="Search Settings",
            placeholder="Search LTX 2.3 settings (for example: model, quantization, dataset, cache, optimizer)",
            lines=1,
        )
        toggle_all_button = gr.Button(
            "Open All Panels",
            variant="secondary",
            elem_classes=["mbtn", "mbtn-indigo"],
        )
        panels_state = gr.State(value="mixed")
    search_results = gr.HTML(visible=False)

    accordions = []
    panel_search_specs = []

    def tracked_panel(label, *, keywords="", open=False, **kwargs):
        accordion = gr.Accordion(label, open=open, **kwargs)
        accordions.append(accordion)
        panel_search_specs.append((label, f"{label} {keywords}".lower()))
        return accordion

    with tracked_panel(
        "Accelerate launch Settings",
        keywords="resource gpu precision distributed multi gpu dynamo cpu threads port",
        open=True,
        elem_classes="flux1_background",
    ):
        accelerate_launch = AccelerateLaunch(config=config)
        # Ensure single-GPU default threads=1 like other tabs when unset is handled by class defaults.

    with tracked_panel(
        "Model Settings",
        keywords="checkpoint dit transformer gemma text encoder vae vocoder model paths",
        open=True,
    ):
        with gr.Row():
            ltx2_checkpoint = reg(
                "ltx2_checkpoint",
                gr.Textbox(
                    label="LTX-2 Checkpoint Path",
                    placeholder="Path to ltx-2.3-22b-dev.safetensors (also provides VAE / audio VAE / vocoder)",
                    value=get_value("ltx2_checkpoint"),
                    info="Single-file LTX-2 / LTX-2.3 checkpoint. Pre-quantized FP8 and INT8 ConvRot checkpoints are supported.",
                ),
            )
            ltx2_checkpoint_button = gr.Button("📂", elem_id="ltx2_checkpoint_button", elem_classes=["mbtn", "mbtn-blue"], size="sm", visible=(not headless))
            ltx2_checkpoint_button.click(
                get_file_path,
                inputs=[ltx2_checkpoint],
                outputs=[ltx2_checkpoint],
                show_progress=False,
            )
        with gr.Row():
            gemma_safetensors = reg(
                "gemma_safetensors",
                gr.Textbox(
                    label="Gemma 3 12B Single File (safetensors)",
                    placeholder="Path to gemma_3_12B_it_fp8_e4m3fn.safetensors (recommended: smaller download)",
                    value=get_value("gemma_safetensors"),
                    info="Single-file Gemma text encoder (weights + tokenizer + config). Leave empty when using the HF folder.",
                ),
            )
            gemma_safetensors_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-lime"], visible=(not headless))
            gemma_safetensors_button.click(
                get_file_path,
                inputs=[gemma_safetensors],
                outputs=[gemma_safetensors],
                show_progress=False,
            )
        with gr.Row():
            gemma_root = reg(
                "gemma_root",
                gr.Textbox(
                    label="Gemma 3 12B HF Directory (alternative)",
                    placeholder="Path to gemma-3-12b-it-qat-q4_0-unquantized folder (enables 8-bit/4-bit loading)",
                    value=get_value("gemma_root"),
                    info="HuggingFace-format Gemma folder. Required for gemma_load_in_8bit / 4bit. Leave empty when using the single file.",
                ),
            )
            gemma_root_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-fuchsia"], visible=(not headless))
            gemma_root_button.click(
                get_folder_path,
                inputs=[gemma_root],
                outputs=[gemma_root],
                show_progress=False,
            )
        with gr.Row():
            ltx_version = reg(
                "ltx_version",
                gr.Dropdown(
                    label="LTX Version",
                    choices=["2.3", "2.0"],
                    value=str(get_value("ltx_version")),
                    info="2.3 = LTX-2.3 22B checkpoint, 2.0 = LTX-2 19B checkpoint. Must match the checkpoint file.",
                ),
            )
            ltx_version_check_mode = reg(
                "ltx_version_check_mode",
                gr.Dropdown(
                    label="Version Check Mode",
                    choices=["off", "warn", "error"],
                    value=get_value("ltx_version_check_mode"),
                    info="How to react when checkpoint metadata does not match the selected LTX version.",
                ),
            )
            ltx2_mode = reg(
                "ltx2_mode",
                gr.Dropdown(
                    label="Training Modality",
                    choices=["video", "av", "audio"],
                    value=get_value("ltx2_mode"),
                    info="video = video-only, av = joint audio+video, audio = audio-only. Caching and training must use the same mode.",
                ),
            )
            vae_dtype = reg(
                "vae_dtype",
                gr.Dropdown(
                    label="VAE Data Type",
                    choices=["bfloat16", "float16", "float32"],
                    value=get_value("vae_dtype"),
                    info="Data type for VAE latent caching and sampling.",
                ),
            )
        with gr.Row():
            gemma_load_in_8bit = reg(
                "gemma_load_in_8bit",
                gr.Checkbox(
                    label="Gemma 8-bit (bitsandbytes)",
                    value=get_value("gemma_load_in_8bit"),
                    info="Load Gemma in 8-bit to save VRAM during text encoder caching/sampling. Requires the HF directory.",
                ),
            )
            gemma_load_in_4bit = reg(
                "gemma_load_in_4bit",
                gr.Checkbox(
                    label="Gemma 4-bit (bitsandbytes)",
                    value=get_value("gemma_load_in_4bit"),
                    info="Load Gemma in 4-bit NF4. Requires the HF directory.",
                ),
            )
            cpu_staged_checkpoint_loading = reg(
                "cpu_staged_checkpoint_loading",
                gr.Checkbox(
                    label="CPU-staged checkpoint loading",
                    value=get_value("cpu_staged_checkpoint_loading"),
                    info="Compatibility mode: load checkpoint tensors on CPU first. Slower, but works around CUDA safetensors loading errors.",
                ),
            )
            ltx2_first_frame_conditioning_p = reg(
                "ltx2_first_frame_conditioning_p",
                gr.Number(
                    label="First-frame conditioning probability",
                    value=get_value("ltx2_first_frame_conditioning_p"),
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    info="Probability of keeping frame 0 clean (I2V-style conditioning). 0.1 default; 0.0 = pure T2V.",
                ),
            )

    with tracked_panel(
        "Attention Settings",
        keywords="sdpa flash attention sage xformers split attention",
        open=False,
    ):
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
        with gr.Row():
            split_attn_target = reg(
                "split_attn_target",
                gr.Dropdown(
                    label="Split Attention Target",
                    choices=["none", "all", "self", "cross", "text_cross", "av_cross", "video", "audio"],
                    value=get_value("split_attn_target"),
                    info="Split attention modules to reduce VRAM. 'none' disables splitting.",
                ),
            )
            split_attn_mode = reg(
                "split_attn_mode",
                gr.Dropdown(
                    label="Split Attention Mode",
                    choices=["batch", "query"],
                    value=get_value("split_attn_mode"),
                    info="Split by batch dimension or query length.",
                ),
            )
            split_attn_chunk_size = reg(
                "split_attn_chunk_size",
                gr.Number(
                    label="Split Attention Chunk Size",
                    value=get_value("split_attn_chunk_size"),
                    precision=0,
                    minimum=0,
                    info="Chunk size for query-based splitting (0 = default 1024).",
                ),
            )

    with tracked_panel(
        "Quantization and Memory Settings",
        keywords="fp8 int8 convrot nf4 block swap offload memory dtype quantization",
        open=True,
    ):
        gr.Markdown(
            "Pick **one** base quantization: FP8 (fast, ~19 GB), **INT8 ConvRot** (higher fidelity than FP8), or NF4 (~10 GB). "
            "INT8 ConvRot rotates weights with a group-wise Hadamard matrix before int8 quantization — recommended for quality."
        )
        with gr.Row():
            fp8_base = reg(
                "fp8_base",
                gr.Checkbox(label="FP8 Base", value=get_value("fp8_base"), info="Keep base model weights in FP8."),
            )
            fp8_scaled = reg(
                "fp8_scaled",
                gr.Checkbox(label="FP8 Scaled", value=get_value("fp8_scaled"), info="Quantize checkpoint weights to scaled FP8 at load time. Use together with FP8 Base."),
            )
            fp8_keep_blocks = reg(
                "fp8_keep_blocks",
                gr.Textbox(
                    label="FP8 Keep Blocks",
                    value=get_value("fp8_keep_blocks"),
                    placeholder="e.g. 0,1,46,47",
                    info="Optional comma list/ranges of transformer blocks kept in bf16 with FP8 Scaled.",
                ),
            )
            fp8_w8a8 = reg(
                "fp8_w8a8",
                gr.Checkbox(label="W8A8 Activations", value=get_value("fp8_w8a8"), info="With FP8 Base+Scaled: also quantize activations (int8 W8A8)."),
            )
        with gr.Row():
            int8_convrot_dynamic = reg(
                "int8_convrot_dynamic",
                gr.Checkbox(
                    label="INT8 ConvRot (quantize at load)",
                    value=get_value("int8_convrot_dynamic"),
                    info="Quantize a standard bf16/fp16 checkpoint to INT8 ConvRot while loading. Higher fidelity than FP8 at similar VRAM.",
                ),
            )
            int8_convrot_base = reg(
                "int8_convrot_base",
                gr.Checkbox(
                    label="INT8 ConvRot (pre-quantized checkpoint)",
                    value=get_value("int8_convrot_base"),
                    info="Load a checkpoint already converted with ltx2_quantize_int8_convrot.py (Comfy-compatible).",
                ),
            )
            int8_convrot_groupsize = reg(
                "int8_convrot_groupsize",
                gr.Dropdown(
                    label="ConvRot Group Size",
                    choices=["auto", "256", "64", "16"],
                    value=str(get_value("int8_convrot_groupsize")),
                    allow_custom_value=True,
                    info="Hadamard rotation group size. 'auto' picks the largest divisor of each layer (256, 64, 16).",
                ),
            )
            int8_convrot_no_mse_clip = reg(
                "int8_convrot_no_mse_clip",
                gr.Checkbox(
                    label="Disable MSE clip",
                    value=get_value("int8_convrot_no_mse_clip"),
                    info="Use plain absmax row scales instead of MSE-optimal clipping (faster load, slightly lower quality).",
                ),
            )
            int8_convrot_write_quality_report = reg(
                "int8_convrot_write_quality_report",
                gr.Checkbox(
                    label="Write quality report",
                    value=get_value("int8_convrot_write_quality_report"),
                    info="Write per-layer INT8 ConvRot reconstruction metrics JSON into the output folder.",
                ),
            )
        with gr.Row():
            nf4_base = reg(
                "nf4_base",
                gr.Checkbox(label="NF4 Base", value=get_value("nf4_base"), info="NF4 4-bit base quantization (~10 GB weights). Lowest VRAM, lowest fidelity."),
            )
            quantize_device = reg(
                "quantize_device",
                gr.Dropdown(
                    label="Quantize Device",
                    choices=["", "cuda", "cpu"],
                    value=get_value("quantize_device"),
                    allow_custom_value=True,
                    info="Device for startup quantization. Empty = backend default (cuda). 'cpu' lowers GPU load-time peaks at the cost of RAM/time.",
                ),
            )
            w8a8_backend = reg(
                "w8a8_backend",
                gr.Dropdown(
                    label="INT8 Matmul Backend",
                    choices=["", "torch", "triton", "cutlass"],
                    value=get_value("w8a8_backend"),
                    info="Backend for int8 matmuls (ConvRot / W8A8). Empty = auto (torch._int_mm when available).",
                ),
            )
        with gr.Row():
            blocks_to_swap = reg(
                "blocks_to_swap",
                gr.Slider(
                    label="Blocks to Swap",
                    minimum=0,
                    maximum=LTX2_MAX_BLOCKS_TO_SWAP,
                    step=1,
                    value=get_value("blocks_to_swap"),
                    info=f"Offload N transformer blocks to CPU RAM (0-{LTX2_MAX_BLOCKS_TO_SWAP}). Higher = less VRAM, slower steps, more system RAM.",
                ),
            )
            use_pinned_memory_for_block_swap = reg(
                "use_pinned_memory_for_block_swap",
                gr.Checkbox(
                    label="Pinned memory for block swap",
                    value=get_value("use_pinned_memory_for_block_swap"),
                    info="Faster block transfers when host RAM permits. Disable if you hit block-swap crashes.",
                ),
            )
            block_swap_h2d_only = reg(
                "block_swap_h2d_only",
                gr.Checkbox(
                    label="H2D-only block swap",
                    value=get_value("block_swap_h2d_only"),
                    info="Stream frozen LoRA base blocks Host-to-Device only. Requires block swap > 0 and gradient checkpointing.",
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
                    info="Offload checkpointed activations to CPU (more RAM, less VRAM).",
                ),
            )
            blockwise_checkpointing = reg(
                "blockwise_checkpointing",
                gr.Checkbox(
                    label="Blockwise checkpointing",
                    value=get_value("blockwise_checkpointing"),
                    info="Checkpoint transformer blocks individually. Lowest peak VRAM, heavy CPU traffic.",
                ),
            )
            blocks_to_checkpoint = reg(
                "blocks_to_checkpoint",
                gr.Number(
                    label="Blocks to checkpoint",
                    value=get_value("blocks_to_checkpoint"),
                    precision=0,
                    minimum=-1,
                    info="-1 = all transformer blocks (with blockwise checkpointing).",
                ),
            )
            ltx2_low_ram_load = reg(
                "ltx2_low_ram_load",
                gr.Checkbox(
                    label="Low main-RAM loading",
                    value=get_value("ltx2_low_ram_load"),
                    info="Write each weight straight to its final device so the checkpoint is never staged whole in main RAM.",
                ),
            )

    with tracked_panel(
        "Dataset Settings",
        keywords="dataset config folder resolution caption batch bucket frames video image toml",
        open=True,
    ):
        with gr.Row():
            dataset_config_mode = reg(
                "dataset_config_mode",
                gr.Dropdown(
                    label="Dataset Configuration Mode",
                    choices=["Generate from Folder Structure", "Use TOML File"],
                    value=get_value("dataset_config_mode"),
                    info="Generate a dataset TOML from a parent folder of subfolders, or supply your own TOML.",
                ),
            )
        with gr.Group() as dataset_generation_group:
            with gr.Row():
                parent_folder_path = reg(
                    "parent_folder_path",
                    gr.Textbox(
                        label="Parent Folder Path",
                        placeholder="Folder containing subfolders of videos/images (e.g. 1_mydataset)",
                        value=get_value("parent_folder_path"),
                        info="Each subfolder becomes a dataset. Prefix with repeats like '3_name'. Captions are .txt files next to the media.",
                    ),
                )
                parent_folder_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-pink"], visible=(not headless))
                parent_folder_button.click(
                    get_folder_path,
                    inputs=[parent_folder_path],
                    outputs=[parent_folder_path],
                    show_progress=False,
                )
            with gr.Row():
                dataset_resolution_width = reg(
                    "dataset_resolution_width",
                    gr.Number(label="Resolution Width", value=get_value("dataset_resolution_width"), precision=0, minimum=64, step=32,
                              info="Training width (32 px steps for LTX-2). Default 768."),
                )
                dataset_resolution_height = reg(
                    "dataset_resolution_height",
                    gr.Number(label="Resolution Height", value=get_value("dataset_resolution_height"), precision=0, minimum=64, step=32,
                              info="Training height (32 px steps for LTX-2). Default 512."),
                )
                dataset_batch_size = reg(
                    "dataset_batch_size",
                    gr.Number(label="Dataset Batch Size", value=get_value("dataset_batch_size"), precision=0, minimum=1,
                              info="Per-bucket batch size used during training."),
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
                                info="Create caption files for media without one."),
                )
            gr.Markdown("**🎬 Video Frame Settings** — LTX-2 frame counts must be 8k+1 (1, 9, 17, 25, 33, 41, 49, ...). Videos are resampled to the target FPS (default 25).")
            with gr.Row():
                dataset_num_frames = reg(
                    "dataset_num_frames",
                    gr.Number(label="Target Frames", value=get_value("dataset_num_frames"), precision=0, minimum=1,
                              info="Frames per training sample. Rounded down to 8k+1 automatically."),
                )
                dataset_frame_extraction = reg(
                    "dataset_frame_extraction",
                    gr.Dropdown(
                        label="Frame Extraction",
                        choices=["head", "chunk", "slide", "uniform", "full"],
                        value=get_value("dataset_frame_extraction"),
                        info="How segments are pulled from each video: head = first N frames, chunk = consecutive chunks, slide = sliding window, uniform = evenly spaced samples, full = whole video.",
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
                    gr.Number(label="Max Frames", value=get_value("dataset_max_frames"), precision=0, minimum=1,
                              info="Upper bound on frames used from each video (default 129)."),
                )
                dataset_source_fps = reg(
                    "dataset_source_fps",
                    gr.Number(label="Source FPS Override", value=get_value("dataset_source_fps"), minimum=0,
                              info="0 = auto-detect from each video. Set only for variable-frame-rate videos with wrong detection."),
                )
                dataset_target_fps = reg(
                    "dataset_target_fps",
                    gr.Number(label="Target FPS", value=get_value("dataset_target_fps"), minimum=1,
                              info="Videos are resampled to this rate during caching (LTX-2 default 25)."),
                )
                generate_toml_button = gr.Button("🛠️ Generate Dataset Config", variant="secondary", elem_classes=["mbtn", "mbtn-orange"])
        with gr.Row():
            dataset_config = reg(
                "dataset_config",
                gr.Textbox(
                    label="Dataset Config TOML",
                    placeholder="Path to dataset config .toml (auto-filled when generated)",
                    value=get_value("dataset_config"),
                    info="The dataset TOML used by caching and training.",
                ),
            )
            dataset_config_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-gold"], visible=(not headless))
            dataset_config_button.click(
                get_file_path,
                inputs=[dataset_config],
                outputs=[dataset_config],
                show_progress=False,
            )
            generated_toml_path = reg(
                "generated_toml_path",
                gr.Textbox(label="Generated TOML Path", value=get_value("generated_toml_path"), interactive=False,
                           info="Last generated dataset TOML (reused on Start Training when present)."),
            )
        dataset_status = gr.Textbox(label="Dataset Generation Status", lines=5, interactive=False, value="")

    with tracked_panel(
        "Audio / AV Settings",
        keywords="audio video av loss waveform mel spectrogram conditioning source",
        open=False,
    ):
        with gr.Row():
            ltx2_audio_source = reg(
                "ltx2_audio_source",
                gr.Dropdown(
                    label="Audio Source (caching)",
                    choices=["video", "audio_files"],
                    value=get_value("ltx2_audio_source"),
                    info="For av/audio modes: take audio from the video container or from external .wav files next to the videos.",
                ),
            )
            separate_audio_buckets = reg(
                "separate_audio_buckets",
                gr.Checkbox(
                    label="Separate audio buckets",
                    value=get_value("separate_audio_buckets"),
                    info="Keep audio/non-audio items in separate batches. Effectively required for mixed AV datasets with batch size > 1.",
                ),
            )

    with tracked_panel(
        "Caching Settings",
        keywords="cache latent text encoder device batch workers skip existing",
        open=False,
    ):
        gr.Markdown("Latent and text-encoder caches are (re)built automatically before each training start; existing cache files are skipped.")
        with gr.Row():
            cache_latents = reg(
                "cache_latents",
                gr.Checkbox(label="Cache Latents", value=get_value("cache_latents"), info="Run VAE latent caching before training."),
            )
            caching_latent_device = reg(
                "caching_latent_device",
                gr.Dropdown(label="Latent Device", choices=["cuda", "cpu"], value=get_value("caching_latent_device"), allow_custom_value=True,
                            info="Device for VAE encoding (first GPU from gpu_ids is used for 'cuda')."),
            )
            caching_latent_batch_size = reg(
                "caching_latent_batch_size",
                gr.Number(label="Latent Batch Size", value=get_value("caching_latent_batch_size"), precision=0, minimum=1),
            )
            caching_latent_num_workers = reg(
                "caching_latent_num_workers",
                gr.Number(label="Latent Workers", value=get_value("caching_latent_num_workers"), precision=0, minimum=0),
            )
            caching_latent_skip_existing = reg(
                "caching_latent_skip_existing",
                gr.Checkbox(label="Skip Existing", value=get_value("caching_latent_skip_existing")),
            )
            caching_latent_keep_cache = reg(
                "caching_latent_keep_cache",
                gr.Checkbox(label="Keep Cache", value=get_value("caching_latent_keep_cache"), info="Keep cache files for removed/changed media."),
            )
            caching_latent_atomic_writes = reg(
                "caching_latent_atomic_writes",
                gr.Checkbox(label="Atomic Writes", value=get_value("caching_latent_atomic_writes"),
                            info="Write caches to a temp file first, then replace. Safer on crashes."),
            )
        with gr.Row():
            caching_latent_vae_spatial_tile_size = reg(
                "caching_latent_vae_spatial_tile_size",
                gr.Number(label="VAE Spatial Tile (px)", value=get_value("caching_latent_vae_spatial_tile_size"), precision=0, minimum=0, step=32,
                          info="0 = no spatial tiling. Otherwise >= 64 and divisible by 32; reduces caching VRAM."),
            )
            caching_latent_vae_spatial_tile_overlap = reg(
                "caching_latent_vae_spatial_tile_overlap",
                gr.Number(label="Spatial Overlap (px)", value=get_value("caching_latent_vae_spatial_tile_overlap"), precision=0, minimum=0, step=32),
            )
            caching_latent_vae_temporal_tile_size = reg(
                "caching_latent_vae_temporal_tile_size",
                gr.Number(label="VAE Temporal Tile (frames)", value=get_value("caching_latent_vae_temporal_tile_size"), precision=0, minimum=0, step=8,
                          info="0 = no temporal tiling. Otherwise >= 16 and divisible by 8."),
            )
            caching_latent_vae_temporal_tile_overlap = reg(
                "caching_latent_vae_temporal_tile_overlap",
                gr.Number(label="Temporal Overlap (frames)", value=get_value("caching_latent_vae_temporal_tile_overlap"), precision=0, minimum=0, step=8),
            )
            caching_latent_vae_chunk_size = reg(
                "caching_latent_vae_chunk_size",
                gr.Number(label="VAE Chunk Size", value=get_value("caching_latent_vae_chunk_size"), precision=0, minimum=0,
                          info="CausalConv3d chunking (0 = disabled)."),
            )
        with gr.Row():
            cache_text_encoder_outputs = reg(
                "cache_text_encoder_outputs",
                gr.Checkbox(label="Cache Text Encoder Outputs", value=get_value("cache_text_encoder_outputs"),
                            info="Run Gemma text-embedding caching before training."),
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
            caching_teo_atomic_writes = reg(
                "caching_teo_atomic_writes",
                gr.Checkbox(label="Atomic Writes", value=get_value("caching_teo_atomic_writes")),
            )
            cache_before_connector = reg(
                "cache_before_connector",
                gr.Checkbox(label="Cache pre-connector features", value=get_value("cache_before_connector"),
                            info="Also save pre-connector text features (needed for connector LoRA training)."),
            )

    with tracked_panel(
        "Network Settings (LoRA)",
        keywords="network lora rank dim alpha dropout weights module",
        open=True,
    ):
        with gr.Row():
            network_module = reg(
                "network_module",
                gr.Textbox(label="Network Module", value="networks.lora_ltx2", interactive=False,
                           info="Fixed to networks.lora_ltx2 for LTX-2 LoRA training."),
            )
            network_dim = reg(
                "network_dim",
                gr.Slider(label="Network Dimension (Rank)", minimum=1, maximum=256, step=1, value=get_value("network_dim"),
                          info="LoRA rank. 32 is the LTX-2 doc default; 128 for maximum-capacity character/style training."),
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
            lora_target_preset = reg(
                "lora_target_preset",
                gr.Dropdown(
                    label="LoRA Target Preset",
                    choices=["t2v", "v2v", "video_sa", "video_sa_ff", "video_sa_ca_ff", "audio", "audio_v2a", "audio_ref_ic", "av_ic", "video_ref_only_av", "full"],
                    value=get_value("lora_target_preset"),
                    info="Which modules receive LoRA. t2v = video attention+FFN (standard); audio presets for audio training; full = everything.",
                ),
            )
            train_connectors = reg(
                "train_connectors",
                gr.Checkbox(label="Train text connectors", value=get_value("train_connectors"),
                            info="Add LoRA on the Gemma embeddings connector. Requires 'Cache pre-connector features' during TE caching."),
            )
            no_convert_to_comfy = reg(
                "no_convert_to_comfy",
                gr.Checkbox(label="Skip ComfyUI export", value=get_value("no_convert_to_comfy"),
                            info="By default both musubi (.safetensors) and ComfyUI (.comfy.safetensors) LoRA files are saved. Check to save only the musubi format."),
            )
            no_save_original_lora = reg(
                "no_save_original_lora",
                gr.Checkbox(label="Keep only ComfyUI file", value=get_value("no_save_original_lora"),
                            info="Delete the original after ComfyUI conversion. WARNING: resume requires the original format."),
            )
        with gr.Row():
            network_weights = reg(
                "network_weights",
                gr.Textbox(label="Network Weights (warm start)", value=get_value("network_weights"),
                           placeholder="Optional path to existing LoRA weights to continue training from"),
            )
            base_weights = reg(
                "base_weights",
                gr.Textbox(label="Base Weights (merge before training)", value=get_value("base_weights"),
                           placeholder="Optional LoRA weights merged into the base before training"),
            )
            base_weights_multiplier = reg(
                "base_weights_multiplier",
                gr.Number(label="Base Weights Multiplier", value=get_value("base_weights_multiplier"), step=0.05),
            )
            dim_from_weights = reg(
                "dim_from_weights",
                gr.Checkbox(label="Dim from weights", value=get_value("dim_from_weights"),
                            info="Determine rank from the loaded network weights."),
            )
        with gr.Row():
            network_args = reg(
                "network_args",
                gr.Textbox(label="Network Args", value=get_value("network_args"),
                           placeholder='Space separated, e.g. "loraplus_lr_ratio=4" "exclude_patterns=[...]"',
                           info="Extra network arguments passed to networks.lora_ltx2."),
            )
            scale_weight_norms = reg(
                "scale_weight_norms",
                gr.Number(label="Scale Weight Norms", value=get_value("scale_weight_norms"), minimum=0.0, step=0.1,
                          info="0 = disabled. Scales weights when their norm exceeds this value."),
            )

    with tracked_panel(
        "Optimizer and Scheduler Settings",
        keywords="optimizer learning rate scheduler warmup decay cycles gradient norm",
        open=True,
    ):
        with gr.Row():
            optimizer_type = reg(
                "optimizer_type",
                gr.Dropdown(
                    label="Optimizer Type",
                    choices=add_automagic_optimizer_choices(["AdamW", "AdamW8bit", "AdaFactor", "came", "Lion", "Lion8bit", "prodigyplusscheduleFree"]),
                    allow_custom_value=True,
                    value=get_value("optimizer_type"),
                    info="AdamW8bit is a robust default. AdaFactor matches our other demo presets.",
                ),
            )
            optimizer_args = reg(
                "optimizer_args",
                gr.Textbox(
                    label="Optimizer Arguments",
                    value=get_value("optimizer_args"),
                    placeholder='e.g. "scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01"',
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
                          info="1e-4 is the LTX-2 doc recommendation for rank 32; lower slightly for very high ranks."),
            )
            max_grad_norm = reg(
                "max_grad_norm",
                gr.Number(label="Max Gradient Norm", value=get_value("max_grad_norm"), step=0.1, minimum=0,
                          info="Gradient clipping (0 = disabled)."),
            )
            gradient_accumulation_steps = reg(
                "gradient_accumulation_steps",
                gr.Number(label="Gradient Accumulation Steps", value=get_value("gradient_accumulation_steps"), precision=0, minimum=1),
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

    with tracked_panel(
        "Training Settings",
        keywords="training epochs steps workers seed gradient accumulation checkpointing precision",
        open=True,
    ):
        with gr.Row():
            max_train_epochs = reg(
                "max_train_epochs",
                gr.Number(label="Max Train Epochs", value=get_value("max_train_epochs"), precision=0, minimum=0,
                          info="0 = use max train steps instead."),
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
        with gr.Row():
            timestep_sampling = reg(
                "timestep_sampling",
                gr.Dropdown(
                    label="Timestep Sampling",
                    choices=["shifted_logit_normal", "sigma", "uniform", "sigmoid", "shift", "logsnr"],
                    value=get_value("timestep_sampling"),
                    info="shifted_logit_normal is the LTX-2 sequence-length-adaptive method (recommended).",
                ),
            )
            discrete_flow_shift = reg(
                "discrete_flow_shift",
                gr.Number(label="Discrete Flow Shift", value=get_value("discrete_flow_shift"), step=0.1,
                          info="Only used by 'shift' style sampling; shifted_logit_normal manages shift automatically."),
            )
            min_timestep = reg(
                "min_timestep",
                gr.Number(label="Min Timestep", value=get_value("min_timestep"), precision=0, minimum=0, maximum=1000),
            )
            max_timestep = reg(
                "max_timestep",
                gr.Number(label="Max Timestep", value=get_value("max_timestep"), precision=0, minimum=0, maximum=1000),
            )

    with tracked_panel(
        "Saving Settings",
        keywords="save output checkpoint resume state precision epochs steps retention",
        open=True,
    ):
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

    with tracked_panel(
        "Sample Generation Settings",
        keywords="sample generation prompts output width height frames steps guidance seed negative",
        open=False,
    ):
        gr.Markdown(
            "Samples are generated during training from a prompt file (one prompt per line). "
            "Prompt lines are auto-augmented with width/height/frames/steps/guidance/seed unless disabled."
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
        with gr.Row():
            sample_prompts = reg(
                "sample_prompts",
                gr.Textbox(label="Sample Prompts File", value=get_value("sample_prompts"),
                           placeholder="Path to a .txt file with one prompt per line"),
            )
            sample_prompts_button = gr.Button("📂", size="sm", elem_classes=["mbtn", "mbtn-forest"], visible=(not headless))
            sample_prompts_button.click(get_file_path, inputs=[sample_prompts], outputs=[sample_prompts], show_progress=False)
        with gr.Row():
            width = reg("width", gr.Number(label="Sample Width", value=get_value("width"), precision=0, minimum=64, step=32))
            height = reg("height", gr.Number(label="Sample Height", value=get_value("height"), precision=0, minimum=64, step=32))
            sample_num_frames = reg(
                "sample_num_frames",
                gr.Number(label="Sample Frames", value=get_value("sample_num_frames"), precision=0, minimum=1,
                          info="8k+1 values recommended (e.g. 49, 121)."),
            )
            sample_steps = reg(
                "sample_steps",
                gr.Number(label="Sample Steps", value=get_value("sample_steps"), precision=0, minimum=1),
            )
            sample_seed = reg("sample_seed", gr.Number(label="Sample Seed", value=get_value("sample_seed"), precision=0, minimum=0))
            sample_cfg_scale = reg(
                "sample_cfg_scale",
                gr.Number(label="Sample Guidance Scale", value=get_value("sample_cfg_scale"), step=0.1),
            )
        with gr.Row():
            sample_sampling_preset = reg(
                "sample_sampling_preset",
                gr.Dropdown(
                    label="Sampling Preset",
                    choices=["defaults", "legacy", "ltx20", "ltx23", "ltx23_hq", "distilled_two_stage"],
                    value=get_value("sample_sampling_preset"),
                    info="'defaults' resolves per LTX version (LTX-2.3: 30 steps, 768x512, 121 frames).",
                ),
            )
            sample_sampler = reg(
                "sample_sampler",
                gr.Dropdown(label="Sampler", choices=["auto", "euler", "res_2s"], value=get_value("sample_sampler")),
            )
            sample_sigma_schedule = reg(
                "sample_sigma_schedule",
                gr.Dropdown(label="Sigma Schedule", choices=["auto", "ltx", "ltx_latent", "ltx23_distilled"], value=get_value("sample_sigma_schedule")),
            )
            sample_negative_prompt = reg(
                "sample_negative_prompt",
                gr.Textbox(label="Negative Prompt", value=get_value("sample_negative_prompt")),
            )
        with gr.Row():
            sample_with_offloading = reg(
                "sample_with_offloading",
                gr.Checkbox(label="Offload during sampling", value=get_value("sample_with_offloading"),
                            info="Offload DiT to CPU between sampling prompts to save VRAM."),
            )
            sample_tiled_vae = reg(
                "sample_tiled_vae",
                gr.Checkbox(label="Tiled VAE decode", value=get_value("sample_tiled_vae"), info="Reduces sampling VRAM."),
            )
            sample_vae_tile_size = reg(
                "sample_vae_tile_size",
                gr.Number(label="VAE Tile Size", value=get_value("sample_vae_tile_size"), precision=0, minimum=0, step=32),
            )
            sample_vae_tile_overlap = reg(
                "sample_vae_tile_overlap",
                gr.Number(label="VAE Tile Overlap", value=get_value("sample_vae_tile_overlap"), precision=0, minimum=0, step=32),
            )
            sample_vae_temporal_tile_size = reg(
                "sample_vae_temporal_tile_size",
                gr.Number(label="VAE Temporal Tile", value=get_value("sample_vae_temporal_tile_size"), precision=0, minimum=0, step=8),
            )
            sample_vae_temporal_tile_overlap = reg(
                "sample_vae_temporal_tile_overlap",
                gr.Number(label="Temporal Overlap", value=get_value("sample_vae_temporal_tile_overlap"), precision=0, minimum=0, step=8),
            )
        with gr.Row():
            sample_merge_audio = reg(
                "sample_merge_audio",
                gr.Checkbox(label="Merge audio into sample mp4", value=get_value("sample_merge_audio")),
            )
            sample_disable_audio = reg(
                "sample_disable_audio",
                gr.Checkbox(label="Disable audio previews", value=get_value("sample_disable_audio")),
            )
            sample_audio_only = reg(
                "sample_audio_only",
                gr.Checkbox(label="Audio-only previews", value=get_value("sample_audio_only")),
            )
            sample_output_dir = reg(
                "sample_output_dir",
                gr.Textbox(label="Enhanced Prompts Output Dir", value=get_value("sample_output_dir"),
                           placeholder="Optional folder for the auto-enhanced prompt file"),
            )
            disable_prompt_enhancement = reg(
                "disable_prompt_enhancement",
                gr.Checkbox(label="Disable prompt enhancement", value=get_value("disable_prompt_enhancement"),
                            info="Use the prompt file exactly as written."),
            )

    with tracked_panel(
        "Logging / Metadata / HuggingFace",
        keywords="logging tensorboard wandb metadata author license tags huggingface repository upload",
        open=False,
    ):
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

    with tracked_panel(
        "Additional Parameters",
        keywords="additional cli parameters arguments debug timesteps",
        open=False,
    ):
        with gr.Row():
            additional_parameters = reg(
                "additional_parameters",
                gr.Textbox(
                    label="Additional CLI Parameters",
                    value=get_value("additional_parameters"),
                    placeholder='Appended verbatim to the training command, e.g. --audio_loss_weight 1.0 --caption_dropout_rate 0.1',
                    info="Escape hatch for any ltx2_train_network.py argument not exposed above.",
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

    # Order the registrations to match LTX2_PARAM_KEYS exactly.
    registration_map = dict(registrations)
    missing = [key for key in LTX2_PARAM_KEYS if key not in registration_map]
    extra = [key for key, _ in registrations if key not in set(LTX2_PARAM_KEYS)]
    assert not missing and not extra, f"LTX-2 GUI parameter mismatch. Missing: {missing} Extra: {extra}"
    settings_list = [registration_map[key] for key in LTX2_PARAM_KEYS]

    # Dataset config mode visibility toggle.
    dataset_config_mode.change(
        fn=lambda mode: gr.Group(visible=(mode == "Generate from Folder Structure")),
        inputs=[dataset_config_mode],
        outputs=[dataset_generation_group],
        show_progress=False,
    )

    # Round target frames to 8k+1 as the user edits.
    dataset_num_frames.blur(
        fn=lambda frames: round_frames_to_ltx2(frames),
        inputs=[dataset_num_frames],
        outputs=[dataset_num_frames],
        show_progress=False,
    )

    generate_toml_button.click(
        generate_ltx2_dataset_toml,
        inputs=settings_list,
        outputs=[dataset_config, generated_toml_path, dataset_status],
        show_progress=True,
    )

    with gr.Column(), gr.Group():
        with gr.Row():
            print_button = gr.Button("Print Command", variant="secondary", elem_classes=["mbtn", "mbtn-slate"])
            toggle_all_button_bottom = gr.Button(
                "Open All Panels",
                variant="secondary",
                elem_classes=["mbtn", "mbtn-purple"],
            )
        executor = CommandExecutor(headless=headless)

    run_state = gr.Textbox(value=str(train_state_value), visible=False)

    def search_and_open_panels(query):
        query = str(query or "").strip().lower()
        if not query:
            return [
                "closed",
                gr.Button(value="Open All Panels"),
                gr.Button(value="Open All Panels"),
                gr.HTML(value="", visible=False),
            ] + [gr.Accordion(open=False) for _ in accordions]

        matches = [query in searchable_text for _, searchable_text in panel_search_specs]
        matched_names = [
            label
            for (label, _), matched in zip(panel_search_specs, matches)
            if matched
        ]
        if matched_names:
            items = "".join(f"<li>{name}</li>" for name in matched_names)
            result = f"<strong>Matching panels</strong><ul>{items}</ul>"
            button_label = f"Reset Search ({len(matched_names)} panel{'s' if len(matched_names) != 1 else ''} filtered)"
        else:
            result = "<strong>No settings found.</strong>"
            button_label = "Reset Search"

        return [
            "search",
            gr.Button(value=button_label),
            gr.Button(value=button_label),
            gr.HTML(value=result, visible=True),
        ] + [gr.Accordion(open=matched) for matched in matches]

    search_input.change(
        fn=search_and_open_panels,
        inputs=[search_input],
        outputs=[panels_state, toggle_all_button, toggle_all_button_bottom, search_results] + accordions,
        show_progress=False,
    )

    def toggle_panels(current_state):
        # A toggle during filtered search acts as Reset Search, matching the
        # behavior of the established trainer tabs.
        open_panels = current_state not in {"open", "search"}
        new_state = "open" if open_panels else "closed"
        button_label = "Hide All Panels" if open_panels else "Open All Panels"
        return [
            new_state,
            gr.Button(value=button_label),
            gr.Button(value=button_label),
            gr.Textbox(value=""),
            gr.HTML(value="", visible=False),
        ] + [gr.Accordion(open=open_panels) for _ in accordions]

    for button in (toggle_all_button, toggle_all_button_bottom):
        button.click(
            fn=toggle_panels,
            inputs=[panels_state],
            outputs=[
                panels_state,
                toggle_all_button,
                toggle_all_button_bottom,
                search_input,
                search_results,
            ]
            + accordions,
            show_progress=False,
        )

    configuration.button_open_config.click(
        ltx2_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )
    configuration.button_load_config.click(
        ltx2_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )
    configuration.button_save_config.click(
        ltx2_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )
    print_button.click(
        ltx2_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )
    executor.button_run.click(
        ltx2_gui_actions,
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
