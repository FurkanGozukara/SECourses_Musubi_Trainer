"""Turn the recorded legacy trainer schema into ParamSpecs with the unified layout.

``legacy_schema.json`` (written by ``tools/generate_unified_schema.py``) records each trainer's
settings exactly as the original tabs define them. This module decides *where* each setting
appears in the unified UI: its section (most important sections first), its position inside the
section, whether it is an advanced setting, and which widget it uses when trainers disagree.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from .param_spec import ALL_MODES, FULL, LORA, SECTION_IDS, ParamSpec

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(APP_DIR)
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_schema.json")

# ------------------------------------------------------------------------------------------------
# Section layout. Within a section, keys are listed most important first; keys a section does not
# list are appended after the listed ones in the trainer's own order.
# ------------------------------------------------------------------------------------------------
SECTION_KEYS: dict[str, list[str]] = {
    "dataset": [
        "dataset_config_mode",
        "parent_folder_path",
        "dataset_config",
        "dataset_resolution_width",
        "dataset_resolution_height",
        "dataset_batch_size",
        "dataset_caption_extension",
        "create_missing_captions",
        "caption_strategy",
        "dataset_enable_bucket",
        "dataset_bucket_no_upscale",
        # video datasets
        "dataset_num_frames",
        "target_frames",
        "frame_extraction",
        "dataset_frame_extraction",
        "frame_stride",
        "dataset_frame_stride",
        "frame_sample",
        "dataset_frame_sample",
        "max_frames",
        "dataset_max_frames",
        "source_fps",
        "dataset_source_fps",
        "dataset_target_fps",
        "auto_normalize_target_frames",
        "allow_experimental_duration",
        "ltx2_audio_source",
        "separate_audio_buckets",
        # control / reference images
        "use_control_images",
        "control_directory_name",
        "dataset_control_directory",
        "no_resize_control",
        "control_resolution_width",
        "control_resolution_height",
        "dataset_qwen_image_edit_no_resize_control",
        "dataset_qwen_image_edit_control_resolution_width",
        "dataset_qwen_image_edit_control_resolution_height",
        "auto_generate_black_control_images",
        "dataset_cache_directory",
        "generated_toml_path",
    ],
    "output": [
        "output_dir",
        "output_name",
        "save_every_n_epochs",
        "save_every_n_steps",
        "save_precision",
        "mem_eff_save",
        "save_last_n_epochs",
        "save_last_n_steps",
        "resume",
        "save_state",
        "save_state_on_train_end",
        "save_last_n_epochs_state",
        "save_last_n_steps_state",
        "convert_to_diffusers",
        "diffusers_output_dir",
        "convert_to_safetensors",
        "safetensors_output_dir",
        "no_convert_to_comfy",
        "no_save_original_lora",
    ],
    "training": [
        "max_train_epochs",
        "max_train_steps",
        "learning_rate",
        "optimizer_type",
        "optimizer_args",
        "lr_scheduler",
        "lr_warmup_steps",
        "max_grad_norm",
        "gradient_accumulation_steps",
        "seed",
        "lr_decay_steps",
        "lr_scheduler_num_cycles",
        "lr_scheduler_power",
        "lr_scheduler_timescale",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_type",
        "lr_scheduler_args",
    ],
    "model": [
        "training_mode",
        "model_family",
        "model_version",
        "zimage_variant",
        "is_edit",
        "model_type",
        "task",
        "model_variant",
        "ltx2_mode",
        "ltx_version",
        "pretrained_model_name_or_path",
        "dit",
        "dit_high_noise",
        "timestep_boundary",
        "ltx2_checkpoint",
        "vae",
        "ae",
        "video_vae",
        "audio_vae",
        "vocoder",
        "text_encoder",
        "text_encoder_qwen",
        "text_encoder_clip",
        "t5",
        "clip",
        "clip_l",
        "t5xxl",
        "gemma_root",
        "gemma_safetensors",
        "ltx2_text_encoder_checkpoint",
        "ltx2_video_vae",
        "ltx2_audio_vae",
        "training_adapter_path",
        "training_adapter_multiplier",
        "turbo_dit",
        "turbo_dit_cache",
        "unconditional_dit",
        "use_unconditional_dit_for_lora_sampling",
        "dit_dtype",
        "vae_dtype",
        "text_encoder_dtype",
        "clip_vision_dtype",
        "one_frame",
        "force_v2_1_time_embedding",
        "dit_in_channels",
        "num_layers",
        "ltx_version_check_mode",
    ],
    "network": [
        "network_dim",
        "network_alpha",
        "network_dropout",
        "network_args",
        "lora_target_preset",
        "train_connectors",
        "network_weights",
        "dim_from_weights",
        "network_module",
        "custom_network_module",
        "scale_weight_norms",
        "base_weights",
        "base_weights_multiplier",
        "training_comment",
        "no_metadata",
    ],
    "finetune": [
        "dit_variant",
        "fused_backward_pass",
        "full_fp16",
        "block_swap_optimizer_patch_params",
        "freeze_text_embeddings",
        "ltx2_block_swap_trainable_ring",
        "save_merged_checkpoint",
        "ltx2_fft_merge_after_training",
    ],
    "memory": [
        "blocks_to_swap",
        "fp8_base",
        "fp8_scaled",
        "gradient_checkpointing",
        "full_bf16",
        "convrot_int8",
        "convrot_int8_bwd",
        "fp8_text_encoder",
        "fp8_vl",
        "fp8_llm",
        "fp8_t5",
        "gradient_checkpointing_cpu_offload",
        "use_pinned_memory_for_block_swap",
        "block_swap_h2d_only",
        "block_swap_ring_size",
        "offload_inactive_dit",
        "img_in_txt_in_offloading",
        "text_encoder_blocks_to_swap",
        "vae_cache_cpu",
        "vae_tiling",
        "vae_chunk_size",
        "vae_spatial_tile_sample_min_size",
        "sdpa",
        "flash_attn",
        "flash3",
        "sage_attn",
        "xformers",
        "split_attn",
        "split_attn_target",
        "split_attn_mode",
        "split_attn_chunk_size",
        "use_legacy_sdpa",
        "compile",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_cache_size_limit",
        "compile_resident_blocks_only",
        "cuda_allow_tf32",
        "cuda_cudnn_benchmark",
        "faster_model_loading",
        "disable_numpy_memmap",
        "disable_mmap",
        "fp8_keep_blocks",
        "fp8_w8a8",
        "w8a8_backend",
        "nf4_base",
        "int8_convrot_base",
        "int8_convrot_dynamic",
        "int8_convrot_groupsize",
        "int8_convrot_no_mse_clip",
        "int8_convrot_write_quality_report",
        "quantize_device",
        "blockwise_checkpointing",
        "blocks_to_checkpoint",
        "ltx2_low_ram_load",
        "cpu_staged_checkpoint_loading",
        "gemma_load_in_8bit",
        "gemma_load_in_4bit",
        "prune_adaln",
    ],
    "samples": [
        "sample_prompts",
        "sample_every_n_epochs",
        "sample_every_n_steps",
        "sample_at_first",
        "sample_width",
        "sample_height",
        "width",
        "height",
        "sample_num_frames",
        "sample_steps",
        "sample_cfg_scale",
        "sample_guidance_scale",
        "sample_discrete_flow_shift",
        "sample_seed",
        "sample_negative_prompt",
        "sample_sampling_preset",
        "sample_sampler",
        "sample_sigma_schedule",
        "sample_merge_audio",
        "sample_disable_audio",
        "sample_audio_only",
        "h3_allow_experimental_sample_duration",
        "sample_with_offloading",
        "sample_tiled_vae",
        "sample_vae_tile_size",
        "sample_vae_tile_overlap",
        "sample_vae_temporal_tile_size",
        "sample_vae_temporal_tile_overlap",
        "sample_output_dir",
        "disable_prompt_enhancement",
    ],
    "caching": [
        "cache_latents",
        "cache_text_encoder_outputs",
        "caching_latent_skip_existing",
        "caching_teo_skip_existing",
        "caching_latent_device",
        "caching_teo_device",
        "caching_latent_batch_size",
        "caching_teo_batch_size",
        "caching_teo_text_encoder_dtype",
        "caching_teo_text_cache_dtype",
        "caching_teo_fp8_text_encoder",
        "caching_teo_fp8_vl",
        "caching_teo_fp8_llm",
        "caching_teo_fp8_te",
        "caching_teo_quantized_qwen",
        "caching_latent_num_workers",
        "caching_teo_num_workers",
        "caching_latent_keep_cache",
        "caching_teo_keep_cache",
    ],
    "advanced": [
        "timestep_sampling",
        "discrete_flow_shift",
        "flow_shift",
        "weighting_scheme",
        "sigmoid_scale",
        "logit_mean",
        "logit_std",
        "mode_scale",
        "min_timestep",
        "max_timestep",
        "preserve_distribution_shape",
        "num_timestep_buckets",
        "guidance_scale",
        "noise_scale_start",
        "noise_scale_end",
        "noise_clip_std",
        "scheduler_scale",
        "independent_time",
        "video_only",
        "audio_loss_weight",
        "ltx2_first_frame_conditioning_p",
        "dino_loss_weight",
        "dino_loss_every_n_steps",
        "skip_t2i_visual_dummy",
        "show_timesteps",
    ],
    "hardware": [
        "gpu_ids",
        "mixed_precision",
        "max_data_loader_n_workers",
        "persistent_data_loader_workers",
        "multi_gpu",
        "num_processes",
        "num_machines",
        "main_process_port",
        "num_cpu_threads_per_process",
        "dynamo_backend",
        "dynamo_mode",
        "dynamo_use_fullgraph",
        "dynamo_use_dynamic",
        "extra_accelerate_launch_args",
        "ddp_timeout",
        "ddp_gradient_as_bucket_view",
        "ddp_static_graph",
    ],
    "logging": [
        "log_with",
        "logging_dir",
        "log_prefix",
        "log_tracker_name",
        "wandb_run_name",
        "wandb_api_key",
        "log_tracker_config",
        "log_config",
        "debug_mode",
        "additional_parameters",
    ],
    "metadata": [
        "metadata_title",
        "metadata_author",
        "metadata_description",
        "metadata_tags",
        "metadata_license",
        "metadata_reso",
        "metadata_arch",
        "huggingface_repo_id",
        "huggingface_token",
        "huggingface_repo_type",
        "huggingface_repo_visibility",
        "huggingface_path_in_repo",
        "save_state_to_huggingface",
        "resume_from_huggingface",
        "async_upload",
    ],
}

# Key-name prefixes that place unlisted keys.
PREFIX_SECTIONS = (
    ("caching_", "caching"),
    ("sample_", "samples"),
    ("dataset_", "dataset"),
    ("compile_", "memory"),
    ("int8_", "memory"),
    ("h3_", "model_specific"),
    ("metadata_", "metadata"),
    ("huggingface_", "metadata"),
    ("lr_", "training"),
    ("dynamo_", "hardware"),
    ("ddp_", "hardware"),
)

# Original panel title fragments -> section (fallback for keys the tables above do not place).
LEGACY_PANEL_SECTIONS = (
    ("full fine-tuning", "finetune"),
    ("lora settings", "network"),
    ("network settings", "network"),
    ("training dataset", "dataset"),
    ("dataset settings", "dataset"),
    ("audio / av", "dataset"),
    ("sample generation", "samples"),
    ("caching", "caching"),
    ("text encoder outputs caching", "caching"),
    ("learning rate", "training"),
    ("save models", "output"),
    ("huggingface", "metadata"),
    ("metadata", "metadata"),
    ("accelerate", "hardware"),
    ("quantization", "memory"),
    ("memory", "memory"),
    ("attention", "memory"),
    ("torch compile", "memory"),
    ("flow matching", "advanced"),
    ("audio supervision", "advanced"),
    ("model files", "model"),
    ("model settings", "model"),
    ("advanced settings", "logging"),
)

# Settings most people never touch: hidden until "Show advanced settings" is on.
ADVANCED_KEYS = frozenset(
    {
        "dataset_cache_directory",
        "generated_toml_path",
        "save_last_n_epochs_state",
        "save_last_n_steps_state",
        "save_state_on_train_end",
        "lr_decay_steps",
        "lr_scheduler_num_cycles",
        "lr_scheduler_power",
        "lr_scheduler_timescale",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_type",
        "lr_scheduler_args",
        "dit_in_channels",
        "num_layers",
        "force_v2_1_time_embedding",
        "ltx_version_check_mode",
        "clip_vision_dtype",
        "network_module",
        "custom_network_module",
        "dim_from_weights",
        "scale_weight_norms",
        "base_weights",
        "base_weights_multiplier",
        "training_comment",
        "no_metadata",
        "full_fp16",
        "block_swap_optimizer_patch_params",
        "block_swap_ring_size",
        "block_swap_h2d_only",
        "use_pinned_memory_for_block_swap",
        "img_in_txt_in_offloading",
        "vae_chunk_size",
        "vae_spatial_tile_sample_min_size",
        "split_attn_target",
        "split_attn_mode",
        "split_attn_chunk_size",
        "use_legacy_sdpa",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_cache_size_limit",
        "compile_resident_blocks_only",
        "cuda_allow_tf32",
        "cuda_cudnn_benchmark",
        "faster_model_loading",
        "disable_numpy_memmap",
        "disable_mmap",
        "int8_convrot_dynamic",
        "int8_convrot_groupsize",
        "int8_convrot_no_mse_clip",
        "int8_convrot_write_quality_report",
        "quantize_device",
        "w8a8_backend",
        "sample_output_dir",
        "disable_prompt_enhancement",
        "sample_with_offloading",
        "sample_vae_tile_size",
        "sample_vae_tile_overlap",
        "sample_vae_temporal_tile_size",
        "sample_vae_temporal_tile_overlap",
        "caching_latent_device",
        "caching_teo_device",
        "caching_latent_batch_size",
        "caching_teo_batch_size",
        "caching_latent_num_workers",
        "caching_teo_num_workers",
        "caching_latent_keep_cache",
        "caching_teo_keep_cache",
        "caching_latent_debug_mode",
        "caching_latent_console_width",
        "caching_latent_console_back",
        "caching_latent_console_num_images",
        "caching_latent_atomic_writes",
        "caching_teo_atomic_writes",
        "caching_latent_cache_seed",
        "caching_latent_vae_spatial_tile_size",
        "caching_latent_vae_spatial_tile_overlap",
        "caching_latent_vae_temporal_tile_size",
        "caching_latent_vae_temporal_tile_overlap",
        "caching_latent_vae_chunk_size",
        "cache_before_connector",
        "text_encoder_attn_mode",
        "nvfp4_scaled_mm",
        "audio_vae_scaling_factor",
        "max_length",
        "logit_mean",
        "logit_std",
        "mode_scale",
        "preserve_distribution_shape",
        "num_timestep_buckets",
        "show_timesteps",
        "num_machines",
        "main_process_port",
        "num_cpu_threads_per_process",
        "dynamo_backend",
        "dynamo_mode",
        "dynamo_use_fullgraph",
        "dynamo_use_dynamic",
        "extra_accelerate_launch_args",
        "ddp_timeout",
        "ddp_gradient_as_bucket_view",
        "ddp_static_graph",
        "log_prefix",
        "log_tracker_name",
        "log_tracker_config",
        "log_config",
        "metadata_reso",
        "metadata_arch",
        "huggingface_repo_type",
        "huggingface_repo_visibility",
        "huggingface_path_in_repo",
        "save_state_to_huggingface",
        "resume_from_huggingface",
        "async_upload",
    }
)

# Settings shown only for one training type even though the original tab kept them visible.
LORA_ONLY_KEYS = frozenset(SECTION_KEYS["network"])
FULL_ONLY_KEYS = frozenset(SECTION_KEYS["finetune"]) - {"mem_eff_save", "full_fp16"}
# Full fine-tuning trains the base weights themselves, so every trainer's full path turns these off
# (full_finetune_gui.normalize_image_training_parameters; Krea 2's full-DiT script has no Turbo DiT sampling):
# shown in Fine-Tuning they would look active while the run silently ignored them.
LORA_ONLY_EVERYWHERE = frozenset(
    {
        "fp8_base", "fp8_scaled", "convrot_int8", "convrot_int8_bwd", "block_swap_h2d_only", "turbo_dit", "turbo_dit_cache",
        "use_unconditional_dit_for_lora_sampling",
    }
)
# ...and the reverse: every LoRA path drops these (training_mode_runtime_exclusions), except where the backend
# also uses them for LoRA (LTX 2.3 writes its LoRA EMA / teacher weights with the memory-efficient saver).
FULL_ONLY_EXCEPT = {"mem_eff_save": frozenset({"ltx2"})}

# Fine-tuning derives these from Full BF16 (full_finetune_gui.normalize_image_training_parameters sets the
# trained dtype bfloat16 / float32 and the save precision bf16 / fp32): only LoRA lets the user choose them.
DERIVED_IN_FULL = frozenset({"dit_dtype", "save_precision"})

# The Training Type selector at the top of the workspace drives these; they are never shown.
HIDDEN_KEYS = frozenset({"training_mode", "generated_toml_path"})

# Fields with a browse button. Values: "file" | "folder".
PATH_KINDS = {
    "parent_folder_path": "folder",
    "dataset_config": "file",
    "output_dir": "folder",
    "resume": "folder",
    "logging_dir": "folder",
    "sample_prompts": "file",
    "sample_output_dir": "folder",
    "network_weights": "file",
    "base_weights": "file",
    "dit": "file",
    "dit_high_noise": "file",
    "vae": "file",
    "video_vae": "file",
    "audio_vae": "file",
    "vocoder": "folder",
    "text_encoder": "file",
    "text_encoder_qwen": "folder",
    "text_encoder_clip": "folder",
    "t5": "file",
    "clip": "file",
    "ltx2_checkpoint": "file",
    "gemma_root": "folder",
    "gemma_safetensors": "file",
    "ltx2_text_encoder_checkpoint": "file",
    "ltx2_video_vae": "file",
    "ltx2_audio_vae": "file",
    "training_adapter_path": "file",
    "turbo_dit": "file",
    "unconditional_dit": "file",
    "diffusers_output_dir": "folder",
    "safetensors_output_dir": "folder",
    "h3_guidance_loss_uncond_cache": "file",
}
# Folder names / paths typed without a browse button: cleaned like paths (Explorer's "Copy as path"
# quotes, separators) before the trainer or its dataset generator sees them.
PATH_TEXT_KEYS = frozenset({"dataset_cache_directory", "dataset_control_directory"})

# Counts the classic code reads with int() although the trainer never declares them (accelerate launch
# arguments, dataset generation, caching, samples): a fraction there is a typo, never a setting. Not
# fps or sigma values, which are real fractions.
WHOLE_NUMBER_KEYS = frozenset(
    {
        "main_process_port", "num_processes", "num_machines", "num_cpu_threads_per_process",
        "dataset_batch_size", "dataset_resolution_width", "dataset_resolution_height",
        "control_resolution_width", "control_resolution_height",
        "dataset_qwen_image_edit_control_resolution_width", "dataset_qwen_image_edit_control_resolution_height",
        "caching_latent_batch_size", "caching_latent_num_workers", "caching_teo_batch_size", "caching_teo_num_workers",
        "caching_latent_console_num_images", "caching_latent_console_width",
        "sample_width", "sample_height", "sample_steps", "sample_seed", "sample_num_frames",
        "target_frames", "max_frames", "frame_stride", "frame_sample", "num_frames",
        "vae_chunk_size", "vae_spatial_tile_sample_min_size", "dit_in_channels",
    }
)
TEXTAREA_KEYS = frozenset({"network_args", "additional_parameters", "sample_negative_prompt", "metadata_description"})

# Label polish where the original label is a raw argument name.
# Help texts where a classic tab's own text is wrong in the unified UI (one text serves every trainer).
INFO_OVERRIDES = {
    "create_missing_captions": "When Start builds the dataset, write a caption for every image or video that has none (Missing Caption Strategy decides the text).",
    # the classic tabs said "no commas": the unified UI reads every way of typing these lists (unified/arguments.py)
    "optimizer_args": (
        "key=value items separated by spaces, commas or new lines, e.g. weight_decay=0.01 betas=(0.9,0.999). "
        "Values are Python literals; true / false work too."
    ),
    "lr_scheduler_args": "key=value items separated by spaces, commas or new lines, e.g. T_max=100 eta_min=1e-6.",
    "network_args": "key=value items separated by spaces, commas or new lines, e.g. conv_dim=4 conv_alpha=1.",
}

LABEL_OVERRIDES = {
    "caching_latent_device": "Latent Cache Device",
    "caching_teo_device": "Text Cache Device",
    "caching_latent_batch_size": "Latent Cache Batch Size",
    "caching_latent_num_workers": "Latent Cache Workers",
    "caching_teo_batch_size": "Text Cache Batch Size",
    "caching_teo_num_workers": "Text Cache Workers",
    "huggingface_repo_id": "Hugging Face Repo ID",
    "huggingface_token": "Hugging Face Token",
    "huggingface_repo_type": "Hugging Face Repo Type",
    "huggingface_repo_visibility": "Hugging Face Repo Visibility",
    "huggingface_path_in_repo": "Hugging Face Path in Repo",
    "metadata_author": "Metadata Author",
    "metadata_description": "Metadata Description",
    "metadata_license": "Metadata License",
    "metadata_tags": "Metadata Tags",
    "metadata_title": "Metadata Title",
    # trainers whose components carry no label of their own
    "model_version": "🧩 Model Version",
    "zimage_variant": "🧩 Z-Image Variant",
    "no_resize_control": "Keep Control Image Size (No Resize)",
    "control_resolution_width": "Control Image Width",
    "control_resolution_height": "Control Image Height",
    "fp8_text_encoder": "FP8 Text Encoder",
    "caching_teo_fp8_text_encoder": "FP8 Text Encoder While Caching",
    "fp8_llm": "FP8 LLM Text Encoder",
    "caching_teo_fp8_llm": "FP8 LLM While Caching",
    "training_adapter_path": "Training Adapter Path",
    "training_adapter_multiplier": "Training Adapter Strength",
    "xformers": "xFormers Attention",
}
LABEL_WORDS = {
    "vae": "VAE", "fp8": "FP8", "fp16": "FP16", "bf16": "BF16", "int8": "INT8", "lr": "LR", "cfg": "CFG", "llm": "LLM",
    "dit": "DiT", "teo": "Text", "img": "Image", "txt": "Text", "id": "ID", "gpu": "GPU", "cpu": "CPU", "ema": "EMA",
    "sdpa": "SDPA", "toml": "TOML", "hf": "HF", "lora": "LoRA", "clip": "CLIP", "t5": "T5", "ip": "IP",
}


def pretty_label(key: str, legacy_label) -> str:
    """The trainer's own label, or readable words when it only has a raw key such as ``fp8_llm``."""
    label = str(legacy_label or "").strip()
    if key in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[key]
    if label and label != key and not ("_" in label and " " not in label):
        return label
    return " ".join(LABEL_WORDS.get(word, word.capitalize()) for word in key.split("_") if word)

# Device fields: a free-text box in some trainers, a dropdown in others.
DEVICE_CHOICES = ("cuda", "cuda:0", "cuda:1", "cpu")

FAMILY_OF_CLASS = {
    "Checkbox": "bool",
    "Number": "number",
    "Slider": "number",
    "Textbox": "str",
    "Dropdown": "str",
    "Radio": "str",
}


def _family(entry: dict) -> str:
    if entry.get("cls") == "Dropdown" and entry.get("multiselect"):
        return "list"
    if entry.get("cls") == "CheckboxGroup":
        return "list"
    return FAMILY_OF_CLASS.get(entry.get("cls"), "str")


def resolve_placeholders(value):
    if isinstance(value, str):
        if value.startswith("{APP}"):
            return os.path.normpath(APP_DIR + value[len("{APP}") :])
        if value.startswith("{ROOT}"):
            return os.path.normpath(ROOT_DIR + value[len("{ROOT}") :])
        return value
    if isinstance(value, list):
        return [resolve_placeholders(item) for item in value]
    return value


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def legacy_keys(backend: str) -> list[str]:
    return list(load_schema()["backends"][backend]["keys"])


@lru_cache(maxsize=None)
def _arg_types(backend: str) -> dict:
    return dict(load_schema().get("arg_types", {}).get(backend, {}))


# Where the trainer reads the runtime TOML differently from its command line: Wan's timestep_boundary is
# type=int on the command line, but a TOML float up to 1.0 is a fraction of the timesteps (the classic field
# is 0.0-1.0 in 0.01 steps).
ARG_TYPE_OVERRIDES = {("wan", "timestep_boundary"): {"type": "float", "default": None}}
# 0 there would change training (Max Timestep 0: no noise at all); the classic tabs read it as "not set"
ZERO_MEANS_UNSET = frozenset({"max_timestep"})


def arg_type(backend: str, key: str) -> dict | None:
    """The training script's argparse description of ``key`` (type, nargs, default), or None for
    settings only the GUI uses. Recorded by tools/trainer_arg_types.py."""
    return ARG_TYPE_OVERRIDES.get((backend, key)) or _arg_types(backend).get(key)


def arg_alias_dest(backend: str, key: str) -> str | None:
    """The argparse dest that the option named ``key`` sets on the command line (``ltx2_mode`` -> ``ltx_mode``), or
    None when ``key`` is a dest itself or no option. The trainers fill only dests from the runtime TOML, so a key with
    an alias name is ignored there (Guardrail 20). Recorded by tools/generate_unified_schema.py."""
    return load_schema().get("arg_aliases", {}).get(backend, {}).get(key)


def legacy_entry(backend: str, key: str) -> dict:
    return load_schema()["backends"][backend]["params"][key]


def section_for(key: str, entry: dict) -> str:
    for section, keys in SECTION_KEYS.items():
        if key in keys:
            return section
    for prefix, section in PREFIX_SECTIONS:
        if key.startswith(prefix):
            return section
    parents = " / ".join(entry.get("parents") or []).lower()
    for fragment, section in LEGACY_PANEL_SECTIONS:
        if fragment in parents:
            return section
    return "model_specific"


def priority_for(key: str, section: str, legacy_index: int) -> int:
    ordered = SECTION_KEYS.get(section, [])
    if key in ordered:
        return ordered.index(key) * 10
    return 1000 + legacy_index


def modes_for(key: str, entry: dict) -> frozenset:
    parents = " / ".join(entry.get("parents") or []).lower()
    if key in LORA_ONLY_KEYS or key in LORA_ONLY_EVERYWHERE or "lora settings" in parents:
        return frozenset({LORA})
    if key in FULL_ONLY_KEYS or "full fine-tuning" in parents:
        return frozenset({FULL})
    return ALL_MODES


def _normalize_choices(choices) -> tuple | None:
    if not isinstance(choices, list):
        return None
    normalized = []
    for choice in choices:
        if isinstance(choice, list) and len(choice) == 2:
            label, value = choice
            normalized.append((str(label), value) if label != value else value)
        else:
            normalized.append(choice)
    return tuple(normalized)


def _kind_for(key: str, entry: dict, family: str) -> str:
    cls = entry.get("cls")
    if family == "bool":
        return "checkbox"
    if family == "list":
        return "dropdown"
    if family == "number":
        return "slider" if cls == "Slider" or key == "blocks_to_swap" else "number"
    if cls == "Radio":
        return "radio"
    if cls == "Dropdown" or key in ("caching_latent_device", "caching_teo_device"):
        return "dropdown"
    path_kind = PATH_KINDS.get(key) or entry.get("browse")
    if path_kind:
        return path_kind
    if key in TEXTAREA_KEYS or (entry.get("lines") or 1) > 1:
        return "textarea"
    return "text"


def build_param_specs(backend: str) -> list[ParamSpec]:
    """Ordered ParamSpecs for one legacy trainer (order = the trainer's own positional order)."""
    backend_schema = load_schema()["backends"][backend]
    mode_choices = (backend_schema["params"].get("training_mode") or {}).get("choices") or []
    supports_full = any("fine" in str(choice).lower() for choice in mode_choices)
    specs = []
    for index, key in enumerate(backend_schema["keys"]):
        entry = backend_schema["params"][key]
        family = _family(entry)
        kind = _kind_for(key, entry, family)
        default = resolve_placeholders(entry.get("value"))
        if kind == "text" and (arg_type(backend, key) or {}).get("type") == "bool":
            # an on/off trainer flag that a classic tab shows as a text box (resume_from_huggingface)
            from .values import parse_bool

            kind, family = "checkbox", "bool"
            try:
                default = parse_bool(default)
            except ValueError:
                default = False
        section = section_for(key, entry)
        choices = _normalize_choices(entry.get("choices"))
        allow_custom = bool(entry.get("allow_custom_value"))
        if key in ("caching_latent_device", "caching_teo_device"):
            choices = choices or DEVICE_CHOICES
            allow_custom = True
        if kind == "dropdown" and entry.get("cls") == "Textbox":
            allow_custom = True
        label = pretty_label(key, entry.get("label"))
        visible = bool(entry.get("visible", True)) and key not in HIDDEN_KEYS
        minimum, maximum, step = entry.get("minimum"), entry.get("maximum"), entry.get("step")
        precision = entry.get("precision")
        extras = {"family": family, "legacy_cls": entry.get("cls"), "precision": precision, "interactive": entry.get("interactive")}
        modes = frozenset({LORA}) if supports_full and key in DERIVED_IN_FULL else modes_for(key, entry)
        if supports_full and key in FULL_ONLY_EXCEPT and backend not in FULL_ONLY_EXCEPT[key]:
            modes = frozenset({FULL})
        specs.append(
            ParamSpec(
                key=key,
                kind=kind,
                label=str(label),
                default=default,
                info=INFO_OVERRIDES.get(key) or str(entry.get("info") or ""),
                choices=choices,
                minimum=minimum,
                maximum=maximum,
                step=step,
                section=section if section in SECTION_IDS else "model_specific",
                priority=priority_for(key, section, index),
                modes=modes,
                visible=visible,
                advanced=key in ADVANCED_KEYS,
                allow_custom_value=allow_custom,
                multiselect=bool(entry.get("multiselect")),
                lines=int(entry.get("lines") or 1),
                placeholder=str(entry.get("placeholder") or ""),
                extras=extras,
            )
        )
    return specs


def component_uid(key: str, family: str, conflicting: bool) -> str:
    """Shared component id: the key, or key#family when trainers disagree on the value type."""
    return f"{key}#{family}" if conflicting else key
