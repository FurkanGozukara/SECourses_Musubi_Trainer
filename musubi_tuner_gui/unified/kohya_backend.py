"""kohya sd-scripts backend of the unified trainer UI: FLUX.1 Dev and SDXL, LoRA and full fine-tuning.

The settings use the same keys as the musubi trainers wherever the meaning is the same (``output_dir``,
``learning_rate``, ``optimizer_type``, ``blocks_to_swap``, ``dataset_resolution_width`` ...). Settings that only exist in
kohya keep their kohya / sd-scripts names (``pretrained_model_name_or_path``, ``ae``, ``clip_l``, ``t5xxl``,
``guidance_scale``, ``reg_data_dir``, ``unet_lr`` ...). The kohya_ss GUI JSON presets used model-prefixed duplicates
(``flux1_clip_l``, ``sdxl_no_half_vae``, ``sdxl_blocks_to_swap``, ``flux_fused_backward_pass`` ...); they are folded into
one key per meaning by :func:`convert_kohya_gui_json`.

Training runs ``accelerate launch <APP>/sd-scripts/<script>.py --config_file <runtime TOML>`` exactly like the
kohya_ss GUI, with the runtime TOML written into ``output_dir`` through ``common_gui.SaveConfigFileToRun`` (Guardrail 1
logging normalisation) and ``training_mode`` persisted in it (Guardrail 2).

Relative paths in presets are resolved against the install root (the folder that holds ``SECourses_Musubi_Trainer``);
``../``-prefixed paths are resolved against the app folder, like the musubi presets' ``output_dir = "../Training_Outputs_X"``.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import toml

from ..class_accelerate_launch import AccelerateLaunch
from ..common_gui import (
    SaveConfigFileToRun,
    _training_model_directories,
    requires_native_compile_toolchain,
    scriptdir,
    setup_environment,
)
from ..custom_logging import setup_logging
from ..full_finetune_gui import (
    FULL_FINE_TUNING_MODE,
    LORA_TRAINING_MODE,
    TRAINING_MODE_CHOICES,
    infer_training_mode_from_loaded_config,
    is_full_fine_tuning,
    normalize_training_mode,
)
from .param_spec import ALL_MODES, FULL, LORA, SECTION_IDS, ParamSpec
from .schema import arg_type
from .values import clean_path, parse_number, strip_quotes

log = setup_logging()

__all__ = [
    "KOHYA_MODELS",
    "KOHYA_HINTS",
    "kohya_param_specs",
    "kohya_defaults",
    "convert_kohya_gui_json",
    "relocate_model_paths",
    "load_kohya_preset",
    "kohya_validation_problems",
    "kohya_whole_number",
    "kohya_steps_example",
    "kohya_steps_note",
    "train_kohya_model",
    "build_kohya_preset_toml",
    "kohya_preset_file_name",
]

APP_DIR = Path(scriptdir).resolve()
ROOT_DIR = APP_DIR.parent
SD_SCRIPTS_DIR = APP_DIR / "sd-scripts"

FOLDER_DATASET_MODE = "Generate from Folder Structure"
TOML_DATASET_MODE = "Use TOML File"

LORA_ONLY = frozenset({LORA})
FULL_ONLY = frozenset({FULL})


def _fs(path: os.PathLike | str) -> str:
    return str(path).replace("\\", "/")


KOHYA_MODELS: dict[str, dict[str, Any]] = {
    "flux1_dev": {
        "display_name": "FLUX.1 Dev (kohya sd-scripts)",
        "short_name": "FLUX.1 Dev",
        "train_scripts": {LORA: "flux_train_network.py", FULL: "flux_train.py"},
        "model_dir": _fs(ROOT_DIR / "Training_Models_FLUX_1"),
        "model_dir_name": "Training_Models_FLUX_1",
        "legacy_model_dir_names": ("FLUX_Training_Models",),
        # Download_Train_Models.py of the Kohya FLUX app (repo MonsterMMORPG/Kohya_Train)
        "model_files": {
            "pretrained_model_name_or_path": "flux1-dev.safetensors",
            "ae": "ae.safetensors",
            "clip_l": "clip_l.safetensors",
            "t5xxl": "t5xxl_fp16.safetensors",
        },
        "alternative_model_files": {
            "pretrained_model_name_or_path": (
                "flux1-dev-fp8.safetensors",
                "flux1-krea-dev.safetensors",
                "FLUX-SRPO-bf16.safetensors",
            ),
        },
        "download_repo": "MonsterMMORPG/Kohya_Train",
        "output_dir_name": "Training_Outputs_FLUX_1",
        "preset_folder": "FLUX_1_Dev_Training_Configs",
        "network_module": "networks.lora_flux",
        "max_blocks_to_swap": 35,
        "training_types": (LORA, FULL),
    },
    "sdxl": {
        "display_name": "SDXL (kohya sd-scripts)",
        "short_name": "SDXL",
        "train_scripts": {LORA: "sdxl_train_network.py", FULL: "sdxl_train.py"},
        "model_dir": _fs(ROOT_DIR / "Training_Models_SDXL"),
        "model_dir_name": "Training_Models_SDXL",
        "legacy_model_dir_names": ("SDXL_Training_Models",),
        # Download_Train_Models.py of the Kohya FLUX app (repo MonsterMMORPG/Wan_GGUF); the shipped presets use RealVis v5
        "model_files": {"pretrained_model_name_or_path": "SDXL_RealVis_v5_FP32.safetensors"},
        "alternative_model_files": {"pretrained_model_name_or_path": ("SDXL_CyberRealistic_v8_FP32.safetensors",)},
        "default_vae": "stabilityai/sdxl-vae",
        "download_repo": "MonsterMMORPG/Wan_GGUF",
        "output_dir_name": "Training_Outputs_SDXL",
        "preset_folder": "SDXL_Training_Configs",
        "network_module": "networks.lora",
        "max_blocks_to_swap": 68,
        "training_types": (LORA, FULL),
    },
}

MODEL_PATH_KEYS = ("pretrained_model_name_or_path", "ae", "clip_l", "t5xxl", "vae", "network_weights")
_KOHYA_MODEL_DIR_NAMES = (
    "Training_Models_FLUX_1",
    "Training_Models_SDXL",
    "FLUX_Training_Models",
    "SDXL_Training_Models",
)
_MODEL_FILE_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")

# ----------------------------------------------------------------------------------------------------------------------
# choices
# ----------------------------------------------------------------------------------------------------------------------

OPTIMIZER_CHOICES = (
    "Adafactor",
    "AdamW",
    "AdamW8bit",
    "PagedAdamW8bit",
    "PagedAdamW32bit",
    "AdamWScheduleFree",
    "RAdamScheduleFree",
    "SGDScheduleFree",
    "Lion",
    "Lion8bit",
    "PagedLion8bit",
    "Prodigy",
    "prodigyplus.ProdigyPlusScheduleFree",
    "pytorch_optimizer.CAME",
    "bitsandbytes.optim.AdEMAMix8bit",
    "bitsandbytes.optim.PagedAdEMAMix8bit",
    "DAdaptAdam",
    "DAdaptLion",
    "SGDNesterov",
    "SGDNesterov8bit",
)
LR_SCHEDULER_CHOICES = (
    "constant",
    "constant_with_warmup",
    "cosine",
    "cosine_with_restarts",
    "cosine_with_min_lr",
    "linear",
    "polynomial",
    "piecewise_constant",
    "inverse_sqrt",
    "warmup_stable_decay",
    "adafactor",
)
SAMPLER_CHOICES = (
    "ddim",
    "pndm",
    "lms",
    "euler",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "dpmsolver++",  # plain "dpmsolver" fails in sd-scripts with diffusers >= 0.26 (final_sigmas_type)
    "dpmsingle",
    "k_lms",
    "k_euler",
    "k_euler_a",
    "k_dpm_2",
    "k_dpm_2_a",
)
DYNAMO_BACKEND_CHOICES = (
    "no",
    "eager",
    "aot_eager",
    "inductor",
    "aot_ts_nvfuser",
    "nvprims_nvfuser",
    "cudagraphs",
    "ofi",
    "fx2trt",
    "onnxrt",
    "tensorrt",
    "ipex",
)

# ----------------------------------------------------------------------------------------------------------------------
# defaults (from the best measured 24 GB preset of each model / training type)
# ----------------------------------------------------------------------------------------------------------------------

_ADAFACTOR_ARGS = "scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01"


def _model(model_key: str) -> dict[str, Any]:
    try:
        return KOHYA_MODELS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown kohya model '{model_key}'. Choose one of: {', '.join(KOHYA_MODELS)}.") from exc


def _training_type(value: object) -> str:
    """'lora' / 'full' from a training type ('lora', 'full') or a training_mode label."""
    text = str(value or "").strip().casefold()
    if text in (LORA, "lora_training"):
        return LORA
    if text in (FULL, "fine_tuning", "full_fine_tuning", "dreambooth"):
        return FULL
    try:
        return FULL if is_full_fine_tuning(value) else LORA
    except ValueError as exc:
        raise ValueError(f"Unknown training type '{value}'. Use 'lora' or 'full'.") from exc


def _default_model_paths(model_key: str) -> dict[str, str]:
    model = _model(model_key)
    paths = {key: _fs(Path(model["model_dir"]) / name) for key, name in model["model_files"].items()}
    if "default_vae" in model:
        paths["vae"] = model["default_vae"]
    return paths


def _common_defaults(model_key: str) -> dict[str, Any]:
    model = _model(model_key)
    return {
        # dataset
        "dataset_config_mode": FOLDER_DATASET_MODE,
        "parent_folder_path": "",
        "reg_data_dir": "",
        "dataset_config": "",
        "dataset_resolution_width": 1024,
        "dataset_resolution_height": 1024,
        "dataset_batch_size": 1,
        "dataset_caption_extension": ".txt",
        "dataset_enable_bucket": False,
        "dataset_bucket_no_upscale": True,
        "prior_loss_weight": 1.0,
        "min_bucket_reso": 256,
        "max_bucket_reso": 2048,
        "bucket_reso_steps": 64,
        "shuffle_caption": False,
        "keep_tokens": 0,
        "caption_dropout_rate": 0.0,
        "flip_aug": False,
        "random_crop": False,
        # output
        "output_dir": _fs(ROOT_DIR / model["output_dir_name"]),
        "save_every_n_epochs": 25,
        "save_every_n_steps": 0,
        "save_last_n_epochs": 0,
        "save_last_n_steps": 0,
        "save_model_as": "safetensors",
        "save_state": False,
        "save_state_on_train_end": False,
        "save_last_n_epochs_state": 0,
        "save_last_n_steps_state": 0,
        "resume": "",
        # training
        "max_train_epochs": 200,
        "max_train_steps": 0,
        "optimizer_type": "Adafactor",
        "optimizer_args": _ADAFACTOR_ARGS,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "lr_scheduler_num_cycles": 1,
        "lr_scheduler_power": 1.0,
        "lr_scheduler_type": "",
        "lr_scheduler_args": "",
        "gradient_accumulation_steps": 1,
        "seed": 1,
        # model / precision
        **_default_model_paths(model_key),
        "mixed_precision": "bf16",
        "full_bf16": True,
        "full_fp16": False,
        # LoRA
        "network_module": model["network_module"],
        "network_dropout": 0.0,
        "network_args": "",
        "network_weights": "",
        "dim_from_weights": False,
        "scale_weight_norms": 0.0,
        "unet_lr": 0.0,
        # memory
        "blocks_to_swap": 0,
        "gradient_checkpointing": True,
        "highvram": False,
        "disable_mmap_load_safetensors": False,
        "compile": False,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_dynamic": "auto",
        "compile_fullgraph": False,
        "compile_cache_size_limit": 0,
        # samples
        "sample_prompts": "",
        "sample_every_n_epochs": 0,
        "sample_every_n_steps": 0,
        "sample_at_first": False,
        # caching
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "vae_batch_size": 4,
        "skip_cache_check": False,
        # advanced
        "loss_type": "l2",
        "huber_c": 0.1,
        "huber_schedule": "snr",
        "huber_scale": 1.0,
        "ip_noise_gamma": 0.0,
        # hardware
        "gpu_ids": "0",
        "multi_gpu": False,
        "num_processes": 1,
        "num_machines": 1,
        "main_process_port": 0,
        "num_cpu_threads_per_process": 2,
        "max_data_loader_n_workers": 0,
        "persistent_data_loader_workers": False,
        "dynamo_backend": "no",
        "dynamo_mode": "default",
        "dynamo_use_fullgraph": False,
        "dynamo_use_dynamic": False,
        "extra_accelerate_launch_args": "",
        "ddp_timeout": 0,
        "ddp_gradient_as_bucket_view": False,
        "ddp_static_graph": False,
        # logging
        "log_with": "",
        "logging_dir": "",
        "log_prefix": "",
        "log_tracker_name": "",
        "log_tracker_config": "",
        "wandb_run_name": "",
        "wandb_api_key": "",
        "log_config": False,
        "additional_parameters": "",
        # metadata
        "training_comment": "",
        "metadata_title": "",
        "metadata_author": "",
        "metadata_description": "",
        "metadata_license": "",
        "metadata_tags": "",
        "huggingface_repo_id": "",
        "huggingface_token": "",
        "huggingface_repo_type": "",
        "huggingface_repo_visibility": "",
        "huggingface_path_in_repo": "",
        "save_state_to_huggingface": False,
        "resume_from_huggingface": "",
        "async_upload": False,
    }


def _flux_defaults(training_type: str) -> dict[str, Any]:
    values = {
        # FLUX_LoRA_Training_LoRA_Tab_Configs/24GB_GPUs_21.61GB_Quality_1_LoRA.json
        "output_name": "My_FLUX_1_Dev_LoRA",
        "learning_rate": 5e-05,
        "text_encoder_lr": 5e-05,
        "t5xxl_lr": 0.0,
        "max_grad_norm": 1.0,  # the LoRA tab drops 0 values, so sd-scripts' default 1.0 was what ran
        "save_precision": "fp32",
        "network_dim": 128,
        "network_alpha": 128.0,
        "train_blocks": "all",
        "train_double_block_indices": "all",
        "train_single_block_indices": "all",
        "blocks_to_swap": 13,
        "cpu_offload_checkpointing": False,
        "fp8_base": False,
        "fp8_scaled": False,
        "fp8_fast": False,
        "fp8_base_unet": False,
        "highvram": True,
        "disable_numpy_memmap": False,
        "compile": True,
        "cache_text_encoder_outputs_on_cpu": False,
        "timestep_sampling": "sigmoid",
        "discrete_flow_shift": 3.1582,
        "sigmoid_scale": 1.0,
        "guidance_scale": 1.0,
        "model_prediction_type": "raw",
        "apply_t5_attn_mask": True,
        "t5xxl_max_token_length": 512,
        # full fine-tuning keys (values of the DreamBooth 24 GB preset)
        "fused_backward_pass": True,
        "blockwise_fused_optimizers": False,
        "mem_eff_save": True,
    }
    if training_type == FULL:
        # FLUX_DreamBooth_Tab_Fine_Tuning_Training_Configs/24GB_GPUs_22.14GB_VRAM_Quality_1.json
        values.update(
            {
                "output_name": "My_FLUX_1_Dev_Fine_Tuning",
                "learning_rate": 4e-06,
                "max_grad_norm": 1.0,
                "save_precision": "bf16",
                "blocks_to_swap": 10,
                "highvram": False,
                "apply_t5_attn_mask": False,
            }
        )
    return values


def _sdxl_defaults(training_type: str) -> dict[str, Any]:
    values = {
        # No SDXL LoRA presets exist in the kohya app: these follow the SDXL fine-tuning presets (Adafactor, constant LR,
        # SDXL noise offset, fp32 SDXL VAE) with LoRA learning rates; U-Net only so text-encoder outputs can be cached.
        "output_name": "My_SDXL_LoRA",
        "learning_rate": 1e-04,
        "text_encoder_lr": 0.0,
        "max_grad_norm": 1.0,
        "save_precision": "fp16",
        "network_dim": 64,
        "network_alpha": 32.0,
        "max_train_epochs": 100,
        "save_every_n_epochs": 20,
        "fp8_base": False,
        "sdpa": True,
        "xformers": False,
        "no_half_vae": True,
        "max_token_length": 75,
        "vae_batch_size": 2,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "noise_offset": 0.0357,
        "ip_noise_gamma": 0.1,
        "min_snr_gamma": 0.0,
        "debiased_estimation_loss": False,
        "min_timestep": 0,
        "max_timestep": 1000,
        "sample_sampler": "euler_a",
        # full fine-tuning keys (values of the DreamBooth 24 GB preset)
        "learning_rate_te1": 3e-06,
        "learning_rate_te2": 0.0,
        "fused_backward_pass": False,
    }
    if training_type == FULL:
        # SDXL_DreamBooth_Fine_Tuning_Tab_Configs/24GB_GPUs_17.30GB_Quality_1.json
        values.update(
            {
                "output_name": "My_SDXL_Fine_Tuning",
                "learning_rate": 8e-06,
                "max_grad_norm": 0.0,  # --max_grad_norm=0.0 in the preset's additional parameters
                "save_precision": "bf16",
                "max_train_epochs": 1,  # the preset trains 1 epoch with folder repeats + regularization images
                "save_every_n_epochs": 1,
                "save_every_n_steps": 451,
                "seed": 0,
                "sdpa": False,  # the preset ran without xformers / SDPA ("none")
                "cache_text_encoder_outputs": False,  # text encoder 1 is trained
                "cache_text_encoder_outputs_to_disk": False,
                "dataset_caption_extension": ".caption",  # "" in the preset = sd-scripts default .caption
                "num_cpu_threads_per_process": 4,
            }
        )
    return values


def kohya_defaults(model_key: str, training_type: str) -> dict[str, Any]:
    """A value for every key of :func:`kohya_param_specs` (``training_type`` is ``"lora"`` or ``"full"``)."""
    _model(model_key)
    training_type = _training_type(training_type)
    values = _common_defaults(model_key)
    values.update(_flux_defaults(training_type) if model_key == "flux1_dev" else _sdxl_defaults(training_type))
    values["training_mode"] = FULL_FINE_TUNING_MODE if training_type == FULL else LORA_TRAINING_MODE
    keys = [spec.key for spec in _build_specs(model_key)]
    missing = [key for key in keys if key not in values]
    if missing:  # programming error: every key needs a default
        raise RuntimeError(f"kohya defaults missing for: {', '.join(missing)}")
    return {key: values[key] for key in keys}


# ----------------------------------------------------------------------------------------------------------------------
# parameter specs
# ----------------------------------------------------------------------------------------------------------------------


def _build_specs(model_key: str) -> list[ParamSpec]:
    model = _model(model_key)
    flux = model_key == "flux1_dev"
    sdxl = model_key == "sdxl"
    specs: list[ParamSpec] = []

    def add(key, kind, label, info="", section="model_specific", priority=500, modes=ALL_MODES, **kwargs):
        specs.append(ParamSpec(key=key, kind=kind, label=label, info=info, section=section, priority=priority, modes=modes, **kwargs))

    # ---------------- dataset
    add(
        "dataset_config_mode",
        "radio",
        "Dataset Source",
        "Folder: kohya <repeats>_<name> sub-folders (e.g. 1_ohwx man) inside the training folder. TOML: a kohya dataset config file.",
        "dataset",
        5,
        choices=(("Kohya folder structure (<repeats>_<name> sub-folders)", FOLDER_DATASET_MODE), ("Kohya dataset TOML file", TOML_DATASET_MODE)),
    )
    add(
        "parent_folder_path",
        "folder",
        "Training Images Folder (train_data_dir)",
        "Parent folder of <repeats>_<trigger class> sub-folders with images and .txt captions, e.g. 1_ohwx man. Repeats multiply each epoch.",
        "dataset",
        10,
        placeholder="e.g. C:/datasets/my_person (contains 1_ohwx man)",
    )
    add(
        "reg_data_dir",
        "folder",
        "Regularization Images Folder (optional)",
        "Optional parent folder of <repeats>_<class> sub-folders (e.g. 1_man) with class images; doubles the steps per epoch.",
        "dataset",
        15,
    )
    add(
        "dataset_config",
        "file",
        "Dataset TOML File",
        "Only used with 'Kohya dataset TOML file': a sd-scripts dataset config ([[datasets]] / [[datasets.subsets]]).",
        "dataset",
        20,
        file_types=(".toml",),
    )
    add("dataset_resolution_width", "number", "Resolution Width", "Training width in pixels (multiple of 64). 1024 is the native FLUX/SDXL size.", "dataset", 30, minimum=256, maximum=4096, step=64)
    add("dataset_resolution_height", "number", "Resolution Height", "Training height in pixels (multiple of 64). 1024 is the native FLUX/SDXL size.", "dataset", 31, minimum=256, maximum=4096, step=64)
    add("dataset_batch_size", "number", "Batch Size", "Images per step. 1 is the measured default; higher needs more VRAM and usually a higher learning rate.", "dataset", 40, minimum=1, maximum=64, step=1)
    add(
        "dataset_caption_extension",
        "text",
        "Caption Extension",
        "Caption file extension next to each image (.txt). Images without a caption use the folder's class name as caption.",
        "dataset",
        50,
    )
    add(
        "dataset_enable_bucket",
        "checkbox",
        "Enable Aspect-Ratio Buckets",
        "Train non-square images at their own aspect ratio. Off (presets): every image is resized/cropped to the resolution.",
        "dataset",
        60,
    )
    add("dataset_bucket_no_upscale", "checkbox", "Do Not Upscale Small Images", "With buckets: never upscale images smaller than the resolution.", "dataset", 61)
    add("prior_loss_weight", "number", "Regularization Loss Weight", "Loss weight of regularization images (prior_loss_weight). 1.0 recommended.", "dataset", 70, advanced=True, minimum=0, maximum=10, step=0.05)
    add("min_bucket_reso", "number", "Min Bucket Resolution", "Smallest bucket side in pixels (256).", "dataset", 80, advanced=True, minimum=64, maximum=4096, step=64)
    add("max_bucket_reso", "number", "Max Bucket Resolution", "Largest bucket side in pixels (2048).", "dataset", 81, advanced=True, minimum=256, maximum=8192, step=64)
    add("bucket_reso_steps", "number", "Bucket Resolution Steps", "Bucket size granularity in pixels; 64 recommended (must be divisible by 32).", "dataset", 82, advanced=True, minimum=32, maximum=256, step=32)
    if sdxl:
        add(
            "max_token_length",
            "dropdown",
            "Max Caption Tokens",
            "CLIP token limit of a caption: 75 (default), 150 or 225 for long captions.",
            "dataset",
            85,
            advanced=True,
            choices=(75, 150, 225),
        )
    add("shuffle_caption", "checkbox", "Shuffle Caption Tags", "Shuffle comma-separated caption tags each step (tag captions only). Not possible with cached text encoder outputs.", "dataset", 90, advanced=True)
    add("keep_tokens", "number", "Keep First N Tags", "With shuffling: keep the first N comma-separated tags in place (e.g. the trigger word).", "dataset", 91, advanced=True, minimum=0, maximum=32, step=1)
    add("caption_dropout_rate", "number", "Caption Dropout Rate", "Probability of training a step with an empty caption (0 = off). Not possible with cached text encoder outputs.", "dataset", 92, advanced=True, minimum=0, maximum=1, step=0.01)
    add("flip_aug", "checkbox", "Horizontal Flip Augmentation", "Randomly mirror images. Avoid for faces/text whose asymmetry matters.", "dataset", 93, advanced=True)
    add("random_crop", "checkbox", "Random Crop", "Random instead of center crop. Not possible together with latent caching.", "dataset", 94, advanced=True)

    # ---------------- output
    add("output_dir", "folder", "Output Folder", "Where checkpoints, the runtime config and logs are written.", "output", 10)
    add("output_name", "text", "Output Name", "File name of the trained model (without extension).", "output", 20)
    add("save_every_n_epochs", "number", "Save Every N Epochs", "Save a checkpoint every N epochs (0 = only at the end). Presets: 25 of 200 epochs.", "output", 30, minimum=0, step=1)
    add("save_every_n_steps", "number", "Save Every N Steps", "Save a checkpoint every N steps (0 = off).", "output", 31, minimum=0, step=1)
    add("save_last_n_epochs", "number", "Keep Last N Epoch Saves", "Delete older epoch checkpoints, keeping the newest N (0 = keep all).", "output", 40, advanced=True, minimum=0, step=1)
    add("save_last_n_steps", "number", "Keep Last N Steps of Saves", "Delete step checkpoints older than N steps (0 = keep all).", "output", 41, advanced=True, minimum=0, step=1)
    add(
        "save_precision",
        "dropdown",
        "Save Precision",
        "Precision of the saved weights. LoRA presets: fp32; fine-tunes: bf16 (half the file size).",
        "output",
        50,
        choices=("fp32", "bf16", "fp16"),
    )
    add(
        "save_model_as",
        "dropdown",
        "Save Format",
        "safetensors (recommended)" + ("" if flux else " or ckpt for old tools") + ".",
        "output",
        55,
        advanced=True,
        choices=("safetensors",) if flux else ("safetensors", "ckpt"),
    )
    add("save_state", "checkbox", "Save Training State", "Also save optimizer state with each checkpoint so training can be resumed (large).", "output", 60, advanced=True)
    add("save_state_on_train_end", "checkbox", "Save State at the End", "Save the resumable training state once training finishes.", "output", 61, advanced=True)
    add("save_last_n_epochs_state", "number", "Keep Last N Epoch States", "Keep only the newest N epoch states (0 = keep all).", "output", 62, advanced=True, minimum=0, step=1)
    add("save_last_n_steps_state", "number", "Keep Last N Steps of States", "Keep only states newer than N steps (0 = keep all).", "output", 63, advanced=True, minimum=0, step=1)
    add("resume", "folder", "Resume From State Folder", "A saved training state folder (…-state) to continue a run from. Empty = new run.", "output", 70, advanced=True)

    # ---------------- training
    add(
        "training_mode",
        "radio",
        "Training Mode",
        "LoRA: small adapter (flux_train_network / sdxl_train_network). Full Fine-Tuning: trains the whole model (DreamBooth: flux_train / sdxl_train).",
        "training",
        0,
        choices=tuple(TRAINING_MODE_CHOICES),
    )
    add(
        "max_train_epochs",
        "number",
        "Max Training Epochs",
        "Epochs to train (one epoch = every image × its folder repeats). Overrides steps when > 0. Presets: 200 with 1 repeat.",
        "training",
        10,
        minimum=0,
        step=1,
    )
    add("max_train_steps", "number", "Max Training Steps", "Used only when epochs = 0. sd-scripts falls back to 1600 steps when both are 0.", "training", 11, minimum=0, step=1)
    add(
        "learning_rate",
        "number",
        "Learning Rate",
        (
            "FLUX LoRA presets: 5e-5 (1e-4 for 4-GPU runs); FLUX fine-tune: 4e-6 (1e-5 at batch 7)."
            if flux
            else "SDXL LoRA: 1e-4 (U-Net); SDXL fine-tune presets: 8e-6 for the U-Net."
        ),
        "training",
        20,
        minimum=0,
    )
    add("unet_lr", "number", "Transformer / U-Net LR (LoRA)", "Separate learning rate for the " + ("FLUX transformer" if flux else "U-Net") + " LoRA. 0 = use Learning Rate.", "training", 21, modes=LORA_ONLY, advanced=True, minimum=0)
    add(
        "text_encoder_lr",
        "number",
        "Text Encoder LR (LoRA)",
        (
            "CLIP-L LoRA learning rate; 0 = train only the FLUX transformer. Presets: 5e-5 (CLIP-L trained, T5 outputs cached)."
            if flux
            else "LoRA learning rate of both CLIP text encoders; 0 = U-Net only (needed to cache text encoder outputs). Try 1e-5 to 5e-5."
        ),
        "training",
        22,
        modes=LORA_ONLY,
        minimum=0,
    )
    if flux:
        add(
            "t5xxl_lr",
            "number",
            "T5-XXL LR (LoRA)",
            "T5-XXL LoRA learning rate; only used with train_t5xxl=True in Network Arguments (T5 outputs are then not cached). 0 = Text Encoder LR.",
            "training",
            23,
            modes=LORA_ONLY,
            advanced=True,
            minimum=0,
        )
    add(
        "optimizer_type",
        "dropdown",
        "Optimizer",
        "Adafactor with a fixed LR is what every preset uses (low VRAM, stable). AdamW8bit is a common alternative.",
        "training",
        30,
        choices=OPTIMIZER_CHOICES,
        allow_custom_value=True,
    )
    add("optimizer_args", "text", "Optimizer Arguments", "Space-separated key=value pairs. Adafactor presets: " + _ADAFACTOR_ARGS, "training", 31)
    add("lr_scheduler", "dropdown", "LR Scheduler", "constant is used by every preset.", "training", 40, choices=LR_SCHEDULER_CHOICES)
    add("lr_warmup_steps", "number", "LR Warmup Steps", "Warm-up steps, or a ratio of the total steps when below 1 (e.g. 0.05). 0 for constant.", "training", 41, minimum=0)
    add("lr_scheduler_num_cycles", "number", "LR Scheduler Cycles", "Restarts for cosine_with_restarts.", "training", 42, advanced=True, minimum=1, step=1)
    add("lr_scheduler_power", "number", "LR Scheduler Power", "Power of the polynomial scheduler.", "training", 43, advanced=True, minimum=0)
    add("lr_scheduler_type", "text", "Custom Scheduler Class", "Optional full class name of another scheduler (e.g. CosineAnnealingLR). Empty = use LR Scheduler.", "training", 44, advanced=True)
    add(
        "lr_scheduler_args",
        "text",
        "Scheduler Arguments",
        "Space-separated key=value pairs, text in quotes: min_lr=1e-6 (cosine_with_min_lr), step_rules='1:10,0.1:20,0.01' (piecewise_constant) or the custom scheduler's arguments.",
        "training",
        45,
        advanced=True,
    )
    add(
        "max_grad_norm",
        "number",
        "Max Gradient Norm",
        "Gradient clipping; 0 = off. FLUX presets: 1.0, SDXL fine-tune presets: 0.",
        "training",
        50,
        minimum=0,
        step=0.1,
    )
    add("gradient_accumulation_steps", "number", "Gradient Accumulation Steps", "Accumulate N steps before an optimizer step (effective batch = batch × N). Must be 1 with fused backward pass.", "training", 60, minimum=1, step=1)
    add("seed", "number", "Seed", "Training seed; 0 = random.", "training", 70, minimum=0, step=1)

    # ---------------- model files and precision
    if flux:
        add(
            "pretrained_model_name_or_path",
            "file",
            "FLUX.1 Dev Model",
            "FLUX.1 dev transformer (flux1-dev.safetensors, ~23.8 GB) or a FLUX.1 dev fine-tune such as flux1-krea-dev / FLUX-SRPO.",
            "model",
            10,
            file_types=(".safetensors", ".sft"),
        )
        add("ae", "file", "FLUX VAE (AE)", "FLUX.1 autoencoder ae.safetensors.", "model", 20, file_types=(".safetensors", ".sft"))
        add("clip_l", "file", "CLIP-L Text Encoder", "clip_l.safetensors.", "model", 30, file_types=(".safetensors",))
        add("t5xxl", "file", "T5-XXL Text Encoder", "t5xxl_fp16.safetensors (fp16 recommended over fp8 for caching quality).", "model", 40, file_types=(".safetensors",))
    else:
        add(
            "pretrained_model_name_or_path",
            "file",
            "SDXL Base Model",
            "SDXL checkpoint (.safetensors). Realistic person presets were made with RealVis v5 FP32; any SDXL 1.0 fine-tune works.",
            "model",
            10,
            file_types=(".safetensors", ".ckpt"),
        )
        add(
            "vae",
            "file",
            "SDXL VAE (optional)",
            "VAE file, folder or Hugging Face id. Presets: stabilityai/sdxl-vae (downloaded once). Empty = the checkpoint's VAE.",
            "model",
            20,
            file_types=(".safetensors",),
        )
        add("no_half_vae", "checkbox", "No Half VAE (fp32 VAE)", "Keep the VAE in fp32; the original SDXL VAE produces NaNs in fp16/bf16. Recommended on.", "model", 25)
    add("mixed_precision", "dropdown", "Mixed Precision", "bf16 for RTX 30xx and newer (all presets). fp16 only for older GPUs.", "model", 50, choices=("no", "fp16", "bf16"))
    add("full_bf16", "checkbox", "Full BF16 Training", "Keep weights and gradients in bf16 (all presets; saves a lot of VRAM). Needs Mixed Precision bf16.", "model", 60)
    add("full_fp16", "checkbox", "Full FP16 Training", "fp16 weights and gradients (older GPUs only; needs Mixed Precision fp16).", "model", 61, advanced=True)

    # ---------------- LoRA network
    add(
        "network_module",
        "dropdown",
        "Network Module",
        "networks.lora_flux for FLUX.1 LoRA." if flux else "networks.lora for SDXL LoRA (LoCon: add conv_dim / conv_alpha network arguments).",
        "network",
        5,
        modes=LORA_ONLY,
        advanced=True,
        choices=("networks.lora_flux", "networks.oft_flux") if flux else ("networks.lora", "networks.oft"),
        allow_custom_value=True,
    )
    add(
        "network_dim",
        "number",
        "Network Rank (Dim)",
        "LoRA rank. FLUX presets: 128 (64 on 8 GB, 48 on 6 GB)." if flux else "LoRA rank. 32-64 for a person or style, 128 for complex concepts.",
        "network",
        10,
        modes=LORA_ONLY,
        minimum=1,
        maximum=1024,
        step=1,
    )
    add(
        "network_alpha",
        "number",
        "Network Alpha",
        "Scales the LoRA by alpha/rank. FLUX presets use alpha = rank (128)." if flux else "Scales the LoRA by alpha/rank; rank/2 is a common choice.",
        "network",
        11,
        modes=LORA_ONLY,
        minimum=0,
        maximum=1024,
    )
    add("network_dropout", "number", "Network Dropout", "Dropout of the LoRA layers (0 = off; 0.1 can reduce overfitting).", "network", 20, modes=LORA_ONLY, advanced=True, minimum=0, maximum=1, step=0.01)
    add("network_args", "text", "Network Arguments", "Extra space-separated key=value network arguments (e.g. conv_dim=8 conv_alpha=4).", "network", 30, modes=LORA_ONLY, advanced=True)
    add("network_weights", "file", "Continue From LoRA", "Start from an existing LoRA file of the same rank (empty = new LoRA).", "network", 40, modes=LORA_ONLY, advanced=True, file_types=(".safetensors",))
    add("dim_from_weights", "checkbox", "Rank From LoRA File", "Take rank/alpha from the 'Continue From LoRA' file.", "network", 41, modes=LORA_ONLY, advanced=True)
    add("scale_weight_norms", "number", "Scale Weight Norms", "Max-norm regularization of LoRA weights (0 = off; 1.0 if used).", "network", 50, modes=LORA_ONLY, advanced=True, minimum=0)
    if flux:
        add(
            "train_blocks",
            "dropdown",
            "Train Blocks",
            "Which FLUX blocks get LoRA modules: all (presets), single or double.",
            "network",
            60,
            modes=LORA_ONLY,
            advanced=True,
            choices=("all", "single", "double"),
        )
        add("train_double_block_indices", "text", "Double Block Indices", "Double blocks to train (19): 'all', 'none' or e.g. 0-5,8.", "network", 61, modes=LORA_ONLY, advanced=True)
        add("train_single_block_indices", "text", "Single Block Indices", "Single blocks to train (38): 'all', 'none' or e.g. 0-9.", "network", 62, modes=LORA_ONLY, advanced=True)

    # ---------------- fine-tuning
    if flux:
        add(
            "fused_backward_pass",
            "checkbox",
            "Fused Backward Pass",
            "Apply Adafactor updates during the backward pass (big VRAM saving; all fine-tune presets). Adafactor only, accumulation 1.",
            "finetune",
            10,
            modes=FULL_ONLY,
        )
        add("blockwise_fused_optimizers", "checkbox", "Blockwise Fused Optimizers", "Alternative to fused backward pass that works with any optimizer (one optimizer per block).", "finetune", 11, modes=FULL_ONLY, advanced=True)
        add("mem_eff_save", "checkbox", "Memory-Efficient Save", "Save the 23 GB checkpoint without an extra copy in RAM (presets: on).", "finetune", 20, modes=FULL_ONLY)
    else:
        add(
            "learning_rate_te1",
            "number",
            "Text Encoder 1 LR (CLIP-L)",
            "Trains SDXL text encoder 1 with this LR; 0 = frozen. Quality 1 presets: 3e-6; 0 on 6-8 GB GPUs.",
            "finetune",
            10,
            modes=FULL_ONLY,
            minimum=0,
        )
        add("learning_rate_te2", "number", "Text Encoder 2 LR (OpenCLIP-G)", "Trains text encoder 2; 0 = frozen (all presets).", "finetune", 11, modes=FULL_ONLY, minimum=0)
        add("fused_backward_pass", "checkbox", "Fused Backward Pass", "Apply Adafactor updates during the backward pass to save VRAM. Adafactor only, accumulation 1.", "finetune", 20, modes=FULL_ONLY)

    # ---------------- memory and speed
    max_swap = model["max_blocks_to_swap"]
    add(
        "blocks_to_swap",
        "slider",
        "Blocks to Swap",
        (
            f"FLUX blocks kept in RAM and streamed to the GPU (0-{max_swap}). More = less VRAM, slower. 24 GB presets: 13 (LoRA), 10 (fine-tune)."
            if flux
            else f"SDXL transformer blocks kept in RAM (0-{max_swap}). 0 on 16 GB+; 50-68 on 8-12 GB GPUs."
        ),
        "memory",
        10,
        minimum=0,
        maximum=max_swap,
        step=1,
    )
    add("gradient_checkpointing", "checkbox", "Gradient Checkpointing", "Recompute activations in the backward pass: large VRAM saving for ~20% speed. On except 48-80 GB presets.", "memory", 20)
    if flux:
        add("cpu_offload_checkpointing", "checkbox", "CPU Offload Checkpointing", "Offload checkpointed activations to RAM. Not together with block swap.", "memory", 25, advanced=True)
        add("fp8_base", "checkbox", "FP8 Base Model", "Load the frozen FLUX weights in fp8 (plain cast). Prefer FP8 Scaled.", "memory", 30, modes=LORA_ONLY, advanced=True)
        add("fp8_scaled", "checkbox", "FP8 Scaled Base Model", "Scaled fp8 FLUX weights for LoRA: about 11 GB less weight memory, slightly lower quality. The FP8_Scaled presets spend it on fewer swapped blocks (faster steps) at the same GPU tier.", "memory", 31, modes=LORA_ONLY)
        add("fp8_fast", "checkbox", "Fast FP8 Matmul", "Use fp8 matmul with FP8 Scaled (RTX 40xx+). Faster, small quality cost.", "memory", 32, modes=LORA_ONLY, advanced=True)
        add("fp8_base_unet", "checkbox", "FP8 Transformer Only", "fp8 for the FLUX transformer only, text encoders stay in bf16.", "memory", 33, modes=LORA_ONLY, advanced=True)
    else:
        add("fp8_base", "checkbox", "FP8 Base U-Net", "Load the frozen U-Net in fp8 for LoRA (saves VRAM, lowers quality).", "memory", 30, modes=LORA_ONLY, advanced=True)
        add("sdpa", "checkbox", "SDPA Attention", "PyTorch scaled-dot-product attention (fast, low VRAM). The fine-tune presets were measured without it.", "memory", 40)
        add("xformers", "checkbox", "xformers Attention", "xformers memory-efficient attention (alternative to SDPA).", "memory", 41)
    add("highvram", "checkbox", "High VRAM Mode", "Skip cache clearing during latent caching (faster when VRAM is plentiful; LoRA presets: on).", "memory", 50, advanced=True)
    add(
        "disable_mmap_load_safetensors",
        "checkbox",
        "Classic Safetensors Loading",
        "Revert to the old model loading (uses more RAM). The 4-GPU presets turn it on.",
        "memory",
        51,
        advanced=True,
    )
    if flux:
        add("disable_numpy_memmap", "checkbox", "Disable NumPy Memmap", "Turn off the numpy memory-mapped loading optimisation if it causes problems.", "memory", 52, advanced=True)
    add("compile", "checkbox", "torch.compile", "Compile the model blocks (FLUX presets: on; faster after the first steps). Needs Triton and a C++ compiler.", "memory", 60)
    add("compile_backend", "dropdown", "Compile Backend", "inductor recommended.", "memory", 61, advanced=True, choices=("inductor", "eager", "aot_eager", "cudagraphs"))
    add("compile_mode", "dropdown", "Compile Mode", "default recommended.", "memory", 62, advanced=True, choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"))
    add("compile_dynamic", "dropdown", "Compile Dynamic Shapes", "auto lets PyTorch decide.", "memory", 63, advanced=True, choices=("auto", "true", "false"))
    add("compile_fullgraph", "checkbox", "Compile Full Graph", "Require one graph per block.", "memory", 64, advanced=True)
    add("compile_cache_size_limit", "number", "Compile Cache Size Limit", "torch._dynamo cache size limit (0 = PyTorch default). Raise it with many bucket sizes.", "memory", 65, advanced=True, minimum=0, step=1)

    # ---------------- samples
    add(
        "sample_prompts",
        "file",
        "Sample Prompts",
        "Prompt file (one prompt per line, kohya syntax: --w 1024 --h 1024 --s 28 --d 42) or the prompt text itself.",
        "samples",
        10,
        file_types=(".txt",),
        lines=3,
    )
    add("sample_every_n_epochs", "number", "Sample Every N Epochs", "Render the sample prompts every N epochs (0 = off).", "samples", 20, minimum=0, step=1)
    add("sample_every_n_steps", "number", "Sample Every N Steps", "Render the sample prompts every N steps (0 = off).", "samples", 21, minimum=0, step=1)
    add("sample_at_first", "checkbox", "Sample Before Training", "Render the samples once before the first step (baseline).", "samples", 30)
    if sdxl:
        add("sample_sampler", "dropdown", "Sample Sampler", "Scheduler for sample images (euler_a in the presets).", "samples", 40, advanced=True, choices=SAMPLER_CHOICES)

    # ---------------- caching
    add("cache_latents", "checkbox", "Cache Latents", "Encode all images with the VAE once before training (faster, less VRAM). Not with random crop.", "caching", 10)
    add("cache_latents_to_disk", "checkbox", "Cache Latents to Disk", "Keep the latent cache next to the images so later runs skip encoding.", "caching", 11)
    add(
        "cache_text_encoder_outputs",
        "checkbox",
        "Cache Text Encoder Outputs",
        (
            "Encode captions once (T5 is then unloaded, big VRAM saving). With a CLIP-L LR only T5 is cached. Not with caption shuffle/dropout."
            if flux
            else "Encode captions once and unload the text encoders. Needs text encoder LR 0. Not with caption shuffle/dropout."
        ),
        "caching",
        20,
    )
    add("cache_text_encoder_outputs_to_disk", "checkbox", "Cache Text Encoder Outputs to Disk", "Keep the text-encoder cache next to the images for later runs.", "caching", 21)
    if flux:
        add(
            "cache_text_encoder_outputs_on_cpu",
            "checkbox",
            "Encode Captions on CPU",
            "Run T5 caching on the CPU to save VRAM during caching (slower; used by the 8-10 GB presets).",
            "caching",
            22,
            advanced=True,
        )
    add("vae_batch_size", "number", "Latent Caching Batch Size", "Images per VAE batch while caching latents (presets: 4, 2 on small GPUs).", "caching", 30, advanced=True, minimum=1, maximum=64, step=1)
    add("skip_cache_check", "checkbox", "Skip Cache Validation", "Trust existing cache files without checking their content (faster start).", "caching", 40, advanced=True)

    # ---------------- advanced training
    if flux:
        add(
            "timestep_sampling",
            "dropdown",
            "Timestep Sampling",
            "sigmoid (all presets), shift, flux_shift (resolution dependent), uniform or sigma.",
            "advanced",
            10,
            choices=("sigmoid", "shift", "flux_shift", "uniform", "sigma"),
        )
        add("discrete_flow_shift", "number", "Discrete Flow Shift", "Shift for 'shift' sampling and the sampler; presets: 3.1582.", "advanced", 11, minimum=0, maximum=100, step=0.0001)
        add("sigmoid_scale", "number", "Sigmoid Scale", "Scale of sigmoid/shift timestep sampling (1.0).", "advanced", 12, advanced=True, minimum=0, maximum=10, step=0.05)
    add("loss_type", "dropdown", "Loss Type", "l2 (presets), l1, huber or smooth_l1.", "advanced", 20, advanced=True, choices=("l2", "l1", "huber", "smooth_l1"))
    add("huber_c", "number", "Huber C", "Huber / smooth L1 decay parameter (0.1).", "advanced", 21, advanced=True, minimum=0, step=0.01)
    add("huber_schedule", "dropdown", "Huber Schedule", "Huber parameter schedule (snr).", "advanced", 22, advanced=True, choices=("snr", "exponential", "constant"))
    add("huber_scale", "number", "Huber Scale", "Huber loss scale (1.0).", "advanced", 23, advanced=True, minimum=0, step=0.05)
    add(
        "ip_noise_gamma",
        "number",
        "Input Perturbation Noise",
        "Input perturbation noise (0 = off). SDXL fine-tune presets: 0.1." if sdxl else "Input perturbation noise (0 = off; ~0.1 if used).",
        "advanced",
        30,
        advanced=flux,
        minimum=0,
        maximum=1,
        step=0.01,
    )
    if sdxl:
        add("noise_offset", "number", "Noise Offset", "Noise offset (SDXL presets: 0.0357, the value SDXL was trained with). 0 = off.", "advanced", 40, minimum=0, maximum=1, step=0.0001)
        add("min_snr_gamma", "number", "Min SNR Gamma", "Min-SNR loss weighting (0 = off; 5 if used).", "advanced", 41, advanced=True, minimum=0, maximum=20, step=0.5)
        add("debiased_estimation_loss", "checkbox", "Debiased Estimation Loss", "Debiased estimation loss weighting.", "advanced", 42, advanced=True)
        add("min_timestep", "number", "Min Timestep", "Lowest training timestep (0).", "advanced", 50, advanced=True, minimum=0, maximum=999, step=1)
        add("max_timestep", "number", "Max Timestep", "Highest training timestep (1000).", "advanced", 51, advanced=True, minimum=1, maximum=1000, step=1)

    # ---------------- model specific
    if flux:
        add("guidance_scale", "number", "Guidance Scale (Training)", "Distilled-guidance value fed to FLUX.1 dev while training; 1.0 in every preset.", "model_specific", 10, minimum=0, maximum=10, step=0.1)
        add("model_prediction_type", "dropdown", "Model Prediction Type", "raw (presets), additive or sigma_scaled.", "model_specific", 20, advanced=True, choices=("raw", "additive", "sigma_scaled"))
        add("apply_t5_attn_mask", "checkbox", "Apply T5 Attention Mask", "Mask T5 padding tokens. On in the Quality 1 LoRA presets, off in the others.", "model_specific", 30)
        add("t5xxl_max_token_length", "number", "T5-XXL Max Tokens", "Caption token limit of T5-XXL (512 for dev).", "model_specific", 40, advanced=True, minimum=64, maximum=512, step=1)

    # ---------------- hardware and launch
    add("gpu_ids", "text", "GPU IDs", "Comma-separated GPU indices for accelerate (e.g. 0 or 0,1,2,3).", "hardware", 10)
    add("multi_gpu", "checkbox", "Multi GPU", "Distributed training over the GPU IDs (set Number of Processes to the GPU count).", "hardware", 20)
    add("num_processes", "number", "Number of Processes", "One process per GPU (4 for the 4x GPU presets).", "hardware", 21, minimum=1, step=1)
    add("num_machines", "number", "Number of Machines", "Machines in a multi-node run.", "hardware", 22, advanced=True, minimum=1, step=1)
    add("main_process_port", "number", "Main Process Port", "Port for multi-GPU runs (0 = accelerate default 29500).", "hardware", 23, advanced=True, minimum=0, maximum=65535, step=1)
    add("num_cpu_threads_per_process", "slider", "CPU Threads per Process", "CPU threads accelerate gives each process.", "hardware", 30, advanced=True, minimum=1, maximum=max(os.cpu_count() or 1, 2), step=1)
    add("max_data_loader_n_workers", "number", "Data Loader Workers", "0 = load in the main process (presets, lowest RAM). 2 can speed up uncached data.", "hardware", 40, minimum=0, maximum=32, step=1)
    add("persistent_data_loader_workers", "checkbox", "Persistent Data Loader Workers", "Keep workers alive between epochs (needs workers > 0).", "hardware", 41, advanced=True)
    add("dynamo_backend", "dropdown", "Accelerate Dynamo Backend", "accelerate-level torch dynamo (no = off; use the torch.compile option instead).", "hardware", 50, advanced=True, choices=DYNAMO_BACKEND_CHOICES)
    add("dynamo_mode", "dropdown", "Dynamo Mode", "Dynamo optimisation mode (only with a dynamo backend).", "hardware", 51, advanced=True, choices=("", "default", "reduce-overhead", "max-autotune"))
    add("dynamo_use_fullgraph", "checkbox", "Dynamo Full Graph", "Whole-graph dynamo compilation.", "hardware", 52, advanced=True)
    add("dynamo_use_dynamic", "checkbox", "Dynamo Dynamic Shapes", "Dynamic shape tracing for dynamo.", "hardware", 53, advanced=True)
    add("extra_accelerate_launch_args", "text", "Extra accelerate launch Arguments", "Passed to accelerate launch, e.g. --same_network --machine_rank 1.", "hardware", 60, advanced=True)
    add("ddp_timeout", "number", "DDP Timeout (minutes)", "Distributed timeout (0 = default 30 minutes).", "hardware", 70, advanced=True, minimum=0, step=1)
    add("ddp_gradient_as_bucket_view", "checkbox", "DDP Gradient as Bucket View", "DDP memory optimisation for multi-GPU.", "hardware", 71, advanced=True)
    add("ddp_static_graph", "checkbox", "DDP Static Graph", "DDP static graph optimisation for multi-GPU.", "hardware", 72, advanced=True)

    # ---------------- logging
    add("log_with", "dropdown", "Logging", "TensorBoard / WandB logging (empty = off). Logs go to Output Folder/logs unless a folder is set.", "logging", 10, choices=("", "tensorboard", "wandb", "all"))
    add("logging_dir", "folder", "Logging Folder", "Base log folder (relative = inside the output folder). Empty = <output>/logs. Used only when logging is on.", "logging", 20)
    add("log_prefix", "text", "Log Folder Prefix", "Prefix of the timestamped log sub-folder.", "logging", 30, advanced=True)
    add("log_tracker_name", "text", "Tracker Name", "Project name for the tracker (WandB project).", "logging", 31, advanced=True)
    add("log_tracker_config", "file", "Tracker Config", "TOML with tracker init arguments.", "logging", 32, advanced=True, file_types=(".toml",))
    add("wandb_run_name", "text", "WandB Run Name", "Empty = output name.", "logging", 40, advanced=True)
    add("wandb_api_key", "text", "WandB API Key", "Logs in to WandB before training.", "logging", 41, advanced=True)
    add("log_config", "checkbox", "Log Training Config", "Send the (sanitised) training arguments to the tracker.", "logging", 50, advanced=True)
    add(
        "additional_parameters",
        "textarea",
        "Additional sd-scripts Arguments",
        "Extra command-line arguments appended after --config_file, e.g. --network_train_text_encoder_only.",
        "logging",
        90,
        lines=2,
    )

    # ---------------- metadata and Hugging Face
    add("training_comment", "text", "Training Comment", "Stored in the LoRA metadata (ss_training_comment).", "metadata", 5, modes=LORA_ONLY, advanced=True)
    add("metadata_title", "text", "Metadata Title", "Model title in the file metadata (default: output name).", "metadata", 10, advanced=True)
    add("metadata_author", "text", "Metadata Author", "", "metadata", 11, advanced=True)
    add("metadata_description", "text", "Metadata Description", "", "metadata", 12, advanced=True)
    add("metadata_license", "text", "Metadata License", "", "metadata", 13, advanced=True)
    add("metadata_tags", "text", "Metadata Tags", "Comma-separated tags.", "metadata", 14, advanced=True)
    add("huggingface_repo_id", "text", "Hugging Face Repo ID", "Upload checkpoints to this repo (empty = no upload).", "metadata", 20, advanced=True)
    add("huggingface_token", "text", "Hugging Face Token", "Write token for the upload.", "metadata", 21, advanced=True)
    add("huggingface_repo_type", "text", "Hugging Face Repo Type", "model (default) or dataset.", "metadata", 22, advanced=True)
    add("huggingface_repo_visibility", "text", "Hugging Face Repo Visibility", "public or private (default private).", "metadata", 23, advanced=True)
    add("huggingface_path_in_repo", "text", "Hugging Face Path in Repo", "Folder inside the repo.", "metadata", 24, advanced=True)
    add("save_state_to_huggingface", "checkbox", "Upload Training State", "Also upload the resumable training state.", "metadata", 25, advanced=True)
    add("resume_from_huggingface", "text", "Resume From Hugging Face", "repo_id/path_in_repo:revision:repo_type of a saved state to resume from.", "metadata", 26, advanced=True)
    add("async_upload", "checkbox", "Async Upload", "Upload in the background while training continues.", "metadata", 27, advanced=True)

    section_order = {section: index for index, section in enumerate(SECTION_IDS)}
    specs.sort(key=lambda spec: (section_order[spec.section], spec.priority))
    return specs


@lru_cache(maxsize=None)
def _specs_with_defaults(model_key: str) -> tuple[ParamSpec, ...]:
    specs = _build_specs(model_key)
    lora_defaults = kohya_defaults(model_key, LORA)
    full_defaults = kohya_defaults(model_key, FULL)
    return tuple(spec.with_(default=(lora_defaults if LORA in spec.modes else full_defaults)[spec.key]) for spec in specs)


def kohya_param_specs(model_key: str) -> list[ParamSpec]:
    """Every setting of the kohya backend for ``model_key``, ordered by section and priority."""
    _model(model_key)
    return list(_specs_with_defaults(model_key))


def _spec_map(model_key: str) -> dict[str, ParamSpec]:
    return {spec.key: spec for spec in _specs_with_defaults(model_key)}


# whole-number settings without an int argparse option in sd-scripts: the resolution (written as whole pixels), the batch
# size (train_batch_size), SDXL's block swap (read from the TOML as a block count) and accelerate launch's counts
_WHOLE_NUMBER_KEYS = frozenset(
    {
        "dataset_resolution_width", "dataset_resolution_height", "dataset_batch_size", "blocks_to_swap", "num_processes",
        "num_machines", "main_process_port", "num_cpu_threads_per_process",
    }
)


def kohya_whole_number(model_key: str, key: str) -> bool:
    """sd-scripts (or accelerate launch) reads this setting as an int: argparse type=int (recorded in the schema by
    tools/trainer_arg_types.py) or one of the counts above. Float options keep their fractions, and so does
    lr_warmup_steps (int or float: below 1 a ratio of the steps)."""
    return key in _WHOLE_NUMBER_KEYS or (arg_type(f"kohya:{model_key}", key) or {}).get("type") == "int"


# ----------------------------------------------------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------------------------------------------------


def _clean_text(value: object) -> str:
    """Typed text without padding and surrounding quotes (straight, curly or backticks, as copy-paste brings them)."""
    return strip_quotes(str(value if value is not None else ""))


def _looks_like_hf_repo_id(value: str) -> bool:
    """``owner/name`` Hugging Face ids (accepted by sd-scripts for SDXL base models and VAEs)."""
    text = _clean_text(value)
    if not text or "\\" in text or text.startswith((".", "/", "~")) or re.match(r"^[A-Za-z]:", text):
        return False
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", text):
        return False
    if text.lower().endswith(_MODEL_FILE_EXTENSIONS):
        return False
    return not (ROOT_DIR / text).exists() and not (APP_DIR / text).exists()


def _resolve_path(value: object) -> str:
    """Absolute forward-slash path. Quotes, ``file:///`` URLs, ``%VAR%`` and ``~`` are handled like every other path
    field (values.clean_path). Relative paths: existing under CWD/app/root first, else '../' = app-relative and
    anything else = install-root-relative (the preset convention)."""
    text = clean_path(value)
    if not text:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(text))
    if os.path.isabs(expanded):
        return _fs(os.path.normpath(expanded))
    for base in (Path.cwd(), APP_DIR, ROOT_DIR):
        candidate = base / expanded
        try:
            if candidate.exists():
                return _fs(os.path.normpath(candidate.resolve()))
        except OSError:
            continue
    base = APP_DIR if expanded.replace("\\", "/").startswith("../") else ROOT_DIR
    return _fs(os.path.normpath(base / expanded))


def _search_directories() -> list[list[Path]]:
    """Model folders, most specific first: the kohya folders, then every other Training_Models* folder."""
    kohya = []
    for name in _KOHYA_MODEL_DIR_NAMES:
        for base in (ROOT_DIR, APP_DIR):
            folder = base / name
            if folder.is_dir():
                kohya.append(folder.resolve())
    seen = {os.path.normcase(str(folder)) for folder in kohya}
    others = []
    for folder in _training_model_directories():
        identity = os.path.normcase(str(folder))
        if identity not in seen:
            seen.add(identity)
            others.append(folder)
    return [kohya, others]


def _find_unique_file(filename: str, directories: list[Path], max_depth: int = 2) -> tuple[Path | None, int]:
    wanted = filename.casefold()
    matches: dict[str, Path] = {}
    for directory in directories:
        base_depth = len(directory.parts)
        try:
            for current, dirs, files in os.walk(directory):
                if len(Path(current).parts) - base_depth >= max_depth:
                    dirs[:] = []
                for name in files:
                    if name.casefold() == wanted:
                        path = Path(current, name).resolve()
                        matches[os.path.normcase(str(path))] = path
        except OSError:
            continue
    if len(matches) == 1:
        return next(iter(matches.values())), 1
    return None, len(matches)


def _relocate_one(value: object) -> tuple[object, str | None]:
    if not isinstance(value, str):
        return value, None
    text = _clean_text(value)
    if not text or _looks_like_hf_repo_id(text):
        return value, None
    resolved = _resolve_path(text)
    if os.path.exists(resolved):
        # existing absolute paths are never changed; existing relative ones (preset convention) become absolute
        return (value, None) if os.path.isabs(os.path.expandvars(os.path.expanduser(text))) else (resolved, None)
    filename = re.split(r"[\\/]", text)[-1]
    if not filename:
        return value, None
    for tier in _search_directories():
        found, count = _find_unique_file(filename, tier)
        if found is not None:
            return _fs(found), f"Relocated '{value}' to '{_fs(found)}'."
        if count > 1:
            return value, f"Cannot relocate '{value}': {count} files named {filename} were found."
    return value, None


def relocate_model_paths(values: dict) -> dict:
    """Model paths that do not exist are replaced by the unique file with the same name in Training_Models_FLUX_1 /
    Training_Models_SDXL / FLUX_Training_Models / SDXL_Training_Models (then any other Training_Models* folder).
    Existing relative paths become absolute; Hugging Face ids and unmatched values are kept."""
    result = dict(values)
    for key in MODEL_PATH_KEYS:
        if key in result:
            relocated, message = _relocate_one(result[key])
            if message:
                log.info(message)
            result[key] = relocated
    return result


# ----------------------------------------------------------------------------------------------------------------------
# kohya_ss GUI JSON -> our values
# ----------------------------------------------------------------------------------------------------------------------

# sd-scripts defaults that the kohya_ss LoRA tab silently fell back to: it drops every 0 value from its runtime TOML
_LORA_TAB_ZERO_FALLBACKS = {
    "max_grad_norm": 1.0,
    "prior_loss_weight": 1.0,
    "guidance_scale": 3.5,
    "discrete_flow_shift": 3.0,
    "huber_c": 0.1,
    "huber_scale": 1.0,
    "lr_scheduler_power": 1.0,
    "lr_scheduler_num_cycles": 1,
    "bucket_reso_steps": 64,
    "min_bucket_reso": 256,
    "max_bucket_reso": 1024,
    "network_alpha": 1.0,
    "t5xxl_max_token_length": 512,
}

# JSON keys that describe the old GUI / other model families and carry no setting for this backend
_JSON_GUI_ONLY_KEYS = {
    "model_list",
    "flux1_checkbox",
    "sdxl_checkbox",
    "sd3_checkbox",
    "sdxl",
    "v2",
    "v_parameterization",
    "LoRA_type",
    "LyCORIS_preset",
    "save_as_bool",
    "epoch",
    "lr_warmup",
    "optimizer",
    "max_resolution",
    "train_data_dir",
    "train_batch_size",
    "caption_extension",
    "enable_bucket",
    "bucket_no_upscale",
    "xformers",
    "flux1_cache_text_encoder_outputs",
    "flux1_cache_text_encoder_outputs_to_disk",
    "flux1_cache_text_encoder_outputs_on_cpu",
    "sdxl_cache_text_encoder_outputs",
    "sdxl_no_half_vae",
    "sdxl_blocks_to_swap",
    "flux1_clip_l",
    "flux1_t5xxl",
    "flux_fused_backward_pass",
    "advanced_disable_mmap_load_safetensors",
    "sdxl_disable_mmap_load_safetensors",
    "text_encoder_lr",
    "unet_lr",
}

_FLUX_LORA_EXTRA_NETWORK_ARGS = (
    "img_attn_dim",
    "img_mlp_dim",
    "img_mod_dim",
    "single_dim",
    "txt_attn_dim",
    "txt_mlp_dim",
    "txt_mod_dim",
    "single_mod_dim",
    "in_dims",
)

# JSON keys copied 1:1 when the target key exists for the model
_JSON_DIRECT_KEYS = (
    "reg_data_dir",
    "dataset_config",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "prior_loss_weight",
    "shuffle_caption",
    "keep_tokens",
    "caption_dropout_rate",
    "flip_aug",
    "random_crop",
    "output_dir",
    "output_name",
    "save_every_n_epochs",
    "save_every_n_steps",
    "save_last_n_epochs",
    "save_last_n_steps",
    "save_state",
    "save_state_on_train_end",
    "save_last_n_epochs_state",
    "save_last_n_steps_state",
    "save_model_as",
    "resume",
    "max_train_steps",
    "learning_rate",
    "optimizer_args",
    "lr_scheduler",
    "lr_scheduler_num_cycles",
    "lr_scheduler_power",
    "lr_scheduler_type",
    "lr_scheduler_args",
    "max_grad_norm",
    "gradient_accumulation_steps",
    "seed",
    "pretrained_model_name_or_path",
    "ae",
    "vae",
    "mixed_precision",
    "full_bf16",
    "full_fp16",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_weights",
    "dim_from_weights",
    "scale_weight_norms",
    "train_blocks",
    "train_double_block_indices",
    "train_single_block_indices",
    "learning_rate_te1",
    "learning_rate_te2",
    "fused_backward_pass",
    "blockwise_fused_optimizers",
    "mem_eff_save",
    "blocks_to_swap",
    "gradient_checkpointing",
    "cpu_offload_checkpointing",
    "fp8_base",
    "fp8_scaled",
    "fp8_fast",
    "fp8_base_unet",
    "highvram",
    "disable_mmap_load_safetensors",
    "disable_numpy_memmap",
    "compile",
    "compile_backend",
    "compile_mode",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_cache_size_limit",
    "sample_prompts",
    "sample_every_n_epochs",
    "sample_every_n_steps",
    "sample_sampler",
    "cache_latents",
    "cache_latents_to_disk",
    "vae_batch_size",
    "skip_cache_check",
    "timestep_sampling",
    "discrete_flow_shift",
    "loss_type",
    "huber_c",
    "huber_schedule",
    "huber_scale",
    "ip_noise_gamma",
    "noise_offset",
    "min_snr_gamma",
    "debiased_estimation_loss",
    "min_timestep",
    "max_timestep",
    "guidance_scale",
    "model_prediction_type",
    "apply_t5_attn_mask",
    "t5xxl_max_token_length",
    "max_token_length",
    "gpu_ids",
    "multi_gpu",
    "num_processes",
    "num_machines",
    "main_process_port",
    "num_cpu_threads_per_process",
    "max_data_loader_n_workers",
    "persistent_data_loader_workers",
    "dynamo_backend",
    "dynamo_mode",
    "dynamo_use_fullgraph",
    "dynamo_use_dynamic",
    "extra_accelerate_launch_args",
    "log_with",
    "logging_dir",
    "log_tracker_name",
    "log_tracker_config",
    "wandb_run_name",
    "wandb_api_key",
    "log_config",
    "training_comment",
    "metadata_title",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "save_state_to_huggingface",
    "resume_from_huggingface",
    "async_upload",
)


def _is_active(value: object) -> bool:
    return value not in (None, "", False, 0, 0.0, [], "none", "None")


def _number(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


# settings that hold sd-scripts argument lists (a runtime TOML stores them as lists)
_ARGUMENT_KEYS = ("optimizer_args", "lr_scheduler_args", "network_args", "additional_parameters")


def _coerce_to_spec(spec: ParamSpec | None, value: object, whole: bool = False) -> object:
    """Bring a loaded value into the widget's type (ints for integer fields, bools for checkboxes). Numbers are read like
    every other field (values.parse_number: ' 1e-4 ', '0,0001'; ``whole``: '10,000' is ten thousand); text that is no
    number keeps the default."""
    if spec is None or value is None:
        return value
    if spec.kind == "checkbox":
        if isinstance(value, str):
            return value.strip().casefold() in ("1", "true", "yes", "on")
        return bool(value)
    if spec.kind in ("number", "slider") and not isinstance(value, bool):
        try:
            number = parse_number(value, label=spec.label or spec.key, whole=whole)
        except ValueError:
            number = None
        if number is None:
            number = _number(spec.default)
        integer_field = isinstance(spec.default, int) and not isinstance(spec.default, bool)
        return int(number) if integer_field and float(number).is_integer() else float(number)
    if spec.kind in ("text", "textarea", "file", "folder"):
        if isinstance(value, (list, tuple)):
            return _join_args(value) if spec.key in _ARGUMENT_KEYS else " ".join(str(item) for item in value)
        return "" if value is None else str(value)
    return value


def _split_args(text: object) -> list[str]:
    """Split a command-line fragment; keeps Windows backslashes and handles quotes."""
    if isinstance(text, (list, tuple)):
        return [str(item) for item in text if str(item).strip()]
    raw = str(text or "").strip()
    if not raw:
        return []
    lexer = shlex.shlex(raw, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return raw.split()


def _arg_items(text: object) -> list[str]:
    """Whitespace-separated items of an argument list. Quotes stay, because sd-scripts reads optimizer and scheduler
    values with ast.literal_eval (mode='triangular2', step_rules='1:10,0.1:20,0.01'); only one layer of quotes around
    a whole item goes, which keeps the space in "betas=(0.9, 0.999)"."""
    if isinstance(text, (list, tuple)):
        words = [str(item) for item in text]
    else:
        raw = str(text or "").strip()
        lexer = shlex.shlex(raw, posix=False)
        lexer.whitespace_split = True
        lexer.commenters = ""
        try:
            words = list(lexer)
        except ValueError:  # an unclosed quote: plain words (the argument check names the broken item)
            words = raw.split()
    items = []
    for word in words:
        word = word.strip()
        if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
            word = word[1:-1].strip()
        if word:
            items.append(word)
    return items


# a comma separates two key=value items only where the next key starts ("a=1,b=2", "a=1, b=2")
_NEXT_KEY_RE = re.compile(r",\s*(?=[A-Za-z_][\w.]*=)")


def _split_at_item_commas(word: str) -> list[str]:
    """'a=1,b=2' -> 'a=1', 'b=2', but only after a complete key=value (one '=' outside quotes): sd-scripts' own
    network_reg_lrs=single_blocks_(\\d|10)_=1e-3,double_blocks=2e-3 and quoted values stay one item."""
    parts, start, quote, equals = [], 0, "", 0
    for index, char in enumerate(word):
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "=":
            equals += 1
        elif char == "," and equals == 1 and _NEXT_KEY_RE.match(word, index):
            parts.append(word[start:index])
            start, equals = index + 1, 0
    parts.append(word[start:])
    return parts


def _split_key_value_args(text: object, raw_values: bool = False) -> list[str]:
    """``key=value`` items of sd-scripts' optimizer / scheduler / network arguments. Like the kohya_ss GUI one item per
    whitespace-separated word, and a comma also separates items where the next ``key=`` follows a complete one. Commas
    inside a value stay: ``betas=0.9,0.999`` is one tuple argument (sd-scripts reads each value with
    ``ast.literal_eval``). ``raw_values``: network modules read their values as plain text, so double quotes go (the
    kohya_ss GUI removed them all)."""
    items = []
    for word in _arg_items(text):
        for part in _split_at_item_commas(word):
            part = part.strip().strip(",").strip()
            if raw_values:
                part = part.replace('"', "")
            if part:
                items.append(part)
    return items


def _join_args(items) -> str:
    """List items (a runtime TOML's optimizer_args ...) as the text of one field; an item with spaces gets quotes, so it
    stays one item when the field is read again."""
    words = []
    for item in items:
        item = str(item).strip()
        if any(char.isspace() for char in item):
            quote = "'" if '"' in item and "'" not in item else '"'
            item = f"{quote}{item}{quote}"
        if item:
            words.append(item)
    return " ".join(words)


def _argument_options(text: object, raw_values: bool = False) -> dict[str, str]:
    """{key: value} of an argument list (items without '=' are reported by _argument_problems)."""
    options = {}
    for item in _split_key_value_args(text, raw_values):
        key, separator, value = item.partition("=")
        if separator:
            options[key.strip()] = value.strip()
    return options


def _literal_value(text: str, default: object = None) -> object:
    """``text`` read the way sd-scripts reads optimizer / scheduler values (ast.literal_eval); ``default`` if it cannot."""
    try:
        return ast.literal_eval(text)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return default


_UNREADABLE = object()


def _argument_problems(label: str, text: object, literal: bool) -> list[str]:
    """What sd-scripts fails on in an argument list, only after the models are loaded (the kohya_ss GUI checked the format
    first: common_gui.validate_args_setting). Optimizer / scheduler arguments are split at their one '=' and the value is
    read with ast.literal_eval (``literal``); network arguments are split at the first '='."""
    malformed, unreadable = [], []
    for item in _split_key_value_args(text, raw_values=not literal):
        key, separator, value = item.partition("=")
        if not separator or not key.isidentifier() or not value or (literal and "=" in value):
            malformed.append(item)
        elif literal and _literal_value(value, _UNREADABLE) is _UNREADABLE:
            unreadable.append(item)
    problems = []
    if malformed:
        rule = "one '=' each, no spaces around it" if literal else "no spaces around '='"
        problems.append(
            f"{label}: write key=value items separated by spaces ({rule}; quote a whole item that holds spaces, "
            f"e.g. \"betas=(0.9, 0.999)\"). Not usable: {', '.join(malformed)}"
        )
    if unreadable:
        problems.append(
            f"{label}: sd-scripts reads every value as a Python literal, so text needs quotes (e.g. mode='triangular2'). "
            f"Not readable: {', '.join(unreadable)}"
        )
    return problems


def _adafactor_relative_step(optimizer_args: object) -> bool:
    """Whether sd-scripts runs Adafactor with relative_step (its default; warmup_init=True also turns it on)."""
    options = {}
    for item in _split_key_value_args(optimizer_args):
        key, _, value = item.partition("=")
        options[key.strip().casefold()] = value.strip().strip("'\"").casefold()
    return options.get("relative_step", "true") in ("true", "1") or options.get("warmup_init", "false") in ("true", "1")


def _with_argument(text: object, flag: str) -> str:
    """Additional sd-scripts arguments with ``flag`` appended once."""
    current = str(text or "").strip()
    return current if flag in _split_args(current) else f"{current} {flag}".strip()


def _without_argument(text: object, flag: str) -> str:
    """Additional sd-scripts arguments without ``flag``; everything else stays as typed."""
    current = str(text or "")
    removed = re.sub(rf"(?<!\S){re.escape(flag)}(?!\S)", "", current)
    return current if removed == current else re.sub(r"[ \t]{2,}", " ", removed).strip()


# trains only the text encoder LoRA (an sd-scripts option; the kohya_ss LoRA tab set it for unet_lr 0)
_TEXT_ENCODER_ONLY = "--network_train_text_encoder_only"


def _fold_additional_parameters(text: str, values: dict, notes: list[str], allowed: set[str]) -> str:
    """Move well-known flags of the kohya JSON 'additional_parameters' into their settings."""
    tokens = _split_args(text)
    remaining: list[str] = []
    index = 0
    folded: list[str] = []
    flag_keys = {"no_half_vae", "train_text_encoder", "cache_latents", "cache_latents_to_disk", "gradient_checkpointing", "full_bf16", "sdpa", "xformers"}
    value_keys = {"max_grad_norm": float, "learning_rate_te1": float, "learning_rate_te2": float, "learning_rate": float, "seed": int}
    while index < len(tokens):
        token = tokens[index]
        name, _, inline = token.lstrip("-").partition("=")
        if token.startswith("--") and name in flag_keys and not inline:
            if name == "train_text_encoder":
                folded.append(token)  # derived from the text encoder learning rates
            elif name in allowed:
                values[name] = True
                folded.append(token)
            else:
                remaining.append(token)
            index += 1
            continue
        if token.startswith("--") and name in value_keys and name in allowed:
            raw = inline
            consumed = 1
            if not inline and index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                raw = tokens[index + 1]
                consumed = 2
            try:
                values[name] = value_keys[name](float(raw))
                folded.append(" ".join(tokens[index : index + consumed]))
                index += consumed
                continue
            except (TypeError, ValueError):
                pass
        remaining.append(token)
        index += 1
    if folded:
        notes.append(f"Moved additional_parameters into settings: {', '.join(folded)}.")
    return " ".join(remaining)


_ON_WORDS = ("true", "1", "yes", "on")
_OFF_WORDS = ("", "false", "0", "no", "off", "none")


def _split_resume(resume: object, from_huggingface: object) -> tuple[str, str]:
    """(Resume From State Folder, Resume From Hugging Face) of a kohya config. sd-scripts keeps the Hugging Face location
    in resume and reads resume_from_huggingface as on/off (the kohya_ss GUI stored the text "True"); here Resume From
    Hugging Face holds the location itself."""
    location = str(resume or "").strip()
    flag = "" if from_huggingface is None else str(from_huggingface).strip()
    if flag.casefold() in _OFF_WORDS:
        return location, ""
    if flag.casefold() in _ON_WORDS:
        return "", location
    return location, flag  # the location typed into the Hugging Face field itself


def kohya_steps_example(epoch: int, processes: int, batch: int, accumulation: int) -> tuple[int, int, int] | None:
    """(images, sd-scripts steps, kohya_ss GUI steps) for the first image count from 15 up (x repeats, one bucket) where
    the converted epochs train more steps than the kohya_ss GUI did; None when they never do. The GUI trained
    ceil(images / batch / accumulation x epoch) steps on every GPU (lora_gui.py / dreambooth_gui.py); sd-scripts trains
    epoch x processes epochs of ceil(batches / processes / accumulation) steps (train_network.py), so it rounds every
    epoch up to whole steps."""
    processes, batch, accumulation = max(processes, 1), max(batch, 1), max(accumulation, 1)
    if epoch <= 0 or processes * batch * accumulation == 1:
        return None
    for images in range(15, 45):
        kohya_steps = math.ceil(images / batch / accumulation * epoch)
        sd_steps = epoch * processes * math.ceil(math.ceil(images / batch) / processes / accumulation)
        if sd_steps != kohya_steps:
            return images, sd_steps, kohya_steps
    return None


def kohya_steps_note(epoch: int, processes: int, batch: int, accumulation: int) -> str:
    """Conversion note on the step count of the kohya_ss GUI's epochs ('' when the converted epochs train the same)."""
    example = kohya_steps_example(epoch, processes, batch, accumulation)
    if example is None:
        return ""
    images, sd_steps, kohya_steps = example
    batch, accumulation = max(batch, 1), max(accumulation, 1)
    return (
        f"Steps: the kohya_ss GUI trained ceil(images x repeats / batch {batch} / accumulation {accumulation} x epoch {epoch}) "
        f"steps, while sd-scripts rounds every epoch up to whole steps per GPU: when images x repeats is not a multiple of "
        f"{max(processes, 1) * batch * accumulation} (batch x GPUs x accumulation) this trains more steps ({images} images: "
        f"{sd_steps} here, {kohya_steps} in the kohya_ss app). To match a kohya_ss run exactly, set Max Training Epochs 0 and "
        f"Max Training Steps to its step count."
    )


def convert_kohya_gui_json(data: dict, model_key: str, training_type: str) -> tuple[dict, list[str]]:
    """kohya_ss GUI JSON (LoRA tab or DreamBooth tab) -> values of this backend (only its keys) + notes."""
    _model(model_key)
    training_type = _training_type(training_type)
    flux = model_key == "flux1_dev"
    specs = _spec_map(model_key)
    allowed = set(specs)
    lora = training_type == LORA
    notes: list[str] = []
    out: dict[str, Any] = {"training_mode": LORA_TRAINING_MODE if lora else FULL_FINE_TUNING_MODE}
    consumed: set[str] = set()

    def get(key, default=None):
        consumed.add(key)
        return data.get(key, default)

    # model family check
    if "flux1_checkbox" in data or "sdxl" in data or "sdxl_checkbox" in data:
        json_flux = bool(data.get("flux1_checkbox"))
        json_sdxl = bool(data.get("sdxl") or data.get("sdxl_checkbox"))
        if flux and not json_flux:
            notes.append("This JSON was not saved for FLUX.1 (flux1_checkbox is off).")
        if not flux and not json_sdxl:
            notes.append("This JSON was not saved for SDXL (sdxl checkbox is off).")

    for key in _JSON_DIRECT_KEYS:
        if key in data and key in allowed:
            out[key] = data[key]
            consumed.add(key)
    if "resume" in data or "resume_from_huggingface" in data:
        out["resume"], out["resume_from_huggingface"] = _split_resume(data.get("resume"), data.get("resume_from_huggingface"))
    if str(data.get("logging_dir") or "").strip() and not str(data.get("log_with") or "").strip():
        out["log_with"] = "tensorboard"  # sd-scripts logs to TensorBoard when only logging_dir is set
        notes.append("logging_dir without log_with: sd-scripts then logged to TensorBoard; Logging set to tensorboard.")

    # dataset
    if "train_data_dir" in data:
        out["parent_folder_path"] = get("train_data_dir") or ""
    dataset_config = str(data.get("dataset_config") or "").strip()
    out["dataset_config_mode"] = TOML_DATASET_MODE if dataset_config else FOLDER_DATASET_MODE
    resolution = str(get("max_resolution") or "").replace(" ", "")
    if resolution:
        parts = [part for part in resolution.split(",") if part]
        try:
            width = int(float(parts[0]))
            height = int(float(parts[1])) if len(parts) > 1 else width
            out["dataset_resolution_width"], out["dataset_resolution_height"] = width, height
        except (ValueError, IndexError):
            notes.append(f"Could not read max_resolution '{resolution}'.")
    if "train_batch_size" in data:
        out["dataset_batch_size"] = _int(get("train_batch_size"), 1) or 1
    if "caption_extension" in data:
        extension = str(get("caption_extension") or "").strip()
        if not extension:
            extension = ".caption"
            notes.append("caption_extension was empty: the kohya_ss GUI then used sd-scripts' default '.caption' (folder-name captions for .txt datasets).")
        out["dataset_caption_extension"] = extension
    if "enable_bucket" in data:
        out["dataset_enable_bucket"] = bool(get("enable_bucket"))
    if "bucket_no_upscale" in data:
        out["dataset_bucket_no_upscale"] = bool(get("bucket_no_upscale"))
    if "max_token_length" in allowed and "max_token_length" in data:
        token_length = _int(data.get("max_token_length"), 75)
        out["max_token_length"] = token_length if token_length in (150, 225) else 75

    # training length and optimizer
    epoch = _int(get("epoch"), 0)
    json_max_epochs = _int(data.get("max_train_epochs"), 0)
    json_max_steps = _int(data.get("max_train_steps"), 0)
    consumed.update({"max_train_epochs", "max_train_steps"})
    if json_max_epochs > 0:
        out["max_train_epochs"], out["max_train_steps"] = json_max_epochs, json_max_steps
    elif json_max_steps > 0:
        out["max_train_epochs"], out["max_train_steps"] = 0, json_max_steps
    elif not {"epoch", "max_train_epochs", "max_train_steps"} & set(data):
        pass  # the file sets no training length: keep the current one
    elif dataset_config or epoch <= 0:
        # the GUI computed no steps then (it skips the calculation for a dataset TOML), so sd-scripts trained its default
        out["max_train_epochs"], out["max_train_steps"] = 0, 1600
        reason = f"epoch {epoch} is not used with a dataset TOML" if dataset_config else "epoch was 0"
        notes.append(f"In the kohya_ss GUI {reason}: sd-scripts trained its default 1600 steps; converted to Max Training Steps 1600.")
    else:
        # The GUI turned epoch into steps (images x repeats / batch x epoch) without dividing by the GPU count, while
        # every step trains one batch on each GPU: N GPUs trained epoch x N passes over the dataset (sd-scripts' epochs).
        processes = max(_int(data.get("num_processes"), 1), 1)
        out["max_train_epochs"], out["max_train_steps"] = epoch * processes, 0
        if processes > 1:
            notes.append(
                f"epoch {epoch} on {processes} GPUs: the kohya_ss GUI computed the steps of {epoch} epochs for one GPU and every GPU "
                f"trained them, which is {epoch * processes} epochs over the dataset; converted to Max Training Epochs {epoch * processes}."
            )
        batch, accumulation = _int(data.get("train_batch_size"), 1), _int(data.get("gradient_accumulation_steps"), 1)
        steps_note = kohya_steps_note(epoch, processes, batch, accumulation)
        if steps_note:
            notes.append(steps_note)
    if "optimizer" in data:
        out["optimizer_type"] = get("optimizer")
    # lora_gui.py / dreambooth_gui.py: int(lr_warmup_steps) when above 0 (0.05 meant no warmup), else lr_warmup % as a
    # ratio; with a dataset TOML lr_warmup_steps was replaced by lr_warmup % of max_train_steps
    warmup_steps = _number(get("lr_warmup_steps"), 0.0)
    warmup_percent = _number(get("lr_warmup"), 0.0)
    if dataset_config:
        warmup_steps = round(int(warmup_percent) * json_max_steps / 100) if json_max_steps > 0 and warmup_percent else 0
    if warmup_steps > 0:
        out["lr_warmup_steps"] = int(warmup_steps)
        if not float(warmup_steps).is_integer():
            notes.append(f"lr_warmup_steps {warmup_steps:g}: the kohya_ss GUI trained int(lr_warmup_steps) = {int(warmup_steps)} warmup steps; converted to {int(warmup_steps)}.")
    elif warmup_percent:
        out["lr_warmup_steps"] = warmup_percent / 100
        notes.append(f"lr_warmup {warmup_percent:g}% -> lr_warmup_steps {warmup_percent / 100:g} (ratio of the total steps).")
    else:
        out["lr_warmup_steps"] = 0
    if "save_precision" in data:
        precision = str(get("save_precision") or "").strip()
        out["save_precision"] = {"float": "fp32", "fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}.get(precision, "fp32")
    if "seed" in out:
        out["seed"] = max(_int(out["seed"], 0), 0)

    # precision / memory variants that the old GUI kept per model
    if flux:
        clip_l = data.get("flux1_clip_l") or data.get("clip_l") or ""
        t5xxl = data.get("flux1_t5xxl") or data.get("t5xxl") or ""
        consumed.update({"flux1_clip_l", "clip_l", "flux1_t5xxl", "t5xxl"})
        if clip_l:
            out["clip_l"] = clip_l
        if t5xxl:
            out["t5xxl"] = t5xxl
        if "flux1_cache_text_encoder_outputs" in data:
            out["cache_text_encoder_outputs"] = bool(get("flux1_cache_text_encoder_outputs"))
        if "flux1_cache_text_encoder_outputs_to_disk" in data:
            out["cache_text_encoder_outputs_to_disk"] = bool(get("flux1_cache_text_encoder_outputs_to_disk"))
        out["cache_text_encoder_outputs_on_cpu"] = bool(get("flux1_cache_text_encoder_outputs_on_cpu", False))
        if not lora:
            out["fused_backward_pass"] = bool(get("flux_fused_backward_pass", data.get("fused_backward_pass", False)))
            consumed.add("fused_backward_pass")
        if "xformers" in data:
            consumed.add("xformers")
            notes.append("xformers / sdpa have no effect on FLUX.1 in sd-scripts (FLUX always uses PyTorch SDPA); ignored.")
    else:
        if "sdxl_cache_text_encoder_outputs" in data:
            out["cache_text_encoder_outputs"] = bool(get("sdxl_cache_text_encoder_outputs"))
        out["cache_text_encoder_outputs_to_disk"] = bool(data.get("cache_text_encoder_outputs_to_disk", False)) and out.get("cache_text_encoder_outputs", True)
        if "sdxl_no_half_vae" in data:
            out["no_half_vae"] = bool(get("sdxl_no_half_vae"))
        if "sdxl_blocks_to_swap" in data:
            out["blocks_to_swap"] = _int(get("sdxl_blocks_to_swap"), 0)
        attention = str(get("xformers") or "").strip().casefold()
        out["sdpa"] = attention == "sdpa"
        out["xformers"] = attention == "xformers"
    # DreamBooth tab: SDXL used sdxl_disable_mmap_load_safetensors, FLUX advanced_disable_mmap_load_safetensors;
    # the LoRA tab's own disable_mmap_load_safetensors key is copied directly above.
    mmap_key = "sdxl_disable_mmap_load_safetensors" if not flux else "advanced_disable_mmap_load_safetensors"
    consumed.update({"advanced_disable_mmap_load_safetensors", "sdxl_disable_mmap_load_safetensors"})
    if not lora and mmap_key in data:
        out["disable_mmap_load_safetensors"] = bool(data[mmap_key])

    # LoRA network
    if lora:
        lora_type = str(get("LoRA_type") or "").strip()
        module = {
            "Flux1": "networks.lora_flux",
            "Flux1 OFT": "networks.oft_flux",
            "Standard": "networks.lora",
            "Kohya LoCon": "networks.lora",
            "LoRA-FA": "networks.lora_fa",
            "Kohya DyLoRA": "networks.dylora",
        }.get(lora_type)
        if lora_type and module is None:
            notes.append(f"LoRA type '{lora_type}' needs LyCORIS, which this backend does not set up; using {_model(model_key)['network_module']}.")
        out["network_module"] = module or _model(model_key)["network_module"]
        extra_args: list[str] = []
        if lora_type == "Kohya LoCon":
            extra_args += [f"conv_dim={get('conv_dim')}", f"conv_alpha={get('conv_alpha')}"]
        if flux and lora_type == "Flux1":
            for key in _FLUX_LORA_EXTRA_NETWORK_ARGS:
                value = get(key)
                if value:
                    extra_args.append(f"{key}={value}")
            if get("train_lora_ggpo"):
                extra_args += [f"ggpo_beta={get('ggpo_beta')}", f"ggpo_sigma={get('ggpo_sigma')}"]
            if get("split_qkv"):
                extra_args.append("split_qkv=True")
            if get("train_t5xxl"):
                extra_args.append("train_t5xxl=True")
            if get("split_mode"):
                # the GUI forced train_blocks=single; sd-scripts then swapped 18 blocks unless blocks_to_swap was set
                out["train_blocks"] = "single"
                if _int(out.get("blocks_to_swap"), 0) <= 0:
                    out["blocks_to_swap"] = 18
                    notes.append("split_mode is deprecated in sd-scripts, which then swapped 18 blocks: converted to train_blocks=single and Blocks to Swap 18.")
                else:
                    notes.append("split_mode is deprecated in sd-scripts (ignored next to blocks_to_swap): converted to train_blocks=single.")
        out["network_args"] = " ".join(extra_args)
        if "network_dropout" in out:
            out["network_dropout"] = _number(out["network_dropout"])
        learning_rate = _number(data.get("learning_rate"), 0.0)
        unet_lr = _number(get("unet_lr"), 0.0)
        text_encoder_lr = _number(get("text_encoder_lr"), 0.0)
        t5xxl_lr = _number(get("t5xxl_lr"), 0.0)
        if t5xxl_lr and flux:
            # the GUI wrote text_encoder_lr = [text_encoder_lr, t5xxl_lr]; a T5-XXL LoRA exists only with train_t5xxl
            out["t5xxl_lr"] = t5xxl_lr
            if not data.get("train_t5xxl"):
                notes.append(f"t5xxl_lr {t5xxl_lr:g} is kept; it trains nothing without train_t5xxl (no T5-XXL LoRA then).")
            if text_encoder_lr == 0:
                notes.append("t5xxl_lr with text_encoder_lr 0: the kohya_ss GUI refused to start (a T5-XXL LR needs a text encoder LR).")
        out["unet_lr"] = 0.0 if unet_lr == learning_rate else unet_lr
        if unet_lr and unet_lr == learning_rate:
            notes.append(f"unet_lr {unet_lr:g} equals learning_rate: stored as 0 (= use Learning Rate), same training.")
        # What the LoRA tab trained (lora_gui.py): unet_lr and text_encoder_lr both 0 = both parts at learning_rate, only
        # unet_lr 0 = only the text encoder. Here Text Encoder LR 0 means "transformer / U-Net only".
        out["text_encoder_lr"] = text_encoder_lr
        if unet_lr == 0 and text_encoder_lr == 0 and learning_rate > 0:
            out["text_encoder_lr"] = learning_rate
            notes.append(f"unet_lr and text_encoder_lr were 0: kohya trained the text encoder too, at learning_rate; Text Encoder LR set to {learning_rate:g}.")
        text_encoder_only = unet_lr == 0 and text_encoder_lr > 0
    else:
        text_encoder_only = False
        for key in ("LoRA_type", "unet_lr", "text_encoder_lr", "network_dim", "network_alpha"):
            consumed.add(key)

    # emulate the kohya_ss LoRA tab: every 0 value was dropped, so sd-scripts used its own default
    if lora:
        for key, fallback in _LORA_TAB_ZERO_FALLBACKS.items():
            if key in out and key in allowed and _number(out[key], 1.0) == 0:
                out[key] = fallback
                notes.append(f"{key} = 0 is dropped by the kohya_ss LoRA tab, so sd-scripts used {fallback:g}; converted to {fallback:g}.")

    # additional parameters
    additional = str(get("additional_parameters") or "")
    out["additional_parameters"] = _fold_additional_parameters(additional, out, notes, allowed)
    if text_encoder_only:
        out["additional_parameters"] = _with_argument(out["additional_parameters"], "--network_train_text_encoder_only")
        notes.append("unet_lr was 0 with a text encoder LR: kohya trained only the text encoder; kept with --network_train_text_encoder_only in Additional sd-scripts Arguments.")
    if not flux and not lora:
        # sdxl_train derives text encoder training from the learning rates (the GUI passed --train_text_encoder for > 0)
        out.setdefault("learning_rate_te1", _number(data.get("learning_rate_te1"), 0.0))
        out.setdefault("learning_rate_te2", _number(data.get("learning_rate_te2"), 0.0))

    # ignore values for keys this model/type does not have, coerce types
    result: dict[str, Any] = {}
    for key, value in out.items():
        if key not in allowed:
            continue
        spec = specs[key]
        if training_type not in spec.modes and key != "training_mode":
            continue
        value = _coerce_to_spec(spec, value, kohya_whole_number(model_key, key))
        if spec.choices and value not in spec.choice_values() and not spec.allow_custom_value:
            notes.append(f"{key} = {value!r} is not a valid choice; keeping the default.")
            continue
        result[key] = value

    consumed.update(_JSON_GUI_ONLY_KEYS & set(data))
    ignored = sorted(key for key in data if key not in consumed and key not in allowed)
    active = [f"{key}={data[key]!r}" for key in ignored if _is_active(data[key])]
    if active:
        notes.append("kohya_ss settings not used by this trainer (ignored): " + ", ".join(active) + ".")
    quiet = len(ignored) - len(active)
    if quiet:
        notes.append(f"{quiet} other kohya_ss keys were empty, off or irrelevant for {_model(model_key)['short_name']} and were ignored.")
    return result, notes


# ----------------------------------------------------------------------------------------------------------------------
# loading presets / runtime TOMLs
# ----------------------------------------------------------------------------------------------------------------------

# sd-scripts runtime-TOML names -> our keys (for reloading a run's config)
_RUNTIME_TO_KEY = {
    "train_data_dir": "parent_folder_path",
    "train_batch_size": "dataset_batch_size",
    "caption_extension": "dataset_caption_extension",
    "enable_bucket": "dataset_enable_bucket",
    "bucket_no_upscale": "dataset_bucket_no_upscale",
}
_PRESET_META_KEYS = ("preset_category", "preset_model", "preset_training_type", "preset_format")

# Runtime TOMLs (this backend and the kohya_ss GUI) leave disabled options out, so a key that is absent there means
# sd-scripts' default (the kohya_ss LoRA tab also drops every 0: its network_alpha 0 ran as 1). Values in our terms;
# launch-only keys (gpu_ids, multi_gpu, dynamo_*) are never in a runtime TOML.
_RUNTIME_ABSENT_VALUES: dict[str, Any] = {
    **{
        key: False
        for key in (
            "dataset_enable_bucket", "dataset_bucket_no_upscale", "shuffle_caption", "flip_aug", "random_crop", "save_state",
            "save_state_on_train_end", "full_bf16", "full_fp16", "gradient_checkpointing", "cpu_offload_checkpointing",
            "fp8_base", "fp8_scaled", "fp8_fast", "fp8_base_unet", "sdpa", "xformers", "highvram",
            "disable_mmap_load_safetensors", "disable_numpy_memmap", "compile", "compile_fullgraph", "sample_at_first",
            "cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk",
            "cache_text_encoder_outputs_on_cpu", "skip_cache_check", "debiased_estimation_loss", "apply_t5_attn_mask",
            "no_half_vae", "dim_from_weights", "persistent_data_loader_workers", "ddp_gradient_as_bucket_view",
            "ddp_static_graph", "log_config", "save_state_to_huggingface", "async_upload", "fused_backward_pass",
            "blockwise_fused_optimizers", "mem_eff_save",
        )
    },
    **{
        key: 0
        for key in (
            "save_every_n_epochs", "save_every_n_steps", "save_last_n_epochs", "save_last_n_steps", "save_last_n_epochs_state",
            "save_last_n_steps_state", "max_train_epochs", "keep_tokens", "caption_dropout_rate", "lr_warmup_steps", "seed",
            "unet_lr", "network_dropout", "scale_weight_norms", "blocks_to_swap", "compile_cache_size_limit",
            "sample_every_n_epochs", "sample_every_n_steps", "ip_noise_gamma", "noise_offset", "min_snr_gamma",
            "min_timestep", "ddp_timeout",
        )
    },
    **{
        key: ""
        for key in (
            "reg_data_dir", "network_weights", "resume", "sample_prompts", "vae", "logging_dir", "log_with", "log_prefix",
            "log_tracker_name", "log_tracker_config", "wandb_run_name", "wandb_api_key", "metadata_title", "metadata_author",
            "metadata_description", "metadata_license", "metadata_tags", "huggingface_repo_id", "huggingface_token",
            "huggingface_repo_type", "huggingface_repo_visibility", "huggingface_path_in_repo", "training_comment",
            "network_args", "lr_scheduler_type", "lr_scheduler_args", "optimizer_args", "resume_from_huggingface",
        )
    },
    "max_grad_norm": 1.0,
    "learning_rate": 2e-06,
    "optimizer_type": "AdamW",
    "lr_scheduler": "constant",
    "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": 1.0,
    "gradient_accumulation_steps": 1,
    "dataset_batch_size": 1,
    "min_bucket_reso": 256,
    "max_bucket_reso": 1024,
    "bucket_reso_steps": 64,
    "mixed_precision": "no",
    "loss_type": "l2",
    "huber_c": 0.1,
    "huber_schedule": "snr",
    "huber_scale": 1.0,
    "max_data_loader_n_workers": 8,
    "sample_sampler": "ddim",
    "t5xxl_lr": 0.0,
    "max_timestep": 1000,
    "max_token_length": 75,
    "dataset_caption_extension": ".caption",
    "prior_loss_weight": 1.0,
    "vae_batch_size": 1,
    "sigmoid_scale": 1.0,
    "guidance_scale": 3.5,
    "discrete_flow_shift": 3.0,
    "t5xxl_max_token_length": 512,
    "model_prediction_type": "sigma_scaled",
    "timestep_sampling": "sigma",
    "compile_dynamic": "auto",
    "train_blocks": "all",
    "train_double_block_indices": "all",
    "train_single_block_indices": "all",
}
# the same for LoRA runs only (a fine-tuning run's TOML has no network keys: the LoRA fields keep their values)
_RUNTIME_ABSENT_LORA_VALUES: dict[str, Any] = {"network_alpha": 1.0}


def _is_runtime_toml(data: dict) -> bool:
    """A training run's config (sd-scripts names, always has 'resolution') rather than a preset of our keys."""
    return "resolution" in data and "preset_format" not in data


def _from_runtime_toml(data: dict, model_key: str, notes: list[str]) -> dict:
    """Map sd-scripts names of a runtime TOML (written by train_kohya_model or the kohya_ss GUI) to our keys."""
    spec_map = _spec_map(model_key)
    out: dict[str, Any] = {}
    runtime = _is_runtime_toml(data)
    if runtime:
        sd_names = {value: key for key, value in _RUNTIME_TO_KEY.items()}
        lora_run = bool(str(data.get("network_module") or "").strip())
        absent = {**_RUNTIME_ABSENT_VALUES, **(_RUNTIME_ABSENT_LORA_VALUES if lora_run else {})}
        for key, value in absent.items():
            if key in spec_map and sd_names.get(key, key) not in data:
                out[key] = value
        if "max_train_steps" not in data:
            out["max_train_steps"] = 0 if data.get("max_train_epochs") else 1600
        if lora_run and "text_encoder_lr" in spec_map and "text_encoder_lr" not in data and not data.get("network_train_unet_only"):
            out["text_encoder_lr"] = _number(data.get("learning_rate"), 2e-06)  # sd-scripts trains the text encoder at learning_rate
        if not lora_run and "learning_rate_te1" in spec_map:
            training_te = bool(data.get("train_text_encoder"))
            for key in ("learning_rate_te1", "learning_rate_te2"):
                if key not in data:
                    out[key] = _number(data.get("learning_rate"), 2e-06) if training_te else 0.0
    for key, value in data.items():
        if key in spec_map:
            out[key] = value
        elif key in _RUNTIME_TO_KEY:
            out[_RUNTIME_TO_KEY[key]] = value
    if runtime and str(data.get("logging_dir") or "").strip() and not str(data.get("log_with") or "").strip():
        out["log_with"] = "tensorboard"  # sd-scripts logs to TensorBoard when only logging_dir is set
        notes.append("logging_dir without log_with: sd-scripts logged this run to TensorBoard; Logging set to tensorboard.")
    if "train_data_dir" in data:
        out["dataset_config_mode"] = FOLDER_DATASET_MODE
    elif str(data.get("dataset_config") or "").strip():
        out["dataset_config_mode"] = TOML_DATASET_MODE
    resolution = data.get("resolution")
    if resolution not in (None, "") and "dataset_resolution_width" not in data:
        parts = [part for part in re.split(r"[,x ]+", str(resolution).strip("()[] ")) if part]
        try:
            out["dataset_resolution_width"] = int(float(parts[0]))
            out["dataset_resolution_height"] = int(float(parts[1] if len(parts) > 1 else parts[0]))
        except (ValueError, IndexError):
            notes.append(f"Could not read resolution '{resolution}'.")
    if out.get("save_precision") == "float":
        out["save_precision"] = "fp32"
    if isinstance(data.get("text_encoder_lr"), list):
        # [CLIP-L, T5-XXL] for FLUX (the second one trains only with train_t5xxl)
        rates = [_number(rate) for rate in data["text_encoder_lr"]]
        out["text_encoder_lr"] = rates[0] if rates else 0.0
        if "t5xxl_lr" in spec_map:
            out["t5xxl_lr"] = rates[1] if len(rates) > 1 and rates[1] != out["text_encoder_lr"] else 0.0
    if "network_train_unet_only" in data and data.get("network_train_unet_only"):
        out["text_encoder_lr"] = 0.0
    if "optimizer" in data and "optimizer_type" not in data:
        out["optimizer_type"] = data["optimizer"]
    network_args = data.get("network_args")
    if isinstance(network_args, list):
        remaining = []
        for item in network_args:
            name, _, value = str(item).partition("=")
            if name in ("train_double_block_indices", "train_single_block_indices", "train_blocks") and name in spec_map:
                out[name] = value
            else:
                remaining.append(str(item))
        out["network_args"] = _join_args(remaining)
    for key in ("optimizer_args", "lr_scheduler_args"):
        if isinstance(data.get(key), list):
            out[key] = _join_args(data[key])
    if "max_token_length" in out:
        out["max_token_length"] = _int(out["max_token_length"], 75) if _int(out["max_token_length"], 75) in (150, 225) else 75
    if "resume_from_huggingface" in data:
        out["resume"], out["resume_from_huggingface"] = _split_resume(data.get("resume"), data["resume_from_huggingface"])
    if "sample_prompts" in out and str(out["sample_prompts"]).replace("\\", "/").endswith("sample/prompt.txt"):
        notes.append(f"sample_prompts points to a prompt file written by a previous run: {out['sample_prompts']}")
    return out


def _detect_json_training_type(data: dict) -> str:
    return LORA if ("LoRA_type" in data or "network_dim" in data) else FULL


def load_kohya_preset(path: str, model_key: str, current: dict) -> tuple[dict, list[str]]:
    """Load one of our TOML presets, a runtime TOML or a raw kohya_ss GUI .json into ``current`` (missing keys keep
    their current value). Model paths are relocated and ``training_mode`` follows the file (Guardrail 2)."""
    _model(model_key)
    notes: list[str] = []
    file_path = clean_path(path)
    if not file_path or not os.path.isfile(file_path):
        raise ValueError(f"Preset file not found: {path or '(empty)'}")
    values = dict(current or {})
    spec_map = _spec_map(model_key)

    if file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{file_path} is not a kohya_ss GUI configuration (expected a JSON object).")
        training_type = _detect_json_training_type(data)
        loaded, convert_notes = convert_kohya_gui_json(data, model_key, training_type)
        notes.extend(convert_notes)
        mode = loaded.get("training_mode", LORA_TRAINING_MODE)
    else:
        with open(file_path, "r", encoding="utf-8-sig") as handle:
            data = toml.load(handle)
        preset_model = data.get("preset_model")
        if preset_model and preset_model != model_key:
            notes.append(f"This preset was made for '{preset_model}', not '{model_key}'.")
        loaded = _from_runtime_toml(data, model_key, notes)
        if "additional_parameters" in spec_map:
            current = loaded.get("additional_parameters", values.get("additional_parameters"))
            if data.get("network_train_text_encoder_only"):
                loaded["additional_parameters"] = _with_argument(current, _TEXT_ENCODER_ONLY)
                notes.append(f"This run trained only the text encoder (network_train_text_encoder_only): kept with {_TEXT_ENCODER_ONLY} in Additional sd-scripts Arguments.")
            elif _is_runtime_toml(data):
                # runtime TOMLs record it as network_train_text_encoder_only: a flag left on screen by an earlier load goes
                loaded["additional_parameters"] = _without_argument(current, _TEXT_ENCODER_ONLY)
        if "training_mode" in data:
            mode = infer_training_mode_from_loaded_config(data)
        elif str(data.get("preset_training_type") or "").strip():
            mode = FULL_FINE_TUNING_MODE if _training_type(data["preset_training_type"]) == FULL else LORA_TRAINING_MODE
        else:
            mode = infer_training_mode_from_loaded_config(data)
        unknown = sorted(
            key
            for key in data
            if key not in spec_map and key not in _RUNTIME_TO_KEY and key not in _PRESET_META_KEYS
            and key not in ("resolution", "network_train_unet_only", "network_train_text_encoder_only", "train_text_encoder", "optimizer", "console_log_simple")
        )
        if unknown:
            notes.append("Keys not used by this trainer (ignored): " + ", ".join(unknown) + ".")

    for key, value in loaded.items():
        if key in spec_map and key != "training_mode":
            values[key] = _coerce_to_spec(spec_map[key], value, kohya_whole_number(model_key, key))
    values["training_mode"] = mode
    relocated = relocate_model_paths(values)
    for key in MODEL_PATH_KEYS:
        if key in values and relocated.get(key) != values.get(key):
            notes.append(f"{key}: {values.get(key)} -> {relocated[key]}")
    return relocated, notes


def kohya_preset_file_name(json_name: str) -> str:
    """'24GB_GPUs_21.61GB_Quality_1_LoRA.json' -> '24_GB_GPUs_21.61_GB_Quality_1.toml' (preset_library naming)."""
    stem = Path(json_name).stem
    stem = re.sub(r"_LoRA(?=_|$)", "", stem)
    stem = re.sub(r"_VRAM(?=_|$)", "", stem)
    stem = re.sub(r"^(\d+(?:\.\d+)?)GB_GPUs?_(\d+(?:\.\d+)?)GB", r"\1_GB_GPUs_\2_GB", stem)
    return stem + ".toml"


def build_kohya_preset_toml(model_key: str, training_type: str, values: dict, description: str, comments: tuple[str, ...] = ()) -> str:
    """Text of a shipped preset: '# description:' line, metadata keys, then every setting (sorted)."""
    training_type = _training_type(training_type)
    keys = [spec.key for spec in _build_specs(model_key)]
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"Preset values missing: {', '.join(missing)}")
    document: dict[str, Any] = {
        "preset_category": "image",
        "preset_model": model_key,
        "preset_training_type": training_type,
        "preset_format": 1,
    }
    for key in sorted(keys):
        value = values[key]
        if value is None:
            raise ValueError(f"Preset value of '{key}' is None (TOML cannot store it).")
        document[key] = value
    header = [f"# description: {description}"] + [f"# {line}" for line in comments]
    return "\n".join(header) + "\n" + toml.dumps(document)


# ----------------------------------------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------------------------------------

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl")


def _find_accelerate_launch(python_cmd: str) -> list[str]:
    """Same lookup as modern_image_lora_gui._find_accelerate_launch."""
    import shutil

    python_dir = os.path.dirname(python_cmd)
    fallback = os.path.join(python_dir, "accelerate.exe" if sys.platform == "win32" else "accelerate")
    if os.path.isfile(fallback):
        return [fallback, "launch"]
    accelerate_path = shutil.which("accelerate")
    if accelerate_path:
        return [accelerate_path, "launch"]
    return [python_cmd, "-m", "accelerate.commands.launch"]


def _folder_repeats(name: str) -> int:
    """Repeats of a '<repeats>_<name>' folder exactly as sd-scripts reads them (config_util.extract_dreambooth_params:
    int() of the text before the first '_'; it skips folders below 1), so '10' and '1_' train too."""
    try:
        return int(name.split("_")[0])
    except ValueError:
        return 0


def _kohya_subfolders(folder: str) -> tuple[list[str], int]:
    """Valid '<repeats>_<name>' sub-folders and their image count."""
    valid, images = [], 0
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isdir(path):
            continue
        if _folder_repeats(name) >= 1:
            valid.append(name)
            images += sum(1 for item in os.listdir(path) if item.lower().endswith(_IMAGE_EXTENSIONS))
    return valid, images


def _validate_kohya_folder(folder: str, label: str, errors: list[str]) -> None:
    if not folder:
        errors.append(f"{label} is empty.")
        return
    if not os.path.isdir(folder):
        errors.append(f"{label} does not exist: {folder}")
        return
    subfolders, images = _kohya_subfolders(folder)
    if not subfolders:
        errors.append(
            f"{label} has no kohya '<repeats>_<name>' sub-folder (e.g. '1_ohwx man' holding the images): {folder}"
        )
    elif images == 0:
        errors.append(f"{label} has no images in its sub-folders ({', '.join(subfolders)}): {folder}")


def _write_prompt_file(text: str, output_dir: str) -> str:
    path = os.path.join(output_dir, "sample", "prompt.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.strip() + "\n")
    return _fs(path)


def _log_with_from_arguments(tokens: list[str]) -> tuple[str, list[str]]:
    """Take --log_with out of the extra CLI arguments (it must reach the TOML so logging_dir is set, Guardrail 1)."""
    log_with, rest, index = "", [], 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--log_with="):
            log_with = token.split("=", 1)[1].strip()
        elif token == "--log_with" and index + 1 < len(tokens):
            log_with = tokens[index + 1].strip()
            index += 1
        else:
            rest.append(token)
        index += 1
    return log_with, rest


def _looks_like_prompt_file(text: str) -> bool:
    """Single-line Sample Prompts text naming a .txt file (C:/x/prompts.txt, prompts.txt), not a prompt that ends in
    '.txt' (a photo of ohwx man reading notes.txt)."""
    text = strip_quotes(text.strip())
    if "\n" in text or not text.lower().endswith(".txt") or " --" in text:
        return False
    if not re.search(r"\s", text) or re.match(r"^(?:[A-Za-z]:|[\\/~]|\.{1,2}[\\/])", text):
        return True
    return bool(re.search(r"[\\/]", text)) and "," not in text


# OFT modules use network_alpha as the limit of the rotation: alpha 0 trains nothing (oft.py / oft_flux.py)
_OFT_MODULES = ("networks.oft", "networks.oft_flux")


def _validated_values(model_key: str, values: dict) -> tuple[dict, str, list[str]]:
    """Merge with defaults, resolve paths and validate. Returns (values, training type, warnings)."""
    model = _model(model_key)
    flux = model_key == "flux1_dev"
    try:
        mode = normalize_training_mode(values.get("training_mode"))
    except ValueError as exc:
        raise ValueError(f"Training mode must be '{LORA_TRAINING_MODE}' or '{FULL_FINE_TUNING_MODE}'.") from exc
    training_type = FULL if mode == FULL_FINE_TUNING_MODE else LORA
    v = kohya_defaults(model_key, training_type)
    spec_map = _spec_map(model_key)
    errors: list[str] = []
    warnings: list[str] = []
    for key, value in (values or {}).items():
        spec = spec_map.get(key)
        if spec is None:
            continue
        whole = kohya_whole_number(model_key, key)
        v[key] = _coerce_to_spec(spec, value, whole)
        if spec.kind not in ("number", "slider") or training_type not in spec.modes or isinstance(value, bool):
            continue
        # in the words of the field check, so Start refuses these on its own too (and the UI lists them once)
        try:
            parse_number(value, label=spec.label or key, whole=whole)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if whole and isinstance(v[key], float) and not v[key].is_integer():
            errors.append(f"{spec.label or key} must be a whole number (got {value}).")
    v["training_mode"] = mode

    # paths
    for key in ("output_dir", "parent_folder_path", "reg_data_dir", "dataset_config", "network_weights", "log_tracker_config"):
        v[key] = _resolve_path(v.get(key))
    resume = _clean_text(v.get("resume"))
    v["resume"] = _resolve_path(resume) if resume else ""
    for key in ("pretrained_model_name_or_path", "ae", "clip_l", "t5xxl", "vae"):
        if key in v:
            text = _clean_text(v[key])
            v[key] = text if (not text or _looks_like_hf_repo_id(text)) else _resolve_path(text)
    extra_arguments = _split_args(v["additional_parameters"])

    if not v["output_dir"]:
        errors.append("Output Folder is empty.")
    output_name = _clean_text(v.get("output_name"))
    if not output_name:
        errors.append("Output Name is empty.")
    elif re.search(r'[<>:"/\\|?*]', output_name):
        errors.append(f"Output Name contains characters that are not allowed in file names: {output_name}")
    v["output_name"] = output_name

    # model files
    base = v["pretrained_model_name_or_path"]
    if not base:
        errors.append(f"{model['short_name']} base model path is empty.")
    elif flux and not os.path.isfile(base):
        errors.append(f"FLUX.1 model file not found: {base} (download {model['model_files']['pretrained_model_name_or_path']} into {model['model_dir']}).")
    elif not flux and not (os.path.exists(base) or _looks_like_hf_repo_id(base)):
        errors.append(f"SDXL model not found: {base} (download {model['model_files']['pretrained_model_name_or_path']} into {model['model_dir']}).")
    if flux:
        for key, label in (("ae", "FLUX VAE (ae)"), ("clip_l", "CLIP-L"), ("t5xxl", "T5-XXL")):
            if not v[key]:
                errors.append(f"{label} path is empty.")
            elif not os.path.isfile(v[key]):
                errors.append(f"{label} file not found: {v[key]} (expected {model['model_files'][key]} in {model['model_dir']}).")
    elif v.get("vae") and not (os.path.exists(v["vae"]) or _looks_like_hf_repo_id(v["vae"])):
        errors.append(f"SDXL VAE not found: {v['vae']} (use a .safetensors file, a Diffusers folder, a Hugging Face id or leave it empty).")

    # files the run reads later: the kohya_ss GUI checked them before starting, sd-scripts fails only after loading the
    # models (train_network.py: resume, network_weights; the tracker init)
    hf_resume = str(v.get("resume_from_huggingface") or "").strip()
    if v["resume"] and hf_resume:
        errors.append("Set either Resume From State Folder or Resume From Hugging Face, not both.")
    elif v["resume"] and not os.path.isdir(v["resume"]):
        errors.append(f"Resume From State Folder not found: {v['resume']} (a saved training state folder, e.g. <output>/<name>-000025-state).")
    elif hf_resume and not re.match(r"^[^/\\:\s]+/[^/\\:\s]+", hf_resume):
        errors.append(f"Resume From Hugging Face must be repo_id/path_in_repo[:revision[:repo_type]] (e.g. me/my-lora/state:main:model), got '{hf_resume}'.")
    if training_type == LORA and v["network_weights"] and not os.path.isfile(v["network_weights"]):
        errors.append(f"Continue From LoRA file not found: {v['network_weights']}")
    if training_type == LORA and v["dim_from_weights"] and not v["network_weights"]:
        errors.append("Rank From LoRA File needs a Continue From LoRA file.")
    if v["log_tracker_config"] and not os.path.isfile(v["log_tracker_config"]):
        errors.append(f"Tracker Config file not found: {v['log_tracker_config']}")

    # dataset
    if v["dataset_config_mode"] == TOML_DATASET_MODE:
        if not v["dataset_config"]:
            errors.append("Dataset TOML File is empty (or switch Dataset Source to the kohya folder structure).")
        elif not os.path.isfile(v["dataset_config"]):
            errors.append(f"Dataset TOML File not found: {v['dataset_config']}")
    else:
        _validate_kohya_folder(v["parent_folder_path"], "Training Images Folder", errors)
        if v["reg_data_dir"]:
            _validate_kohya_folder(v["reg_data_dir"], "Regularization Images Folder", errors)
    for key, label in (("dataset_resolution_width", "Resolution Width"), ("dataset_resolution_height", "Resolution Height")):
        size = _int(v.get(key), 0)
        if size < 64 or size % 32:
            errors.append(f"{label} must be a positive multiple of 32 (64 recommended), got {v.get(key)}.")
    if _int(v.get("dataset_batch_size"), 0) < 1:
        errors.append("Batch Size must be at least 1.")
    if _int(v.get("bucket_reso_steps"), 0) < 1:
        errors.append("Bucket Resolution Steps must be greater than 0.")

    # length and learning rates
    if _int(v["max_train_epochs"]) <= 0 and _int(v["max_train_steps"]) <= 0:
        errors.append("Set Max Training Epochs or Max Training Steps above 0.")
    lr = _number(v["learning_rate"])
    if training_type == LORA:
        text_encoder_lr = _number(v["text_encoder_lr"])
        if lr <= 0 and _number(v["unet_lr"]) <= 0 and text_encoder_lr <= 0:
            errors.append("Set a learning rate above 0.")
        if _int(v["network_dim"]) < 1:
            errors.append("Network Rank (Dim) must be at least 1.")
        if _number(v["network_alpha"]) < 0:
            errors.append("Network Alpha cannot be negative (0 = alpha equals the rank, no scaling).")
        elif _number(v["network_alpha"]) == 0 and str(v["network_module"]).strip() in _OFT_MODULES:
            errors.append("Network Alpha 0 trains nothing with OFT: networks.oft / networks.oft_flux use alpha as the limit of the rotation. Use a small value such as 1e-3.")
        if _TEXT_ENCODER_ONLY in extra_arguments and text_encoder_lr <= 0:
            errors.append(f"{_TEXT_ENCODER_ONLY} trains only the text encoder: set Text Encoder LR above 0 (at 0 nothing would be trained).")
        if flux and _number(v["t5xxl_lr"]) > 0 and text_encoder_lr <= 0:
            errors.append("T5-XXL LR needs a Text Encoder LR above 0 (as in the kohya_ss GUI: at 0 no text encoder is trained).")
    elif lr <= 0:
        errors.append("Set a learning rate above 0.")
    if _number(v["lr_warmup_steps"]) < 0:
        errors.append("LR Warmup Steps cannot be negative.")
    if _int(v["gradient_accumulation_steps"]) < 1:
        errors.append("Gradient Accumulation Steps must be at least 1.")

    # precision
    mixed = str(v["mixed_precision"] or "no")
    if v["full_bf16"] and v["full_fp16"]:
        errors.append("Choose either Full BF16 or Full FP16, not both.")
    if v["full_bf16"] and mixed != "bf16":
        errors.append("Full BF16 Training needs Mixed Precision 'bf16'.")
    if v["full_fp16"] and mixed != "fp16":
        errors.append("Full FP16 Training needs Mixed Precision 'fp16'.")

    # memory
    swap = _int(v["blocks_to_swap"])
    if swap < 0 or swap > model["max_blocks_to_swap"]:
        errors.append(f"Blocks to Swap must be between 0 and {model['max_blocks_to_swap']} for {model['short_name']}.")
    if flux and swap > 0 and v.get("cpu_offload_checkpointing"):
        errors.append("CPU Offload Checkpointing cannot be combined with Blocks to Swap; set one of them to off/0.")

    # argument lists: sd-scripts splits every optimizer / scheduler argument at '=' and reads the value with
    # ast.literal_eval, network arguments at the first '=' (the kohya_ss GUI refused malformed ones: validate_args_setting)
    errors += _argument_problems("Optimizer Arguments", v["optimizer_args"], literal=True)
    errors += _argument_problems("Scheduler Arguments", v["lr_scheduler_args"], literal=True)
    if training_type == LORA:
        errors += _argument_problems("Network Arguments", v["network_args"], literal=False)

    # optimizer / scheduler: sd-scripts builds the 'adafactor' scheduler only for Adafactor with relative_step, and then
    # picks it itself (with relative_step=False the bare name fails when the scheduler is created)
    optimizer = str(v["optimizer_type"] or "").strip()
    scheduler = str(v["lr_scheduler"] or "").strip()
    custom_scheduler = str(v["lr_scheduler_type"] or "").strip()
    relative_step = optimizer.casefold() == "adafactor" and _adafactor_relative_step(v["optimizer_args"])
    if scheduler.casefold().startswith("adafactor") and not custom_scheduler:
        if optimizer.casefold() != "adafactor":
            errors.append("The 'adafactor' LR Scheduler works only with the Adafactor optimizer; choose another LR Scheduler (the presets use constant).")
        elif not relative_step:
            errors.append("The 'adafactor' LR Scheduler needs relative_step=True in Optimizer Arguments; with relative_step=False choose another LR Scheduler (the presets use constant).")
    # sd-scripts creates these schedulers without a warmup and raises "... does not require num_warmup_steps" (after the
    # models are loaded); the kohya_ss GUI set the warmup to 0 for constant
    if _number(v["lr_warmup_steps"]) > 0:
        if custom_scheduler:
            errors.append("LR Warmup Steps must be 0 with a Custom Scheduler Class (sd-scripts gives it no warmup).")
        elif relative_step:
            errors.append("LR Warmup Steps must be 0 with Adafactor's relative_step (sd-scripts then runs its adafactor scheduler, which has no warmup): set relative_step=False in Optimizer Arguments, as the presets do, or the warmup to 0.")
        elif scheduler == "constant":
            errors.append("LR Warmup Steps must be 0 with the constant LR Scheduler; choose constant_with_warmup for a warmup.")
    if not custom_scheduler and not relative_step:
        options = _argument_options(v["lr_scheduler_args"])
        if scheduler == "cosine_with_min_lr":
            ratio = any(token.split("=", 1)[0] == "--lr_scheduler_min_lr_ratio" for token in extra_arguments)
            if "min_lr_rate" in options:
                errors.append("cosine_with_min_lr: use min_lr=… in Scheduler Arguments or --lr_scheduler_min_lr_ratio in Additional sd-scripts Arguments; min_lr_rate clashes with the one sd-scripts passes itself.")
            elif ("min_lr" in options) == ratio:
                errors.append("The cosine_with_min_lr LR Scheduler needs one minimum: min_lr=1e-6 in Scheduler Arguments or --lr_scheduler_min_lr_ratio 0.1 in Additional sd-scripts Arguments.")
        elif scheduler == "piecewise_constant":
            if "step_rules" not in options:
                errors.append("The piecewise_constant LR Scheduler needs step_rules in Scheduler Arguments, e.g. step_rules='1:10,0.1:20,0.01' (LR multiplier:steps pairs, then the last multiplier).")
            elif not isinstance(_literal_value(options["step_rules"], ""), str):
                errors.append("step_rules must be text in quotes, e.g. step_rules='1:10,0.1:20,0.01'.")

    # full fine-tuning
    if training_type == FULL and v.get("fused_backward_pass"):
        if optimizer.casefold() != "adafactor":
            errors.append("Fused Backward Pass works only with the Adafactor optimizer.")
        if _int(v["gradient_accumulation_steps"]) != 1:
            errors.append("Fused Backward Pass needs Gradient Accumulation Steps = 1.")
        if flux and v.get("blockwise_fused_optimizers"):
            errors.append("Choose either Fused Backward Pass or Blockwise Fused Optimizers, not both.")

    # caching compatibility (sd-scripts asserts these after loading the models)
    cache_te = bool(v["cache_text_encoder_outputs"] or v["cache_text_encoder_outputs_to_disk"])
    if cache_te and (v["shuffle_caption"] or _number(v["caption_dropout_rate"]) > 0):
        errors.append("Caption shuffle / caption dropout cannot be used with Cache Text Encoder Outputs.")
    if v["cache_latents"] and v["random_crop"]:
        errors.append("Random Crop cannot be used with Cache Latents.")
    if not flux and cache_te:
        if training_type == LORA and _number(v["text_encoder_lr"]) > 0:
            errors.append("SDXL: Cache Text Encoder Outputs needs Text Encoder LR = 0 (U-Net only LoRA).")
        if training_type == FULL and (_number(v["learning_rate_te1"]) > 0 or _number(v["learning_rate_te2"]) > 0):
            errors.append("SDXL: text encoders cannot be trained while their outputs are cached; set both text encoder LRs to 0 or turn caching off.")
    # networks.lora_flux reads train_t5xxl as the text "True"; flux_train_network.py then refuses cached outputs
    if flux and training_type == LORA and cache_te and _argument_options(v["network_args"], raw_values=True).get("train_t5xxl") == "True":
        errors.append("train_t5xxl=True in Network Arguments trains T5-XXL, so its outputs cannot be cached: turn Cache Text Encoder Outputs (and to Disk) off.")

    # samples
    sampling = _int(v["sample_every_n_epochs"]) > 0 or _int(v["sample_every_n_steps"]) > 0 or bool(v["sample_at_first"])
    prompts = str(v.get("sample_prompts") or "").strip()
    if sampling and not prompts:
        errors.append("Sample generation is on but Sample Prompts is empty.")
    if prompts:
        resolved = _resolve_path(prompts) if "\n" not in prompts else ""
        if resolved and os.path.isfile(resolved):
            v["sample_prompts"] = resolved
        elif _looks_like_prompt_file(prompts):
            errors.append(f"Sample prompts file not found: {prompts}")

    # hardware / logging
    if _int(v["num_processes"]) < 1 or _int(v["num_machines"]) < 1:
        errors.append("Number of Processes and Number of Machines must be at least 1.")
    gpu_ids = str(v.get("gpu_ids") or "").replace(" ", "")
    if gpu_ids and not re.fullmatch(r"\d+(,\d+)*", gpu_ids):
        errors.append(f"GPU IDs must be comma-separated numbers (e.g. 0 or 0,1), got '{v.get('gpu_ids')}'.")
    v["gpu_ids"] = gpu_ids
    if str(v.get("log_with") or "") not in ("", "tensorboard", "wandb", "all"):
        errors.append("Logging must be empty, tensorboard, wandb or all.")

    if errors:
        raise ValueError(f"{model['short_name']} {'LoRA' if training_type == LORA else 'fine-tuning'} settings need attention:\n- " + "\n- ".join(errors))
    if training_type == LORA and flux and _number(v["text_encoder_lr"]) > 0 and v["cache_text_encoder_outputs"]:
        warnings.append("CLIP-L LoRA is trained (Text Encoder LR > 0): only T5-XXL outputs are cached.")
    return v, training_type, warnings


def kohya_validation_problems(model_key: str, values: dict) -> list[str]:
    """What Start would refuse in these settings, one problem per item (empty: they pass). Reads, never writes, files."""
    try:
        _validated_values(model_key, values)
    except ValueError as exc:
        problems = [line[2:] for line in str(exc).splitlines() if line.startswith("- ")]
        return problems or [str(exc)]
    return []


def _build_sd_scripts_config(model_key: str, v: dict, training_type: str) -> list[tuple[str, object]]:
    """Runtime TOML entries in sd-scripts names (kohya_ss GUI mapping); disabled options are left out."""
    flux = model_key == "flux1_dev"
    lora = training_type == LORA
    cfg: dict[str, object] = {"training_mode": v["training_mode"]}

    def put(key, value, when=True):
        if when and value is not None:
            cfg[key] = value

    # dataset
    if v["dataset_config_mode"] == TOML_DATASET_MODE:
        put("dataset_config", v["dataset_config"])
    else:
        put("train_data_dir", v["parent_folder_path"])
        put("reg_data_dir", v["reg_data_dir"], bool(v["reg_data_dir"]))
    put("resolution", f"{_int(v['dataset_resolution_width'])},{_int(v['dataset_resolution_height'])}")
    put("train_batch_size", _int(v["dataset_batch_size"], 1))
    extension = str(v["dataset_caption_extension"] or "").strip()
    put("caption_extension", extension, bool(extension))
    put("enable_bucket", True, bool(v["dataset_enable_bucket"]))
    put("bucket_no_upscale", True, bool(v["dataset_bucket_no_upscale"]))
    put("min_bucket_reso", _int(v["min_bucket_reso"], 256))
    put("max_bucket_reso", _int(v["max_bucket_reso"], 2048))
    put("bucket_reso_steps", _int(v["bucket_reso_steps"], 64))
    put("shuffle_caption", True, bool(v["shuffle_caption"]))
    put("keep_tokens", _int(v["keep_tokens"]), _int(v["keep_tokens"]) > 0)
    put("caption_dropout_rate", _number(v["caption_dropout_rate"]), _number(v["caption_dropout_rate"]) > 0)
    put("flip_aug", True, bool(v["flip_aug"]))
    put("random_crop", True, bool(v["random_crop"]))
    put("prior_loss_weight", _number(v["prior_loss_weight"], 1.0))
    if not flux and _int(v.get("max_token_length"), 75) in (150, 225):
        put("max_token_length", _int(v["max_token_length"]))

    # output
    put("output_dir", v["output_dir"])
    put("output_name", v["output_name"])
    put("save_model_as", v["save_model_as"] or "safetensors")
    put("save_precision", {"fp32": "float"}.get(v["save_precision"], v["save_precision"]))
    for key in ("save_every_n_epochs", "save_every_n_steps", "save_last_n_epochs", "save_last_n_steps", "save_last_n_epochs_state", "save_last_n_steps_state"):
        put(key, _int(v[key]), _int(v[key]) > 0)
    put("save_state", True, bool(v["save_state"]))
    put("save_state_on_train_end", True, bool(v["save_state_on_train_end"]))
    put("resume", v["resume"], bool(v["resume"]))

    # training
    put("max_train_epochs", _int(v["max_train_epochs"]), _int(v["max_train_epochs"]) > 0)
    put("max_train_steps", _int(v["max_train_steps"]), _int(v["max_train_steps"]) > 0)
    put("learning_rate", _number(v["learning_rate"]))
    put("optimizer_type", str(v["optimizer_type"]).strip())
    optimizer_args = _split_key_value_args(v["optimizer_args"])
    put("optimizer_args", optimizer_args, bool(optimizer_args))
    put("lr_scheduler", v["lr_scheduler"])
    warmup = _number(v["lr_warmup_steps"])
    put("lr_warmup_steps", int(warmup) if warmup >= 1 else warmup, warmup > 0)
    put("lr_scheduler_num_cycles", max(_int(v["lr_scheduler_num_cycles"], 1), 1))
    put("lr_scheduler_power", _number(v["lr_scheduler_power"], 1.0))
    put("lr_scheduler_type", str(v["lr_scheduler_type"]).strip(), bool(str(v["lr_scheduler_type"]).strip()))
    scheduler_args = _split_key_value_args(v["lr_scheduler_args"])
    put("lr_scheduler_args", scheduler_args, bool(scheduler_args))
    put("max_grad_norm", _number(v["max_grad_norm"]))  # always written: sd-scripts' default is 1.0
    put("seed", _int(v["seed"]), _int(v["seed"]) > 0)
    put("gradient_accumulation_steps", max(_int(v["gradient_accumulation_steps"], 1), 1))

    # LoRA network
    if lora:
        text_encoder_lr = _number(v["text_encoder_lr"])
        t5xxl_lr = _number(v.get("t5xxl_lr"))  # FLUX: [CLIP-L, T5-XXL] like the kohya_ss GUI (T5 needs train_t5xxl)
        put("unet_lr", _number(v["unet_lr"]), _number(v["unet_lr"]) > 0)
        put("text_encoder_lr", [text_encoder_lr, t5xxl_lr if t5xxl_lr > 0 else text_encoder_lr], text_encoder_lr > 0)
        put("network_train_unet_only", True, text_encoder_lr <= 0)
        # written here instead of the command line, so reloading this run's TOML keeps it (as with the kohya_ss GUI)
        put("network_train_text_encoder_only", True, _TEXT_ENCODER_ONLY in _split_args(v["additional_parameters"]))
        put("network_module", str(v["network_module"]).strip())
        put("network_dim", _int(v["network_dim"]))
        put("network_alpha", _number(v["network_alpha"]))
        put("network_dropout", _number(v["network_dropout"]), _number(v["network_dropout"]) > 0)
        network_args = _split_key_value_args(v["network_args"], raw_values=True)
        if flux:
            names = {arg.partition("=")[0] for arg in network_args}
            for key in ("train_double_block_indices", "train_single_block_indices"):
                value = str(v.get(key) or "").strip()
                if value and value.casefold() != "all" and key not in names:
                    network_args.append(f"{key}={value}")
            if str(v.get("train_blocks") or "all") != "all" and "train_blocks" not in names:
                network_args.append(f"train_blocks={v['train_blocks']}")
        put("network_args", network_args, bool(network_args))
        put("network_weights", v["network_weights"], bool(v["network_weights"]))
        put("dim_from_weights", True, bool(v["dim_from_weights"]))
        put("scale_weight_norms", _number(v["scale_weight_norms"]), _number(v["scale_weight_norms"]) > 0)
        put("training_comment", v["training_comment"], bool(str(v["training_comment"]).strip()))
    else:
        put("fused_backward_pass", True, bool(v["fused_backward_pass"]))
        if flux:
            put("blockwise_fused_optimizers", True, bool(v["blockwise_fused_optimizers"]))
            put("mem_eff_save", True, bool(v["mem_eff_save"]))
        else:
            te1, te2 = _number(v["learning_rate_te1"]), _number(v["learning_rate_te2"])
            if te1 > 0 or te2 > 0:
                put("train_text_encoder", True)
                put("learning_rate_te1", te1)
                put("learning_rate_te2", te2)

    # model files and precision
    put("pretrained_model_name_or_path", v["pretrained_model_name_or_path"])
    if flux:
        put("ae", v["ae"])
        put("clip_l", v["clip_l"])
        put("t5xxl", v["t5xxl"])
    else:
        put("vae", v["vae"], bool(v["vae"]))
        put("no_half_vae", True, bool(v["no_half_vae"]))
    put("mixed_precision", v["mixed_precision"])
    put("full_bf16", True, bool(v["full_bf16"]))
    put("full_fp16", True, bool(v["full_fp16"]))

    # memory and speed
    put("blocks_to_swap", _int(v["blocks_to_swap"]), _int(v["blocks_to_swap"]) > 0)
    put("gradient_checkpointing", True, bool(v["gradient_checkpointing"]))
    if flux:
        put("cpu_offload_checkpointing", True, bool(v["cpu_offload_checkpointing"]))
        put("disable_numpy_memmap", True, bool(v["disable_numpy_memmap"]))
        if lora:
            for key in ("fp8_base", "fp8_scaled", "fp8_fast", "fp8_base_unet"):
                put(key, True, bool(v[key]))
    else:
        if lora:
            put("fp8_base", True, bool(v["fp8_base"]))
        put("sdpa", True, bool(v["sdpa"]))
        put("xformers", True, bool(v["xformers"]) and not bool(v["sdpa"]))
    put("highvram", True, bool(v["highvram"]))
    put("disable_mmap_load_safetensors", True, bool(v["disable_mmap_load_safetensors"]))
    if v["compile"]:
        put("compile", True)
        put("compile_backend", v["compile_backend"])
        put("compile_mode", v["compile_mode"])
        put("compile_dynamic", v["compile_dynamic"], str(v["compile_dynamic"]) != "auto")
        put("compile_fullgraph", True, bool(v["compile_fullgraph"]))
        put("compile_cache_size_limit", _int(v["compile_cache_size_limit"]), _int(v["compile_cache_size_limit"]) > 0)

    # samples
    put("sample_prompts", v["sample_prompts"], bool(str(v["sample_prompts"]).strip()))
    put("sample_every_n_epochs", _int(v["sample_every_n_epochs"]), _int(v["sample_every_n_epochs"]) > 0)
    put("sample_every_n_steps", _int(v["sample_every_n_steps"]), _int(v["sample_every_n_steps"]) > 0)
    put("sample_at_first", True, bool(v["sample_at_first"]))
    if not flux:
        put("sample_sampler", v["sample_sampler"])

    # caching
    put("cache_latents", True, bool(v["cache_latents"]))
    put("cache_latents_to_disk", True, bool(v["cache_latents_to_disk"]))
    put("cache_text_encoder_outputs", True, bool(v["cache_text_encoder_outputs"]))
    put("cache_text_encoder_outputs_to_disk", True, bool(v["cache_text_encoder_outputs_to_disk"]))
    if flux:
        put("cache_text_encoder_outputs_on_cpu", True, bool(v["cache_text_encoder_outputs_on_cpu"]))
    put("vae_batch_size", max(_int(v["vae_batch_size"], 1), 1))
    put("skip_cache_check", True, bool(v["skip_cache_check"]))

    # advanced training
    if flux:
        put("timestep_sampling", v["timestep_sampling"])
        put("discrete_flow_shift", _number(v["discrete_flow_shift"], 3.0))
        put("sigmoid_scale", _number(v["sigmoid_scale"], 1.0))
        put("model_prediction_type", v["model_prediction_type"])
        put("guidance_scale", _number(v["guidance_scale"], 1.0))
        put("apply_t5_attn_mask", True, bool(v["apply_t5_attn_mask"]))
        put("t5xxl_max_token_length", _int(v["t5xxl_max_token_length"], 512))
    else:
        put("noise_offset", _number(v["noise_offset"]), _number(v["noise_offset"]) > 0)
        put("min_snr_gamma", _number(v["min_snr_gamma"]), _number(v["min_snr_gamma"]) > 0)
        put("debiased_estimation_loss", True, bool(v["debiased_estimation_loss"]))
        put("min_timestep", _int(v["min_timestep"]), _int(v["min_timestep"]) > 0)
        put("max_timestep", _int(v["max_timestep"]), 0 < _int(v["max_timestep"]) < 1000)
    put("loss_type", v["loss_type"])
    put("huber_c", _number(v["huber_c"], 0.1))
    put("huber_schedule", v["huber_schedule"])
    put("huber_scale", _number(v["huber_scale"], 1.0))
    put("ip_noise_gamma", _number(v["ip_noise_gamma"]), _number(v["ip_noise_gamma"]) > 0)

    # data loading / distributed
    put("max_data_loader_n_workers", max(_int(v["max_data_loader_n_workers"]), 0))
    put("persistent_data_loader_workers", True, bool(v["persistent_data_loader_workers"]) and _int(v["max_data_loader_n_workers"]) > 0)
    put("ddp_timeout", _int(v["ddp_timeout"]), _int(v["ddp_timeout"]) > 0)
    put("ddp_gradient_as_bucket_view", True, bool(v["ddp_gradient_as_bucket_view"]))
    put("ddp_static_graph", True, bool(v["ddp_static_graph"]))

    # console: plain log lines, without Rich's time / level / file:line columns and line wrapping, so the training
    # status panel reads checkpoint and sample lines exactly
    put("console_log_simple", True)

    # logging (normalised by SaveConfigFileToRun: no empty/root logging_dir, output_dir/logs when on)
    log_with = str(v["log_with"] or "").strip()
    put("log_with", log_with, bool(log_with))
    put("logging_dir", str(v["logging_dir"] or "").strip(), bool(log_with) and bool(str(v["logging_dir"] or "").strip()))
    if log_with:
        put("log_prefix", v["log_prefix"], bool(str(v["log_prefix"]).strip()))
        put("log_tracker_name", v["log_tracker_name"], bool(str(v["log_tracker_name"]).strip()))
        put("log_tracker_config", v["log_tracker_config"], bool(v["log_tracker_config"]))
        put("log_config", True, bool(v["log_config"]))
        if log_with in ("wandb", "all"):
            put("wandb_run_name", str(v["wandb_run_name"]).strip() or v["output_name"])
            put("wandb_api_key", v["wandb_api_key"], bool(str(v["wandb_api_key"]).strip()))

    # metadata / Hugging Face
    for key in ("metadata_title", "metadata_author", "metadata_description", "metadata_license", "metadata_tags"):
        put(key, str(v[key]).strip(), bool(str(v[key]).strip()))
    for key in ("huggingface_repo_id", "huggingface_token", "huggingface_repo_type", "huggingface_repo_visibility", "huggingface_path_in_repo"):
        put(key, str(v[key]).strip(), bool(str(v[key]).strip()))
    put("save_state_to_huggingface", True, bool(v["save_state_to_huggingface"]))
    hf_resume = str(v["resume_from_huggingface"]).strip()
    if hf_resume:
        # sd-scripts: --resume holds the Hugging Face location, --resume_from_huggingface is an on/off flag
        put("resume", hf_resume, True)
        put("resume_from_huggingface", True, True)
    put("async_upload", True, bool(v["async_upload"]))
    return list(cfg.items())


def _command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if (" " in part or not part) else part for part in command)


def train_kohya_model(model_key: str, values: dict, *, print_only: bool, runner=None) -> str:
    """Validate, write the runtime TOML into output_dir and build ``accelerate launch <sd-scripts script> --config_file``.

    ``print_only``: return the command and the TOML text without launching. Otherwise call
    ``runner.execute_command(run_cmd=<list>, env=<dict>)`` (non-blocking) and return a short status line.
    """
    model = _model(model_key)
    v, training_type, warnings = _validated_values(model_key, values)
    output_dir = v["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # sample prompt text typed into the field (kohya_ss GUI behaviour): write it to <output>/sample/prompt.txt
    prompts = str(v["sample_prompts"] or "").strip()
    if prompts and not os.path.isfile(prompts):
        v["sample_prompts"] = _write_prompt_file(prompts, output_dir)

    extra_tokens = _split_args(v["additional_parameters"])
    cli_log_with, extra_tokens = _log_with_from_arguments(extra_tokens)
    if training_type == LORA:  # in the runtime TOML as network_train_text_encoder_only
        extra_tokens = [token for token in extra_tokens if token != _TEXT_ENCODER_ONLY]
    if cli_log_with and not str(v["log_with"] or "").strip():
        v["log_with"] = cli_log_with

    env = None
    if not print_only:
        if runner is None or not hasattr(runner, "execute_command"):
            raise ValueError("A runner with execute_command(run_cmd, **kwargs) is required to start training.")
        env = setup_environment(compile_requested=requires_native_compile_toolchain(v))
        if v["compile"] and env.get("MUSUBI_TORCH_COMPILE_READY") == "0":
            v["compile"] = False
            warnings.append(f"torch.compile is off for this run: {env.get('MUSUBI_TORCH_COMPILE_DETAIL', 'toolchain unavailable')}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_path = _fs(os.path.join(output_dir, f"{v['output_name']}_{timestamp}.toml"))
    SaveConfigFileToRun(
        parameters=_build_sd_scripts_config(model_key, v, training_type),
        file_path=config_path,
        exclusion=[],
    )

    python_cmd = sys.executable
    command = _find_accelerate_launch(python_cmd)
    command = AccelerateLaunch.run_cmd(
        run_cmd=command,
        dynamo_backend=v["dynamo_backend"],
        dynamo_mode=v["dynamo_mode"],
        dynamo_use_fullgraph=v["dynamo_use_fullgraph"],
        dynamo_use_dynamic=v["dynamo_use_dynamic"],
        num_processes=v["num_processes"],
        num_machines=v["num_machines"],
        multi_gpu=v["multi_gpu"],
        gpu_ids=v["gpu_ids"],
        main_process_port=v["main_process_port"],
        num_cpu_threads_per_process=v["num_cpu_threads_per_process"],
        mixed_precision=v["mixed_precision"],
        extra_accelerate_launch_args=v["extra_accelerate_launch_args"],
    )
    script = _fs(SD_SCRIPTS_DIR / model["train_scripts"][training_type])
    command += [script, "--config_file", config_path] + extra_tokens

    with open(config_path, "r", encoding="utf-8") as handle:
        toml_text = handle.read()
    kind = "LoRA" if training_type == LORA else "full fine-tuning"
    if print_only:
        lines = [f"# {model['short_name']} {kind}: kohya sd-scripts {model['train_scripts'][training_type]}", _command_text(command), "", f"# runtime config: {config_path}", toml_text]
        lines += [f"# note: {warning}" for warning in warnings]
        log.info(f"Print-only {model['short_name']} {kind} command: {_command_text(command)} (config: {config_path})")
        return "\n".join(lines)

    log.info(f"Starting {model['short_name']} {kind}: {_command_text(command)}")
    runner.execute_command(run_cmd=command, env=env)
    status = f"{model['short_name']} {kind} started with {model['train_scripts'][training_type]} (config: {config_path})."
    if warnings:
        status += " " + " ".join(warnings)
    return status


# ----------------------------------------------------------------------------------------------------------------------
# hints shown next to the model
# ----------------------------------------------------------------------------------------------------------------------

KOHYA_HINTS: dict[str, dict[str, list[str]]] = {
    "flux1_dev": {
        LORA: [
            "Dataset: 15-30 sharp images of the subject in one '1_ohwx man'-style folder (1 repeat); the presets train 200 epochs and save every 25, then pick the best checkpoint.",
            "Captions: '.txt' next to each image, or none to use the folder name ('ohwx man') as the caption for every image.",
            "Pick the preset of your GPU tier: Quality 1 keeps the full bf16 model with block swap; FP8 Scaled presets fit the same tier with fewer swapped blocks (faster steps) for a small quality loss.",
            "Learning rate 5e-5 with Adafactor (fixed LR, constant schedule) at batch 1; the 4x GPU presets use 1e-4 and also 200 epochs, spread over the 4 GPUs.",
            "Rank 128 / alpha 128 in the 10 GB+ presets; 8 GB presets drop to rank 64 at 768 px and 6 GB to rank 48 at 512 px.",
            "Text Encoder LR 5e-5 also trains a CLIP-L LoRA; T5-XXL outputs are cached, so caption shuffle/dropout must stay off.",
            "Block swap moves FLUX blocks to RAM: more blocks = less VRAM, slower steps. Keep 64 GB+ system RAM for the low-VRAM presets.",
            "Keep training images at the preset resolution (buckets are off): crop/resize them to 1024x1024 beforehand.",
        ],
        FULL: [
            "Full fine-tuning (DreamBooth) writes a complete 23.8 GB FLUX.1 dev checkpoint per save: plan disk space (200 epochs, save every 25 = 8 files).",
            "Learning rate 4e-6 with Adafactor and fused backward pass in every preset; the batch-7 48 GB preset uses 1e-5.",
            "Fused backward pass needs Adafactor and gradient accumulation 1; it keeps optimizer state small enough for 24 GB with block swap 10.",
            "10-12 GB GPUs use block swap 31-35 and need a lot of system RAM (64 GB+ recommended) for the swapped blocks.",
            "Dataset: 15-30 images in a '1_ohwx man' folder; images without captions use the folder name. Text encoders are not trained.",
            "Extract a LoRA from the fine-tuned model afterwards if you need a small file; fine-tuning usually gives the best likeness.",
        ],
    },
    "sdxl": {
        LORA: [
            "No measured SDXL LoRA presets exist yet: the defaults follow the SDXL fine-tuning presets (Adafactor, constant LR, fp32 SDXL VAE, noise offset 0.0357) at rank 64 / alpha 32.",
            "Learning rate 1e-4 for the U-Net with Adafactor; lower it to 5e-5 if samples overcook. Text Encoder LR 0 lets the text encoder outputs be cached.",
            "Dataset: 20-40 images at 1024x1024 in '<repeats>_ohwx man' folders; 100 epochs at 1 repeat ≈ 2000-4000 steps.",
            "Use a good SDXL base such as RealVis v5 for photos; keep No Half VAE on with the original SDXL VAE.",
            "12 GB GPUs: SDPA + gradient checkpointing is enough; 8 GB: add block swap or fp8 base.",
        ],
        FULL: [
            "These DreamBooth presets train 1 epoch over '<repeats>_ohwx man' folders (e.g. 150 repeats) plus a regularization folder of class images ('1_man'); regularization doubles the steps.",
            "Quality 1 (10 GB+) also trains text encoder 1 at 3e-6 (U-Net 8e-6); the 6-8 GB presets freeze both text encoders and cache their outputs.",
            "Captions: the presets use the folder name as caption (caption extension '.caption'); set '.txt' to use your own caption files.",
            "Checkpoints are full SDXL models (~6.9 GB in bf16) saved every 451 steps; pick the best one with an XY plot.",
            "10-12 GB GPUs swap 50-68 of SDXL's 70 transformer blocks to RAM; 16 GB swaps 20; 24 GB+ needs no swap.",
            "Max gradient norm 0 (no clipping) and noise offset 0.0357 are part of the tuned presets.",
        ],
    },
}
