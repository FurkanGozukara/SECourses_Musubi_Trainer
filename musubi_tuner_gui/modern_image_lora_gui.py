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
from functools import partial

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
    SaveConfigFile,
    SaveConfigFileToRun,
    get_file_path,
    get_file_path_or_save_as,
    get_folder_path,
    print_command_and_toml,
    run_cmd_advanced_training,
    save_executed_script,
    scriptdir,
    setup_environment,
    validate_block_swap_options,
)
from .custom_logging import setup_logging
from .dataset_config_generator import (
    generate_dataset_config_from_folders,
    save_dataset_config,
    validate_dataset_config,
)

log = setup_logging()


@dataclass(frozen=True)
class ModernImageArchitecture:
    key: str
    display_name: str
    slug: str
    network_module: str
    train_script: str
    latent_cache_script: str
    text_cache_script: str
    max_blocks_to_swap: int
    model_folder: str
    model_filenames: dict[str, str]

    @property
    def is_ideogram(self) -> bool:
        return self.key == "ideogram4"

    @property
    def is_krea(self) -> bool:
        return self.key == "krea2"


ARCHITECTURES = {
    "ideogram4": ModernImageArchitecture(
        key="ideogram4",
        display_name="Ideogram 4",
        slug="ideogram4",
        network_module="networks.lora_ideogram4",
        train_script="ideogram4_train_network.py",
        latent_cache_script="ideogram4_cache_latents.py",
        text_cache_script="ideogram4_cache_text_encoder_outputs.py",
        max_blocks_to_swap=33,
        model_folder="Training_Models_Ideogram_4",
        model_filenames={
            "dit": "ideogram4_fp8_scaled.safetensors",
            "vae": "flux2-vae.safetensors",
            "text_encoder": "qwen3vl_8b_fp8_scaled.safetensors",
            "unconditional_dit": "ideogram4_unconditional_fp8_scaled.safetensors",
        },
    ),
    "krea2": ModernImageArchitecture(
        key="krea2",
        display_name="Krea 2",
        slug="krea2",
        network_module="networks.lora_krea2",
        train_script="krea2_train_network.py",
        latent_cache_script="krea2_cache_latents.py",
        text_cache_script="krea2_cache_text_encoder_outputs.py",
        max_blocks_to_swap=26,
        model_folder="Training_Models_Krea_2",
        model_filenames={
            "dit": "Krea_2_Raw_Base.safetensors",
            "vae": "qwen_image_vae.safetensors",
            "text_encoder": "qwen3vl_4b_bf16.safetensors",
            "turbo_dit": "Krea_2_Turbo.safetensors",
        },
    ),
}


MODERN_IMAGE_PARAM_KEYS = [
    # Accelerate
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
    # Advanced
    "additional_parameters",
    "debug_mode",
    # Dataset
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
    # Model
    "dit",
    "vae",
    "text_encoder",
    "dit_dtype",
    "vae_dtype",
    "unconditional_dit",
    "use_unconditional_dit_for_lora_sampling",
    "turbo_dit",
    "turbo_dit_cache",
    "fp8_base",
    "fp8_scaled",
    "disable_numpy_memmap",
    "blocks_to_swap",
    "use_pinned_memory_for_block_swap",
    "block_swap_h2d_only",
    "block_swap_ring_size",
    # Compile
    "compile",
    "compile_backend",
    "compile_mode",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_cache_size_limit",
    # Architecture-specific and schedule
    "sampler_preset",
    "initial_sigma",
    "validate_caption_structure",
    "warn_on_caption_issues",
    "log_loss_stats",
    "timestep_sampling",
    "weighting_scheme",
    "discrete_flow_shift",
    "sigmoid_scale",
    "min_timestep",
    "max_timestep",
    # Training
    "sdpa",
    "flash_attn",
    "sage_attn",
    "xformers",
    "split_attn",
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
    # Sampling
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
    # Caching
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
    "caching_teo_text_cache_dtype",
    "caching_teo_text_encoder_dtype",
    # Optimizer
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
    # Network
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
    # Save
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
    # Metadata
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "metadata_reso",
    "metadata_arch",
    # Hugging Face
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "save_state_to_huggingface",
    "resume_from_huggingface",
    "async_upload",
]


@dataclass
class ModernImageWorkflow:
    parameters: list[tuple[str, object]]
    config_path: str
    latent_cache_command: list[str] | None
    text_cache_command: list[str] | None
    train_command: list[str]

    @property
    def commands(self) -> list[list[str]]:
        commands = []
        if self.latent_cache_command:
            commands.append(self.latent_cache_command)
        if self.text_cache_command:
            commands.append(self.text_cache_command)
        commands.append(self.train_command)
        return commands


executors: dict[str, CommandExecutor] = {}
train_state_values = {key: time.time() for key in ARCHITECTURES}


def get_architecture(key: str) -> ModernImageArchitecture:
    try:
        return ARCHITECTURES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown modern image architecture: {key}") from exc


def _trainer_script_path(filename: str) -> str:
    return os.path.normpath(os.path.join(scriptdir, "musubi-tuner", "src", "musubi_tuner", filename))


def _find_accelerate_launch(python_cmd: str) -> list[str]:
    python_dir = os.path.dirname(python_cmd)
    fallback = os.path.join(python_dir, "accelerate.exe" if sys.platform == "win32" else "accelerate")
    if os.path.isfile(fallback):
        return [fallback, "launch"]
    accelerate_path = shutil.which("accelerate")
    if accelerate_path:
        return [accelerate_path, "launch"]
    return [python_cmd, "-m", "accelerate.commands.launch"]


def _coerce_cli_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        return [part.strip('"\'') for part in shlex.split(text, posix=False) if part.strip('"\'')]
    except ValueError:
        return text.replace(",", " ").split()


def _coerce_float_list(value: object) -> list[float]:
    output = []
    for item in _coerce_cli_list(value):
        try:
            output.append(float(item))
        except (TypeError, ValueError):
            log.warning("Ignoring invalid base weight multiplier: %s", item)
    return output


def _resolve_device(value: object, gpu_ids: object) -> str:
    device = str(value or "cuda").strip()
    if device == "cuda":
        first_gpu = str(gpu_ids or "").split(",")[0].strip()
        return f"cuda:{first_gpu}" if first_gpu else "cuda:0"
    return device


def _require_file(param_dict: dict, key: str, label: str, required: bool = True) -> str:
    value = str(param_dict.get(key) or "").strip()
    if not value:
        if required:
            raise ValueError(f"{label} is required.")
        return ""
    if not os.path.isfile(value):
        raise ValueError(f"{label} does not exist: {value}")
    return os.path.abspath(value)


def _replace_parameter(parameters: list[tuple[str, object]], key: str, value: object) -> list[tuple[str, object]]:
    return [(name, value if name == key else current) for name, current in parameters]


def _normalize_modern_parameters(
    spec: ModernImageArchitecture,
    parameters: list[tuple[str, object]],
    *,
    validate_paths: bool = True,
) -> tuple[dict, list[tuple[str, object]]]:
    param_dict = dict(parameters)
    param_dict["network_module"] = spec.network_module
    parameters = _replace_parameter(parameters, "network_module", spec.network_module)

    save_precision = str(param_dict.get("save_precision") or "bf16").lower()
    if save_precision == "float":
        save_precision = "fp32"
    if save_precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("save_precision must be bf16, fp16, or fp32.")
    param_dict["save_precision"] = save_precision
    parameters = _replace_parameter(parameters, "save_precision", save_precision)

    mixed_precision = str(param_dict.get("mixed_precision") or "bf16")
    if param_dict.get("full_bf16") and mixed_precision != "bf16":
        raise ValueError("Full BF16 training requires mixed_precision='bf16'.")
    if param_dict.get("full_fp16") and mixed_precision != "fp16":
        raise ValueError("Full FP16 training requires mixed_precision='fp16'.")
    if param_dict.get("full_bf16") and param_dict.get("full_fp16"):
        raise ValueError("Full BF16 and Full FP16 cannot both be enabled.")

    attention_flags = [name for name in ("sdpa", "flash_attn", "xformers") if bool(param_dict.get(name))]
    if bool(param_dict.get("sage_attn")):
        raise ValueError(f"SageAttention is not available for {spec.display_name} training in this backend.")
    if not attention_flags:
        param_dict["sdpa"] = True
        parameters = _replace_parameter(parameters, "sdpa", True)
    elif len(attention_flags) > 1:
        raise ValueError("Select exactly one attention backend: SDPA, FlashAttention, or xformers.")

    try:
        blocks_to_swap = int(param_dict.get("blocks_to_swap", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("blocks_to_swap must be an integer.") from exc
    if not 0 <= blocks_to_swap <= spec.max_blocks_to_swap:
        raise ValueError(f"blocks_to_swap for {spec.display_name} must be between 0 and {spec.max_blocks_to_swap}.")
    param_dict["blocks_to_swap"] = blocks_to_swap
    parameters = _replace_parameter(parameters, "blocks_to_swap", blocks_to_swap)
    validate_block_swap_options(param_dict, lora_training=True)

    for key in (
        "caching_latent_batch_size",
        "caching_latent_num_workers",
        "caching_teo_batch_size",
        "caching_teo_num_workers",
    ):
        value = param_dict.get(key)
        if value is not None and int(value) < 1:
            raise ValueError(f"{key} must be at least 1.")

    if spec.is_ideogram:
        if str(param_dict.get("weighting_scheme") or "none") != "none":
            raise ValueError("Ideogram 4 supports weighting_scheme='none' only.")
        param_dict["fp8_base"] = False
        param_dict["fp8_scaled"] = False
        parameters = _replace_parameter(_replace_parameter(parameters, "fp8_base", False), "fp8_scaled", False)
        if param_dict.get("warn_on_caption_issues") and not param_dict.get("validate_caption_structure"):
            param_dict["warn_on_caption_issues"] = False
            parameters = _replace_parameter(parameters, "warn_on_caption_issues", False)
        if param_dict.get("use_unconditional_dit_for_lora_sampling") and not str(
            param_dict.get("unconditional_dit") or ""
        ).strip():
            raise ValueError("Official asymmetric sampling requires an unconditional Ideogram 4 DiT.")
    else:
        param_dict["dit_dtype"] = "bfloat16"
        parameters = _replace_parameter(parameters, "dit_dtype", "bfloat16")
        fp8_base = bool(param_dict.get("fp8_base"))
        fp8_scaled = bool(param_dict.get("fp8_scaled"))
        if fp8_base != fp8_scaled:
            raise ValueError("Krea 2 requires fp8_base and fp8_scaled to be enabled together.")
        turbo_dit = str(param_dict.get("turbo_dit") or "").strip()
        if param_dict.get("turbo_dit_cache") and not turbo_dit:
            raise ValueError("turbo_dit_cache requires a Krea 2 Turbo DiT path.")
        if turbo_dit and blocks_to_swap:
            raise ValueError("Krea 2 Turbo sample generation cannot be combined with blocks_to_swap.")

    base_weights = _coerce_cli_list(param_dict.get("base_weights"))
    base_weight_multipliers = _coerce_float_list(param_dict.get("base_weights_multiplier"))
    param_dict["base_weights"] = base_weights or None
    param_dict["base_weights_multiplier"] = base_weight_multipliers or None
    parameters = _replace_parameter(parameters, "base_weights", base_weights or None)
    parameters = _replace_parameter(parameters, "base_weights_multiplier", base_weight_multipliers or None)

    if not validate_paths:
        return param_dict, parameters

    dataset_config = str(param_dict.get("dataset_config") or "").strip()
    if not dataset_config or not os.path.isfile(dataset_config):
        raise ValueError(f"Dataset config does not exist: {dataset_config or '(empty)'}")
    dataset_config = os.path.abspath(dataset_config)
    param_dict["dataset_config"] = dataset_config
    parameters = _replace_parameter(parameters, "dataset_config", dataset_config)

    dit_path = _require_file(param_dict, "dit", f"{spec.display_name} DiT")
    vae_path = _require_file(param_dict, "vae", f"{spec.display_name} VAE")
    text_encoder_path = _require_file(param_dict, "text_encoder", f"{spec.display_name} text encoder")
    for key, value in (("dit", dit_path), ("vae", vae_path), ("text_encoder", text_encoder_path)):
        param_dict[key] = value
        parameters = _replace_parameter(parameters, key, value)

    sample_prompts = str(param_dict.get("sample_prompts") or "").strip()
    if sample_prompts:
        if not os.path.isfile(sample_prompts):
            raise ValueError(f"Sample prompt file does not exist: {sample_prompts}")
        sample_prompts = os.path.abspath(sample_prompts)
        param_dict["sample_prompts"] = sample_prompts
        parameters = _replace_parameter(parameters, "sample_prompts", sample_prompts)

    if spec.is_ideogram:
        unconditional = _require_file(
            param_dict,
            "unconditional_dit",
            "Ideogram 4 unconditional DiT",
            required=False,
        )
        if unconditional:
            param_dict["unconditional_dit"] = unconditional
            parameters = _replace_parameter(parameters, "unconditional_dit", unconditional)
    else:
        turbo = _require_file(param_dict, "turbo_dit", "Krea 2 Turbo DiT", required=False)
        if turbo:
            param_dict["turbo_dit"] = turbo
            parameters = _replace_parameter(parameters, "turbo_dit", turbo)

    output_dir = str(param_dict.get("output_dir") or "").strip()
    output_name = str(param_dict.get("output_name") or "").strip()
    if not output_dir:
        raise ValueError("Output directory is required.")
    if not output_name:
        raise ValueError("Output name is required.")
    output_dir = os.path.abspath(output_dir)
    param_dict["output_dir"] = output_dir
    parameters = _replace_parameter(parameters, "output_dir", output_dir)

    return param_dict, parameters


def build_modern_cache_commands(
    spec: ModernImageArchitecture,
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
            "--vae",
            str(param_dict["vae"]),
            "--device",
            _resolve_device(param_dict.get("caching_latent_device"), gpu_ids),
        ]
        if spec.is_ideogram and param_dict.get("vae_dtype"):
            latent_command.extend(["--vae_dtype", str(param_dict["vae_dtype"])])
        for key, flag in (
            ("caching_latent_batch_size", "--batch_size"),
            ("caching_latent_num_workers", "--num_workers"),
        ):
            value = param_dict.get(key)
            if value is not None:
                latent_command.extend([flag, str(int(value))])
        if param_dict.get("caching_latent_skip_existing"):
            latent_command.append("--skip_existing")
        if param_dict.get("caching_latent_keep_cache"):
            latent_command.append("--keep_cache")

    text_command = None
    if bool(param_dict.get("cache_text_encoder_outputs", True)):
        text_command = [
            python_cmd,
            _trainer_script_path(spec.text_cache_script),
            "--dataset_config",
            dataset_config,
            "--text_encoder",
            str(param_dict["text_encoder"]),
            "--device",
            _resolve_device(param_dict.get("caching_teo_device"), gpu_ids),
        ]
        for key, flag in (
            ("caching_teo_batch_size", "--batch_size"),
            ("caching_teo_num_workers", "--num_workers"),
        ):
            value = param_dict.get(key)
            if value is not None:
                text_command.extend([flag, str(int(value))])
        if param_dict.get("caching_teo_skip_existing"):
            text_command.append("--skip_existing")
        if param_dict.get("caching_teo_keep_cache"):
            text_command.append("--keep_cache")
        if spec.is_ideogram:
            text_command.extend(
                ["--text_cache_dtype", str(param_dict.get("caching_teo_text_cache_dtype") or "bf16")]
            )
            if param_dict.get("validate_caption_structure"):
                text_command.append("--validate_caption_structure")
            if param_dict.get("warn_on_caption_issues"):
                text_command.append("--warn_on_caption_issues")
        elif param_dict.get("caching_teo_text_encoder_dtype"):
            text_command.extend(
                ["--text_encoder_dtype", str(param_dict["caching_teo_text_encoder_dtype"])]
            )

    return latent_command, text_command


def _run_config_exclusions(spec: ModernImageArchitecture, parameters: list[tuple[str, object]]) -> list[str]:
    exclusions = {
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
        "sample_output_dir",
        "disable_prompt_enhancement",
        "sample_width",
        "sample_height",
        "sample_steps",
        "sample_seed",
        "sample_negative_prompt",
        "sample_cfg_scale",
        "mem_eff_save",
    }
    exclusions.update(
        key
        for key, _ in parameters
        if key.startswith("caching_") or key in {"cache_latents", "cache_text_encoder_outputs"}
    )
    if spec.is_ideogram:
        exclusions.update({"turbo_dit", "turbo_dit_cache", "fp8_base", "fp8_scaled"})
    else:
        exclusions.update(
            {
                # Krea 2 currently fixes DiT compute to bfloat16 in its trainer.
                "dit_dtype",
                "unconditional_dit",
                "use_unconditional_dit_for_lora_sampling",
                "sampler_preset",
                "initial_sigma",
                "validate_caption_structure",
                "warn_on_caption_issues",
                "log_loss_stats",
            }
        )
    return sorted(exclusions)


def _debug_parameters(debug_mode: object) -> str:
    return {
        "Show Timesteps (Image)": "--show_timesteps image",
        "Show Timesteps (Console)": "--show_timesteps console",
        "RCM Debug Save": "--rcm_debug_save",
        "Enable Logging (TensorBoard)": "--log_with tensorboard",
        "Enable Logging (WandB)": "--log_with wandb",
        "Enable Logging (All)": "--log_with all",
    }.get(str(debug_mode or ""), "")


def prepare_modern_image_workflow(
    spec_key: str,
    parameters: list[tuple[str, object]],
    config_path: str,
    *,
    validate_paths: bool = True,
    python_cmd: str | None = None,
) -> ModernImageWorkflow:
    spec = get_architecture(spec_key)
    param_dict, parameters = _normalize_modern_parameters(spec, parameters, validate_paths=validate_paths)
    python_cmd = python_cmd or sys.executable

    if validate_paths:
        os.makedirs(str(param_dict["output_dir"]), exist_ok=True)

    SaveConfigFileToRun(
        parameters=parameters,
        file_path=config_path,
        exclusion=_run_config_exclusions(spec, parameters),
        mandatory_keys=["dataset_config", "dit", "vae", "network_module", "output_dir", "output_name"],
    )

    latent_command, text_command = build_modern_cache_commands(spec, param_dict, python_cmd=python_cmd)

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
    train_command.extend([_trainer_script_path(spec.train_script), "--config_file", config_path])

    additional = str(param_dict.get("additional_parameters") or "").strip()
    debug = _debug_parameters(param_dict.get("debug_mode"))
    if debug:
        additional = f"{additional} {debug}".strip()
    train_command = run_cmd_advanced_training(run_cmd=train_command, additional_parameters=additional)

    return ModernImageWorkflow(
        parameters=parameters,
        config_path=config_path,
        latent_cache_command=latent_command,
        text_cache_command=text_command,
        train_command=train_command,
    )


def _create_dataset_config(
    spec: ModernImageArchitecture,
    *,
    parent_folder_path: object,
    resolution_width: object,
    resolution_height: object,
    caption_extension: object,
    create_missing_captions: object,
    caption_strategy: object,
    batch_size: object,
    enable_bucket: object,
    bucket_no_upscale: object,
    cache_directory: object,
    output_dir: object,
) -> tuple[str, list[str]]:
    parent = str(parent_folder_path or "").strip()
    if not parent or not os.path.isdir(parent):
        raise ValueError(f"Dataset parent folder does not exist: {parent or '(empty)'}")
    save_dir = str(output_dir or "").strip() or parent
    os.makedirs(save_dir, exist_ok=True)

    config, messages = generate_dataset_config_from_folders(
        parent_folder=parent,
        resolution=(int(resolution_width), int(resolution_height)),
        caption_extension=str(caption_extension or ".txt"),
        create_missing_captions=bool(create_missing_captions),
        caption_strategy=str(caption_strategy or "folder_name"),
        batch_size=int(batch_size),
        enable_bucket=bool(enable_bucket),
        bucket_no_upscale=bool(bucket_no_upscale),
        cache_directory_name=str(cache_directory or "cache_dir"),
        control_directory_name=f"__{spec.slug}_no_control__",
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(save_dir, f"{spec.slug}_dataset_{timestamp}.toml")
    save_dataset_config(config, path)
    valid, validation_messages = validate_dataset_config(path)
    messages.extend(validation_messages)
    if not valid:
        raise ValueError("\n".join(messages))
    messages.append(f"Saved dataset TOML: {path}")
    return os.path.abspath(path), messages


def generate_modern_dataset_toml(
    spec_key: str,
    parent_folder_path,
    dataset_resolution_width,
    dataset_resolution_height,
    dataset_caption_extension,
    create_missing_captions,
    caption_strategy,
    dataset_batch_size,
    dataset_enable_bucket,
    dataset_bucket_no_upscale,
    dataset_cache_directory,
    output_dir,
):
    spec = get_architecture(spec_key)
    try:
        path, messages = _create_dataset_config(
            spec,
            parent_folder_path=parent_folder_path,
            resolution_width=dataset_resolution_width,
            resolution_height=dataset_resolution_height,
            caption_extension=dataset_caption_extension,
            create_missing_captions=create_missing_captions,
            caption_strategy=caption_strategy,
            batch_size=dataset_batch_size,
            enable_bucket=dataset_enable_bucket,
            bucket_no_upscale=dataset_bucket_no_upscale,
            cache_directory=dataset_cache_directory,
            output_dir=output_dir,
        )
        return path, path, "\n".join(messages)
    except Exception as exc:
        message = f"Failed to generate dataset TOML: {exc}"
        log.error(message)
        return "", "", message


def _enhance_sample_prompts(
    spec: ModernImageArchitecture,
    param_dict: dict,
    parameters: list[tuple[str, object]],
) -> tuple[dict, list[tuple[str, object]]]:
    prompt_path = str(param_dict.get("sample_prompts") or "").strip()
    if not prompt_path or bool(param_dict.get("disable_prompt_enhancement")):
        return param_dict, parameters
    if not os.path.isfile(prompt_path):
        return param_dict, parameters

    width = int(param_dict.get("sample_width") or 1024)
    height = int(param_dict.get("sample_height") or 1024)
    steps = int(param_dict.get("sample_steps") or (20 if spec.is_ideogram else 28))
    seed = int(param_dict.get("sample_seed") if param_dict.get("sample_seed") is not None else 42)
    negative = str(param_dict.get("sample_negative_prompt") or "").strip()
    cfg_scale = float(param_dict.get("sample_cfg_scale") or (7.0 if spec.is_ideogram else 5.5))

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
            enhanced = re.sub(
                r"(?<!\S)-(?P<flag>w|h|s|d|l|n)(?=\s|=)",
                r"--\g<flag>",
                stripped,
                flags=re.IGNORECASE,
            )
            if not has_flag(enhanced, "w"):
                enhanced += f" --w {width}"
            if not has_flag(enhanced, "h"):
                enhanced += f" --h {height}"
            if not spec.is_ideogram:
                if not has_flag(enhanced, "s"):
                    enhanced += f" --s {steps}"
                if not has_flag(enhanced, "l"):
                    enhanced += f" --l {cfg_scale}"
            if not has_flag(enhanced, "d"):
                enhanced += f" --d {seed}"
            if negative and not has_flag(enhanced, "n"):
                enhanced += f" --n {negative}"
            enhanced_lines.append(enhanced)

    sample_dir = str(param_dict.get("sample_output_dir") or "").strip()
    output_dir = sample_dir or str(param_dict.get("output_dir") or "").strip()
    if not output_dir:
        return param_dict, parameters
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_name = str(param_dict.get("output_name") or spec.slug)
    enhanced_path = os.path.join(output_dir, f"{output_name}_enhanced_prompts_{timestamp}.txt")
    with open(enhanced_path, "w", encoding="utf-8") as handle:
        handle.write("# Generated by SECourses Musubi Trainer\n")
        for line in enhanced_lines:
            handle.write(f"{line}\n")

    enhanced_path = os.path.abspath(enhanced_path)
    param_dict["sample_prompts"] = enhanced_path
    parameters = _replace_parameter(parameters, "sample_prompts", enhanced_path)
    return param_dict, parameters


def save_modern_configuration(
    spec_key: str,
    save_as: bool,
    file_path: str,
    parameters: list[tuple[str, object]],
):
    spec = get_architecture(spec_key)
    original_path = file_path
    if save_as or not file_path:
        file_path = get_file_path_or_save_as(
            file_path,
            default_extension=".toml",
            extension_name="TOML files",
        )
    if not file_path:
        return original_path, gr.update(value="No file selected.", visible=True)
    if not file_path.lower().endswith(".toml"):
        file_path = os.path.splitext(file_path)[0] + ".toml"

    try:
        _, parameters = _normalize_modern_parameters(spec, parameters, validate_paths=False)
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


def _config_value_for_component(key: str, value: object, default: object, spec: ModernImageArchitecture):
    if key == "network_module":
        return spec.network_module
    if key == "dit_dtype" and spec.is_krea:
        return "bfloat16"
    if key == "save_precision" and value in (None, ""):
        return "bf16"
    if isinstance(value, list):
        if key in {"optimizer_args", "lr_scheduler_args", "network_args"}:
            return " ".join(str(item) for item in value)
        if key in {"base_weights", "base_weights_multiplier"}:
            return " ".join(str(item) for item in value)
        if not isinstance(default, list):
            return value[0] if value else default
    return value


def open_modern_configuration(
    spec_key: str,
    ask_for_file: bool,
    file_path: str,
    parameters: list[tuple[str, object]],
):
    spec = get_architecture(spec_key)
    original_path = file_path
    if ask_for_file:
        file_path = get_file_path_or_save_as(
            file_path,
            default_extension=".toml",
            extension_name="TOML files",
        )
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
        with open(file_path, "r", encoding="utf-8") as handle:
            data = toml.load(handle)
        values = []
        for key, default in parameters:
            value = data.get(key, default)
            values.append(_config_value_for_component(key, value, default, spec))
        message = f"Configuration loaded: {os.path.basename(file_path)}"
        gr.Info(message)
        return tuple([file_path, gr.update(value=message, visible=True)] + values)
    except Exception as exc:
        message = f"Failed to load configuration: {exc}"
        log.exception(message)
        gr.Error(message)
        return tuple([original_path, gr.update(value=message, visible=True)] + [value for _, value in parameters])


def _command_line(command: list[str]) -> str:
    if platform.system() == "Windows":
        return subprocess.list2cmdline([str(item) for item in command])
    return shlex.join([str(item) for item in command])


def _build_workflow_script(spec: ModernImageArchitecture, workflow: ModernImageWorkflow) -> tuple[str, str]:
    if platform.system() == "Windows":
        lines = ["@echo off", "setlocal"]
        for index, command in enumerate(workflow.commands):
            label = "training" if index == len(workflow.commands) - 1 else "caching"
            lines.append(f"echo Starting {spec.display_name} {label}...")
            lines.append(_command_line(command))
            lines.append("if errorlevel 1 exit /b %errorlevel%")
        suffix = ".bat"
    else:
        lines = ["#!/bin/bash", "set -e"]
        for index, command in enumerate(workflow.commands):
            label = "training" if index == len(workflow.commands) - 1 else "caching"
            lines.append(f'echo "Starting {spec.display_name} {label}..."')
            lines.append(_command_line(command))
        suffix = ".sh"

    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as handle:
        handle.write(content)
        path = handle.name
    if platform.system() != "Windows":
        os.chmod(path, os.stat(path).st_mode | 0o100)
    return path, content


def train_modern_image_model(
    spec_key: str,
    headless: bool,
    print_only: bool,
    parameters: list[tuple[str, object]],
):
    spec = get_architecture(spec_key)
    param_dict = dict(parameters)

    if str(param_dict.get("dataset_config_mode") or "") == "Generate from Folder Structure":
        generated = str(param_dict.get("generated_toml_path") or "").strip()
        if not generated or not os.path.isfile(generated):
            generated, _ = _create_dataset_config(
                spec,
                parent_folder_path=param_dict.get("parent_folder_path"),
                resolution_width=param_dict.get("dataset_resolution_width"),
                resolution_height=param_dict.get("dataset_resolution_height"),
                caption_extension=param_dict.get("dataset_caption_extension"),
                create_missing_captions=param_dict.get("create_missing_captions"),
                caption_strategy=param_dict.get("caption_strategy"),
                batch_size=param_dict.get("dataset_batch_size"),
                enable_bucket=param_dict.get("dataset_enable_bucket"),
                bucket_no_upscale=param_dict.get("dataset_bucket_no_upscale"),
                cache_directory=param_dict.get("dataset_cache_directory"),
                output_dir=param_dict.get("output_dir"),
            )
        param_dict["dataset_config"] = generated
        parameters = _replace_parameter(parameters, "dataset_config", generated)
        parameters = _replace_parameter(parameters, "generated_toml_path", generated)

    param_dict, parameters = _enhance_sample_prompts(spec, param_dict, parameters)

    raw_output_dir = str(param_dict.get("output_dir") or "").strip()
    if not raw_output_dir:
        raise ValueError("Output directory is required.")
    output_dir = os.path.abspath(raw_output_dir)
    output_name = str(param_dict.get("output_name") or spec.slug)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_path = os.path.join(output_dir, f"{output_name}_{timestamp}.toml")
    workflow = prepare_modern_image_workflow(spec_key, parameters, config_path)

    if print_only:
        for command in workflow.commands[:-1]:
            print_command_and_toml(command, "")
        print_command_and_toml(workflow.train_command, workflow.config_path)
        return None

    executor = executors.get(spec_key)
    if executor is None:
        raise RuntimeError(f"{spec.display_name} command executor is not initialized.")

    script_path, script_content = _build_workflow_script(spec, workflow)
    save_executed_script(
        script_content=script_content,
        config_name=output_name,
        script_type=spec.slug,
    )
    env = setup_environment()
    command = [script_path] if platform.system() == "Windows" else ["bash", script_path]
    executor.execute_command(
        run_cmd=command,
        env=env,
        shell=platform.system() == "Windows",
    )

    train_state_values[spec_key] = time.time()
    return (
        gr.Button(visible=False or headless),
        gr.Row(visible=True),
        gr.Button(interactive=True),
        gr.Textbox(value=f"{spec.display_name} workflow in progress..."),
        gr.Textbox(value=train_state_values[spec_key]),
    )


def modern_image_gui_actions(
    spec_key: str,
    action: str,
    ask_for_file: bool,
    config_file_name: str,
    headless: bool,
    print_only: bool,
    *args,
):
    parameters = list(zip(MODERN_IMAGE_PARAM_KEYS, args))
    if action == "open_configuration":
        return open_modern_configuration(spec_key, ask_for_file, config_file_name, parameters)
    if action == "save_configuration":
        return save_modern_configuration(spec_key, ask_for_file, config_file_name, parameters)
    if action == "train_model":
        return train_modern_image_model(spec_key, headless, print_only, parameters)
    raise ValueError(f"Unknown GUI action: {action}")


def _default_model_paths(spec: ModernImageArchitecture) -> dict[str, str]:
    model_dir = os.path.normpath(os.path.join(scriptdir, "..", spec.model_folder))
    defaults = {}
    for key, filename in spec.model_filenames.items():
        candidate = os.path.join(model_dir, filename)
        defaults[key] = candidate if os.path.isfile(candidate) else ""
    return defaults


def _architecture_defaults(spec: ModernImageArchitecture) -> dict[str, object]:
    defaults = {
        "mixed_precision": "bf16",
        "num_cpu_threads_per_process": 1,
        "num_processes": 1,
        "num_machines": 1,
        "multi_gpu": False,
        "gpu_ids": "",
        "main_process_port": 0,
        "dynamo_backend": "no",
        "dynamo_mode": "",
        "dynamo_use_fullgraph": False,
        "dynamo_use_dynamic": False,
        "extra_accelerate_launch_args": "",
        "dataset_config_mode": "Generate from Folder Structure",
        "dataset_resolution_width": 1024,
        "dataset_resolution_height": 1024,
        "dataset_caption_extension": ".txt",
        "create_missing_captions": True,
        "caption_strategy": "folder_name",
        "dataset_batch_size": 1,
        "dataset_enable_bucket": True,
        "dataset_bucket_no_upscale": False,
        "dataset_cache_directory": "cache_dir",
        "dit_dtype": "bfloat16",
        "vae_dtype": "bfloat16",
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
        "weighting_scheme": "none",
        "sigmoid_scale": 1.0,
        "min_timestep": 0,
        "max_timestep": 1000,
        "sdpa": True,
        "flash_attn": False,
        "sage_attn": False,
        "xformers": False,
        "split_attn": False,
        "max_train_steps": 1600,
        "max_data_loader_n_workers": 2,
        "persistent_data_loader_workers": True,
        "seed": 42,
        "gradient_checkpointing": True,
        "gradient_accumulation_steps": 1,
        "optimizer_type": "adamw8bit",
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "network_module": spec.network_module,
        "network_dim": 32,
        "network_alpha": 32,
        "network_dropout": 0,
        "output_dir": os.path.join(scriptdir, "outputs", spec.slug),
        "output_name": f"my-{spec.slug}-lora",
        "save_precision": "bf16",
        "save_every_n_epochs": 1,
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
        "sample_width": 1024,
        "sample_height": 1024,
        "sample_seed": 42,
        "disable_prompt_enhancement": False,
    }
    defaults.update(_default_model_paths(spec))
    if spec.is_ideogram:
        defaults.update(
            {
                "timestep_sampling": "ideogram4_shift",
                "discrete_flow_shift": 1.0,
                "sampler_preset": "V4_DEFAULT_20",
                "initial_sigma": 1.004,
                "validate_caption_structure": False,
                "warn_on_caption_issues": False,
                "log_loss_stats": False,
                "caching_teo_text_cache_dtype": "bf16",
                "sample_steps": 20,
                "sample_cfg_scale": 7.0,
                "fp8_base": False,
                "fp8_scaled": False,
            }
        )
    else:
        defaults.update(
            {
                "timestep_sampling": "krea2_shift",
                "discrete_flow_shift": 2.5,
                "caching_teo_text_encoder_dtype": "bfloat16",
                "sample_steps": 28,
                "sample_cfg_scale": 5.5,
                "fp8_base": True,
                "fp8_scaled": True,
                "turbo_dit_cache": False,
            }
        )
    return defaults


def _prepare_config(spec: ModernImageArchitecture, config) -> GUIConfig:
    defaults = _architecture_defaults(spec)
    if isinstance(config, dict):
        data = dict(config)
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
        for key in ("dit", "vae", "text_encoder", "unconditional_dit", "turbo_dit", "output_dir"):
            if not data.get(key) and defaults.get(key):
                data[key] = defaults[key]
        return type("GUIConfigAdapter", (), {"config": data, "get": data.get})()

    if not hasattr(config, "config"):
        config = GUIConfig()
    if not getattr(config, "config", None):
        config.config = defaults.copy()
    else:
        for key, value in defaults.items():
            if key not in config.config:
                config.config[key] = value
        for key in ("dit", "vae", "text_encoder", "unconditional_dit", "turbo_dit", "output_dir"):
            if not config.config.get(key) and defaults.get(key):
                config.config[key] = defaults[key]
    return config


def modern_image_lora_tab(spec_key: str, headless: bool = False, config: GUIConfig | dict = {}):
    spec = get_architecture(spec_key)
    config = _prepare_config(spec, config)

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    with gr.Accordion("Configuration file Settings", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    with gr.Row():
        search_input = gr.Textbox(
            label="Search Settings",
            placeholder=f"Search {spec.display_name} settings",
            lines=1,
        )
        toggle_all_button = gr.Button("Open All Panels", variant="secondary")
        panels_state = gr.State(value="closed")
    search_results = gr.HTML(visible=False)

    accordions = []

    accelerate_accordion = gr.Accordion("Accelerate launch Settings", open=False, elem_classes="flux1_background")
    accordions.append(accelerate_accordion)
    with accelerate_accordion:
        accelerate = AccelerateLaunch(config=config)

    save_accordion = gr.Accordion(
        "Save Models and Resume Training Settings",
        open=False,
        elem_classes="samples_background",
    )
    accordions.append(save_accordion)
    with save_accordion:
        save_load = SaveLoadSettings(headless=headless, config=config, show_mem_eff_save=False)

    dataset_accordion = gr.Accordion(
        f"{spec.display_name} Training Dataset",
        open=False,
        elem_classes="samples_background",
    )
    accordions.append(dataset_accordion)
    with dataset_accordion:
        dataset_config_mode = gr.Radio(
            label="Dataset Configuration Method",
            choices=["Use TOML File", "Generate from Folder Structure"],
            value=config.get("dataset_config_mode", "Generate from Folder Structure"),
        )

        def toggle_dataset_mode(mode):
            return (
                gr.Row(visible=mode == "Use TOML File"),
                gr.Column(visible=mode == "Generate from Folder Structure"),
            )

        with gr.Row(visible=config.get("dataset_config_mode") == "Use TOML File") as toml_mode_row:
            dataset_config = gr.Textbox(
                label="Dataset Config",
                value=config.get("dataset_config", ""),
                info="Dataset TOML used by both cache stages and training.",
            )
            dataset_config_button = gr.Button("Browse", visible=not headless)
            dataset_config_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".toml",
                    extension_name="TOML files",
                ),
                outputs=[dataset_config],
            )

        with gr.Column(
            visible=config.get("dataset_config_mode") != "Use TOML File"
        ) as folder_mode_column:
            with gr.Row():
                parent_folder_path = gr.Textbox(
                    label="Parent Dataset Folder",
                    value=config.get("parent_folder_path", ""),
                    info="Parent folder containing repeat-prefixed training folders.",
                )
                parent_folder_button = gr.Button("Browse", visible=not headless)
                parent_folder_button.click(
                    fn=lambda: get_folder_path(folder_path=""),
                    outputs=[parent_folder_path],
                )
            with gr.Row():
                dataset_resolution_width = gr.Number(
                    label="Dataset Width",
                    value=config.get("dataset_resolution_width", 1024),
                    minimum=16,
                    step=16,
                )
                dataset_resolution_height = gr.Number(
                    label="Dataset Height",
                    value=config.get("dataset_resolution_height", 1024),
                    minimum=16,
                    step=16,
                )
                dataset_batch_size = gr.Number(
                    label="Dataset Batch Size",
                    value=config.get("dataset_batch_size", 1),
                    minimum=1,
                    step=1,
                )
            with gr.Row():
                dataset_caption_extension = gr.Textbox(
                    label="Caption Extension",
                    value=config.get("dataset_caption_extension", ".txt"),
                )
                create_missing_captions = gr.Checkbox(
                    label="Create Missing Captions",
                    value=bool(config.get("create_missing_captions", True)),
                )
                caption_strategy = gr.Dropdown(
                    label="Missing Caption Strategy",
                    choices=["folder_name", "empty"],
                    value=config.get("caption_strategy", "folder_name"),
                )
            with gr.Row():
                dataset_enable_bucket = gr.Checkbox(
                    label="Enable Buckets",
                    value=bool(config.get("dataset_enable_bucket", True)),
                )
                dataset_bucket_no_upscale = gr.Checkbox(
                    label="Do Not Upscale Buckets",
                    value=bool(config.get("dataset_bucket_no_upscale", False)),
                )
                dataset_cache_directory = gr.Textbox(
                    label="Cache Directory Name",
                    value=config.get("dataset_cache_directory", "cache_dir"),
                )
            generated_toml_path = gr.Textbox(
                label="Generated Dataset TOML",
                value=config.get("generated_toml_path", ""),
                interactive=False,
            )
            dataset_status = gr.Textbox(label="Dataset Status", interactive=False)
            generate_dataset_button = gr.Button("Generate Dataset TOML", variant="primary")
            generate_dataset_button.click(
                fn=partial(generate_modern_dataset_toml, spec_key),
                inputs=[
                    parent_folder_path,
                    dataset_resolution_width,
                    dataset_resolution_height,
                    dataset_caption_extension,
                    create_missing_captions,
                    caption_strategy,
                    dataset_batch_size,
                    dataset_enable_bucket,
                    dataset_bucket_no_upscale,
                    dataset_cache_directory,
                    save_load.output_dir,
                ],
                outputs=[dataset_config, generated_toml_path, dataset_status],
            )

        dataset_config_mode.change(
            fn=toggle_dataset_mode,
            inputs=[dataset_config_mode],
            outputs=[toml_mode_row, folder_mode_column],
        )

    model_accordion = gr.Accordion(
        f"{spec.display_name} Model Settings",
        open=False,
        elem_classes="preset_background",
    )
    accordions.append(model_accordion)
    with model_accordion:
        with gr.Row():
            dit = gr.Textbox(
                label=f"{spec.display_name} {'Conditional ' if spec.is_ideogram else 'RAW '}DiT",
                value=config.get("dit", ""),
                info="Required training transformer checkpoint.",
            )
            dit_button = gr.Button("Browse", visible=not headless)
            dit_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".safetensors",
                    extension_name="Model files",
                ),
                outputs=[dit],
            )
        with gr.Row():
            vae = gr.Textbox(
                label="VAE",
                value=config.get("vae", ""),
                info="Flux2 VAE for Ideogram 4; Qwen-Image VAE for Krea 2.",
            )
            vae_button = gr.Button("Browse", visible=not headless)
            vae_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".safetensors",
                    extension_name="Model files",
                ),
                outputs=[vae],
            )
            text_encoder = gr.Textbox(
                label="Qwen3-VL Text Encoder",
                value=config.get("text_encoder", ""),
                info="Required for text caching and training-time samples.",
            )
            text_encoder_button = gr.Button("Browse", visible=not headless)
            text_encoder_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".safetensors",
                    extension_name="Model files",
                ),
                outputs=[text_encoder],
            )
        with gr.Row():
            dit_dtype = gr.Dropdown(
                label="DiT Compute Dtype",
                choices=["bfloat16"] if spec.is_krea else ["bfloat16", "float16"],
                value="bfloat16" if spec.is_krea else config.get("dit_dtype", "bfloat16"),
                interactive=not spec.is_krea,
                info="Krea 2 fixes DiT compute to bfloat16 in the backend. Ideogram 4 supports both listed dtypes.",
            )
            vae_dtype = gr.Dropdown(
                label="VAE Compute Dtype",
                choices=["bfloat16", "float16"],
                value=config.get("vae_dtype", "bfloat16"),
            )
            disable_numpy_memmap = gr.Checkbox(
                label="Disable NumPy Memmap",
                value=bool(config.get("disable_numpy_memmap", False)),
                info="Use regular reads instead of memory-mapped model loading.",
            )

        with gr.Row(visible=spec.is_ideogram):
            unconditional_dit = gr.Textbox(
                label="Unconditional DiT (Optional)",
                value=config.get("unconditional_dit", ""),
                info="Optional. Needed only for official asymmetric CFG samples.",
            )
            unconditional_button = gr.Button("Browse", visible=not headless and spec.is_ideogram)
            unconditional_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".safetensors",
                    extension_name="Model files",
                ),
                outputs=[unconditional_dit],
            )
            use_unconditional_dit_for_lora_sampling = gr.Checkbox(
                label="Use Official Asymmetric CFG for LoRA Samples",
                value=bool(config.get("use_unconditional_dit_for_lora_sampling", False)),
                info="Opt in to the optional unconditional DiT during training samples.",
            )
        if not spec.is_ideogram:
            unconditional_dit = gr.Textbox(value="", visible=False)
            use_unconditional_dit_for_lora_sampling = gr.Checkbox(value=False, visible=False)

        with gr.Row(visible=spec.is_krea):
            turbo_dit = gr.Textbox(
                label="Krea 2 Turbo DiT (Optional Samples)",
                value=config.get("turbo_dit", ""),
                info="Optional distilled checkpoint for training-time samples. Cannot be used with block swap.",
            )
            turbo_button = gr.Button("Browse", visible=not headless and spec.is_krea)
            turbo_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".safetensors",
                    extension_name="Model files",
                ),
                outputs=[turbo_dit],
            )
            turbo_dit_cache = gr.Checkbox(
                label="Keep Turbo Weights in CPU RAM",
                value=bool(config.get("turbo_dit_cache", False)),
                info="Faster sampling with roughly one extra DiT copy in host RAM.",
            )
        if not spec.is_krea:
            turbo_dit = gr.Textbox(value="", visible=False)
            turbo_dit_cache = gr.Checkbox(value=False, visible=False)

        with gr.Row():
            fp8_base = gr.Checkbox(
                label="FP8 Base",
                value=bool(config.get("fp8_base", False)),
                visible=spec.is_krea,
                info="Krea 2 only. Must be enabled together with Scaled FP8.",
            )
            fp8_scaled = gr.Checkbox(
                label="Scaled FP8",
                value=bool(config.get("fp8_scaled", False)),
                visible=spec.is_krea,
                info="Krea 2 only. Must be enabled together with FP8 Base.",
            )
            blocks_to_swap = gr.Number(
                label="Blocks to Swap",
                value=config.get("blocks_to_swap", 0),
                minimum=0,
                maximum=spec.max_blocks_to_swap,
                step=1,
                info=f"Maximum for {spec.display_name}: {spec.max_blocks_to_swap}.",
            )
            use_pinned_memory_for_block_swap = gr.Checkbox(
                label="Pinned Memory for Block Swap",
                value=bool(config.get("use_pinned_memory_for_block_swap", False)),
            )
        with gr.Row():
            block_swap_h2d_only = gr.Checkbox(
                label="H2D-Only Block Swap",
                value=bool(config.get("block_swap_h2d_only", False)),
                info="LoRA only. Requires block swap and gradient checkpointing.",
            )
            block_swap_ring_size = gr.Number(
                label="H2D Ring Size",
                value=config.get("block_swap_ring_size", 2),
                minimum=1,
                step=1,
                info="2 overlaps copy and compute; 1 minimizes VRAM.",
            )

    compile_accordion = gr.Accordion("Torch Compile Settings", open=False)
    accordions.append(compile_accordion)
    with compile_accordion:
        with gr.Row():
            compile_enabled = gr.Checkbox(
                label="Enable torch.compile",
                value=bool(config.get("compile", False)),
            )
            compile_backend = gr.Dropdown(
                label="Compile Backend",
                choices=["inductor", "eager", "aot_eager", "cudagraphs"],
                value=config.get("compile_backend", "inductor"),
            )
            compile_mode = gr.Dropdown(
                label="Compile Mode",
                choices=["default", "reduce-overhead", "max-autotune"],
                value=config.get("compile_mode", "default"),
            )
        with gr.Row():
            compile_dynamic = gr.Dropdown(
                label="Dynamic Shapes",
                choices=["auto", "true", "false"],
                value=config.get("compile_dynamic", "auto"),
            )
            compile_fullgraph = gr.Checkbox(
                label="Fullgraph",
                value=bool(config.get("compile_fullgraph", False)),
            )
            compile_cache_size_limit = gr.Number(
                label="Compile Cache Size Limit",
                value=config.get("compile_cache_size_limit", 0),
                minimum=0,
                step=1,
            )

    schedule_accordion = gr.Accordion(
        "Flow Matching and Timestep Settings",
        open=False,
        elem_classes="preset_background",
    )
    accordions.append(schedule_accordion)
    with schedule_accordion:
        timestep_choices = [
            "ideogram4_shift",
            "krea2_shift",
            "shift",
            "sigma",
            "uniform",
            "sigmoid",
            "flux_shift",
            "logsnr",
        ]
        with gr.Row():
            timestep_sampling = gr.Dropdown(
                label="Timestep Sampling",
                choices=timestep_choices,
                value=config.get(
                    "timestep_sampling",
                    "ideogram4_shift" if spec.is_ideogram else "krea2_shift",
                ),
                info="Architecture-specific shift is recommended for variable-resolution datasets.",
            )
            weighting_scheme = gr.Dropdown(
                label="Weighting Scheme",
                choices=["none"] if spec.is_ideogram else ["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"],
                value=config.get("weighting_scheme", "none"),
            )
            discrete_flow_shift = gr.Number(
                label="Discrete Flow Shift",
                value=config.get("discrete_flow_shift", 1.0 if spec.is_ideogram else 2.5),
                step=0.1,
            )
        with gr.Row():
            sigmoid_scale = gr.Number(
                label="Sigmoid Scale",
                value=config.get("sigmoid_scale", 1.0),
                step=0.1,
            )
            min_timestep = gr.Number(
                label="Minimum Timestep",
                value=config.get("min_timestep", 0),
                minimum=0,
                maximum=999,
                step=1,
            )
            max_timestep = gr.Number(
                label="Maximum Timestep",
                value=config.get("max_timestep", 1000),
                minimum=1,
                maximum=1000,
                step=1,
            )
        with gr.Row(visible=spec.is_ideogram):
            sampler_preset = gr.Dropdown(
                label="Ideogram 4 Sampler Preset",
                choices=["V4_DEFAULT_20", "V4_QUALITY_48", "V4_TURBO_12"],
                value=config.get("sampler_preset", "V4_DEFAULT_20"),
            )
            initial_sigma = gr.Number(
                label="Initial Sigma",
                value=config.get("initial_sigma", 1.004),
                step=0.001,
            )
            validate_caption_structure = gr.Checkbox(
                label="Validate Structured Captions",
                value=bool(config.get("validate_caption_structure", False)),
            )
            warn_on_caption_issues = gr.Checkbox(
                label="Warn Instead of Failing Caption Validation",
                value=bool(config.get("warn_on_caption_issues", False)),
            )
            log_loss_stats = gr.Checkbox(
                label="Log Ideogram Loss Diagnostics",
                value=bool(config.get("log_loss_stats", False)),
            )
        if not spec.is_ideogram:
            sampler_preset = gr.Dropdown(choices=["V4_DEFAULT_20"], value="V4_DEFAULT_20", visible=False)
            initial_sigma = gr.Number(value=1.004, visible=False)
            validate_caption_structure = gr.Checkbox(value=False, visible=False)
            warn_on_caption_issues = gr.Checkbox(value=False, visible=False)
            log_loss_stats = gr.Checkbox(value=False, visible=False)

    training_accordion = gr.Accordion(
        "Training Settings",
        open=False,
        elem_classes="preset_background",
    )
    accordions.append(training_accordion)
    with training_accordion:
        training = TrainingSettings(
            headless=headless,
            config=config,
            supported_attention={"sdpa", "flash_attn", "xformers"},
        )

    sample_accordion = gr.Accordion(
        "Sample Generation Settings",
        open=False,
        elem_classes="samples_background",
    )
    accordions.append(sample_accordion)
    with sample_accordion:
        with gr.Row():
            sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                value=config.get("sample_every_n_steps", 0),
                minimum=0,
                step=1,
            )
            sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs",
                value=config.get("sample_every_n_epochs", 0),
                minimum=0,
                step=1,
            )
            sample_at_first = gr.Checkbox(
                label="Sample Before Training",
                value=bool(config.get("sample_at_first", False)),
            )
        with gr.Row():
            sample_prompts = gr.Textbox(
                label="Sample Prompts File",
                value=config.get("sample_prompts", ""),
            )
            sample_prompts_button = gr.Button("Browse", visible=not headless)
            sample_prompts_button.click(
                fn=lambda: get_file_path(
                    file_path="",
                    default_extension=".txt",
                    extension_name="Text files",
                ),
                outputs=[sample_prompts],
            )
            disable_prompt_enhancement = gr.Checkbox(
                label="Disable Prompt Defaults",
                value=bool(config.get("disable_prompt_enhancement", False)),
            )
        with gr.Row():
            sample_output_dir = gr.Textbox(
                label="Sample Output Directory",
                value=config.get("sample_output_dir", ""),
            )
            sample_output_button = gr.Button("Browse", visible=not headless)
            sample_output_button.click(
                fn=lambda: get_folder_path(folder_path=""),
                outputs=[sample_output_dir],
            )
        with gr.Row():
            sample_width = gr.Number(
                label="Sample Width",
                value=config.get("sample_width", 1024),
                minimum=16,
                step=16,
            )
            sample_height = gr.Number(
                label="Sample Height",
                value=config.get("sample_height", 1024),
                minimum=16,
                step=16,
            )
            sample_steps = gr.Number(
                label="Sample Steps",
                value=config.get("sample_steps", 20 if spec.is_ideogram else 28),
                minimum=1,
                step=1,
                interactive=not spec.is_ideogram,
                info="Ideogram 4 uses the selected sampler preset.",
            )
            sample_seed = gr.Number(
                label="Sample Seed",
                value=config.get("sample_seed", 42),
                step=1,
            )
        with gr.Row():
            sample_negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=config.get("sample_negative_prompt", ""),
            )
            sample_cfg_scale = gr.Number(
                label="Sample CFG Scale",
                value=config.get("sample_cfg_scale", 7.0 if spec.is_ideogram else 5.5),
                step=0.1,
                interactive=not spec.is_ideogram,
                info="Krea RAW defaults to 5.5; Krea Turbo should use 1. Ideogram guidance comes from its preset.",
            )

    caching_accordion = gr.Accordion(
        "Caching Settings",
        open=False,
        elem_classes="samples_background",
    )
    accordions.append(caching_accordion)
    with caching_accordion:
        with gr.Row():
            cache_latents = gr.Checkbox(
                label="Run Latent Caching",
                value=bool(config.get("cache_latents", True)),
            )
            caching_latent_device = gr.Textbox(
                label="Latent Cache Device",
                value=config.get("caching_latent_device", "cuda"),
            )
            caching_latent_batch_size = gr.Number(
                label="Latent Cache Batch Size",
                value=config.get("caching_latent_batch_size", 1),
                minimum=1,
                step=1,
            )
            caching_latent_num_workers = gr.Number(
                label="Latent Cache Workers",
                value=config.get("caching_latent_num_workers", 2),
                minimum=1,
                step=1,
            )
        with gr.Row():
            caching_latent_skip_existing = gr.Checkbox(
                label="Skip Existing Latent Caches",
                value=bool(config.get("caching_latent_skip_existing", True)),
            )
            caching_latent_keep_cache = gr.Checkbox(
                label="Keep Unreferenced Latent Caches",
                value=bool(config.get("caching_latent_keep_cache", True)),
            )
        with gr.Row():
            cache_text_encoder_outputs = gr.Checkbox(
                label="Run Text Encoder Caching",
                value=bool(config.get("cache_text_encoder_outputs", True)),
            )
            caching_teo_device = gr.Textbox(
                label="Text Cache Device",
                value=config.get("caching_teo_device", "cuda"),
            )
            caching_teo_batch_size = gr.Number(
                label="Text Cache Batch Size",
                value=config.get("caching_teo_batch_size", 1),
                minimum=1,
                step=1,
            )
            caching_teo_num_workers = gr.Number(
                label="Text Cache Workers",
                value=config.get("caching_teo_num_workers", 2),
                minimum=1,
                step=1,
            )
        with gr.Row():
            caching_teo_skip_existing = gr.Checkbox(
                label="Skip Existing Text Caches",
                value=bool(config.get("caching_teo_skip_existing", True)),
            )
            caching_teo_keep_cache = gr.Checkbox(
                label="Keep Unreferenced Text Caches",
                value=bool(config.get("caching_teo_keep_cache", True)),
            )
            caching_teo_text_cache_dtype = gr.Dropdown(
                label="Ideogram Text Cache Dtype",
                choices=["bf16", "fp8_e4m3fn", "float32"],
                value=config.get("caching_teo_text_cache_dtype", "bf16"),
                visible=spec.is_ideogram,
                info="FP8 greatly reduces Ideogram 4 text-cache disk usage.",
            )
            caching_teo_text_encoder_dtype = gr.Dropdown(
                label="Krea Text Encoder Dtype",
                choices=["bfloat16", "float16"],
                value=config.get("caching_teo_text_encoder_dtype", "bfloat16"),
                visible=spec.is_krea,
            )

    optimizer_accordion = gr.Accordion(
        "Learning Rate, Optimizer and Scheduler Settings",
        open=False,
        elem_classes="flux1_rank_layers_background",
    )
    accordions.append(optimizer_accordion)
    with optimizer_accordion:
        optimizer = OptimizerAndScheduler(headless=headless, config=config)

    network_accordion = gr.Accordion("LoRA Settings", open=False, elem_classes="flux1_background")
    accordions.append(network_accordion)
    with network_accordion:
        network = Network(headless=headless, config=config)
        network.network_module.value = spec.network_module
        network.network_module.interactive = False

    advanced_accordion = gr.Accordion("Advanced Settings", open=False, elem_classes="samples_background")
    accordions.append(advanced_accordion)
    with advanced_accordion:
        advanced = AdvancedTraining(headless=headless, training_type="lora", config=config)

    metadata_accordion = gr.Accordion(
        "Metadata Settings",
        open=False,
        elem_classes="flux1_rank_layers_background",
    )
    accordions.append(metadata_accordion)
    with metadata_accordion:
        metadata = MetaData(config=config)

    huggingface_accordion = gr.Accordion(
        "HuggingFace Settings",
        open=False,
        elem_classes="huggingface_background",
    )
    accordions.append(huggingface_accordion)
    with huggingface_accordion:
        huggingface = HuggingFace(config=config)

    executors[spec_key] = CommandExecutor(headless=headless)
    executor = executors[spec_key]
    run_state = gr.Textbox(value=train_state_values[spec_key], visible=False)

    with gr.Row():
        print_button = gr.Button("Print Training Workflow")
        toggle_all_button_bottom = gr.Button("Open All Panels", variant="secondary")

    settings_list = [
        # Accelerate
        accelerate.mixed_precision,
        accelerate.num_cpu_threads_per_process,
        accelerate.num_processes,
        accelerate.num_machines,
        accelerate.multi_gpu,
        accelerate.gpu_ids,
        accelerate.main_process_port,
        accelerate.dynamo_backend,
        accelerate.dynamo_mode,
        accelerate.dynamo_use_fullgraph,
        accelerate.dynamo_use_dynamic,
        accelerate.extra_accelerate_launch_args,
        # Advanced
        advanced.additional_parameters,
        advanced.debug_mode,
        # Dataset
        dataset_config_mode,
        dataset_config,
        parent_folder_path,
        dataset_resolution_width,
        dataset_resolution_height,
        dataset_caption_extension,
        create_missing_captions,
        caption_strategy,
        dataset_batch_size,
        dataset_enable_bucket,
        dataset_bucket_no_upscale,
        dataset_cache_directory,
        generated_toml_path,
        # Model
        dit,
        vae,
        text_encoder,
        dit_dtype,
        vae_dtype,
        unconditional_dit,
        use_unconditional_dit_for_lora_sampling,
        turbo_dit,
        turbo_dit_cache,
        fp8_base,
        fp8_scaled,
        disable_numpy_memmap,
        blocks_to_swap,
        use_pinned_memory_for_block_swap,
        block_swap_h2d_only,
        block_swap_ring_size,
        # Compile
        compile_enabled,
        compile_backend,
        compile_mode,
        compile_dynamic,
        compile_fullgraph,
        compile_cache_size_limit,
        # Architecture and schedule
        sampler_preset,
        initial_sigma,
        validate_caption_structure,
        warn_on_caption_issues,
        log_loss_stats,
        timestep_sampling,
        weighting_scheme,
        discrete_flow_shift,
        sigmoid_scale,
        min_timestep,
        max_timestep,
        # Training
        training.sdpa,
        training.flash_attn,
        training.sage_attn,
        training.xformers,
        training.split_attn,
        training.max_train_steps,
        training.max_train_epochs,
        training.max_data_loader_n_workers,
        training.persistent_data_loader_workers,
        training.seed,
        training.gradient_checkpointing,
        training.gradient_checkpointing_cpu_offload,
        training.gradient_accumulation_steps,
        training.full_bf16,
        training.full_fp16,
        training.logging_dir,
        training.log_with,
        training.log_prefix,
        training.log_tracker_name,
        training.wandb_run_name,
        training.log_tracker_config,
        training.wandb_api_key,
        training.log_config,
        training.ddp_timeout,
        training.ddp_gradient_as_bucket_view,
        training.ddp_static_graph,
        # Sampling
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_at_first,
        sample_prompts,
        sample_output_dir,
        disable_prompt_enhancement,
        sample_width,
        sample_height,
        sample_steps,
        sample_seed,
        sample_negative_prompt,
        sample_cfg_scale,
        # Caching
        cache_latents,
        caching_latent_device,
        caching_latent_batch_size,
        caching_latent_num_workers,
        caching_latent_skip_existing,
        caching_latent_keep_cache,
        cache_text_encoder_outputs,
        caching_teo_device,
        caching_teo_batch_size,
        caching_teo_num_workers,
        caching_teo_skip_existing,
        caching_teo_keep_cache,
        caching_teo_text_cache_dtype,
        caching_teo_text_encoder_dtype,
        # Optimizer
        optimizer.optimizer_type,
        optimizer.optimizer_args,
        optimizer.learning_rate,
        optimizer.max_grad_norm,
        optimizer.lr_scheduler,
        optimizer.lr_warmup_steps,
        optimizer.lr_decay_steps,
        optimizer.lr_scheduler_num_cycles,
        optimizer.lr_scheduler_power,
        optimizer.lr_scheduler_timescale,
        optimizer.lr_scheduler_min_lr_ratio,
        optimizer.lr_scheduler_type,
        optimizer.lr_scheduler_args,
        # Network
        network.no_metadata,
        network.network_weights,
        network.network_module,
        network.network_dim,
        network.network_alpha,
        network.network_dropout,
        network.network_args,
        network.training_comment,
        network.dim_from_weights,
        network.scale_weight_norms,
        network.base_weights,
        network.base_weights_multiplier,
        # Save
        save_load.output_dir,
        save_load.output_name,
        save_load.resume,
        save_load.save_precision,
        save_load.save_every_n_epochs,
        save_load.save_last_n_epochs,
        save_load.save_every_n_steps,
        save_load.save_last_n_steps,
        save_load.save_last_n_epochs_state,
        save_load.save_last_n_steps_state,
        save_load.save_state,
        save_load.save_state_on_train_end,
        save_load.mem_eff_save,
        # Metadata
        metadata.metadata_author,
        metadata.metadata_description,
        metadata.metadata_license,
        metadata.metadata_tags,
        metadata.metadata_title,
        metadata.metadata_reso,
        metadata.metadata_arch,
        # Hugging Face
        huggingface.huggingface_repo_id,
        huggingface.huggingface_token,
        huggingface.huggingface_repo_type,
        huggingface.huggingface_repo_visibility,
        huggingface.huggingface_path_in_repo,
        huggingface.save_state_to_huggingface,
        huggingface.resume_from_huggingface,
        huggingface.async_upload,
    ]
    assert len(settings_list) == len(MODERN_IMAGE_PARAM_KEYS), (
        f"{spec.display_name}: settings ({len(settings_list)}) do not match keys "
        f"({len(MODERN_IMAGE_PARAM_KEYS)})"
    )

    panel_keywords = {
        0: "accelerate gpu precision distributed dynamo",
        1: "save output checkpoint resume precision",
        2: "dataset captions resolution buckets cache folder toml",
        3: f"model dit vae text encoder fp8 block swap {spec.display_name.lower()}",
        4: "compile torch inductor",
        5: "timestep flow schedule weighting sampler caption",
        6: "training attention steps epochs gradient logging ddp",
        7: "sample prompts cfg seed",
        8: "cache latent text encoder device dtype",
        9: "optimizer learning rate scheduler",
        10: "lora network rank alpha dropout base weights",
        11: "advanced parameters debug",
        12: "metadata author license tags",
        13: "huggingface upload repository token",
    }
    panel_names = [
        "Accelerate launch Settings",
        "Save Models and Resume Training Settings",
        f"{spec.display_name} Training Dataset",
        f"{spec.display_name} Model Settings",
        "Torch Compile Settings",
        "Flow Matching and Timestep Settings",
        "Training Settings",
        "Sample Generation Settings",
        "Caching Settings",
        "Learning Rate, Optimizer and Scheduler Settings",
        "LoRA Settings",
        "Advanced Settings",
        "Metadata Settings",
        "HuggingFace Settings",
    ]

    def search_settings(query):
        query = str(query or "").strip().lower()
        if not query:
            return gr.update(value="", visible=False)
        matches = [
            panel_names[index]
            for index, keywords in panel_keywords.items()
            if query in keywords or query in panel_names[index].lower()
        ]
        if not matches:
            return gr.update(value=f"<strong>No settings found for '{query}'.</strong>", visible=True)
        items = "".join(f"<li>{name}</li>" for name in matches)
        return gr.update(value=f"<strong>Matching panels</strong><ul>{items}</ul>", visible=True)

    search_input.change(
        fn=search_settings,
        inputs=[search_input],
        outputs=[search_results],
        show_progress=False,
    )

    def toggle_panels(state):
        open_panels = state != "open"
        new_state = "open" if open_panels else "closed"
        label = "Hide All Panels" if open_panels else "Open All Panels"
        return [new_state, gr.Button(value=label), gr.Button(value=label)] + [
            gr.Accordion(open=open_panels) for _ in accordions
        ]

    for button in (toggle_all_button, toggle_all_button_bottom):
        button.click(
            fn=toggle_panels,
            inputs=[panels_state],
            outputs=[panels_state, toggle_all_button, toggle_all_button_bottom] + accordions,
            show_progress=False,
        )

    action = partial(modern_image_gui_actions, spec_key)
    configuration.button_open_config.click(
        action,
        inputs=[
            gr.Textbox(value="open_configuration", visible=False),
            dummy_true,
            configuration.config_file_name,
            dummy_headless,
            dummy_false,
        ]
        + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )
    configuration.button_load_config.click(
        action,
        inputs=[
            gr.Textbox(value="open_configuration", visible=False),
            dummy_false,
            configuration.config_file_name,
            dummy_headless,
            dummy_false,
        ]
        + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )
    configuration.button_save_config.click(
        action,
        inputs=[
            gr.Textbox(value="save_configuration", visible=False),
            dummy_false,
            configuration.config_file_name,
            dummy_headless,
            dummy_false,
        ]
        + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )
    print_button.click(
        action,
        inputs=[
            gr.Textbox(value="train_model", visible=False),
            dummy_false,
            configuration.config_file_name,
            dummy_headless,
            dummy_true,
        ]
        + settings_list,
        show_progress=False,
    )
    executor.button_run.click(
        action,
        inputs=[
            gr.Textbox(value="train_model", visible=False),
            dummy_false,
            configuration.config_file_name,
            dummy_headless,
            dummy_false,
        ]
        + settings_list,
        outputs=[
            executor.button_run,
            executor.stop_row,
            executor.button_stop_training,
            executor.training_status,
            run_state,
        ],
        show_progress=False,
    )
    executor.button_stop_training.click(
        executor.kill_command,
        inputs=[],
        outputs=[
            executor.button_run,
            executor.stop_row,
            executor.button_stop_training,
            executor.training_status,
        ],
        js="() => { if (confirm('Stop training/caching?')) { return []; } else { throw new Error('Cancelled'); } }",
    )

    return settings_list
