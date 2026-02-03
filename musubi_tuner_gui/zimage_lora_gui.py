import os
import re
import shlex
import shutil
import sys
import time
from datetime import datetime

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
    generate_script_content,
    get_file_path,
    get_file_path_or_save_as,
    get_folder_path,
    print_command_and_toml,
    run_cmd_advanced_training,
    save_executed_script,
    scriptdir,
    setup_environment,
)
from .custom_logging import setup_logging
from .dataset_config_generator import generate_dataset_config_from_folders, save_dataset_config, validate_dataset_config

log = setup_logging()

executor = None
train_state_value = time.time()


def _find_accelerate_launch(python_cmd: str) -> list[str]:
    accelerate_path = shutil.which("accelerate")
    if accelerate_path:
        return [rf"{accelerate_path}", "launch"]

    python_dir = os.path.dirname(python_cmd)
    accelerate_fallback = os.path.join(python_dir, "accelerate.exe" if sys.platform == "win32" else "accelerate")
    if os.path.exists(accelerate_fallback) and os.access(accelerate_fallback, os.X_OK):
        return [rf"{accelerate_fallback}", "launch"]

    log.warning("Accelerate binary not found, using Python module fallback")
    return [python_cmd, "-m", "accelerate.commands.launch"]


def _get_debug_parameters_for_mode(debug_mode: str) -> str:
    debug_params = {
        "Show Timesteps (Image)": "--show_timesteps image",
        "Show Timesteps (Console)": "--show_timesteps console",
        "RCM Debug Save": "--rcm_debug_save",
        "Enable Logging (TensorBoard)": "--log_with tensorboard --logging_dir ./logs",
        "Enable Logging (WandB)": "--log_with wandb",
        "Enable Logging (All)": "--log_with all --logging_dir ./logs",
    }
    return debug_params.get(debug_mode, "")


def _generate_zimage_dataset_toml(
    parent_folder_path: str,
    dataset_resolution_width: int,
    dataset_resolution_height: int,
    dataset_caption_extension: str,
    create_missing_captions: bool,
    caption_strategy: str,
    dataset_batch_size: int,
    dataset_enable_bucket: bool,
    dataset_bucket_no_upscale: bool,
    dataset_cache_directory: str,
    output_dir: str,
):
    try:
        parent_folder_path = (parent_folder_path or "").strip()
        if not parent_folder_path:
            raise ValueError("Parent folder path is required.")
        if not os.path.isdir(parent_folder_path):
            raise ValueError(f"Parent folder does not exist: {parent_folder_path}")

        save_dir = (output_dir or "").strip() or parent_folder_path
        os.makedirs(save_dir, exist_ok=True)

        cfg, msgs = generate_dataset_config_from_folders(
            parent_folder=parent_folder_path,
            resolution=(int(dataset_resolution_width), int(dataset_resolution_height)),
            caption_extension=(dataset_caption_extension or ".txt"),
            create_missing_captions=bool(create_missing_captions),
            caption_strategy=caption_strategy or "folder_name",
            batch_size=int(dataset_batch_size),
            enable_bucket=bool(dataset_enable_bucket),
            bucket_no_upscale=bool(dataset_bucket_no_upscale),
            cache_directory_name=dataset_cache_directory or "cache_dir",
            # Z-Image does not support control images; avoid adding control_directory entries.
            control_directory_name="__zimage_no_control__",
        )

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(save_dir, f"zimage_dataset_{ts}.toml")
        save_dataset_config(cfg, out_path)

        is_valid, v_msgs = validate_dataset_config(out_path)
        status_lines = []
        status_lines.extend(msgs)
        status_lines.extend(v_msgs)
        if not is_valid:
            status_lines.append("[ERROR] Dataset TOML validation failed. Please review warnings/errors above.")
        else:
            status_lines.append(f"[OK] Saved dataset TOML: {out_path}")

        return out_path, out_path, "\n".join(status_lines)
    except Exception as e:
        err = f"[ERROR] Failed to generate dataset TOML: {e}"
        log.error(err)
        return "", "", err


def save_zimage_configuration(save_as_bool, file_path, parameters):
    original_file_path = file_path

    if save_as_bool or not file_path:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")

    if not file_path:
        return original_file_path, gr.update(value="No file selected.", visible=True)

    try:
        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "save_as_bool", "headless", "print_only"],
        )
        msg = f"Configuration saved: {os.path.basename(file_path)}"
        gr.Info(msg)
        return file_path, gr.update(value=msg, visible=True)
    except Exception as e:
        msg = f"Failed to save configuration: {e}"
        log.error(msg)
        gr.Error(msg)
        return original_file_path, gr.update(value=msg, visible=True)


def open_zimage_configuration(ask_for_file, file_path, parameters):
    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")

    if not file_path:
        values = [original_file_path, gr.update(value="", visible=False)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    if ask_for_file and not os.path.isfile(file_path):
        msg = f"New configuration file will be created at: {os.path.basename(file_path)}"
        gr.Info(msg)
        values = [file_path, gr.update(value=msg, visible=True)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    if not os.path.isfile(file_path):
        msg = f"Config file does not exist: {file_path}"
        gr.Error(msg)
        values = [original_file_path, gr.update(value=msg, visible=True)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
    except Exception as e:
        msg = f"Failed to load configuration: {e}"
        log.error(msg)
        gr.Error(msg)
        values = [original_file_path, gr.update(value=msg, visible=True)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    numeric_fields = {
        "dataset_resolution_width",
        "dataset_resolution_height",
        "dataset_batch_size",
        "num_cpu_threads_per_process",
        "num_processes",
        "num_machines",
        "main_process_port",
        "blocks_to_swap",
        "max_train_steps",
        "max_train_epochs",
        "max_data_loader_n_workers",
        "seed",
        "gradient_accumulation_steps",
        "discrete_flow_shift",
        "sigmoid_scale",
        "min_timestep",
        "max_timestep",
        "learning_rate",
        "max_grad_norm",
        "lr_warmup_steps",
        "lr_decay_steps",
        "lr_scheduler_num_cycles",
        "lr_scheduler_power",
        "lr_scheduler_timescale",
        "lr_scheduler_min_lr_ratio",
        "network_dim",
        "network_alpha",
        "network_dropout",
        "scale_weight_norms",
        "sample_every_n_steps",
        "sample_every_n_epochs",
        "sample_width",
        "sample_height",
        "sample_steps",
        "sample_guidance_scale",
        "sample_seed",
        "sample_cfg_scale",
        "save_every_n_epochs",
        "save_every_n_steps",
        "save_last_n_epochs",
        "save_last_n_steps",
        "save_last_n_epochs_state",
        "save_last_n_steps_state",
        "ddp_timeout",
        "compile_cache_size_limit",
        "caching_latent_batch_size",
        "caching_latent_num_workers",
        "caching_teo_batch_size",
        "caching_teo_num_workers",
    }

    list_to_str_fields = {"optimizer_args", "lr_scheduler_args", "network_args"}

    loaded_values = []
    for key, default_value in parameters:
        if key in data:
            v = data[key]
            if isinstance(v, list) and key in numeric_fields:
                v = v[0] if v else None
            elif isinstance(v, list) and key in list_to_str_fields:
                v = " ".join(str(x) for x in v)
            loaded_values.append(v)
        else:
            loaded_values.append(default_value)

    msg = f"Loaded configuration: {os.path.basename(file_path)}"
    gr.Info(msg)
    return tuple([file_path, gr.update(value=msg, visible=True)] + loaded_values)


def _maybe_create_enhanced_sample_prompts_zimage(param_dict: dict, parameters: list[tuple[str, object]]) -> tuple[dict, list[tuple[str, object]]]:
    sample_prompts = (param_dict.get("sample_prompts") or "").strip()
    if not sample_prompts or not sample_prompts.lower().endswith(".txt"):
        return param_dict, parameters
    if not os.path.isfile(sample_prompts):
        return param_dict, parameters

    output_dir = (param_dict.get("output_dir") or "").strip()
    if not output_dir:
        return param_dict, parameters

    sample_out_dir = (param_dict.get("sample_output_dir") or "").strip() or output_dir
    os.makedirs(sample_out_dir, exist_ok=True)

    width = int(param_dict.get("sample_width") or 1024)
    height = int(param_dict.get("sample_height") or 1024)
    steps = int(param_dict.get("sample_steps") or 25)
    seed = param_dict.get("sample_seed", None)
    neg = (param_dict.get("sample_negative_prompt") or "").strip()
    cfg_scale = float(param_dict.get("sample_cfg_scale") or 4.0)

    def has_flag(s: str, flag: str) -> bool:
        return re.search(rf"(?:^|\\s)--{re.escape(flag)}\\s+", s) is not None

    enhanced_lines = []
    with open(sample_prompts, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                enhanced_lines.append(line)
                continue

            s = stripped
            if not has_flag(s, "w"):
                s += f" --w {width}"
            if not has_flag(s, "h"):
                s += f" --h {height}"
            if not has_flag(s, "s"):
                s += f" --s {steps}"
            if seed is not None and int(seed) >= 0 and not has_flag(s, "d"):
                s += f" --d {int(seed)}"
            if not has_flag(s, "l"):
                s += f" --l {cfg_scale}"
            if neg and not has_flag(s, "n"):
                s += f" --n {neg}"

            enhanced_lines.append(s)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = (param_dict.get("output_name") or "zimage").strip() or "zimage"
    enhanced_path = os.path.join(sample_out_dir, f"{out_name}_enhanced_prompts_{ts}.txt")
    with open(enhanced_path, "w", encoding="utf-8") as f:
        f.write("# Enhanced prompt file generated by SECourses Musubi Trainer\n")
        f.write(f"# Original file: {sample_prompts}\n\n")
        for l in enhanced_lines:
            f.write(l + "\n")

    param_dict["sample_prompts"] = enhanced_path
    parameters = [(k, (enhanced_path if k == "sample_prompts" else v)) for k, v in parameters]
    return param_dict, parameters


def train_zimage_model(headless: bool, print_only: bool, parameters):
    global train_state_value

    python_cmd = sys.executable
    run_cmd = _find_accelerate_launch(python_cmd)

    param_dict = dict(parameters)

    training_mode = (param_dict.get("training_mode") or "LoRA Training").strip()

    # Prefer generated dataset config when using folder mode.
    dataset_config_mode = (param_dict.get("dataset_config_mode") or "").strip()
    if dataset_config_mode == "Generate from Folder Structure":
        gen_path = (param_dict.get("generated_toml_path") or "").strip()
        if gen_path:
            param_dict["dataset_config"] = gen_path
            parameters = [(k, (gen_path if k == "dataset_config" else v)) for k, v in parameters]

    dataset_config = (param_dict.get("dataset_config") or "").strip()
    if not dataset_config:
        raise ValueError("[ERROR] Dataset config is required. Generate a dataset TOML or set dataset_config.")
    if not os.path.exists(dataset_config):
        raise ValueError(f"[ERROR] Dataset config file does not exist: {dataset_config}")

    dit_path = (param_dict.get("dit") or "").strip()
    vae_path = (param_dict.get("vae") or "").strip()
    te_path = (param_dict.get("text_encoder") or "").strip()

    if not dit_path or not os.path.exists(dit_path):
        raise ValueError("[ERROR] DiT checkpoint path is required and must exist.")
    if not vae_path or not os.path.exists(vae_path):
        raise ValueError("[ERROR] VAE checkpoint path is required and must exist.")
    if not te_path or not os.path.exists(te_path):
        raise ValueError("[ERROR] Text Encoder checkpoint path is required and must exist.")

    output_dir = (param_dict.get("output_dir") or "").strip()
    output_name = (param_dict.get("output_name") or "").strip()
    if not output_dir:
        raise ValueError("[ERROR] Output directory is required.")
    if not output_name:
        raise ValueError("[ERROR] Output name is required.")

    # Enforce correct LoRA module for LoRA training only.
    if training_mode == "LoRA Training":
        required_network_module = "networks.lora_zimage"
        if (param_dict.get("network_module") or "").strip() != required_network_module:
            param_dict["network_module"] = required_network_module
            parameters = [(k, (required_network_module if k == "network_module" else v)) for k, v in parameters]

    # Optional pre-steps
    latent_cache_cmd = None
    teo_cache_cmd = None

    # Match WAN tab behavior: run caching unless explicitly disabled.
    if param_dict.get("caching_latent_skip_existing") is not False:
        latent_cache_cmd = [
            python_cmd,
            f"{scriptdir}/musubi-tuner/src/musubi_tuner/zimage_cache_latents.py",
            "--dataset_config",
            dataset_config,
            "--vae",
            vae_path,
        ]

        caching_device = (param_dict.get("caching_latent_device") or "cuda").strip()
        if caching_device == "cuda":
            if (param_dict.get("gpu_ids") or "").strip():
                gpu_ids = str(param_dict.get("gpu_ids")).split(",")
                caching_device = f"cuda:{gpu_ids[0].strip()}"
            else:
                caching_device = "cuda:0"
        latent_cache_cmd.extend(["--device", caching_device])

        if param_dict.get("caching_latent_batch_size") is not None:
            latent_cache_cmd.extend(["--batch_size", str(int(param_dict.get("caching_latent_batch_size")))])
        if param_dict.get("caching_latent_num_workers") is not None:
            latent_cache_cmd.extend(["--num_workers", str(int(param_dict.get("caching_latent_num_workers")))])

        if bool(param_dict.get("caching_latent_skip_existing")):
            latent_cache_cmd.append("--skip_existing")
        if bool(param_dict.get("caching_latent_keep_cache", False)):
            latent_cache_cmd.append("--keep_cache")

    if param_dict.get("caching_teo_skip_existing") is not False:
        teo_cache_cmd = [
            python_cmd,
            f"{scriptdir}/musubi-tuner/src/musubi_tuner/zimage_cache_text_encoder_outputs.py",
            "--dataset_config",
            dataset_config,
            "--text_encoder",
            te_path,
        ]

        if bool(param_dict.get("caching_teo_fp8_llm", False)):
            teo_cache_cmd.append("--fp8_llm")

        teo_device = (param_dict.get("caching_teo_device") or "cuda").strip()
        if teo_device == "cuda":
            if (param_dict.get("gpu_ids") or "").strip():
                gpu_ids = str(param_dict.get("gpu_ids")).split(",")
                teo_device = f"cuda:{gpu_ids[0].strip()}"
            else:
                teo_device = "cuda:0"
        teo_cache_cmd.extend(["--device", teo_device])

        if param_dict.get("caching_teo_batch_size") is not None:
            teo_cache_cmd.extend(["--batch_size", str(int(param_dict.get("caching_teo_batch_size")))])
        if param_dict.get("caching_teo_num_workers") is not None:
            teo_cache_cmd.extend(["--num_workers", str(int(param_dict.get("caching_teo_num_workers")))])

        if bool(param_dict.get("caching_teo_skip_existing")):
            teo_cache_cmd.append("--skip_existing")
        if bool(param_dict.get("caching_teo_keep_cache", False)):
            teo_cache_cmd.append("--keep_cache")

    # Accelerate launch flags
    run_cmd = AccelerateLaunch.run_cmd(
        run_cmd=run_cmd,
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

    if training_mode == "DreamBooth Fine-Tuning":
        run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/zimage_train.py")
    else:
        run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/zimage_train_network.py")

    if print_only:
        if latent_cache_cmd:
            print_command_and_toml(latent_cache_cmd, "")
        if teo_cache_cmd:
            print_command_and_toml(teo_cache_cmd, "")
        print_command_and_toml(run_cmd, "")
        return

    os.makedirs(output_dir, exist_ok=True)
    formatted_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_path = os.path.join(output_dir, f"{output_name}_{formatted_datetime}.toml")

    # Enhance prompt file defaults if user provided sample_prompts.
    param_dict, parameters = _maybe_create_enhanced_sample_prompts_zimage(param_dict, parameters)

    pattern_exclusion = [k for k, _ in parameters if k.startswith("caching_latent_") or k.startswith("caching_teo_")]

    dreambooth_mode = training_mode == "DreamBooth Fine-Tuning"
    exclusion_extra = []
    if dreambooth_mode:
        exclusion_extra.extend(
            [
                "network_module",
                "network_dim",
                "network_alpha",
                "network_dropout",
                "network_args",
                "network_weights",
                "training_comment",
                "dim_from_weights",
                "scale_weight_norms",
                "base_weights",
                "base_weights_multiplier",
                "no_metadata",
            ]
        )

    SaveConfigFileToRun(
        parameters=parameters,
        file_path=cfg_path,
        exclusion=[
            "file_path",
            "save_as",
            "save_as_bool",
            "headless",
            "print_only",
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
            # GUI-only dataset generation fields
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
            # GUI-only variant selector and prompt helper
            "zimage_variant",
            "sample_cfg_scale",
            "training_mode",
            # Cache-only toggle
            "caching_teo_fp8_llm",
        ]
        + pattern_exclusion
        + exclusion_extra,
        mandatory_keys=["dataset_config", "dit", "vae", "text_encoder"]
        + ([] if dreambooth_mode else ["network_module"]),
    )

    run_cmd.append("--config_file")
    run_cmd.append(cfg_path)

    additional_params = (param_dict.get("additional_parameters") or "").strip()
    debug_mode_selected = (param_dict.get("debug_mode") or "None").strip()
    if debug_mode_selected and debug_mode_selected != "None":
        debug_params = _get_debug_parameters_for_mode(debug_mode_selected)
        if debug_params:
            additional_params = (additional_params + " " + debug_params).strip() if additional_params else debug_params

    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, additional_parameters=additional_params)

    # Match Qwen/WAN launch strategy:
    # 1) run latent caching synchronously (so failures surface before training),
    # 2) then run (optional) text-encoder caching + training via a wrapper script so Stop cancels both.
    env = setup_environment()

    if latent_cache_cmd:
        log.info("Running latent caching...")
        latent_cache_script = generate_script_content(latent_cache_cmd, "Z-Image latent caching")
        save_executed_script(script_content=latent_cache_script, config_name=output_name, script_type="zimage_latent_cache")
        try:
            gr.Info("Starting latent caching... This may take a while.")
            import subprocess

            subprocess.run(latent_cache_cmd, env=setup_environment(), check=True)
            gr.Info("Latent caching completed successfully!")
        except subprocess.CalledProcessError as e:
            log.error(f"Latent caching failed with return code {e.returncode}")
            gr.Warning(f"Latent caching failed with return code {e.returncode}")
            raise RuntimeError(f"Latent caching failed with return code {e.returncode}")
        except FileNotFoundError:
            raise RuntimeError(f"Python executable not found: {python_cmd}")

    if teo_cache_cmd:
        import platform
        import tempfile
        import subprocess

        if platform.system() == "Windows":
            script_ext = ".bat"
            teo_cache_cmd_str = " ".join([f"\"{arg}\"" if " " in str(arg) else str(arg) for arg in teo_cache_cmd])
            run_cmd_str = " ".join([f"\"{arg}\"" if " " in str(arg) else str(arg) for arg in run_cmd])
            script_content = f"""@echo off
echo Starting text encoder output caching...
{teo_cache_cmd_str}
if %errorlevel% neq 0 (
    echo Text encoder output caching failed with error code %errorlevel%
    exit /b %errorlevel%
)
echo Text encoder output caching completed successfully.
echo Starting training...
{run_cmd_str}
"""
        else:
            script_ext = ".sh"
            teo_cache_cmd_str = shlex.join(teo_cache_cmd)
            run_cmd_str = shlex.join(run_cmd)
            script_content = f"""#!/bin/bash
set -e
echo "Starting text encoder output caching..."
{teo_cache_cmd_str}
echo "Text encoder output caching completed successfully."
echo "Starting training..."
{run_cmd_str}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=script_ext, delete=False, encoding="utf-8") as f:
            temp_script = f.name
            f.write(script_content)

        if platform.system() != "Windows":
            import stat

            os.chmod(temp_script, os.stat(temp_script).st_mode | stat.S_IEXEC)

        save_executed_script(script_content=script_content, config_name=output_name, script_type="zimage")
        final_cmd = [temp_script] if platform.system() == "Windows" else ["bash", temp_script]
        gr.Info("Starting text encoder caching followed by training...")
        executor.execute_command(run_cmd=final_cmd, env=env, shell=True if platform.system() == "Windows" else False)
    else:
        training_script = generate_script_content(run_cmd, "Z-Image training")
        save_executed_script(script_content=training_script, config_name=output_name, script_type="zimage")
        gr.Info("Starting training...")
        executor.execute_command(run_cmd=run_cmd, env=env)

    train_state_value = time.time()
    return (
        gr.Button(visible=False or headless),
        gr.Row(visible=True),
        gr.Button(interactive=True),
        gr.Textbox(value="Training in progress..."),
        gr.Textbox(value=train_state_value),
    )


ZIMAGE_PARAM_KEYS = [
    # accelerate_launch
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
    # advanced_training
    "additional_parameters",
    "debug_mode",
    # dataset
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
    # gui-only model selector
    "training_mode",
    "zimage_variant",
    # model
    "dit",
    "vae",
    "text_encoder",
    "fp8_base",
    "fp8_scaled",
    "fp8_llm",
    "disable_numpy_memmap",
    "blocks_to_swap",
    "use_pinned_memory_for_block_swap",
    "img_in_txt_in_offloading",
    # torch compile
    "compile",
    "compile_backend",
    "compile_mode",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_cache_size_limit",
    # schedule
    "timestep_sampling",
    "weighting_scheme",
    "discrete_flow_shift",
    "sigmoid_scale",
    "min_timestep",
    "max_timestep",
    # training settings
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
    # samples
    "sample_every_n_steps",
    "sample_every_n_epochs",
    "sample_at_first",
    "sample_prompts",
    "sample_output_dir",
    "sample_width",
    "sample_height",
    "sample_steps",
    "sample_guidance_scale",
    "sample_seed",
    "sample_negative_prompt",
    "sample_cfg_scale",
    # caching
    "caching_latent_device",
    "caching_latent_batch_size",
    "caching_latent_num_workers",
    "caching_latent_skip_existing",
    "caching_latent_keep_cache",
    "caching_teo_device",
    "caching_teo_batch_size",
    "caching_teo_num_workers",
    "caching_teo_skip_existing",
    "caching_teo_keep_cache",
    "caching_teo_fp8_llm",
    # optimizer/scheduler
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
    # network
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
    # save/load
    "output_dir",
    "output_name",
    "resume",
    "save_every_n_epochs",
    "save_last_n_epochs",
    "save_every_n_steps",
    "save_last_n_steps",
    "save_last_n_epochs_state",
    "save_last_n_steps_state",
    "save_state",
    "save_state_on_train_end",
    "mem_eff_save",
    # metadata
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    # huggingface
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_repo_type",
    "huggingface_repo_visibility",
    "huggingface_path_in_repo",
    "save_state_to_huggingface",
    "resume_from_huggingface",
    "async_upload",
]


def zimage_gui_actions(action: str, ask_for_file: bool, config_file_name: str, headless: bool, print_only: bool, *args):
    if action == "open_configuration":
        return open_zimage_configuration(ask_for_file, config_file_name, list(zip(ZIMAGE_PARAM_KEYS, args)))
    if action == "save_configuration":
        return save_zimage_configuration(ask_for_file, config_file_name, list(zip(ZIMAGE_PARAM_KEYS, args)))
    if action == "train_model":
        return train_zimage_model(headless=headless, print_only=print_only, parameters=list(zip(ZIMAGE_PARAM_KEYS, args)))


def zimage_lora_tab(headless=False, config: GUIConfig = {}):
    global executor

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    defaults = {
        "training_mode": "LoRA Training",
        "zimage_variant": "base",
        "network_module": "networks.lora_zimage",
        "output_name": "my-zimage-lora",
        # Torch compile defaults
        "compile": False,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_dynamic": "auto",
        "compile_fullgraph": False,
        "compile_cache_size_limit": 0,
        "mixed_precision": "bf16",
        "num_cpu_threads_per_process": 1,
        "sdpa": True,
        "optimizer_type": "adamw8bit",
        "learning_rate": 1e-4,
        "gradient_checkpointing": True,
        "timestep_sampling": "shift",
        "weighting_scheme": "none",
        "discrete_flow_shift": 2.0,
        "dataset_config_mode": "Generate from Folder Structure",
        "dataset_resolution_width": 1024,
        "dataset_resolution_height": 1024,
        "dataset_enable_bucket": True,
        "dataset_cache_directory": "cache_dir",
        # Caching defaults
        "caching_latent_device": "cuda",
        "caching_latent_batch_size": 1,
        "caching_latent_num_workers": 2,
        "caching_latent_skip_existing": True,
        "caching_latent_keep_cache": True,
        "caching_teo_device": "cuda",
        "caching_teo_batch_size": 8,
        "caching_teo_num_workers": 2,
        "caching_teo_skip_existing": True,
        "caching_teo_keep_cache": True,
        "caching_teo_fp8_llm": False,
        # Samples
        "sample_width": 1024,
        "sample_height": 1024,
        "sample_steps": 25,
        "sample_cfg_scale": 4.0,
    }

    if isinstance(config, dict):
        config.update({k: v for k, v in defaults.items() if k not in config})
        config = type("GUIConfig", (), {"config": config, "get": config.get})()
    elif hasattr(config, "config"):
        if not getattr(config, "config", None):
            config.config = defaults.copy()
        else:
            config.config.update({k: v for k, v in defaults.items() if k not in config.config})

    with gr.Accordion("Configuration file Settings", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    # Add search functionality and unified toggle button
    with gr.Row():
        with gr.Column(scale=2):
            search_input = gr.Textbox(
                label="🔍 Search Settings",
                placeholder="Type to search and filter panels (e.g., 'model', 'learning', 'fp8', 'cache', 'epochs')",
                lines=1,
                interactive=True,
                info="Filter settings panels by keyword.",
            )
        with gr.Column(scale=1):
            toggle_all_btn = gr.Button(
                value="Open All Panels",
                variant="secondary",
                size="lg",
                elem_id="toggle-all-btn"
            )
            # Hidden state to track if panels are open or closed
            panels_state = gr.State(value="closed")  # Default state is closed

    # Hidden elements for search functionality (not displayed)
    search_results_row = gr.Row(visible=False)
    search_results = gr.HTML(visible=False)

    # Create accordion references
    accordions = []

    accelerate_accordion = gr.Accordion("Accelerate launch Settings", open=False, elem_classes="flux1_background")
    accordions.append(accelerate_accordion)
    with accelerate_accordion:
        accelerate_launch = AccelerateLaunch(config=config)

    save_load_accordion = gr.Accordion("Save Models and Resume Training Settings", open=False, elem_classes="samples_background")
    accordions.append(save_load_accordion)
    with save_load_accordion:
        save_load = SaveLoadSettings(headless=headless, config=config)

    dataset_accordion = gr.Accordion("Z-Image Training Dataset", open=False, elem_classes="samples_background")
    accordions.append(dataset_accordion)
    with dataset_accordion:
        with gr.Row():
            dataset_config_mode = gr.Radio(
                label="Dataset Configuration Method",
                choices=["Use TOML File", "Generate from Folder Structure"],
                value=config.get("dataset_config_mode", "Generate from Folder Structure"),
                info="Choose how to configure your dataset: provide a TOML file or auto-generate from folder structure",
            )

        def _toggle_dataset_mode(mode):
            toml_visible = mode == "Use TOML File"
            folder_visible = mode == "Generate from Folder Structure"
            return gr.Row(visible=toml_visible), gr.Column(visible=folder_visible)

        with gr.Row(
            visible=config.get("dataset_config_mode", "Generate from Folder Structure") == "Use TOML File"
        ) as toml_mode_row:
            dataset_config = gr.Textbox(
                label="dataset_config",
                value=config.get("dataset_config", ""),
                interactive=True,
                info="Path to dataset TOML used by caching and training.",
            )
            dataset_config_btn = gr.Button("📁", size="lg", visible=not headless)
            dataset_config_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".toml", extension_name="TOML files"),
                outputs=[dataset_config],
            )

        with gr.Column(
            visible=config.get("dataset_config_mode", "Generate from Folder Structure") == "Generate from Folder Structure"
        ) as folder_mode_column:
            with gr.Row():
                parent_folder_path = gr.Textbox(
                    label="parent_folder_path",
                    value=config.get("parent_folder_path", ""),
                    interactive=True,
                    info="Parent folder used to generate a dataset TOML (only in Generate mode).",
                )
                parent_folder_btn = gr.Button("📂", size="lg", visible=not headless)
                parent_folder_btn.click(fn=lambda: get_folder_path(folder_path=""), outputs=[parent_folder_path])

            with gr.Row():
                dataset_resolution_width = gr.Number(
                    label="dataset_resolution_width",
                    value=config.get("dataset_resolution_width", 1024),
                    step=8,
                    interactive=True,
                    info="Target dataset width in pixels (multiples of 8 recommended).",
                )
                dataset_resolution_height = gr.Number(
                    label="dataset_resolution_height",
                    value=config.get("dataset_resolution_height", 1024),
                    step=8,
                    interactive=True,
                    info="Target dataset height in pixels (multiples of 8 recommended).",
                )

            with gr.Row():
                dataset_caption_extension = gr.Textbox(
                    label="dataset_caption_extension",
                    value=config.get("dataset_caption_extension", ".txt"),
                    interactive=True,
                    info="Caption file extension to read/write (e.g., .txt).",
                )
                create_missing_captions = gr.Checkbox(
                    label="create_missing_captions",
                    value=bool(config.get("create_missing_captions", True)),
                    info="Create empty captions when missing during TOML generation.",
                )
                caption_strategy = gr.Dropdown(
                    label="caption_strategy",
                    choices=["folder_name", "empty"],
                    value=config.get("caption_strategy", "folder_name"),
                    interactive=True,
                    info="folder_name uses the folder name as caption; empty writes blank captions.",
                )

            with gr.Row():
                dataset_batch_size = gr.Number(
                    label="dataset_batch_size",
                    value=config.get("dataset_batch_size", 1),
                    minimum=1,
                    step=1,
                    interactive=True,
                    info="Batch size recorded in the dataset config.",
                )
                dataset_enable_bucket = gr.Checkbox(
                    label="dataset_enable_bucket",
                    value=bool(config.get("dataset_enable_bucket", True)),
                    info="Enable resolution bucketing in the dataset config.",
                )
                dataset_bucket_no_upscale = gr.Checkbox(
                    label="dataset_bucket_no_upscale",
                    value=bool(config.get("dataset_bucket_no_upscale", False)),
                    info="Do not upscale smaller images when bucketing.",
                )

            with gr.Row():
                dataset_cache_directory = gr.Textbox(
                    label="dataset_cache_directory",
                    value=config.get("dataset_cache_directory", "cache_dir"),
                    interactive=True,
                    info="Cache directory name for latent/text encoder outputs.",
                )
                generated_toml_path = gr.Textbox(
                    label="generated_toml_path",
                    value=config.get("generated_toml_path", ""),
                    interactive=False,
                    info="Read-only: path of the generated dataset TOML.",
                )

            with gr.Row():
                dataset_status = gr.Textbox(label="Dataset status", value="", interactive=False, info="Validation and status messages.")

            with gr.Row():
                generate_toml_btn = gr.Button("Generate dataset TOML", variant="primary")
                generate_toml_btn.click(
                    fn=_generate_zimage_dataset_toml,
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
            fn=_toggle_dataset_mode,
            inputs=[dataset_config_mode],
            outputs=[toml_mode_row, folder_mode_column],
        )

    model_accordion = gr.Accordion("Z-Image Model Settings", open=False, elem_classes="preset_background")
    accordions.append(model_accordion)
    with model_accordion:
        with gr.Row():
            training_mode = gr.Radio(
                label="Training Mode",
                choices=["LoRA Training", "DreamBooth Fine-Tuning"],
                value=config.get("training_mode", "LoRA Training"),
                info="LoRA: parameter-efficient. DreamBooth: full fine-tuning (more VRAM).",
            )
        with gr.Row():
            zimage_variant = gr.Dropdown(
                label="zimage_variant",
                choices=[("Base", "base"), ("Turbo", "turbo")],
                value=config.get("zimage_variant", "base"),
                interactive=True,
                info="Select Base vs Turbo defaults (your actual DiT checkpoint decides what you train).",
            )
        with gr.Row():
            dit = gr.Textbox(
                label="DiT (Base Model) Checkpoint Path",
                value=config.get("dit", ""),
                placeholder="Path to DiT base model checkpoint (Z_Image_BF16.safetensors)",
                interactive=True,
                info="Required DiT checkpoint. Examples: Z_Image_BF16.safetensors or Z_Image_Turbo_BF16.safetensors.",
            )
            dit_btn = gr.Button("📁", size="lg", visible=not headless)
            dit_btn.click(fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"), outputs=[dit])
        with gr.Row():
            vae = gr.Textbox(
                label="vae",
                value=config.get("vae", ""),
                placeholder="Example: Z_Image_Train_VAE.safetensors",
                interactive=True,
                info="Required VAE checkpoint (e.g., Z_Image_Train_VAE.safetensors).",
            )
            vae_btn = gr.Button("📁", size="lg", visible=not headless)
            vae_btn.click(fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"), outputs=[vae])
        with gr.Row():
            text_encoder = gr.Textbox(
                label="text_encoder",
                value=config.get("text_encoder", ""),
                placeholder="Example: qwen_3_8b.safetensors",
                interactive=True,
                info="Required Qwen3 text encoder (Z_Image_Training_Text_Encoder.safetensors).",
            )
            te_btn = gr.Button("📁", size="lg", visible=not headless)
            te_btn.click(fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"), outputs=[text_encoder])
        with gr.Row():
            fp8_base = gr.Checkbox(
                label="fp8_base",
                value=bool(config.get("fp8_base", False)),
                info="Use FP8 for the base model (DiT) to reduce VRAM.",
            )
            fp8_scaled = gr.Checkbox(
                label="fp8_scaled",
                value=bool(config.get("fp8_scaled", False)),
                info="Use scaled FP8 for the base model (recommended with fp8_base).",
            )
            fp8_llm = gr.Checkbox(
                label="fp8_llm",
                value=bool(config.get("fp8_llm", False)),
                info="Use FP8 for the Qwen3 text encoder during training.",
            )
        with gr.Row():
            disable_numpy_memmap = gr.Checkbox(
                label="disable_numpy_memmap",
                value=bool(config.get("disable_numpy_memmap", False)),
                info="Disable numpy memmap when loading weights (uses more RAM but can avoid mmap issues).",
            )
            blocks_to_swap = gr.Number(
                label="blocks_to_swap",
                value=config.get("blocks_to_swap", 0),
                minimum=0,
                step=1,
                interactive=True,
                info="Offload N transformer blocks to CPU for VRAM savings.",
            )
            use_pinned_memory_for_block_swap = gr.Checkbox(
                label="use_pinned_memory_for_block_swap",
                value=bool(config.get("use_pinned_memory_for_block_swap", False)),
                info="Use pinned memory for faster CPU<->GPU transfers (uses more shared memory on Windows).",
            )
            img_in_txt_in_offloading = gr.Checkbox(
                label="img_in_txt_in_offloading",
                value=bool(config.get("img_in_txt_in_offloading", False)),
                info="Offload img_in and txt_in tensors to CPU to reduce VRAM.",
            )

    torch_compile_accordion = gr.Accordion("Torch Compile Settings", open=False)
    accordions.append(torch_compile_accordion)
    with torch_compile_accordion:
        gr.Markdown(
            """⚠️ **Important:** If you get errors with torch.compile just disable it. It can increase speed and slightly reduce VRAM with no quality loss."""
        )

        with gr.Row():
            compile = gr.Checkbox(
                label="Enable torch.compile",
                info="Enable torch.compile for faster training (requires PyTorch 2.1+, Triton for CUDA). Disable gradient checkpointing for best results!",
                value=bool(config.get("compile", False)),
                interactive=True,
            )
            compile_backend = gr.Dropdown(
                label="Compile Backend",
                info="Backend for torch.compile (default: inductor)",
                choices=["inductor", "cudagraphs", "eager", "aot_eager", "aot_ts_nvfuser"],
                value=config.get("compile_backend", "inductor"),
                interactive=True,
            )
            compile_mode = gr.Dropdown(
                label="Compile Mode",
                info="Optimization mode for torch.compile",
                choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                value=config.get("compile_mode", "default"),
                interactive=True,
            )

        with gr.Row():
            compile_dynamic = gr.Dropdown(
                label="Dynamic Shapes",
                info="Dynamic shape handling: auto (default), true (enable), false (disable)",
                choices=["auto", "true", "false"],
                value=config.get("compile_dynamic", "auto"),
                allow_custom_value=False,
                interactive=True,
            )
            compile_fullgraph = gr.Checkbox(
                label="Fullgraph Mode",
                info="Enable fullgraph mode in torch.compile (may fail with complex models)",
                value=bool(config.get("compile_fullgraph", False)),
                interactive=True,
            )
            compile_cache_size_limit = gr.Number(
                label="Cache Size Limit",
                info="Set torch._dynamo.config.cache_size_limit (0 = use PyTorch default, typically 8-32)",
                value=config.get("compile_cache_size_limit", 0),
                step=1,
                minimum=0,
                interactive=True,
            )

    schedule_accordion = gr.Accordion("Flow Matching and Timestep Settings", open=False, elem_classes="flux1_background")
    accordions.append(schedule_accordion)
    with schedule_accordion:
        with gr.Row():
            timestep_sampling = gr.Dropdown(
                label="timestep_sampling",
                choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "qwen_shift", "logsnr", "qinglong_flux", "qinglong_qwen"],
                value=config.get("timestep_sampling", "shift"),
                interactive=True,
                info="Timestep sampling method for flow matching (shift recommended for Z-Image).",
            )
            weighting_scheme = gr.Dropdown(
                label="weighting_scheme",
                choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
                value=config.get("weighting_scheme", "none"),
                interactive=True,
                info="Loss weighting scheme for timestep distribution (none disables weighting).",
            )
        with gr.Row():
            discrete_flow_shift = gr.Number(
                label="discrete_flow_shift",
                value=config.get("discrete_flow_shift", 2.0),
                step=0.1,
                interactive=True,
                info="Shift factor used by shift/flux_shift sampling.",
            )
            sigmoid_scale = gr.Number(
                label="sigmoid_scale",
                value=config.get("sigmoid_scale", 1.0),
                step=0.1,
                interactive=True,
                info="Scale for sigmoid/shift timestep sampling.",
            )
        with gr.Row():
            min_timestep = gr.Number(
                label="min_timestep",
                value=config.get("min_timestep", 0),
                step=1,
                interactive=True,
                info="Minimum timestep clamp (0-999).",
            )
            max_timestep = gr.Number(
                label="max_timestep",
                value=config.get("max_timestep", 1000),
                step=1,
                interactive=True,
                info="Maximum timestep clamp (1-1000).",
            )

    training_accordion = gr.Accordion("Training Settings", open=False, elem_classes="preset_background")
    accordions.append(training_accordion)
    with training_accordion:
        training = TrainingSettings(headless=headless, config=config)

    sample_accordion = gr.Accordion("Sample Generation Settings", open=False, elem_classes="samples_background")
    accordions.append(sample_accordion)
    with sample_accordion:
        _variant_init = (config.get("zimage_variant", "base") or "base").strip().lower()
        _default_sample_steps = 20 if _variant_init == "turbo" else 25
        _default_cfg_scale = 1.0 if _variant_init == "turbo" else 4.0
        with gr.Row():
            sample_every_n_steps = gr.Number(
                label="sample_every_n_steps",
                value=config.get("sample_every_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
                info="Generate samples every N steps (0 disables).",
            )
            sample_every_n_epochs = gr.Number(
                label="sample_every_n_epochs",
                value=config.get("sample_every_n_epochs", 0),
                minimum=0,
                step=1,
                interactive=True,
                info="Generate samples every N epochs (overrides steps).",
            )
            sample_at_first = gr.Checkbox(
                label="sample_at_first",
                value=bool(config.get("sample_at_first", False)),
                info="Generate samples before training starts.",
            )
        with gr.Row():
            sample_prompts = gr.Textbox(
                label="sample_prompts",
                value=config.get("sample_prompts", ""),
                interactive=True,
                info="Path to a prompt file (one prompt per line).",
            )
            sample_prompts_btn = gr.Button("📁", size="lg", visible=not headless)
            sample_prompts_btn.click(fn=lambda: get_file_path(file_path="", default_extension=".txt", extension_name="Text files"), outputs=[sample_prompts])
        with gr.Row():
            sample_output_dir = gr.Textbox(
                label="sample_output_dir",
                value=config.get("sample_output_dir", ""),
                interactive=True,
                info="Folder to save generated samples.",
            )
            sample_output_dir_btn = gr.Button("📂", size="lg", visible=not headless)
            sample_output_dir_btn.click(fn=lambda: get_folder_path(folder_path=""), outputs=[sample_output_dir])
        with gr.Row():
            sample_width = gr.Number(
                label="sample_width",
                value=config.get("sample_width", 1024),
                step=8,
                interactive=True,
                info="Sample width in pixels (multiples of 8 recommended).",
            )
            sample_height = gr.Number(
                label="sample_height",
                value=config.get("sample_height", 1024),
                step=8,
                interactive=True,
                info="Sample height in pixels (multiples of 8 recommended).",
            )
        with gr.Row():
            sample_steps = gr.Number(
                label="sample_steps",
                value=config.get("sample_steps", _default_sample_steps),
                step=1,
                interactive=True,
                info="Sampling steps (Base ~25, Turbo ~20 by default).",
            )
            sample_guidance_scale = gr.Number(
                label="sample_guidance_scale",
                value=config.get("sample_guidance_scale", 0.0),
                step=0.1,
                interactive=True,
                info="Legacy guidance scale; Z-Image uses sample_cfg_scale (--l) instead.",
            )
            sample_seed = gr.Number(
                label="sample_seed",
                value=config.get("sample_seed", 42),
                step=1,
                interactive=True,
                info="Seed for sample generation.",
            )
        with gr.Row():
            sample_cfg_scale = gr.Number(
                label="sample_cfg_scale",
                value=config.get("sample_cfg_scale", _default_cfg_scale),
                step=0.1,
                interactive=True,
                info="CFG scale for Z-Image sampling (kohya flag: --l).",
            )
            sample_negative_prompt = gr.Textbox(
                label="sample_negative_prompt",
                value=config.get("sample_negative_prompt", ""),
                interactive=True,
                info="Optional negative prompt (used when cfg_scale > 1).",
            )

        # Variant-based suggestions (doesn't force anything, but helps defaults)
        def _variant_defaults(variant: str):
            if variant == "turbo":
                return gr.update(value=1.0), gr.update(value=20)
            return gr.update(value=4.0), gr.update(value=25)

        zimage_variant.change(fn=_variant_defaults, inputs=[zimage_variant], outputs=[sample_cfg_scale, sample_steps])

    caching_accordion = gr.Accordion("Caching Settings", open=False, elem_classes="samples_background")
    accordions.append(caching_accordion)
    with caching_accordion:
        with gr.Row():
            caching_latent_device = gr.Textbox(
                label="caching_latent_device",
                value=config.get("caching_latent_device", "cuda"),
                interactive=True,
                info="Device for latent caching (e.g., cuda, cuda:0, cpu).",
            )
            caching_latent_batch_size = gr.Number(
                label="caching_latent_batch_size",
                value=config.get("caching_latent_batch_size", 1),
                step=1,
                interactive=True,
                info="Batch size for latent caching.",
            )
            caching_latent_num_workers = gr.Number(
                label="caching_latent_num_workers",
                value=config.get("caching_latent_num_workers", 2),
                step=1,
                interactive=True,
                info="Dataloader workers for latent caching.",
            )
        with gr.Row():
            caching_latent_skip_existing = gr.Checkbox(
                label="caching_latent_skip_existing",
                value=bool(config.get("caching_latent_skip_existing", True)),
                info="Skip existing cache entries (set false to disable latent caching step).",
            )
            caching_latent_keep_cache = gr.Checkbox(
                label="caching_latent_keep_cache",
                value=bool(config.get("caching_latent_keep_cache", True)),
                info="Keep latent cache after training.",
            )
        with gr.Row():
            caching_teo_device = gr.Textbox(
                label="caching_teo_device",
                value=config.get("caching_teo_device", "cuda"),
                interactive=True,
                info="Device for text encoder output caching.",
            )
            caching_teo_batch_size = gr.Number(
                label="caching_teo_batch_size",
                value=config.get("caching_teo_batch_size", 8),
                step=1,
                interactive=True,
                info="Batch size for text encoder output caching.",
            )
            caching_teo_num_workers = gr.Number(
                label="caching_teo_num_workers",
                value=config.get("caching_teo_num_workers", 2),
                step=1,
                interactive=True,
                info="Dataloader workers for text encoder output caching.",
            )
        with gr.Row():
            caching_teo_skip_existing = gr.Checkbox(
                label="caching_teo_skip_existing",
                value=bool(config.get("caching_teo_skip_existing", True)),
                info="Skip existing caches (set false to disable text encoder caching step).",
            )
            caching_teo_keep_cache = gr.Checkbox(
                label="caching_teo_keep_cache",
                value=bool(config.get("caching_teo_keep_cache", True)),
                info="Keep text encoder output cache after training.",
            )
            caching_teo_fp8_llm = gr.Checkbox(
                label="caching_teo_fp8_llm",
                value=bool(config.get("caching_teo_fp8_llm", False)),
                interactive=True,
                info="Use FP8 for the Qwen3 text encoder during caching.",
            )

    optimizer_accordion = gr.Accordion("Learning Rate, Optimizer and Scheduler Settings", open=False, elem_classes="flux1_rank_layers_background")
    accordions.append(optimizer_accordion)
    with optimizer_accordion:
        optim = OptimizerAndScheduler(headless=headless, config=config)

    network_accordion = gr.Accordion("LoRA Settings", open=False, elem_classes="flux1_background")
    accordions.append(network_accordion)
    with network_accordion:
        network = Network(headless=headless, config=config)

    advanced_accordion = gr.Accordion("Advanced Settings", open=False, elem_classes="samples_background")
    accordions.append(advanced_accordion)
    with advanced_accordion:
        advanced = AdvancedTraining(headless=headless, config=config)

    metadata_accordion = gr.Accordion("Metadata Settings", open=False, elem_classes="flux1_rank_layers_background")
    accordions.append(metadata_accordion)
    with metadata_accordion:
        metadata = MetaData(config=config)

    huggingface_accordion = gr.Accordion("HuggingFace Settings", open=False, elem_classes="huggingface_background")
    accordions.append(huggingface_accordion)
    with huggingface_accordion:
        huggingface = HuggingFace(config=config)

    executor = CommandExecutor(headless=headless)
    run_state = gr.Textbox(value=train_state_value, visible=False)

    with gr.Column(), gr.Group():
        with gr.Row():
            button_print = gr.Button("Print training command")
            toggle_all_btn_bottom = gr.Button(
                value="Open All Panels",
                variant="secondary"
            )

    settings_list = [
        # accelerate_launch
        accelerate_launch.mixed_precision,
        accelerate_launch.num_cpu_threads_per_process,
        accelerate_launch.num_processes,
        accelerate_launch.num_machines,
        accelerate_launch.multi_gpu,
        accelerate_launch.gpu_ids,
        accelerate_launch.main_process_port,
        accelerate_launch.dynamo_backend,
        accelerate_launch.dynamo_mode,
        accelerate_launch.dynamo_use_fullgraph,
        accelerate_launch.dynamo_use_dynamic,
        accelerate_launch.extra_accelerate_launch_args,
        # advanced_training
        advanced.additional_parameters,
        advanced.debug_mode,
        # dataset
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
        # variant
        training_mode,
        zimage_variant,
        # model
        dit,
        vae,
        text_encoder,
        fp8_base,
        fp8_scaled,
        fp8_llm,
        disable_numpy_memmap,
        blocks_to_swap,
        use_pinned_memory_for_block_swap,
        img_in_txt_in_offloading,
        # torch compile
        compile,
        compile_backend,
        compile_mode,
        compile_dynamic,
        compile_fullgraph,
        compile_cache_size_limit,
        # schedule
        timestep_sampling,
        weighting_scheme,
        discrete_flow_shift,
        sigmoid_scale,
        min_timestep,
        max_timestep,
        # training settings
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
        # samples
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_at_first,
        sample_prompts,
        sample_output_dir,
        sample_width,
        sample_height,
        sample_steps,
        sample_guidance_scale,
        sample_seed,
        sample_negative_prompt,
        sample_cfg_scale,
        # caching
        caching_latent_device,
        caching_latent_batch_size,
        caching_latent_num_workers,
        caching_latent_skip_existing,
        caching_latent_keep_cache,
        caching_teo_device,
        caching_teo_batch_size,
        caching_teo_num_workers,
        caching_teo_skip_existing,
        caching_teo_keep_cache,
        caching_teo_fp8_llm,
        # optimizer/scheduler
        optim.optimizer_type,
        optim.optimizer_args,
        optim.learning_rate,
        optim.max_grad_norm,
        optim.lr_scheduler,
        optim.lr_warmup_steps,
        optim.lr_decay_steps,
        optim.lr_scheduler_num_cycles,
        optim.lr_scheduler_power,
        optim.lr_scheduler_timescale,
        optim.lr_scheduler_min_lr_ratio,
        optim.lr_scheduler_type,
        optim.lr_scheduler_args,
        # network
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
        # save/load
        save_load.output_dir,
        save_load.output_name,
        save_load.resume,
        save_load.save_every_n_epochs,
        save_load.save_last_n_epochs,
        save_load.save_every_n_steps,
        save_load.save_last_n_steps,
        save_load.save_last_n_epochs_state,
        save_load.save_last_n_steps_state,
        save_load.save_state,
        save_load.save_state_on_train_end,
        save_load.mem_eff_save,
        # metadata
        metadata.metadata_author,
        metadata.metadata_description,
        metadata.metadata_license,
        metadata.metadata_tags,
        metadata.metadata_title,
        # huggingface
        huggingface.huggingface_repo_id,
        huggingface.huggingface_token,
        huggingface.huggingface_repo_type,
        huggingface.huggingface_repo_visibility,
        huggingface.huggingface_path_in_repo,
        huggingface.save_state_to_huggingface,
        huggingface.resume_from_huggingface,
        huggingface.async_upload,
    ]

    assert len(settings_list) == len(ZIMAGE_PARAM_KEYS), f"settings_list ({len(settings_list)}) != keys ({len(ZIMAGE_PARAM_KEYS)})"

    # Add handler for search functionality
    def search_settings(query):
        if not query or len(query.strip()) < 1:
            return gr.Row(visible=False), ""

        query_lower = query.lower().strip()
        results = []

        # Comprehensive parameter map for Z-Image
        parameter_map = {
            # Model Variant
            "training_mode": ("Z-Image Model Settings", "Training Mode (LoRA/DreamBooth)"),
            "zimage_variant": ("Z-Image Model Settings", "Z-Image Variant (Base/Turbo)"),
            "base": ("Z-Image Model Settings", "Base Model"),
            "turbo": ("Z-Image Model Settings", "Turbo Model (Distilled)"),
            "variant": ("Z-Image Model Settings", "Model Variant"),

            # Model Paths
            "dit": ("Z-Image Model Settings", "DiT (Base Model) Checkpoint Path"),
            "vae": ("Z-Image Model Settings", "VAE Checkpoint Path"),
            "text_encoder": ("Z-Image Model Settings", "Text Encoder (Qwen3) Path"),

            # FP8 and Memory
            "fp8_base": ("Z-Image Model Settings", "FP8 for Base Model (DiT)"),
            "fp8_scaled": ("Z-Image Model Settings", "Scaled FP8 for Base Model"),
            "fp8_llm": ("Z-Image Model Settings", "FP8 for Qwen3 Text Encoder"),
            "blocks_to_swap": ("Z-Image Model Settings", "Blocks to Swap to CPU"),
            "blocks": ("Z-Image Model Settings", "Blocks to Swap"),
            "swap": ("Z-Image Model Settings", "Blocks to Swap"),
            "cpu": ("Z-Image Model Settings", "CPU Offloading"),
            "disable_numpy_memmap": ("Z-Image Model Settings", "Disable NumPy Memmap"),
            "use_pinned_memory_for_block_swap": ("Z-Image Model Settings", "Use Pinned Memory for Block Swap"),
            "img_in_txt_in_offloading": ("Z-Image Model Settings", "Image-in-Text Input Offloading"),

            # Torch Compile
            "compile": ("Torch Compile Settings", "Enable torch.compile"),
            "compile_backend": ("Torch Compile Settings", "Compile Backend"),
            "compile_mode": ("Torch Compile Settings", "Compile Mode"),
            "compile_dynamic": ("Torch Compile Settings", "Dynamic Shapes"),
            "compile_fullgraph": ("Torch Compile Settings", "Fullgraph Mode"),
            "compile_cache_size_limit": ("Torch Compile Settings", "Cache Size Limit"),

            # Dataset Settings
            "dataset": ("Z-Image Training Dataset", "Dataset Configuration"),
            "dataset_config_mode": ("Z-Image Training Dataset", "Dataset Configuration Method"),
            "dataset_config": ("Z-Image Training Dataset", "Dataset Config File"),
            "parent_folder": ("Z-Image Training Dataset", "Parent Folder Path"),
            "resolution": ("Z-Image Training Dataset", "Resolution"),
            "caption": ("Z-Image Training Dataset", "Caption Settings"),
            "bucket": ("Z-Image Training Dataset", "Bucketing"),
            "batch_size": ("Z-Image Training Dataset", "Batch Size"),
            "cache_directory": ("Z-Image Training Dataset", "Cache Directory"),

            # Flow Matching
            "timestep": ("Flow Matching and Timestep Settings", "Timestep Sampling"),
            "weighting": ("Flow Matching and Timestep Settings", "Weighting Scheme"),
            "discrete_flow_shift": ("Flow Matching and Timestep Settings", "Discrete Flow Shift"),
            "sigmoid": ("Flow Matching and Timestep Settings", "Sigmoid Scale"),

            # Training Settings
            "sdpa": ("Training Settings", "Use SDPA"),
            "flash_attn": ("Training Settings", "Use FlashAttention"),
            "sage_attn": ("Training Settings", "Use SageAttention"),
            "xformers": ("Training Settings", "Use xformers"),
            "split_attn": ("Training Settings", "Split Attention"),
            "max_train_steps": ("Training Settings", "Max Training Steps"),
            "max_train_epochs": ("Training Settings", "Max Training Epochs"),
            "epochs": ("Training Settings", "Training Epochs"),
            "steps": ("Training Settings", "Training Steps"),
            "seed": ("Training Settings", "Random Seed"),
            "gradient": ("Training Settings", "Gradient Settings"),
            "gradient_checkpointing": ("Training Settings", "Gradient Checkpointing"),
            "gradient_accumulation": ("Training Settings", "Gradient Accumulation Steps"),
            "full_bf16": ("Training Settings", "Full BF16"),
            "full_fp16": ("Training Settings", "Full FP16"),
            "workers": ("Training Settings", "DataLoader Workers"),
            "persistent": ("Training Settings", "Persistent DataLoader Workers"),

            # Optimizer Settings
            "optimizer": ("Learning Rate, Optimizer and Scheduler Settings", "Optimizer Type"),
            "learning_rate": ("Learning Rate, Optimizer and Scheduler Settings", "Learning Rate"),
            "lr": ("Learning Rate, Optimizer and Scheduler Settings", "Learning Rate"),
            "adamw": ("Learning Rate, Optimizer and Scheduler Settings", "AdamW Optimizer"),
            "scheduler": ("Learning Rate, Optimizer and Scheduler Settings", "LR Scheduler"),
            "warmup": ("Learning Rate, Optimizer and Scheduler Settings", "LR Warmup Steps"),
            "max_grad_norm": ("Learning Rate, Optimizer and Scheduler Settings", "Max Gradient Norm"),

            # Network/LoRA Settings
            "lora": ("LoRA Settings", "LoRA Configuration"),
            "network_module": ("LoRA Settings", "Network Module"),
            "network_dim": ("LoRA Settings", "Network Dimension (LoRA Rank)"),
            "network_alpha": ("LoRA Settings", "Network Alpha"),
            "network_dropout": ("LoRA Settings", "Network Dropout"),
            "network_args": ("LoRA Settings", "Network Arguments"),
            "rank": ("LoRA Settings", "LoRA Rank"),
            "alpha": ("LoRA Settings", "LoRA Alpha"),
            "lora_zimage": ("LoRA Settings", "LoRA Z-Image Module"),

            # Caching
            "cache": ("Caching Settings", "Caching Configuration"),
            "latent_cache": ("Caching Settings", "Latent Caching"),
            "text_encoder_cache": ("Caching Settings", "Text Encoder Output Caching"),
            "teo_cache": ("Caching Settings", "Text Encoder Output Caching"),
            "caching_device": ("Caching Settings", "Caching Device"),
            "skip_existing": ("Caching Settings", "Skip Existing Cache"),
            "keep_cache": ("Caching Settings", "Keep Cache Files"),
            "fp8_llm_cache": ("Caching Settings", "FP8 for Text Encoder Caching"),

            # Samples
            "sample": ("Sample Generation Settings", "Sample Generation"),
            "sample_prompts": ("Sample Generation Settings", "Sample Prompts File"),
            "sample_steps": ("Sample Generation Settings", "Sample Steps"),
            "sample_cfg_scale": ("Sample Generation Settings", "Sample CFG Scale"),
            "guidance": ("Sample Generation Settings", "Guidance Scale"),
            "cfg": ("Sample Generation Settings", "CFG Scale"),

            # Save/Load
            "output": ("Save Models and Resume Training Settings", "Output Settings"),
            "output_dir": ("Save Models and Resume Training Settings", "Output Directory"),
            "output_name": ("Save Models and Resume Training Settings", "Output Name"),
            "save_every": ("Save Models and Resume Training Settings", "Save Frequency"),
            "resume": ("Save Models and Resume Training Settings", "Resume Training"),
            "save_state": ("Save Models and Resume Training Settings", "Save Optimizer State"),
            "mem_eff_save": ("Save Models and Resume Training Settings", "Memory Efficient Save"),

            # Accelerate
            "mixed_precision": ("Accelerate launch Settings", "Mixed Precision"),
            "bf16": ("Accelerate launch Settings", "BF16 Mixed Precision"),
            "num_processes": ("Accelerate launch Settings", "Number of Processes"),
            "multi_gpu": ("Accelerate launch Settings", "Multi GPU"),
            "gpu_ids": ("Accelerate launch Settings", "GPU IDs"),
            "dynamo": ("Accelerate launch Settings", "Dynamo Backend"),

            # Metadata
            "metadata": ("Metadata Settings", "Model Metadata"),
            "author": ("Metadata Settings", "Metadata Author"),
            "description": ("Metadata Settings", "Metadata Description"),
            "license": ("Metadata Settings", "Metadata License"),
            "tags": ("Metadata Settings", "Metadata Tags"),

            # HuggingFace
            "huggingface": ("HuggingFace Settings", "HuggingFace Upload"),
            "repo_id": ("HuggingFace Settings", "Repository ID"),
            "huggingface_token": ("HuggingFace Settings", "HuggingFace Token"),

            # Advanced
            "additional_parameters": ("Advanced Settings", "Additional Parameters"),
            "debug": ("Advanced Settings", "Debug Mode"),
        }

        # Search through parameter map
        for param, (location, display_name) in parameter_map.items():
            if query_lower in param.lower() or query_lower in display_name.lower() or param.lower() in query_lower:
                score = 0
                if query_lower == param.lower():
                    score = 100
                elif param.lower().startswith(query_lower):
                    score = 80
                elif query_lower in param.lower():
                    score = 60
                elif query_lower in display_name.lower():
                    score = 40

                results.append((location, display_name, param, score))

        # Sort by score
        results.sort(key=lambda x: x[3], reverse=True)

        # Remove duplicates
        seen = set()
        unique_results = []
        for item in results:
            key = (item[0], item[1])
            if key not in seen:
                seen.add(key)
                unique_results.append(item[:3])

        if not unique_results:
            html = f"<div style='padding: 10px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;'>"
            html += f"<strong>No results found for '{query}'</strong><br>"
            html += f"<span style='color: #666; font-size: 0.9em;'>Try: learning, optimizer, fp8, vram, epochs, batch, cache, sample, zimage, turbo</span>"
            html += "</div>"
            return gr.Row(visible=True), html

        # Format results
        html = f"<div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>"
        html += f"<strong>Found {len(unique_results)} result(s) for '{query}':</strong><br><br>"

        unique_panels = set()
        for location, display_name, param in unique_results:
            panel_name = location.split('→')[0].strip()
            unique_panels.add(panel_name)

        html += f"<div style='margin-bottom: 10px; padding: 8px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 3px; color: #155724;'>"
        html += f"✅ <strong>Opening {len(unique_panels)} relevant panel{'s' if len(unique_panels) > 1 else ''}:</strong> "
        html += ", ".join(sorted(unique_panels))
        html += "</div>"

        for location, display_name, param in unique_results[:10]:
            html += f"<div style='margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px; border-left: 3px solid #007bff;'>"
            html += f"📍 <strong>{location}</strong><br>"
            html += f"<span style='margin-left: 20px; color: #333;'>→ {display_name}</span>"
            html += f"</div>"

        if len(unique_results) > 10:
            html += f"<div style='color: #666; margin-top: 5px;'>... and {len(unique_results) - 10} more results</div>"

        html += "</div>"

        return gr.Row(visible=True), html

    # Modified search functionality to open relevant panels
    def search_and_open_panels(query):
        if not query or len(query.strip()) < 1:
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            return [gr.Row(visible=False), "", gr.Button(value="Open All Panels"), gr.Button(value="Open All Panels"), "closed"] + accordion_states

        results_row, results_html = search_settings(query)

        panels_to_open = set()

        panel_map = {
            "Accelerate launch Settings": 0,
            "Save Models and Resume Training Settings": 1,
            "Z-Image Training Dataset": 2,
            "Z-Image Model Settings": 3,
            "Torch Compile Settings": 4,
            "Flow Matching and Timestep Settings": 5,
            "Training Settings": 6,
            "Sample Generation Settings": 7,
            "Caching Settings": 8,
            "Learning Rate, Optimizer and Scheduler Settings": 9,
            "LoRA Settings": 10,
            "Advanced Settings": 11,
            "Metadata Settings": 12,
            "HuggingFace Settings": 13,
        }

        import re
        panel_pattern = r'📍 <strong>([^<]+)</strong>'
        matches = re.findall(panel_pattern, results_html)

        for match in matches:
            base_panel = match.split('→')[0].strip()
            if base_panel in panel_map:
                panels_to_open.add(panel_map[base_panel])

        accordion_states = []
        for i in range(len(accordions)):
            if i in panels_to_open:
                accordion_states.append(gr.Accordion(open=True))
            else:
                accordion_states.append(gr.Accordion(open=False))

        if panels_to_open:
            button_text = f"Reset Search ({len(panels_to_open)} panel{'s' if len(panels_to_open) > 1 else ''} filtered)"
            state = "search"
        else:
            button_text = "Open All Panels"
            state = "closed"

        return [gr.Row(visible=False), "", gr.Button(value=button_text), gr.Button(value=button_text), state] + accordion_states

    search_input.change(
        search_and_open_panels,
        inputs=[search_input],
        outputs=[search_results_row, search_results, toggle_all_btn, toggle_all_btn_bottom, panels_state] + accordions,
        show_progress=False,
    )

    # Add handler for unified toggle button
    def toggle_all_panels(current_state):
        if current_state == "search":
            new_state = "closed"
            new_button_text = "Open All Panels"
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            search_value = ""
            results_visibility = gr.Row(visible=False)
            results_content = ""
        elif current_state == "closed":
            new_state = "open"
            new_button_text = "Hide All Panels"
            accordion_states = [gr.Accordion(open=True) for _ in accordions]
            search_value = gr.Textbox(value="")
            results_visibility = gr.Row(visible=False)
            results_content = ""
        else:
            new_state = "closed"
            new_button_text = "Open All Panels"
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            search_value = gr.Textbox(value="")
            results_visibility = gr.Row(visible=False)
            results_content = ""

        return [new_state, gr.Button(value=new_button_text), gr.Button(value=new_button_text), search_value, results_visibility, results_content] + accordion_states

    toggle_all_btn.click(
        toggle_all_panels,
        inputs=[panels_state],
        outputs=[panels_state, toggle_all_btn, toggle_all_btn_bottom, search_input, search_results_row, search_results] + accordions,
        show_progress=False,
    )

    toggle_all_btn_bottom.click(
        toggle_all_panels,
        inputs=[panels_state],
        outputs=[panels_state, toggle_all_btn, toggle_all_btn_bottom, search_input, search_results_row, search_results] + accordions,
        show_progress=False,
    )

    configuration.button_open_config.click(
        zimage_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )
    configuration.button_load_config.click(
        zimage_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )
    configuration.button_save_config.click(
        zimage_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )

    button_print.click(
        zimage_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        zimage_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[executor.button_run, executor.stop_row, executor.button_stop_training, executor.training_status, run_state],
        show_progress=False,
    )

    executor.button_stop_training.click(
        executor.kill_command,
        inputs=[],
        outputs=[executor.button_run, executor.stop_row, executor.button_stop_training, executor.training_status],
        js="() => { if (confirm('Stop training/caching?')) { return []; } else { throw new Error('Cancelled'); } }",
    )

    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.stop_row, executor.button_stop_training, executor.training_status],
        show_progress=False,
    )
