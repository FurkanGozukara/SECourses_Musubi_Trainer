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


def _generate_flux2_dataset_toml(
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
    control_directory_name: str,
    no_resize_control: bool,
    control_resolution_width: int,
    control_resolution_height: int,
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

        control_resolution = None
        try:
            crw = int(control_resolution_width or 0)
            crh = int(control_resolution_height or 0)
            if crw > 0 and crh > 0:
                control_resolution = (crw, crh)
        except Exception:
            control_resolution = None

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
            control_directory_name=control_directory_name or "control_images",
            no_resize_control=bool(no_resize_control),
            control_resolution=control_resolution,
        )

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(save_dir, f"flux2_dataset_{ts}.toml")
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


def save_flux2_configuration(save_as_bool, file_path, parameters):
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


def open_flux2_configuration(ask_for_file, file_path, parameters):
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
        "control_resolution_width",
        "control_resolution_height",
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
        "save_every_n_epochs",
        "save_every_n_steps",
        "save_last_n_epochs",
        "save_last_n_steps",
        "save_last_n_epochs_state",
        "save_last_n_steps_state",
        "ddp_timeout",
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


def _maybe_create_enhanced_sample_prompts(param_dict: dict, parameters: list[tuple[str, object]]) -> tuple[dict, list[tuple[str, object]]]:
    """
    Musubi-tuner sample generation reads per-line settings from the prompt file (kohya format).
    If the user provides a plain prompt file without flags, sampling falls back to small 256x256 defaults.
    We write an enhanced prompt file with sane defaults and point sample_prompts to it.
    """
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
    steps = int(param_dict.get("sample_steps") or 50)
    guidance = float(param_dict.get("sample_guidance_scale") or 4.0)
    seed = param_dict.get("sample_seed", None)
    neg = (param_dict.get("sample_negative_prompt") or "").strip()

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
            if not has_flag(s, "g"):
                s += f" --g {guidance}"
            if seed is not None and int(seed) >= 0 and not has_flag(s, "d"):
                s += f" --d {int(seed)}"
            if neg and not has_flag(s, "n"):
                s += f" --n {neg}"

            enhanced_lines.append(s)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_name = (param_dict.get("output_name") or "flux2").strip() or "flux2"
    enhanced_path = os.path.join(sample_out_dir, f"{out_name}_enhanced_prompts_{ts}.txt")
    with open(enhanced_path, "w", encoding="utf-8") as f:
        f.write("# Enhanced prompt file generated by SECourses Musubi Trainer\n")
        f.write(f"# Original file: {sample_prompts}\n\n")
        for l in enhanced_lines:
            f.write(l + "\n")

    param_dict["sample_prompts"] = enhanced_path
    parameters = [(k, (enhanced_path if k == "sample_prompts" else v)) for k, v in parameters]
    return param_dict, parameters


def train_flux2_model(headless: bool, print_only: bool, parameters):
    global train_state_value

    python_cmd = sys.executable
    run_cmd = _find_accelerate_launch(python_cmd)

    param_dict = dict(parameters)

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

    # This tab is for FLUX.2 dev.
    model_version = "dev"
    param_dict["model_version"] = model_version
    parameters = [(k, ("dev" if k == "model_version" else v)) for k, v in parameters]

    dit_path = (param_dict.get("dit") or "").strip()
    vae_path = (param_dict.get("vae") or "").strip()
    te_path = (param_dict.get("text_encoder") or "").strip()

    if not dit_path or not os.path.exists(dit_path):
        raise ValueError("[ERROR] DiT checkpoint path is required and must exist.")
    if not vae_path or not os.path.exists(vae_path):
        raise ValueError("[ERROR] AE/VAE checkpoint path is required and must exist.")
    if not te_path or not os.path.exists(te_path):
        raise ValueError("[ERROR] Text Encoder checkpoint path is required and must exist.")

    output_dir = (param_dict.get("output_dir") or "").strip()
    output_name = (param_dict.get("output_name") or "").strip()
    if not output_dir:
        raise ValueError("[ERROR] Output directory is required.")
    if not output_name:
        raise ValueError("[ERROR] Output name is required.")

    # Enforce correct LoRA module.
    required_network_module = "networks.lora_flux_2"
    if (param_dict.get("network_module") or "").strip() != required_network_module:
        param_dict["network_module"] = required_network_module
        parameters = [(k, (required_network_module if k == "network_module" else v)) for k, v in parameters]

    # Optional pre-steps
    latent_cache_cmd = None
    teo_cache_cmd = None

    # Match WAN tab behavior: run caching unless explicitly disabled.
    # - True: run and pass --skip_existing
    # - False: do not run caching
    # - missing/None: run without --skip_existing (overwrite/recompute)
    if param_dict.get("caching_latent_skip_existing") is not False:
        latent_cache_cmd = [
            python_cmd,
            f"{scriptdir}/musubi-tuner/src/musubi_tuner/flux_2_cache_latents.py",
            "--model_version",
            model_version,
            "--dataset_config",
            dataset_config,
            "--vae",
            vae_path,
        ]
        vae_dtype = (param_dict.get("vae_dtype") or "").strip()
        if vae_dtype:
            latent_cache_cmd.extend(["--vae_dtype", vae_dtype])

        caching_device = (param_dict.get("caching_latent_device") or "cuda").strip()
        if caching_device == "cuda" and (param_dict.get("gpu_ids") or "").strip():
            gpu_ids = str(param_dict.get("gpu_ids")).split(",")
            caching_device = f"cuda:{gpu_ids[0].strip()}"
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
            f"{scriptdir}/musubi-tuner/src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
            "--model_version",
            model_version,
            "--dataset_config",
            dataset_config,
            "--text_encoder",
            te_path,
        ]

        # fp8_text_encoder is not supported for dev.
        if False:
            teo_cache_cmd.append("--fp8_text_encoder")

        teo_device = (param_dict.get("caching_teo_device") or "cuda").strip()
        if teo_device == "cuda" and (param_dict.get("gpu_ids") or "").strip():
            gpu_ids = str(param_dict.get("gpu_ids")).split(",")
            teo_device = f"cuda:{gpu_ids[0].strip()}"
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

    run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/flux_2_train_network.py")

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

    # Improve sample generation defaults by enhancing the prompt file if provided.
    param_dict, parameters = _maybe_create_enhanced_sample_prompts(param_dict, parameters)

    # Exclude caching keys from training config file.
    pattern_exclusion = [k for k, _ in parameters if k.startswith("caching_latent_") or k.startswith("caching_teo_")]

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
            "control_directory_name",
            "no_resize_control",
            "control_resolution_width",
            "control_resolution_height",
            "caching_teo_fp8_text_encoder",
            "fp8_text_encoder",
        ]
        + pattern_exclusion,
        mandatory_keys=["dataset_config", "dit", "vae", "text_encoder", "model_version", "network_module"],
    )

    run_cmd.append("--config_file")
    run_cmd.append(cfg_path)

    # Debug mode and extra args
    additional_params = (param_dict.get("additional_parameters") or "").strip()
    debug_mode_selected = (param_dict.get("debug_mode") or "None").strip()
    if debug_mode_selected and debug_mode_selected != "None":
        debug_params = _get_debug_parameters_for_mode(debug_mode_selected)
        if debug_params:
            additional_params = (additional_params + " " + debug_params).strip() if additional_params else debug_params

    run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, additional_parameters=additional_params)

    env = setup_environment()

    # If caching is enabled, run caching + training in one wrapper script so Stop cancels everything.
    if latent_cache_cmd or teo_cache_cmd:
        import platform
        import tempfile

        cmds = []
        if latent_cache_cmd:
            cmds.append(("Latent caching", latent_cache_cmd))
        if teo_cache_cmd:
            cmds.append(("Text encoder output caching", teo_cache_cmd))
        cmds.append(("Training", run_cmd))

        if platform.system() == "Windows":
            script_ext = ".bat"
            lines = ["@echo off"]
            for title, cmd in cmds:
                cmd_str = " ".join([f"\"{arg}\"" if " " in str(arg) else str(arg) for arg in cmd])
                lines.append(f"echo Starting {title}...")
                lines.append(cmd_str)
                lines.append("if %errorlevel% neq 0 (")
                lines.append(f"  echo {title} failed with error code %errorlevel%")
                lines.append("  exit /b %errorlevel%")
                lines.append(")")
            script_content = "\n".join(lines) + "\n"
        else:
            script_ext = ".sh"
            lines = ["#!/bin/bash", "set -e"]
            for title, cmd in cmds:
                lines.append(f'echo "Starting {title}..."')
                lines.append(shlex.join(cmd))
            script_content = "\n".join(lines) + "\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=script_ext, delete=False, encoding="utf-8") as f:
            temp_script = f.name
            f.write(script_content)

        if platform.system() != "Windows":
            import stat

            os.chmod(temp_script, os.stat(temp_script).st_mode | stat.S_IEXEC)

        save_executed_script(script_content=script_content, config_name=output_name, script_type="flux2")
        final_cmd = [temp_script] if platform.system() == "Windows" else ["bash", temp_script]
        executor.execute_command(run_cmd=final_cmd, env=env, shell=True if platform.system() == "Windows" else False)
    else:
        training_script = generate_script_content(run_cmd, "FLUX.2 training")
        save_executed_script(script_content=training_script, config_name=output_name, script_type="flux2")
        executor.execute_command(run_cmd=run_cmd, env=env)

    train_state_value = time.time()
    return (
        gr.Button(visible=False or headless),
        gr.Row(visible=True),
        gr.Button(interactive=True),
        gr.Textbox(value="Training in progress..."),
        gr.Textbox(value=train_state_value),
    )


def flux2_gui_actions(action: str, ask_for_file: bool, config_file_name: str, headless: bool, print_only: bool, *args):
    if action == "open_configuration":
        return open_flux2_configuration(ask_for_file, config_file_name, list(zip(FLUX2_PARAM_KEYS, args)))
    if action == "save_configuration":
        return save_flux2_configuration(ask_for_file, config_file_name, list(zip(FLUX2_PARAM_KEYS, args)))
    if action == "train_model":
        return train_flux2_model(headless=headless, print_only=print_only, parameters=list(zip(FLUX2_PARAM_KEYS, args)))


FLUX2_PARAM_KEYS = [
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
    "control_directory_name",
    "no_resize_control",
    "control_resolution_width",
    "control_resolution_height",
    # model
    "model_version",
    "dit",
    "vae",
    "vae_dtype",
    "text_encoder",
    "fp8_base",
    "fp8_scaled",
    "fp8_text_encoder",
    "disable_numpy_memmap",
    "blocks_to_swap",
    "use_pinned_memory_for_block_swap",
    "img_in_txt_in_offloading",
    # schedule
    "timestep_sampling",
    "weighting_scheme",
    "discrete_flow_shift",
    "sigmoid_scale",
    "min_timestep",
    "max_timestep",
    # training_settings
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
    "caching_teo_fp8_text_encoder",
    # optimizer and scheduler
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


def flux2_lora_tab(headless=False, config: GUIConfig = {}):
    global executor

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    defaults = {
        "model_version": "dev",
        "network_module": "networks.lora_flux_2",
        "output_name": "my-flux2-lora",
        "mixed_precision": "bf16",
        "num_cpu_threads_per_process": 1,
        "sdpa": True,
        "optimizer_type": "adamw8bit",
        "learning_rate": 1e-4,
        "gradient_checkpointing": True,
        "timestep_sampling": "flux2_shift",
        "weighting_scheme": "none",
        "dataset_config_mode": "Generate from Folder Structure",
        "dataset_resolution_width": 1024,
        "dataset_resolution_height": 1024,
        "dataset_enable_bucket": True,
        "dataset_cache_directory": "cache_dir",
        "control_directory_name": "control_images",
        "no_resize_control": True,
        "control_resolution_width": 2024,
        "control_resolution_height": 2024,
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
        "caching_teo_fp8_text_encoder": False,
    }

    # Align with other tabs: accept either a dict or a GUIConfig.
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

    with gr.Accordion("Accelerate Launch", open=False):
        accelerate_launch = AccelerateLaunch(config=config)

    with gr.Accordion("Save / Resume", open=False):
        save_load = SaveLoadSettings(headless=headless, config=config)

    with gr.Accordion("Advanced", open=False):
        advanced = AdvancedTraining(headless=headless, config=config)

    with gr.Accordion("Dataset", open=False):
        with gr.Row():
            dataset_config_mode = gr.Dropdown(
                label="Dataset Config Mode",
                choices=["Generate from Folder Structure", "Use TOML File"],
                value=config.get("dataset_config_mode", "Generate from Folder Structure"),
                interactive=True,
            )

        with gr.Row():
            dataset_config = gr.Textbox(
                label="dataset_config",
                value=config.get("dataset_config", ""),
                placeholder="Path to dataset config TOML",
                interactive=True,
            )
            dataset_config_btn = gr.Button("Browse", size="lg", visible=not headless)
            dataset_config_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".toml", extension_name="TOML files"),
                outputs=[dataset_config],
            )

        with gr.Row():
            parent_folder_path = gr.Textbox(
                label="parent_folder_path",
                value=config.get("parent_folder_path", ""),
                placeholder="Folder containing subfolders like 1_concept, 10_concept2...",
                interactive=True,
            )
            parent_folder_btn = gr.Button("Browse", size="lg", visible=not headless)
            parent_folder_btn.click(fn=lambda: get_folder_path(folder_path=""), outputs=[parent_folder_path])

        with gr.Row():
            dataset_resolution_width = gr.Number(
                label="dataset_resolution_width",
                value=config.get("dataset_resolution_width", 1024),
                step=8,
                interactive=True,
            )
            dataset_resolution_height = gr.Number(
                label="dataset_resolution_height",
                value=config.get("dataset_resolution_height", 1024),
                step=8,
                interactive=True,
            )

        with gr.Row():
            dataset_caption_extension = gr.Textbox(
                label="dataset_caption_extension",
                value=config.get("dataset_caption_extension", ".txt"),
                interactive=True,
            )
            create_missing_captions = gr.Checkbox(
                label="create_missing_captions",
                value=bool(config.get("create_missing_captions", True)),
            )
            caption_strategy = gr.Dropdown(
                label="caption_strategy",
                choices=["folder_name", "empty"],
                value=config.get("caption_strategy", "folder_name"),
                interactive=True,
            )

        with gr.Row():
            dataset_batch_size = gr.Number(
                label="dataset_batch_size",
                value=config.get("dataset_batch_size", 1),
                minimum=1,
                step=1,
                interactive=True,
            )
            dataset_enable_bucket = gr.Checkbox(
                label="dataset_enable_bucket",
                value=bool(config.get("dataset_enable_bucket", True)),
            )
            dataset_bucket_no_upscale = gr.Checkbox(
                label="dataset_bucket_no_upscale",
                value=bool(config.get("dataset_bucket_no_upscale", False)),
            )

        with gr.Row():
            dataset_cache_directory = gr.Textbox(
                label="dataset_cache_directory",
                value=config.get("dataset_cache_directory", "cache_dir"),
                interactive=True,
            )
            generated_toml_path = gr.Textbox(
                label="generated_toml_path",
                value=config.get("generated_toml_path", ""),
                interactive=False,
            )

        with gr.Accordion("Control Images (optional)", open=False):
            with gr.Row():
                control_directory_name = gr.Textbox(
                    label="control_directory_name",
                    value=config.get("control_directory_name", "control_images"),
                    interactive=True,
                )
                no_resize_control = gr.Checkbox(
                    label="no_resize_control",
                    value=bool(config.get("no_resize_control", True)),
                )
            with gr.Row():
                control_resolution_width = gr.Number(
                    label="control_resolution_width",
                    value=config.get("control_resolution_width", 2024),
                    step=8,
                    interactive=True,
                )
                control_resolution_height = gr.Number(
                    label="control_resolution_height",
                    value=config.get("control_resolution_height", 2024),
                    step=8,
                    interactive=True,
                )

        with gr.Row():
            dataset_status = gr.Textbox(label="Dataset status", value="", interactive=False)

        with gr.Row():
            generate_toml_btn = gr.Button("Generate dataset TOML", variant="primary")
            generate_toml_btn.click(
                fn=_generate_flux2_dataset_toml,
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
                    control_directory_name,
                    no_resize_control,
                    control_resolution_width,
                    control_resolution_height,
                    save_load.output_dir,
                ],
                outputs=[dataset_config, generated_toml_path, dataset_status],
            )

    with gr.Accordion("Model", open=False):
        with gr.Row():
            model_version = gr.Dropdown(
                label="model_version",
                choices=[("dev (Mistral 3)", "dev")],
                value="dev",
                interactive=False,
            )
        with gr.Row():
            dit = gr.Textbox(label="dit", value=config.get("dit", ""), interactive=True)
            dit_btn = gr.Button("Browse", size="lg", visible=not headless)
            dit_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"),
                outputs=[dit],
            )
        with gr.Row():
            vae = gr.Textbox(label="vae", value=config.get("vae", ""), interactive=True)
            vae_btn = gr.Button("Browse", size="lg", visible=not headless)
            vae_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"),
                outputs=[vae],
            )
        with gr.Row():
            vae_dtype = gr.Dropdown(
                label="vae_dtype",
                choices=["", "float32", "bfloat16"],
                value=config.get("vae_dtype", ""),
                interactive=True,
            )
        with gr.Row():
            text_encoder = gr.Textbox(label="text_encoder", value=config.get("text_encoder", ""), interactive=True)
            te_btn = gr.Button("Browse", size="lg", visible=not headless)
            te_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"),
                outputs=[text_encoder],
            )
        with gr.Row():
            fp8_base = gr.Checkbox(label="fp8_base", value=bool(config.get("fp8_base", False)))
            fp8_scaled = gr.Checkbox(label="fp8_scaled", value=bool(config.get("fp8_scaled", False)))
            fp8_text_encoder = gr.Checkbox(
                label="fp8_text_encoder",
                value=False,
                interactive=False,
                info="Not supported for FLUX.2 dev.",
            )
        with gr.Row():
            disable_numpy_memmap = gr.Checkbox(label="disable_numpy_memmap", value=bool(config.get("disable_numpy_memmap", False)))
            blocks_to_swap = gr.Number(label="blocks_to_swap", value=config.get("blocks_to_swap", 0), minimum=0, step=1, interactive=True)
            use_pinned_memory_for_block_swap = gr.Checkbox(
                label="use_pinned_memory_for_block_swap",
                value=bool(config.get("use_pinned_memory_for_block_swap", False)),
            )
            img_in_txt_in_offloading = gr.Checkbox(
                label="img_in_txt_in_offloading",
                value=bool(config.get("img_in_txt_in_offloading", False)),
            )

    with gr.Accordion("Flow / Schedule", open=False):
        with gr.Row():
            timestep_sampling = gr.Dropdown(
                label="timestep_sampling",
                choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "qwen_shift", "logsnr", "qinglong_flux", "qinglong_qwen"],
                value=config.get("timestep_sampling", "flux2_shift"),
                interactive=True,
            )
            weighting_scheme = gr.Dropdown(
                label="weighting_scheme",
                choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
                value=config.get("weighting_scheme", "none"),
                interactive=True,
            )
        with gr.Row():
            discrete_flow_shift = gr.Number(
                label="discrete_flow_shift",
                value=config.get("discrete_flow_shift", 1.0),
                step=0.1,
                interactive=True,
            )
            sigmoid_scale = gr.Number(
                label="sigmoid_scale",
                value=config.get("sigmoid_scale", 1.0),
                step=0.1,
                interactive=True,
            )
        with gr.Row():
            min_timestep = gr.Number(label="min_timestep", value=config.get("min_timestep", 0), step=1, interactive=True)
            max_timestep = gr.Number(label="max_timestep", value=config.get("max_timestep", 1000), step=1, interactive=True)

    with gr.Accordion("Training", open=False):
        training = TrainingSettings(headless=headless, config=config)

    with gr.Accordion("Samples (optional)", open=False):
        with gr.Row():
            sample_every_n_steps = gr.Number(label="sample_every_n_steps", value=config.get("sample_every_n_steps", 0), minimum=0, step=1, interactive=True)
            sample_every_n_epochs = gr.Number(label="sample_every_n_epochs", value=config.get("sample_every_n_epochs", 0), minimum=0, step=1, interactive=True)
            sample_at_first = gr.Checkbox(label="sample_at_first", value=bool(config.get("sample_at_first", False)))
        with gr.Row():
            sample_prompts = gr.Textbox(label="sample_prompts", value=config.get("sample_prompts", ""), interactive=True)
            sample_prompts_btn = gr.Button("Browse", size="lg", visible=not headless)
            sample_prompts_btn.click(fn=lambda: get_file_path(file_path="", default_extension=".txt", extension_name="Text files"), outputs=[sample_prompts])
        with gr.Row():
            sample_output_dir = gr.Textbox(label="sample_output_dir", value=config.get("sample_output_dir", ""), interactive=True)
            sample_output_dir_btn = gr.Button("Browse", size="lg", visible=not headless)
            sample_output_dir_btn.click(fn=lambda: get_folder_path(folder_path=""), outputs=[sample_output_dir])
        with gr.Row():
            sample_width = gr.Number(label="sample_width", value=config.get("sample_width", 1024), step=8, interactive=True)
            sample_height = gr.Number(label="sample_height", value=config.get("sample_height", 1024), step=8, interactive=True)
        with gr.Row():
            sample_steps = gr.Number(label="sample_steps", value=config.get("sample_steps", 50), step=1, interactive=True)
            sample_guidance_scale = gr.Number(label="sample_guidance_scale", value=config.get("sample_guidance_scale", 4.0), step=0.1, interactive=True)
            sample_seed = gr.Number(label="sample_seed", value=config.get("sample_seed", 42), step=1, interactive=True)
        with gr.Row():
            sample_negative_prompt = gr.Textbox(label="sample_negative_prompt", value=config.get("sample_negative_prompt", ""), interactive=True)

    with gr.Accordion("Caching (optional)", open=False):
        with gr.Row():
            caching_latent_device = gr.Textbox(label="caching_latent_device", value=config.get("caching_latent_device", "cuda"), interactive=True)
            caching_latent_batch_size = gr.Number(label="caching_latent_batch_size", value=config.get("caching_latent_batch_size", 1), step=1, interactive=True)
            caching_latent_num_workers = gr.Number(label="caching_latent_num_workers", value=config.get("caching_latent_num_workers", 2), step=1, interactive=True)
        with gr.Row():
            caching_latent_skip_existing = gr.Checkbox(label="caching_latent_skip_existing", value=bool(config.get("caching_latent_skip_existing", True)))
            caching_latent_keep_cache = gr.Checkbox(label="caching_latent_keep_cache", value=bool(config.get("caching_latent_keep_cache", True)))
        with gr.Row():
            caching_teo_device = gr.Textbox(label="caching_teo_device", value=config.get("caching_teo_device", "cuda"), interactive=True)
            caching_teo_batch_size = gr.Number(label="caching_teo_batch_size", value=config.get("caching_teo_batch_size", 8), step=1, interactive=True)
            caching_teo_num_workers = gr.Number(label="caching_teo_num_workers", value=config.get("caching_teo_num_workers", 2), step=1, interactive=True)
        with gr.Row():
            caching_teo_skip_existing = gr.Checkbox(label="caching_teo_skip_existing", value=bool(config.get("caching_teo_skip_existing", True)))
            caching_teo_keep_cache = gr.Checkbox(label="caching_teo_keep_cache", value=bool(config.get("caching_teo_keep_cache", True)))
            caching_teo_fp8_text_encoder = gr.Checkbox(
                label="caching_teo_fp8_text_encoder",
                value=False,
                interactive=False,
                info="Not supported for FLUX.2 dev.",
            )

    with gr.Accordion("Optimizer / Scheduler", open=False):
        optim = OptimizerAndScheduler(headless=headless, config=config)

    with gr.Accordion("Network (LoRA)", open=False):
        network = Network(headless=headless, config=config)

    with gr.Accordion("Metadata", open=False):
        metadata = MetaData(config=config)

    with gr.Accordion("HuggingFace", open=False):
        huggingface = HuggingFace(config=config)

    executor = CommandExecutor(headless=headless)
    run_state = gr.Textbox(value=train_state_value, visible=False)
    button_print = gr.Button("Print command (no run)", variant="secondary")

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
        control_directory_name,
        no_resize_control,
        control_resolution_width,
        control_resolution_height,
        # model
        model_version,
        dit,
        vae,
        vae_dtype,
        text_encoder,
        fp8_base,
        fp8_scaled,
        fp8_text_encoder,
        disable_numpy_memmap,
        blocks_to_swap,
        use_pinned_memory_for_block_swap,
        img_in_txt_in_offloading,
        # schedule
        timestep_sampling,
        weighting_scheme,
        discrete_flow_shift,
        sigmoid_scale,
        min_timestep,
        max_timestep,
        # training_settings
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
        caching_teo_fp8_text_encoder,
        # optimizer and scheduler
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

    # Sanity: keep the wiring correct when this file is edited.
    assert len(settings_list) == len(FLUX2_PARAM_KEYS), f"settings_list ({len(settings_list)}) != FLUX2_PARAM_KEYS ({len(FLUX2_PARAM_KEYS)})"

    configuration.button_open_config.click(
        flux2_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )
    configuration.button_load_config.click(
        flux2_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )
    configuration.button_save_config.click(
        flux2_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )

    button_print.click(
        flux2_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        flux2_gui_actions,
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
