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
    list_files,
    normalize_path,
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

# Model version choices
_FLUX2_MODEL_VERSIONS = [("dev (Mistral 3)", "dev")]
_KLEIN_MODEL_VERSIONS = [
    ("klein-4b (distilled)", "klein-4b"),
    ("klein-base-4b", "klein-base-4b"),
    ("klein-9b (distilled)", "klein-9b"),
    ("klein-base-9b", "klein-base-9b"),
]


def _get_debug_parameters_for_mode(debug_mode: str) -> str:
    debug_params = {
        "Show Timesteps (Image)": "--show_timesteps image",
        "Show Timesteps (Console)": "--show_timesteps console",
        "RCM Debug Save": "--rcm_debug_save",
        "Enable Logging (TensorBoard)": "--log_with tensorboard",
        "Enable Logging (WandB)": "--log_with wandb",
        "Enable Logging (All)": "--log_with all",
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


def _generate_flux_dataset_toml(
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
        out_path = os.path.join(save_dir, f"flux_dataset_{ts}.toml")
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


def save_flux_configuration(save_as_bool, file_path, parameters):
    original_file_path = file_path

    if save_as_bool or not file_path:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")

    if not file_path:
        return gr.update(value=original_file_path), gr.update(value="No file selected.", visible=True)

    def _config_file_dropdown_update(value: str):
        value = normalize_path(value) if isinstance(value, str) and value else value
        directory = value
        if isinstance(directory, str) and directory:
            try:
                if not os.path.isdir(directory):
                    directory = os.path.dirname(directory) or "."
            except OSError:
                directory = os.path.dirname(directory) or "."
        else:
            directory = "."

        choices = [""]
        try:
            choices.extend(list(list_files(directory, exts=[".toml", ".json"], all=True)))
        except Exception:
            pass

        if isinstance(value, str) and value and value not in choices:
            choices.insert(1, value)

        return gr.update(value=value, choices=choices)

    try:
        SaveConfigFile(
            parameters=parameters,
            file_path=file_path,
            exclusion=["file_path", "save_as", "save_as_bool", "headless", "print_only"],
        )
        msg = f"Configuration saved: {os.path.basename(file_path)}"
        gr.Info(msg)
        return _config_file_dropdown_update(file_path), gr.update(value=msg, visible=True)
    except Exception as e:
        msg = f"Failed to save configuration: {e}"
        log.error(msg)
        gr.Error(msg)
        return _config_file_dropdown_update(original_file_path), gr.update(value=msg, visible=True)


def open_flux_configuration(ask_for_file, file_path, parameters):
    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path_or_save_as(file_path, default_extension=".toml", extension_name="TOML files")

    if not file_path:
        values = [original_file_path, gr.update(value="", visible=False)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    def _config_file_dropdown_update(value: str):
        value = normalize_path(value) if isinstance(value, str) and value else value
        directory = value
        if isinstance(directory, str) and directory:
            try:
                if not os.path.isdir(directory):
                    directory = os.path.dirname(directory) or "."
            except OSError:
                directory = os.path.dirname(directory) or "."
        else:
            directory = "."

        choices = [""]
        try:
            choices.extend(list(list_files(directory, exts=[".toml", ".json"], all=True)))
        except Exception:
            pass

        if isinstance(value, str) and value and value not in choices:
            choices.insert(1, value)

        return gr.update(value=value, choices=choices)

    if ask_for_file and not os.path.isfile(file_path):
        msg = f"New configuration file will be created at: {os.path.basename(file_path)}"
        gr.Info(msg)
        values = [_config_file_dropdown_update(file_path), gr.update(value=msg, visible=True)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    if not os.path.isfile(file_path):
        msg = f"Config file does not exist: {file_path}"
        gr.Error(msg)
        values = [_config_file_dropdown_update(original_file_path), gr.update(value=msg, visible=True)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
    except Exception as e:
        msg = f"Failed to load configuration: {e}"
        log.error(msg)
        gr.Error(msg)
        values = [_config_file_dropdown_update(original_file_path), gr.update(value=msg, visible=True)]
        values.extend([v for _, v in parameters])
        return tuple(values)

    # Some older presets may not include model_family. Infer it from model_version when possible so the UI can be updated.
    loaded_model_version_raw = (data.get("model_version") or "").strip()
    inferred_family = "FLUX Klein" if loaded_model_version_raw in {v for _, v in _KLEIN_MODEL_VERSIONS} else None
    loaded_model_family = (data.get("model_family") or inferred_family or "").strip() or None
    if loaded_model_family not in {"FLUX.2", "FLUX Klein"}:
        loaded_model_family = None

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
        "compile_cache_size_limit",
    }

    list_to_str_fields = {"optimizer_args", "lr_scheduler_args", "network_args"}

    loaded_values = []
    for key, default_value in parameters:
        if key == "model_family":
            # Ensure the model_family output is always normalized for downstream UI updates.
            v = (data.get(key) or loaded_model_family or default_value or "FLUX.2").strip()
            if v not in {"FLUX.2", "FLUX Klein"}:
                v = "FLUX.2"
            loaded_values.append(v)
            continue

        if key in data:
            v = data[key]
            if isinstance(v, list) and key in numeric_fields:
                v = v[0] if v else None
            elif isinstance(v, list) and key in list_to_str_fields:
                v = " ".join(str(x) for x in v)
            loaded_values.append(v)
        else:
            loaded_values.append(default_value)

    # Prevent dropdown validation errors when loading Klein presets into the combined FLUX UI.
    # Gradio validates the loaded value against the component's current choices, so we must update choices + value together.
    try:
        family_idx = next(i for i, (k, _) in enumerate(parameters) if k == "model_family")
        mv_idx = next(i for i, (k, _) in enumerate(parameters) if k == "model_version")
    except StopIteration:
        family_idx = None
        mv_idx = None

    if family_idx is not None and mv_idx is not None:
        family_val = (loaded_values[family_idx] or "FLUX.2").strip()
        mv_val = (loaded_values[mv_idx] or "").strip()

        if family_val == "FLUX.2":
            loaded_values[mv_idx] = gr.update(choices=_FLUX2_MODEL_VERSIONS, value="dev", interactive=False)
        else:
            if mv_val not in {v for _, v in _KLEIN_MODEL_VERSIONS}:
                mv_val = "klein-base-9b"
            loaded_values[mv_idx] = gr.update(choices=_KLEIN_MODEL_VERSIONS, value=mv_val, interactive=True)

        # Keep other family-dependent controls consistent after loading (programmatic updates do not trigger .change).
        def _idx_for_key(search_key: str):
            try:
                return next(i for i, (k, _) in enumerate(parameters) if k == search_key)
            except StopIteration:
                return None

        def _maybe_set_checkbox(key: str, *, value: bool, interactive: bool, info: str):
            idx = _idx_for_key(key)
            if idx is None:
                return
            loaded_values[idx] = gr.update(value=value, interactive=interactive, info=info)

        if family_val == "FLUX.2":
            _maybe_set_checkbox(
                "fp8_text_encoder",
                value=False,
                interactive=False,
                info="Not supported for FLUX.2 dev. Available for FLUX Klein.",
            )
            _maybe_set_checkbox(
                "caching_teo_fp8_text_encoder",
                value=False,
                interactive=False,
                info="Not supported for FLUX.2 dev. Available for FLUX Klein.",
            )
        else:
            fp8_idx = _idx_for_key("fp8_text_encoder")
            teo_fp8_idx = _idx_for_key("caching_teo_fp8_text_encoder")
            fp8_default = loaded_values[fp8_idx] if fp8_idx is not None else False
            teo_fp8_default = loaded_values[teo_fp8_idx] if teo_fp8_idx is not None else False
            fp8_val = bool(data.get("fp8_text_encoder", fp8_default))
            teo_fp8_val = bool(data.get("caching_teo_fp8_text_encoder", teo_fp8_default))
            _maybe_set_checkbox(
                "fp8_text_encoder",
                value=fp8_val,
                interactive=True,
                info="FP8 for text encoder (supported in Klein)",
            )
            _maybe_set_checkbox(
                "caching_teo_fp8_text_encoder",
                value=teo_fp8_val,
                interactive=True,
                info="FP8 for text encoder caching (supported in Klein)",
            )

    msg = f"Loaded configuration: {os.path.basename(file_path)}"
    gr.Info(msg)
    return tuple([_config_file_dropdown_update(file_path), gr.update(value=msg, visible=True)] + loaded_values)


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
    if bool(param_dict.get("disable_prompt_enhancement", False)):
        log.info("FLUX prompt enhancement disabled; using original sample prompt file.")
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
        return re.search(rf"(?<!\S)--{re.escape(flag)}(?:\s+|=)", s, flags=re.IGNORECASE) is not None

    enhanced_lines = []
    with open(sample_prompts, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                enhanced_lines.append(line)
                continue

            s = stripped
            # Normalize common short/equals forms to canonical Kohya-style options that musubi parses.
            s = re.sub(r"(?<!\S)-(?P<flag>fs|w|h|s|d|l|n|g)(?=\s|=)", r"--\g<flag>", s, flags=re.IGNORECASE)
            s = re.sub(r"(?<!\S)--(?P<flag>fs|w|h|s|d|l|n|g)=(?P<value>[^\s]+)", r"--\g<flag> \g<value>", s, flags=re.IGNORECASE)
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
    out_name = (param_dict.get("output_name") or "flux").strip() or "flux"
    enhanced_path = os.path.join(sample_out_dir, f"{out_name}_enhanced_prompts_{ts}.txt")
    with open(enhanced_path, "w", encoding="utf-8") as f:
        f.write("# Enhanced prompt file generated by SECourses Musubi Trainer\n")
        f.write(f"# Original file: {sample_prompts}\n\n")
        for l in enhanced_lines:
            f.write(l + "\n")

    param_dict["sample_prompts"] = enhanced_path
    parameters = [(k, (enhanced_path if k == "sample_prompts" else v)) for k, v in parameters]
    return param_dict, parameters


def train_flux_model(headless: bool, print_only: bool, parameters):
    global train_state_value

    python_cmd = sys.executable
    run_cmd = _find_accelerate_launch(python_cmd)

    param_dict = dict(parameters)

    training_mode = (param_dict.get("training_mode") or "LoRA Training").strip()
    if training_mode != "LoRA Training":
        # FLUX training path uses flux_2_train_network.py which is LoRA/network training.
        # Some users select "DreamBooth Fine-Tuning" expecting it to mean "fine-tune"; instead of hard-failing,
        # fall back to LoRA training and keep going.
        gr.Warning("DreamBooth fine-tuning is not supported for FLUX. Continuing with LoRA Training.")
        training_mode = "LoRA Training"
        param_dict["training_mode"] = training_mode
        parameters = [(k, (training_mode if k == "training_mode" else v)) for k, v in parameters]

    # Prefer generated dataset config when using folder mode.
    dataset_config_mode = (param_dict.get("dataset_config_mode") or "").strip()
    if dataset_config_mode == "Generate from Folder Structure":
        gen_path = (param_dict.get("generated_toml_path") or "").strip()
        if gen_path:
            param_dict["dataset_config"] = gen_path
            parameters = [(k, (gen_path if k == "dataset_config" else v)) for k, v in parameters]

    dataset_config = (param_dict.get("dataset_config") or "").strip()
    if not dataset_config and dataset_config_mode == "Generate from Folder Structure":
        # Auto-generate the dataset TOML at train-time if the user selected folder mode but didn't click Generate.
        parent_folder_path = (param_dict.get("parent_folder_path") or "").strip()
        if parent_folder_path:
            out_path, gen_path, status = _generate_flux_dataset_toml(
                parent_folder_path=parent_folder_path,
                dataset_resolution_width=int(param_dict.get("dataset_resolution_width") or 1024),
                dataset_resolution_height=int(param_dict.get("dataset_resolution_height") or 1024),
                dataset_caption_extension=(param_dict.get("dataset_caption_extension") or ".txt"),
                create_missing_captions=bool(param_dict.get("create_missing_captions", True)),
                caption_strategy=(param_dict.get("caption_strategy") or "folder_name"),
                dataset_batch_size=int(param_dict.get("dataset_batch_size") or 1),
                dataset_enable_bucket=bool(param_dict.get("dataset_enable_bucket", True)),
                dataset_bucket_no_upscale=bool(param_dict.get("dataset_bucket_no_upscale", False)),
                dataset_cache_directory=(param_dict.get("dataset_cache_directory") or "cache_dir"),
                control_directory_name=(param_dict.get("control_directory_name") or "control_images"),
                no_resize_control=bool(param_dict.get("no_resize_control", True)),
                control_resolution_width=int(param_dict.get("control_resolution_width") or 2024),
                control_resolution_height=int(param_dict.get("control_resolution_height") or 2024),
                output_dir=(param_dict.get("output_dir") or ""),
            )
            if not out_path:
                raise ValueError(status or "[ERROR] Failed to auto-generate dataset TOML from folder structure.")

            gr.Info("Dataset TOML was auto-generated from folder structure for this run.")
            param_dict["dataset_config"] = out_path
            param_dict["generated_toml_path"] = gen_path
            dataset_config = out_path
            parameters = [(k, (out_path if k == "dataset_config" else v)) for k, v in parameters]
            parameters = [(k, (gen_path if k == "generated_toml_path" else v)) for k, v in parameters]

    if not dataset_config:
        raise ValueError("[ERROR] Dataset config is required. Generate a dataset TOML or set dataset_config.")
    if not os.path.exists(dataset_config):
        raise ValueError(f"[ERROR] Dataset config file does not exist: {dataset_config}")

    # Get model family and version
    model_family = (param_dict.get("model_family") or "FLUX.2").strip()
    model_version = (param_dict.get("model_version") or "dev").strip()

    # Validate model version based on family
    if model_family == "FLUX.2":
        model_version = "dev"
    else:  # FLUX Klein
        allowed = {v for _, v in _KLEIN_MODEL_VERSIONS}
        if model_version not in allowed:
            raise ValueError(f"[ERROR] Invalid model_version for FLUX Klein: {model_version}")

    param_dict["model_version"] = model_version
    parameters = [(k, (model_version if k == "model_version" else v)) for k, v in parameters]

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
            f"{scriptdir}/musubi-tuner/src/musubi_tuner/flux_2_cache_text_encoder_outputs.py",
            "--model_version",
            model_version,
            "--dataset_config",
            dataset_config,
            "--text_encoder",
            te_path,
        ]

        # fp8_text_encoder: only supported for Klein family
        if model_family == "FLUX Klein":
            fp8_te_for_cache = bool(param_dict.get("caching_teo_fp8_text_encoder", False) or param_dict.get("fp8_text_encoder", False))
            if fp8_te_for_cache:
                teo_cache_cmd.append("--fp8_text_encoder")

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

    # Exclude caching and GUI-only keys from training config file.
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
            "model_family",  # GUI-only selector
            "training_mode",  # GUI-only selector
            "disable_prompt_enhancement",
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

    # Match Qwen/WAN launch strategy:
    # 1) run latent caching synchronously (so failures surface before training),
    # 2) then run (optional) text-encoder caching + training via a wrapper script so Stop cancels both.
    env = setup_environment()

    script_type = "flux2" if model_family == "FLUX.2" else "flux_klein"

    if latent_cache_cmd:
        log.info("Running latent caching...")
        latent_cache_script = generate_script_content(latent_cache_cmd, f"{model_family} latent caching")
        save_executed_script(script_content=latent_cache_script, config_name=output_name, script_type=f"{script_type}_latent_cache")
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

        save_executed_script(script_content=script_content, config_name=output_name, script_type=script_type)
        final_cmd = [temp_script] if platform.system() == "Windows" else ["bash", temp_script]
        gr.Info("Starting text encoder caching followed by training...")
        executor.execute_command(run_cmd=final_cmd, env=env, shell=True if platform.system() == "Windows" else False)
    else:
        training_script = generate_script_content(run_cmd, f"{model_family} training")
        save_executed_script(script_content=training_script, config_name=output_name, script_type=script_type)
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


def flux_gui_actions(action: str, ask_for_file: bool, config_file_name: str, headless: bool, print_only: bool, *args):
    if action == "open_configuration":
        return open_flux_configuration(ask_for_file, config_file_name, list(zip(FLUX_PARAM_KEYS, args)))
    if action == "save_configuration":
        return save_flux_configuration(ask_for_file, config_file_name, list(zip(FLUX_PARAM_KEYS, args)))
    if action == "train_model":
        return train_flux_model(headless=headless, print_only=print_only, parameters=list(zip(FLUX_PARAM_KEYS, args)))


FLUX_PARAM_KEYS = [
    # model family selector (GUI only)
    "model_family",
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
    "training_mode",
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
    "disable_prompt_enhancement",
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


def flux_lora_tab(headless=False, config: GUIConfig = {}):
    global executor

    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    defaults = {
        "model_family": "FLUX.2",
        "training_mode": "LoRA Training",
        "model_version": "dev",
        "network_module": "networks.lora_flux_2",
        "output_name": "my-flux-lora",
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
        "fp8_text_encoder": False,
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
        # Sample defaults (FLUX.2)
        "disable_prompt_enhancement": False,
        "sample_steps": 50,
        "sample_guidance_scale": 4.0,
        # Torch compile defaults
        "compile": False,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_dynamic": "auto",
        "compile_fullgraph": False,
        "compile_cache_size_limit": 0,
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

    initial_model_family = (config.get("model_family", "FLUX.2") or "FLUX.2").strip()
    if initial_model_family not in {"FLUX.2", "FLUX Klein"}:
        initial_model_family = "FLUX.2"

    with gr.Accordion("Configuration file Settings", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    # Model Family Selector (Top level)
    with gr.Row():
        model_family = gr.Dropdown(
            label="🎯 Model Family",
            choices=["FLUX.2", "FLUX Klein"],
            value=config.get("model_family", "FLUX.2"),
            interactive=True,
            info="Select which FLUX model family to train",
        )

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

    dataset_accordion = gr.Accordion("FLUX Training Dataset", open=False, elem_classes="samples_background")
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
                placeholder="Path to dataset config TOML",
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
                    placeholder="Folder containing subfolders like 1_concept, 10_concept2...",
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
                    info="Target dataset width in pixels (multiples of 16 recommended for FLUX).",
                )
                dataset_resolution_height = gr.Number(
                    label="dataset_resolution_height",
                    value=config.get("dataset_resolution_height", 1024),
                    step=8,
                    interactive=True,
                    info="Target dataset height in pixels (multiples of 16 recommended for FLUX).",
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

            with gr.Accordion("Control Images (optional)", open=False):
                with gr.Row():
                    control_directory_name = gr.Textbox(
                        label="control_directory_name",
                        value=config.get("control_directory_name", "control_images"),
                        interactive=True,
                        info="Subfolder name for control/reference images (optional).",
                    )
                    no_resize_control = gr.Checkbox(
                        label="no_resize_control",
                        value=bool(config.get("no_resize_control", True)),
                        info="Do not resize control images during preprocessing.",
                    )
                with gr.Row():
                    control_resolution_width = gr.Number(
                        label="control_resolution_width",
                        value=config.get("control_resolution_width", 2024),
                        step=8,
                        interactive=True,
                        info="Optional control image width (set >0 to force resize).",
                    )
                    control_resolution_height = gr.Number(
                        label="control_resolution_height",
                        value=config.get("control_resolution_height", 2024),
                        step=8,
                        interactive=True,
                        info="Optional control image height (set >0 to force resize).",
                    )

            with gr.Row():
                dataset_status = gr.Textbox(label="Dataset status", value="", interactive=False, info="Validation and status messages.")

            with gr.Row():
                generate_toml_btn = gr.Button("Generate dataset TOML", variant="primary")
                generate_toml_btn.click(
                    fn=_generate_flux_dataset_toml,
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

        dataset_config_mode.change(
            fn=_toggle_dataset_mode,
            inputs=[dataset_config_mode],
            outputs=[toml_mode_row, folder_mode_column],
        )

    model_accordion = gr.Accordion("FLUX Model Settings", open=False, elem_classes="preset_background")
    accordions.append(model_accordion)
    with model_accordion:
        initial_model_version = (config.get("model_version") or "").strip()
        if initial_model_family == "FLUX.2":
            initial_model_version = "dev"
        else:
            if initial_model_version not in {v for _, v in _KLEIN_MODEL_VERSIONS}:
                initial_model_version = "klein-base-9b"

        with gr.Row():
            training_mode = gr.Radio(
                label="Training Mode",
                choices=["LoRA Training", "DreamBooth Fine-Tuning"],
                value=config.get("training_mode", "LoRA Training"),
                info="LoRA is supported. DreamBooth fine-tuning is not supported for FLUX yet.",
            )
        with gr.Row():
            model_version = gr.Dropdown(
                label="model_version",
                choices=_FLUX2_MODEL_VERSIONS if initial_model_family == "FLUX.2" else _KLEIN_MODEL_VERSIONS,
                value=initial_model_version,
                interactive=(initial_model_family == "FLUX Klein"),
                info="Model version (changes based on Model Family selection above)",
            )
        with gr.Row():
            dit = gr.Textbox(
                label="dit",
                value=config.get("dit", ""),
                placeholder="Example: FLUX_2_Dev_BF16.safetensors",
                interactive=True,
                info="Required DiT checkpoint. Examples: FLUX_2_Dev_BF16.safetensors or FLUX2-Klein-Base-9B.safetensors.",
            )
            dit_btn = gr.Button("📁", size="lg", visible=not headless)
            dit_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"),
                outputs=[dit],
            )
        with gr.Row():
            vae = gr.Textbox(
                label="vae",
                value=config.get("vae", ""),
                placeholder="Example: FLUX_2_Klein_Train_VAE.safetensors",
                interactive=True,
                info="Required VAE/AE checkpoint (e.g., FLUX_2_Klein_Train_VAE.safetensors).",
            )
            vae_btn = gr.Button("📁", size="lg", visible=not headless)
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
                info="Optional VAE dtype override for caching/training (bfloat16 saves VRAM).",
            )
        with gr.Row():
            text_encoder = gr.Textbox(
                label="text_encoder",
                value=config.get("text_encoder", ""),
                placeholder="Example: Mistral3_FLUX2_BF16.safetensors",
                interactive=True,
                info="Required text encoder. Examples: Mistral3_FLUX2_BF16.safetensors (dev) or qwen_3_8b.safetensors (Klein).",
            )
            te_btn = gr.Button("📁", size="lg", visible=not headless)
            te_btn.click(
                fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Model files"),
                outputs=[text_encoder],
            )
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
            fp8_text_encoder = gr.Checkbox(
                label="fp8_text_encoder",
                value=(bool(config.get("fp8_text_encoder", False)) if initial_model_family == "FLUX Klein" else False),
                interactive=(initial_model_family == "FLUX Klein"),
                info=("FP8 for text encoder (supported in Klein)" if initial_model_family == "FLUX Klein" else "Not supported for FLUX.2 dev. Available for FLUX Klein."),
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
                info="Offload N transformer blocks to CPU for VRAM savings. Max recommended: FLUX.2 dev = 29; FLUX Klein 9B = 16; FLUX Klein 4B = 13.",
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
                value=config.get("timestep_sampling", "flux2_shift"),
                interactive=True,
                info="Timestep sampling method for flow matching (flux2_shift recommended for FLUX.2).",
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
                value=config.get("discrete_flow_shift", 1.0),
                step=0.1,
                interactive=True,
                info="Shift factor used by shift/flux_shift/flux2_shift sampling.",
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
                info="Path to a prompt file (one prompt per line). Missing defaults are auto-added unless disabled below.",
            )
            sample_prompts_btn = gr.Button("📁", size="lg", visible=not headless)
            sample_prompts_btn.click(fn=lambda: get_file_path(file_path="", default_extension=".txt", extension_name="Text files"), outputs=[sample_prompts])
        with gr.Row():
            disable_prompt_enhancement = gr.Checkbox(
                label="Disable Automatic Prompt Enhancement",
                value=bool(config.get("disable_prompt_enhancement", False)),
                info="Use prompt file as-is (do not auto-add default --w/--h/--s/--g/--d/--n values).",
            )
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
                info="Sample width in pixels (multiples of 16 recommended).",
            )
            sample_height = gr.Number(
                label="sample_height",
                value=config.get("sample_height", 1024),
                step=8,
                interactive=True,
                info="Sample height in pixels (multiples of 16 recommended).",
            )
        with gr.Row():
            sample_steps = gr.Number(
                label="sample_steps",
                value=config.get("sample_steps", 50),
                step=1,
                interactive=True,
                info="Sampling steps (Klein distilled uses ~4; base uses ~50).",
            )
            sample_guidance_scale = gr.Number(
                label="sample_guidance_scale",
                value=config.get("sample_guidance_scale", 4.0),
                step=0.1,
                interactive=True,
                info="Guidance scale for sampling (distilled ~1.0, base ~4.0).",
            )
            sample_seed = gr.Number(
                label="sample_seed",
                value=config.get("sample_seed", 42),
                step=1,
                interactive=True,
                info="Seed for sample generation.",
            )
        with gr.Row():
            sample_negative_prompt = gr.Textbox(
                label="sample_negative_prompt",
                value=config.get("sample_negative_prompt", ""),
                interactive=True,
                info="Optional negative prompt for CFG sampling.",
            )

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
            caching_teo_fp8_text_encoder = gr.Checkbox(
                label="caching_teo_fp8_text_encoder",
                value=(
                    bool(config.get("caching_teo_fp8_text_encoder", False)) if initial_model_family == "FLUX Klein" else False
                ),
                interactive=(initial_model_family == "FLUX Klein"),
                info=(
                    "FP8 for text encoder caching (supported in Klein)"
                    if initial_model_family == "FLUX Klein"
                    else "Not supported for FLUX.2 dev. Available for FLUX Klein."
                ),
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

    # Dynamic UI updates based on model family selection.
    # NOTE: This runs on programmatic updates too (e.g. when loading a preset),
    # so it must preserve already-loaded values instead of always forcing defaults.
    def update_model_settings(
        family: str,
        current_model_version: str,
        current_fp8_text_encoder: bool,
        current_caching_teo_fp8_text_encoder: bool,
        current_sample_steps,
        current_sample_guidance_scale,
    ):
        family = (family or "").strip()
        current_model_version = (current_model_version or "").strip()

        def _as_int(value, default: int):
            try:
                if value is None or value == "":
                    return default
                return int(value)
            except (TypeError, ValueError):
                return default

        def _as_float(value, default: float):
            try:
                if value is None or value == "":
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        if family == "FLUX.2":
            return (
                gr.update(choices=_FLUX2_MODEL_VERSIONS, value="dev", interactive=False, info="FLUX.2 dev (Mistral 3)"),
                gr.update(value=False, interactive=False, info="Not supported for FLUX.2 dev."),
                gr.update(value=False, interactive=False, info="Not supported for FLUX.2 dev."),
                gr.update(value=_as_int(current_sample_steps, 50)),
                gr.update(value=_as_float(current_sample_guidance_scale, 4.0)),
            )

        # FLUX Klein
        allowed_klein = {v for _, v in _KLEIN_MODEL_VERSIONS}
        if current_model_version not in allowed_klein:
            current_model_version = "klein-base-9b"

        # Recommended sampling defaults for Klein:
        # - distilled klein-4b / klein-9b: 4 steps @ guidance 1.0
        # - base klein models: 50 steps @ guidance 4.0
        is_distilled = current_model_version in {"klein-4b", "klein-9b"}
        default_steps = 4 if is_distilled else 50
        default_guidance = 1.0 if is_distilled else 4.0

        return (
            gr.update(choices=_KLEIN_MODEL_VERSIONS, value=current_model_version, interactive=True, info="Select FLUX Klein model version"),
            gr.update(value=bool(current_fp8_text_encoder), interactive=True, info="FP8 for text encoder (supported in Klein)"),
            gr.update(value=bool(current_caching_teo_fp8_text_encoder), interactive=True, info="FP8 for text encoder caching (supported in Klein)"),
            gr.update(value=_as_int(current_sample_steps, default_steps)),
            gr.update(value=_as_float(current_sample_guidance_scale, default_guidance)),
        )

    model_family.change(
        fn=update_model_settings,
        inputs=[model_family, model_version, fp8_text_encoder, caching_teo_fp8_text_encoder, sample_steps, sample_guidance_scale],
        outputs=[model_version, fp8_text_encoder, caching_teo_fp8_text_encoder, sample_steps, sample_guidance_scale],
    )

    # Klein-specific: Update sampling defaults based on model version
    def update_klein_sampling_defaults(version: str):
        v = (version or "").strip().lower()
        if v in {"klein-4b", "klein-9b"}:  # Distilled models
            return gr.Number(value=4), gr.Number(value=1.0)
        return gr.Number(value=50), gr.Number(value=4.0)  # Base models

    model_version.change(
        fn=update_klein_sampling_defaults,
        inputs=[model_version],
        outputs=[sample_steps, sample_guidance_scale],
    )

    settings_list = [
        # model family
        model_family,
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
        training_mode,
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
        disable_prompt_enhancement,
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
    assert len(settings_list) == len(FLUX_PARAM_KEYS), f"settings_list ({len(settings_list)}) != FLUX_PARAM_KEYS ({len(FLUX_PARAM_KEYS)})"

    # Add handler for search functionality
    def search_settings(query):
        if not query or len(query.strip()) < 1:
            return gr.Row(visible=False), ""

        query_lower = query.lower().strip()
        results = []

        # Comprehensive parameter map
        parameter_map = {
            # Model Selection
            "model_family": ("FLUX Model Settings", "Model Family (FLUX.2 or Klein)"),
            "training_mode": ("FLUX Model Settings", "Training Mode (LoRA/DreamBooth)"),
            "model_version": ("FLUX Model Settings", "Model Version"),
            "flux": ("FLUX Model Settings", "FLUX models"),
            "klein": ("FLUX Model Settings", "FLUX Klein models"),
            "dev": ("FLUX Model Settings", "FLUX.2 dev model"),

            # Model Paths
            "dit": ("FLUX Model Settings", "DiT Checkpoint Path"),
            "vae": ("FLUX Model Settings", "VAE/AE Checkpoint Path"),
            "text_encoder": ("FLUX Model Settings", "Text Encoder Path"),
            "vae_dtype": ("FLUX Model Settings", "VAE Data Type"),

            # FP8 and Memory
            "fp8_base": ("FLUX Model Settings", "FP8 for Base Model (DiT)"),
            "fp8_scaled": ("FLUX Model Settings", "Scaled FP8 for Base Model"),
            "fp8_text_encoder": ("FLUX Model Settings", "FP8 for Text Encoder (Klein only)"),
            "blocks_to_swap": ("FLUX Model Settings", "Blocks to Swap to CPU"),
            "blocks": ("FLUX Model Settings", "Blocks to Swap"),
            "swap": ("FLUX Model Settings", "Blocks to Swap"),
            "cpu": ("FLUX Model Settings", "CPU Offloading"),
            "disable_numpy_memmap": ("FLUX Model Settings", "Disable NumPy Memmap"),
            "use_pinned_memory_for_block_swap": ("FLUX Model Settings", "Use Pinned Memory for Block Swap"),
            "img_in_txt_in_offloading": ("FLUX Model Settings", "Image-in-Text Input Offloading"),

            # Torch Compile
            "compile": ("Torch Compile Settings", "Enable torch.compile"),
            "compile_backend": ("Torch Compile Settings", "Compile Backend"),
            "compile_mode": ("Torch Compile Settings", "Compile Mode"),
            "compile_dynamic": ("Torch Compile Settings", "Dynamic Shapes"),
            "compile_fullgraph": ("Torch Compile Settings", "Fullgraph Mode"),
            "compile_cache_size_limit": ("Torch Compile Settings", "Cache Size Limit"),

            # Dataset Settings
            "dataset": ("FLUX Training Dataset", "Dataset Configuration"),
            "dataset_config_mode": ("FLUX Training Dataset", "Dataset Configuration Method"),
            "dataset_config": ("FLUX Training Dataset", "Dataset Config File"),
            "parent_folder": ("FLUX Training Dataset", "Parent Folder Path"),
            "resolution": ("FLUX Training Dataset", "Resolution"),
            "caption": ("FLUX Training Dataset", "Caption Settings"),
            "bucket": ("FLUX Training Dataset", "Bucketing"),
            "control": ("FLUX Training Dataset", "Control Images"),
            "control_directory": ("FLUX Training Dataset", "Control Directory Name"),
            "no_resize_control": ("FLUX Training Dataset", "No Resize Control Images"),
            "batch_size": ("FLUX Training Dataset", "Batch Size"),
            "cache_directory": ("FLUX Training Dataset", "Cache Directory"),

            # Flow Matching
            "timestep": ("Flow Matching and Timestep Settings", "Timestep Sampling"),
            "weighting": ("Flow Matching and Timestep Settings", "Weighting Scheme"),
            "flow_shift": ("Flow Matching and Timestep Settings", "Flow Shift"),
            "discrete_flow_shift": ("Flow Matching and Timestep Settings", "Discrete Flow Shift"),
            "sigmoid": ("Flow Matching and Timestep Settings", "Sigmoid Scale"),
            "flux2_shift": ("Flow Matching and Timestep Settings", "FLUX2 Shift Sampling"),

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
            "dim_from_weights": ("LoRA Settings", "Auto-Determine Rank from Weights"),
            "scale_weight_norms": ("LoRA Settings", "Scale Weight Norms"),
            "base_weights": ("LoRA Settings", "Base Weights"),

            # Caching
            "cache": ("Caching Settings", "Caching Configuration"),
            "latent_cache": ("Caching Settings", "Latent Caching"),
            "text_encoder_cache": ("Caching Settings", "Text Encoder Output Caching"),
            "caching_device": ("Caching Settings", "Caching Device"),
            "skip_existing": ("Caching Settings", "Skip Existing Cache"),
            "keep_cache": ("Caching Settings", "Keep Cache Files"),

            # Samples
            "sample": ("Sample Generation Settings", "Sample Generation"),
            "sample_prompts": ("Sample Generation Settings", "Sample Prompts File"),
            "disable_prompt_enhancement": ("Sample Generation Settings", "Disable Automatic Prompt Enhancement"),
            "sample_steps": ("Sample Generation Settings", "Sample Steps"),
            "guidance": ("Sample Generation Settings", "Guidance Scale"),

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
            html += f"<span style='color: #666; font-size: 0.9em;'>Try: learning, optimizer, fp8, vram, epochs, batch, cache, sample, model</span>"
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
            "FLUX Training Dataset": 2,
            "FLUX Model Settings": 3,
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
        flux_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )
    configuration.button_load_config.click(
        flux_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )
    configuration.button_save_config.click(
        flux_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )

    button_print.click(
        flux_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        flux_gui_actions,
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
