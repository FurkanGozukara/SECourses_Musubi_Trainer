import gradio as gr
import os
import subprocess
import time
import toml

from datetime import datetime
from .class_accelerate_launch import AccelerateLaunch
from .class_advanced_training import AdvancedTraining
from .class_command_executor import CommandExecutor
from .class_configuration_file import ConfigurationFile
from .class_gui_config import GUIConfig
from .class_latent_caching import LatentCaching
from .class_network import Network
from .class_optimizer_and_scheduler import OptimizerAndScheduler
from .class_save_load import SaveLoadSettings
from .class_text_encoder_outputs_caching import TextEncoderOutputsCaching
from .class_training import TrainingSettings
from .common_gui import (
    get_file_path,
    get_saveasfile_path,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    SaveConfigFileToRun,
    scriptdir,
    setup_environment,
)
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .custom_logging import setup_logging

log = setup_logging()

executor = None
huggingface = None
train_state_value = time.time()


class QwenImageModel:
    """Qwen Image specific model settings"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.dataset_config = gr.Textbox(
                label="Dataset Config File",
                info="REQUIRED: Path to TOML file containing dataset configuration (images, captions, batch size, resolution, etc.)",
                placeholder='e.g., /path/to/dataset.toml',
                value=str(self.config.get("dataset_config", "")),
            )

        with gr.Row():
            self.dit = gr.Textbox(
                label="DiT (Base Model) Checkpoint Path",
                placeholder="Path to DiT base model checkpoint (qwen_image_bf16.safetensors)",
                value=self.config.get("dit", ""),
            )

            self.dit_dtype = gr.Dropdown(
                label="DiT Data Type",
                info="⚙️ HARDCODED: bfloat16 is required and hardcoded for Qwen Image. Do not change - other dtypes will cause errors",
                choices=["bfloat16"],
                value=self.config.get("dit_dtype", "bfloat16"),
                interactive=False,  # Hardcoded for Qwen Image
            )

        with gr.Row():
            self.vae = gr.Textbox(
                label="VAE Checkpoint Path",
                info="REQUIRED: Path to VAE model (diffusion_pytorch_model.safetensors from Qwen/Qwen-Image). NOT ComfyUI VAE!",
                placeholder="e.g., /path/to/vae/diffusion_pytorch_model.safetensors",
                value=self.config.get("vae", ""),
            )
            # Note: vae_dtype is not supported in Qwen Image, removed from GUI

        # Qwen Image specific text encoder
        with gr.Row():
            self.text_encoder = gr.Textbox(
                label="Text Encoder (Qwen2.5-VL) Path",
                info="REQUIRED: Path to Qwen2.5-VL text encoder model (qwen_2.5_vl_7b.safetensors from Comfy-Org/Qwen-Image_ComfyUI)",
                placeholder="e.g., /path/to/text_encoder/qwen_2.5_vl_7b.safetensors",
                value=self.config.get("text_encoder", ""),
            )

            self.fp8_vl = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                info="✅ ENABLED by default. FP8 quantization for Qwen2.5-VL saves ~8GB VRAM with minimal quality loss. Disable only if you have 24GB+ VRAM",
                value=self.config.get("fp8_vl", True),
            )

        # Qwen Image specific options
        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="Use FP8 for Base Model (DiT)",
                info="FP8 quantization for DiT model saves ~12GB VRAM. May slightly reduce quality. Useful for 16GB GPUs. Combine with fp8_scaled",
                value=self.config.get("fp8_base", False),
            )
            
            self.fp8_scaled = gr.Checkbox(
                label="Use Scaled FP8 for DiT",
                info="Improved FP8 quantization method. REQUIRED when fp8_base is enabled. Better quality than standard FP8",
                value=self.config.get("fp8_scaled", False),
            )
            
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap to CPU",
                info="Swap DiT blocks to CPU to save VRAM. 16=24GB→16GB, 45=42GB→12GB. Requires 64GB+ RAM. Slows training significantly",
                value=self.config.get("blocks_to_swap", 0),
                minimum=0,
                maximum=50,
                step=1,
                interactive=True,
            )

        # Flow matching parameters
        with gr.Row():
            self.timestep_sampling = gr.Dropdown(
                label="Timestep Sampling Method",
                info="'shift' uses fixed discrete_flow_shift value. 'qwen_shift' calculates dynamic shift based on image resolution (ignores discrete_flow_shift parameter)",
                choices=["shift", "qwen_shift", "uniform", "sigmoid", "logsnr", "qinglong_flux", "qinglong_qwen"],
                value=self.config.get("timestep_sampling", "shift"),
                interactive=True,
            )

            self.discrete_flow_shift = gr.Number(
                label="Discrete Flow Shift",
                info="⚠️ Only used with 'shift' method. Qwen Image optimal: 3.0. 'qwen_shift' automatically calculates dynamic shift (0.5-0.9) based on image resolution",
                value=self.config.get("discrete_flow_shift", 3.0),
                step=0.1,
                interactive=True,
            )

            self.weighting_scheme = gr.Dropdown(
                label="Weighting Scheme",
                info="✅ 'none' recommended for Qwen Image. Advanced: 'logit_normal' for different timestep emphasis, 'mode' for SD3-style weighting",
                choices=["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"],
                value=self.config.get("weighting_scheme", "none"),
                interactive=True,
            )

        # Weighting scheme parameters
        with gr.Row():
            self.logit_mean = gr.Number(
                label="Logit Mean",
                info="Mean for 'logit_normal' weighting. 0.0=balanced, negative=favor early timesteps, positive=favor late timesteps",
                value=self.config.get("logit_mean", 0.0),
                minimum=-3.0,
                maximum=3.0,
                step=0.001,
                interactive=True,
            )

            self.logit_std = gr.Number(
                label="Logit Std",
                info="Standard deviation for 'logit_normal'. 1.0=normal distribution, <1.0=concentrated, >1.0=spread out timestep sampling",
                value=self.config.get("logit_std", 1.0),
                minimum=0.1,
                maximum=5.0,
                step=0.001,
                interactive=True,
            )

            self.mode_scale = gr.Number(
                label="Mode Scale",
                info="Scale for 'mode' weighting scheme. Default 1.29 from SD3. Higher values = more emphasis on certain timesteps",
                value=self.config.get("mode_scale", 1.29),
                minimum=0.1,
                maximum=5.0,
                step=0.001,
                interactive=True,
            )

        with gr.Row():
            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
                info="Visualization mode for timestep debugging. 'image' saves visual plots, 'console' prints to terminal. Leave empty for no visualization",
                choices=["image", "console"],
                allow_custom_value=True,
                value=self.config.get("show_timesteps", None),
                interactive=True,
            )


def qwen_image_gui_actions(
    # action type
    action_type,
    # control
    bool_value,
    file_path,
    headless,
    print_only,
    # accelerate_launch
    mixed_precision,
    num_cpu_threads_per_process,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    main_process_port,
    dynamo_backend,
    dynamo_mode,
    dynamo_use_fullgraph,
    dynamo_use_dynamic,
    extra_accelerate_launch_args,
    # advanced_training
    additional_parameters,
    dataset_config,
    sdpa,
    flash_attn,
    sage_attn,
    xformers,
    split_attn,
    max_train_steps,
    max_train_epochs,
    max_data_loader_n_workers,
    persistent_data_loader_workers,
    seed,
    gradient_checkpointing,
    gradient_accumulation_steps,
    logging_dir,
    log_with,
    log_prefix,
    log_tracker_name,
    wandb_run_name,
    log_tracker_config,
    wandb_api_key,
    log_config,
    ddp_timeout,
    ddp_gradient_as_bucket_view,
    ddp_static_graph,
    sample_every_n_steps,
    sample_at_first,
    sample_every_n_epochs,
    sample_prompts,
    # Latent Caching
    caching_latent_device,
    caching_latent_batch_size,
    caching_latent_num_workers,
    caching_latent_skip_existing,
    caching_latent_keep_cache,
    caching_latent_debug_mode,
    caching_latent_console_width,
    caching_latent_console_back,
    caching_latent_console_num_images,
    
    # Text Encoder Outputs Caching
    caching_teo_text_encoder,  # Single text encoder for Qwen Image
    caching_teo_device,
    caching_teo_fp8_vl,
    caching_teo_batch_size,
    caching_teo_num_workers,
    caching_teo_skip_existing,
    caching_teo_keep_cache,
    optimizer_type,
    optimizer_args,
    learning_rate,
    max_grad_norm,
    lr_scheduler,
    lr_warmup_steps,
    lr_decay_steps,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    lr_scheduler_timescale,
    lr_scheduler_min_lr_ratio,
    lr_scheduler_type,
    lr_scheduler_args,
    dit,
    dit_dtype,
    vae,
    text_encoder,  # Qwen Image text encoder
    fp8_vl,       # Qwen Image specific
    fp8_base,
    fp8_scaled,   # Qwen Image specific
    blocks_to_swap,
    timestep_sampling,
    discrete_flow_shift,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    show_timesteps,
    no_metadata,
    network_weights,
    network_module,
    network_dim,
    network_alpha,
    network_dropout,
    network_args,
    training_comment,
    dim_from_weights,
    scale_weight_norms,
    base_weights,
    base_weights_multiplier,
    output_dir,
    output_name,
    resume,
    save_every_n_epochs,
    save_every_n_steps,
    save_last_n_epochs,
    save_last_n_epochs_state,
    save_last_n_steps,
    save_last_n_steps_state,
    save_state,
    save_state_on_train_end,
    huggingface_repo_id,
    huggingface_token,
    huggingface_repo_type,
    huggingface_repo_visibility,
    huggingface_path_in_repo,
    save_state_to_huggingface,
    resume_from_huggingface,
    async_upload,
    metadata_author,
    metadata_description,
    metadata_license,
    metadata_tags,
    metadata_title,
):
    parameters = [(k, v) for k, v in locals().items() if k not in ["action_type", "bool_value", "headless", "print_only"]]
    
    if action_type == "save_configuration":
        log.info("Save configuration...")
        return save_qwen_image_configuration(
            save_as_bool=bool_value,
            file_path=file_path,
            parameters=parameters,
        )
        
    if action_type == "open_configuration":
        log.info("Open configuration...")
        return open_qwen_image_configuration(
            ask_for_file=bool_value,
            file_path=file_path,
            parameters=parameters,
        )
        
    if action_type == "train_model":
        log.info("Train Qwen Image model...")
        return train_qwen_image_model(
            headless=headless,
            print_only=print_only,
            parameters=parameters,
        )


def save_qwen_image_configuration(save_as_bool, file_path, parameters):
    original_file_path = file_path

    if save_as_bool:
        log.info("Save as...")
        file_path = get_saveasfile_path(
            file_path, defaultextension=".toml", extension_name="TOML files (*.toml)"
        )
    else:
        log.info("Save...")
        if file_path == None or file_path == "":
            file_path = get_saveasfile_path(
                file_path,
                defaultextension=".toml",
                extension_name="TOML files (*.toml)",
            )

    log.debug(file_path)

    if file_path == None or file_path == "":
        return original_file_path

    destination_directory = os.path.dirname(file_path)

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    SaveConfigFile(
        parameters=parameters,
        file_path=file_path,
        exclusion=[
            "file_path",
            "save_as",
            "save_as_bool",
        ],
    )

    return file_path


def open_qwen_image_configuration(ask_for_file, file_path, parameters):
    original_file_path = file_path

    if ask_for_file:
        file_path = get_file_path(
            file_path, default_extension=".toml", extension_name="TOML files (*.toml)"
        )

    if not file_path == "" and not file_path == None:
        if not os.path.isfile(file_path):
            log.error(f"Config file {file_path} does not exist.")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            my_data = toml.load(f)
            log.info("Loading config...")
    else:
        file_path = original_file_path
        my_data = {}

    values = [file_path]
    for key, value in parameters:
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            toml_value = my_data.get(key)
            values.append(toml_value if toml_value is not None else value)

    return tuple(values)


def train_qwen_image_model(headless, print_only, parameters):
    import sys
    
    # Use Python directly instead of uv for better compatibility
    python_cmd = sys.executable
    run_cmd = [python_cmd, "-m", "accelerate", "launch"]

    param_dict = dict(parameters)
    
    # Validate required parameters before starting
    if not param_dict.get("dataset_config"):
        raise ValueError("❌ Dataset config file is required for training. Please specify a path to your dataset configuration TOML file.")
    if not param_dict.get("vae"):
        raise ValueError("❌ VAE checkpoint path is required for training. Please download and specify the VAE model path (diffusion_pytorch_model.safetensors from Qwen/Qwen-Image).")
    if not param_dict.get("dit"):
        raise ValueError("❌ DiT checkpoint path is required for training. Please download and specify the DiT model path (qwen_image_bf16.safetensors from Comfy-Org/Qwen-Image_ComfyUI).")
    if not param_dict.get("output_dir"):
        raise ValueError("❌ Output directory is required. Please specify where to save your trained LoRA model.")
    if not param_dict.get("output_name"):
        raise ValueError("❌ Output name is required. Please specify a name for your trained LoRA model.")
    
    # Cache latents using Qwen Image specific script
    run_cache_latent_cmd = [python_cmd, "./musubi-tuner/qwen_image_cache_latents.py",
                            "--dataset_config", str(param_dict.get("dataset_config")),
                            "--vae", str(param_dict.get("vae"))
    ]
    
    # Note: vae_dtype is not supported in Qwen Image
        
    if param_dict.get("caching_latent_device"):
        run_cache_latent_cmd.append("--device")
        run_cache_latent_cmd.append(str(param_dict.get("caching_latent_device")))
    
    if param_dict.get("caching_latent_batch_size"):
        run_cache_latent_cmd.append("--batch_size")
        run_cache_latent_cmd.append(str(param_dict.get("caching_latent_batch_size")))
    
    if param_dict.get("caching_latent_num_workers"):
        run_cache_latent_cmd.append("--num_workers")
        run_cache_latent_cmd.append(str(param_dict.get("caching_latent_num_workers")))
        
    if param_dict.get("caching_latent_skip_existing"):
        run_cache_latent_cmd.append("--skip_existing")
        
    if param_dict.get("caching_latent_keep_cache"):
        run_cache_latent_cmd.append("--keep_cache")

    log.info(f"Executing command: {run_cache_latent_cmd}")
    log.info("Caching latents...")
    try:
        result = subprocess.run(run_cache_latent_cmd, env=setup_environment(), 
                               capture_output=True, text=True, check=True)
        log.debug("Latent caching completed.")
    except subprocess.CalledProcessError as e:
        log.error(f"Latent caching failed with return code {e.returncode}")
        log.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Latent caching failed: {e.stderr}")
    except FileNotFoundError as e:
        log.error(f"Command not found: {e}")
        log.error("Please ensure Python is installed and accessible in your PATH")
        raise RuntimeError(f"Python executable not found: {python_cmd}")
    
    # Cache text encoder outputs using Qwen Image specific script
    run_cache_teo_cmd = [python_cmd, "./musubi-tuner/qwen_image_cache_text_encoder_outputs.py",
                            "--dataset_config", str(param_dict.get("dataset_config"))
    ]
    
    if param_dict.get("text_encoder"):
        run_cache_teo_cmd.append("--text_encoder")
        run_cache_teo_cmd.append(str(param_dict.get("text_encoder")))   
                            
    if param_dict.get("caching_teo_fp8_vl"):
        run_cache_teo_cmd.append("--fp8_vl")
        
    if param_dict.get("caching_teo_device"):
        run_cache_teo_cmd.append("--device")
        run_cache_teo_cmd.append(str(param_dict.get("caching_teo_device")))
    
    if param_dict.get("caching_teo_batch_size"):
        run_cache_teo_cmd.append("--batch_size")
        run_cache_teo_cmd.append(str(param_dict.get("caching_teo_batch_size")))
        
    if param_dict.get("caching_teo_skip_existing"):
        run_cache_teo_cmd.append("--skip_existing")
        
    if param_dict.get("caching_teo_keep_cache"):
        run_cache_teo_cmd.append("--keep_cache")
        
    if param_dict.get("caching_teo_num_workers"):
        run_cache_teo_cmd.append("--num_workers")
        run_cache_teo_cmd.append(str(param_dict.get("caching_teo_num_workers")))

    log.info(f"Executing command: {run_cache_teo_cmd}")
    log.info("Caching text encoder outputs...")
    try:
        result = subprocess.run(run_cache_teo_cmd, env=setup_environment(),
                               capture_output=True, text=True, check=True)
        log.debug("Text encoder output caching completed.")
    except subprocess.CalledProcessError as e:
        log.error(f"Text encoder caching failed with return code {e.returncode}")
        log.error(f"Error output: {e.stderr}")
        raise RuntimeError(f"Text encoder caching failed: {e.stderr}")
    except FileNotFoundError as e:
        log.error(f"Command not found: {e}")
        raise RuntimeError(f"Python executable not found: {python_cmd}")

    # Setup accelerate launch command
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

    # Use Qwen Image specific training script
    run_cmd.append(f"{scriptdir}/musubi-tuner/qwen_image_train_network.py")

    if print_only:
        print_command_and_toml(run_cmd, "")
    else:
        # Save config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(param_dict.get('output_dir'), f"{param_dict.get('output_name')}_{formatted_datetime}.toml")

        log.info(f"Saving training config to {file_path}...")

        pattern_exclusion = []
        for key, _ in parameters:
            if key.startswith('caching_latent_') or key.startswith('caching_teo_'):
                pattern_exclusion.append(key)

        SaveConfigFileToRun(
            parameters=parameters,
            file_path=file_path,
            exclusion=[
                "file_path",
                "save_as",
                "save_as_bool",
                "headless",
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
            ] + pattern_exclusion,
        )
        
        run_cmd.append("--config_file")
        run_cmd.append(f"{file_path}")

        run_cmd_params = {
            "additional_parameters": param_dict.get("additional_parameters"),
        }

        run_cmd = run_cmd_advanced_training(run_cmd=run_cmd, **run_cmd_params)

        env = setup_environment()

        executor.execute_command(run_cmd=run_cmd, env=env)

        train_state_value = time.time()

        return (
            gr.Button(visible=False or headless),
            gr.Button(visible=True),
            gr.Textbox(value=train_state_value),
        )


class QwenImageTrainingSettings:
    """Qwen Image specific training settings with optimal defaults"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.sdpa = gr.Checkbox(
                label="Use SDPA for CrossAttention",
                info="✅ RECOMMENDED for Qwen Image. PyTorch's Scaled Dot Product Attention - fastest and most memory efficient",
                value=self.config.get("sdpa", True),
            )

            self.flash_attn = gr.Checkbox(
                label="Use FlashAttention for CrossAttention",
                info="Memory-efficient attention implementation. Requires FlashAttention library. Enable split_attn if using this",
                value=self.config.get("flash_attn", False),
            )

            self.sage_attn = gr.Checkbox(
                label="Use SageAttention for CrossAttention",
                info="Alternative attention implementation. Requires SageAttention library. Enable split_attn if using this",
                value=self.config.get("sage_attn", False),
            )

            self.xformers = gr.Checkbox(
                label="Use xformers for CrossAttention",
                info="Memory-efficient attention from xformers library. Enable split_attn if using this",
                value=self.config.get("xformers", False),
            )

            self.split_attn = gr.Checkbox(
                label="Split Attention",
                info="⚠️ REQUIRED if using FlashAttention/SageAttention/xformers. Splits attention computation to reduce memory usage",
                value=self.config.get("split_attn", False),
            )

        with gr.Row():
            self.max_train_steps = gr.Number(
                label="Max Training Steps",
                info="Total training steps. 1600 steps ≈ 1-2 hours on RTX 4090. Ignored if Max Training Epochs is set",
                value=self.config.get("max_train_steps", 1600),
                minimum=100,
                step=100,
                interactive=True,
            )

            self.max_train_epochs = gr.Number(
                label="Max Training Epochs",
                info="✅ RECOMMENDED: 16 epochs for Qwen Image. Overrides max_train_steps. 1 epoch = full pass through dataset",
                value=self.config.get("max_train_epochs", 16),
                minimum=1,
                maximum=100,
                step=1,
                interactive=True,
            )

            self.max_data_loader_n_workers = gr.Number(
                label="Max DataLoader Workers",
                info="✅ 2 recommended for Qwen Image stability. Higher values = faster data loading but more RAM usage and potential instability",
                value=self.config.get("max_data_loader_n_workers", 2),
                minimum=0,
                maximum=16,
                step=1,
                interactive=True,
            )

            self.persistent_data_loader_workers = gr.Checkbox(
                label="Persistent DataLoader Workers",
                info="✅ ENABLED: Keeps data loading processes alive between epochs. Faster epoch transitions but uses more RAM",
                value=self.config.get("persistent_data_loader_workers", True),
            )

        with gr.Row():
            self.seed = gr.Number(
                label="Random Seed for Training",
                info="42 = reproducible results (same output every time). 0 or empty = random seed each run. Useful for comparing training runs",
                value=self.config.get("seed", 42),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.gradient_checkpointing = gr.Checkbox(
                label="Enable Gradient Checkpointing",
                info="✅ ENABLED: Trades computation for memory. Essential for Qwen Image training. Saves ~50% VRAM but increases training time ~20%",
                value=self.config.get("gradient_checkpointing", True),
            )

            self.gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                info="Simulate larger batch size. 1 = update every step, 4 = accumulate 4 steps then update. Useful for small VRAM",
                value=self.config.get("gradient_accumulation_steps", 1),
                minimum=1,
                maximum=32,
                step=1,
                interactive=True,
            )

        # Logging settings
        with gr.Row():
            self.logging_dir = gr.Textbox(
                label="Logging Directory",
                info="Directory for training logs and TensorBoard data. Leave empty to disable logging. Example: ./logs",
                placeholder="e.g., ./logs or /path/to/logs",
                value=self.config.get("logging_dir", ""),
            )

            self.log_with = gr.Dropdown(
                label="Logging Tool",
                info="TensorBoard = local logs, WandB = cloud tracking, 'all' = both. Requires logging_dir for TensorBoard",
                choices=["tensorboard", "wandb", "all"],
                allow_custom_value=True,
                value=self.config.get("log_with", ""),
                interactive=True,
            )

        with gr.Row():
            self.log_prefix = gr.Textbox(
                label="Log Prefix",
                info="Prefix added to log directory names with timestamp. Example: 'qwen-lora' creates 'qwen-lora-20241201120000'",
                placeholder="e.g., qwen-lora or my-experiment",
                value=self.config.get("log_prefix", ""),
            )

            self.log_tracker_name = gr.Textbox(
                label="Log Tracker Name",
                info="Custom name for the tracker instance. Used in TensorBoard/WandB interface. Defaults to script name",
                placeholder="e.g., qwen-image-training or my-custom-name",
                value=self.config.get("log_tracker_name", ""),
            )

        # Sample generation settings
        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                info="Generate test images every N training steps. 100-500 recommended. Leave empty to disable. Requires sample_prompts file",
                value=self.config.get("sample_every_n_steps", None),
                minimum=1,
                step=1,
                interactive=True,
            )

            self.sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs",
                info="Generate test images every N epochs. 1-4 recommended. Leave empty to disable. Overrides sample_every_n_steps",
                value=self.config.get("sample_every_n_epochs", None),
                minimum=1,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.sample_at_first = gr.Checkbox(
                label="Sample at First",
                info="Generate test images before training starts. Useful to verify prompts and base model quality",
                value=self.config.get("sample_at_first", False),
            )

            self.sample_prompts = gr.Textbox(
                label="Sample Prompts File",
                info="Path to text file with prompts (one per line). Required for sample generation. Example: 'A cat\nA dog\nA house'",
                placeholder="e.g., /path/to/prompts.txt",
                value=self.config.get("sample_prompts", ""),
            )

        # Additional settings
        with gr.Row():
            self.wandb_run_name = gr.Textbox(
                label="WandB Run Name",
                info="Custom name for WandB experiment. Auto-generated if empty. Only used when log_with includes 'wandb'",
                placeholder="e.g., qwen-lora-experiment-v1",
                value=self.config.get("wandb_run_name", ""),
            )

            self.log_tracker_config = gr.Textbox(
                label="Log Tracker Config",
                info="Path to JSON config file for logging parameters. Advanced users only. Leave empty for defaults",
                placeholder="e.g., /path/to/tracker_config.json",
                value=self.config.get("log_tracker_config", ""),
            )

        with gr.Row():
            self.wandb_api_key = gr.Textbox(
                label="WandB API Key",
                info="WandB API key for authentication. Get from wandb.ai/authorize. Can also set WANDB_API_KEY env variable",
                placeholder="Enter your WandB API key (optional if env var set)",
                type="password",
                value=self.config.get("wandb_api_key", ""),
            )

            self.log_config = gr.Checkbox(
                label="Log Training Configuration",
                info="Include all training parameters in logs. Useful for experiment tracking and reproducibility",
                value=self.config.get("log_config", False),
            )

        # DDP settings
        with gr.Row():
            self.ddp_timeout = gr.Number(
                label="DDP Timeout (minutes)",
                info="Distributed training timeout in minutes. Increase if training crashes on multi-GPU setups. Leave empty for default (30min)",
                value=self.config.get("ddp_timeout", None),
                minimum=1,
                maximum=1440,
                step=1,
                interactive=True,
            )

            self.ddp_gradient_as_bucket_view = gr.Checkbox(
                label="DDP Gradient as Bucket View",
                info="Optimization for distributed training. May improve performance but can cause instability in some cases",
                value=self.config.get("ddp_gradient_as_bucket_view", False),
            )

            self.ddp_static_graph = gr.Checkbox(
                label="DDP Static Graph",
                info="Enables static graph optimization for distributed training. May improve performance if model architecture doesn't change",
                value=self.config.get("ddp_static_graph", False),
            )

        with gr.Row():
            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
                info="Debug timestep distribution. 'image' saves visual plots, 'console' prints to terminal. Leave empty for no visualization",
                choices=["image", "console"],
                allow_custom_value=True,
                value=self.config.get("show_timesteps", None),
                interactive=True,
            )


class QwenImageOptimizerSettings:
    """Qwen Image specific optimizer settings with optimal defaults"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.optimizer_type = gr.Dropdown(
                label="Optimizer Type",
                info="adamw8bit recommended for Qwen Image training (memory efficient)",
                choices=["adamw8bit", "AdamW", "AdaFactor", "bitsandbytes.optim.AdEMAMix8bit", "bitsandbytes.optim.PagedAdEMAMix8bit"],
                allow_custom_value=True,
                value=self.config.get("optimizer_type", "adamw8bit"),
            )

            self.learning_rate = gr.Number(
                label="Learning Rate",
                info="✅ 1e-4 (0.0001) recommended for Qwen Image. Too high = instability, too low = slow learning. Typical range: 1e-6 to 1e-3",
                value=self.config.get("learning_rate", 1e-4),
                minimum=1e-7,
                maximum=1e-2,
                step=1e-6,
                interactive=True,
            )

        with gr.Row():
            self.optimizer_args = gr.Textbox(
                label="Optimizer Arguments",
                info="Extra optimizer parameters as key=value pairs. Common: weight_decay=0.01 (regularization), betas=0.9,0.999 (momentum)",
                placeholder='e.g. "weight_decay=0.01 betas=0.9,0.999"',
                value=self.config.get("optimizer_args", ""),
            )

            self.max_grad_norm = gr.Number(
                label="Max Gradient Norm",
                info="Gradient clipping to prevent exploding gradients. 1.0 = good default, 0 = disabled. Higher = allow larger gradients",
                value=self.config.get("max_grad_norm", 1.0),
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                interactive=True,
            )

        # Learning rate scheduler settings
        with gr.Row():
            self.lr_scheduler = gr.Dropdown(
                label="Learning Rate Scheduler",
                info="✅ 'constant' recommended for most cases. 'cosine' = gradual decrease, 'linear' = linear decrease, 'constant_with_warmup' = warm start",
                choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "adafactor"],
                value=self.config.get("lr_scheduler", "constant"),
                interactive=True,
            )

            self.lr_warmup_steps = gr.Number(
                label="LR Warmup Steps",
                info="Gradually increase LR for stability. Integer = steps, float <1 = ratio of total steps. 0 = no warmup. Try 100-500 steps",
                value=self.config.get("lr_warmup_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_decay_steps = gr.Number(
                label="LR Decay Steps",
                info="Number of steps to decay learning rate (0 for no decay, or ratio <1 for percentage of total steps)",
                value=self.config.get("lr_decay_steps", 0),
                step=1,
                interactive=True,
            )

            self.lr_scheduler_num_cycles = gr.Number(
                label="LR Scheduler Cycles",
                info="Number of restart cycles for 'cosine_with_restarts' scheduler. More cycles = more LR restarts during training",
                value=self.config.get("lr_scheduler_num_cycles", 1),
                minimum=1,
                maximum=10,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_power = gr.Number(
                label="LR Scheduler Power",
                info="Polynomial decay power for 'polynomial' scheduler. 1.0 = linear, >1.0 = slower initial decay, <1.0 = faster initial decay",
                value=self.config.get("lr_scheduler_power", 1.0),
                minimum=0.1,
                maximum=5.0,
                step=0.1,
                interactive=True,
            )

            self.lr_scheduler_timescale = gr.Number(
                label="LR Scheduler Timescale",
                info="Inverse sqrt scheduler timescale. Defaults to warmup steps. Advanced parameter, leave empty for auto",
                value=self.config.get("lr_scheduler_timescale", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_min_lr_ratio = gr.Number(
                label="LR Min Ratio",
                info="Minimum learning rate as ratio of initial LR. 0.1 = LR won't go below 10% of initial. 0 = can reach zero",
                value=self.config.get("lr_scheduler_min_lr_ratio", 0.0),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                interactive=True,
            )

            self.lr_scheduler_type = gr.Textbox(
                label="Custom Scheduler Module",
                info="Full path to custom scheduler class (e.g., 'torch.optim.lr_scheduler.CosineAnnealingLR'). Leave empty to use built-in schedulers",
                value=self.config.get("lr_scheduler_type", ""),
            )

        with gr.Row():
            self.lr_scheduler_args = gr.Textbox(
                label="Scheduler Arguments",
                info="Extra scheduler parameters as key=value pairs. Example: T_max=100 for CosineAnnealing, eta_min=1e-7 for minimum LR",
                placeholder='e.g. "T_max=100 eta_min=1e-7"',
                value=self.config.get("lr_scheduler_args", ""),
            )


class QwenImageNetworkSettings:
    """Qwen Image specific network settings with optimal defaults"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.no_metadata = gr.Checkbox(
                label="No Metadata",
                info="Exclude training metadata from saved LoRA file. Reduces file size slightly but loses training info for future reference",
                value=self.config.get("no_metadata", False),
            )

            self.network_weights = gr.Textbox(
                label="Pretrained Network Weights",
                info="Path to existing LoRA weights to continue training from. Leave empty to start from scratch",
                placeholder="e.g., /path/to/existing_lora.safetensors",
                value=self.config.get("network_weights", ""),
            )

        with gr.Row():
            self.network_module = gr.Textbox(
                label="Network Module",
                info="⚙️ AUTO-SET: LoRA implementation for Qwen Image. 'networks.lora_qwen_image' is automatically selected. Do not change",
                placeholder="networks.lora_qwen_image",
                value=self.config.get("network_module", "networks.lora_qwen_image"),
                interactive=False,  # Will be auto-selected
            )

            self.network_dim = gr.Number(
                label="Network Dimension (Rank)",
                info="✅ LoRA rank/dimension. 32 recommended for Qwen Image. Higher = more capacity but larger files. Range: 8-128",
                value=self.config.get("network_dim", 32),
                minimum=1,
                maximum=512,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.network_alpha = gr.Number(
                label="Network Alpha",
                info="✅ LoRA scaling factor. 1.0 recommended for Qwen Image. Higher = stronger LoRA effect. Formula: alpha/rank = final scaling",
                value=self.config.get("network_alpha", 1.0),
                minimum=0.1,
                maximum=512.0,
                step=0.1,
                interactive=True,
            )

            self.network_dropout = gr.Number(
                label="Network Dropout",
                info="Dropout rate for regularization. 0.0 = no dropout, 0.1 = 10% dropout. Helps prevent overfitting",
                value=self.config.get("network_dropout", 0.0),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                interactive=True,
            )

        with gr.Row():
            self.network_args = gr.Textbox(
                label="Network Arguments",
                info="Advanced LoRA parameters as key=value pairs. Common: conv_dim=4 (train convolution layers), conv_alpha=1",
                placeholder='e.g. "conv_dim=4 conv_alpha=1"',
                value=self.config.get("network_args", ""),
            )

            self.training_comment = gr.Textbox(
                label="Training Comment",
                info="Descriptive comment saved in LoRA metadata. Useful for tracking what this model was trained for",
                placeholder='e.g. "Anime style character training on 500 images"',
                value=self.config.get("training_comment", ""),
            )

        with gr.Row():
            self.dim_from_weights = gr.Checkbox(
                label="Auto-Determine Rank from Weights",
                info="Automatically detect network_dim from pretrained LoRA weights. Only works when network_weights is specified",
                value=self.config.get("dim_from_weights", False),
            )

            self.scale_weight_norms = gr.Number(
                label="Scale Weight Norms",
                info="Scale weights to prevent exploding gradients. 0.0 = disabled, 1.0 = good starting point",
                value=self.config.get("scale_weight_norms", 0.0),
                minimum=0.0,
                step=0.1,
                interactive=True,
            )

        with gr.Row():
            self.base_weights = gr.Textbox(
                label="LoRA Base Weights",
                info="Path to pre-existing LoRA weights to merge into model before training. Useful for fine-tuning existing LoRAs",
                placeholder="Path to LoRA .safetensors file (optional)",
                value=self.config.get("base_weights", ""),
            )

            self.base_weights_multiplier = gr.Number(
                label="Base Weights Multiplier",
                info="Strength multiplier for base weights (1.0 = full strength, 0.5 = half strength). Only used if base_weights is specified",
                value=self.config.get("base_weights_multiplier", 1.0),
                minimum=0.0,
                maximum=2.0,
                step=0.1,
                interactive=True,
            )


class QwenImageSaveLoadSettings:
    """Qwen Image specific save/load settings with optimal defaults"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.output_dir = gr.Textbox(
                label="Output Directory",
                info="REQUIRED: Directory where trained LoRA model will be saved. Must exist or be creatable",
                placeholder="e.g., ./models/trained or /path/to/output",
                value=self.config.get("output_dir", ""),
            )

            self.output_name = gr.Textbox(
                label="Output Name",
                info="REQUIRED: Base filename for saved LoRA (without extension). Example: 'my-qwen-lora' creates 'my-qwen-lora.safetensors'",
                placeholder="e.g., my-qwen-lora or character-style-v1",
                value=self.config.get("output_name", ""),
            )

        with gr.Row():
            self.resume = gr.Textbox(
                label="Resume Training State",
                info="Path to .safetensors state file to resume interrupted training. Includes optimizer state, step count, etc.",
                placeholder="e.g., /path/to/training_state.safetensors",
                value=self.config.get("resume", ""),
            )

        with gr.Row():
            self.save_every_n_epochs = gr.Number(
                label="Save Every N Epochs",
                info="✅ 1 recommended. Save checkpoint every N epochs for backup and progress tracking. 0 = save only at end",
                value=self.config.get("save_every_n_epochs", 1),
                minimum=0,
                maximum=50,
                step=1,
                interactive=True,
            )

            self.save_every_n_steps = gr.Number(
                label="Save Every N Steps",
                info="Save checkpoint every N training steps. Leave empty to disable. Overrides save_every_n_epochs. Useful for long training",
                value=self.config.get("save_every_n_steps", None),
                minimum=1,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_epochs = gr.Number(
                label="Save Last N Epochs",
                info="Keep only last N epoch checkpoints (removes older ones). Example: 3 = keep only last 3 checkpoints. Leave empty to keep all",
                value=self.config.get("save_last_n_epochs", None),
                minimum=1,
                step=1,
                interactive=True,
            )

            self.save_last_n_epochs_state = gr.Number(
                label="Save Last N Epochs State",
                info="Keep last N optimizer states (larger files). 0=keep all. Overrides save_last_n_epochs for state files only",
                value=self.config.get("save_last_n_epochs_state", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_steps = gr.Number(
                label="Save Last N Steps",
                info="Keep only last N step checkpoints. 0=keep all. Example: 3=keep only last 3 step saves, delete older ones",
                value=self.config.get("save_last_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.save_last_n_steps_state = gr.Number(
                label="Save Last N Steps State",
                info="Keep last N optimizer states for step saves. 0=keep all. Overrides save_last_n_steps for state files only",
                value=self.config.get("save_last_n_steps_state", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_state = gr.Checkbox(
                label="Save Optimizer States with Checkpoints",
                info="Saves complete training state (optimizer, scheduler) for exact resume. Increases file size significantly but allows perfect training continuation",
                value=self.config.get("save_state", False),
            )

            self.save_state_on_train_end = gr.Checkbox(
                label="Save State on Training End",
                info="Save complete training state when training finishes, even if 'Save Optimizer States' is disabled. Useful for resuming training later",
                value=self.config.get("save_state_on_train_end", False),
            )


class QwenImageLatentCaching:
    """Qwen Image specific latent caching - removes VAE options not supported"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.caching_latent_device = gr.Textbox(
                label="Caching Device",
                info="Device for latent caching. 'cuda' = GPU (faster), 'cpu' = CPU (slower but uses less VRAM)",
                placeholder="cuda, cpu, cuda:0, cuda:1",
                value=self.config.get("caching_latent_device", "cuda"),
            )

            self.caching_latent_batch_size = gr.Number(
                label="Caching Batch Size",
                info="How many images to process at once. 4=conservative for 16GB VRAM, 8-16=faster with more VRAM. Reduce if OOM errors",
                value=self.config.get("caching_latent_batch_size", 4),
                minimum=1,
                maximum=64,
                step=1,
                interactive=True,
            )

            self.caching_latent_num_workers = gr.Number(
                label="Data Loading Workers",
                info="Parallel workers for loading images. 8=good default. Higher=faster loading but more RAM usage. 0=single-threaded",
                value=self.config.get("caching_latent_num_workers", 8),
                minimum=0,
                maximum=32,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.caching_latent_skip_existing = gr.Checkbox(
                label="Skip Existing Cache Files",
                info="Skip caching if cache files already exist. Disable to force re-caching all files",
                value=self.config.get("caching_latent_skip_existing", True),
            )

            self.caching_latent_keep_cache = gr.Checkbox(
                label="Keep Cache Files",
                info="Keep cached latent files after training. Recommended to enable for faster re-training with same dataset",
                value=self.config.get("caching_latent_keep_cache", True),
            )

        # Debug options (for compatibility)
        with gr.Row():
            self.caching_latent_debug_mode = gr.Dropdown(
                label="Debug Mode",
                info="Debug visualization for latent caching. 'image' saves debug images, 'console' prints debug info, 'video' for video models. Leave empty for no debugging",
                choices=["image", "console", "video"],
                allow_custom_value=True,
                value=self.config.get("caching_latent_debug_mode", None),
                interactive=True,
            )

            self.caching_latent_console_width = gr.Number(
                label="Console Width",
                info="Terminal width for debug console output formatting. Standard terminal = 80-120 characters",
                value=self.config.get("caching_latent_console_width", 80),
                minimum=40,
                maximum=200,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.caching_latent_console_back = gr.Textbox(
                label="Console Background",
                info="Background character for debug console visualization. Usually leave empty or use space/dot",
                placeholder="e.g., ' ' or '.' or leave empty",
                value=self.config.get("caching_latent_console_back", ""),
                interactive=True,
            )

            self.caching_latent_console_num_images = gr.Number(
                label="Console Number of Images",
                info="Max images to show in debug console. 0=no limit. Useful to prevent console spam with large datasets",
                value=self.config.get("caching_latent_console_num_images", 0),
                minimum=0,
                step=1,
                interactive=True,
            )


class QwenImageTextEncoderOutputsCaching:
    """Qwen Image specific text encoder caching"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.caching_teo_text_encoder = gr.Textbox(
                label="Text Encoder (Qwen2.5-VL) Path",
                info="Path to Qwen2.5-VL for text encoder caching. Usually same as main Text Encoder path above",
                placeholder="e.g., /path/to/qwen_2.5_vl_7b.safetensors",
                value=self.config.get("caching_teo_text_encoder", ""),
            )

            # Note: text_encoder_dtype not used in Qwen Image caching

        with gr.Row():
            self.caching_teo_device = gr.Textbox(
                label="Caching Device",
                info="Device for text encoder caching. 'cuda' = GPU (faster), 'cpu' = CPU (slower, use if GPU runs out of memory)",
                placeholder="cuda, cpu, cuda:0, cuda:1",
                value=self.config.get("caching_teo_device", "cuda"),
            )

            self.caching_teo_fp8_vl = gr.Checkbox(
                label="Use FP8 for VL Model",
                info="✅ ENABLED: Use FP8 quantization during caching to save VRAM. Usually matches main FP8 VL setting",
                value=self.config.get("caching_teo_fp8_vl", True),
            )

        with gr.Row():
            self.caching_teo_batch_size = gr.Number(
                label="Caching Batch Size",
                info="Text encoder batch size. 16=good default. Higher=faster but more VRAM. Reduce if getting OOM errors",
                value=self.config.get("caching_teo_batch_size", 16),
                minimum=1,
                maximum=128,
                step=1,
                interactive=True,
            )

            self.caching_teo_num_workers = gr.Number(
                label="Data Loading Workers",
                info="Parallel workers for loading text data. 8=good default. Higher=faster but more RAM usage",
                value=self.config.get("caching_teo_num_workers", 8),
                minimum=0,
                maximum=32,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.caching_teo_skip_existing = gr.Checkbox(
                label="Skip Existing Cache Files",
                info="Skip text encoder caching if cache files already exist. Disable to force re-caching all text embeddings",
                value=self.config.get("caching_teo_skip_existing", True),
            )

            self.caching_teo_keep_cache = gr.Checkbox(
                label="Keep Cache Files",
                info="Keep cached text encoder files after training. Recommended to enable for faster re-training with same text data",
                value=self.config.get("caching_teo_keep_cache", True),
            )


# Note: Default loading is now handled by TabConfigManager in the main GUI
# These functions are kept for reference but are no longer used


def qwen_image_lora_tab(
    headless=False,
    config: GUIConfig = {},
):
    # Configuration is now managed by TabConfigManager
    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    # Setup Configuration Files Gradio
    with gr.Accordion("Configuration file Settings", open=False):
        # Show configuration status
        # Check if this is a default config by looking for Qwen-specific values
        is_using_defaults = (hasattr(config, 'config') and 
                            config.config.get("discrete_flow_shift") == 3.0 and 
                            config.config.get("timestep_sampling") == "shift" and
                            config.config.get("fp8_vl") == True)
        
        if is_using_defaults:
            config_status = """
            ✅ **Qwen Image Optimal Defaults Active**
            
            **Key Optimizations Applied:**
            - Discrete Flow Shift: 3.0 (optimal for Qwen Image)
            - Optimizer: adamw8bit (memory efficient, recommended)
            - Learning Rate: 1e-4 (higher than default, per Qwen examples)
            - Mixed Precision: bf16 (strongly recommended)
            - SDPA Attention: Enabled (fastest for Qwen Image)
            - Gradient Checkpointing: Enabled (memory savings)
            """
        elif hasattr(config, 'config') and config.config:
            config_status = "📄 **Custom configuration loaded** - You may want to verify optimal settings are applied"
        else:
            config_status = "⚠️ **No configuration** - Default values will be used"
        
        gr.Markdown(config_status)
        configuration = ConfigurationFile(headless=headless, config=config)

    with gr.Accordion("Accelerate launch Settings", open=False, elem_classes="flux1_background"), gr.Column():
        accelerate_launch = AccelerateLaunch(config=config)
        # Note: bf16 mixed precision is STRONGLY recommended for Qwen Image
        
    with gr.Accordion("Qwen Image Model Settings", open=True, elem_classes="preset_background"):
        qwen_model = QwenImageModel(headless=headless, config=config)
        
    with gr.Accordion("Caching", open=True, elem_classes="samples_background"):
        with gr.Tab("Latent caching"):
            qwenLatentCaching = QwenImageLatentCaching(headless=headless, config=config)
                
        with gr.Tab("Text encoder caching"):
            qwenTeoCaching = QwenImageTextEncoderOutputsCaching(headless=headless, config=config)
        
    with gr.Accordion("Save Load Settings", open=True, elem_classes="samples_background"):
        saveLoadSettings = QwenImageSaveLoadSettings(headless=headless, config=config)
        
    with gr.Accordion("Optimizer and Scheduler Settings", open=True, elem_classes="flux1_rank_layers_background"):
        OptimizerAndSchedulerSettings = QwenImageOptimizerSettings(headless=headless, config=config)
        
    with gr.Accordion("Network Settings", open=True, elem_classes="flux1_background"):
        network = QwenImageNetworkSettings(headless=headless, config=config)
        
    with gr.Accordion("Training Settings", open=True, elem_classes="preset_background"):
        trainingSettings = QwenImageTrainingSettings(headless=headless, config=config)

    with gr.Accordion("Advanced Settings", open=False, elem_classes="samples_background"):
        gr.Markdown("**Additional Parameters**: Add custom training parameters as key=value pairs (e.g., `custom_param=value`). These will be appended to the training command.")
        advanced_training = AdvancedTraining(
            headless=headless, training_type="lora", config=config
        )

    with gr.Accordion("Metadata Settings", open=False, elem_classes="flux1_rank_layers_background"), gr.Group():
        metadata = MetaData(config=config)

    global huggingface
    with gr.Accordion("HuggingFace Settings", open=False, elem_classes="huggingface_background"):
        huggingface = HuggingFace(config=config)

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
        advanced_training.additional_parameters,
        
        # Dataset Settings
        qwen_model.dataset_config,
        
        # trainingSettings (flash3 not supported for Qwen Image)
        trainingSettings.sdpa,
        trainingSettings.flash_attn,
        trainingSettings.sage_attn,
        trainingSettings.xformers,
        trainingSettings.split_attn,
        trainingSettings.max_train_steps,
        trainingSettings.max_train_epochs,
        trainingSettings.max_data_loader_n_workers,
        trainingSettings.persistent_data_loader_workers,
        trainingSettings.seed,
        trainingSettings.gradient_checkpointing,
        trainingSettings.gradient_accumulation_steps,
        trainingSettings.logging_dir,
        trainingSettings.log_with,
        trainingSettings.log_prefix,
        trainingSettings.log_tracker_name,
        trainingSettings.wandb_run_name,
        trainingSettings.log_tracker_config,
        trainingSettings.wandb_api_key,
        trainingSettings.log_config,
        trainingSettings.ddp_timeout,
        trainingSettings.ddp_gradient_as_bucket_view,
        trainingSettings.ddp_static_graph,
        trainingSettings.sample_every_n_steps,
        trainingSettings.sample_at_first,
        trainingSettings.sample_every_n_epochs,
        trainingSettings.sample_prompts,
        
        # Qwen Image Latent Caching
        qwenLatentCaching.caching_latent_device,
        qwenLatentCaching.caching_latent_batch_size,
        qwenLatentCaching.caching_latent_num_workers,
        qwenLatentCaching.caching_latent_skip_existing,
        qwenLatentCaching.caching_latent_keep_cache,
        qwenLatentCaching.caching_latent_debug_mode,
        qwenLatentCaching.caching_latent_console_width,
        qwenLatentCaching.caching_latent_console_back,
        qwenLatentCaching.caching_latent_console_num_images,
        
        # Qwen Image Text Encoder Outputs Caching
        qwenTeoCaching.caching_teo_text_encoder,
        qwenTeoCaching.caching_teo_device,
        qwenTeoCaching.caching_teo_fp8_vl,
        qwenTeoCaching.caching_teo_batch_size,
        qwenTeoCaching.caching_teo_num_workers,
        qwenTeoCaching.caching_teo_skip_existing,
        qwenTeoCaching.caching_teo_keep_cache,
        
        # OptimizerAndSchedulerSettings
        OptimizerAndSchedulerSettings.optimizer_type,
        OptimizerAndSchedulerSettings.optimizer_args,
        OptimizerAndSchedulerSettings.learning_rate,
        OptimizerAndSchedulerSettings.max_grad_norm,
        OptimizerAndSchedulerSettings.lr_scheduler,
        OptimizerAndSchedulerSettings.lr_warmup_steps,
        OptimizerAndSchedulerSettings.lr_decay_steps,
        OptimizerAndSchedulerSettings.lr_scheduler_num_cycles,
        OptimizerAndSchedulerSettings.lr_scheduler_power,
        OptimizerAndSchedulerSettings.lr_scheduler_timescale,
        OptimizerAndSchedulerSettings.lr_scheduler_min_lr_ratio,
        OptimizerAndSchedulerSettings.lr_scheduler_type,
        OptimizerAndSchedulerSettings.lr_scheduler_args,
        
        # Qwen Image model settings
        qwen_model.dit,
        qwen_model.dit_dtype,
        qwen_model.vae,
        qwen_model.text_encoder,
        qwen_model.fp8_vl,
        qwen_model.fp8_base,
        qwen_model.fp8_scaled,
        qwen_model.blocks_to_swap,
        qwen_model.timestep_sampling,
        qwen_model.discrete_flow_shift,
        qwen_model.weighting_scheme,
        qwen_model.logit_mean,
        qwen_model.logit_std,
        qwen_model.mode_scale,
        qwen_model.show_timesteps,
        
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
        
        # saveLoadSettings
        saveLoadSettings.output_dir,
        saveLoadSettings.output_name,
        saveLoadSettings.resume,
        saveLoadSettings.save_every_n_epochs,
        saveLoadSettings.save_every_n_steps,
        saveLoadSettings.save_last_n_epochs,
        saveLoadSettings.save_last_n_epochs_state,
        saveLoadSettings.save_last_n_steps,
        saveLoadSettings.save_last_n_steps_state,
        saveLoadSettings.save_state,
        saveLoadSettings.save_state_on_train_end,
        
        # huggingface
        huggingface.huggingface_repo_id,
        huggingface.huggingface_token,
        huggingface.huggingface_repo_type,
        huggingface.huggingface_repo_visibility,
        huggingface.huggingface_path_in_repo,
        huggingface.save_state_to_huggingface,
        huggingface.resume_from_huggingface,
        huggingface.async_upload,
        
        # metadata
        metadata.metadata_author,
        metadata.metadata_description,
        metadata.metadata_license,
        metadata.metadata_tags,
        metadata.metadata_title,
    ]

    run_state = gr.Textbox(value=train_state_value, visible=False)

    with gr.Column(), gr.Group():
        with gr.Row():
            button_print = gr.Button("Print training command")

    global executor
    executor = CommandExecutor(headless=headless)

    configuration.button_open_config.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name] + settings_list,
        show_progress=False,
    )

    configuration.button_load_config.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name] + settings_list,
        show_progress=False,
    )

    configuration.button_save_config.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name],
        show_progress=False,
    )

    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[executor.button_run, executor.button_stop_training],
    )

    button_print.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[executor.button_run, executor.button_stop_training, run_state],
        show_progress=False,
    )

    executor.button_stop_training.click(
        executor.kill_command,
        outputs=[executor.button_run, executor.button_stop_training],
    )