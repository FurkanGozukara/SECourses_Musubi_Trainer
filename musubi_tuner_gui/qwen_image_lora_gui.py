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
                label="Dataset Config",
                placeholder='Path to the dataset config file',
                value=str(self.config.get("dataset_config", "")),
            )

        with gr.Row():
            self.dit = gr.Textbox(
                label="DiT Checkpoint Path",
                placeholder="Path to DiT checkpoint",
                value=self.config.get("dit", ""),
            )

            self.dit_dtype = gr.Dropdown(
                label="DiT Data Type",
                info="Data type for DiT (bfloat16 recommended and hardcoded for Qwen Image)",
                choices=["bfloat16"],
                value=self.config.get("dit_dtype", "bfloat16"),
                interactive=False,  # Hardcoded for Qwen Image
            )

        with gr.Row():
            self.vae = gr.Textbox(
                label="VAE Checkpoint Path",
                placeholder="Path to VAE checkpoint",
                value=self.config.get("vae", ""),
            )
            # Note: vae_dtype is not supported in Qwen Image, removed from GUI

        # Qwen Image specific text encoder
        with gr.Row():
            self.text_encoder = gr.Textbox(
                label="Text Encoder (Qwen2.5-VL) Path",
                placeholder="Path to Qwen2.5-VL text encoder",
                value=self.config.get("text_encoder", ""),
            )

            self.fp8_vl = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                info="Use FP8 for Qwen2.5-VL text encoder (STRONGLY recommended for <16GB VRAM)",
                value=self.config.get("fp8_vl", True),
            )

        # Qwen Image specific options
        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="Use FP8 for Base Model",
                info="Use FP8 quantization for DiT base model (memory savings)",
                value=self.config.get("fp8_base", False),
            )
            
            self.fp8_scaled = gr.Checkbox(
                label="Use Scaled FP8 for DiT",
                info="Use scaled FP8 for DiT (recommended when using fp8_base)",
                value=self.config.get("fp8_scaled", False),
            )
            
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap",
                info="Number of blocks to swap to CPU (for memory savings)",
                value=self.config.get("blocks_to_swap", None),
                step=1,
                interactive=True,
            )

        # Flow matching parameters
        with gr.Row():
            self.timestep_sampling = gr.Dropdown(
                label="Timestep Sampling Method",
                info="'shift' recommended for Qwen Image, 'qwen_shift' uses dynamic shift",
                choices=["shift", "qwen_shift", "uniform", "sigmoid"],
                value=self.config.get("timestep_sampling", "shift"),
                interactive=True,
            )

            self.discrete_flow_shift = gr.Number(
                label="Discrete Flow Shift",
                info="Qwen Image uses 3.0 (lower than other models), qwen_shift ignores this",
                value=self.config.get("discrete_flow_shift", 3.0),
                step=0.1,
                interactive=True,
            )

            self.weighting_scheme = gr.Dropdown(
                label="Weighting Scheme",
                info="Weighting scheme for timestep distribution (none recommended)",
                choices=["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"],
                value=self.config.get("weighting_scheme", "none"),
                interactive=True,
            )

        # Weighting scheme parameters
        with gr.Row():
            self.logit_mean = gr.Number(
                label="Logit Mean",
                info="Mean for 'logit_normal' weighting scheme",
                value=self.config.get("logit_mean", 0.0),
                step=0.001,
                interactive=True,
            )

            self.logit_std = gr.Number(
                label="Logit Std",
                info="Standard deviation for 'logit_normal' weighting scheme",
                value=self.config.get("logit_std", 1.0),
                step=0.001,
                interactive=True,
            )

            self.mode_scale = gr.Number(
                label="Mode Scale",
                info="Scale of mode weighting scheme",
                value=self.config.get("mode_scale", 1.29),
                step=0.001,
                interactive=True,
            )

        with gr.Row():
            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
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
    run_cmd = [rf"uv", "run", "accelerate", "launch"]

    param_dict = dict(parameters)
    
    # Cache latents using Qwen Image specific script
    run_cache_latent_cmd = ["uv", "run", "./musubi-tuner/qwen_image_cache_latents.py",
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
    subprocess.run(run_cache_latent_cmd, env=setup_environment())
    log.debug("Latent caching completed.")
    
    # Cache text encoder outputs using Qwen Image specific script
    run_cache_teo_cmd = ["uv", "run", "./musubi-tuner/qwen_image_cache_text_encoder_outputs.py",
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
    subprocess.run(run_cache_teo_cmd, env=setup_environment())
    log.debug("Text encoder output caching completed.")

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
    run_cmd.append(rf"{scriptdir}/musubi-tuner/qwen_image_train_network.py")

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
        run_cmd.append(rf"{file_path}")

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
                info="SDPA is recommended for Qwen Image (faster than others)",
                value=self.config.get("sdpa", True),
            )

            self.flash_attn = gr.Checkbox(
                label="FlashAttention",
                info="Use FlashAttention for CrossAttention (use split_attn if enabled)",
                value=self.config.get("flash_attn", False),
            )

            self.sage_attn = gr.Checkbox(
                label="SageAttention",
                info="Use SageAttention for CrossAttention (use split_attn if enabled)",
                value=self.config.get("sage_attn", False),
            )

            self.xformers = gr.Checkbox(
                label="xformers",
                info="Use xformers for CrossAttention (use split_attn if enabled)",
                value=self.config.get("xformers", False),
            )

            self.split_attn = gr.Checkbox(
                label="Split Attention",
                info="REQUIRED if using anything other than SDPA (reduces memory)",
                value=self.config.get("split_attn", False),
            )

        with gr.Row():
            self.max_train_steps = gr.Number(
                label="Max Training Steps",
                info="Maximum number of training steps (default 1600)",
                value=self.config.get("max_train_steps", 1600),
                interactive=True,
            )

            self.max_train_epochs = gr.Number(
                label="Max Training Epochs",
                info="Overrides max_train_steps (16 recommended for Qwen Image)",
                value=self.config.get("max_train_epochs", 16),
            )

            self.max_data_loader_n_workers = gr.Number(
                label="Max DataLoader Workers",
                info="Qwen Image recommends 2 workers for stability",
                value=self.config.get("max_data_loader_n_workers", 2),
                interactive=True,
            )

            self.persistent_data_loader_workers = gr.Checkbox(
                label="Persistent DataLoader Workers",
                info="Keep workers alive between epochs (recommended)",
                value=self.config.get("persistent_data_loader_workers", True),
            )

        with gr.Row():
            self.seed = gr.Number(
                label="Random Seed for Training",
                info="Set to 42 for reproducible results",
                value=self.config.get("seed", 42),
            )

            self.gradient_checkpointing = gr.Checkbox(
                label="Enable Gradient Checkpointing",
                info="STRONGLY recommended for memory savings with Qwen Image",
                value=self.config.get("gradient_checkpointing", True),
            )

            self.gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                info="Number of steps to accumulate gradients before backward pass",
                value=self.config.get("gradient_accumulation_steps", 1),
                interactive=True,
            )

        # Logging settings
        with gr.Row():
            self.logging_dir = gr.Textbox(
                label="Logging Directory",
                placeholder="Directory for TensorBoard logs",
                value=self.config.get("logging_dir", ""),
            )

            self.log_with = gr.Dropdown(
                label="Logging Tool",
                choices=["tensorboard", "wandb", "all"],
                allow_custom_value=True,
                value=self.config.get("log_with", None),
                interactive=True,
            )

        with gr.Row():
            self.log_prefix = gr.Textbox(
                label="Log Prefix",
                placeholder="Prefix for log directory names",
                value=self.config.get("log_prefix", ""),
            )

            self.log_tracker_name = gr.Textbox(
                label="Log Tracker Name",
                placeholder="Name for tracker",
                value=self.config.get("log_tracker_name", ""),
            )

        # Sample generation settings
        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                info="Generate samples during training",
                value=self.config.get("sample_every_n_steps", None),
                step=1,
                interactive=True,
            )

            self.sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs",
                info="Generate samples every N epochs",
                value=self.config.get("sample_every_n_epochs", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.sample_at_first = gr.Checkbox(
                label="Sample at First",
                info="Generate sample before training starts",
                value=self.config.get("sample_at_first", False),
            )

            self.sample_prompts = gr.Textbox(
                label="Sample Prompts File",
                placeholder="Path to prompts file for sample generation",
                value=self.config.get("sample_prompts", ""),
            )

        # Additional settings
        with gr.Row():
            self.wandb_run_name = gr.Textbox(
                label="WandB Run Name",
                placeholder="Name for WandB run",
                value=self.config.get("wandb_run_name", ""),
            )

            self.log_tracker_config = gr.Textbox(
                label="Log Tracker Config",
                placeholder="Path to tracker config file",
                value=self.config.get("log_tracker_config", ""),
            )

        with gr.Row():
            self.wandb_api_key = gr.Textbox(
                label="WandB API Key",
                placeholder="WandB API key",
                type="password",
                value=self.config.get("wandb_api_key", ""),
            )

            self.log_config = gr.Checkbox(
                label="Log Training Configuration",
                info="Log training config to tracker",
                value=self.config.get("log_config", False),
            )

        # DDP settings
        with gr.Row():
            self.ddp_timeout = gr.Number(
                label="DDP Timeout (minutes)",
                info="DDP timeout in minutes",
                value=self.config.get("ddp_timeout", None),
                step=1,
                interactive=True,
            )

            self.ddp_gradient_as_bucket_view = gr.Checkbox(
                label="DDP Gradient as Bucket View",
                value=self.config.get("ddp_gradient_as_bucket_view", False),
            )

            self.ddp_static_graph = gr.Checkbox(
                label="DDP Static Graph",
                value=self.config.get("ddp_static_graph", False),
            )

        with gr.Row():
            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
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
            self.optimizer_type = gr.Textbox(
                label="Optimizer Type",
                info="adamw8bit recommended for Qwen Image training",
                placeholder="adamw8bit (recommended)",
                value=self.config.get("optimizer_type", "adamw8bit"),
            )

            self.learning_rate = gr.Number(
                label="Learning Rate",
                info="Qwen Image example uses 1e-4 (higher than default 2e-6)",
                value=self.config.get("learning_rate", 1e-4),
                step=1e-6,
                interactive=True,
            )

        with gr.Row():
            self.optimizer_args = gr.Textbox(
                label="Optimizer Arguments",
                placeholder='e.g. "weight_decay=0.01 betas=0.9,0.999"',
                value=self.config.get("optimizer_args", ""),
            )

            self.max_grad_norm = gr.Number(
                label="Max Gradient Norm",
                info="Gradient clipping (0 to disable)",
                value=self.config.get("max_grad_norm", 1.0),
                step=0.1,
                interactive=True,
            )

        # Learning rate scheduler settings
        with gr.Row():
            self.lr_scheduler = gr.Dropdown(
                label="Learning Rate Scheduler",
                choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup", "adafactor"],
                value=self.config.get("lr_scheduler", "constant"),
                interactive=True,
            )

            self.lr_warmup_steps = gr.Number(
                label="LR Warmup Steps",
                info="Warmup steps or ratio (float <1)",
                value=self.config.get("lr_warmup_steps", 0),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_decay_steps = gr.Number(
                label="LR Decay Steps",
                value=self.config.get("lr_decay_steps", 0),
                step=1,
                interactive=True,
            )

            self.lr_scheduler_num_cycles = gr.Number(
                label="LR Scheduler Cycles",
                info="For cosine_with_restarts",
                value=self.config.get("lr_scheduler_num_cycles", 1),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_power = gr.Number(
                label="LR Scheduler Power",
                info="For polynomial scheduler",
                value=self.config.get("lr_scheduler_power", 1.0),
                step=0.1,
                interactive=True,
            )

            self.lr_scheduler_timescale = gr.Number(
                label="LR Scheduler Timescale",
                value=self.config.get("lr_scheduler_timescale", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.lr_scheduler_min_lr_ratio = gr.Number(
                label="LR Min Ratio",
                info="Minimum LR as ratio of initial LR",
                value=self.config.get("lr_scheduler_min_lr_ratio", None),
                step=0.01,
                interactive=True,
            )

            self.lr_scheduler_type = gr.Textbox(
                label="Custom Scheduler Module",
                value=self.config.get("lr_scheduler_type", ""),
            )

        with gr.Row():
            self.lr_scheduler_args = gr.Textbox(
                label="Scheduler Arguments",
                placeholder='e.g. "T_max=100"',
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
                info="Don't save metadata in output model",
                value=self.config.get("no_metadata", False),
            )

            self.network_weights = gr.Textbox(
                label="Network Weights",
                placeholder="Pretrained LoRA weights to start from",
                value=self.config.get("network_weights", ""),
            )

        with gr.Row():
            self.network_module = gr.Textbox(
                label="Network Module",
                info="Auto-selected for Qwen Image (lora_qwen_image)",
                placeholder="networks.lora",
                value=self.config.get("network_module", "networks.lora"),
                interactive=False,  # Will be auto-selected
            )

            self.network_dim = gr.Number(
                label="Network Dimension (Rank)",
                info="LoRA rank - 32 recommended for Qwen Image",
                value=self.config.get("network_dim", 32),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.network_alpha = gr.Number(
                label="Network Alpha",
                info="LoRA alpha (1.0 recommended)",
                value=self.config.get("network_alpha", 1.0),
                step=0.1,
                interactive=True,
            )

            self.network_dropout = gr.Number(
                label="Network Dropout",
                info="Dropout rate (None for no dropout)",
                value=self.config.get("network_dropout", None),
                step=0.01,
                interactive=True,
            )

        with gr.Row():
            self.network_args = gr.Textbox(
                label="Network Arguments",
                placeholder="Additional network arguments",
                value=self.config.get("network_args", ""),
            )

            self.training_comment = gr.Textbox(
                label="Training Comment",
                placeholder="Comment for metadata",
                value=self.config.get("training_comment", ""),
            )

        with gr.Row():
            self.dim_from_weights = gr.Checkbox(
                label="Dim from Weights",
                info="Auto-determine rank from network weights",
                value=self.config.get("dim_from_weights", False),
            )

            self.scale_weight_norms = gr.Number(
                label="Scale Weight Norms",
                info="Scale weights to prevent exploding gradients",
                value=self.config.get("scale_weight_norms", None),
                step=0.1,
                interactive=True,
            )

        with gr.Row():
            self.base_weights = gr.Textbox(
                label="Base Weights",
                placeholder="LoRA weights to merge before training",
                value=self.config.get("base_weights", ""),
            )

            self.base_weights_multiplier = gr.Number(
                label="Base Weights Multiplier",
                info="Multiplier for base weights",
                value=self.config.get("base_weights_multiplier", None),
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
                placeholder="Directory to save trained model",
                value=self.config.get("output_dir", ""),
            )

            self.output_name = gr.Textbox(
                label="Output Name",
                placeholder="Base name for saved model files",
                value=self.config.get("output_name", ""),
            )

        with gr.Row():
            self.resume = gr.Textbox(
                label="Resume",
                placeholder="Path to state file to resume training",
                value=self.config.get("resume", ""),
            )

        with gr.Row():
            self.save_every_n_epochs = gr.Number(
                label="Save Every N Epochs",
                info="Save checkpoint every N epochs (1 recommended)",
                value=self.config.get("save_every_n_epochs", 1),
                step=1,
                interactive=True,
            )

            self.save_every_n_steps = gr.Number(
                label="Save Every N Steps",
                value=self.config.get("save_every_n_steps", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_epochs = gr.Number(
                label="Save Last N Epochs",
                info="Keep only last N epoch checkpoints",
                value=self.config.get("save_last_n_epochs", None),
                step=1,
                interactive=True,
            )

            self.save_last_n_epochs_state = gr.Number(
                label="Save Last N Epochs State",
                value=self.config.get("save_last_n_epochs_state", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_steps = gr.Number(
                label="Save Last N Steps",
                value=self.config.get("save_last_n_steps", None),
                step=1,
                interactive=True,
            )

            self.save_last_n_steps_state = gr.Number(
                label="Save Last N Steps State",
                value=self.config.get("save_last_n_steps_state", None),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_state = gr.Checkbox(
                label="Save State",
                info="Save optimizer states with checkpoints",
                value=self.config.get("save_state", False),
            )

            self.save_state_on_train_end = gr.Checkbox(
                label="Save State on Train End",
                info="Save state at end of training",
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
                label="Device",
                placeholder="Device for caching (e.g., cuda, cpu)",
                value=self.config.get("caching_latent_device", "cuda"),
            )

            self.caching_latent_batch_size = gr.Number(
                label="Batch Size",
                info="Conservative batch size for memory efficiency",
                value=self.config.get("caching_latent_batch_size", 4),
                step=1,
                interactive=True,
            )

            self.caching_latent_num_workers = gr.Number(
                label="Number of Workers",
                info="Number of workers for dataset loading",
                value=self.config.get("caching_latent_num_workers", 8),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.caching_latent_skip_existing = gr.Checkbox(
                label="Skip Existing",
                value=self.config.get("caching_latent_skip_existing", True),
            )

            self.caching_latent_keep_cache = gr.Checkbox(
                label="Keep Cache",
                value=self.config.get("caching_latent_keep_cache", True),
            )

        # Debug options (for compatibility)
        with gr.Row():
            self.caching_latent_debug_mode = gr.Dropdown(
                label="Debug Mode",
                choices=["image", "console", "video"],
                allow_custom_value=True,
                value=self.config.get("caching_latent_debug_mode", None),
                interactive=True,
            )

            self.caching_latent_console_width = gr.Number(
                label="Console Width",
                value=self.config.get("caching_latent_console_width", 80),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.caching_latent_console_back = gr.Textbox(
                label="Console Background",
                value=self.config.get("caching_latent_console_back", ""),
                interactive=True,
            )

            self.caching_latent_console_num_images = gr.Number(
                label="Console Number of Images",
                value=self.config.get("caching_latent_console_num_images", None),
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
                placeholder="Path to Qwen2.5-VL text encoder",
                value=self.config.get("caching_teo_text_encoder", ""),
            )

            # Note: text_encoder_dtype not used in Qwen Image caching

        with gr.Row():
            self.caching_teo_device = gr.Textbox(
                label="Device",
                placeholder="Device for caching (e.g., cuda, cpu)",
                value=self.config.get("caching_teo_device", "cuda"),
            )

            self.caching_teo_fp8_vl = gr.Checkbox(
                label="Use FP8 for VL Model",
                info="Use FP8 for Qwen2.5-VL during caching (recommended for <16GB VRAM)",
                value=self.config.get("caching_teo_fp8_vl", True),
            )

        with gr.Row():
            self.caching_teo_batch_size = gr.Number(
                label="Batch Size",
                value=self.config.get("caching_teo_batch_size", 16),
                step=1,
                interactive=True,
            )

            self.caching_teo_num_workers = gr.Number(
                label="Number of Workers",
                value=self.config.get("caching_teo_num_workers", 8),
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.caching_teo_skip_existing = gr.Checkbox(
                label="Skip Existing",
                value=self.config.get("caching_teo_skip_existing", True),
            )

            self.caching_teo_keep_cache = gr.Checkbox(
                label="Keep Cache",
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
            config_status = "✅ **Qwen Image optimal defaults loaded**"
        elif hasattr(config, 'config') and config.config:
            config_status = "📄 **Custom configuration loaded**"
        else:
            config_status = "⚠️ **No configuration**"
        
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

    with gr.Accordion("Advanced Settings", open=True, elem_classes="samples_background"):
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