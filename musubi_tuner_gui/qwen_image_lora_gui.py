import gradio as gr
import os
import re
import shutil
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
    get_file_path_or_save_as,
    get_folder_path,
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
from .dataset_config_generator import (
    generate_dataset_config_from_folders,
    save_dataset_config,
    validate_dataset_config
)

log = setup_logging()

executor = None
huggingface = None
train_state_value = time.time()


class QwenImageDataset:
    """Qwen Image dataset configuration settings"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        # Dataset configuration mode selection
        with gr.Row():
            self.dataset_config_mode = gr.Radio(
                label="Dataset Configuration Method",
                choices=["Use TOML File", "Generate from Folder Structure"],
                value=self.config.get("dataset_config_mode", "Use TOML File"),
                info="Choose how to configure your dataset: provide a TOML file or auto-generate from folder structure"
            )
        
        # TOML file mode
        with gr.Row(visible=self.config.get("dataset_config_mode", "Use TOML File") == "Use TOML File") as self.toml_mode_row:
            self.dataset_config = gr.Textbox(
                label="Dataset Config File",
                info="REQUIRED: Path to TOML file for training. This is the file path the model will use. Must exist before starting training!",
                placeholder='e.g., /path/to/dataset.toml',
                value=str(self.config.get("dataset_config", "")),
            )
        
        # Folder structure mode
        with gr.Column(visible=self.config.get("dataset_config_mode", "Use TOML File") == "Generate from Folder Structure") as self.folder_mode_column:
            with gr.Row():
                with gr.Column(scale=8):
                    self.parent_folder_path = gr.Textbox(
                        label="Parent Folder Path",
                        info="Path to parent folder containing subfolders with images. Each subfolder can have format: [repeats]_[name] (e.g., 3_ohwx, 2_style)",
                        placeholder="e.g., C:\\Users\\Name\\Pictures\\training_data",
                        value=self.config.get("parent_folder_path", "")
                    )
                self.parent_folder_button = gr.Button(
                    "📂", 
                    elem_id="parent_folder_button", 
                    size="sm"
                )
            
            with gr.Row():
                self.dataset_resolution_width = gr.Number(
                    label="Resolution Width",
                    value=self.config.get("dataset_resolution_width", 960),
                    minimum=64,
                    maximum=4096,
                    step=64,
                    info="Width of training images"
                )
                self.dataset_resolution_height = gr.Number(
                    label="Resolution Height",
                    value=self.config.get("dataset_resolution_height", 544),
                    minimum=64,
                    maximum=4096,
                    step=64,
                    info="Height of training images"
                )
            
            with gr.Row():
                self.dataset_caption_extension = gr.Textbox(
                    label="Caption Extension",
                    value=self.config.get("dataset_caption_extension", ".txt"),
                    info="File extension for caption files (e.g., .txt, .caption)"
                )
                self.dataset_batch_size = gr.Number(
                    label="Batch Size",
                    value=self.config.get("dataset_batch_size", 1),
                    minimum=1,
                    maximum=64,
                    step=1,
                    info="Training batch size per dataset"
                )
            
            with gr.Row():
                self.create_missing_captions = gr.Checkbox(
                    label="Create Missing Captions",
                    value=self.config.get("create_missing_captions", True),
                    info="Automatically create caption files for images that don't have them"
                )
                self.caption_strategy = gr.Dropdown(
                    label="Caption Strategy",
                    choices=["folder_name", "empty"],
                    value=self.config.get("caption_strategy", "folder_name"),
                    info="folder_name: Use folder name (without repeat prefix) as caption | empty: Create empty caption files",
                    visible=self.config.get("create_missing_captions", True)
                )
            
            with gr.Row():
                self.dataset_enable_bucket = gr.Checkbox(
                    label="Enable Bucketing",
                    value=self.config.get("dataset_enable_bucket", False),
                    info="Enable aspect ratio bucketing to train on images with different aspect ratios"
                )
                self.dataset_bucket_no_upscale = gr.Checkbox(
                    label="Bucket No Upscale",
                    value=self.config.get("dataset_bucket_no_upscale", False),
                    info="Don't upscale images when bucketing (maintains original size if smaller than target)"
                )
            
            with gr.Row():
                self.dataset_cache_directory = gr.Textbox(
                    label="Cache Directory Name",
                    value=self.config.get("dataset_cache_directory", "cache_dir"),
                    info="Cache folder name (relative) or full path (absolute). Each dataset gets its own cache directory to avoid conflicts"
                )
                self.dataset_control_directory = gr.Textbox(
                    label="Control Directory Name",
                    value=self.config.get("dataset_control_directory", "edit_images"),
                    info="Name for control/edit images directory (for image editing). Expected inside each dataset folder"
                )
            
            with gr.Row():
                self.dataset_qwen_image_edit_no_resize_control = gr.Checkbox(
                    label="Qwen Image Edit: No Resize Control",
                    value=self.config.get("dataset_qwen_image_edit_no_resize_control", False),
                    info="Don't resize control images to target resolution (keeps original control image size)"
                )
            
            with gr.Row():
                self.generate_toml_button = gr.Button(
                    "Generate Dataset Configuration",
                    variant="primary"
                )
                self.generated_toml_path = gr.Textbox(
                    label="Generated TOML Path",
                    value=self.config.get("generated_toml_path", ""),
                    info="Display only. This path is auto-copied to 'Dataset Config File' field. Training ALWAYS uses the 'Dataset Config File' path.",
                    interactive=False
                )
            
            with gr.Row():
                self.copy_generated_path_button = gr.Button(
                    "📋 Copy Generated TOML Path to Dataset Config",
                    variant="secondary"
                )
            
            self.dataset_status = gr.Textbox(
                label="Dataset Status",
                value="",
                lines=5,
                interactive=False,
                info="Status messages from dataset configuration generation"
            )
    
    def setup_dataset_ui_events(self, saveLoadSettings=None):
        """Setup event handlers for dataset configuration UI"""
        
        def toggle_dataset_mode(mode):
            """Toggle visibility of dataset configuration modes"""
            is_toml = mode == "Use TOML File"
            return (
                gr.Row(visible=is_toml),  # toml_mode_row
                gr.Column(visible=not is_toml),  # folder_mode_column
            )
        
        def toggle_caption_strategy(create_missing):
            """Toggle visibility of caption strategy dropdown"""
            return gr.Dropdown(visible=create_missing)
        
        def browse_parent_folder():
            """Browse for parent folder"""
            folder_path = get_folder_path()
            return folder_path if folder_path else gr.update()
        
        def generate_dataset_config(
            parent_folder,
            width, height,
            caption_ext,
            create_missing,
            caption_strat,
            batch_size,
            enable_bucket,
            bucket_no_upscale,
            cache_dir,
            control_dir,
            qwen_no_resize,
            output_dir  # Add output_dir parameter
        ):
            """Generate dataset configuration from folder structure"""
            try:
                if not parent_folder:
                    return "", "", "[ERROR] Please specify a parent folder path containing your dataset folders"
                
                if not os.path.exists(parent_folder):
                    return "", "", f"[ERROR] Parent folder does not exist: {parent_folder}"
                
                # Create caption files if requested
                if create_missing:
                    subfolder_paths = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) 
                                     if os.path.isdir(os.path.join(parent_folder, d))]
                    
                    for folder_path in subfolder_paths:
                        # Parse folder name to get name without repeat count
                        folder_name = os.path.basename(folder_path)
                        match = re.match(r"^(\d+)_(.+)$", folder_name)
                        
                        if match and caption_strat == "folder_name":
                            # Use the name part only (without repeat count)
                            caption_text = match.group(2)
                        elif caption_strat == "folder_name":
                            # Use full folder name if no repeat format
                            caption_text = folder_name
                        else:
                            # Empty caption
                            caption_text = ""
                        
                        # Create caption files for all images without captions
                        for img_file in os.listdir(folder_path):
                            img_path = os.path.join(folder_path, img_file)
                            if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                                caption_path = os.path.splitext(img_path)[0] + caption_ext
                                if not os.path.exists(caption_path):
                                    with open(caption_path, 'w', encoding='utf-8') as f:
                                        f.write(caption_text)
                
                # Generate the dataset configuration
                config, messages = generate_dataset_config_from_folders(
                    parent_folder=parent_folder,
                    resolution=(int(width), int(height)),
                    caption_extension=caption_ext,
                    create_missing_captions=False,  # We already created them above
                    caption_strategy="folder_name",
                    batch_size=int(batch_size),
                    enable_bucket=enable_bucket,
                    bucket_no_upscale=bucket_no_upscale,
                    cache_directory_name=cache_dir,
                    control_directory_name=control_dir,
                    qwen_image_edit_no_resize_control=qwen_no_resize
                )
                
                # Check if config generation was successful
                if not config or not config.get("datasets"):
                    return "", "", "[ERROR] Failed to generate configuration. Check your folder structure.\n" + "\n".join(messages)
                
                # Generate output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Use output_dir from saveLoadSettings if available
                if output_dir and os.path.exists(output_dir):
                    output_path = os.path.join(output_dir, f"dataset_config_{timestamp}.toml")
                else:
                    # Default to parent folder
                    output_path = os.path.join(parent_folder, f"dataset_config_{timestamp}.toml")
                
                # Save the configuration
                save_dataset_config(config, output_path)
                
                # Get dataset info for status message
                num_datasets = len(config.get("datasets", []))
                
                status_msg = f"[SUCCESS] Generated dataset configuration:\n"
                status_msg += f"  Output: {output_path}\n"
                status_msg += f"  Datasets: {num_datasets}\n"
                status_msg += f"\n" + "\n".join(messages)
                
                if create_missing:
                    status_msg += f"\n  Caption files created with strategy: {caption_strat}"
                
                # Return both paths - output_path for dataset_config field and display
                return output_path, output_path, status_msg
                
            except Exception as e:
                error_msg = f"[ERROR] Failed to generate dataset configuration:\n{str(e)}"
                log.error(error_msg)
                import traceback
                traceback.print_exc()
                return "", "", error_msg
        
        def copy_generated_path(generated_path):
            """Copy generated TOML path to dataset config field"""
            if generated_path:
                return generated_path
            return gr.update()
        
        # Event handlers
        self.dataset_config_mode.change(
            fn=toggle_dataset_mode,
            inputs=[self.dataset_config_mode],
            outputs=[self.toml_mode_row, self.folder_mode_column]
        )
        
        self.create_missing_captions.change(
            fn=toggle_caption_strategy,
            inputs=[self.create_missing_captions],
            outputs=[self.caption_strategy]
        )
        
        self.parent_folder_button.click(
            fn=browse_parent_folder,
            outputs=[self.parent_folder_path]
        )
        
        # Bind generate button
        if hasattr(self, 'generate_toml_button'):
            # Pass output_dir from saveLoadSettings if available
            if saveLoadSettings and hasattr(saveLoadSettings, 'output_dir'):
                self.generate_toml_button.click(
                    fn=generate_dataset_config,
                    inputs=[
                        self.parent_folder_path,
                        self.dataset_resolution_width,
                        self.dataset_resolution_height,
                        self.dataset_caption_extension,
                        self.create_missing_captions,
                        self.caption_strategy,
                        self.dataset_batch_size,
                        self.dataset_enable_bucket,
                        self.dataset_bucket_no_upscale,
                        self.dataset_cache_directory,
                        self.dataset_control_directory,
                        self.dataset_qwen_image_edit_no_resize_control,
                        saveLoadSettings.output_dir  # Pass output_dir
                    ],
                    outputs=[self.dataset_config, self.generated_toml_path, self.dataset_status]
                )
            else:
                # Fallback without output_dir
                self.generate_toml_button.click(
                    fn=lambda *args: generate_dataset_config(*args, None),  # Pass None for output_dir
                    inputs=[
                        self.parent_folder_path,
                        self.dataset_resolution_width,
                        self.dataset_resolution_height,
                        self.dataset_caption_extension,
                        self.create_missing_captions,
                        self.caption_strategy,
                        self.dataset_batch_size,
                        self.dataset_enable_bucket,
                        self.dataset_bucket_no_upscale,
                        self.dataset_cache_directory,
                        self.dataset_control_directory,
                        self.dataset_qwen_image_edit_no_resize_control,
                    ],
                    outputs=[self.dataset_config, self.generated_toml_path, self.dataset_status]
                )
        
        # Bind copy button
        if hasattr(self, 'copy_generated_path_button'):
            self.copy_generated_path_button.click(
                fn=copy_generated_path,
                inputs=[self.generated_toml_path],
                outputs=[self.dataset_config]
            )


class QwenImageModel:
    """Qwen Image specific model settings"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            with gr.Column(scale=4):
                self.dit = gr.Textbox(
                    label="DiT (Base Model) Checkpoint Path",
                    placeholder="Path to DiT base model checkpoint (qwen_image_bf16.safetensors)",
                    value=self.config.get("dit", ""),
                    info="e.g. qwen_image_bf16.safetensors : Must be standard bf16/fp16 model. FP8 pre-quantized models NOT supported. FP8 conversion happens automatically if enabled below"
                )
            self.dit_button = gr.Button(
                "📁",
                size="sm",
                elem_id="dit_button"
            )
            with gr.Column(scale=1):
                self.dit_dtype = gr.Dropdown(
                    label="DiT Computation Data Type",
                    info="[HARDCODED] bfloat16 computation precision is required for Qwen Image training. FP8 settings below control weight storage only",
                    choices=["bfloat16"],
                    value=self.config.get("dit_dtype", "bfloat16"),
                    interactive=False,  # Hardcoded for Qwen Image
                )

        with gr.Row():
            with gr.Column(scale=4):
                self.vae = gr.Textbox(
                    label="VAE Checkpoint Path",
                    info="e.g. qwen_train_vae.safetensors : REQUIRED: Path to VAE model (diffusion_pytorch_model.safetensors from Qwen/Qwen-Image). NOT ComfyUI VAE!",
                    placeholder="e.g., /path/to/vae/diffusion_pytorch_model.safetensors",
                    value=self.config.get("vae", ""),
                )
            self.vae_button = gr.Button(
                "📁",
                size="sm",
                elem_id="vae_button"
            )
            with gr.Column(scale=1):
                self.vae_dtype = gr.Dropdown(
                    label="VAE Data Type",
                    info="Data type for VAE model. bfloat16 = Qwen Image default, float16 = faster, float32 = highest precision",
                    choices=["bfloat16", "float16", "float32"],
                    value=self.config.get("vae_dtype", "bfloat16"),
                    interactive=True,
                )

        # Qwen Image specific text encoder
        with gr.Row():
            with gr.Column(scale=4):
                self.text_encoder = gr.Textbox(
                    label="Text Encoder (Qwen2.5-VL) Path",
                    info="e.g. qwen_2.5_vl_7b_bf16.safetensors : REQUIRED: Path to Qwen2.5-VL text encoder model (qwen_2.5_vl_7b.safetensors from Comfy-Org/Qwen-Image_ComfyUI)",
                    placeholder="e.g., /path/to/text_encoder/qwen_2.5_vl_7b.safetensors",
                    value=self.config.get("text_encoder", ""),
                )
            self.text_encoder_button = gr.Button(
                "📁",
                size="sm",
                elem_id="text_encoder_button"
            )
            with gr.Column(scale=1):
                self.text_encoder_dtype = gr.Dropdown(
                    label="Text Encoder Data Type",
                    info="Data type for Qwen2.5-VL text encoder. float16 = faster, bfloat16 = better precision",
                    choices=["float16", "bfloat16", "float32"],
                    value=self.config.get("text_encoder_dtype", "float16"),
                    interactive=True,
                )

        # VAE optimization settings  
        with gr.Row():
            self.vae_tiling = gr.Checkbox(
                label="VAE Tiling",
                info="Enable spatial tiling for VAE to reduce VRAM usage during encoding/decoding",
                value=self.config.get("vae_tiling", False),
            )
            
            self.vae_chunk_size = gr.Number(
                label="VAE Chunk Size",
                info="Chunk size for CausalConv3d in VAE. Higher = faster but more VRAM. 0 = auto/disabled",
                value=self.config.get("vae_chunk_size", 0),
                minimum=0,
                maximum=128,
                step=1,
                interactive=True,
            )
            
            self.vae_spatial_tile_sample_min_size = gr.Number(
                label="VAE Spatial Tile Min Size", 
                info="Minimum spatial tile size for VAE. Auto-enables vae_tiling if set. 0 = disabled, 256 = typical",
                value=self.config.get("vae_spatial_tile_sample_min_size", 0),
                minimum=0,
                maximum=1024,
                step=32,
                interactive=True,
            )

            self.fp8_vl = gr.Checkbox(
                label="Use FP8 for Text Encoder",
                info="FP8 quantization for Qwen2.5-VL saves ~8GB VRAM with minimal quality loss. Enable for GPUs with <16GB VRAM",
                value=self.config.get("fp8_vl", False),
            )

        # Qwen Image specific options
        gr.Markdown("""
        ### FP8 Quantization Options
        **Important:** FP8 is an on-the-fly optimization. You MUST provide standard bf16 models as input. 
        Pre-quantized FP8 models are NOT supported. Training outputs LoRA weights only - base models remain in bf16 format.
        
        **GPU Compatibility:** Both options work on ALL CUDA GPUs (RTX 2000/3000/4000). 
        For 16GB GPUs, enable BOTH checkboxes to reduce VRAM from 24GB → 12GB.
        """)
        
        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="Use FP8 for Base Model (DiT)",
                info="Converts bf16 model to FP8 on-the-fly, saving ~12GB VRAM (24GB→12GB). INPUT: Requires standard bf16 model. OUTPUT: Cannot save FP8 models. Always enable with fp8_scaled below",
                value=self.config.get("fp8_base", False),
            )
            
            self.fp8_scaled = gr.Checkbox(
                label="Use Scaled FP8 for DiT",
                info="REQUIRED with fp8_base for best quality. RTX 4000: native support (fast). RTX 2000/3000: automatic fallback (same quality, slightly slower)",
                value=self.config.get("fp8_scaled", False),
            )
            
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap to CPU",
                info="Swap DiT blocks to CPU to save VRAM. Qwen Image has 60 total blocks, max swap is 59. 16=24GB→16GB, 45=42GB→12GB. Requires 64GB+ RAM. Slows training significantly",
                value=self.config.get("blocks_to_swap", 0),
                minimum=0,
                maximum=59,
                step=1,
                interactive=True,
            )

        # Additional model settings
        with gr.Row():
            self.guidance_scale = gr.Number(
                label="Guidance Scale", 
                info="Classifier-free guidance scale. Default: 1.0. Higher values = stronger prompt adherence",
                value=self.config.get("guidance_scale", 1.0),
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                interactive=True,
            )
            
            self.img_in_txt_in_offloading = gr.Checkbox(
                label="Image-in-Text Input Offloading",
                info="Memory optimization for mixed image-text inputs. Enable for VRAM savings with complex inputs",
                value=self.config.get("img_in_txt_in_offloading", False),
            )

            self.edit = gr.Checkbox(
                label="Enable Qwen-Image-Edit Mode",
                info="Enable image editing training with control images. Requires control_image_path in dataset configuration and Qwen-Image-Edit DiT model",
                value=self.config.get("edit", False),
            )

        # Flow matching parameters
        with gr.Row():
            self.timestep_sampling = gr.Dropdown(
                label="Timestep Sampling Method",
                info="[RECOMMENDED] 'qwen_shift' = dynamic shift per resolution (best for Qwen). 'shift' = fixed shift. 'sigma' = musubi tuner default",
                choices=["shift", "qwen_shift", "sigma", "uniform", "sigmoid", "flux_shift", "logsnr", "qinglong_flux", "qinglong_qwen"],
                value=self.config.get("timestep_sampling", "qwen_shift"),
                interactive=True,
            )

            self.discrete_flow_shift = gr.Number(
                label="Discrete Flow Shift",
                info="[NOTE] Only used with 'shift' method. Qwen Image optimal: 2.2. 'qwen_shift' automatically calculates dynamic shift (0.5-0.9) based on image resolution",
                value=self.config.get("discrete_flow_shift", 2.2),
                step=0.1,
                interactive=True,
            )

            self.weighting_scheme = gr.Dropdown(
                label="Weighting Scheme",
                info="[RECOMMENDED] 'none' recommended for Qwen Image. Advanced: 'logit_normal' for different timestep emphasis, 'mode' for SD3-style weighting",
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

        # Advanced timestep parameters  
        with gr.Row():
            self.sigmoid_scale = gr.Number(
                label="Sigmoid Scale", 
                info="Scale factor for sigmoid timestep sampling. Only used with 'sigmoid' or some shift methods. Default: 1.0",
                value=self.config.get("sigmoid_scale", 1.0),
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                interactive=True,
            )
            
            self.min_timestep = gr.Number(
                label="Min Timestep",
                info="Minimum timestep for training (0-999). Leave empty for default (0). Constrains timestep sampling range",
                value=self.config.get("min_timestep", None),
                minimum=0,
                maximum=999,
                step=1,
                interactive=True,
            )
            
            self.max_timestep = gr.Number(
                label="Max Timestep", 
                info="Maximum timestep for training (1-1000). Leave empty for default (1000). Constrains timestep sampling range",
                value=self.config.get("max_timestep", None),
                minimum=1,
                maximum=1000,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.preserve_distribution_shape = gr.Checkbox(
                label="Preserve Distribution Shape",
                info="Use rejection sampling to preserve original timestep distribution when using min/max timestep constraints",
                value=self.config.get("preserve_distribution_shape", False),
            )

            self.num_timestep_buckets = gr.Number(
                label="Num Timestep Buckets",
                info="Number of buckets for uniform timestep sampling. 0=disabled (default), 4-10=bucketed sampling for better training stability",
                value=self.config.get("num_timestep_buckets", 0),
                minimum=0,
                maximum=100,
                step=1,
                interactive=True,
            )

            self.show_timesteps = gr.Dropdown(
                label="Show Timesteps",
                info="Visualization mode for timestep debugging. 'image' saves visual plots, 'console' prints to terminal. Leave empty for no visualization",
                choices=["image", "console", ""],
                allow_custom_value=True,
                value=self.config.get("show_timesteps", ""),
                interactive=True,
            )
    
    def setup_model_ui_events(self):
        """Setup event handlers for model configuration UI"""
        
        # Add file browse button handlers for model paths
        self.dit_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Safetensors files (*.safetensors)"),
            outputs=[self.dit]
        )
        
        self.vae_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Safetensors files (*.safetensors)"),
            outputs=[self.vae]
        )
        
        self.text_encoder_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".safetensors", extension_name="Safetensors files (*.safetensors)"),
            outputs=[self.text_encoder]
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
    # dataset configuration parameters
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
    dataset_control_directory,
    dataset_qwen_image_edit_no_resize_control,
    generated_toml_path,
    sdpa,
    flash_attn,
    sage_attn,
    xformers,
    flash3,
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
    text_encoder_dtype,
    vae,
    vae_dtype,
    vae_tiling,
    vae_chunk_size,
    vae_spatial_tile_sample_min_size,
    text_encoder,  # Qwen Image text encoder
    fp8_vl,       # Qwen Image specific
    fp8_base,
    fp8_scaled,   # Qwen Image specific
    blocks_to_swap,
    guidance_scale,
    img_in_txt_in_offloading,
    edit,
    timestep_sampling,
    discrete_flow_shift,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    sigmoid_scale,
    min_timestep,
    max_timestep,
    preserve_distribution_shape,
    num_timestep_buckets,
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
    # Define numeric fields that should never be lists
    # NOTE: Exclude fields that are legitimately lists like optimizer_args, lr_scheduler_args, network_args
    numeric_fields = [
        'learning_rate', 'max_grad_norm', 'guidance_scale', 'logit_mean', 'logit_std',
        'mode_scale', 'sigmoid_scale', 'lr_scheduler_power', 'lr_scheduler_timescale',
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',  # base_weights_multiplier is a Number in Qwen Image GUI
        'vae_chunk_size', 'vae_spatial_tile_sample_min_size', 'blocks_to_swap',
        'min_timestep', 'max_timestep', 'discrete_flow_shift', 'network_dropout',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'num_timestep_buckets', 'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers'
    ]
    
    # Create parameters list, ensuring numeric fields are not lists
    parameters = []
    local_vars = locals().copy()  # Make a copy to avoid modification during iteration
    for k, v in local_vars.items():
        if k not in ["action_type", "bool_value", "headless", "print_only", "numeric_fields", "parameters", "local_vars"]:
            # If it's a numeric field and the value is a list, take the first element
            if k in numeric_fields and isinstance(v, list):
                log.warning(f"Converting list to single value for numeric field '{k}': {v} -> {v[0] if v else None}")
                v = v[0] if v else None
            # Also check for any other unexpected lists in numeric-like fields
            elif isinstance(v, list) and k not in ['optimizer_args', 'lr_scheduler_args', 'network_args', 
                                                     'base_weights', 'extra_accelerate_launch_args',
                                                     'gpu_ids', 'additional_parameters']:
                # Only log if it's not the parameters list itself (which would be recursive)
                if 'parameters' not in str(v)[:100]:  # Check first 100 chars to avoid full recursion
                    log.warning(f"Unexpected list value for field '{k}': {v}")
            parameters.append((k, v))
    
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
        gr.Info("Training is starting... Please check the console for progress.")
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
        # Auto-append .toml extension if not present
        if file_path and not file_path.endswith('.toml'):
            file_path = file_path + '.toml'
            log.info(f"Auto-appending .toml extension: {file_path}")
        elif file_path == None or file_path == "":
            file_path = get_saveasfile_path(
                file_path,
                defaultextension=".toml",
                extension_name="TOML files (*.toml)",
            )

    log.debug(file_path)

    if file_path == None or file_path == "":
        gr.Info("Save cancelled")
        return original_file_path, gr.update(value="Save cancelled", visible=True)

    destination_directory = os.path.dirname(file_path)

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # Process parameters to handle list values properly
    processed_params = []
    # NOTE: Exclude fields that are legitimately lists like optimizer_args, lr_scheduler_args, network_args
    numeric_fields = [
        'learning_rate', 'max_grad_norm', 'guidance_scale', 'logit_mean', 'logit_std',
        'mode_scale', 'sigmoid_scale', 'lr_scheduler_power', 'lr_scheduler_timescale',
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',  # base_weights_multiplier is a Number in Qwen Image GUI
        'vae_chunk_size', 'vae_spatial_tile_sample_min_size', 'blocks_to_swap',
        'min_timestep', 'max_timestep', 'discrete_flow_shift', 'network_dropout',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'num_timestep_buckets', 'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers'
    ]
    
    for key, value in parameters:
        # If value is a list and it's not supposed to be (like from a Number component)
        # take the first element or convert to appropriate type
        if isinstance(value, list) and len(value) > 0 and key in numeric_fields:
            # These should be single numeric values
            value = value[0] if value else None
        processed_params.append((key, value))

    try:
        SaveConfigFile(
            parameters=processed_params,
            file_path=file_path,
            exclusion=[
                "file_path",
                "save_as",
                "save_as_bool",
            ],
        )
        
        # Show success message with timestamp
        config_name = os.path.basename(file_path)
        save_time = datetime.now().strftime("%I:%M:%S %p")  # Format: 01:32:23 PM
        success_msg = f"Configuration saved successfully to: {config_name} - Saved at {save_time}"
        log.info(success_msg)
        gr.Info(success_msg)
        
        return file_path, gr.update(value=success_msg, visible=True)
    except Exception as e:
        error_msg = f"Failed to save configuration: {str(e)}"
        log.error(error_msg)
        gr.Error(error_msg)
        return original_file_path, gr.update(value=error_msg, visible=True)


def open_qwen_image_configuration(ask_for_file, file_path, parameters):
    original_file_path = file_path
    status_msg = ""

    if ask_for_file:
        # Use the new function that allows both file selection and folder navigation
        file_path = get_file_path_or_save_as(
            file_path, default_extension=".toml", extension_name="TOML files"
        )

    if not file_path == "" and not file_path == None:
        # Check if it's a new file (doesn't exist yet) - that's OK if user typed a new name
        if not os.path.isfile(file_path) and ask_for_file:
            # If user selected a path but the file doesn't exist, it's a new config
            # We'll return the path so it can be used for saving later
            status_msg = f"New configuration file will be created at: {os.path.basename(file_path)}"
            log.info(status_msg)
            gr.Info(status_msg)
            # Return the new file path with empty/default values for configuration
            values = [file_path, gr.update(value=status_msg, visible=True)]
            for key, value in parameters:
                if not key in ["ask_for_file", "apply_preset", "file_path"]:
                    values.append(value)  # Keep current values
            return tuple(values)
        elif not os.path.isfile(file_path):
            error_msg = f"Config file {file_path} does not exist."
            log.error(error_msg)
            gr.Error(error_msg)
            # Return with error status
            values = [original_file_path, gr.update(value=error_msg, visible=True)]
            for key, value in parameters:
                if not key in ["ask_for_file", "apply_preset", "file_path"]:
                    values.append(value)
            return tuple(values)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                my_data = toml.load(f)
                config_name = os.path.basename(file_path)
                status_msg = f"Configuration loaded successfully from: {config_name}"
                log.info(status_msg)
                gr.Info(status_msg)
        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            log.error(error_msg)
            gr.Error(error_msg)
            # Return with error status
            values = [original_file_path, gr.update(value=error_msg, visible=True)]
            for key, value in parameters:
                if not key in ["ask_for_file", "apply_preset", "file_path"]:
                    values.append(value)
            return tuple(values)
    else:
        file_path = original_file_path
        my_data = {}
        if ask_for_file:
            status_msg = "Load cancelled"
            gr.Info(status_msg)

    # Define minimum value constraints to prevent validation errors
    # Based on actual Gradio component definitions
    minimum_constraints = {
        # Accelerate Launch components (shared with Musubi Tuner)
        "num_processes": 1,           # minimum=1
        "num_machines": 1,           # minimum=1
        "num_cpu_threads_per_process": 1,  # minimum=1 (Slider)
        "main_process_port": 0,      # minimum=0
        # Components with minimum=0
        "vae_chunk_size": 0,
        "vae_spatial_tile_sample_min_size": 0,
        "blocks_to_swap": 0,
        "min_timestep": 0,
        "max_data_loader_n_workers": 0,
        "seed": 0,
        "max_grad_norm": 0.0,
        "lr_warmup_steps": 0,
        # Components with minimum=0.1
        "guidance_scale": 0.1,
        "logit_std": 0.1,
        "mode_scale": 0.1,
        "sigmoid_scale": 0.1,
        "lr_scheduler_power": 0.1,
        "network_alpha": 0.1,
        # Components with minimum=1
        "max_timestep": 1,
        "max_train_epochs": 1,
        "gradient_accumulation_steps": 1,
        "sample_every_n_steps": 1,
        "sample_every_n_epochs": 1,
        "ddp_timeout": 1,
        "lr_scheduler_num_cycles": 1,
        "network_dim": 1,
        "save_every_n_steps": 1,
        "save_last_n_epochs": 1,
        "caching_latent_batch_size": 1,
        "caching_teo_batch_size": 1,
        # Components with minimum=100
        "max_train_steps": 100,
        # Components with minimum=1e-7
        "learning_rate": 1e-7,
        # Components with minimum=-3.0
        "logit_mean": -3.0
    }

    # Parameters that should be None when their value is 0 (optional parameters)
    optional_parameters = {
        "ddp_timeout", "sample_every_n_steps", "sample_every_n_epochs", 
        "save_every_n_steps", "save_last_n_epochs", "max_timestep", "min_timestep"
    }

    # NOTE: Exclude fields that are legitimately lists like optimizer_args, lr_scheduler_args, network_args
    numeric_fields = [
        'learning_rate', 'max_grad_norm', 'guidance_scale', 'logit_mean', 'logit_std',
        'mode_scale', 'sigmoid_scale', 'lr_scheduler_power', 'lr_scheduler_timescale',
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',  # base_weights_multiplier is a Number in Qwen Image GUI
        'vae_chunk_size', 'vae_spatial_tile_sample_min_size', 'blocks_to_swap',
        'min_timestep', 'max_timestep', 'discrete_flow_shift', 'network_dropout',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'num_timestep_buckets', 'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers'
    ]
    
    values = [file_path, gr.update(value=status_msg, visible=True)]
    for key, value in parameters:
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            toml_value = my_data.get(key)
            if toml_value is not None:
                # Handle list values that should be single values
                if isinstance(toml_value, list) and key in numeric_fields:
                    log.info(f"[CONFIG] Converting list to single value for numeric field '{key}': {toml_value} -> {toml_value[0] if toml_value else None}")
                    toml_value = toml_value[0] if toml_value else None
                elif isinstance(toml_value, list) and key not in ['optimizer_args', 'lr_scheduler_args', 'network_args', 
                                                                  'base_weights', 'base_weights_multiplier', 'extra_accelerate_launch_args',
                                                                  'gpu_ids', 'additional_parameters']:
                    log.warning(f"[CONFIG] Unexpected list value for field '{key}': {toml_value} (type: {type(toml_value)})")
                
                # Convert 0 to None for optional parameters to avoid minimum constraint violations
                if key in optional_parameters and toml_value == 0:
                    toml_value = None
                elif key in minimum_constraints and toml_value is not None:
                    # Apply minimum constraints if the parameter has one
                    min_val = minimum_constraints[key]
                    try:
                        if toml_value < min_val:
                            log.warning(f"Parameter '{key}' value {toml_value} is below minimum {min_val}, adjusting to minimum")
                            toml_value = min_val
                    except (TypeError, ValueError) as e:
                        log.warning(f"Could not compare {key} value {toml_value} with minimum {min_val}: {e}")
                
                # Final check before appending
                if isinstance(toml_value, list) and key in numeric_fields:
                    log.error(f"[CONFIG ERROR] Still have list for numeric field '{key}' after processing: {toml_value}")
                    toml_value = None  # Fallback to None to prevent error
                    
                values.append(toml_value)
            else:
                # Use original default value if not found in config
                # Check if the default value is a list and should be a single value
                if isinstance(value, list) and key in numeric_fields:
                    log.info(f"[DEFAULT] Converting list to single value for numeric field '{key}': {value} -> {value[0] if value else None}")
                    value = value[0] if value else None
                elif isinstance(value, list) and key not in ['optimizer_args', 'lr_scheduler_args', 'network_args', 
                                                             'base_weights', 'base_weights_multiplier', 'extra_accelerate_launch_args',
                                                             'gpu_ids', 'additional_parameters']:
                    log.warning(f"[DEFAULT] Unexpected list value for field '{key}': {value}")
                    
                values.append(value)

    # Final validation before returning
    result_values = []
    for i, v in enumerate(values):
        if isinstance(v, list):
            # Get parameter name for this index
            param_name = "unknown"
            if i > 0 and i <= len(parameters):
                param_name = parameters[i-1][0]  # -1 because values[0] is file_path
            
            # Only log verbose error if it's not the parameters list itself
            if param_name != 'parameters' and 'parameters' not in str(v)[:50]:
                log.debug(f"[VALIDATION] Processing list value at index {i} (param: {param_name})")
            # Try to fix it
            if param_name in numeric_fields:
                fixed_value = v[0] if v else None
                log.info(f"[VALIDATION FIX] Converted {param_name} to: {fixed_value}")
                result_values.append(fixed_value)
            elif param_name in ['optimizer_args', 'lr_scheduler_args', 'network_args', 
                               'base_weights', 'extra_accelerate_launch_args',
                               'gpu_ids', 'additional_parameters']:
                # These should remain as lists, but optimizer_args, lr_scheduler_args, and network_args
                # need to be converted to space-separated strings for the GUI textbox
                if param_name in ['optimizer_args', 'lr_scheduler_args', 'network_args']:
                    # Convert list to space-separated string for textbox display
                    result_values.append(" ".join(v) if isinstance(v, list) else v)
                else:
                    # Keep as list for other parameters
                    result_values.append(v)
            else:
                # Unknown list - try to convert if it looks numeric
                if v and len(v) == 1 and isinstance(v[0], (int, float, type(None))):
                    log.warning(f"[VALIDATION] Converting unexpected single-element list {param_name}: {v} -> {v[0]}")
                    result_values.append(v[0])
                else:
                    log.warning(f"[VALIDATION] Keeping list for param {param_name}")
                    result_values.append(v)
        else:
            result_values.append(v)

    return tuple(result_values)


def train_qwen_image_model(headless, print_only, parameters):
    import sys
    import json
    
    # Use Python directly instead of uv for better compatibility
    python_cmd = sys.executable
    
    # Find accelerate using shutil.which (like Kohya does)
    accelerate_path = shutil.which("accelerate")
    
    if accelerate_path:
        # Found accelerate in PATH
        log.debug(f"Found accelerate at: {accelerate_path}")
        run_cmd = [rf"{accelerate_path}", "launch"]
    else:
        # Fallback: try to find accelerate in the venv's Scripts/bin directory
        python_dir = os.path.dirname(python_cmd)
        if sys.platform == "win32":
            accelerate_fallback = os.path.join(python_dir, "accelerate.exe")
        else:
            accelerate_fallback = os.path.join(python_dir, "accelerate")
        
        if os.path.exists(accelerate_fallback) and os.access(accelerate_fallback, os.X_OK):
            log.debug(f"Found accelerate via fallback at: {accelerate_fallback}")
            run_cmd = [rf"{accelerate_fallback}", "launch"]
        else:
            # Last resort: run accelerate through Python using the commands.launch module
            log.warning("Accelerate binary not found, using Python module fallback")
            run_cmd = [python_cmd, "-m", "accelerate.commands.launch"]

    param_dict = dict(parameters)
    
    # Always use the Dataset Config File path for training
    effective_dataset_config = param_dict.get("dataset_config")
    dataset_mode = param_dict.get("dataset_config_mode", "Use TOML File")
    
    # Validate dataset config path
    if not effective_dataset_config or effective_dataset_config.strip() == "":
        if dataset_mode == "Generate from Folder Structure":
            raise ValueError(
                "[ERROR] Dataset config file path is empty!\n"
                "After generating a dataset configuration:\n"
                "1. The path should be auto-copied to 'Dataset Config File' field, OR\n"
                "2. Click 'Copy Generated TOML Path to Dataset Config' button to manually copy it\n"
                "The model always uses the path in 'Dataset Config File' for training."
            )
        else:
            raise ValueError(
                "[ERROR] Dataset config file path is empty!\n"
                "Please enter a path in the 'Dataset Config File' field.\n"
                "This is the path that will be used for training."
            )
    
    if not os.path.exists(effective_dataset_config):
            raise ValueError(
                f"[ERROR] Dataset config file does not exist: {effective_dataset_config}\n"
                "Please check the file path or create the dataset configuration file first."
            )
    
    # Validate the TOML file can be parsed
    try:
        with open(effective_dataset_config, 'r', encoding='utf-8') as f:
            dataset_config_content = toml.load(f)
        if not dataset_config_content.get("datasets"):
            raise ValueError(
                f"[ERROR] Invalid dataset config: No datasets defined in {effective_dataset_config}\n"
                "The TOML file must contain at least one [[datasets]] section."
            )
    except toml.TomlDecodeError as e:
        raise ValueError(
            f"[ERROR] Invalid TOML format in dataset config: {effective_dataset_config}\n"
            f"Error: {str(e)}"
        )
    except Exception as e:
        raise ValueError(
            f"[ERROR] Failed to read dataset config: {effective_dataset_config}\n"
            f"Error: {str(e)}"
        )
    
    # Update param_dict with effective config
    param_dict["dataset_config"] = effective_dataset_config
    # Validate VAE checkpoint
    vae_path = param_dict.get("vae")
    if not vae_path:
        raise ValueError(
            "[ERROR] VAE checkpoint path is required for training!\n"
            "Please download and specify the VAE model path.\n"
            "Expected file: pytorch_model.pt or similar from Qwen/Qwen-Image"
        )
    if not os.path.exists(vae_path):
        raise ValueError(
            f"[ERROR] VAE checkpoint file does not exist: {vae_path}\n"
            "Please check the file path or download the VAE model first."
        )
    
    # Validate DiT checkpoint
    dit_path = param_dict.get("dit")
    if not dit_path:
        raise ValueError(
            "[ERROR] DiT checkpoint path is required for training!\n"
            "Please download and specify the DiT model path.\n"
            "Expected file: mp_rank_00_model_states_fp8.safetensors or similar"
        )
    if not os.path.exists(dit_path):
        raise ValueError(
            f"[ERROR] DiT checkpoint file does not exist: {dit_path}\n"
            "Please check the file path or download the DiT model first."
        )
    if not param_dict.get("output_dir"):
        raise ValueError("[ERROR] Output directory is required. Please specify where to save your trained LoRA model.")
    if not param_dict.get("output_name"):
        raise ValueError("[ERROR] Output name is required. Please specify a name for your trained LoRA model.")
    
    # Only do file operations and caching if not in print_only mode
    if not print_only:
        # Create output directory if it doesn't exist
        output_dir = param_dict.get("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]  # Remove last 3 digits of microseconds to get milliseconds
        
        # Save dataset TOML to output directory (NO JSON saving anymore)
        dataset_toml_dest = os.path.join(output_dir, f"{timestamp}.toml")
        if os.path.exists(effective_dataset_config):
            shutil.copy2(effective_dataset_config, dataset_toml_dest)
            log.info(f"Dataset config saved to: {dataset_toml_dest}")
            print(f"\n[INFO] Dataset configuration backed up to output directory:")
            print(f"       {dataset_toml_dest}")
            print(f"\n[INFO] Configuration file saved with timestamp: {timestamp}")
            print(f"       This file will be preserved with your trained model for reproducibility.\n")
    
        # Cache latents using Qwen Image specific script
        run_cache_latent_cmd = [python_cmd, "./musubi-tuner/src/musubi_tuner/qwen_image_cache_latents.py",
                                "--dataset_config", str(param_dict.get("dataset_config")),
                                "--vae", str(param_dict.get("vae"))
        ]
        
        # Note: vae_dtype is not supported in Qwen Image
            
        # Determine the device for caching latents
        caching_device = param_dict.get("caching_latent_device", "cuda")
        if caching_device == "cuda" and param_dict.get("gpu_ids"):
            # If gpu_ids is specified in accelerate config, use the first GPU ID
            gpu_ids = str(param_dict.get("gpu_ids")).split(",")
            caching_device = f"cuda:{gpu_ids[0].strip()}"
            log.info(f"Using GPU ID from accelerate config for latent caching: {caching_device}")
        
        if caching_device:
            run_cache_latent_cmd.append("--device")
            run_cache_latent_cmd.append(str(caching_device))
        
        if param_dict.get("caching_latent_batch_size") is not None:
            run_cache_latent_cmd.append("--batch_size")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_batch_size")))
        
        if param_dict.get("caching_latent_num_workers") is not None:
            run_cache_latent_cmd.append("--num_workers")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_num_workers")))
            
        if param_dict.get("caching_latent_skip_existing"):
            run_cache_latent_cmd.append("--skip_existing")
            
        if param_dict.get("caching_latent_keep_cache"):
            run_cache_latent_cmd.append("--keep_cache")
        
        # VAE optimization parameters for latent caching
        if param_dict.get("vae_tiling"):
            run_cache_latent_cmd.append("--vae_tiling")
        
        if param_dict.get("vae_chunk_size") is not None and param_dict.get("vae_chunk_size") > 0:
            run_cache_latent_cmd.append("--vae_chunk_size")
            run_cache_latent_cmd.append(str(param_dict.get("vae_chunk_size")))
        
        if param_dict.get("vae_spatial_tile_sample_min_size") is not None and param_dict.get("vae_spatial_tile_sample_min_size") > 0:
            run_cache_latent_cmd.append("--vae_spatial_tile_sample_min_size")
            run_cache_latent_cmd.append(str(param_dict.get("vae_spatial_tile_sample_min_size")))
        
        # VAE dtype parameter - SKIP for Qwen Image as it's not supported
        # if param_dict.get("vae_dtype") is not None and param_dict.get("vae_dtype") != "":
        #     run_cache_latent_cmd.append("--vae_dtype")
        #     run_cache_latent_cmd.append(str(param_dict.get("vae_dtype")))
        
        # Debug parameters for latent caching
        if param_dict.get("caching_latent_debug_mode") is not None and param_dict.get("caching_latent_debug_mode") != "":
            run_cache_latent_cmd.append("--debug_mode")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_debug_mode")))
        
        if param_dict.get("caching_latent_console_width") is not None:
            run_cache_latent_cmd.append("--console_width")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_console_width")))
        
        if param_dict.get("caching_latent_console_back") is not None and param_dict.get("caching_latent_console_back") != "":
            run_cache_latent_cmd.append("--console_back")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_console_back")))
        
        if param_dict.get("caching_latent_console_num_images") is not None:
            run_cache_latent_cmd.append("--console_num_images")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_console_num_images")))

        log.info(f"Executing command: {run_cache_latent_cmd}")
        log.info("Caching latents...")
        try:
            # Run without capture_output to show progress in real-time
            gr.Info("Starting latent caching... This may take a while.")
            result = subprocess.run(run_cache_latent_cmd, env=setup_environment(), check=True)
            log.debug("Latent caching completed.")
            gr.Info("Latent caching completed successfully!")
        except subprocess.CalledProcessError as e:
            log.error(f"Latent caching failed with return code {e.returncode}")
            gr.Warning(f"Latent caching failed with return code {e.returncode}")
            raise RuntimeError(f"Latent caching failed with return code {e.returncode}")
        except FileNotFoundError as e:
            log.error(f"Command not found: {e}")
            log.error("Please ensure Python is installed and accessible in your PATH")
            raise RuntimeError(f"Python executable not found: {python_cmd}")
    
        # Cache text encoder outputs using Qwen Image specific script
        run_cache_teo_cmd = [python_cmd, "./musubi-tuner/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
                                "--dataset_config", str(param_dict.get("dataset_config"))
        ]
    
        # Validate text encoder for caching
        text_encoder_path = param_dict.get("text_encoder")
        if text_encoder_path and text_encoder_path != "":
            if not os.path.exists(text_encoder_path):
                log.warning(f"Text encoder file does not exist: {text_encoder_path}")
                log.warning("Text encoder caching will use text_encoder from caching_teo_text_encoder if specified")
            else:
                run_cache_teo_cmd.append("--text_encoder")
                run_cache_teo_cmd.append(str(text_encoder_path))
        
        # Check for caching-specific text encoder
        caching_text_encoder_path = param_dict.get("caching_teo_text_encoder")
        if caching_text_encoder_path and caching_text_encoder_path != "":
            if not os.path.exists(caching_text_encoder_path):
                raise ValueError(
                    f"[ERROR] Text encoder file for caching does not exist: {caching_text_encoder_path}\n"
                    "Please check the file path or download the Qwen2.5-VL model first."
                )
            # Override with caching-specific text encoder
            if "--text_encoder" in run_cache_teo_cmd:
                idx = run_cache_teo_cmd.index("--text_encoder")
                run_cache_teo_cmd[idx + 1] = str(caching_text_encoder_path)
            else:
                run_cache_teo_cmd.append("--text_encoder")
                run_cache_teo_cmd.append(str(caching_text_encoder_path))   
                                
        if param_dict.get("caching_teo_fp8_vl"):
            run_cache_teo_cmd.append("--fp8_vl")
        
        # Determine the device for caching text encoder outputs
        teo_caching_device = param_dict.get("caching_teo_device", "cuda")
        if teo_caching_device == "cuda" and param_dict.get("gpu_ids"):
            # If gpu_ids is specified in accelerate config, use the first GPU ID
            gpu_ids = str(param_dict.get("gpu_ids")).split(",")
            teo_caching_device = f"cuda:{gpu_ids[0].strip()}"
            log.info(f"Using GPU ID from accelerate config for text encoder caching: {teo_caching_device}")
        
        if teo_caching_device:
            run_cache_teo_cmd.append("--device")
            run_cache_teo_cmd.append(str(teo_caching_device))
        
        if param_dict.get("caching_teo_batch_size") is not None:
            run_cache_teo_cmd.append("--batch_size")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_batch_size")))
            
        if param_dict.get("caching_teo_skip_existing"):
            run_cache_teo_cmd.append("--skip_existing")
            
        if param_dict.get("caching_teo_keep_cache"):
            run_cache_teo_cmd.append("--keep_cache")
            
        if param_dict.get("caching_teo_num_workers") is not None:
            run_cache_teo_cmd.append("--num_workers")
            run_cache_teo_cmd.append(str(param_dict.get("caching_teo_num_workers")))
            
        if param_dict.get("edit"):
            run_cache_teo_cmd.append("--edit")

        # Store the text encoder caching command to be run as part of training
        teo_cache_cmd = run_cache_teo_cmd
        teo_cache_env = setup_environment()
    else:
        teo_cache_cmd = None
        teo_cache_env = None

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
    run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py")

    if print_only:
        print_command_and_toml(run_cmd, "")
    else:
        # Save config file for model
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(param_dict.get('output_dir'), f"{param_dict.get('output_name')}_{formatted_datetime}.toml")

        log.info(f"Saving training config to {file_path}...")

        # Validate sample generation settings
        # If sample_prompts is empty or missing, disable sample generation to avoid errors
        sample_prompts_provided = False
        for key, value in parameters:
            if key == 'sample_prompts' and value and value.strip():
                sample_prompts_provided = True
                break
        
        if not sample_prompts_provided:
            # Disable sample generation if no prompt file is provided
            modified_params = []
            for key, value in parameters:
                if key in ['sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first']:
                    if key == 'sample_every_n_epochs' and value and value != 0:
                        log.warning(f"Disabling {key}={value} because no sample_prompts file was provided")
                        modified_params.append((key, 0))
                    elif key == 'sample_every_n_steps' and value and value != 0:
                        log.warning(f"Disabling {key}={value} because no sample_prompts file was provided")
                        modified_params.append((key, 0))
                    elif key == 'sample_at_first' and value:
                        log.warning(f"Disabling {key}={value} because no sample_prompts file was provided")
                        modified_params.append((key, False))
                    else:
                        modified_params.append((key, value))
                else:
                    modified_params.append((key, value))
            parameters = modified_params

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

        # Create a wrapper script that runs both text encoder caching and training
        if teo_cache_cmd:
            # Create a combined command that runs caching first, then training
            import tempfile
            import platform
            
            # Create a temporary script to run both commands
            if platform.system() == "Windows":
                script_ext = ".bat"
                script_content = f"""@echo off
echo Starting text encoder output caching...
{' '.join(teo_cache_cmd)}
if %errorlevel% neq 0 (
    echo Text encoder caching failed with error code %errorlevel%
    exit /b %errorlevel%
)
echo Text encoder caching completed successfully!
echo Starting training...
{' '.join(run_cmd)}
"""
            else:
                script_ext = ".sh"
                script_content = f"""#!/bin/bash
echo "Starting text encoder output caching..."
{' '.join(teo_cache_cmd)}
if [ $? -ne 0 ]; then
    echo "Text encoder caching failed with error code $?"
    exit $?
fi
echo "Text encoder caching completed successfully!"
echo "Starting training..."
{' '.join(run_cmd)}
"""
            
            # Write the script to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=script_ext, delete=False) as f:
                temp_script = f.name
                f.write(script_content)
            
            # Make script executable on Unix-like systems
            if platform.system() != "Windows":
                import stat
                os.chmod(temp_script, os.stat(temp_script).st_mode | stat.S_IEXEC)
            
            # Execute the combined script
            if platform.system() == "Windows":
                final_cmd = [temp_script]
            else:
                final_cmd = ["bash", temp_script]
            
            log.info("Starting combined text encoder caching and training process...")
            gr.Info("Starting text encoder caching followed by training...")
            executor.execute_command(run_cmd=final_cmd, env=env, shell=True if platform.system() == "Windows" else False)
        else:
            # No text encoder caching needed, just run training
            executor.execute_command(run_cmd=run_cmd, env=env)

        train_state_value = time.time()

        # Return immediately to show stop button
        return (
            gr.Button(visible=False or headless),  # Hide start button
            gr.Row(visible=True),  # Show stop row
            gr.Button(interactive=True),  # Enable stop button by default
            gr.Textbox(value="Training in progress..."),  # Update status
            gr.Textbox(value=train_state_value),  # Trigger state change
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
                info="[RECOMMENDED] PyTorch's Scaled Dot Product Attention - fastest and most memory efficient for Qwen Image. PRIORITY 1: If multiple selected, SDPA takes precedence",
                value=self.config.get("sdpa", True),
            )

            self.flash_attn = gr.Checkbox(
                label="Use FlashAttention for CrossAttention",
                info="Memory-efficient attention implementation. Requires FlashAttention library. Enable split_attn if using this. PRIORITY 2: Used only if SDPA is disabled",
                value=self.config.get("flash_attn", False),
            )

            self.sage_attn = gr.Checkbox(
                label="Use SageAttention for CrossAttention",
                info="Alternative attention implementation. Requires SageAttention library. Enable split_attn if using this. PRIORITY 3: Used only if SDPA & FlashAttn are disabled",
                value=self.config.get("sage_attn", False),
            )

            self.xformers = gr.Checkbox(
                label="Use xformers for CrossAttention",
                info="Memory-efficient attention from xformers library. Enable split_attn if using this. PRIORITY 4: Lowest priority, used only if all others are disabled",
                value=self.config.get("xformers", False),
            )

        with gr.Row():
            self.flash3 = gr.Checkbox(
                label="Use FlashAttention 3",
                info="[EXPERIMENTAL] FlashAttention 3 support. Not confirmed to work with Qwen Image yet. Requires FlashAttention 3 library",
                value=self.config.get("flash3", False),
            )

            self.split_attn = gr.Checkbox(
                label="Split Attention",
                info="[REQUIRED] if using FlashAttention/SageAttention/xformers/flash3. Splits attention computation to reduce memory usage",
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
                info="[RECOMMENDED] 16 epochs for Qwen Image. Overrides max_train_steps. 1 epoch = full pass through dataset",
                value=self.config.get("max_train_epochs", 16),
                minimum=1,
                maximum=9999,
                step=1,
                interactive=True,
            )

            self.max_data_loader_n_workers = gr.Number(
                label="Max DataLoader Workers",
                info="[RECOMMENDED] 2 recommended for Qwen Image stability. Higher values = faster data loading but more RAM usage and potential instability",
                value=self.config.get("max_data_loader_n_workers", 2),
                minimum=0,
                maximum=16,
                step=1,
                interactive=True,
            )

            self.persistent_data_loader_workers = gr.Checkbox(
                label="Persistent DataLoader Workers",
                info="[ENABLED] Keeps data loading processes alive between epochs. Faster epoch transitions but uses more RAM",
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
                info="[ENABLED] Trades computation for memory. Essential for Qwen Image training. Saves ~50% VRAM but increases training time ~20%",
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
            with gr.Column(scale=4):
                self.logging_dir = gr.Textbox(
                    label="Logging Directory",
                    info="Directory for training logs and TensorBoard data. Leave empty to disable logging. Example: ./logs",
                    placeholder="e.g., ./logs or /path/to/logs",
                    value=self.config.get("logging_dir", ""),
                )
            self.logging_dir_button = gr.Button(
                "📂",
                size="sm",
                elem_id="logging_dir_button"
            )

            with gr.Column(scale=4):
                self.log_with = gr.Dropdown(
                    label="Logging Tool",
                    info="TensorBoard = local logs, WandB = cloud tracking, 'all' = both, (none) = no logging. Requires logging_dir for TensorBoard",
                    choices=[("(none)", ""), ("tensorboard", "tensorboard"), ("wandb", "wandb"), ("all", "all")],
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
                info="Generate test images every N training steps. 0 = disable, 100-500 recommended. Requires sample_prompts file",
                value=self.config.get("sample_every_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs",
                info="Generate test images every N epochs. 0 = disable, 1-4 recommended. Overrides sample_every_n_steps",
                value=self.config.get("sample_every_n_epochs", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.sample_at_first = gr.Checkbox(
                label="Sample at First",
                info="Generate test images before training starts. Useful to verify prompts and base model quality",
                value=self.config.get("sample_at_first", False),
            )

            with gr.Column(scale=4):
                self.sample_prompts = gr.Textbox(
                    label="Sample Prompts File",
                    info="Path to text file with prompts (one per line). Required for sample generation. Example: 'A cat\nA dog\nA house'",
                    placeholder="e.g., /path/to/prompts.txt",
                    value=self.config.get("sample_prompts", ""),
                )
            self.sample_prompts_button = gr.Button(
                "📂",
                size="sm",
                elem_id="sample_prompts_button"
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
                info="Distributed training timeout in minutes. 0 = use default (30min). Increase if training crashes on multi-GPU setups",
                value=self.config.get("ddp_timeout", 0),
                minimum=0,
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
                choices=["image", "console", ""],
                allow_custom_value=True,
                value=self.config.get("show_timesteps", ""),
                interactive=True,
            )
        
        # Add click handler for logging directory folder button
        self.logging_dir_button.click(
            fn=lambda: get_folder_path(),
            outputs=[self.logging_dir]
        )
        
        # Add click handler for sample prompts file button
        self.sample_prompts_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".txt", extension_name="Text files"),
            outputs=[self.sample_prompts]
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
                info="[RECOMMENDED] adamw8bit for Qwen Image (memory efficient, confirmed in official examples). AdamW = standard, AdaFactor = adaptive LR",
                choices=[
                    "adamw8bit", 
                    "AdamW", 
                    "AdaFactor", 
                    "bitsandbytes.optim.AdEMAMix8bit", 
                    "bitsandbytes.optim.PagedAdEMAMix8bit",
                    "torch.optim.Adam",
                    "torch.optim.SGD"
                ],
                allow_custom_value=True,
                value=self.config.get("optimizer_type", "adamw8bit"),
            )

            self.learning_rate = gr.Number(
                label="Learning Rate",
                info="[RECOMMENDED] 5e-5 (0.00005) for Qwen Image. Too high = instability, too low = slow learning. Typical range: 1e-6 to 1e-3",
                value=self.config.get("learning_rate", 5e-5),
                minimum=1e-7,
                maximum=1e-2,
                step=1e-6,
                interactive=True,
            )

        with gr.Row():
            self.optimizer_args = gr.Textbox(
                label="Optimizer Arguments",
                info="Extra optimizer parameters as key=value pairs. Space separated. e.g. scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01",
                placeholder='e.g. "scale_parameter=False relative_step=False warmup_init=False weight_decay=0.01"',
                value=" ".join(self.config.get("optimizer_args", []) or []) if isinstance(self.config.get("optimizer_args", []), list) else self.config.get("optimizer_args", ""),
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
                info="[RECOMMENDED] 'constant' for most cases. 'cosine' = gradual decrease, 'linear' = linear decrease, 'constant_with_warmup' = warm start",
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
                info="Extra scheduler parameters as key=value pairs. Space separated. e.g. T_max=100 eta_min=1e-7 last_epoch=-1",
                placeholder='e.g. "T_max=100 eta_min=1e-7 last_epoch=-1"',
                value=" ".join(self.config.get("lr_scheduler_args", []) or []) if isinstance(self.config.get("lr_scheduler_args", []), list) else self.config.get("lr_scheduler_args", ""),
            )


class QwenImageNetworkSettings:
    """Qwen Image specific LoRA network settings with optimal defaults"""
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
                label="Network Weights (LoRA Weights)",
                info="Path to existing LoRA weights to continue training from. Leave empty to start from scratch",
                placeholder="e.g., /path/to/existing_lora.safetensors",
                value=self.config.get("network_weights", ""),
            )

        with gr.Row():
            self.network_module = gr.Textbox(
                label="Network Module (LoRA Type)",
                info="[AUTO-SET] LoRA implementation for Qwen Image. 'networks.lora_qwen_image' is automatically selected. Do not change",
                placeholder="networks.lora_qwen_image",
                value=self.config.get("network_module", "networks.lora_qwen_image"),
                interactive=False,  # Will be auto-selected
            )

            self.network_dim = gr.Number(
                label="Network Dimension (LoRA Rank)",
                info="[RECOMMENDED] LoRA rank/dimension. 16 for Qwen Image. Higher = more capacity but larger files. Range: 8-128",
                value=self.config.get("network_dim", 16),
                minimum=1,
                maximum=512,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.network_alpha = gr.Number(
                label="Network Alpha (LoRA Alpha)",
                info="[RECOMMENDED] LoRA scaling factor. 1.0 for Qwen Image. Higher = stronger LoRA effect. Formula: alpha/rank = final scaling",
                value=self.config.get("network_alpha", 1.0),
                minimum=0.1,
                maximum=512.0,
                step=0.1,
                interactive=True,
            )

            self.network_dropout = gr.Number(
                label="Network Dropout (LoRA Dropout)",
                info="Dropout rate for LoRA regularization. 0.0 = no dropout, 0.1 = 10% dropout. Helps prevent overfitting",
                value=self.config.get("network_dropout", 0.0),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                interactive=True,
            )

        with gr.Row():
            self.network_args = gr.Textbox(
                label="Network Arguments (LoRA Args)",
                info="Advanced LoRA parameters as key=value pairs. Space separated. e.g. conv_dim=4 conv_alpha=1 algo=locon",
                placeholder='e.g. "conv_dim=4 conv_alpha=1 algo=locon"',
                value=" ".join(self.config.get("network_args", []) or []) if isinstance(self.config.get("network_args", []), list) else self.config.get("network_args", ""),
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
            with gr.Column(scale=4):
                self.output_dir = gr.Textbox(
                    label="Output Directory",
                    info="REQUIRED: Directory where trained LoRA model will be saved. Must exist or be creatable",
                    placeholder="e.g., ./models/trained or /path/to/output",
                    value=self.config.get("output_dir", ""),
                )
            self.output_dir_button = gr.Button(
                "📂",
                size="sm",
                elem_id="output_dir_button"
            )

            with gr.Column(scale=4):
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
                info="[RECOMMENDED] 1 recommended. Save checkpoint every N epochs for backup and progress tracking. 0 = save only at end",
                value=self.config.get("save_every_n_epochs", 1),
                minimum=0,
                maximum=50,
                step=1,
                interactive=True,
            )

            self.save_every_n_steps = gr.Number(
                label="Save Every N Steps",
                info="Save checkpoint every N training steps. 0 = disable. Overrides save_every_n_epochs. Useful for long training",
                value=self.config.get("save_every_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_epochs = gr.Number(
                label="Save Last N Epochs",
                info="Keep only last N epoch checkpoints (removes older ones). 0 = keep all, 3 = keep only last 3 checkpoints",
                value=self.config.get("save_last_n_epochs", 0),
                minimum=0,
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
        
        # Add click handler for output directory folder button
        self.output_dir_button.click(
            fn=lambda: get_folder_path(),
            outputs=[self.output_dir]
        )


class QwenImageLatentCaching:
    """Qwen Image specific latent caching - removes VAE options not supported"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        # Add informative text about latent caching
        gr.Markdown("""
        ### 🖼️ Latent Caching Info
        
        **IMPORTANT:** Both Latent AND Text Encoder caching are REQUIRED for training!
        
        **What is cached:** Image latent representations from VAE encoder  
        **Where cached:** In `cache_dir` folder inside each dataset directory (each dataset has its own cache)  
        **When to re-cache (uncheck Skip Existing):**
        - Changed training images
        - Changed VAE model
        - Changed resolution
        - Cache files are corrupted
        
        **Cache files:** `*_qi.safetensors` files in each dataset's cache_dir
        
        ⚠️ **Note:** Each dataset MUST have its own cache directory. Skip Existing checks each dataset's cache separately.
        """)
        
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
                choices=["image", "console", "video", ""],
                allow_custom_value=True,
                value=self.config.get("caching_latent_debug_mode", ""),
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
        # Add informative text about text encoder caching
        gr.Markdown("""
        ### 📝 Text Encoder Caching Info
        
        **What is cached:** Caption text embeddings from Qwen2.5-VL model  
        **Where cached:** In `cache_dir` folder inside each dataset directory (each dataset has its own cache)  
        **When to re-cache (uncheck Skip Existing):**
        - Changed captions or added new captions
        - Changed text encoder model
        - Cache files are corrupted
        - Changed training resolution
        
        **Cache files:** `*_qi_te.safetensors` files in each dataset's cache_dir
        
        ⚠️ **Note:** If caching re-runs even with "Skip Existing" checked:
        - This may happen on first run (no cache exists yet)
        - Check each dataset folder contains its own cache_dir with `*_qi_te.safetensors` files
        - Each dataset needs its own unique cache directory (musubi-tuner requirement)
        """)
        
        with gr.Row():
            self.caching_teo_text_encoder = gr.Textbox(
                label="Text Encoder (Qwen2.5-VL) Path",
                info="Path to Qwen2.5-VL for text encoder caching. Leave empty to use the main Text Encoder path from Model Settings above",
                placeholder="e.g., /path/to/qwen_2.5_vl_7b.safetensors (leave empty to use main path)",
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
                info="Use FP8 quantization during caching to save VRAM. Should match main FP8 VL setting. Enable for <16GB VRAM",
                value=self.config.get("caching_teo_fp8_vl", False),
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
                info="✅ KEEP CHECKED: Skip if cache already exists (fast). UNCHECK ONLY to re-cache when: captions changed, text encoder changed, or cache corrupted",
                value=self.config.get("caching_teo_skip_existing", True),
            )

            self.caching_teo_keep_cache = gr.Checkbox(
                label="Keep Cache Files",
                info="✅ KEEP CHECKED: Keep cache for faster re-training. Cache stays in cache_dir folder even if images removed from dataset",
                value=self.config.get("caching_teo_keep_cache", True),
            )


# Note: Default loading is now handled by TabConfigManager in the main GUI
# These functions are kept for reference but are no longer used


def qwen_image_lora_tab(
    headless=False,
    config: GUIConfig = {},
):
    # Add custom CSS for larger button text
    gr.HTML("""
    <style>
        #toggle-all-btn {
            font-size: 1.2rem !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
        }
        #toggle-all-btn button {
            font-size: 1.2rem !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
        }
    </style>
    """)
    
    # Configuration is now managed by TabConfigManager
    dummy_true = gr.Checkbox(value=True, visible=False)
    dummy_false = gr.Checkbox(value=False, visible=False)
    dummy_headless = gr.Checkbox(value=headless, visible=False)

    # Setup Configuration Files Gradio
    with gr.Accordion("Configuration file Settings", open=True):
        # Show configuration status
        # Check if this is a default config by looking for Qwen-specific values
        is_using_defaults = (hasattr(config, 'config') and 
                            config.config.get("discrete_flow_shift") == 3.0 and 
                            config.config.get("timestep_sampling") == "shift" and
                            config.config.get("fp8_vl") == True)
        
        if is_using_defaults:
            config_status = """
            [OK] **Qwen Image Optimal Defaults Active**
            
            **Key Optimizations Applied:**
            - Discrete Flow Shift: 3.0 (optimal for Qwen Image)
            - Optimizer: adamw8bit (memory efficient, recommended)
            - Learning Rate: 5e-5 (recommended for Qwen Image, per latest docs)
            - Mixed Precision: bf16 (strongly recommended)
            - SDPA Attention: Enabled (fastest for Qwen Image)
            - Gradient Checkpointing: Enabled (memory savings)
            """
        elif hasattr(config, 'config') and config.config:
            config_status = "[INFO] **Custom configuration loaded** - You may want to verify optimal settings are applied"
        else:
            config_status = "[WARNING] **No configuration** - Default values will be used"
        
        gr.Markdown(config_status)
        configuration = ConfigurationFile(headless=headless, config=config)

    # Add unified toggle button to control all accordions
    with gr.Row():
        with gr.Column(scale=1):
            toggle_all_btn = gr.Button(
                value="Open All Panels", 
                variant="secondary", 
                size="lg",
                elem_id="toggle-all-btn"
            )
            # Hidden state to track if panels are open or closed
            panels_state = gr.State(value="closed")  # Default state is closed

    # Create accordion references
    accordions = []
    
    accelerate_accordion = gr.Accordion("Accelerate launch Settings", open=False, elem_classes="flux1_background")
    accordions.append(accelerate_accordion)
    with accelerate_accordion, gr.Column():
        accelerate_launch = AccelerateLaunch(config=config)
        # Note: bf16 mixed precision is STRONGLY recommended for Qwen Image
        
    # Save Load Settings - moved before Model Settings for better workflow
    save_load_accordion = gr.Accordion("Save Load Settings", open=False, elem_classes="samples_background")
    accordions.append(save_load_accordion)
    with save_load_accordion:
        saveLoadSettings = QwenImageSaveLoadSettings(headless=headless, config=config)
    
    # Qwen Image Training Dataset accordion - placed before Model Settings for better workflow
    qwen_dataset_accordion = gr.Accordion("Qwen Image Training Dataset", open=False, elem_classes="samples_background")
    accordions.append(qwen_dataset_accordion)
    with qwen_dataset_accordion:
        qwen_dataset = QwenImageDataset(headless=headless, config=config)
        qwen_dataset.setup_dataset_ui_events(saveLoadSettings)  # Pass saveLoadSettings for output_dir access
        
    # Qwen Image Model Settings accordion - contains model paths and settings
    qwen_model_accordion = gr.Accordion("Qwen Image Model Settings", open=False, elem_classes="preset_background")
    accordions.append(qwen_model_accordion)
    with qwen_model_accordion:
        qwen_model = QwenImageModel(headless=headless, config=config)
        qwen_model.setup_model_ui_events()  # Setup model UI events
        
    caching_accordion = gr.Accordion("Caching", open=False, elem_classes="samples_background")
    accordions.append(caching_accordion)
    with caching_accordion:
        with gr.Tab("Latent caching"):
            qwenLatentCaching = QwenImageLatentCaching(headless=headless, config=config)
                
        with gr.Tab("Text encoder caching"):
            qwenTeoCaching = QwenImageTextEncoderOutputsCaching(headless=headless, config=config)
        
    optimizer_accordion = gr.Accordion("Optimizer and Scheduler Settings", open=False, elem_classes="flux1_rank_layers_background")
    accordions.append(optimizer_accordion)
    with optimizer_accordion:
        OptimizerAndSchedulerSettings = QwenImageOptimizerSettings(headless=headless, config=config)
        
    network_accordion = gr.Accordion("LoRA Settings", open=False, elem_classes="flux1_background")
    accordions.append(network_accordion)
    with network_accordion:
        network = QwenImageNetworkSettings(headless=headless, config=config)
        
    training_accordion = gr.Accordion("Training Settings", open=False, elem_classes="preset_background")
    accordions.append(training_accordion)
    with training_accordion:
        trainingSettings = QwenImageTrainingSettings(headless=headless, config=config)

    advanced_accordion = gr.Accordion("Advanced Settings", open=False, elem_classes="samples_background")
    accordions.append(advanced_accordion)
    with advanced_accordion:
        gr.Markdown("**Additional Parameters**: Add custom training parameters as key=value pairs (e.g., `custom_param=value`). These will be appended to the training command.")
        advanced_training = AdvancedTraining(
            headless=headless, training_type="lora", config=config
        )

    metadata_accordion = gr.Accordion("Metadata Settings", open=False, elem_classes="flux1_rank_layers_background")
    accordions.append(metadata_accordion)
    with metadata_accordion, gr.Group():
        metadata = MetaData(config=config)

    global huggingface
    huggingface_accordion = gr.Accordion("HuggingFace Settings", open=False, elem_classes="huggingface_background")
    accordions.append(huggingface_accordion)
    with huggingface_accordion:
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
        
        # Dataset Settings - new parameters
        qwen_dataset.dataset_config_mode,
        qwen_dataset.dataset_config,
        qwen_dataset.parent_folder_path,
        qwen_dataset.dataset_resolution_width,
        qwen_dataset.dataset_resolution_height,
        qwen_dataset.dataset_caption_extension,
        qwen_dataset.create_missing_captions,
        qwen_dataset.caption_strategy,
        qwen_dataset.dataset_batch_size,
        qwen_dataset.dataset_enable_bucket,
        qwen_dataset.dataset_bucket_no_upscale,
        qwen_dataset.dataset_cache_directory,
        qwen_dataset.dataset_control_directory,
        qwen_dataset.dataset_qwen_image_edit_no_resize_control,
        qwen_dataset.generated_toml_path,
        
        # trainingSettings
        trainingSettings.sdpa,
        trainingSettings.flash_attn,
        trainingSettings.sage_attn,
        trainingSettings.xformers,
        trainingSettings.flash3,
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
        qwen_model.text_encoder_dtype,
        qwen_model.vae,
        qwen_model.vae_dtype,
        qwen_model.vae_tiling,
        qwen_model.vae_chunk_size,
        qwen_model.vae_spatial_tile_sample_min_size,
        qwen_model.text_encoder,
        qwen_model.fp8_vl,
        qwen_model.fp8_base,
        qwen_model.fp8_scaled,
        qwen_model.blocks_to_swap,
        qwen_model.guidance_scale,
        qwen_model.img_in_txt_in_offloading,
        qwen_model.edit,
        qwen_model.timestep_sampling,
        qwen_model.discrete_flow_shift,
        qwen_model.weighting_scheme,
        qwen_model.logit_mean,
        qwen_model.logit_std,
        qwen_model.mode_scale,
        qwen_model.sigmoid_scale,
        qwen_model.min_timestep,
        qwen_model.max_timestep,
        qwen_model.preserve_distribution_shape,
        qwen_model.num_timestep_buckets,
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

    # Add handler for unified toggle button
    def toggle_all_panels(current_state):
        if current_state == "closed":
            # Open all panels and update button text
            new_state = "open"
            new_button_text = "Hide All Panels"
            accordion_states = [gr.Accordion(open=True) for _ in accordions]
        else:
            # Close all panels and update button text
            new_state = "closed"
            new_button_text = "Open All Panels"
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
        
        return [new_state, gr.Button(value=new_button_text)] + accordion_states
    
    toggle_all_btn.click(
        toggle_all_panels,
        inputs=[panels_state],
        outputs=[panels_state, toggle_all_btn] + accordions,
        show_progress=False,
    )

    configuration.button_open_config.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_true, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )

    configuration.button_load_config.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,  # Allow load button to work during training
    )

    configuration.button_save_config.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,  # Allow save button to work during training
    )
    
    # Auto-load configuration when a valid config file is selected
    configuration.config_file_name.change(
        fn=lambda config_name, *args: (
            qwen_image_gui_actions("open_configuration", False, config_name, dummy_headless.value, False, *args)
            if config_name and config_name.endswith('.json')
            else ([config_name, ""] + [gr.update() for _ in settings_list])
        ),
        inputs=[configuration.config_file_name] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )

    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[
            executor.button_run,
            executor.stop_row,
            executor.button_stop_training,
            executor.training_status,
        ],
    )

    button_print.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        qwen_image_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), dummy_false, configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[
            executor.button_run,
            executor.stop_row,
            executor.button_stop_training,
            executor.training_status,
            run_state,
        ],
        show_progress=False,
    )
    
    # Wire up stop button with JavaScript confirmation
    executor.button_stop_training.click(
        executor.kill_command,
        inputs=[],
        outputs=[
            executor.button_run,
            executor.stop_row,
            executor.button_stop_training,
            executor.training_status,
        ],
        js="() => { if (confirm('Are you sure you want to stop training?')) { return []; } else { throw new Error('Cancelled'); } }",
    )