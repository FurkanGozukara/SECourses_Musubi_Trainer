import gradio as gr
import os
import re
import shlex
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


def get_debug_parameters_for_mode(debug_mode: str) -> str:
    """
    Convert selected debug mode to command-line parameters for training.

    Parameters:
        debug_mode (str): The selected debug mode

    Returns:
        str: Command-line parameters for the selected debug mode
    """
    debug_params = {
        "Show Timesteps (Image)": "--show_timesteps image",
        "Show Timesteps (Console)": "--show_timesteps console",
        "RCM Debug Save": "--rcm_debug_save",
        "Enable Logging (TensorBoard)": "--log_with tensorboard --logging_dir ./logs",
        "Enable Logging (WandB)": "--log_with wandb",
        "Enable Logging (All)": "--log_with all --logging_dir ./logs",
    }
    return debug_params.get(debug_mode, "")


def manage_additional_parameters(additional_params: str, args_to_add: list = None, args_to_remove: list = None) -> str:
    """
    Manage additional parameters by adding or removing specific arguments without losing user-written values.
    
    Args:
        additional_params: The current additional_parameters string
        args_to_add: List of argument strings to add (e.g., ['--disable_numpy_memmap', '--metadata_arch "qwen-image-edit-plus"'])
        args_to_remove: List of argument strings to remove (e.g., ['--disable_numpy_memmap'])
    
    Returns:
        Modified additional_parameters string with requested args added/removed
    """
    if args_to_add is None:
        args_to_add = []
    if args_to_remove is None:
        args_to_remove = []
    
    if not additional_params and not args_to_add:
        return ""
    
    # Parse existing parameters into a list of tokens
    # Split by spaces, but preserve quoted strings
    import shlex
    try:
        if additional_params.strip():
            existing_args = shlex.split(additional_params)
        else:
            existing_args = []
    except Exception:
        # Fallback: simple split if shlex fails
        existing_args = additional_params.split() if additional_params.strip() else []
    
    # Remove arguments that should be removed
    args_to_remove_normalized = []
    for arg in args_to_remove:
        # Normalize: remove leading dashes, handle both --arg and arg forms
        # Also handle quoted args like '--metadata_arch "qwen-image-edit-plus"'
        normalized = arg.lstrip('-').split()[0]  # Get first word before any quotes
        args_to_remove_normalized.append(normalized)
    
    # Build list of args to keep (excluding ones to remove)
    filtered_args = []
    i = 0
    while i < len(existing_args):
        arg = existing_args[i]
        # Check if this arg should be removed
        should_remove = False
        normalized_arg = arg.lstrip('-')
        
        for remove_arg in args_to_remove_normalized:
            if normalized_arg == remove_arg:
                should_remove = True
                # If it's a flag with a value (like --metadata_arch "value"), skip the value too
                if i + 1 < len(existing_args) and not existing_args[i + 1].startswith('-'):
                    i += 1  # Skip the value
                break
        
        if not should_remove:
            filtered_args.append(arg)
        i += 1
    
    # Add new arguments (avoid duplicates)
    args_to_add_normalized = []
    for arg in args_to_add:
        # Normalize for duplicate checking
        normalized = arg.lstrip('-').split()[0]  # Get first word (e.g., 'disable_numpy_memmap' from '--disable_numpy_memmap')
        args_to_add_normalized.append(normalized)
    
    # Check which args to add are not already present
    existing_normalized = [arg.lstrip('-').split()[0] for arg in filtered_args]
    for arg in args_to_add:
        normalized = arg.lstrip('-').split()[0]
        if normalized not in existing_normalized:
            # Parse the arg to add (handle quoted strings properly)
            try:
                parsed = shlex.split(arg)
                filtered_args.extend(parsed)
            except Exception:
                # Fallback: simple append
                filtered_args.append(arg)
    
    # Return as space-separated string
    return ' '.join(filtered_args) if filtered_args else ""


def upsert_parameter(parameters, key: str, value):
    """Return a new parameter list where `key` is set to `value` exactly once."""
    updated: list[tuple] = []
    replaced = False
    for k, v in parameters:
        if k == key:
            if not replaced:
                updated.append((k, value))
                replaced = True
            # Skip duplicate instances of the same key
        else:
            updated.append((k, v))

    if not replaced:
        updated.append((key, value))

    return updated


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
                    label="Control Directory Name (Edit Mode)",
                    value=self.config.get("dataset_control_directory", "edit_images"),
                    info="[EDIT MODE] Directory containing control/reference images for Qwen-Image-Edit training. Only used when edit=true in Model Settings. Place control images in this subfolder within each dataset folder.\n\n📁 Dataset Structure:\n• Training images: dataset_folder/images/\n• Control images: dataset_folder/edit_images/\n• Captions: dataset_folder/images/*.txt\n\n🖼️ File Naming:\n• Single control: image1.jpg → edit_images/image1.png\n• Multiple controls (Edit-2509): image1.jpg → edit_images/image1_0.png, image1_1.png, image1_2.png\n• Supports: .png, .jpg, .jpeg, .webp formats"
                )
            
            with gr.Row():
                self.dataset_qwen_image_edit_no_resize_control = gr.Checkbox(
                    label="Qwen Image Edit: No Resize Control",
                    value=self.config.get("dataset_qwen_image_edit_no_resize_control", False),
                    info="[EDIT MODE ONLY] When checked, control images keep their original size (no resizing). Use when your control images are already the correct size. Ignored in normal LoRA training."
                )
                
                self.dataset_qwen_image_edit_control_resolution_width = gr.Number(
                    label="Control Image Width",
                    value=self.config.get("dataset_qwen_image_edit_control_resolution_width", 1328),
                    minimum=0,
                    maximum=4096,
                    step=64,
                    info="[EDIT MODE ONLY] Width for control images. 0 = use training resolution. 1024 = Official Qwen-Image-Edit default. 1328 = Optimal for Qwen models. Only used when edit=true. Cannot be used with 'No Resize Control'."
                )
                
                self.dataset_qwen_image_edit_control_resolution_height = gr.Number(
                    label="Control Image Height",
                    value=self.config.get("dataset_qwen_image_edit_control_resolution_height", 1328),
                    minimum=0,
                    maximum=4096,
                    step=64,
                    info="[EDIT MODE ONLY] Height for control images. 0 = use training resolution. 1024 = Official Qwen-Image-Edit default. 1328 = Optimal for Qwen models. Set both width & height for custom control resolution."
                )
                
                self.auto_generate_black_control_images = gr.Checkbox(
                    label="Auto Generate Black Control Images",
                    value=self.config.get("auto_generate_black_control_images", False),
                    info="[EDIT MODE ONLY] Automatically generate pitch black PNG images as control images. Uses same filenames as training images."
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
            control_res_width,  # Add control resolution width
            control_res_height,  # Add control resolution height
            auto_generate_black,  # Add auto generate black control images
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
                
                # Auto-generate black control images if requested
                if auto_generate_black:
                    from PIL import Image
                    import numpy as np
                    
                    # Determine control image dimensions
                    control_width = control_res_width if control_res_width > 0 else 1024
                    control_height = control_res_height if control_res_height > 0 else 1024
                    
                    subfolder_paths = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) 
                                     if os.path.isdir(os.path.join(parent_folder, d))]
                    
                    for folder_path in subfolder_paths:
                        # Create control directory
                        control_folder = os.path.join(folder_path, control_dir)
                        os.makedirs(control_folder, exist_ok=True)
                        
                        # Generate black control images for each training image
                        for img_file in os.listdir(folder_path):
                            img_path = os.path.join(folder_path, img_file)
                            if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                                # Create black control image with numbered suffix for Edit-2509 compatibility
                                base_name = os.path.splitext(img_file)[0]
                                control_img_path = os.path.join(control_folder, f"{base_name}_0.png")
                                
                                if not os.path.exists(control_img_path):
                                    # Create pitch black image
                                    black_image = Image.new('RGB', (control_width, control_height), (0, 0, 0))
                                    black_image.save(control_img_path, 'PNG')
                
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
                
                # Add control resolution settings to datasets with control directories
                if control_res_width > 0 and control_res_height > 0:
                    for dataset_entry in config.get("datasets", []):
                        if "control_directory" in dataset_entry:
                            dataset_entry["qwen_image_edit_control_resolution"] = [int(control_res_width), int(control_res_height)]
                            messages.append(f"[OK] Added control resolution {control_res_width}x{control_res_height} to dataset")
                
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
                        self.dataset_qwen_image_edit_control_resolution_width,
                        self.dataset_qwen_image_edit_control_resolution_height,
                        self.auto_generate_black_control_images,
                        saveLoadSettings.output_dir  # Pass output_dir
                    ],
                    outputs=[self.dataset_config, self.generated_toml_path, self.dataset_status]
                )
            else:
                # Fallback without output_dir
                self.generate_toml_button.click(
                    fn=lambda *args: generate_dataset_config(*args, None),  # Pass all args + None for output_dir
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
                        self.dataset_qwen_image_edit_control_resolution_width,
                        self.dataset_qwen_image_edit_control_resolution_height,
                        self.auto_generate_black_control_images,
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
        # Training Mode Selection (placed at top for visibility)
        with gr.Row():
            gr.Markdown("## Training Mode Configuration")
        
        with gr.Row():
            self.training_mode = gr.Radio(
                label="Training Mode",
                choices=["LoRA Training", "DreamBooth Fine-Tuning"],
                value=self.config.get("training_mode", "LoRA Training"),
                info="LoRA: Efficient parameter-efficient fine-tuning. Faster to train on lower VRAM GPUs with FP8 Scaled."
            )
        
        # Qwen-Image-Edit mode toggle (placed after training mode)
        with gr.Row():
            self.edit = gr.Checkbox(
                label="Enable Qwen-Image-Edit Mode",
                info="Enable image editing training with control images. Requires control_image_path in dataset configuration and Qwen-Image-Edit DiT model",
                value=self.config.get("edit", False),
            )
            self.edit_plus = gr.Checkbox(
                label="Enable Qwen-Image-Edit-2509 Mode",
                info="Enable Edit-2509 training with multiple control images (up to 3). Requires Qwen-Image-Edit-2509 DiT model. Cannot be used with regular Edit mode.",
                value=self.config.get("edit_plus", False),
            )
            self.faster_model_loading = gr.Checkbox(
                label="Faster Model Loading (Uses more RAM but speeds up model loading speed - Enable for RunPod)",
                info="Disables numpy memmap for faster model loading. Uses more RAM but speeds up loading, especially useful for RunPod and similar environments.",
                value=self.config.get("faster_model_loading", False),
            )
            self.use_pinned_memory_for_block_swap = gr.Checkbox(
                label="Use Pinned Memory for Block Swapping (Faster on Windows - Requires more RAM)",
                info="Uses more system RAM but speeds up training. The speed up maybe significant depending on system settings. To work, go to Advanced Graphics settings in System > Display > Graphics as in tutorial video and disable Hardware-Accelerated GPU Scheduling and restart your PC.",
                value=self.config.get("use_pinned_memory_for_block_swap", False),
            )
        
        # Torch Compile Settings - for faster training with torch.compile
        self.torch_compile_accordion = gr.Accordion("Torch Compile Settings", open=False)
        with self.torch_compile_accordion:
            gr.Markdown(
                """⚠️ **Important:** If you get errors with torch.compile just disable it but it should work out of box with 0-Trade-off. It increases speed and slightly reduces VRAM with 0 quality loss."""
            )
            
            with gr.Row():
                self.compile = gr.Checkbox(
                    label="Enable torch.compile",
                    info="Enable torch.compile for faster training (requires PyTorch 2.1+, Triton for CUDA). Works with SDXL and FLUX. Disable gradient checkpointing for best results!",
                    value=self.config.get("compile", False),
                    interactive=True,
                )
                
                self.compile_backend = gr.Dropdown(
                    label="Compile Backend",
                    info="Backend for torch.compile (default: inductor)",
                    choices=["inductor", "cudagraphs", "eager", "aot_eager", "aot_ts_nvfuser"],
                    value=self.config.get("compile_backend", "inductor"),
                    interactive=True,
                )
                
                self.compile_mode = gr.Dropdown(
                    label="Compile Mode",
                    info="Optimization mode for torch.compile",
                    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    value=self.config.get("compile_mode", "default"),
                    interactive=True,
                )
            
            with gr.Row():
                self.compile_dynamic = gr.Dropdown(
                    label="Dynamic Shapes",
                    info="Dynamic shape handling: auto (default), true (enable), false (disable)",
                    choices=["auto", "true", "false"],
                    value=self.config.get("compile_dynamic", "auto"),
                    allow_custom_value=False,
                    interactive=True,
                )
                
                self.compile_fullgraph = gr.Checkbox(
                    label="Fullgraph Mode",
                    info="Enable fullgraph mode in torch.compile (may fail with complex models)",
                    value=self.config.get("compile_fullgraph", False),
                    interactive=True,
                )
                
                self.compile_cache_size_limit = gr.Number(
                    label="Cache Size Limit",
                    info="Set torch._dynamo.config.cache_size_limit (0 = use PyTorch default, typically 8-32)",
                    value=self.config.get("compile_cache_size_limit", 0),
                    step=1,
                    minimum=0,
                    interactive=True,
                )
        
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
        
        # Add DiT input channels parameter
        with gr.Row():
            self.dit_in_channels = gr.Number(
                label="DiT Input Channels",
                info="[DO NOT CHANGE] VAE latent channels that DiT expects. MUST be 16 for Qwen Image (VAE encodes images to 16-channel latents). Different from SD (4 channels). Changing this will cause training to fail. This represents the depth of encoded image data.",
                value=self.config.get("dit_in_channels", 16),
                # minimum=1,  # Removed: Let backend handle validation
                # maximum=32,
                step=1,
                interactive=True,
            )
            
            self.num_layers = gr.Number(
                label="Number of DiT Layers",
                info="🔧 ADVANCED: Number of transformer layers in the DiT model. Leave empty for auto-detection (60 layers for standard Qwen Image models). Only modify when using custom or pruned models. Incorrect values will cause training failure. Supported range: 1-60 layers.",
                value=self.config.get("num_layers", None) if self.config.get("num_layers") not in ["", None] else None,
                # minimum=1,  # Removed: Let backend handle validation
                # maximum=60,
                step=1,
                interactive=True,
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
            label="Use FP8 for Text Encoder (Qwen2.5-VL)",
            info="[🔥 RECOMMENDED for lower VRAM] FP8 quantization for Qwen2.5-VL text encoder. Reduces VRAM with <1% quality loss. MUST match caching_teo_fp8_vl setting if using cached text encoder outputs. RTX 4000+ has native FP8 support",
            value=self.config.get("fp8_vl", False),
        )

        # Qwen Image specific options
        gr.Markdown("""
        ### FP8 Quantization Options
        **Important:** FP8 is an on-the-fly optimization. You MUST provide standard bf16 models as input. 
        Pre-quantized FP8 models are NOT supported. Training outputs LoRA weights only - base models remain in bf16 format.
        
        **GPU Compatibility:** Both options work on ALL CUDA GPUs (RTX 2000/3000/4000). 
        For lower VRAM GPUs, enable BOTH checkboxes to reduce VRAM usage.
        """)
        
        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="FP8 for Base Model (DiT) [On-The-Fly Conversion]",
                info="[⚠️ ALWAYS USE WITH fp8_scaled] Converts BF16 model → FP8 during loading. Reduces VRAM usage. INPUT: Standard BF16 model file. OUTPUT: Cannot save as FP8. For pre-quantized FP8 models, leave this OFF",
                value=self.config.get("fp8_base", False),
            )

            self.fp8_scaled = gr.Checkbox(
                label="Scaled FP8 (Block-wise Scaling) [🎯 CRITICAL for Quality]",
                info="[✅ MUST ENABLE with fp8_base] Implements block-wise FP8 scaling for ~2-3% better quality vs naive FP8. RTX 4000+: Hardware accelerated. RTX 3000: Software emulation (same quality, 5-10% slower). Without this, expect visible quality degradation",
                value=self.config.get("fp8_scaled", False),
            )
            
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap to CPU",
                info="Swap DiT blocks to CPU to save VRAM. Qwen Image has 60 total blocks, max swap is 59. Higher values save more VRAM but require more RAM. Slows training significantly",
                value=self.config.get("blocks_to_swap", 0),
                minimum=0,
                maximum=59,
                step=1,
                interactive=True,
            )

        # FP8 validation warning
        with gr.Row():
            self.fp8_warning = gr.Markdown(
                "",
                visible=False,
                elem_classes=["warning-text"]
            )

        # Add FP8 validation
        def validate_fp8_settings(fp8_base, fp8_scaled, fp8_vl):
            warnings = []
            if fp8_base and not fp8_scaled:
                warnings.append("🔴 **CRITICAL WARNING**: Using fp8_base without fp8_scaled will cause significant quality degradation! Always enable both together for proper block-wise scaling.")
            if warnings:
                return gr.update(value="\n\n".join(warnings), visible=True)
            return gr.update(value="", visible=False)

        # Edit mode mutual exclusion logic
        def handle_edit_mode_change(edit_value, edit_plus_value):
            """Ensure edit and edit_plus are mutually exclusive"""
            if edit_value and edit_plus_value:
                # If both are checked, keep the one that was just checked and uncheck the other
                return gr.update(value=False), gr.update(value=True)
            return gr.update(), gr.update()
        
        def handle_edit_plus_mode_change(edit_value, edit_plus_value):
            """Ensure edit and edit_plus are mutually exclusive"""
            if edit_value and edit_plus_value:
                # If both are checked, keep the one that was just checked and uncheck the other
                return gr.update(value=False), gr.update(value=True)
            return gr.update(), gr.update()

        # Connect edit mode mutual exclusion
        self.edit.change(
            fn=handle_edit_mode_change,
            inputs=[self.edit, self.edit_plus],
            outputs=[self.edit, self.edit_plus]
        )
        self.edit_plus.change(
            fn=handle_edit_plus_mode_change,
            inputs=[self.edit, self.edit_plus],
            outputs=[self.edit, self.edit_plus]
        )

        # Connect FP8 validation
        self.fp8_base.change(
            fn=validate_fp8_settings,
            inputs=[self.fp8_base, self.fp8_scaled, self.fp8_vl],
            outputs=[self.fp8_warning]
        )
        self.fp8_scaled.change(
            fn=validate_fp8_settings,
            inputs=[self.fp8_base, self.fp8_scaled, self.fp8_vl],
            outputs=[self.fp8_warning]
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
            
            self.flow_shift = gr.Number(
                label="Flow Shift (Advanced)",
                info="[ADVANCED] Controls noise schedule in flow matching. Default 7.0 is optimal for most cases. Higher values (8-10) = smoother but slower convergence. Lower values (5-6) = faster but may be unstable. Only change if you understand flow matching theory.",
                value=self.config.get("flow_shift", 7.0),
                minimum=0.0,
                maximum=20.0,
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
                minimum=-10.0,
                maximum=10.0,
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
                minimum=0,
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
                # minimum=0,  # Removed: Let backend handle validation
                # maximum=100,
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
    debug_mode,
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
    dataset_qwen_image_edit_control_resolution_width,
    dataset_qwen_image_edit_control_resolution_height,
    auto_generate_black_control_images,
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
    gradient_checkpointing_cpu_offload,
    gradient_accumulation_steps,
    full_bf16,
    full_fp16,
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
    sample_output_dir,
    disable_prompt_enhancement,
    sample_width,
    sample_height,
    sample_steps,
    sample_guidance_scale,
    sample_seed,
    sample_discrete_flow_shift,
    sample_cfg_scale,
    sample_negative_prompt,
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
    fused_backward_pass,
    training_mode,  # Add training mode parameter
    dit,
    dit_dtype,
    dit_in_channels,
    num_layers,
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
    edit_plus,
    faster_model_loading,
    use_pinned_memory_for_block_swap,
    # Torch Compile settings
    compile,
    compile_backend,
    compile_mode,
    compile_dynamic,
    compile_fullgraph,
    compile_cache_size_limit,
    timestep_sampling,
    discrete_flow_shift,
    flow_shift,
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
    custom_network_module,  # NEW parameter
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
    convert_to_diffusers,  # NEW parameter
    diffusers_output_dir,  # NEW parameter
    convert_to_safetensors,  # NEW parameter
    safetensors_output_dir,  # NEW parameter
    mem_eff_save,
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
        'min_timestep', 'max_timestep', 'discrete_flow_shift', 'flow_shift', 'dit_in_channels', 'num_layers', 'network_dropout',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_qwen_image_edit_control_resolution_width', 'dataset_qwen_image_edit_control_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'num_timestep_buckets', 'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers',
        'sample_width', 'sample_height', 'sample_steps', 'sample_guidance_scale', 'sample_seed',
        'sample_discrete_flow_shift', 'sample_cfg_scale'
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

    if destination_directory and not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    # Process parameters based on training mode
    param_dict = dict(parameters)
    training_mode = param_dict.get("training_mode", "LoRA Training")
    
    # Apply training mode specific modifications
    modified_params = []

    # Always include training_mode so it persists in saved configs
    modified_params.append(("training_mode", training_mode))

    for key, value in parameters:
        # Migrate old debug mode values
        if key == "debug_mode" and isinstance(value, str):
            migration_map = {
                "Dataset Debug (Image)": "None",  # Dataset debugging moved to caching section
                "Dataset Debug (Console)": "None",  # Dataset debugging moved to caching section
                "Dataset Debug (Video)": "None",  # Dataset debugging moved to caching section
            }
            if value in migration_map:
                old_value = value
                value = migration_map[value]
                log.info(f"Migrated debug_mode from '{old_value}' to '{value}'")

        if training_mode == "DreamBooth Fine-Tuning":
            # For DreamBooth/Fine-tuning, we need to disable network parameters
            if key == "network_module":
                # Set to empty or None to disable LoRA
                modified_params.append((key, ""))
                log.info("DreamBooth mode: Disabling network_module for full fine-tuning")
            elif key in ["network_dim", "network_alpha", "network_dropout", "network_args", "network_weights"]:
                # Skip network-specific parameters for DreamBooth
                log.info(f"DreamBooth mode: Skipping LoRA parameter {key}")
                continue
            elif key == "fused_backward_pass":
                # Respect user's choice for fused_backward_pass (DreamBooth only)
                if value:
                    log.info("DreamBooth mode: fused_backward_pass enabled by user (reduces VRAM with AdaFactor)")
                else:
                    log.info("DreamBooth mode: fused_backward_pass disabled by user")
                modified_params.append((key, value))
            else:
                modified_params.append((key, value))
        else:
            # LoRA Training mode - keep all parameters as is
            if key == "network_module" and (not value or value == ""):
                # Ensure network_module is set for LoRA
                modified_params.append((key, "networks.lora_qwen_image"))
                log.info("LoRA mode: Setting network_module to networks.lora_qwen_image")
            elif key == "fused_backward_pass":
                # For LoRA mode, always disable (not effective) regardless of user setting
                modified_params.append((key, False))
                if value:
                    log.info("LoRA mode: fused_backward_pass disabled (not effective for LoRA training)")
            else:
                modified_params.append((key, value))
    
    parameters = modified_params
    
    # Process parameters to handle list values properly
    processed_params = []
    # NOTE: Exclude fields that are legitimately lists like optimizer_args, lr_scheduler_args, network_args
    numeric_fields = [
        'learning_rate', 'max_grad_norm', 'guidance_scale', 'logit_mean', 'logit_std',
        'mode_scale', 'sigmoid_scale', 'lr_scheduler_power', 'lr_scheduler_timescale',
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',  # base_weights_multiplier is a Number in Qwen Image GUI
        'vae_chunk_size', 'vae_spatial_tile_sample_min_size', 'blocks_to_swap',
        'min_timestep', 'max_timestep', 'discrete_flow_shift', 'flow_shift', 'dit_in_channels', 'num_layers', 'network_dropout',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_qwen_image_edit_control_resolution_width', 'dataset_qwen_image_edit_control_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'num_timestep_buckets', 'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers',
        'sample_width', 'sample_height', 'sample_steps', 'sample_guidance_scale', 'sample_seed',
        'sample_discrete_flow_shift', 'sample_cfg_scale'
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
                "headless",
                "print_only",
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

    # REMOVED: All minimum constraints to prevent Gradio bounds errors
    # Backend will handle parameter validation
    minimum_constraints = {}

    # Parameters that should be None when their value is 0 (optional parameters)
    # NOTE: Only include parameters where 0 truly means "disabled/not set"
    # DO NOT include parameters where 0 is a valid functional value
    optional_parameters = {
        "sample_every_n_steps", "sample_every_n_epochs", 
        "save_every_n_steps", "save_every_n_epochs", "max_timestep", "min_timestep",
        "network_dim", "num_layers",  # NEW: These can be None for auto-detection
        "max_train_epochs"  # NEW: 0 means use max_train_steps instead
        # Removed: "ddp_timeout" (0 = use default 30min timeout - VALID)
        # Removed: "save_last_n_epochs" (0 = keep all epochs - VALID)
    }

    # NOTE: Exclude fields that are legitimately lists like optimizer_args, lr_scheduler_args, network_args
    numeric_fields = [
        'learning_rate', 'max_grad_norm', 'guidance_scale', 'logit_mean', 'logit_std',
        'mode_scale', 'sigmoid_scale', 'lr_scheduler_power', 'lr_scheduler_timescale',
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',  # base_weights_multiplier is a Number in Qwen Image GUI
        'vae_chunk_size', 'vae_spatial_tile_sample_min_size', 'blocks_to_swap',
        'min_timestep', 'max_timestep', 'discrete_flow_shift', 'flow_shift', 'dit_in_channels', 'num_layers', 'network_dropout',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_qwen_image_edit_control_resolution_width', 'dataset_qwen_image_edit_control_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'num_timestep_buckets', 'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers',
        'sample_width', 'sample_height', 'sample_steps', 'sample_guidance_scale', 'sample_seed',
        'sample_discrete_flow_shift', 'sample_cfg_scale', 'compile_cache_size_limit'
    ]
    
    # Process parameters and track which ones are included
    values = [file_path, gr.update(value=status_msg, visible=True)]
    included_params = []  # Track which parameters are actually included

    for key, value in parameters:
        if not key in ["ask_for_file", "apply_preset", "file_path"]:
            included_params.append(key)  # Track this parameter
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

                # Convert empty strings to None for optional numeric parameters
                if key == "num_layers" and (toml_value == "" or toml_value is None):
                    toml_value = None
                # Convert 0 to None for optional parameters to avoid minimum constraint violations
                elif key in optional_parameters and toml_value == 0:
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

    # Final validation before returning - now we can properly match parameters to values
    result_values = []
    param_index = 0  # Index into included_params
    for i, v in enumerate(values):
        if i < 2:  # file_path and gr.update
            result_values.append(v)
        else:
            # This is a parameter value
            param_name = included_params[param_index] if param_index < len(included_params) else "unknown"

            if isinstance(v, list):
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
                        result_values.append(" ".join(str(item) for item in v) if isinstance(v, list) else v)
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

            param_index += 1

    return tuple(result_values)


def generate_enhanced_prompt_file(
    original_prompt_file: str,
    output_dir: str,
    output_name: str,
    sample_width: int = 1328,
    sample_height: int = 1328,
    sample_steps: int = 20,
    sample_guidance_scale: float = 1.0,
    sample_seed: int = -1,
    sample_discrete_flow_shift: float = 0,
    sample_cfg_scale: float = 1.0,
    sample_negative_prompt: str = ""
) -> str:
    """
    Generate an enhanced prompt file with GUI defaults added to prompts that don't have parameters.
    
    Args:
        original_prompt_file: Path to the original prompt file
        output_dir: Directory to save the enhanced prompt file
        output_name: Base name for the output file
        sample_width: Default width for samples
        sample_height: Default height for samples
        sample_steps: Default number of denoising steps
        sample_guidance_scale: Default guidance scale
        sample_seed: Default seed (-1 for random)
        
    Returns:
        Path to the enhanced prompt file
    """
    try:
        # Read original prompt file
        if not os.path.exists(original_prompt_file):
            log.error(f"Original prompt file not found: {original_prompt_file}")
            return None
            
        with open(original_prompt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each line
        enhanced_lines = []
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                enhanced_lines.append(line)
                continue
            
            # Check if line already has parameters
            has_width = '--w ' in line or '-w ' in line
            has_height = '--h ' in line or '-h ' in line
            has_steps = '--s ' in line or '-s ' in line
            has_guidance = '--g ' in line or '-g ' in line
            has_seed = '--d ' in line or '-d ' in line
            has_flow_shift = '--fs ' in line or '-fs ' in line
            has_cfg_scale = '--l ' in line or '-l ' in line
            has_negative = '--n ' in line or '-n ' in line
            
            # Build enhanced line with defaults for missing parameters
            enhanced_line = line
            
            # Add width and height if not present (most important for Qwen)
            if not has_width:
                enhanced_line += f" --w {sample_width}"
            if not has_height:
                enhanced_line += f" --h {sample_height}"
            
            # Add steps if not present
            if not has_steps:
                enhanced_line += f" --s {sample_steps}"
            
            # Add guidance if not present
            if not has_guidance:
                enhanced_line += f" --g {sample_guidance_scale}"
            
            # Add flow shift if not present and not 0 (0 means use training value)
            if not has_flow_shift and sample_discrete_flow_shift != 0:
                enhanced_line += f" --fs {sample_discrete_flow_shift}"
            
            # Add cfg scale if not present and not 1.0 (1.0 means disabled)
            if not has_cfg_scale and sample_cfg_scale != 1.0:
                enhanced_line += f" --l {sample_cfg_scale}"
            
            # Add negative prompt if not present and not empty
            if not has_negative and sample_negative_prompt and sample_negative_prompt.strip():
                enhanced_line += f" --n {sample_negative_prompt}"
            
            # Add seed if not present and not random
            if not has_seed and sample_seed != -1:
                enhanced_line += f" --d {sample_seed}"
            
            enhanced_lines.append(enhanced_line)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        enhanced_filename = f"{output_name}_enhanced_prompts_{timestamp}.txt"
        enhanced_path = os.path.join(output_dir, enhanced_filename)
        
        # Write enhanced prompt file
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            f.write("# Enhanced prompt file generated by Qwen Image LoRA GUI\n")
            f.write(f"# Original file: {original_prompt_file}\n")
            f.write(f"# Defaults applied: width={sample_width}, height={sample_height}, steps={sample_steps}, guidance={sample_guidance_scale}\n")
            if sample_seed != -1:
                f.write(f"# Fixed seed: {sample_seed}\n")
            f.write("#" + "="*70 + "\n\n")
            
            for line in enhanced_lines:
                f.write(line + '\n')
        
        log.info(f"Enhanced prompt file created: {enhanced_path}")
        return enhanced_path
        
    except Exception as e:
        log.error(f"Failed to generate enhanced prompt file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_qwen_image_model(headless, print_only, parameters):
    import sys
    import json
    import os
    import time
    
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
        # Check if this looks like a generated dataset config path (contains timestamp)
        if re.search(r'dataset_config_\d{8}_\d{6}\.toml', effective_dataset_config):
            raise ValueError(
                f"[ERROR] Dataset config file does not exist: {effective_dataset_config}\n"
                "\nThis appears to be a generated dataset config file that may have been deleted or moved.\n"
                "\nTo fix this issue:\n"
                "1. Go to the 'Dataset Config' section\n"
                "2. Set 'Dataset Config Mode' to 'Generate from Folder Structure'\n"
                "3. Click 'Generate Dataset Config' to create a new config file\n"
                "4. The new config path will be automatically set in 'Dataset Config File'\n"
                "5. Save your configuration again\n"
                "\nAlternatively, manually create or locate a valid dataset config file."
            )
        else:
            raise ValueError(
                f"[ERROR] Dataset config file does not exist: {effective_dataset_config}\n"
                "\nPlease check the file path or create the dataset configuration file first.\n"
                "You can generate a new dataset config using the 'Dataset Config' section."
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

    # Ensure the validated dataset config path is propagated to downstream saves/commands
    parameters = upsert_parameter(parameters, "dataset_config", effective_dataset_config)

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
        if param_dict.get("caching_latent_debug_mode") is not None and param_dict.get("caching_latent_debug_mode") not in ["", "None"]:
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
        
        # Add edit mode flags for latent caching
        if param_dict.get("edit"):
            run_cache_latent_cmd.append("--edit")
        if param_dict.get("edit_plus"):
            run_cache_latent_cmd.append("--edit_plus")

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
        if param_dict.get("edit_plus"):
            run_cache_teo_cmd.append("--edit_plus")

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

    # Select the appropriate Qwen Image training script based on training mode
    training_mode = param_dict.get("training_mode", "LoRA Training")
    if training_mode == "DreamBooth Fine-Tuning":
        # Use full fine-tuning script for DreamBooth mode
        run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/qwen_image_train.py")
        log.info("Using qwen_image_train.py for full DreamBooth fine-tuning")
    else:
        # Use network training script for LoRA mode
        run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py")
        log.info("Using qwen_image_train_network.py for LoRA training")

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
        else:
            # Check if prompt enhancement is disabled
            disable_enhancement = param_dict.get('disable_prompt_enhancement', False)
            
            if disable_enhancement:
                # Use original prompts without enhancement
                log.info("Prompt enhancement disabled - using original prompts without Kohya format parameters")
            else:
                # Generate enhanced prompt file with GUI defaults
                original_prompt_file = param_dict.get('sample_prompts')
                
                # Use custom sample output directory if provided, otherwise use output directory
                sample_output_dir = param_dict.get('sample_output_dir', '').strip()
                if sample_output_dir:
                    output_dir = sample_output_dir
                    log.info(f"Using custom sample output directory: {sample_output_dir}")
                else:
                    output_dir = param_dict.get('output_dir')
                    log.info(f"Using default output directory for samples: {output_dir}")
                
                # Create enhanced prompt file
                enhanced_prompt_file = generate_enhanced_prompt_file(
                    original_prompt_file=original_prompt_file,
                    output_dir=output_dir,
                    output_name=param_dict.get('output_name'),
                    sample_width=param_dict.get('sample_width', 1328),
                    sample_height=param_dict.get('sample_height', 1328),
                    sample_steps=param_dict.get('sample_steps', 20),
                    sample_guidance_scale=param_dict.get('sample_guidance_scale', 1.0),
                    sample_seed=param_dict.get('sample_seed', -1),
                    sample_discrete_flow_shift=param_dict.get('sample_discrete_flow_shift', 0),
                    sample_cfg_scale=param_dict.get('sample_cfg_scale', 1.0),
                    sample_negative_prompt=param_dict.get('sample_negative_prompt', '')
                )
                
                if enhanced_prompt_file:
                    # Update parameters to use the enhanced prompt file
                    modified_params = []
                    for key, value in parameters:
                        if key == 'sample_prompts':
                            modified_params.append((key, enhanced_prompt_file))
                            log.info(f"Using enhanced prompt file: {enhanced_prompt_file}")
                        else:
                            modified_params.append((key, value))
                    parameters = modified_params

        # Modify parameters based on training mode
        training_mode = param_dict.get("training_mode", "LoRA Training")
        modified_params = [("training_mode", training_mode)]
        
        for key, value in parameters:
            if training_mode == "DreamBooth Fine-Tuning":
                # For DreamBooth/Fine-tuning, we need to disable network parameters
                if key == "network_module":
                    # Set to empty or None to disable LoRA
                    modified_params.append((key, ""))
                    log.info("DreamBooth mode: Disabling network_module for full fine-tuning")
                elif key in ["network_dim", "network_alpha", "network_dropout", "network_args", "network_weights"]:
                    # Skip network-specific parameters for DreamBooth
                    log.info(f"DreamBooth mode: Skipping LoRA parameter {key}")
                    continue
                elif key == "fused_backward_pass":
                    # Respect user's choice for fused_backward_pass (DreamBooth only)
                    if value:
                        log.info("DreamBooth mode: fused_backward_pass enabled by user (reduces VRAM with AdaFactor)")
                    else:
                        log.info("DreamBooth mode: fused_backward_pass disabled by user")
                    modified_params.append((key, value))
                else:
                    modified_params.append((key, value))
            else:
                # LoRA Training mode - keep all parameters as is
                if key == "network_module":
                    # Handle custom network module selection
                    if value == "custom":
                        # Use the custom_network_module value instead
                        custom_module = param_dict.get("custom_network_module", "")
                        if custom_module and custom_module.strip():
                            modified_params.append((key, custom_module))
                            log.info(f"LoRA mode: Using custom network module: {custom_module}")
                        else:
                            # Fallback to default if custom module not specified
                            modified_params.append((key, "networks.lora_qwen_image"))
                            log.warning("LoRA mode: Custom module selected but not specified, using default networks.lora_qwen_image")
                    elif not value or value == "":
                        # Ensure network_module is set for LoRA
                        modified_params.append((key, "networks.lora_qwen_image"))
                        log.info("LoRA mode: Setting network_module to networks.lora_qwen_image")
                    else:
                        # Use the selected module directly
                        modified_params.append((key, value))
                        log.info(f"LoRA mode: Using network module: {value}")
                elif key == "custom_network_module":
                    # Skip this parameter as it's handled above
                    continue
                elif key in ["convert_to_diffusers", "diffusers_output_dir", "convert_to_safetensors", "safetensors_output_dir"]:
                    # Skip conversion parameters - they are post-training operations
                    log.info(f"Skipping post-training conversion parameter: {key}")
                    continue
                elif key == "fused_backward_pass":
                    # Disable fused_backward_pass for LoRA (not effective)
                    modified_params.append((key, False))
                else:
                    modified_params.append((key, value))
        
        parameters = modified_params
        
        # Handle logging_dir intelligently - create automatic path when empty since training script enables logging automatically
        modified_params = []
        for key, value in parameters:
            if key == "logging_dir":
                output_dir = param_dict.get("output_dir", "")

                # If logging_dir is empty, create automatic path (training script enables TensorBoard when logging_dir is not None)
                if not value or value == "" or value == "/":
                    # Use output_dir as base with a 'logs' subdirectory
                    if output_dir:
                        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        value = os.path.join(output_dir, "logs", f"session_{timestamp}")
                        log.info(f"Auto-generating logging directory: {value}")
                    else:
                        # Fallback to current directory if output_dir is not set
                        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        value = os.path.join(".", "logs", f"session_{timestamp}")
                        log.info(f"Auto-generating logging directory in current folder: {value}")
                # If it's a relative path (doesn't start with / on Linux or drive letter on Windows)
                elif not os.path.isabs(value):
                    # Make it relative to output_dir
                    if output_dir:
                        value = os.path.join(output_dir, value)
                        log.info(f"Using relative logging directory under output_dir: {value}")
                    else:
                        # Keep as is if no output_dir
                        log.info(f"Using relative logging directory: {value}")
                else:
                    # It's an absolute path, use as is
                    log.info(f"Using absolute logging directory: {value}")

                # Ensure the path uses forward slashes for compatibility
                value = value.replace("\\", "/") if value else ""

                modified_params.append((key, value))
            else:
                modified_params.append((key, value))
        
        parameters = modified_params
        
        # Ensure at least one attention option is enabled for training
        attention_opts = ["sdpa", "flash_attn", "flash3", "sage_attn", "xformers"]
        if not any(bool(param_dict.get(opt)) for opt in attention_opts):
            log.warning("No attention option selected; enabling SDPA by default for DreamBooth fine-tuning")
            param_dict["sdpa"] = True
        else:
            # Sanity check: ensure at least one option carries through the parameter list
            if all(not bool(value) for key, value in param_dict.items() if key in attention_opts):
                log.warning("Attention options provided empty/false. Forcing sdpa True to avoid training failure.")
                param_dict["sdpa"] = True

        # Normalize attention values in parameters to match param_dict after all toggles
        normalized_parameters = []
        seen_attn_keys = set()
        for key, value in parameters:
            if key in attention_opts:
                seen_attn_keys.add(key)
                normalized_parameters.append((key, bool(param_dict.get(key))))
            else:
                normalized_parameters.append((key, value))

        for opt in attention_opts:
            if opt not in seen_attn_keys:
                normalized_parameters = upsert_parameter(normalized_parameters, opt, bool(param_dict.get(opt)))

        parameters = normalized_parameters

        # Ensure dataset_config entry survives all transformations
        parameters = upsert_parameter(parameters, "dataset_config", effective_dataset_config)

        # Handle faster_model_loading checkbox: save disable_numpy_memmap to TOML
        faster_model_loading_enabled = param_dict.get("faster_model_loading", False)
        log.info(f"faster_model_loading checkbox value: {faster_model_loading_enabled}")
        if faster_model_loading_enabled:
            parameters = upsert_parameter(parameters, "disable_numpy_memmap", True)
            log.info("Added disable_numpy_memmap = True to TOML parameters")
        else:
            # Remove disable_numpy_memmap if it exists (so it won't be saved to TOML)
            parameters = [(k, v) for k, v in parameters if k != "disable_numpy_memmap"]
            log.info("Removed disable_numpy_memmap from TOML parameters")
        
        # Handle edit_plus checkbox: save metadata_arch to TOML
        edit_plus_enabled = param_dict.get("edit_plus", False)
        log.info(f"edit_plus checkbox value: {edit_plus_enabled}")
        if edit_plus_enabled:
            parameters = upsert_parameter(parameters, "metadata_arch", "qwen-image-edit-plus")
            log.info("Added metadata_arch = 'qwen-image-edit-plus' to TOML parameters")
        else:
            # Remove metadata_arch if edit_plus is disabled (set to None to remove from TOML)
            # Check if it exists and remove it
            parameters = [(k, v) for k, v in parameters if k != "metadata_arch"]
            log.info("Removed metadata_arch from TOML parameters")
        
        # Handle use_pinned_memory_for_block_swap checkbox: check if parameter exists in training script
        use_pinned_memory_enabled = param_dict.get("use_pinned_memory_for_block_swap", False)
        log.info(f"use_pinned_memory_for_block_swap checkbox value: {use_pinned_memory_enabled}")
        
        # Check if the parameter is supported by the training script
        # The parameter is defined in hv_train_network.py (base training module used by Qwen Image)
        # Get project directory (parent of musubi_tuner_gui module)
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check both musubi-tuner and musubi-tuner2 directories (in case user is on a branch)
        possible_paths = [
            os.path.join(project_dir, "musubi-tuner", "src", "musubi_tuner", "hv_train_network.py"),
            os.path.join(project_dir, "musubi-tuner2", "src", "musubi_tuner", "hv_train_network.py"),
        ]
        
        parameter_supported = False
        
        for training_script_path in possible_paths:
            try:
                if os.path.exists(training_script_path):
                    log.info(f"Checking for parameter support in: {training_script_path}")
                    with open(training_script_path, "r", encoding="utf-8") as f:
                        script_content = f.read()
                        # Check for the parameter (with underscores or dashes)
                        if "use_pinned_memory_for_block_swap" in script_content or "use-pinned-memory-for-block-swap" in script_content:
                            parameter_supported = True
                            log.info(f"Found use_pinned_memory_for_block_swap parameter in {training_script_path}")
                            break
                        else:
                            log.info(f"Parameter not found in {training_script_path}")
            except Exception as e:
                log.warning(f"Could not check training script for parameter support: {e}")
        
        if not parameter_supported:
            log.info("use_pinned_memory_for_block_swap parameter not found in any training script")
        
        if use_pinned_memory_enabled and parameter_supported:
            parameters = upsert_parameter(parameters, "use_pinned_memory_for_block_swap", True)
            log.info("Added use_pinned_memory_for_block_swap = True to TOML parameters")
        else:
            # Remove use_pinned_memory_for_block_swap if it exists (so it won't be saved to TOML)
            parameters = [(k, v) for k, v in parameters if k != "use_pinned_memory_for_block_swap"]
            if use_pinned_memory_enabled and not parameter_supported:
                log.info("use_pinned_memory_for_block_swap is not supported by the current training script version. Skipping.")
            else:
                log.info("Removed use_pinned_memory_for_block_swap from TOML parameters")
        
        # Verify parameters are in the list before saving
        disable_numpy_memmap_in_params = any(k == "disable_numpy_memmap" for k, v in parameters)
        metadata_arch_in_params = any(k == "metadata_arch" for k, v in parameters)
        use_pinned_memory_in_params = any(k == "use_pinned_memory_for_block_swap" for k, v in parameters)
        log.info(f"Before SaveConfigFileToRun: disable_numpy_memmap in params: {disable_numpy_memmap_in_params}, metadata_arch in params: {metadata_arch_in_params}, use_pinned_memory_for_block_swap in params: {use_pinned_memory_in_params}")

        pattern_exclusion = []
        for key, _ in parameters:
            if key.startswith('caching_latent_') or key.startswith('caching_teo_'):
                pattern_exclusion.append(key)
        
        # Also exclude training_mode from the TOML as it's not a training parameter

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
                "training_mode",  # Exclude training_mode as it's GUI-only
            ] + pattern_exclusion,
            mandatory_keys=["dataset_config", "dit", "vae", "text_encoder"],
        )
        
        # Verify TOML file was created and check its contents
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                toml_content = toml.load(f)
                disable_numpy_memmap_in_toml = toml_content.get("disable_numpy_memmap", None)
                metadata_arch_in_toml = toml_content.get("metadata_arch", None)
                use_pinned_memory_in_toml = toml_content.get("use_pinned_memory_for_block_swap", None)
                log.info(f"After SaveConfigFileToRun - TOML file contents check:")
                log.info(f"  disable_numpy_memmap = {disable_numpy_memmap_in_toml}")
                log.info(f"  metadata_arch = {metadata_arch_in_toml}")
                log.info(f"  use_pinned_memory_for_block_swap = {use_pinned_memory_in_toml}")
        
        run_cmd.append("--config_file")
        run_cmd.append(f"{file_path}")

        # Handle debug mode selection
        additional_params = param_dict.get("additional_parameters", "")
        debug_mode_selected = param_dict.get("debug_mode", "None")
        if debug_mode_selected != "None":
            debug_params = get_debug_parameters_for_mode(debug_mode_selected)
            if debug_params:
                if additional_params:
                    additional_params += " " + debug_params
                else:
                    additional_params = debug_params

        # Also handle faster_model_loading, edit_plus, and use_pinned_memory_for_block_swap in additional_parameters
        # to ensure they're passed via command line as well (for backward compatibility)
        args_to_add = []
        args_to_remove = []
        
        if faster_model_loading_enabled:
            args_to_add.append("--disable_numpy_memmap")
        else:
            args_to_remove.append("--disable_numpy_memmap")
        
        if edit_plus_enabled:
            args_to_add.append('--metadata_arch "qwen-image-edit-plus"')
        else:
            args_to_remove.append('--metadata_arch "qwen-image-edit-plus"')
        
        # Only add use_pinned_memory_for_block_swap if it's supported by the training script
        if use_pinned_memory_enabled and parameter_supported:
            args_to_add.append("--use_pinned_memory_for_block_swap")
        else:
            args_to_remove.append("--use_pinned_memory_for_block_swap")
        
        # Apply changes to additional_params (preserve user-written values)
        if args_to_add or args_to_remove:
            additional_params = manage_additional_parameters(
                additional_params,
                args_to_add=args_to_add,
                args_to_remove=args_to_remove
            )

        run_cmd_params = {
            "additional_parameters": additional_params,
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
                # Properly quote arguments for Windows batch files
                teo_cache_cmd_str = ' '.join([f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in teo_cache_cmd])
                run_cmd_str = ' '.join([f'"{arg}"' if ' ' in str(arg) else str(arg) for arg in run_cmd])
                script_content = f"""@echo off
echo Starting text encoder output caching...
{teo_cache_cmd_str}
if %errorlevel% neq 0 (
    echo Text encoder caching failed with error code %errorlevel%
    exit /b %errorlevel%
)
echo Text encoder caching completed successfully!
echo Starting training...
{run_cmd_str}
"""
            else:
                script_ext = ".sh"
                # Use shlex.join for proper quoting on Unix-like systems
                teo_cache_cmd_str = shlex.join(teo_cache_cmd)
                run_cmd_str = shlex.join(run_cmd)
                script_content = f"""#!/bin/bash
echo "Starting text encoder output caching..."
{teo_cache_cmd_str}
if [ $? -ne 0 ]; then
    echo "Text encoder caching failed with error code $?"
    exit $?
fi
echo "Text encoder caching completed successfully!"
echo "Starting training..."
{run_cmd_str}
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
                # minimum=100,  # Removed: Let backend handle validation
                step=100,
                interactive=True,
            )

            self.max_train_epochs = gr.Number(
                label="Max Training Epochs",
                info="[RECOMMENDED] 16 epochs for Qwen Image. Overrides max_train_steps. 1 epoch = full pass through dataset. 0 = use max_train_steps instead",
                value=self.config.get("max_train_epochs", 16),
                # minimum=0,  # Removed: Let backend handle validation
                # maximum=9999,
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

            self.gradient_checkpointing_cpu_offload = gr.Checkbox(
                label="Gradient Checkpointing CPU Offload",
                info="Offload activations to CPU when using gradient checkpointing. Reduces VRAM usage but slows training",
                value=self.config.get("gradient_checkpointing_cpu_offload", False),
            )

            self.gradient_accumulation_steps = gr.Number(
                label="Gradient Accumulation Steps",
                info="Simulate larger batch size. 1 = update every step, 4 = accumulate 4 steps then update. Useful for small VRAM",
                value=self.config.get("gradient_accumulation_steps", 1),
                minimum=0,
                maximum=32,
                step=1,
                interactive=True,
            )
        
        # Add full precision training options
        with gr.Row():
            self.full_bf16 = gr.Checkbox(
                label="Full BF16 Training",
                info="[EXPERIMENTAL] Stores gradients in BF16 instead of FP32. Saves ~30% VRAM but may cause training instability. Good for large batch sizes. Monitor loss carefully. Incompatible with mixed_precision='bf16'.",
                value=self.config.get("full_bf16", False),
            )
            
            self.full_fp16 = gr.Checkbox(
                label="Full FP16 Training",
                info="[EXPERIMENTAL] Stores gradients in FP16 instead of FP32. Saves ~30% VRAM but higher risk of gradient underflow. Use only if full_bf16 isn't available. Requires careful learning rate tuning. Incompatible with mixed_precision='fp16'.",
                value=self.config.get("full_fp16", False),
            )

        # Dtype compatibility warning
        with gr.Row():
            self.dtype_warning = gr.Markdown(
                "",
                visible=False,
                elem_classes=["warning-text"]
            )

        # Add validation function for dtype conflicts
        def validate_dtype_settings(full_bf16, full_fp16):
            warnings = []

            # Check full_bf16 and full_fp16 conflict
            if full_bf16 and full_fp16:
                warnings.append("⚠️ **CONFLICT**: Cannot use both full_bf16 and full_fp16 simultaneously. Choose one.")
                warnings.append("💡 **TIP**: Use full_bf16 for better stability, or full_fp16 only if BF16 is not available on your GPU.")

            if warnings:
                return gr.update(value="\n\n".join(warnings), visible=True)
            return gr.update(value="", visible=False)

        # Connect validation to relevant inputs
        for component in [self.full_bf16, self.full_fp16]:
            component.change(
                fn=validate_dtype_settings,
                inputs=[self.full_bf16, self.full_fp16],
                outputs=[self.dtype_warning]
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

        # These sample settings are moved to a dedicated section

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


class QwenImageSampleSettings:
    """Qwen Image specific sample generation settings"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()
    
    def initialize_ui_components(self):
        gr.Markdown("""
        ### Sample Generation Configuration
        Configure test image generation during training. Samples help monitor training progress and quality.
        
        **How it works:**
        - Provide a simple prompt file (one prompt per line)
        - GUI defaults below will be automatically added to prompts that don't specify parameters
        - An enhanced prompt file will be saved to your output directory for reference
        
        **Prompt File Format:**
        - Simple format: `A cat sitting` (will use GUI defaults below)
        - Advanced format: `A cat sitting --w 512 --h 512` (overrides GUI defaults)
        - Mixed: Some prompts can have parameters, others will use defaults
        """)
        
        # Basic sample settings
        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                info="Generate test images every N training steps. 0 = disable",
                value=self.config.get("sample_every_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )
            
            self.sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs", 
                info="Generate test images every N epochs. 0 = disable. Overrides sample_every_n_steps",
                value=self.config.get("sample_every_n_epochs", 0),
                minimum=0,
                step=1,
                interactive=True,
            )
            
            self.sample_at_first = gr.Checkbox(
                label="Sample at First",
                info="Generate test images before training starts to verify setup",
                value=self.config.get("sample_at_first", False),
            )
        
        # Sample prompts file
        with gr.Row():
            with gr.Column(scale=4):
                self.sample_prompts = gr.Textbox(
                    label="Sample Prompts File",
                    info="Path to text/TOML/JSON file with prompts. Required for sample generation",
                    placeholder="e.g., /path/to/prompts.txt",
                    value=self.config.get("sample_prompts", ""),
                )
            self.sample_prompts_button = gr.Button(
                "📂",
                size="sm",
                elem_id="sample_prompts_button"
            )
        
        # Custom output path for samples
        with gr.Row():
            with gr.Column(scale=4):
                self.sample_output_dir = gr.Textbox(
                    label="Custom Sample Output Directory",
                    info="Optional: Custom directory to save samples. If empty, uses output directory where model files will be saved",
                    placeholder="e.g., /path/to/sample/output",
                    value=self.config.get("sample_output_dir", ""),
                )
            self.sample_output_dir_button = gr.Button(
                "📂",
                size="sm",
                elem_id="sample_output_dir_button"
            )
        
        # Prompt enhancement control
        self.disable_prompt_enhancement = gr.Checkbox(
            label="Disable Automatic Prompt Enhancement",
            info="When enabled, uses original prompts without adding Kohya format parameters",
            value=self.config.get("disable_prompt_enhancement", False),
        )
        
        # Default sample parameters
        gr.Markdown("### Default Sample Parameters")
        gr.Markdown("These values will be automatically added to prompts that don't specify them")
        
        with gr.Row():
            self.sample_width = gr.Number(
                label="Default Width",
                info="Default width for samples (Qwen optimal: 1328)",
                value=self.config.get("sample_width", 1328),
                minimum=64,
                maximum=4096,
                step=8,
                interactive=True,
            )
            
            self.sample_height = gr.Number(
                label="Default Height",
                info="Default height for samples (Qwen optimal: 1328)",
                value=self.config.get("sample_height", 1328),
                minimum=64,
                maximum=4096,
                step=8,
                interactive=True,
            )
            
            self.sample_steps = gr.Number(
                label="Default Steps",
                info="Number of denoising steps",
                value=self.config.get("sample_steps", 20),
                # minimum=0,  # Removed: Let backend handle validation
                # maximum=100,
                step=1,
                interactive=True,
            )
        
        with gr.Row():
            self.sample_guidance_scale = gr.Number(
                label="Default Guidance",
                info="Guidance scale (higher = stronger prompt adherence)",
                value=self.config.get("sample_guidance_scale", 1.0),
                minimum=0.1,
                maximum=20.0,
                step=0.1,
                interactive=True,
            )
            
            self.sample_seed = gr.Number(
                label="Default Seed",
                info="Random seed (-1 = random each time)",
                value=self.config.get("sample_seed", -1),
                minimum=-1,
                maximum=2147483647,
                step=1,
                interactive=True,
            )
        
        with gr.Row():
            self.sample_discrete_flow_shift = gr.Number(
                label="Default Flow Shift",
                info="Discrete flow shift (0 = use training value, 2.2 = Qwen optimal)",
                value=self.config.get("sample_discrete_flow_shift", 0),
                minimum=0,
                maximum=10.0,
                step=0.1,
                interactive=True,
            )
            
            self.sample_cfg_scale = gr.Number(
                label="Default CFG Scale",
                info="CFG scale for negative prompt (1.0 = disabled)",
                value=self.config.get("sample_cfg_scale", 1.0),
                minimum=0.0,
                maximum=10.0,
                step=0.1,
                interactive=True,
            )
        
        with gr.Row():
            self.sample_negative_prompt = gr.Textbox(
                label="Default Negative Prompt",
                info="Default negative prompt for all samples (can be overridden per prompt)",
                placeholder="e.g., low quality, blurry, distorted",
                value=self.config.get("sample_negative_prompt", ""),
            )
        
        # File browser buttons
        self.sample_prompts_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".txt", extension_name="Text files"),
            outputs=[self.sample_prompts]
        )
        
        self.sample_output_dir_button.click(
            fn=lambda: get_folder_path(folder_path=""),
            outputs=[self.sample_output_dir]
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
                info="[RECOMMENDED] Use what preset has unless you are an expert.",
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
                info="[RECOMMENDED] Don't change what preset has unless you are an expert. Lower value may learn more details in longer training, higher value may learn faster but lower quality.",
                value=self.config.get("learning_rate", 5e-5),
                # minimum=1e-7,  # Removed: Let backend handle validation
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
        
        # Fused Backward Pass - now visible with smart auto-management
        with gr.Row():
            self.fused_backward_pass = gr.Checkbox(
                label="Fused Backward Pass [🎯 DreamBooth Only]",
                info="💾 [DreamBooth Exclusive] Advanced memory optimization that reduces VRAM usage during gradient computation by using fused operations with AdaFactor optimizer. 🎯 AUTOMATICALLY ENABLED for DreamBooth Fine-Tuning mode and DISABLED for LoRA mode (not effective). ⚠️ Requirements: Must use AdaFactor optimizer. Limitations: Disables gradient accumulation, max_grad_norm should be 0. VRAM Savings: Significant during backward pass.",
                value=self.config.get("fused_backward_pass", True),  # Default True (will be auto-managed)
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
                # minimum=0,  # Removed: Let backend handle validation
                # maximum=10,
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
            self.network_module = gr.Dropdown(
                label="Network Module (LoRA Architecture Type)",
                info="[ADVANCED] LoRA implementation module. 'networks.lora_qwen_image' is the standard Qwen Image LoRA. 'networks.dylora' enables Dynamic LoRA (trains multiple ranks). 'networks.lora_fa' uses LoRA-FA (Frozen-A) for better stability. Custom modules can be specified via text input below",
                choices=[
                    "networks.lora_qwen_image",
                    "networks.dylora",
                    "networks.lora_fa",
                    "custom"
                ],
                value=self.config.get("network_module", "") or "networks.lora_qwen_image",
                allow_custom_value=True,
                interactive=True,
            )

            self.custom_network_module = gr.Textbox(
                label="Custom Network Module Path",
                info="[EXPERT] Full Python module path for custom LoRA implementation. Only used when 'custom' is selected above. Example: 'my_networks.custom_lora_implementation'",
                placeholder="e.g., my_custom_module.QwenImageLoRA",
                value=self.config.get("custom_network_module", ""),
                visible=self.config.get("network_module", "networks.lora_qwen_image") == "custom",
                interactive=True,
            )

            self.network_dim = gr.Number(
                label="Network Dimension (LoRA Rank)",
                info="[CRITICAL] LoRA rank/dimension. QwenImage: 4-8 (low detail), 16-32 (balanced quality/size), 64-128 (high quality). DyLoRA: set to max rank you want to train. Higher = more VRAM + larger file size. 0 = auto-detection.",
                value=self.config.get("network_dim", 16),
                # minimum=0,  # Removed: Let backend handle validation  
                # maximum=512,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.network_alpha = gr.Number(
                label="Network Alpha (LoRA Alpha)",
                info="[CRITICAL] LoRA scaling factor. BEST PRACTICE: Set alpha = rank/2 for stability (e.g., rank 16 → alpha 8). Setting alpha = rank (e.g., 16/16) is common but may need lower learning rates. Lower alpha = stronger regularization",
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
                label="Network Arguments (Advanced LoRA Parameters)",
                info="[EXPERT] Space-separated key=value pairs. DyLoRA: 'unit=8' (rank increments). LoRA-FA: 'use_tucker=True tucker_rank=8'. LyCORIS: 'algo=locon conv_dim=4 conv_alpha=1'. Block-wise: 'block_dims=2,4,4,8,8,8,8,12,12,12,12,16,16,16,16' 'block_alphas=2,2,2,4,4,4,4,6,6,6,6,8,8,8,8'",
                placeholder='e.g. "conv_dim=4 conv_alpha=1 algo=locon" or "unit=8" for DyLoRA',
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

        # Add event handler for network module selection
        self.network_module.change(
            fn=lambda x: gr.update(visible=(x == "custom")),
            inputs=[self.network_module],
            outputs=[self.custom_network_module]
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
                    info="REQUIRED: Base filename for saved LoRA (WITHOUT .safetensors extension). Musubi will automatically add .safetensors. Example: 'my-qwen-lora' → 'my-qwen-lora.safetensors'",
                    placeholder="e.g., my-qwen-lora or character-style-v1",
                    value=self.config.get("output_name", "my-qwen-lora"),
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
                label="Keep Last N Checkpoints",
                info="Keep only last N checkpoint files (removes older ones). 0 = keep all, 3 = keep only last 3 checkpoint files",
                value=self.config.get("save_last_n_epochs", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.save_last_n_epochs_state = gr.Number(
                label="Keep Last N State Files",
                info="Keep last N optimizer state files (larger files). 0=keep all. Separate control for state files only",
                value=self.config.get("save_last_n_epochs_state", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.save_last_n_steps = gr.Number(
                label="Keep Last N Checkpoints (Alt)",
                info="Alternative checkpoint cleanup for step-based saves. Keep only last N checkpoint files. 0=keep all. Only active if 'Save Every N Steps' is used",
                value=self.config.get("save_last_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.save_last_n_steps_state = gr.Number(
                label="Keep Last N State Files (Alt)",
                info="Alternative state file cleanup for step-based saves. Keep last N state files. 0=keep all. Only active if 'Save Every N Steps' is used",
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
            
            self.mem_eff_save = gr.Checkbox(
                label="Memory Efficient Save",
                info="💾 [DreamBooth Only] Dramatically reduces RAM usage during checkpoint saving (from ~40GB to ~20GB). Essential for systems with limited RAM when doing DreamBooth fine-tuning. ⚠️ Only effective with DreamBooth mode, ignored for LoRA training. Note: Optimizer state saves still use normal method and require full RAM.",
                value=self.config.get("mem_eff_save", False),
            )

        # Conversion Options for Qwen Image LoRA
        with gr.Row():
            gr.Markdown("### 🔄 LoRA Format Conversion Options\n**Note**: These conversion features will be executed automatically after successful training completion. The trained LoRA will be converted to the selected formats for better compatibility with different inference tools.")

        with gr.Row():
            self.convert_to_diffusers = gr.Checkbox(
                label="Convert to Diffusers Format",
                info="[NEW] Automatically convert trained LoRA to Diffusers format after training. Creates additional files compatible with diffusers library. Useful for integration with other tools",
                value=self.config.get("convert_to_diffusers", False),
            )

            self.diffusers_output_dir = gr.Textbox(
                label="Diffusers Output Directory",
                info="Directory for converted Diffusers format files. If empty, uses '{output_dir}/diffusers_format'. Only used when conversion is enabled",
                placeholder="e.g., ./models/diffusers or leave empty for auto",
                value=self.config.get("diffusers_output_dir", ""),
                visible=self.config.get("convert_to_diffusers", False),
            )

        with gr.Row():
            self.convert_to_safetensors = gr.Checkbox(
                label="Convert to Safetensors (Alternative Format)",
                info="[ADVANCED] Convert LoRA to alternative safetensors format with different key naming. Useful for compatibility with certain inference tools",
                value=self.config.get("convert_to_safetensors", False),
            )

            self.safetensors_output_dir = gr.Textbox(
                label="Alternative Safetensors Output Directory",
                info="Directory for alternative safetensors format. If empty, uses '{output_dir}/safetensors_alt'. Only used when conversion is enabled",
                placeholder="e.g., ./models/safetensors_alt or leave empty",
                value=self.config.get("safetensors_output_dir", ""),
                visible=self.config.get("convert_to_safetensors", False),
            )

        # Event handlers for conversion options
        self.convert_to_diffusers.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[self.convert_to_diffusers],
            outputs=[self.diffusers_output_dir]
        )

        self.convert_to_safetensors.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[self.convert_to_safetensors],
            outputs=[self.safetensors_output_dir]
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
                # minimum=1,  # Removed: Let backend handle validation
                # maximum=64,
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
                info="Debug visualization for latent caching. 'None' disables display, 'image' saves debug images, 'console' prints debug info, 'video' for video models",
                choices=["None", "image", "console", "video"],
                allow_custom_value=True,
                value=self.config.get("caching_latent_debug_mode", "None"),
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
                # minimum=1,  # Removed: Let backend handle validation
                # maximum=128,
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

    # Add search functionality and unified toggle button
    with gr.Row():
        with gr.Column(scale=2):
            search_input = gr.Textbox(
                label="🔍 Search Settings",
                placeholder="Type to search and filter panels (e.g., 'block', 'learning', 'fp8', 'cache', 'epochs')",
                lines=1,
                interactive=True
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
    with accelerate_accordion, gr.Column():
        accelerate_launch = AccelerateLaunch(config=config)
        # Note: bf16 mixed precision is STRONGLY recommended for Qwen Image
        
    # Save Load Settings - moved before Model Settings for better workflow
    save_load_accordion = gr.Accordion("Save Models and Resume Training Settings", open=False, elem_classes="samples_background")
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
    
    # Add nested Torch Compile accordion to the main accordions list for toggle functionality
    accordions.append(qwen_model.torch_compile_accordion)
        
    caching_accordion = gr.Accordion("Caching", open=False, elem_classes="samples_background")
    accordions.append(caching_accordion)
    with caching_accordion:
        with gr.Tab("Latent caching"):
            qwenLatentCaching = QwenImageLatentCaching(headless=headless, config=config)
                
        with gr.Tab("Text encoder caching"):
            qwenTeoCaching = QwenImageTextEncoderOutputsCaching(headless=headless, config=config)
        
    optimizer_accordion = gr.Accordion("Learning Rate, Optimizer and Scheduler Settings", open=False, elem_classes="flux1_rank_layers_background")
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

    # New Sample Generation Settings section
    sample_accordion = gr.Accordion("Sample Generation Settings", open=False, elem_classes="samples_background")
    accordions.append(sample_accordion)
    with sample_accordion:
        sampleSettings = QwenImageSampleSettings(headless=headless, config=config)

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
        advanced_training.debug_mode,
        
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
        qwen_dataset.dataset_qwen_image_edit_control_resolution_width,
        qwen_dataset.dataset_qwen_image_edit_control_resolution_height,
        qwen_dataset.auto_generate_black_control_images,
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
        trainingSettings.gradient_checkpointing_cpu_offload,
        trainingSettings.gradient_accumulation_steps,
        trainingSettings.full_bf16,
        trainingSettings.full_fp16,
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
        sampleSettings.sample_every_n_steps,
        sampleSettings.sample_at_first,
        sampleSettings.sample_every_n_epochs,
        sampleSettings.sample_prompts,
        sampleSettings.sample_output_dir,
        sampleSettings.disable_prompt_enhancement,
        sampleSettings.sample_width,
        sampleSettings.sample_height,
        sampleSettings.sample_steps,
        sampleSettings.sample_guidance_scale,
        sampleSettings.sample_seed,
        sampleSettings.sample_discrete_flow_shift,
        sampleSettings.sample_cfg_scale,
        sampleSettings.sample_negative_prompt,
        
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
        OptimizerAndSchedulerSettings.fused_backward_pass,
        
        # Qwen Image model settings
        qwen_model.training_mode,  # Add training mode selection
        qwen_model.dit,
        qwen_model.dit_dtype,
        qwen_model.dit_in_channels,
        qwen_model.num_layers,
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
        qwen_model.edit_plus,
        qwen_model.faster_model_loading,
        qwen_model.use_pinned_memory_for_block_swap,
        
        # Torch Compile settings
        qwen_model.compile,
        qwen_model.compile_backend,
        qwen_model.compile_mode,
        qwen_model.compile_dynamic,
        qwen_model.compile_fullgraph,
        qwen_model.compile_cache_size_limit,
        
        qwen_model.timestep_sampling,
        qwen_model.discrete_flow_shift,
        qwen_model.flow_shift,
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
        network.custom_network_module,  # NEW parameter
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
        saveLoadSettings.convert_to_diffusers,  # NEW parameter
        saveLoadSettings.diffusers_output_dir,  # NEW parameter
        saveLoadSettings.convert_to_safetensors,  # NEW parameter
        saveLoadSettings.safetensors_output_dir,  # NEW parameter
        saveLoadSettings.mem_eff_save,
        
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
            toggle_all_btn_bottom = gr.Button(
                value="Open All Panels", 
                variant="secondary"
            )

    global executor
    executor = CommandExecutor(headless=headless)

    # Add handler for search functionality
    def search_settings(query):
        if not query or len(query.strip()) < 1:
            return gr.Row(visible=False), ""
        
        query_lower = query.lower().strip()
        results = []
        
        # Comprehensive parameter map with all actual parameters
        parameter_map = {
            # Model Settings - Main
            "dit": ("Qwen Image Model Settings", "DiT (Base Model) Checkpoint Path"),
            "vae": ("Qwen Image Model Settings", "VAE Checkpoint Path"),
            "text_encoder": ("Qwen Image Model Settings", "Text Encoder (Qwen2.5-VL) Path"),
            "dit_dtype": ("Qwen Image Model Settings", "DiT Computation Data Type"),
            "vae_dtype": ("Qwen Image Model Settings", "VAE Data Type"),
            "text_encoder_dtype": ("Qwen Image Model Settings", "Text Encoder Data Type"),
            "dit_in_channels": ("Qwen Image Model Settings", "DiT Input Channels"),
            "training_mode": ("Qwen Image Model Settings", "Training Mode (LoRA/DreamBooth)"),
            "edit": ("Qwen Image Model Settings", "Enable Qwen-Image-Edit Mode"),
            "edit_plus": ("Qwen Image Model Settings", "Enable Qwen-Image-Edit-2509 Mode"),
            "faster_model_loading": ("Qwen Image Model Settings", "Faster Model Loading (Uses more RAM but speeds up model loading speed - Enable for RunPod)"),
            
            # Torch Compile Settings
            "compile": ("Qwen Image Model Settings", "Enable torch.compile"),
            "torch_compile": ("Qwen Image Model Settings", "Enable torch.compile"),
            "compile_backend": ("Qwen Image Model Settings", "Compile Backend"),
            "compile_mode": ("Qwen Image Model Settings", "Compile Mode"),
            "compile_dynamic": ("Qwen Image Model Settings", "Dynamic Shapes"),
            "compile_fullgraph": ("Qwen Image Model Settings", "Fullgraph Mode"),
            "compile_cache_size_limit": ("Qwen Image Model Settings", "Cache Size Limit"),
            "inductor": ("Qwen Image Model Settings", "Compile Backend"),
            "triton": ("Qwen Image Model Settings", "Enable torch.compile"),
            
            # FP8 and Memory Settings
            "fp8_base": ("Qwen Image Model Settings", "FP8 for Base Model (DiT) (BF16 Model On The Fly Converted)"),
            "fp8_scaled": ("Qwen Image Model Settings", "Scaled FP8 for Base Model (DiT) (BF16 Model On The Fly Converted - Better FP8 Precision)"),
            "fp8_vl": ("Qwen Image Model Settings", "Use FP8 for Text Encoder"),
            "blocks_to_swap": ("Qwen Image Model Settings", "Blocks to Swap to CPU"),
            "blocks": ("Qwen Image Model Settings", "Blocks to Swap to CPU"),
            "swap": ("Qwen Image Model Settings", "Blocks to Swap to CPU"),
            "cpu": ("Qwen Image Model Settings", "Blocks to Swap to CPU"),
            
            # VAE Optimization
            "vae_tiling": ("Qwen Image Model Settings", "VAE Tiling"),
            "vae_chunk_size": ("Qwen Image Model Settings", "VAE Chunk Size"),
            "vae_spatial_tile_sample_min_size": ("Qwen Image Model Settings", "VAE Spatial Tile Min Size"),
            
            # Flow Matching and Guidance
            "guidance_scale": ("Qwen Image Model Settings", "Guidance Scale"),
            "img_in_txt_in_offloading": ("Qwen Image Model Settings", "Image-in-Text Input Offloading"),
            "timestep_sampling": ("Qwen Image Model Settings", "Timestep Sampling Method"),
            "discrete_flow_shift": ("Qwen Image Model Settings", "Discrete Flow Shift"),
            "flow_shift": ("Qwen Image Model Settings", "Flow Shift (Advanced)"),
            "weighting_scheme": ("Qwen Image Model Settings", "Weighting Scheme"),
            "logit_mean": ("Qwen Image Model Settings", "Logit Mean"),
            "logit_std": ("Qwen Image Model Settings", "Logit Std"),
            "mode_scale": ("Qwen Image Model Settings", "Mode Scale"),
            "sigmoid_scale": ("Qwen Image Model Settings", "Sigmoid Scale"),
            "min_timestep": ("Qwen Image Model Settings", "Min Timestep"),
            "max_timestep": ("Qwen Image Model Settings", "Max Timestep"),
            "preserve_distribution_shape": ("Qwen Image Model Settings", "Preserve Distribution Shape"),
            "num_timestep_buckets": ("Qwen Image Model Settings", "Number of Timestep Buckets"),
            
            # Dataset Settings
            "dataset_config": ("Qwen Image Training Dataset", "Dataset Config File"),
            "dataset_config_mode": ("Qwen Image Training Dataset", "Dataset Configuration Method"),
            "parent_folder_path": ("Qwen Image Training Dataset", "Parent Folder Path"),
            "dataset_resolution_width": ("Qwen Image Training Dataset", "Resolution Width"),
            "dataset_resolution_height": ("Qwen Image Training Dataset", "Resolution Height"),
            "dataset_caption_extension": ("Qwen Image Training Dataset", "Caption Extension"),
            "dataset_batch_size": ("Qwen Image Training Dataset", "Batch Size"),
            "create_missing_captions": ("Qwen Image Training Dataset", "Create Missing Captions"),
            "caption_strategy": ("Qwen Image Training Dataset", "Caption Strategy"),
            "dataset_enable_bucket": ("Qwen Image Training Dataset", "Enable Bucketing"),
            "dataset_bucket_no_upscale": ("Qwen Image Training Dataset", "Bucket No Upscale"),
            "dataset_cache_directory": ("Qwen Image Training Dataset", "Cache Directory Name"),
            "dataset_control_directory": ("Qwen Image Training Dataset", "Control Directory Name (Edit Mode)"),
            "dataset_qwen_image_edit_no_resize_control": ("Qwen Image Training Dataset", "Qwen Image Edit: No Resize Control"),
            "dataset_qwen_image_edit_control_resolution_width": ("Qwen Image Training Dataset", "Control Image Width"),
            "dataset_qwen_image_edit_control_resolution_height": ("Qwen Image Training Dataset", "Control Image Height"),
            "auto_generate_black_control_images": ("Qwen Image Training Dataset", "Auto Generate Black Control Images"),
            
            # Training Settings
            "sdpa": ("Training Settings", "Use SDPA for CrossAttention"),
            "flash_attn": ("Training Settings", "Use FlashAttention"),
            "sage_attn": ("Training Settings", "Use SageAttention"),
            "xformers": ("Training Settings", "Use xformers"),
            "flash3": ("Training Settings", "Use FlashAttention 3"),
            "split_attn": ("Training Settings", "Split Attention"),
            "max_train_steps": ("Training Settings", "Max Training Steps"),
            "max_train_epochs": ("Training Settings", "Max Training Epochs"),
            "max_data_loader_n_workers": ("Training Settings", "Max DataLoader Workers"),
            "persistent_data_loader_workers": ("Training Settings", "Persistent DataLoader Workers"),
            "seed": ("Training Settings", "Seed"),
            "gradient_checkpointing": ("Training Settings", "Gradient Checkpointing"),
            "gradient_checkpointing_cpu_offload": ("Training Settings", "Gradient Checkpointing CPU Offload"),
            "gradient_accumulation_steps": ("Training Settings", "Gradient Accumulation Steps"),
            "full_bf16": ("Training Settings", "Full BF16 Gradients"),
            "full_fp16": ("Training Settings", "Full FP16 Gradients"),
            
            # Optimizer Settings
            "optimizer_type": ("Learning Rate, Optimizer and Scheduler Settings", "Optimizer Type"),
            "learning_rate": ("Learning Rate, Optimizer and Scheduler Settings", "Learning Rate"),
            "optimizer_args": ("Learning Rate, Optimizer and Scheduler Settings", "Optimizer Arguments"),
            "max_grad_norm": ("Learning Rate, Optimizer and Scheduler Settings", "Max Gradient Norm"),
            "fused_backward_pass": ("Learning Rate, Optimizer and Scheduler Settings", "Fused Backward Pass"),
            "lr_scheduler": ("Learning Rate, Optimizer and Scheduler Settings", "Learning Rate Scheduler"),
            "lr_warmup_steps": ("Learning Rate, Optimizer and Scheduler Settings", "LR Warmup Steps"),
            "lr_decay_steps": ("Learning Rate, Optimizer and Scheduler Settings", "LR Decay Steps"),
            "lr_scheduler_num_cycles": ("Learning Rate, Optimizer and Scheduler Settings", "LR Scheduler Cycles"),
            "lr_scheduler_power": ("Learning Rate, Optimizer and Scheduler Settings", "LR Scheduler Power"),
            "lr_scheduler_timescale": ("Learning Rate, Optimizer and Scheduler Settings", "LR Scheduler Timescale"),
            "lr_scheduler_min_lr_ratio": ("Learning Rate, Optimizer and Scheduler Settings", "LR Scheduler Min LR Ratio"),
            
            # Network/LoRA Settings
            "network_module": ("LoRA Settings", "Network Module (LoRA Architecture Type)"),
            "custom_network_module": ("LoRA Settings", "Custom Network Module Path"),
            "network_dim": ("LoRA Settings", "Network Dimension (LoRA Rank)"),
            "network_alpha": ("LoRA Settings", "Network Alpha (LoRA Alpha)"),
            "network_dropout": ("LoRA Settings", "Network Dropout (LoRA Dropout)"),
            "network_args": ("LoRA Settings", "Network Arguments (Advanced LoRA Parameters)"),
            "network_weights": ("LoRA Settings", "Network Weights (LoRA Weights)"),
            "dim_from_weights": ("LoRA Settings", "Auto-Determine Rank from Weights"),
            "scale_weight_norms": ("LoRA Settings", "Scale Weight Norms"),
            "base_weights": ("LoRA Settings", "LoRA Base Weights"),
            "base_weights_multiplier": ("LoRA Settings", "Base Weights Multiplier"),
            "training_comment": ("LoRA Settings", "Training Comment"),
            "no_metadata": ("LoRA Settings", "No Metadata"),
            
            # Latent Caching
            "caching_latent_device": ("Caching → Latent caching", "Caching Device"),
            "caching_latent_batch_size": ("Caching → Latent caching", "Caching Batch Size"),
            "caching_latent_num_workers": ("Caching → Latent caching", "Data Loading Workers"),
            "caching_latent_skip_existing": ("Caching → Latent caching", "Skip Existing Cache Files"),
            "caching_latent_keep_cache": ("Caching → Latent caching", "Keep Cache Files"),
            "caching_latent_debug_mode": ("Caching → Latent caching", "Debug Mode"),
            
            # Text Encoder Caching
            "caching_teo_text_encoder": ("Caching → Text encoder caching", "Text Encoder Path"),
            "caching_teo_device": ("Caching → Text encoder caching", "Caching Device"),
            "caching_teo_fp8_vl": ("Caching → Text encoder caching", "Use FP8 for VL Model"),
            "caching_teo_batch_size": ("Caching → Text encoder caching", "Caching Batch Size"),
            "caching_teo_num_workers": ("Caching → Text encoder caching", "Data Loading Workers"),
            "caching_teo_skip_existing": ("Caching → Text encoder caching", "Skip Existing Cache Files"),
            "caching_teo_keep_cache": ("Caching → Text encoder caching", "Keep Cache Files"),
            
            # Save/Load Settings
            "output_dir": ("Save Models and Resume Training Settings", "Output Directory"),
            "output_name": ("Save Models and Resume Training Settings", "Output Name"),
            "resume": ("Save Models and Resume Training Settings", "Resume from State"),
            "save_every_n_epochs": ("Save Models and Resume Training Settings", "Save Every N Epochs"),
            "save_every_n_steps": ("Save Models and Resume Training Settings", "Save Every N Steps"),
            "save_last_n_epochs": ("Save Models and Resume Training Settings", "Save Last N Epochs"),
            "save_last_n_steps": ("Save Models and Resume Training Settings", "Save Last N Steps"),
            "save_state": ("Save Models and Resume Training Settings", "Save Optimizer States with Checkpoints"),
            "save_state_on_train_end": ("Save Models and Resume Training Settings", "Save State on Training End"),
            "convert_to_diffusers": ("Save Models and Resume Training Settings", "Convert to Diffusers Format"),
            "diffusers_output_dir": ("Save Models and Resume Training Settings", "Diffusers Output Directory"),
            "convert_to_safetensors": ("Save Models and Resume Training Settings", "Convert to Safetensors (Alternative Format)"),
            "safetensors_output_dir": ("Save Models and Resume Training Settings", "Alternative Safetensors Output Directory"),
            "mem_eff_save": ("Save Models and Resume Training Settings", "Memory Efficient Save"),
            
            # Sample Generation
            "sample_every_n_steps": ("Sample Generation Settings", "Sample Every N Steps"),
            "sample_every_n_epochs": ("Sample Generation Settings", "Sample Every N Epochs"),
            "sample_at_first": ("Sample Generation Settings", "Sample at First"),
            "sample_prompts": ("Sample Generation Settings", "Sample Prompts File"),
            "sample_width": ("Sample Generation Settings", "Default Width"),
            "sample_height": ("Sample Generation Settings", "Default Height"),
            "sample_steps": ("Sample Generation Settings", "Default Steps"),
            "sample_guidance_scale": ("Sample Generation Settings", "Default Guidance"),
            "sample_seed": ("Sample Generation Settings", "Default Seed"),
            "sample_discrete_flow_shift": ("Sample Generation Settings", "Default Flow Shift"),
            "sample_cfg_scale": ("Sample Generation Settings", "Default CFG Scale"),
            "sample_negative_prompt": ("Sample Generation Settings", "Default Negative Prompt"),
            
            # Accelerate Settings
            "mixed_precision": ("Accelerate launch Settings", "Mixed Precision"),
            "num_processes": ("Accelerate launch Settings", "Number of Processes"),
            "num_machines": ("Accelerate launch Settings", "Number of Machines"),
            "multi_gpu": ("Accelerate launch Settings", "Multi GPU"),
            "gpu_ids": ("Accelerate launch Settings", "GPU IDs"),
            "dynamo_backend": ("Accelerate launch Settings", "Dynamo Backend"),
        }
        
        # Search through parameter map
        for param, (location, display_name) in parameter_map.items():
            # More flexible matching
            if query_lower in param.lower() or query_lower in display_name.lower() or param.lower() in query_lower:
                score = 0
                # Exact match gets highest score
                if query_lower == param.lower():
                    score = 100
                # Starting with query gets high score
                elif param.lower().startswith(query_lower):
                    score = 80
                # Query in param name
                elif query_lower in param.lower():
                    score = 60
                # Query in display name
                elif query_lower in display_name.lower():
                    score = 40
                
                results.append((location, display_name, param, score))
        
        # Sort by score
        results.sort(key=lambda x: x[3], reverse=True)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for item in results:
            key = (item[0], item[1])  # Use location and display_name as key
            if key not in seen:
                seen.add(key)
                unique_results.append(item[:3])  # Remove score from final result
        
        if not unique_results:
            # No results found
            html = f"<div style='padding: 10px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;'>"
            html += f"<strong>No results found for '{query}'</strong><br>"
            html += f"<span style='color: #666; font-size: 0.9em;'>Try searching for: learning, optimizer, fp8, vram, epochs, batch, cache, sample</span>"
            html += "</div>"
            return gr.Row(visible=True), html
        
        # Format results as HTML
        html = f"<div style='padding: 10px; background: #f0f0f0; border-radius: 5px;'>"
        html += f"<strong>Found {len(unique_results)} result(s) for '{query}':</strong><br><br>"
        
        # Get unique panels
        unique_panels = set()
        for location, display_name, param in unique_results:
            panel_name = location.split('→')[0].strip()
            unique_panels.add(panel_name)
        
        html += f"<div style='margin-bottom: 10px; padding: 8px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 3px; color: #155724;'>"
        html += f"✅ <strong>Opening {len(unique_panels)} relevant panel{'s' if len(unique_panels) > 1 else ''}:</strong> "
        html += ", ".join(sorted(unique_panels))
        html += "</div>"
        
        for location, display_name, param in unique_results[:10]:  # Limit to 10 results
            html += f"<div style='margin-bottom: 8px; padding: 8px; background: white; border-radius: 3px; border-left: 3px solid #007bff;'>"
            html += f"📍 <strong>{location}</strong><br>"
            html += f"<span style='margin-left: 20px; color: #333;'>→ {display_name}</span>"
            html += f"</div>"
        
        if len(unique_results) > 10:
            html += f"<div style='color: #666; margin-top: 5px;'>... and {len(unique_results) - 10} more results</div>"
        
        html += "<div style='margin-top: 10px; padding: 8px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 3px;'>"
        html += "💡 <strong>Note:</strong> Only panels with matching settings are opened. Clear search to reset."
        html += "</div>"
        html += "</div>"
        
        return gr.Row(visible=True), html
    
    # Modified search functionality to open relevant panels
    def search_and_open_panels(query):
        if not query or len(query.strip()) < 1:
            # If search is cleared, close all panels
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            return [gr.Row(visible=False), "", gr.Button(value="Open All Panels"), gr.Button(value="Open All Panels"), "closed"] + accordion_states
        
        # Get search results (but we won't display them)
        results_row, results_html = search_settings(query)
        
        # Parse which panels should be opened based on search results
        panels_to_open = set()
        
        # Map panel names to their indices
        panel_map = {
            "Accelerate launch Settings": 0,
            "Save Models and Resume Training Settings": 1,
            "Qwen Image Training Dataset": 2,
            "Qwen Image Model Settings": 3,
            "Caching": 4,
            "Learning Rate, Optimizer and Scheduler Settings": 5,
            "Training Settings": 6,
            "Network Settings": 7,
            "Advanced Training Settings": 8,
            "Sample Generation Settings": 9,
            "Logging Settings": 10,
            "HuggingFace Settings": 11,
            "Metadata Settings": 12,
        }
        
        # Extract panel names from results HTML
        import re
        panel_pattern = r'📍 <strong>([^<]+)</strong>'
        matches = re.findall(panel_pattern, results_html)
        
        for match in matches:
            # Clean up the match and find the base panel name
            base_panel = match.split('→')[0].strip()
            if base_panel in panel_map:
                panels_to_open.add(panel_map[base_panel])
            # Handle special case for Caching with sub-tabs
            elif "Caching" in base_panel:
                panels_to_open.add(panel_map.get("Caching", 4))
        
        # Create accordion states - open only panels with results
        accordion_states = []
        for i in range(len(accordions)):
            if i in panels_to_open:
                accordion_states.append(gr.Accordion(open=True))
            else:
                accordion_states.append(gr.Accordion(open=False))
        
        # Update button text based on state
        if panels_to_open:
            button_text = f"Reset Search ({len(panels_to_open)} panel{'s' if len(panels_to_open) > 1 else ''} filtered)"
            state = "search"
        else:
            button_text = "Open All Panels"
            state = "closed"
        
        # Hide the results display - we just open the panels
        return [gr.Row(visible=False), "", gr.Button(value=button_text), gr.Button(value=button_text), state] + accordion_states
    
    # Connect search functionality with panel control
    search_input.change(
        search_and_open_panels,
        inputs=[search_input],
        outputs=[search_results_row, search_results, toggle_all_btn, toggle_all_btn_bottom, panels_state] + accordions,
        show_progress=False,
    )
    
    # Add handler for unified toggle button
    def toggle_all_panels(current_state):
        if current_state == "search":
            # If in search mode, close all panels and clear search
            new_state = "closed"
            new_button_text = "Open All Panels"
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            search_value = ""  # Clear search
            results_visibility = gr.Row(visible=False)
            results_content = ""
        elif current_state == "closed":
            # Open all panels and update button text
            new_state = "open"
            new_button_text = "Hide All Panels"
            accordion_states = [gr.Accordion(open=True) for _ in accordions]
            search_value = gr.Textbox(value="")  # Keep search as is
            results_visibility = gr.Row(visible=False)
            results_content = ""
        else:  # current_state == "open"
            # Close all panels and update button text
            new_state = "closed"
            new_button_text = "Open All Panels"
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            search_value = gr.Textbox(value="")  # Keep search as is
            results_visibility = gr.Row(visible=False)
            results_content = ""
        
        return [new_state, gr.Button(value=new_button_text), gr.Button(value=new_button_text), search_value, results_visibility, results_content] + accordion_states
    
    toggle_all_btn.click(
        toggle_all_panels,
        inputs=[panels_state],
        outputs=[panels_state, toggle_all_btn, toggle_all_btn_bottom, search_input, search_results_row, search_results] + accordions,
        show_progress=False,
    )

    # Add handler for bottom toggle button - same functionality as top button
    def toggle_all_panels_bottom(current_state):
        # Same logic as toggle_all_panels but returns both button states
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
    
    toggle_all_btn_bottom.click(
        toggle_all_panels_bottom,
        inputs=[panels_state],
        outputs=[panels_state, toggle_all_btn, toggle_all_btn_bottom, search_input, search_results_row, search_results] + accordions,
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