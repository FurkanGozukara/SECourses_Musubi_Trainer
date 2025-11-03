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
    get_dit_model_path,
    get_vae_model_path,
    get_text_encoder_path,
    get_clip_vision_path,
    print_command_and_toml,
    run_cmd_advanced_training,
    SaveConfigFile,
    SaveConfigFileToRun,
    scriptdir,
    setup_environment,
    manage_additional_parameters,
)
from .class_huggingface import HuggingFace
from .class_metadata import MetaData
from .custom_logging import setup_logging
from .dataset_config_generator import (
    generate_wan_dataset_config_from_folders,
    save_dataset_config,
    validate_dataset_config
)

log = setup_logging()

executor = None
huggingface = None
train_state_value = time.time()


def get_debug_parameters_for_mode(debug_mode: str) -> str:
    """
    Convert selected debug mode to command-line parameters.

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


class WanDataset:
    """Wan dataset configuration settings"""
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
                        info="Path to parent folder with subfolder(s) containing videos/images AND their caption files together. Simplest: Create one folder named 1_[name] (e.g., 1_character_videos). Captions MUST be in SAME folder as media!",
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
                    info="Width of training videos/images. Optimal resolutions: 960×960, 1280×720, 720×1280"
                )
                self.dataset_resolution_height = gr.Number(
                    label="Resolution Height",
                    value=self.config.get("dataset_resolution_height", 960),
                    minimum=64,
                    maximum=4096,
                    step=64,
                    info="Height of training videos/images. Optimal resolutions: 960×960, 1280×720, 720×1280"
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
                    info="Automatically create caption files for videos/images that don't have them"
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
                    info="Enable aspect ratio bucketing to train on videos/images with different aspect ratios"
                )
                self.dataset_bucket_no_upscale = gr.Checkbox(
                    label="Bucket No Upscale",
                    value=self.config.get("dataset_bucket_no_upscale", False),
                    info="Don't upscale videos/images when bucketing (maintains original size if smaller than target)"
                )
            
            with gr.Row():
                self.dataset_cache_directory = gr.Textbox(
                    label="Cache Directory Name",
                    value=self.config.get("dataset_cache_directory", "cache_dir"),
                    info="Cache folder name (relative) or full path (absolute). Each dataset gets its own cache directory to avoid conflicts"
                )
            
            # Video Frame Extraction Settings
            gr.Markdown("### 🎬 Video Frame Extraction Settings")
            gr.Markdown("Configure how frames are extracted from videos during training")
            
            with gr.Row():
                self.frame_extraction = gr.Dropdown(
                    label="Frame Extraction Method",
                    choices=["head", "chunk", "slide", "uniform", "full"],
                    value=self.config.get("frame_extraction", "head"),
                    info="""🎯 **head** (Recommended): Extract frames from the beginning of video
📦 **chunk**: Split video into non-overlapping chunks
🔄 **slide**: Sliding window extraction with overlap
📊 **uniform**: Sample frames uniformly across video
🎬 **full**: Use all available frames (up to max_frames)"""
                )
                self.frame_stride = gr.Number(
                    label="Frame Stride",
                    value=self.config.get("frame_stride", 1),
                    minimum=1,
                    maximum=100,
                    step=1,
                    info="Step size for sliding window extraction (only used with 'slide' method)"
                )
            
            with gr.Row():
                self.frame_sample = gr.Number(
                    label="Frame Sample Count",
                    value=self.config.get("frame_sample", 1),
                    minimum=1,
                    maximum=100,
                    step=1,
                    info="Number of samples to extract (only used with 'uniform' method)"
                )
                self.max_frames = gr.Number(
                    label="Maximum Frames",
                    value=self.config.get("max_frames", 129),
                    minimum=1,
                    maximum=1000,
                    step=1,
                    info="Maximum number of frames to extract from any video (used with 'full' method)"
                )
            
            with gr.Row():
                self.source_fps = gr.Number(
                    label="Source FPS (Optional)",
                    value=self.config.get("source_fps", None),
                    minimum=0.0,
                    maximum=120.0,
                    step=0.1,
                    info="Original video FPS for frame rate conversion. Set to 0 or leave empty for automatic detection"
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
                label="Dataset Generation Status",
                value="",
                info="Shows the status and results of dataset configuration generation",
                interactive=False,
                lines=6
            )
        
        # Dataset Preparation Details Section
        with gr.Accordion("📋 Dataset Preparation Details", open=False):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%); padding: 20px; border-radius: 12px; margin: 15px 0;">
                <h2 style="color: #ffffff; margin-top: 0; text-align: center;">🎯 Complete Wan Dataset Guide</h2>
                
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #d1d5db; margin-top: 0;">📏 Frame Count Rule: "N×4+1" Format</h3>
                    <p style="color: #9ca3af; margin: 8px 0;"><strong>✅ Valid Frame Counts:</strong> 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, <strong style="color: #d97706;">81 (most common)</strong>, 85, 89...</p>
                    <p style="color: #dc2626; margin: 8px 0;"><strong>❌ Invalid Counts:</strong> 20, 30, 50, 80, etc. → Will be automatically truncated to nearest valid count</p>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 20px 0;">
                    <div style="background: rgba(59, 130, 246, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                        <h4 style="color: #60a5fa; margin-top: 0;">🎬 81-Frame Video</h4>
                        <p style="color: #9ca3af; font-size: 14px; margin: 5px 0;"><strong>Input:</strong> 81-frame video (960×960, 16fps)</p>
                        <p style="color: #9ca3af; font-size: 14px; margin: 5px 0;"><strong>Processing:</strong></p>
                        <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                            <li>Loads all 81 frames as sequence</li>
                            <li>Latent: [1, 16, 81, 90, 160]</li>
                            <li>Learns full temporal patterns</li>
                            <li>Most comprehensive training</li>
                        </ul>
                        <p style="color: #10b981; font-size: 13px; margin: 5px 0;"><strong>Best for:</strong> T2V, I2V models requiring long video generation</p>
                    </div>
                    
                    <div style="background: rgba(168, 85, 247, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #a855f7;">
                        <h4 style="color: #c084fc; margin-top: 0;">📸 Single JPG Image</h4>
                        <p style="color: #9ca3af; font-size: 14px; margin: 5px 0;"><strong>Input:</strong> Single image (960×960)</p>
                        <p style="color: #9ca3af; font-size: 14px; margin: 5px 0;"><strong>Processing:</strong></p>
                        <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                            <li>Treated as 1-frame video</li>
                            <li>Latent: [1, 16, 1, 90, 160]</li>
                            <li>No temporal learning</li>
                            <li>Most efficient training</li>
                        </ul>
                        <p style="color: #10b981; font-size: 13px; margin: 5px 0;"><strong>Best for:</strong> T2I models, style training, fast training</p>
                    </div>
                    
                    <div style="background: rgba(245, 158, 11, 0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                        <h4 style="color: #fbbf24; margin-top: 0;">⚡ 17-Frame Video</h4>
                        <p style="color: #9ca3af; font-size: 14px; margin: 5px 0;"><strong>Input:</strong> 17-frame video (960×960, 16fps)</p>
                        <p style="color: #9ca3af; font-size: 14px; margin: 5px 0;"><strong>Processing:</strong></p>
                        <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                            <li>Uses all 17 frames</li>
                            <li>Latent: [1, 16, 17, 90, 160]</li>
                            <li>Limited temporal context</li>
                            <li>Balanced training approach</li>
                        </ul>
                        <p style="color: #10b981; font-size: 13px; margin: 5px 0;"><strong>Best for:</strong> Quick motions, transitions, efficient training</p>
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #ffffff; margin-top: 0;">📹 Video Specifications for Wan Models</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <p style="color: #fbbf24; margin: 8px 0;"><strong>🎬 Wan 2.1 & 2.2 Requirements:</strong></p>
                            <ul style="color: #6b7280; font-size: 14px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Frame Rate:</strong> 16 FPS recommended</li>
                                <li><strong>Frame Count:</strong> N×4+1 format (1, 5, 9, 17, 81...)</li>
                                <li><strong>Default Training:</strong> 81 frames (5.06 seconds)</li>
                                <li><strong>Auto Conversion:</strong> 30fps → 16fps (frames skipped)</li>
                            </ul>
                        </div>
                        <div>
                            <p style="color: #059669; margin: 8px 0;"><strong>📐 Supported Resolutions:</strong></p>
                            <ul style="color: #6b7280; font-size: 14px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Optimal:</strong> 960×960 (square), 1280×720 (landscape), 720×1280 (portrait)</li>
                                <li><strong>Standard:</strong> 720×1280, 1280×720</li>
                                <li><strong>Small:</strong> 480×832, 832×480</li>
                                <li><strong>Wan 2.2:</strong> 704×1280, 1280×704</li>
                                <li><strong>T2I Support:</strong> All resolutions + 1024×1024</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="background: rgba(34, 197, 94, 0.2); padding: 12px; border-radius: 6px; margin: 10px 0;">
                        <h4 style="color: #10b981; margin-top: 0;">🔧 Technical Architecture Details (Verified from Code)</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                            <div>
                                <p style="color: #fbbf24; margin: 5px 0;"><strong>🏗️ Model Architecture:</strong></p>
                                <ul style="color: #6b7280; font-size: 12px; margin: 5px 0; padding-left: 15px;">
                                    <li><strong>14B Models:</strong> 40 layers, 5120 dim</li>
                                    <li><strong>1.3B Models:</strong> 30 layers, 1536 dim</li>
                                    <li><strong>Patch Size:</strong> (1, 2, 2) temporal-spatial</li>
                                    <li><strong>VAE Stride:</strong> (4, 8, 8)</li>
                                </ul>
                            </div>
                            <div>
                                <p style="color: #7c3aed; margin: 5px 0;"><strong>🎯 Wan 2.2 Boundaries:</strong></p>
                                <ul style="color: #6b7280; font-size: 12px; margin: 5px 0; padding-left: 15px;">
                                    <li><strong>t2v-A14B:</strong> 0.875 (87.5%)</li>
                                    <li><strong>i2v-A14B:</strong> 0.900 (90%)</li>
                                    <li><strong>Wan 2.1:</strong> No boundary (single model)</li>
                                    <li><strong>Auto-detect:</strong> Uses model config values</li>
                                </ul>
                            </div>
                            <div>
                                <p style="color: #d97706; margin: 5px 0;"><strong>⚙️ Input Dimensions:</strong></p>
                                <ul style="color: #6b7280; font-size: 12px; margin: 5px 0; padding-left: 15px;">
                                    <li><strong>T2V Models:</strong> 16 channels</li>
                                    <li><strong>I2V Models:</strong> 36 channels (image+video)</li>
                                    <li><strong>Fun-Control:</strong> 48 channels (enhanced)</li>
                                    <li><strong>Latent Space:</strong> [B, C, T, H/8, W/8]</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                <div style="background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #ef4444;">
                    <h3 style="color: #f87171; margin-top: 0;">⚠️ Invalid Frame Count Handling</h3>
                    <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px; font-family: monospace; margin: 10px 0;">
                        <p style="color: #fbbf24; margin: 5px 0;">Example: 20-frame video</p>
                        <p style="color: #e5e7eb; margin: 5px 0;">Input: 20 frames → <span style="color: #f87171;">Automatically truncated to 17 frames</span></p>
                        <p style="color: #f87171; margin: 5px 0;">⚠️ Last 3 frames discarded (data loss!)</p>
                    </div>
                    <p style="color: #fbbf24; margin: 8px 0;"><strong>Prevention:</strong> Always use N×4+1 format to avoid truncation</p>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #ffffff; margin-top: 0;">📁 Dataset Structure (REQUIRED Format)</h3>
                    
                    <div style="background: rgba(239, 68, 68, 0.2); padding: 10px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #ef4444;">
                        <p style="color: #fbbf24; margin: 5px 0;"><strong>⚠️ IMPORTANT:</strong> Caption files MUST be in the SAME folder as media files with matching names!</p>
                        <p style="color: #d1d5db; font-size: 13px; margin: 5px 0;">❌ Separate folders (videos/ + captions/) are NOT supported by musubi backend</p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
                        <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #60a5fa; margin-top: 0;">🎬 Video Dataset Structure</h4>
                            <pre style="color: #d1d5db; font-size: 12px; margin: 0; line-height: 1.4;">training_data/
└── 1_ohwx man/
    ├── video1.mp4 (81 frames, 960×960, 16fps)
    ├── video1.txt
    ├── video2.mp4 (81 frames, 960×960, 16fps)
    ├── video2.txt
    ├── video3.mp4 (81 frames, 960×960, 16fps)
    ├── video3.txt
    └── cache_dir/ (auto-generated)</pre>
                            <p style="color: #10b981; font-size: 11px; margin: 5px 0;">✅ Simple: Just one folder with videos + captions together</p>
                        </div>
                        
                        <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #c084fc; margin-top: 0;">📸 Image Dataset Structure</h4>
                            <pre style="color: #d1d5db; font-size: 12px; margin: 0; line-height: 1.4;">training_data/
└── 1_ohwx style/
    ├── image1.jpg (960×960)
    ├── image1.txt
    ├── image2.jpg (960×960)
    ├── image2.txt
    ├── image3.jpg (960×960)
    ├── image3.txt
    └── cache_dir/ (auto-generated)</pre>
                            <p style="color: #10b981; font-size: 11px; margin: 5px 0;">✅ Simple: Just one folder with images + captions together</p>
                        </div>
                    </div>
                    
                    <div style="background: rgba(34, 197, 94, 0.2); padding: 10px; border-radius: 6px; margin: 10px 0;">
                        <p style="color: #4ade80; margin: 5px 0;"><strong>💡 Folder Name Format:</strong></p>
                        <ul style="color: #d1d5db; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                            <li><strong>Start with "1_"</strong> - Keep it simple! (e.g., <code style="background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 3px;">1_ohwx man</code>, <code style="background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 3px;">1_my_dataset</code>)</li>
                            <li>You can have <strong>just ONE folder</strong> or multiple folders if needed</li>
                            <li>The number before "_" is repeat count (use 1 for normal training, higher numbers to oversample specific data)</li>
                        </ul>
                    </div>
                    
                    <div style="background: rgba(168, 85, 247, 0.2); padding: 10px; border-radius: 6px; margin: 10px 0; border-left: 4px solid #a855f7;">
                        <p style="color: #c084fc; margin: 5px 0;"><strong>🎨 Mixed Datasets (Videos + Images):</strong></p>
                        <p style="color: #d1d5db; font-size: 13px; margin: 5px 0;">✅ You CAN mix videos and images, but use SEPARATE folders for each type:</p>
                        <pre style="color: #d1d5db; font-size: 11px; margin: 5px 0; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">training_data/
├── 1_ohwx videos/     ← Only videos here
│   ├── clip1.mp4
│   ├── clip1.txt
│   └── cache_dir/
└── 1_ohwx images/     ← Only images here
    ├── img1.jpg
    ├── img1.txt
    └── cache_dir/</pre>
                        <p style="color: #fbbf24; font-size: 12px; margin: 5px 0;">⚠️ Don't mix videos and images in the same folder - use separate folders!</p>
                    </div>
                    
                    <div style="background: rgba(59, 130, 246, 0.2); padding: 10px; border-radius: 6px; margin: 10px 0;">
                        <p style="color: #60a5fa; margin: 5px 0;"><strong>📂 Advanced: Multiple Datasets with Different Repeat Counts:</strong></p>
                        <pre style="color: #d1d5db; font-size: 11px; margin: 5px 0; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">training_data/
├── 3_main_character/  ← Repeat 3x (important videos)
├── 1_backgrounds/     ← Repeat 1x (normal images)
└── 5_key_scenes/      ← Repeat 5x (very important videos)</pre>
                    </div>
                </div>

                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #ffffff; margin-top: 0;">🎯 Model-Specific Dataset Recommendations</h3>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 15px 0;">
                        <div style="background: rgba(59, 130, 246, 0.2); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #60a5fa; margin-top: 0;">📹 T2V (Text-to-Video)</h4>
                            <ul style="color: #d1d5db; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Primary:</strong> Video datasets (81 frames)</li>
                                <li><strong>Secondary:</strong> Images (1 frame) for variety</li>
                                <li><strong>Best:</strong> Mixed dataset for versatility</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(168, 85, 247, 0.2); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #c084fc; margin-top: 0;">🖼️ I2V (Image-to-Video)</h4>
                            <ul style="color: #d1d5db; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Primary:</strong> Video datasets with first frames</li>
                                <li><strong>Requires:</strong> Control/reference images</li>
                                <li><strong>Structure:</strong> video + conditioning image</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(245, 158, 11, 0.2); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #fbbf24; margin-top: 0;">🎨 T2I (Text-to-Image)</h4>
                            <ul style="color: #d1d5db; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Primary:</strong> Image datasets (1 frame)</li>
                                <li><strong>Optional:</strong> Mix with video first frames</li>
                                <li><strong>Efficient:</strong> Memory efficient, fast training</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(16, 185, 129, 0.2); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #10b981; margin-top: 0;">🎬 FLF2V (First-Last-Frame)</h4>
                            <ul style="color: #d1d5db; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Primary:</strong> Video datasets (81 frames)</li>
                                <li><strong>Special:</strong> Uses first & last frames as guides</li>
                                <li><strong>Advanced:</strong> Interpolation training</li>
                            </ul>
                        </div>
                        
                        <div style="background: rgba(236, 72, 153, 0.2); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #ec4899; margin-top: 0;">🎮 Fun-Control Models</h4>
                            <ul style="color: #d1d5db; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Required:</strong> Video + control video pairs</li>
                                <li><strong>Structure:</strong> Matching control directory</li>
                                <li><strong>Advanced:</strong> Enhanced controllability</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div style="background: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #22c55e;">
                    <h3 style="color: #4ade80; margin-top: 0;">💡 Training Strategy Recommendations</h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <p style="color: #fbbf24; margin: 8px 0;"><strong>🎯 For Best Results:</strong></p>
                            <ul style="color: #d1d5db; font-size: 14px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Consistent Frames:</strong> Same count across dataset</li>
                                <li><strong>81 Frames:</strong> Optimal for full video capability</li>
                                <li><strong>Mixed Training:</strong> Images + videos for versatility</li>
                                <li><strong>Avoid Truncation:</strong> Use N×4+1 format only</li>
                            </ul>
                        </div>
                        <div>
                            <p style="color: #f87171; margin: 8px 0;"><strong>⚠️ Processing Considerations:</strong></p>
                            <ul style="color: #d1d5db; font-size: 14px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>81 frames:</strong> More computational load than shorter sequences</li>
                                <li><strong>1 frame:</strong> Lightest computational load</li>
                                <li><strong>Batch size:</strong> Usually 1 for video training</li>
                                <li><strong>Mixed batches:</strong> Different temporal dimensions processed separately</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px; margin: 15px 0;">
                    <h3 style="color: #ffffff; margin-top: 0;">🔄 Automatic Frame Processing</h3>
                    <p style="color: #e5e7eb; margin: 8px 0;"><strong>What happens during training:</strong></p>
                    <ol style="color: #d1d5db; font-size: 14px; margin: 5px 0; padding-left: 20px;">
                        <li>Videos extracted frame-by-frame (no additional processing)</li>
                        <li>Frame rate conversion: 30fps → 16fps (automatic frame skipping)</li>
                        <li>Frame count validation: Non-N×4+1 counts truncated to nearest valid</li>
                        <li>Latent encoding: Each sequence encoded to latent space</li>
                        <li>Temporal processing: Model learns patterns across time dimension</li>
                        <li>Loss calculation: Computed across entire temporal sequence</li>
                    </ol>
                    <p style="color: #10b981; margin: 8px 0;"><strong>💡 Debug Mode:</strong> Use <code style="background: rgba(0,0,0,0.5); padding: 2px 6px; border-radius: 3px;">--debug_mode video</code> during caching to preview processed frames</p>
                </div>

                <div style="background: rgba(16, 185, 129, 0.2); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #10b981;">
                    <h3 style="color: #10b981; margin-top: 0;">📁 Supported Model File Formats</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <p style="color: #fbbf24; margin: 8px 0;"><strong>✅ SafeTensors Support (.safetensors):</strong></p>
                            <ul style="color: #6b7280; font-size: 14px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>DiT Models:</strong> .safetensors (recommended)</li>
                                <li><strong>T5 Text Encoder:</strong> .safetensors (recommended) or .pth</li>
                                <li><strong>VAE Models:</strong> .safetensors or .pth</li>
                                <li><strong>CLIP Vision:</strong> .safetensors (recommended) or .pth</li>
                            </ul>
                        </div>
                        <div>
                            <p style="color: #7c3aed; margin: 8px 0;"><strong>🔧 VAE Compatibility:</strong></p>
                            <ul style="color: #6b7280; font-size: 14px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>ALL Models:</strong> Use Wan2.1_VAE.pth</li>
                                <li><strong>Wan 2.2 Advanced:</strong> Still uses Wan2.1_VAE.pth</li>
                                <li><strong>Wan2.2_VAE.pth:</strong> Only for 5B models (unsupported)</li>
                                <li><strong>Important:</strong> Don't mix VAE versions!</li>
                            </ul>
                        </div>
                    </div>
                </div>

                <div style="background: rgba(168, 85, 247, 0.2); padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #a855f7;">
                    <h3 style="color: #c084fc; margin-top: 0;">🧪 One Frame Training (Experimental Feature)</h3>
                    
                    <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px; margin: 10px 0;">
                        <h4 style="color: #fbbf24; margin-top: 0;">❓ What is One Frame Training?</h4>
                        <p style="color: #6b7280; font-size: 14px; margin: 5px 0;">A special mode that makes video models behave like image models for image-to-image transformations.</p>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                        <div>
                            <h4 style="color: #10b981; margin-top: 0;">✅ When to Use:</h4>
                            <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Image-to-image transformations</strong></li>
                                <li><strong>Style transfer</strong> with video models</li>
                                <li><strong>Image editing</strong> tasks</li>
                                <li><strong>Single image generation</strong> from video models</li>
                            </ul>
                            
                            <h4 style="color: #059669; margin-top: 15px;">📋 Requirements:</h4>
                            <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>I2V models</strong> (i2v-14B) for basic</li>
                                <li><strong>FLF2V models</strong> (flf2v-14B) for intermediate</li>
                                <li><strong>LoRA training</strong> (required for effectiveness)</li>
                                <li><strong>Special caching</strong> (--one_frame flag)</li>
                            </ul>
                        </div>
                        
                        <div>
                            <h4 style="color: #dc2626; margin-top: 0;">❌ When NOT to Use:</h4>
                            <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Standard video generation</strong></li>
                                <li><strong>Learning temporal patterns</strong></li>
                                <li><strong>Mixed video+image datasets</strong> (auto-handled)</li>
                                <li><strong>Normal T2V/I2V training</strong></li>
                            </ul>
                            
                            <h4 style="color: #d97706; margin-top: 15px;">⚠️ Important Notes:</h4>
                            <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                                <li><strong>Highly experimental</strong> feature</li>
                                <li><strong>Not officially supported</strong></li>
                                <li><strong>Requires special setup</strong></li>
                                <li><strong>Mixed datasets work fine</strong> without this</li>
                            </ul>
                        </div>
                    </div>

                    <div style="background: rgba(34, 197, 94, 0.2); padding: 12px; border-radius: 6px; margin: 10px 0;">
                        <h4 style="color: #10b981; margin-top: 0;">💡 Mixed Dataset Answer:</h4>
                        <p style="color: #6b7280; font-size: 14px; margin: 5px 0;"><strong>For mixed video + image datasets:</strong> You typically <strong>DON'T need</strong> One Frame Training. Normal training automatically handles:</p>
                        <ul style="color: #6b7280; font-size: 13px; margin: 5px 0; padding-left: 15px;">
                            <li>Videos → Processed as multi-frame sequences (e.g., 81 frames)</li>
                            <li>Images → Processed as 1-frame sequences automatically</li>
                            <li>Model learns both temporal (video) and static (image) patterns</li>
                        </ul>
                        <p style="color: #d97706; font-size: 14px; margin: 8px 0;"><strong>Use One Frame Training only for specialized image transformation tasks, not general mixed training.</strong></p>
                    </div>
                </div>
            </div>
            """)
        
        # Set up event handlers for mode switching
        self.dataset_config_mode.change(
            fn=self._toggle_dataset_mode,
            inputs=[self.dataset_config_mode],
            outputs=[self.toml_mode_row, self.folder_mode_column]
        )
        
        # Set up folder button click handler
        self.parent_folder_button.click(
            fn=lambda: get_folder_path(),
            outputs=[self.parent_folder_path]
        )
        
        # Set up caption strategy visibility
        self.create_missing_captions.change(
            fn=lambda x: gr.Dropdown(visible=x),
            inputs=[self.create_missing_captions],
            outputs=[self.caption_strategy]
        )

    def setup_dataset_ui_events(self, saveLoadSettings=None, wan_model_settings=None):
        """Setup event handlers for dataset configuration UI"""

        # Define generate dataset config function
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
            output_dir,  # Add output_dir parameter
            num_frames,  # Add num_frames parameter
            frame_extraction,
            frame_stride,
            frame_sample,
            max_frames,
            source_fps
        ):
            """Generate WAN dataset configuration from folder structure"""
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

                        # Create caption files for all media files (images and videos)
                        for file_name in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, file_name)
                            if os.path.isfile(file_path) and (
                                file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif',
                                                          '.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv'))
                            ):
                                caption_path = os.path.splitext(file_path)[0] + caption_ext
                                if not os.path.exists(caption_path):
                                    with open(caption_path, 'w', encoding='utf-8') as f:
                                        f.write(caption_text)

                # Generate the WAN dataset configuration
                config, messages = generate_wan_dataset_config_from_folders(
                    parent_folder=parent_folder,
                    resolution=(int(width), int(height)),
                    caption_extension=caption_ext,
                    create_missing_captions=False,  # We already created them above
                    caption_strategy="folder_name",
                    batch_size=int(batch_size),
                    enable_bucket=enable_bucket,
                    bucket_no_upscale=bucket_no_upscale,
                    cache_directory_name=cache_dir,
                    num_frames=int(num_frames),
                    frame_extraction=frame_extraction,
                    frame_stride=int(frame_stride),
                    frame_sample=int(frame_sample),
                    max_frames=int(max_frames),
                    source_fps=float(source_fps) if source_fps and source_fps > 0 else None
                )

                # Check if config generation was successful
                if not config or not config.get("datasets"):
                    return "", "", "[ERROR] Failed to generate configuration. Check your folder structure.\n" + "\n".join(messages)

                # Generate output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save in Output Directory if set, otherwise in dataset_tomls directory
                if output_dir and os.path.exists(output_dir):
                    output_path = os.path.join(output_dir, f"wan_dataset_config_{timestamp}.toml")
                else:
                    # Create dataset_tomls directory if it doesn't exist
                    dataset_tomls_dir = os.path.join(scriptdir, "dataset_tomls")
                    os.makedirs(dataset_tomls_dir, exist_ok=True)
                    output_path = os.path.join(dataset_tomls_dir, f"wan_dataset_config_{timestamp}.toml")

                # Save the configuration
                save_dataset_config(config, output_path)

                # Get dataset info for status message
                num_datasets = len(config.get("datasets", []))

                status_msg = f"[SUCCESS] Generated WAN dataset configuration:\n"
                status_msg += f"  Output: {output_path}\n"
                status_msg += f"  Datasets: {num_datasets}\n"
                status_msg += f"\n" + "\n".join(messages)

                if create_missing:
                    status_msg += f"\n  Caption files created with strategy: {caption_strat}"

                # Return both paths - output_path for dataset_config field and display
                return output_path, output_path, status_msg

            except Exception as e:
                error_msg = f"[ERROR] Failed to generate WAN dataset configuration:\n{str(e)}"
                log.error(error_msg)
                import traceback
                traceback.print_exc()
                return "", "", error_msg

        def copy_generated_path(generated_path):
            """Copy generated TOML path to dataset config field"""
            if generated_path:
                return generated_path
            return gr.update()

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
                        saveLoadSettings.output_dir,  # Pass output_dir
                        wan_model_settings.num_frames if wan_model_settings else 81,  # Pass num_frames
                        self.frame_extraction,
                        self.frame_stride,
                        self.frame_sample,
                        self.max_frames,
                        self.source_fps
                    ],
                    outputs=[self.dataset_config, self.generated_toml_path, self.dataset_status]
                )
            else:
                # Fallback without output_dir
                self.generate_toml_button.click(
                    fn=lambda *args: generate_dataset_config(*args, None, wan_model_settings.num_frames if wan_model_settings else 81, *[self.frame_extraction, self.frame_stride, self.frame_sample, self.max_frames, self.source_fps]),  # Pass all args + None for output_dir + num_frames + frame params
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

    def _toggle_dataset_mode(self, mode):
        """Toggle between TOML file mode and folder structure mode"""
        toml_visible = mode == "Use TOML File"
        folder_visible = mode == "Generate from Folder Structure"
        return gr.Row(visible=toml_visible), gr.Column(visible=folder_visible)


class WanModelSettings:
    """Wan model settings configuration"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        # Training Mode Selection
        with gr.Row():
            self.training_mode = gr.Radio(
                label="Training Mode",
                choices=["LoRA Training", "DreamBooth Fine-Tuning"],
                value=self.config.get("training_mode", "LoRA Training"),
                info="LoRA Training: Parameter-efficient, memory friendly | DreamBooth: Full fine-tuning, more memory intensive"
            )

        # Wan Model Selection
        with gr.Row():
            self.task = gr.Dropdown(
                label="Wan Model Type",
                choices=[
                    ("t2v-14B - Text-to-Video 14B (Wan 2.1 Standard) - 38 Max Block Swap", "t2v-14B"),
                    ("t2v-1.3B - Text-to-Video 1.3B (Wan 2.1 Faster/Smaller) - 28 Max Block Swap", "t2v-1.3B"),
                    ("i2v-14B - Image-to-Video 14B (Wan 2.1 Standard) - 38 Max Block Swap", "i2v-14B"),
                    ("t2i-14B - Text-to-Image 14B (Wan 2.1 Standard) - 38 Max Block Swap", "t2i-14B"),
                    ("flf2v-14B - First-Last-Frame-to-Video 14B (Wan 2.1) - 38 Max Block Swap", "flf2v-14B"),
                    ("t2v-1.3B-FC - Text-to-Video 1.3B Fun-Control (Wan 2.1) - 28 Max Block Swap", "t2v-1.3B-FC"),
                    ("t2v-14B-FC - Text-to-Video 14B Fun-Control (Wan 2.1) - 38 Max Block Swap", "t2v-14B-FC"),
                    ("i2v-14B-FC - Image-to-Video 14B Fun-Control (Wan 2.1) - 38 Max Block Swap", "i2v-14B-FC"),
                    ("t2v-A14B - Text-to-Video Advanced Dual-Model (Wan 2.2) - 38 Max Block Swap", "t2v-A14B"),
                    ("i2v-A14B - Image-to-Video Advanced Dual-Model (Wan 2.2) - 38 Max Block Swap", "i2v-A14B")
                ],
                value=self.config.get("task", "t2v-14B"),
                info="Choose your Wan model variant based on version, use case and hardware capabilities"
            )
            self.use_pinned_memory_for_block_swap = gr.Checkbox(
                label="Use Pinned Memory for Block Swapping (Faster on Windows - Requires more RAM)",
                info="Uses more system RAM but speeds up training. The speed up maybe significant depending on system settings. To work, go to Advanced Graphics settings in System > Display > Graphics as in tutorial video and disable Hardware-Accelerated GPU Scheduling and restart your PC. Only effective when blocks_to_swap > 0",
                value=self.config.get("use_pinned_memory_for_block_swap", False),
            )
        
        # Model Information Panel
        with gr.Row():
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: #ffffff; margin-top: 0;">🎯 Wan Model Guide</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; color: #e5e7eb;">
                    <div>
                        <strong style="color: #fbbf24;">📹 Wan 2.1 Standard Models:</strong><br>
                        • <strong>T2V:</strong> Text-to-Video generation<br>
                        • <strong>I2V:</strong> Image-to-Video generation<br>
                        • <strong>T2I:</strong> Text-to-Image generation<br>
                        • <strong>FLF2V:</strong> First-Last-Frame-to-Video
                    </div>
                    <div>
                        <strong style="color: #10b981;">🎮 Wan 2.1 Fun-Control:</strong><br>
                        • <strong>Enhanced Control:</strong> Advanced generation control<br>
                        • <strong>T2V-FC:</strong> Text-to-Video with Fun-Control<br>
                        • <strong>I2V-FC:</strong> Image-to-Video with Fun-Control<br>
                        • <strong>Better Precision:</strong> More controllable outputs
                    </div>
                    <div>
                        <strong style="color: #f472b6;">⚡ Wan 2.2 Advanced:</strong><br>
                        • <strong>Dual-Model System:</strong> High & Low noise models<br>
                        • <strong>Best Quality:</strong> State-of-the-art results<br>
                        • <strong>A14B Models:</strong> Advanced 14B parameters<br>
                        • <strong>Requires:</strong> High-end GPU for training
                    </div>
                    <div>
                        <strong style="color: #a78bfa;">💾 Hardware Requirements:</strong><br>
                        • <strong>1.3B models:</strong> Lower memory requirements<br>
                        • <strong>14B models:</strong> Moderate memory requirements<br>
                        • <strong>A14B models:</strong> Higher memory requirements<br>
                        • <strong>Fun-Control:</strong> Same as base model
                    </div>
                </div>
            </div>
            """)
        
        # Supported Resolutions Info
        with gr.Row():
            self.resolution_info = gr.HTML(
                value="<div style='background: #374151; padding: 10px; border-radius: 5px; color: #d1d5db;'><strong>📐 Supported Resolutions for t2v-14B:</strong> 720×1280 (portrait), 1280×720 (landscape), 480×832 (small portrait), 832×480 (small landscape)</div>",
                visible=True
            )

        # Model file paths - organized in pairs
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        self.dit = gr.Textbox(
                            label="DiT Model Path (Low Noise / Main Model)",
                            info="✨ REQUIRED: Path to main DiT checkpoint (.safetensors). For Wan 2.2 dual training, this serves as the LOW NOISE model (fine details). For single model training, this is the only model used.",
                            placeholder="e.g., /path/to/wan_t2v_14b.safetensors",
                            value=str(self.config.get("dit", "")), lines=3
                        )
                    with gr.Column(min_width=60):
                        self.dit_button = gr.Button("📂", size="md")
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        self.vae = gr.Textbox(
                            label="VAE Model Path",
                            info="REQUIRED: Use Wan2.1_VAE.pth for ALL models (including Wan 2.2 Advanced). Supports .pth and .safetensors formats.",
                            placeholder="e.g., /path/to/Wan2.1_VAE.pth",
                            value=str(self.config.get("vae", "")), lines=3
                        )
                    with gr.Column(min_width=60):
                        self.vae_button = gr.Button("📂", size="md")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        self.t5 = gr.Textbox(
                            label="T5 Text Encoder Path",
                            info="REQUIRED: Path to T5 text encoder. Supports both .safetensors and .pth formats (recommended: umt5-xxl-enc-bf16.safetensors)",
                            placeholder="e.g., /path/to/umt5-xxl-enc-bf16.safetensors",
                            value=str(self.config.get("t5", "")), lines=3
                        )
                    with gr.Column(min_width=60):
                        self.t5_button = gr.Button("📂", size="md")
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        self.clip = gr.Textbox(
                            label="CLIP Vision Model Path",
                            info="REQUIRED: Path to CLIP vision encoder. Supports both .safetensors and .pth formats (recommended: models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors)",
                            placeholder="e.g., /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                            value=str(self.config.get("clip", "")), lines=3
                        )
                    with gr.Column(min_width=60):
                        self.clip_button = gr.Button("📂", size="md")

        # Wan 2.2 Advanced Models Settings
        with gr.Row():
            gr.HTML("""
            <div style="background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%); padding: 15px; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: #ffffff; margin-top: 0;">🚀 Wan 2.2 Dual-Model System</h3>
                <div style="color: #e5e7eb; line-height: 1.5;">
                    <p style="margin: 5px 0;"><strong>How it works:</strong> Wan 2.2 Advanced models use TWO DiT models working together:</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 10px 0;">
                        <div>
                            <strong style="color: #fbbf24;">🔥 High Noise Model:</strong><br>
                            • Handles early denoising steps<br>
                            • Works on heavily noisy images<br>
                            • Rough structure generation
                        </div>
                        <div>
                            <strong style="color: #10b981;">✨ Low Noise Model (Main DiT):</strong><br>
                            • Handles final denoising steps<br>
                            • Works on nearly clean images<br>
                            • Fine detail refinement
                        </div>
                    </div>
                    <p style="margin: 5px 0;"><strong style="color: #f472b6;">Training Options:</strong></p>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li><strong>Single Model:</strong> Leave "High Noise DiT Path" empty (standard Wan 2.1 style)</li>
                        <li><strong>Dual Model:</strong> Provide both DiT paths (recommended for best Wan 2.2 quality)</li>
                        <li><strong>Timestep Boundary:</strong> Controls when to switch models (0.5 = switch at 50% denoising)</li>
                    </ul>
                </div>
            </div>
            """)

        with gr.Row():
            with gr.Column(scale=8):
                self.dit_high_noise = gr.Textbox(
                    label="High Noise DiT Path (Wan 2.2 Advanced Only)",
                    info="🔥 HIGH NOISE MODEL: Path to high noise DiT for t2v-A14B/i2v-A14B. Leave EMPTY for single-model training (Wan 2.1 style). Fill BOTH paths for dual-model training (best quality).",
                    placeholder="e.g., /path/to/wan_t2v_A14b_high_noise.safetensors",
                    value=str(self.config.get("dit_high_noise", ""))
                )
            with gr.Column(scale=1):
                self.dit_high_noise_button = gr.Button("📂", size="sm")

        with gr.Row():
            self.timestep_boundary = gr.Number(
                label="Timestep Boundary (Dual-Model Switch Point)",
                value=self.config.get("timestep_boundary", None),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                info="⚡ SWITCH POINT: When to switch from high→low noise model. Leave empty for auto-detect (uses model default), 0.3=switch at 30%, 0.5=switch at 50%. Only used when High Noise DiT path is provided."
            )
            self.offload_inactive_dit = gr.Checkbox(
                label="Offload Inactive DiT to CPU",
                value=self.config.get("offload_inactive_dit", False),
                info="💾 MEMORY SAVER: Move inactive model to CPU during dual training. Reduces memory usage but adds model swapping time. Useful for systems with limited GPU memory."
            )

        # Data Types and Memory Settings
        with gr.Row():
            self.dit_dtype = gr.Dropdown(
                label="DiT Data Type",
                choices=["bfloat16", "float16"],
                value=self.config.get("dit_dtype", "bfloat16"),
                info="bfloat16=best quality (recommended), float16=faster but may be unstable"
            )
            self.text_encoder_dtype = gr.Dropdown(
                label="Text Encoder Data Type",
                choices=["bfloat16", "float16"],
                value=self.config.get("text_encoder_dtype", "bfloat16"),
                info="Data type for T5 text encoder"
            )

        with gr.Row():
            self.vae_dtype = gr.Dropdown(
                label="VAE Data Type",
                choices=["bfloat16", "float16"],
                value=self.config.get("vae_dtype", "bfloat16"),
                info="Data type for VAE model"
            )
            self.clip_vision_dtype = gr.Dropdown(
                label="CLIP Vision Data Type",
                choices=["bfloat16", "float16"],
                value=self.config.get("clip_vision_dtype", "bfloat16"),
                info="Data type for CLIP vision encoder"
            )

        # Memory Optimization
        with gr.Row():
            self.fp8_base = gr.Checkbox(
                label="FP8 Base Model",
                value=self.config.get("fp8_base", False),
                info="Enable FP8 for DiT model to reduce memory usage (requires fp8_scaled=true)"
            )
            self.fp8_scaled = gr.Checkbox(
                label="FP8 Scaled",
                value=self.config.get("fp8_scaled", False),
                info="REQUIRED when fp8_base=true, provides better quality than standard FP8"
            )

        with gr.Row():
            self.fp8_t5 = gr.Checkbox(
                label="FP8 T5 Text Encoder",
                value=self.config.get("fp8_t5", False),
                info="Enable FP8 for T5 text encoder to reduce memory usage"
            )
            self.blocks_to_swap = gr.Number(
                label="Blocks to Swap",
                value=self.config.get("blocks_to_swap", 0),
                minimum=0,
                maximum=59,
                step=1,
                info="Number of transformer blocks to swap to CPU (0=disabled, higher values save GPU memory but require more system RAM)"
            )

        # VAE Optimization
        with gr.Row():
            self.vae_tiling = gr.Checkbox(
                label="VAE Tiling",
                value=self.config.get("vae_tiling", False),
                info="Enable spatial tiling to reduce memory usage during VAE operations"
            )
            self.vae_chunk_size = gr.Number(
                label="VAE Chunk Size",
                value=self.config.get("vae_chunk_size", 0),
                minimum=0,
                step=1,
                info="0=auto/disabled. Higher=faster but more memory usage"
            )

        # Additional Wan-specific options
        with gr.Row():
            self.vae_cache_cpu = gr.Checkbox(
                label="Cache VAE Features on CPU",
                value=self.config.get("vae_cache_cpu", False),
                info="Cache VAE features on CPU to reduce GPU memory usage"
            )
            self.force_v2_1_time_embedding = gr.Checkbox(
                label="Force Wan 2.1 Time Embedding",
                value=self.config.get("force_v2_1_time_embedding", False),
                info="Force using 2.1 style time embedding even for Wan 2.2 models"
            )

        # Video-specific Settings
        with gr.Row():
            self.num_frames = gr.Number(
                label="Number of Frames",
                value=self.config.get("num_frames", 81),
                minimum=1,
                maximum=200,
                step=1,
                info="Number of frames for video training (81 is default for Wan models)"
            )
            self.one_frame = gr.Checkbox(
                label="One Frame Training (Experimental)",
                value=self.config.get("one_frame", False),
                info="🧪 EXPERIMENTAL: Enable for image-to-image transformations using video models. NOT needed for mixed video+image datasets. Requires I2V/FLF2V models + LoRA training. See Dataset Preparation Details for full explanation."
            )

        # Set up file browser button handlers
        self.dit_button.click(fn=lambda: get_dit_model_path(), outputs=[self.dit])
        self.vae_button.click(fn=lambda: get_vae_model_path(), outputs=[self.vae])
        self.t5_button.click(fn=lambda: get_text_encoder_path(), outputs=[self.t5])
        self.clip_button.click(fn=lambda: get_clip_vision_path(), outputs=[self.clip])
        self.dit_high_noise_button.click(fn=lambda: get_dit_model_path(), outputs=[self.dit_high_noise])

        # Set up conditional visibility for FP8 settings
        self.fp8_base.change(
            fn=lambda x: gr.Checkbox(value=x if x else False),
            inputs=[self.fp8_base],
            outputs=[self.fp8_scaled]
        )
        
        # Set up dynamic resolution info based on selected model
        self.task.change(
            fn=self._update_resolution_info,
            inputs=[self.task],
            outputs=[self.resolution_info]
        )
        
        # Set up dynamic visibility for Wan 2.2 settings
        self.task.change(
            fn=self._update_wan22_visibility,
            inputs=[self.task],
            outputs=[self.dit_high_noise, self.timestep_boundary, self.offload_inactive_dit]
        )

    def _update_resolution_info(self, task):
        """Update resolution information based on selected Wan model"""
        # Based on actual SUPPORTED_SIZES from musubi-tuner/src/musubi_tuner/wan/configs/__init__.py
        resolution_map = {
            "t2v-14B": "720×1280 (portrait), 1280×720 (landscape), 480×832 (small portrait), 832×480 (small landscape) | Architecture: 40 layers, 5120 dim",
            "t2v-1.3B": "480×832 (small portrait), 832×480 (small landscape) | Architecture: 30 layers, 1536 dim",
            "i2v-14B": "720×1280 (portrait), 1280×720 (landscape), 480×832 (small portrait), 832×480 (small landscape) | Architecture: 40 layers, 5120 dim, 36 input channels",
            "t2i-14B": "720×1280, 1280×720, 480×832, 832×480, 1024×1024 (square) | All SIZE_CONFIGS supported | Architecture: 40 layers, 5120 dim",
            "flf2v-14B": "720×1280 (portrait), 1280×720 (landscape), 480×832 (small portrait), 832×480 (small landscape) | First-Last-Frame mode",
            "t2v-1.3B-FC": "480×832 (small portrait), 832×480 (small landscape) | Fun-Control: 48 input channels",
            "t2v-14B-FC": "720×1280 (portrait), 1280×720 (landscape), 480×832 (small portrait), 832×480 (small landscape) | Fun-Control: 48 input channels",
            "i2v-14B-FC": "720×1280 (portrait), 1280×720 (landscape), 480×832 (small portrait), 832×480 (small landscape) | I2V + Fun-Control",
            "t2v-A14B": "720×1280, 1280×720, 480×832, 832×480 | Wan 2.2 Advanced | Boundary: 0.875 (87.5%) | Dual-model system",
            "i2v-A14B": "720×1280, 1280×720, 480×832, 832×480 | Wan 2.2 Advanced | Boundary: 0.900 (90%) | Dual-model system"
        }
        
        resolutions = resolution_map.get(task, "720×1280 (portrait), 1280×720 (landscape)")
        
        return gr.HTML(
            value=f"<div style='background: #374151; padding: 10px; border-radius: 5px; color: #d1d5db;'><strong>📐 {task} Details:</strong> {resolutions}</div>"
        )

    def _update_wan22_visibility(self, task):
        """Update Wan 2.2 settings visibility based on selected model"""
        is_wan22 = task in ["t2v-A14B", "i2v-A14B"]
        
        if is_wan22:
            # Show Wan 2.2 settings with enhanced info
            dit_high_noise = gr.Textbox(
                label="High Noise DiT Path (Wan 2.2 Advanced Only)",
                info="🔥 HIGH NOISE MODEL: Path to high noise DiT for t2v-A14B/i2v-A14B. Leave EMPTY for single-model training (Wan 2.1 style). Fill BOTH paths for dual-model training (best quality).",
                placeholder="e.g., /path/to/wan_t2v_A14b_high_noise.safetensors",
                interactive=True,
                visible=True
            )
            timestep_boundary = gr.Number(
                label="Timestep Boundary (Dual-Model Switch Point)",
                info="⚡ SWITCH POINT: When to switch from high→low noise model. 0.0=auto-detect, 0.3=switch at 30%, 0.5=switch at 50%. Only used when High Noise DiT path is provided.",
                interactive=True,
                visible=True
            )
            offload_inactive_dit = gr.Checkbox(
                label="Offload Inactive DiT to CPU",
                info="💾 MEMORY SAVER: Move inactive model to CPU during dual training. Reduces memory usage but adds model swapping time. Useful for systems with limited GPU memory.",
                interactive=True,
                visible=True
            )
        else:
            # Hide/disable Wan 2.2 settings for non-A14B models
            dit_high_noise = gr.Textbox(
                label="High Noise DiT Path (Not Available for this model)",
                info="❌ This model doesn't support dual-model training. Only Wan 2.2 Advanced models (t2v-A14B, i2v-A14B) support the dual-model system.",
                placeholder="Not available for this model type",
                interactive=False,
                visible=True
            )
            timestep_boundary = gr.Number(
                label="Timestep Boundary (Not Available)",
                info="❌ Only available for Wan 2.2 Advanced models (t2v-A14B, i2v-A14B)",
                interactive=False,
                visible=True
            )
            offload_inactive_dit = gr.Checkbox(
                label="Offload Inactive DiT (Not Available)",
                info="❌ Only needed for dual-model training (Wan 2.2 Advanced models)",
                interactive=False,
                visible=True
            )
        
        return dit_high_noise, timestep_boundary, offload_inactive_dit


class WanSampleSettings:
    """Wan video specific sample generation settings"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.initialize_ui_components()

    def initialize_ui_components(self):
        gr.Markdown("""
        ### Sample Generation Configuration
        Configure test video generation during training. Samples help monitor training progress and quality.

        **How it works:**
        - Provide a simple prompt file (one prompt per line)
        - GUI defaults below will be automatically added to prompts that don't specify parameters
        - An enhanced prompt file will be saved to your output directory for reference

        **Prompt File Format:**
        - Simple format: `A cat playing in the garden` (will use GUI defaults below)
        - Advanced format: `A cat playing in the garden --w 960 --h 960 --f 81` (overrides GUI defaults)
        - Mixed: Some prompts can have parameters, others will use defaults
        """)

        # Basic sample settings
        with gr.Row():
            self.sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                info="Generate test videos every N training steps. 0 = disable",
                value=self.config.get("sample_every_n_steps", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs",
                info="Generate test videos every N epochs. 0 = disable. Overrides sample_every_n_steps",
                value=self.config.get("sample_every_n_epochs", 0),
                minimum=0,
                step=1,
                interactive=True,
            )

            self.sample_at_first = gr.Checkbox(
                label="Sample at First",
                info="Generate test videos before training starts to verify setup",
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
                info="Default width for samples (Wan optimal: 960, 1280, 720)",
                value=self.config.get("sample_width", 960),
                minimum=64,
                maximum=4096,
                step=64,
                interactive=True,
            )

            self.sample_height = gr.Number(
                label="Default Height",
                info="Default height for samples (Wan optimal: 960, 720, 1280)",
                value=self.config.get("sample_height", 960),
                minimum=64,
                maximum=4096,
                step=64,
                interactive=True,
            )

            self.sample_num_frames = gr.Number(
                label="Default Number of Frames",
                info="Number of frames in sample videos (Wan optimal: 81)",
                value=self.config.get("sample_num_frames", 81),
                minimum=1,
                maximum=200,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.sample_steps = gr.Number(
                label="Default Steps",
                info="Number of inference steps for sample generation",
                value=self.config.get("sample_steps", 20),
                minimum=1,
                maximum=100,
                step=1,
                interactive=True,
            )

            self.sample_guidance_scale = gr.Number(
                label="Default Guidance Scale",
                info="Guidance scale for sample generation (higher = stronger prompt adherence, Wan optimal: 7.0)",
                value=self.config.get("sample_guidance_scale", 7.0),
                minimum=1.0,
                maximum=20.0,
                step=0.1,
                interactive=True,
            )

            self.sample_seed = gr.Number(
                label="Default Seed",
                info="Random seed (-1 = random each time)",
                value=self.config.get("sample_seed", 99),
                minimum=-1,
                step=1,
                interactive=True,
            )

        with gr.Row():
            self.sample_negative_prompt = gr.Textbox(
                label="Sample Negative Prompt",
                info="Negative prompt for sample generation",
                value=self.config.get("sample_negative_prompt", ""),
                placeholder="e.g., blurry, low quality, distorted",
            )
        
        # Set up button handlers
        self.sample_prompts_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".txt", extension_name="Text files"),
            outputs=[self.sample_prompts]
        )
        
        self.sample_output_dir_button.click(
            fn=lambda: get_folder_path(folder_path=""),
            outputs=[self.sample_output_dir]
        )


def generate_enhanced_prompt_file(
    original_prompt_file: str,
    output_dir: str,
    output_name: str,
    sample_width: int = 960,
    sample_height: int = 960,
    sample_num_frames: int = 81,
    sample_steps: int = 20,
    sample_guidance_scale: float = 7.0,
    sample_seed: int = 99,
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
        sample_num_frames: Default number of frames for samples
        sample_steps: Default number of denoising steps
        sample_guidance_scale: Default guidance scale
        sample_seed: Default seed (99 for WAN, -1 would be random)
        sample_scheduler: Default scheduler
        sample_negative_prompt: Default negative prompt

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
            has_frames = '--f ' in line or '-f ' in line
            has_steps = '--s ' in line or '-s ' in line
            has_guidance = '--g ' in line or '-g ' in line
            has_seed = '--d ' in line or '-d ' in line
            has_negative = '--n ' in line or '-n ' in line
            has_one_frame = '--of ' in line or '-of ' in line

            # Build enhanced line with defaults for missing parameters
            enhanced_line = line

            # Add width and height if not present
            if not has_width:
                enhanced_line += f" --w {sample_width}"
            if not has_height:
                enhanced_line += f" --h {sample_height}"

            # Add frames if not present
            if not has_frames:
                enhanced_line += f" --f {sample_num_frames}"

            # Add steps if not present
            if not has_steps:
                enhanced_line += f" --s {sample_steps}"

            # Add guidance if not present
            if not has_guidance:
                enhanced_line += f" --g {sample_guidance_scale}"

            # Add negative prompt if not present and not empty
            if not has_negative and sample_negative_prompt and sample_negative_prompt.strip():
                enhanced_line += f" --n {sample_negative_prompt}"

            # Add seed if not present
            if not has_seed:
                enhanced_line += f" --d {sample_seed}"

            # For 1-frame generation, add one_frame parameter if not present
            # This is required for WAN 2.1 single frame (image) generation
            if sample_num_frames == 1 and not has_one_frame:
                enhanced_line += f" --of target_index=1,control_index=0"

            enhanced_lines.append(enhanced_line)

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        enhanced_filename = f"{output_name}_enhanced_prompts_{timestamp}.txt"
        enhanced_path = os.path.join(output_dir, enhanced_filename)

        # Write enhanced file
        with open(enhanced_path, 'w', encoding='utf-8') as f:
            for line in enhanced_lines:
                f.write(line + '\n')

        log.info(f"Enhanced prompt file created: {enhanced_path}")
        return enhanced_path

    except Exception as e:
        log.error(f"Error generating enhanced prompt file: {e}")
        return None


class WanSaveLoadSettings(SaveLoadSettings):
    """Wan-specific save/load settings that extend the base SaveLoadSettings"""
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        super().__init__(headless, config)
        # Override default output name for Wan models
        if hasattr(self, 'output_name'):
            self.output_name.value = self.config.get("output_name", "my-wan-lora")


def wan_gui_actions(
    action: str,
    config_file_name: str,
    headless: bool,
    print_only: bool,
    *args,
):
    """Handle GUI actions for Wan training"""
    log.info(f"Wan GUI Action: {action}")
    
    if action == "open_configuration":
        return open_wan_configuration(True, config_file_name, list(zip(
            [
                # accelerate_launch
                "mixed_precision", "num_cpu_threads_per_process", "num_processes", "num_machines", "multi_gpu", "gpu_ids",
                "main_process_port", "dynamo_backend", "dynamo_mode", "dynamo_use_fullgraph", "dynamo_use_dynamic", "extra_accelerate_launch_args",
                # advanced_training
                "additional_parameters", "debug_mode",
                # Wan Dataset settings
                "dataset_config_mode", "dataset_config", "parent_folder_path", "dataset_resolution_width",
                "dataset_resolution_height", "dataset_caption_extension", "dataset_batch_size",
                "create_missing_captions", "caption_strategy", "dataset_enable_bucket",
                "dataset_bucket_no_upscale", "dataset_cache_directory", "generated_toml_path",
                # Wan Model settings
                "training_mode", "task", "dit", "vae", "t5", "clip",
                "dit_high_noise", "timestep_boundary", "offload_inactive_dit", "dit_dtype",
                "text_encoder_dtype", "vae_dtype", "clip_vision_dtype", "fp8_base", "fp8_scaled",
                "fp8_t5", "blocks_to_swap", "use_pinned_memory_for_block_swap", "vae_tiling", "vae_chunk_size", "vae_cache_cpu", "num_frames", "one_frame", "force_v2_1_time_embedding",
                # training_settings
                "sdpa", "flash_attn", "sage_attn", "xformers", "split_attn", "max_train_steps", "max_train_epochs",
                "max_data_loader_n_workers", "persistent_data_loader_workers", "seed", "gradient_checkpointing",
                "gradient_checkpointing_cpu_offload", "gradient_accumulation_steps", "full_bf16", "full_fp16",
                "logging_dir", "log_with", "log_prefix", "log_tracker_name", "wandb_run_name", "log_tracker_config",
                "wandb_api_key", "log_config", "ddp_timeout", "ddp_gradient_as_bucket_view", "ddp_static_graph",
                # Sample generation settings (from TrainingSettings)
                "sample_every_n_steps", "sample_every_n_epochs", "sample_at_first", "sample_prompts", "sample_output_dir",
                "disable_prompt_enhancement", "sample_width", "sample_height", "sample_num_frames", "sample_steps",
                "sample_guidance_scale", "sample_seed", "sample_negative_prompt",
                # Latent Caching Settings
                "caching_latent_device", "caching_latent_batch_size", "caching_latent_num_workers", "caching_latent_skip_existing",
                "caching_latent_keep_cache", "caching_latent_debug_mode", "caching_latent_console_width",
                "caching_latent_console_back", "caching_latent_console_num_images",
                # Text Encoder Outputs Caching Settings
                "caching_teo_text_encoder1", "caching_teo_text_encoder2", "caching_teo_text_encoder_dtype", "caching_teo_device",
                "caching_teo_fp8_llm", "caching_teo_batch_size", "caching_teo_num_workers", "caching_teo_skip_existing",
                "caching_teo_keep_cache",
                # Optimizer and Scheduler Settings
                "optimizer_type", "optimizer_args", "learning_rate", "max_grad_norm", "lr_scheduler", "lr_warmup_steps",
                "lr_decay_steps", "lr_scheduler_num_cycles", "lr_scheduler_power", "lr_scheduler_timescale",
                "lr_scheduler_min_lr_ratio", "lr_scheduler_type", "lr_scheduler_args",
                # Network Settings
                "no_metadata", "network_weights", "network_module", "network_dim", "network_alpha",
                "network_dropout", "network_args", "training_comment", "dim_from_weights", "scale_weight_norms",
                "base_weights", "base_weights_multiplier",
                # Save/Load Settings
                "output_dir", "output_name", "resume", "save_every_n_epochs", "save_last_n_epochs", "save_every_n_steps",
                "save_last_n_steps", "save_last_n_epochs_state", "save_last_n_steps_state", "save_state",
                "save_state_on_train_end", "mem_eff_save",
                # HuggingFace Settings
                "huggingface_repo_id", "huggingface_token", "huggingface_repo_type", "huggingface_repo_visibility",
                "huggingface_path_in_repo", "save_state_to_huggingface", "resume_from_huggingface", "async_upload",
                # Metadata Settings
                "metadata_author", "metadata_description", "metadata_license", "metadata_tags", "metadata_title"
            ],
            args
        )))
    elif action == "save_configuration":
        return save_wan_configuration(False, config_file_name, list(zip(
            [
                # accelerate_launch
                "mixed_precision", "num_cpu_threads_per_process", "num_processes", "num_machines", "multi_gpu", "gpu_ids",
                "main_process_port", "dynamo_backend", "dynamo_mode", "dynamo_use_fullgraph", "dynamo_use_dynamic", "extra_accelerate_launch_args",
                # advanced_training
                "additional_parameters", "debug_mode",
                # Wan Dataset settings
                "dataset_config_mode", "dataset_config", "parent_folder_path", "dataset_resolution_width",
                "dataset_resolution_height", "dataset_caption_extension", "dataset_batch_size",
                "create_missing_captions", "caption_strategy", "dataset_enable_bucket",
                "dataset_bucket_no_upscale", "dataset_cache_directory", "generated_toml_path",
                # Wan Model settings
                "training_mode", "task", "dit", "vae", "t5", "clip",
                "dit_high_noise", "timestep_boundary", "offload_inactive_dit", "dit_dtype",
                "text_encoder_dtype", "vae_dtype", "clip_vision_dtype", "fp8_base", "fp8_scaled",
                "fp8_t5", "blocks_to_swap", "use_pinned_memory_for_block_swap", "vae_tiling", "vae_chunk_size", "vae_cache_cpu", "num_frames", "one_frame", "force_v2_1_time_embedding",
                # training_settings
                "sdpa", "flash_attn", "sage_attn", "xformers", "split_attn", "max_train_steps", "max_train_epochs",
                "max_data_loader_n_workers", "persistent_data_loader_workers", "seed", "gradient_checkpointing",
                "gradient_checkpointing_cpu_offload", "gradient_accumulation_steps", "full_bf16", "full_fp16",
                "logging_dir", "log_with", "log_prefix", "log_tracker_name", "wandb_run_name", "log_tracker_config",
                "wandb_api_key", "log_config", "ddp_timeout", "ddp_gradient_as_bucket_view", "ddp_static_graph",
                # Sample generation settings (from TrainingSettings)
                "sample_every_n_steps", "sample_every_n_epochs", "sample_at_first", "sample_prompts", "sample_output_dir",
                "disable_prompt_enhancement", "sample_width", "sample_height", "sample_num_frames", "sample_steps",
                "sample_guidance_scale", "sample_seed", "sample_negative_prompt",
                # Latent Caching Settings
                "caching_latent_device", "caching_latent_batch_size", "caching_latent_num_workers", "caching_latent_skip_existing",
                "caching_latent_keep_cache", "caching_latent_debug_mode", "caching_latent_console_width",
                "caching_latent_console_back", "caching_latent_console_num_images",
                # Text Encoder Outputs Caching Settings
                "caching_teo_text_encoder1", "caching_teo_text_encoder2", "caching_teo_text_encoder_dtype", "caching_teo_device",
                "caching_teo_fp8_llm", "caching_teo_batch_size", "caching_teo_num_workers", "caching_teo_skip_existing",
                "caching_teo_keep_cache",
                # Optimizer and Scheduler Settings
                "optimizer_type", "optimizer_args", "learning_rate", "max_grad_norm", "lr_scheduler", "lr_warmup_steps",
                "lr_decay_steps", "lr_scheduler_num_cycles", "lr_scheduler_power", "lr_scheduler_timescale",
                "lr_scheduler_min_lr_ratio", "lr_scheduler_type", "lr_scheduler_args",
                # Network Settings
                "no_metadata", "network_weights", "network_module", "network_dim", "network_alpha",
                "network_dropout", "network_args", "training_comment", "dim_from_weights", "scale_weight_norms",
                "base_weights", "base_weights_multiplier",
                # Save/Load Settings
                "output_dir", "output_name", "resume", "save_every_n_epochs", "save_last_n_epochs", "save_every_n_steps",
                "save_last_n_steps", "save_last_n_epochs_state", "save_last_n_steps_state", "save_state",
                "save_state_on_train_end", "mem_eff_save",
                # HuggingFace Settings
                "huggingface_repo_id", "huggingface_token", "huggingface_repo_type", "huggingface_repo_visibility",
                "huggingface_path_in_repo", "save_state_to_huggingface", "resume_from_huggingface", "async_upload",
                # Metadata Settings
                "metadata_author", "metadata_description", "metadata_license", "metadata_tags", "metadata_title"
            ],
            args
        )))
    elif action == "train_model":
        log.info("Train WAN model...")
        gr.Info("Training is starting... Please check the console for progress.")
        parameters = list(zip(
            [
                # accelerate_launch
                "mixed_precision", "num_cpu_threads_per_process", "num_processes", "num_machines", "multi_gpu", "gpu_ids",
                "main_process_port", "dynamo_backend", "dynamo_mode", "dynamo_use_fullgraph", "dynamo_use_dynamic", "extra_accelerate_launch_args",
                # advanced_training
                "additional_parameters", "debug_mode",
                # Wan Dataset settings
                "dataset_config_mode", "dataset_config", "parent_folder_path", "dataset_resolution_width",
                "dataset_resolution_height", "dataset_caption_extension", "dataset_batch_size",
                "create_missing_captions", "caption_strategy", "dataset_enable_bucket",
                "dataset_bucket_no_upscale", "dataset_cache_directory", "generated_toml_path",
                # Wan Model settings
                "training_mode", "task", "dit", "vae", "t5", "clip",
                "dit_high_noise", "timestep_boundary", "offload_inactive_dit", "dit_dtype",
                "text_encoder_dtype", "vae_dtype", "clip_vision_dtype", "fp8_base", "fp8_scaled",
                "fp8_t5", "blocks_to_swap", "use_pinned_memory_for_block_swap", "vae_tiling", "vae_chunk_size", "vae_cache_cpu", "num_frames", "one_frame", "force_v2_1_time_embedding",
                # training_settings
                "sdpa", "flash_attn", "sage_attn", "xformers", "split_attn", "max_train_steps", "max_train_epochs",
                "max_data_loader_n_workers", "persistent_data_loader_workers", "seed", "gradient_checkpointing",
                "gradient_checkpointing_cpu_offload", "gradient_accumulation_steps", "full_bf16", "full_fp16",
                "logging_dir", "log_with", "log_prefix", "log_tracker_name", "wandb_run_name", "log_tracker_config",
                "wandb_api_key", "log_config", "ddp_timeout", "ddp_gradient_as_bucket_view", "ddp_static_graph",
                # Sample generation settings
                "sample_every_n_steps", "sample_every_n_epochs", "sample_at_first", "sample_prompts", "sample_output_dir",
                "disable_prompt_enhancement", "sample_width", "sample_height", "sample_num_frames", "sample_steps",
                "sample_guidance_scale", "sample_seed", "sample_negative_prompt",
                # Latent Caching Settings
                "caching_latent_device", "caching_latent_batch_size", "caching_latent_num_workers", "caching_latent_skip_existing",
                "caching_latent_keep_cache", "caching_latent_debug_mode", "caching_latent_console_width",
                "caching_latent_console_back", "caching_latent_console_num_images",
                # Text Encoder Outputs Caching Settings
                "caching_teo_text_encoder1", "caching_teo_text_encoder2", "caching_teo_text_encoder_dtype", "caching_teo_device",
                "caching_teo_fp8_llm", "caching_teo_batch_size", "caching_teo_num_workers", "caching_teo_skip_existing",
                "caching_teo_keep_cache",
                # Optimizer and Scheduler Settings
                "optimizer_type", "optimizer_args", "learning_rate", "max_grad_norm", "lr_scheduler", "lr_warmup_steps",
                "lr_decay_steps", "lr_scheduler_num_cycles", "lr_scheduler_power", "lr_scheduler_timescale",
                "lr_scheduler_min_lr_ratio", "lr_scheduler_type", "lr_scheduler_args",
                # Network Settings
                "no_metadata", "network_weights", "network_module", "network_dim", "network_alpha",
                "network_dropout", "network_args", "training_comment", "dim_from_weights", "scale_weight_norms",
                "base_weights", "base_weights_multiplier",
                # Save/Load Settings
                "output_dir", "output_name", "resume", "save_every_n_epochs", "save_last_n_epochs", "save_every_n_steps",
                "save_last_n_steps", "save_last_n_epochs_state", "save_last_n_steps_state", "save_state",
                "save_state_on_train_end", "mem_eff_save",
                # HuggingFace Settings
                "huggingface_repo_id", "huggingface_token", "huggingface_repo_type", "huggingface_repo_visibility",
                "huggingface_path_in_repo", "save_state_to_huggingface", "resume_from_huggingface", "async_upload",
                # Metadata Settings
                "metadata_author", "metadata_description", "metadata_license", "metadata_tags", "metadata_title"
            ],
            args
        ))
        return train_wan_model(
            headless=headless,
            print_only=print_only,
            parameters=parameters,
        )
    
    return "Unknown action"


def train_wan_model(headless, print_only, parameters):
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
    
    # Debug: Log critical caching parameters to diagnose misalignment issues
    log.debug(f"[DEBUG] Critical caching parameters:")
    log.debug(f"  caching_latent_device: {param_dict.get('caching_latent_device')} (type: {type(param_dict.get('caching_latent_device')).__name__})")
    log.debug(f"  caching_latent_batch_size: {param_dict.get('caching_latent_batch_size')} (type: {type(param_dict.get('caching_latent_batch_size')).__name__})")
    log.debug(f"  caching_latent_num_workers: {param_dict.get('caching_latent_num_workers')} (type: {type(param_dict.get('caching_latent_num_workers')).__name__})")
    log.debug(f"  caching_latent_debug_mode: {param_dict.get('caching_latent_debug_mode')} (type: {type(param_dict.get('caching_latent_debug_mode')).__name__})")
    log.debug(f"  debug_mode: {param_dict.get('debug_mode')} (type: {type(param_dict.get('debug_mode')).__name__})")

    # Initialize variables that might be needed for caching
    latent_cache_cmd = None
    teo_cache_cmd = None

    # Always use the Dataset Config File path for training
    effective_dataset_config = param_dict.get("dataset_config")
    dataset_mode = param_dict.get("dataset_config_mode", "Use TOML File")

    # Validate dataset config path
    if dataset_mode == "Use TOML File" and (not effective_dataset_config or effective_dataset_config == ""):
        log.error("Dataset config file is required for WAN training")
        gr.Error("Dataset config file is required for WAN training")
        return

    # Setup latent caching command for WAN
    if param_dict.get("caching_latent_skip_existing") is not False:  # Only cache if skip_existing is not explicitly disabled
        run_cache_latent_cmd = [python_cmd, "./musubi-tuner/src/musubi_tuner/wan_cache_latents.py",
                                "--dataset_config", str(effective_dataset_config)]

        # Add VAE path (required for WAN latent caching)
        vae_path = param_dict.get("vae", "")
        if vae_path and vae_path != "":
            run_cache_latent_cmd.append("--vae")
            run_cache_latent_cmd.append(str(vae_path))
        else:
            log.warning("VAE path not provided for latent caching")
            gr.Warning("VAE path not provided - latent caching may fail")

        # Add CLIP path if provided (for I2V models)
        clip_path = param_dict.get("clip", "")
        if clip_path and clip_path != "":
            run_cache_latent_cmd.append("--clip")
            run_cache_latent_cmd.append(str(clip_path))

        # Add latent caching parameters
        caching_device = param_dict.get("caching_latent_device")
        if caching_device == "cuda" and param_dict.get("gpu_ids"):
            # If gpu_ids is specified in accelerate config, use the first GPU ID
            gpu_ids = str(param_dict.get("gpu_ids")).split(",")
            caching_device = f"cuda:{gpu_ids[0].strip()}"
            log.info(f"Using GPU ID from accelerate config for latent caching: {caching_device}")

        if caching_device:
            run_cache_latent_cmd.append("--device")
            run_cache_latent_cmd.append(str(caching_device))

        # Validate and add batch_size - must be a valid integer
        caching_batch_size = param_dict.get("caching_latent_batch_size")
        if caching_batch_size:
            # Validate it's actually a number, not a string like "cuda"
            try:
                batch_size_int = int(caching_batch_size)
                if batch_size_int > 0:
                    run_cache_latent_cmd.append("--batch_size")
                    run_cache_latent_cmd.append(str(batch_size_int))
                else:
                    log.warning(f"Invalid caching_latent_batch_size: {caching_batch_size} (must be > 0)")
            except (ValueError, TypeError):
                log.error(f"caching_latent_batch_size has invalid value: {caching_batch_size} (type: {type(caching_batch_size)}). Expected integer.")
                log.error(f"caching_latent_device value: {param_dict.get('caching_latent_device')}")
                log.error(f"This suggests parameter misalignment. Check config file loading.")
                raise ValueError(f"Invalid caching_latent_batch_size value: {caching_batch_size}. Expected integer, got {type(caching_batch_size).__name__}")

        if param_dict.get("caching_latent_num_workers"):
            run_cache_latent_cmd.append("--num_workers")
            run_cache_latent_cmd.append(str(param_dict.get("caching_latent_num_workers")))

        if param_dict.get("caching_latent_skip_existing"):
            run_cache_latent_cmd.append("--skip_existing")

        if param_dict.get("caching_latent_keep_cache"):
            run_cache_latent_cmd.append("--keep_cache")

        # Debug mode for latent caching
        debug_mode_val = param_dict.get("caching_latent_debug_mode")
        # Filter out boolean values, empty strings, "None" string, and validate enum
        if debug_mode_val is not None and debug_mode_val not in ["", "None", True, False] and not isinstance(debug_mode_val, bool):
            # Validate it's one of the allowed enum values
            if debug_mode_val in ["image", "console", "video"]:
                run_cache_latent_cmd.append("--debug_mode")
                run_cache_latent_cmd.append(str(debug_mode_val))
            else:
                log.warning(f"Invalid caching_latent_debug_mode value: {debug_mode_val}. Must be one of: image, console, video")

        # Determine if this is I2V training
        task = param_dict.get("task", "t2v-14B")
        if "i2v" in task.lower():
            run_cache_latent_cmd.append("--i2v")

        # Check for one_frame training (only enable if it's explicitly True, not just truthy)
        if param_dict.get("one_frame", False) is True:
            run_cache_latent_cmd.append("--one_frame")

        # VAE cache CPU setting
        if param_dict.get("vae_cache_cpu", False):
            run_cache_latent_cmd.append("--vae_cache_cpu")

        latent_cache_cmd = run_cache_latent_cmd
        log.info(f"WAN latent caching command: {latent_cache_cmd}")
    else:
        log.info("Latent caching skipped (caching_latent_skip_existing is False)")

    # Setup text encoder outputs caching command for WAN
    if param_dict.get("caching_teo_skip_existing") is not False:  # Only cache if skip_existing is not explicitly disabled
        run_cache_teo_cmd = [python_cmd, "./musubi-tuner/src/musubi_tuner/wan_cache_text_encoder_outputs.py",
                            "--dataset_config", str(effective_dataset_config)]

        # Add T5 text encoder path (required for WAN text encoder caching)
        t5_path = param_dict.get("t5", "")
        if not t5_path or t5_path == "":
            # Try to get from text encoder 1 directory
            t5_path = param_dict.get("caching_teo_text_encoder1", "")

        if t5_path and t5_path != "":
            run_cache_teo_cmd.append("--t5")
            run_cache_teo_cmd.append(str(t5_path))

            # Add text encoder caching parameters
            teo_caching_device = param_dict.get("caching_teo_device")
            if teo_caching_device == "cuda" and param_dict.get("gpu_ids"):
                # If gpu_ids is specified in accelerate config, use the first GPU ID
                gpu_ids = str(param_dict.get("gpu_ids")).split(",")
                teo_caching_device = f"cuda:{gpu_ids[0].strip()}"
                log.info(f"Using GPU ID from accelerate config for text encoder caching: {teo_caching_device}")

            if teo_caching_device:
                run_cache_teo_cmd.append("--device")
                run_cache_teo_cmd.append(str(teo_caching_device))

            if param_dict.get("caching_teo_fp8_llm"):
                run_cache_teo_cmd.append("--fp8_t5")

            if param_dict.get("caching_teo_batch_size"):
                run_cache_teo_cmd.append("--batch_size")
                run_cache_teo_cmd.append(str(param_dict.get("caching_teo_batch_size")))

            if param_dict.get("caching_teo_num_workers"):
                run_cache_teo_cmd.append("--num_workers")
                run_cache_teo_cmd.append(str(param_dict.get("caching_teo_num_workers")))

            if param_dict.get("caching_teo_skip_existing"):
                run_cache_teo_cmd.append("--skip_existing")

            if param_dict.get("caching_teo_keep_cache"):
                run_cache_teo_cmd.append("--keep_cache")

            # Note: WAN text encoder caching does not support --text_encoder_dtype
            # The T5 model uses its configured dtype from the model config
            # if param_dict.get("caching_teo_text_encoder_dtype"):
            #     run_cache_teo_cmd.append("--text_encoder_dtype")
            #     run_cache_teo_cmd.append(str(param_dict.get("caching_teo_text_encoder_dtype")))

            teo_cache_cmd = run_cache_teo_cmd
            log.info(f"WAN text encoder caching command: {teo_cache_cmd}")
        else:
            log.warning("T5 text encoder path not provided for text encoder caching")
            gr.Warning("T5 text encoder path not provided - text encoder caching will be skipped")
            teo_cache_cmd = None
    else:
        log.info("Text encoder caching skipped (caching_teo_skip_existing is False)")

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

    # Select the appropriate WAN training script based on training mode
    training_mode = param_dict.get("training_mode", "LoRA Training")
    if training_mode == "DreamBooth Fine-Tuning":
        # Use full fine-tuning script for DreamBooth mode
        run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/wan_train.py")
        log.info("Using wan_train.py for full DreamBooth fine-tuning")
    else:
        # Use network training script for LoRA mode
        run_cmd.append(f"{scriptdir}/musubi-tuner/src/musubi_tuner/wan_train_network.py")
        log.info("Using wan_train_network.py for LoRA training")

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

        # Enhance sample prompts file with GUI defaults if sample generation is enabled
        if sample_prompts_provided:
            # Check if prompt enhancement is disabled
            disable_enhancement = param_dict.get('disable_prompt_enhancement', False)

            if disable_enhancement:
                # Use original prompts without enhancement
                log.info("Prompt enhancement disabled - using original prompts without WAN format parameters")
            else:
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
                    sample_width=param_dict.get('sample_width', 960),
                    sample_height=param_dict.get('sample_height', 960),
                    sample_num_frames=param_dict.get('sample_num_frames', 81),
                    sample_steps=param_dict.get('sample_steps', 20),
                    sample_guidance_scale=param_dict.get('sample_guidance_scale', 7.0),
                    sample_seed=param_dict.get('sample_seed', 99),
                    sample_negative_prompt=param_dict.get('sample_negative_prompt', '')
                )

                if enhanced_prompt_file:
                    # Update parameters to use the enhanced prompt file
                    log.info(f"Using enhanced prompt file for WAN training: {enhanced_prompt_file}")
                    parameters = upsert_parameter(parameters, "sample_prompts", enhanced_prompt_file)
                else:
                    log.warning("Failed to create enhanced prompt file, using original file")

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
                            modified_params.append((key, "networks.lora_wan"))
                            log.warning("LoRA mode: Custom module selected but not specified, using default networks.lora_wan")
                    elif not value or value == "":
                        # Ensure network_module is set for LoRA
                        modified_params.append((key, "networks.lora_wan"))
                        log.info("LoRA mode: Setting network_module to networks.lora_wan")
                    else:
                        # Use the selected module directly
                        modified_params.append((key, value))
                        log.info(f"LoRA mode: Using network module: {value}")
                elif key == "custom_network_module":
                    # Skip this parameter as it's handled above
                    continue
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

        # Ensure dataset_config entry survives all transformations
        parameters = upsert_parameter(parameters, "dataset_config", effective_dataset_config)

        # Handle use_pinned_memory_for_block_swap checkbox: check if parameter exists in training script
        use_pinned_memory_enabled = param_dict.get("use_pinned_memory_for_block_swap", False)
        log.info(f"use_pinned_memory_for_block_swap checkbox value: {use_pinned_memory_enabled}")
        
        # Check if the parameter is supported by the training script
        # Get project directory (parent of musubi_tuner_gui module)
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check both wan_train.py and wan_train_network.py (used for DreamBooth and LoRA respectively)
        # Also check hv_train_network.py which is imported by wan_train_network.py
        # Also check musubi-tuner and musubi-tuner2 directories (in case user is on a branch)
        possible_paths = [
            os.path.join(project_dir, "musubi-tuner", "src", "musubi_tuner", "wan_train.py"),
            os.path.join(project_dir, "musubi-tuner", "src", "musubi_tuner", "wan_train_network.py"),
            os.path.join(project_dir, "musubi-tuner", "src", "musubi_tuner", "hv_train_network.py"),
            os.path.join(project_dir, "musubi-tuner2", "src", "musubi_tuner", "wan_train.py"),
            os.path.join(project_dir, "musubi-tuner2", "src", "musubi_tuner", "wan_train_network.py"),
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
        use_pinned_memory_in_params = any(k == "use_pinned_memory_for_block_swap" for k, v in parameters)
        log.info(f"Before SaveConfigFileToRun: use_pinned_memory_for_block_swap in params: {use_pinned_memory_in_params}")

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
                "training_mode",  # Exclude training_mode as it's GUI-only
                "num_frames",  # Not used in Wan training, only for sample generation
            ] + pattern_exclusion,
            mandatory_keys=["dataset_config", "dit", "vae", "t5", "clip"],
        )

        run_cmd.append("--config_file")
        run_cmd.append(f"{file_path}")

        # Verify TOML file was created and check its contents
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                toml_content = toml.load(f)
                use_pinned_memory_in_toml = toml_content.get("use_pinned_memory_for_block_swap", None)
                log.info(f"After SaveConfigFileToRun - TOML file contents check:")
                log.info(f"  use_pinned_memory_for_block_swap = {use_pinned_memory_in_toml}")

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

        # Handle use_pinned_memory_for_block_swap in additional_parameters
        # to ensure it's passed via command line as well (for backward compatibility)
        args_to_add = []
        args_to_remove = []
        
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

        # Run latent caching first (if needed)
        if latent_cache_cmd:
            log.info("Running latent caching...")
            try:
                gr.Info("Starting latent caching... This may take a while.")
                result = subprocess.run(latent_cache_cmd, env=setup_environment(), check=True)
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

        # Create a wrapper script that runs text encoder caching and training
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
echo Text encoder caching completed successfully.
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
echo "Text encoder caching completed successfully."
echo "Starting training..."
{' '.join(run_cmd)}
"""

            # Write the script to a temporary file
            temp_script = tempfile.NamedTemporaryFile(mode='w', suffix=script_ext, delete=False)
            temp_script.write(script_content)
            temp_script.close()

            # Make the script executable on Unix systems
            if platform.system() != "Windows":
                import stat
                os.chmod(temp_script.name, os.stat(temp_script.name).st_mode | stat.S_IEXEC)

            # Execute the combined script
            if platform.system() == "Windows":
                final_cmd = [temp_script.name]
            else:
                final_cmd = ["bash", temp_script.name]

            log.info("Starting combined text encoder caching and training process...")
            gr.Info("Starting text encoder caching followed by training...")
            executor.execute_command(run_cmd=final_cmd, env=env, shell=True if platform.system() == "Windows" else False)
        elif latent_cache_cmd:
            # Only latent caching was run, now run training
            log.info("Latent caching completed, starting training...")
            gr.Info("Starting training...")
            executor.execute_command(run_cmd=run_cmd, env=env)
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


def open_wan_configuration(ask_for_file, file_path, parameters):
    """Load WAN configuration from TOML file"""
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
                
                # Validate and auto-correct corrupted parameter values from old bugs
                # Bug: one_frame and num_frames could get swapped due to parameter order mismatch
                if 'one_frame' in my_data and 'num_frames' in my_data:
                    one_frame_val = my_data['one_frame']
                    num_frames_val = my_data['num_frames']
                    
                    # Check if one_frame is numeric (should be boolean)
                    if isinstance(one_frame_val, (int, float)):
                        log.warning(f"Detected corrupted one_frame value ({one_frame_val}). Auto-correcting...")
                        # If num_frames is boolean, they got swapped
                        if isinstance(num_frames_val, bool):
                            # Swap them
                            my_data['one_frame'] = num_frames_val
                            my_data['num_frames'] = int(one_frame_val)
                            log.info(f"Swapped: one_frame={num_frames_val}, num_frames={one_frame_val}")
                        else:
                            # num_frames is correct, just fix one_frame
                            my_data['one_frame'] = False  # Default to False
                            log.info(f"Corrected: one_frame=False (was {one_frame_val})")
                    
                    # Check if num_frames is boolean (should be numeric)
                    elif isinstance(num_frames_val, bool):
                        log.warning(f"Detected corrupted num_frames value ({num_frames_val}). Auto-correcting...")
                        my_data['num_frames'] = 81  # Default to 81 for WAN
                        log.info(f"Corrected: num_frames=81 (was {num_frames_val})")
                
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
        "save_every_n_steps", "max_timestep", "min_timestep",
        "network_dim", "num_layers",  # These can be None for auto-detection
        "max_train_epochs",  # 0 means use max_train_steps instead
        "dit_in_channels", "sample_num_frames", "num_timestep_buckets",  # WAN-specific optional params
        "save_last_n_epochs", "save_last_n_steps",  # 0 = keep all = None for musubi tuner
        "save_last_n_epochs_state", "save_last_n_steps_state",  # Same pattern for state files
        "blocks_to_swap", "vae_chunk_size", "vae_spatial_tile_sample_min_size"  # Memory optimization params
        # Removed: "ddp_timeout" (0 = use default 30min timeout - VALID)
    }

    # NOTE: Exclude fields that are legitimately lists like optimizer_args, lr_scheduler_args, network_args
    numeric_fields = [
        'learning_rate', 'max_grad_norm', 'guidance_scale', 'logit_mean', 'logit_std',
        'mode_scale', 'sigmoid_scale', 'lr_scheduler_power', 'lr_scheduler_timescale',
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',
        'vae_chunk_size', 'blocks_to_swap', 'min_timestep', 'max_timestep', 'discrete_flow_shift', 'flow_shift',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers',
        'sample_width', 'sample_height', 'sample_steps', 'sample_guidance_scale', 'sample_seed',
        'timestep_boundary', 'num_frames', 'vae_spatial_tile_sample_min_size',
        # ADD MISSING WAN-SPECIFIC NUMERIC FIELDS:
        'dit_in_channels', 'num_layers', 'network_dropout', 'sample_num_frames', 'num_timestep_buckets'
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
                # Special handling for debug_mode - use "None" as default if missing
                if key == "debug_mode" and toml_value is None:
                    value = "None"
                    log.debug(f"[CONFIG] debug_mode not found in TOML, using default: None")
                
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


def save_wan_configuration(save_as_bool, file_path, parameters):
    """Save WAN configuration to TOML file"""
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
                modified_params.append((key, "networks.lora_wan"))
                log.info("LoRA mode: Setting network_module to networks.lora_wan")
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
        'lr_scheduler_min_lr_ratio', 'network_alpha', 'base_weights_multiplier',
        'vae_chunk_size', 'blocks_to_swap', 'min_timestep', 'max_timestep', 'discrete_flow_shift', 'flow_shift',
        'scale_weight_norms', 'dataset_resolution_width', 'dataset_resolution_height',
        'dataset_batch_size', 'max_train_steps', 'max_train_epochs', 'seed',
        'gradient_accumulation_steps', 'sample_every_n_steps', 'sample_every_n_epochs',
        'save_every_n_steps', 'save_every_n_epochs', 'save_last_n_epochs',
        'save_last_n_steps', 'save_last_n_epochs_state', 'save_last_n_steps_state',
        'network_dim', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
        'ddp_timeout', 'max_data_loader_n_workers',
        'num_processes', 'num_machines', 'num_cpu_threads_per_process', 'main_process_port',
        'caching_latent_batch_size', 'caching_latent_num_workers', 'caching_latent_console_width',
        'caching_latent_console_num_images', 'caching_teo_batch_size', 'caching_teo_num_workers',
        'sample_width', 'sample_height', 'sample_steps', 'sample_guidance_scale', 'sample_seed',
        'timestep_boundary', 'num_frames', 'vae_spatial_tile_sample_min_size',
        # ADD MISSING WAN-SPECIFIC NUMERIC FIELDS:
        'dit_in_channels', 'num_layers', 'network_dropout', 'sample_num_frames', 'num_timestep_buckets'
    ]
    
    # Parameters that should be None when their value is 0 (optional parameters)
    # When saving, convert None back to 0 so they get saved properly
    optional_parameters = {
        "sample_every_n_steps", "sample_every_n_epochs",
        "save_every_n_steps", "max_timestep", "min_timestep",
        "network_dim", "num_layers",  # These can be None for auto-detection
        "max_train_epochs",  # 0 means use max_train_steps instead
        "dit_in_channels", "sample_num_frames", "num_timestep_buckets",  # WAN-specific optional params
        "save_last_n_epochs", "save_last_n_steps",  # 0 = keep all = None for musubi tuner
        "save_last_n_epochs_state", "save_last_n_steps_state",  # Same pattern for state files
        "blocks_to_swap", "vae_chunk_size", "vae_spatial_tile_sample_min_size"  # Memory optimization params
        # Removed: "ddp_timeout" (0 = use default 30min timeout - VALID)
    }

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

        # If value is a list and it's not supposed to be (like from a Number component)
        # take the first element or convert to appropriate type
        if isinstance(value, list) and len(value) > 0 and key in numeric_fields:
            # These should be single numeric values
            value = value[0] if value else None
        # Convert string values to appropriate numeric types for numeric fields
        elif isinstance(value, str) and key in numeric_fields:
            try:
                # Try to convert to float first, then to int if it's a whole number
                float_value = float(value)
                if float_value.is_integer():
                    value = int(float_value)
                else:
                    value = float_value
            except (ValueError, AttributeError):
                # If conversion fails, keep as string but log warning
                log.warning(f"Failed to convert string '{value}' to number for field '{key}', keeping as string")
        # Clean up optimizer_args to remove trailing commas that could break parsing
        elif key == "optimizer_args" and isinstance(value, list):
            value = [arg.rstrip(',') for arg in value]
        
        # Convert None back to 0 for optional parameters so they get saved properly
        # (None values are excluded by SaveConfigFile, but 0 is a valid value to save)
        if key in optional_parameters and value is None:
            value = 0
        
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
                # Note: debug_mode is saved - it's used by caching scripts
                # Note: caching_latent_debug_mode is also saved - it's used by caching scripts
            ],
        )

    except Exception as e:
        error_msg = f"Failed to save configuration: {str(e)}"
        log.error(error_msg)
        gr.Error(error_msg)
        return original_file_path, gr.update(value=error_msg, visible=True)

    config_name = os.path.basename(file_path)
    status_msg = f"Configuration saved successfully to: {config_name}"
    log.info(status_msg)
    gr.Info(status_msg)
    return file_path, gr.update(value=status_msg, visible=True)


def wan_lora_tab(
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

    # Apply WAN-specific defaults if not already configured
    wan_defaults = {
        "task": "t2v-14B",
        "training_mode": "LoRA Training",
        "dataset_config_mode": "Generate from Folder Structure",
        "dataset_resolution_width": 960,
        "dataset_resolution_height": 960,
        "dataset_caption_extension": ".txt",
        "create_missing_captions": True,
        "caption_strategy": "folder_name",
        "dataset_batch_size": 1,
        "dataset_enable_bucket": False,
        "dataset_bucket_no_upscale": False,
        "dataset_cache_directory": "cache_dir",
        "dit_dtype": "bfloat16",
        "text_encoder_dtype": "bfloat16",
        "vae_dtype": "bfloat16",
        "clip_vision_dtype": "bfloat16",
        "fp8_base": False,
        "fp8_scaled": False,
        "fp8_t5": False,
        "blocks_to_swap": 0,
        "vae_tiling": False,
        "vae_chunk_size": 0,
        "vae_spatial_tile_sample_min_size": 0,
        "vae_cache_cpu": False,
        "force_v2_1_time_embedding": False,
        "num_frames": 81,
        "one_frame": False,
        "timestep_boundary": None,
        "offload_inactive_dit": False,
        "network_module": "networks.lora_wan",
        "network_dim": 16,
        "network_alpha": 16.0,
        "network_dropout": 0.0,
        "max_train_steps": 90000,
        "max_train_epochs": 200,
        "learning_rate": 1e-4,
        "optimizer_type": "adamw8bit",
        "lr_scheduler": "constant",
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "sdpa": True,
        "seed": 99,
        "gradient_checkpointing": True,
        "mixed_precision": "bf16",
        "sample_width": 960,
        "sample_height": 960,
        "sample_num_frames": 81,
        "sample_steps": 20,
        "sample_guidance_scale": 7.0,
        "sample_seed": 99,
        "sample_scheduler": "unipc",
        "caching_latent_device": "cuda",
        "caching_latent_batch_size": 4,
        "caching_latent_num_workers": 8,
        "caching_latent_skip_existing": True,
        "caching_latent_keep_cache": True,
        "caching_teo_device": "cuda",
        "caching_teo_fp8_llm": False,
        "caching_teo_batch_size": 16,
        "caching_teo_num_workers": 8,
        "caching_teo_skip_existing": True,
        "caching_teo_keep_cache": True,
        "caching_teo_text_encoder_dtype": "bfloat16",
        "output_name": "my-wan-lora",
    }

    # Handle different config types
    if isinstance(config, dict):
        # If config is a dict (default case), apply WAN defaults
        config.update({k: v for k, v in wan_defaults.items() if k not in config})
        # Create a GUIConfig-like object
        config = type('GUIConfig', (), {'config': config, 'get': config.get})()
    elif hasattr(config, 'config'):
        # If config is a GUIConfig object, update its config dict
        if not config.config:
            config.config = wan_defaults.copy()
        else:
            config.config.update({k: v for k, v in wan_defaults.items() if k not in config.config})

    # Setup Configuration Files Gradio
    with gr.Accordion("Configuration file Settings", open=True):
        configuration = ConfigurationFile(headless=headless, config=config)

    # Add search functionality and unified toggle button
    with gr.Row():
        with gr.Column(scale=2):
            search_input = gr.Textbox(
                label="🔍 Search Settings",
                placeholder="Type to search and filter panels (e.g., 'model', 'learning', 'fp8', 'cache', 'epochs')",
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
    
    # Accelerate launch Settings - Collapsed by default for Wan
    accelerate_accordion = gr.Accordion("Accelerate launch Settings", open=False, elem_classes="flux1_background")
    accordions.append(accelerate_accordion)
    with accelerate_accordion, gr.Column():
        accelerate_launch = AccelerateLaunch(config=config)
        # Note: bf16 mixed precision is STRONGLY recommended for Wan models
        
    # Save Load Settings
    save_load_accordion = gr.Accordion("Save Models and Resume Training Settings", open=False, elem_classes="samples_background")
    accordions.append(save_load_accordion)
    with save_load_accordion:
        saveLoadSettings = WanSaveLoadSettings(headless=headless, config=config)
    
    # Wan Training Dataset accordion
    wan_dataset_accordion = gr.Accordion("Wan Training Dataset", open=False, elem_classes="samples_background")
    accordions.append(wan_dataset_accordion)
    with wan_dataset_accordion:
        wan_dataset = WanDataset(headless=headless, config=config)

    # Wan Model Settings accordion
    wan_model_accordion = gr.Accordion("Wan Model Settings", open=False, elem_classes="model_background")
    accordions.append(wan_model_accordion)
    with wan_model_accordion:
        wan_model_settings = WanModelSettings(headless=headless, config=config)
    
    # Setup dataset UI events after wan_model_settings is created
    wan_dataset.setup_dataset_ui_events(saveLoadSettings, wan_model_settings)  # Pass both saveLoadSettings and wan_model_settings

    # Training Settings accordion
    training_accordion = gr.Accordion("Training Settings", open=False, elem_classes="training_background")
    accordions.append(training_accordion)
    with training_accordion:
        training_settings = TrainingSettings(headless=headless, config=config)

    # Network Settings accordion (LoRA specific)
    network_accordion = gr.Accordion("Network Settings", open=False, elem_classes="network_background")
    accordions.append(network_accordion)
    with network_accordion:
        network = Network(headless=headless, config=config)

    # Optimizer and Scheduler Settings
    optimizer_accordion = gr.Accordion("Optimizer and Scheduler Settings", open=False, elem_classes="optimizer_background")
    accordions.append(optimizer_accordion)
    with optimizer_accordion:
        optimizer_scheduler = OptimizerAndScheduler(headless=headless, config=config)

    # Latent Caching Settings
    latent_caching_accordion = gr.Accordion("Latent Caching Settings", open=False, elem_classes="caching_background")
    accordions.append(latent_caching_accordion)
    with latent_caching_accordion:
        latent_caching = LatentCaching(headless=headless, config=config)

    # Text Encoder Outputs Caching Settings
    text_encoder_caching_accordion = gr.Accordion("Text Encoder Outputs Caching Settings", open=False, elem_classes="caching_background")
    accordions.append(text_encoder_caching_accordion)
    with text_encoder_caching_accordion:
        text_encoder_caching = TextEncoderOutputsCaching(headless=headless, config=config, show_text_encoder_dtype=False)

    # Sample Generation Settings
    sample_accordion = gr.Accordion("Sample Generation Settings", open=False, elem_classes="samples_background")
    accordions.append(sample_accordion)
    with sample_accordion:
        sampleSettings = WanSampleSettings(headless=headless, config=config)

    # Advanced Settings
    advanced_accordion = gr.Accordion("Advanced Settings", open=False, elem_classes="advanced_background")
    accordions.append(advanced_accordion)
    with advanced_accordion:
        gr.Markdown("**Additional Parameters**: Add custom training parameters as key=value pairs (e.g., `custom_param=value`). These will be appended to the training command.")
        advanced_training = AdvancedTraining(headless=headless, training_type="lora", config=config)

    # Metadata Settings
    metadata_accordion = gr.Accordion("Metadata Settings", open=False, elem_classes="metadata_background")
    accordions.append(metadata_accordion)
    with metadata_accordion:
        metadata = MetaData(config=config)

    # HuggingFace Settings
    huggingface_accordion = gr.Accordion("HuggingFace Settings", open=False, elem_classes="huggingface_background")
    accordions.append(huggingface_accordion)
    with huggingface_accordion:
        global huggingface
        huggingface = HuggingFace(config=config)

    # Command execution and training controls
    with gr.Row():
        button_print = gr.Button("Print Command", variant="secondary")
    
    global executor
    executor = CommandExecutor(headless=headless)
    run_state = gr.State(value=train_state_value)

    # Collect all settings for processing
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

        # Wan Dataset settings
        wan_dataset.dataset_config_mode,
        wan_dataset.dataset_config,
        wan_dataset.parent_folder_path,
        wan_dataset.dataset_resolution_width,
        wan_dataset.dataset_resolution_height,
        wan_dataset.dataset_caption_extension,
        wan_dataset.dataset_batch_size,
        wan_dataset.create_missing_captions,
        wan_dataset.caption_strategy,
        wan_dataset.dataset_enable_bucket,
        wan_dataset.dataset_bucket_no_upscale,
        wan_dataset.dataset_cache_directory,
        wan_dataset.generated_toml_path,

        # Wan Model settings
        wan_model_settings.training_mode,
        wan_model_settings.task,
        wan_model_settings.dit,
        wan_model_settings.vae,
        wan_model_settings.t5,
        wan_model_settings.clip,
        wan_model_settings.dit_high_noise,
        wan_model_settings.timestep_boundary,
        wan_model_settings.offload_inactive_dit,
        wan_model_settings.dit_dtype,
        wan_model_settings.text_encoder_dtype,
        wan_model_settings.vae_dtype,
        wan_model_settings.clip_vision_dtype,
        wan_model_settings.fp8_base,
        wan_model_settings.fp8_scaled,
        wan_model_settings.fp8_t5,
        wan_model_settings.blocks_to_swap,
        wan_model_settings.use_pinned_memory_for_block_swap,
        wan_model_settings.vae_tiling,
        wan_model_settings.vae_chunk_size,
        wan_model_settings.vae_cache_cpu,
        wan_model_settings.num_frames,
        wan_model_settings.one_frame,
        wan_model_settings.force_v2_1_time_embedding,

        # training_settings
        training_settings.sdpa,
        training_settings.flash_attn,
        training_settings.sage_attn,
        training_settings.xformers,
        training_settings.split_attn,
        training_settings.max_train_steps,
        training_settings.max_train_epochs,
        training_settings.max_data_loader_n_workers,
        training_settings.persistent_data_loader_workers,
        training_settings.seed,
        training_settings.gradient_checkpointing,
        training_settings.gradient_checkpointing_cpu_offload,
        training_settings.gradient_accumulation_steps,
        training_settings.full_bf16,
        training_settings.full_fp16,
        training_settings.logging_dir,
        training_settings.log_with,
        training_settings.log_prefix,
        training_settings.log_tracker_name,
        training_settings.wandb_run_name,
        training_settings.log_tracker_config,
        training_settings.wandb_api_key,
        training_settings.log_config,
        training_settings.ddp_timeout,
        training_settings.ddp_gradient_as_bucket_view,
        training_settings.ddp_static_graph,

        # Sample generation settings
        sampleSettings.sample_every_n_steps,
        sampleSettings.sample_every_n_epochs,
        sampleSettings.sample_at_first,
        sampleSettings.sample_prompts,
        sampleSettings.sample_output_dir,
        sampleSettings.disable_prompt_enhancement,
        sampleSettings.sample_width,
        sampleSettings.sample_height,
        sampleSettings.sample_num_frames,
        sampleSettings.sample_steps,
        sampleSettings.sample_guidance_scale,
        sampleSettings.sample_seed,
        sampleSettings.sample_negative_prompt,

        # Latent Caching Settings
        latent_caching.caching_latent_device,
        latent_caching.caching_latent_batch_size,
        latent_caching.caching_latent_num_workers,
        latent_caching.caching_latent_skip_existing,
        latent_caching.caching_latent_keep_cache,
        latent_caching.caching_latent_debug_mode,
        latent_caching.caching_latent_console_width,
        latent_caching.caching_latent_console_back,
        latent_caching.caching_latent_console_num_images,

        # Text Encoder Outputs Caching Settings
        text_encoder_caching.caching_teo_text_encoder1,
        text_encoder_caching.caching_teo_text_encoder2,
        text_encoder_caching.caching_teo_text_encoder_dtype,
        text_encoder_caching.caching_teo_device,
        text_encoder_caching.caching_teo_fp8_llm,
        text_encoder_caching.caching_teo_batch_size,
        text_encoder_caching.caching_teo_num_workers,
        text_encoder_caching.caching_teo_skip_existing,
        text_encoder_caching.caching_teo_keep_cache,

        # Optimizer and Scheduler Settings
        optimizer_scheduler.optimizer_type,
        optimizer_scheduler.optimizer_args,
        optimizer_scheduler.learning_rate,
        optimizer_scheduler.max_grad_norm,
        optimizer_scheduler.lr_scheduler,
        optimizer_scheduler.lr_warmup_steps,
        optimizer_scheduler.lr_decay_steps,
        optimizer_scheduler.lr_scheduler_num_cycles,
        optimizer_scheduler.lr_scheduler_power,
        optimizer_scheduler.lr_scheduler_timescale,
        optimizer_scheduler.lr_scheduler_min_lr_ratio,
        optimizer_scheduler.lr_scheduler_type,
        optimizer_scheduler.lr_scheduler_args,

        # Network Settings
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

        # Save/Load Settings
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
        saveLoadSettings.mem_eff_save,

        # HuggingFace Settings
        huggingface.huggingface_repo_id,
        huggingface.huggingface_token,
        huggingface.huggingface_repo_type,
        huggingface.huggingface_repo_visibility,
        huggingface.huggingface_path_in_repo,
        huggingface.save_state_to_huggingface,
        huggingface.resume_from_huggingface,
        huggingface.async_upload,

        # Metadata Settings
        metadata.metadata_author,
        metadata.metadata_description,
        metadata.metadata_license,
        metadata.metadata_tags,
        metadata.metadata_title,
    ]

    # Set up toggle all panels functionality
    def toggle_all_panels(current_state):
        """Toggle all accordion panels open/closed"""
        if current_state == "open":
            new_state = "closed"
            new_button_text = "Open All Panels"
            accordion_states = [gr.Accordion(open=False) for _ in accordions]
            search_value = gr.Textbox(value="")
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
        
        return [new_state, gr.Button(value=new_button_text), search_value, results_visibility, results_content] + accordion_states
    
    toggle_all_btn.click(
        toggle_all_panels,
        inputs=[panels_state],
        outputs=[panels_state, toggle_all_btn, search_input, search_results_row, search_results] + accordions,
        show_progress=False,
    )

    # Set up configuration file handlers
    configuration.button_open_config.click(
        wan_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
    )

    configuration.button_load_config.click(
        wan_gui_actions,
        inputs=[gr.Textbox(value="open_configuration", visible=False), configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status] + settings_list,
        show_progress=False,
        queue=False,
    )

    configuration.button_save_config.click(
        wan_gui_actions,
        inputs=[gr.Textbox(value="save_configuration", visible=False), configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
        outputs=[configuration.config_file_name, configuration.config_status],
        show_progress=False,
        queue=False,
    )

    # Set up training controls
    button_print.click(
        wan_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), configuration.config_file_name, dummy_headless, dummy_true] + settings_list,
        show_progress=False,
    )

    executor.button_run.click(
        wan_gui_actions,
        inputs=[gr.Textbox(value="train_model", visible=False), configuration.config_file_name, dummy_headless, dummy_false] + settings_list,
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

    run_state.change(
        fn=executor.wait_for_training_to_end,
        outputs=[
            executor.button_run,
            executor.stop_row,
            executor.button_stop_training,
            executor.training_status,
        ],
    )
