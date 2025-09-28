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
                        info="Path to parent folder containing subfolders with videos/images. Each subfolder can have format: [repeats]_[name] (e.g., 3_ohwx, 2_style)",
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
                self.generated_toml_path = gr.Textbox(
                    label="Generated TOML Path",
                    value=self.config.get("generated_toml_path", ""),
                    info="Path where the generated TOML configuration will be saved (auto-set when generating)"
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
                    <h3 style="color: #ffffff; margin-top: 0;">📁 Dataset Structure Examples</h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
                        <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #60a5fa; margin-top: 0;">🎬 Video Dataset Structure</h4>
                            <pre style="color: #d1d5db; font-size: 12px; margin: 0; line-height: 1.4;">training_data/
├── videos/
│   ├── video1.mp4 (81 frames, 960×960, 16fps)
│   ├── video2.mp4 (81 frames, 960×960, 16fps)
│   └── video3.mp4 (81 frames, 960×960, 16fps)
├── captions/
│   ├── video1.txt
│   ├── video2.txt
│   └── video3.txt
└── cache_dir/ (auto-generated)</pre>
                        </div>
                        
                        <div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 6px;">
                            <h4 style="color: #c084fc; margin-top: 0;">📸 Image Dataset Structure</h4>
                            <pre style="color: #d1d5db; font-size: 12px; margin: 0; line-height: 1.4;">training_data/
├── images/
│   ├── image1.jpg (960×960)
│   ├── image2.jpg (960×960)
│   └── image3.jpg (960×960)
├── captions/
│   ├── image1.txt
│   ├── image2.txt
│   └── image3.txt
└── cache_dir/ (auto-generated)</pre>
                        </div>
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
                    ("t2v-14B - Text-to-Video 14B (Wan 2.1 Standard)", "t2v-14B"),
                    ("t2v-1.3B - Text-to-Video 1.3B (Wan 2.1 Faster/Smaller)", "t2v-1.3B"),
                    ("i2v-14B - Image-to-Video 14B (Wan 2.1 Standard)", "i2v-14B"),
                    ("t2i-14B - Text-to-Image 14B (Wan 2.1 Standard)", "t2i-14B"),
                    ("flf2v-14B - First-Last-Frame-to-Video 14B (Wan 2.1)", "flf2v-14B"),
                    ("t2v-1.3B-FC - Text-to-Video 1.3B Fun-Control (Wan 2.1)", "t2v-1.3B-FC"),
                    ("t2v-14B-FC - Text-to-Video 14B Fun-Control (Wan 2.1)", "t2v-14B-FC"),
                    ("i2v-14B-FC - Image-to-Video 14B Fun-Control (Wan 2.1)", "i2v-14B-FC"),
                    ("t2v-A14B - Text-to-Video Advanced Dual-Model (Wan 2.2)", "t2v-A14B"),
                    ("i2v-A14B - Image-to-Video Advanced Dual-Model (Wan 2.2)", "i2v-A14B")
                ],
                value=self.config.get("task", "t2v-14B"),
                info="Choose your Wan model variant based on version, use case and hardware capabilities"
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
                    with gr.Column(scale=8):
                        self.dit = gr.Textbox(
                            label="DiT Model Path (Low Noise / Main Model)",
                            info="✨ REQUIRED: Path to main DiT checkpoint (.safetensors). For Wan 2.2 dual training, this serves as the LOW NOISE model (fine details). For single model training, this is the only model used.",
                            placeholder="e.g., /path/to/wan_t2v_14b.safetensors",
                            value=str(self.config.get("dit", ""))
                        )
                    with gr.Column(scale=1):
                        self.dit_button = gr.Button("📂", size="sm")
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=8):
                        self.vae = gr.Textbox(
                            label="VAE Model Path",
                            info="REQUIRED: Use Wan2.1_VAE.pth for ALL models (including Wan 2.2 Advanced). Supports .pth and .safetensors formats.",
                            placeholder="e.g., /path/to/Wan2.1_VAE.pth",
                            value=str(self.config.get("vae", ""))
                        )
                    with gr.Column(scale=1):
                        self.vae_button = gr.Button("📂", size="sm")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=8):
                        self.text_encoder = gr.Textbox(
                            label="T5 Text Encoder Path",
                            info="REQUIRED: Path to T5 text encoder. Supports both .safetensors and .pth formats (recommended: umt5-xxl-enc-bf16.safetensors)",
                            placeholder="e.g., /path/to/umt5-xxl-enc-bf16.safetensors",
                            value=str(self.config.get("text_encoder", ""))
                        )
                    with gr.Column(scale=1):
                        self.text_encoder_button = gr.Button("📂", size="sm")
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=8):
                        self.clip_vision = gr.Textbox(
                            label="CLIP Vision Model Path",
                            info="REQUIRED: Path to CLIP vision encoder. Supports both .safetensors and .pth formats (recommended: models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors)",
                            placeholder="e.g., /path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
                            value=str(self.config.get("clip_vision", ""))
                        )
                    with gr.Column(scale=1):
                        self.clip_vision_button = gr.Button("📂", size="sm")

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
                value=self.config.get("timestep_boundary", 0.0),
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                info="⚡ SWITCH POINT: When to switch from high→low noise model. 0.0=auto-detect, 0.3=switch at 30%, 0.5=switch at 50%. Only used when High Noise DiT path is provided."
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
            self.fp8_vl = gr.Checkbox(
                label="FP8 Vision-Language",
                value=self.config.get("fp8_vl", False),
                info="Enable FP8 for text encoder to reduce memory usage"
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
        self.dit_button.click(fn=lambda: get_file_path(), outputs=[self.dit])
        self.vae_button.click(fn=lambda: get_file_path(), outputs=[self.vae])
        self.text_encoder_button.click(fn=lambda: get_file_path(), outputs=[self.text_encoder])
        self.clip_vision_button.click(fn=lambda: get_file_path(), outputs=[self.clip_vision])
        self.dit_high_noise_button.click(fn=lambda: get_file_path(), outputs=[self.dit_high_noise])

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
                "additional_parameters",
                # Wan Dataset settings
                "dataset_config_mode", "dataset_config", "parent_folder_path", "dataset_resolution_width",
                "dataset_resolution_height", "dataset_caption_extension", "dataset_batch_size",
                "create_missing_captions", "caption_strategy", "dataset_enable_bucket",
                "dataset_bucket_no_upscale", "dataset_cache_directory", "generated_toml_path",
                # Wan Model settings
                "training_mode", "task", "dit", "vae", "text_encoder", "clip_vision",
                "dit_high_noise", "timestep_boundary", "offload_inactive_dit", "dit_dtype",
                "text_encoder_dtype", "vae_dtype", "clip_vision_dtype", "fp8_base", "fp8_scaled",
                "fp8_vl", "blocks_to_swap", "vae_tiling", "vae_chunk_size", "num_frames", "one_frame",
                # training_settings
                "sdpa", "flash_attn", "sage_attn", "xformers", "split_attn", "max_train_steps", "max_train_epochs",
                "max_data_loader_n_workers", "persistent_data_loader_workers", "seed", "gradient_checkpointing",
                "gradient_checkpointing_cpu_offload", "gradient_accumulation_steps", "full_bf16", "full_fp16",
                "logging_dir", "log_with", "log_prefix", "log_tracker_name", "wandb_run_name", "log_tracker_config",
                "wandb_api_key", "log_config", "ddp_timeout", "ddp_gradient_as_bucket_view", "ddp_static_graph",
                # Sample generation settings (from TrainingSettings)
                "sample_every_n_steps", "sample_every_n_epochs", "sample_at_first", "sample_prompts", "sample_width",
                "sample_height", "sample_num_frames", "sample_steps", "sample_guidance_scale", "sample_seed",
                "sample_scheduler", "sample_negative_prompt",
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
                "output_dir", "output_name", "resume", "save_every_n_epochs", "save_every_n_steps", "save_last_n_epochs",
                "save_last_n_epochs_state", "save_last_n_steps", "save_last_n_steps_state", "save_state",
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
                "additional_parameters",
                # Wan Dataset settings
                "dataset_config_mode", "dataset_config", "parent_folder_path", "dataset_resolution_width",
                "dataset_resolution_height", "dataset_caption_extension", "dataset_batch_size",
                "create_missing_captions", "caption_strategy", "dataset_enable_bucket",
                "dataset_bucket_no_upscale", "dataset_cache_directory", "generated_toml_path",
                # Wan Model settings
                "training_mode", "task", "dit", "vae", "text_encoder", "clip_vision",
                "dit_high_noise", "timestep_boundary", "offload_inactive_dit", "dit_dtype",
                "text_encoder_dtype", "vae_dtype", "clip_vision_dtype", "fp8_base", "fp8_scaled",
                "fp8_vl", "blocks_to_swap", "vae_tiling", "vae_chunk_size", "num_frames", "one_frame",
                # training_settings
                "sdpa", "flash_attn", "sage_attn", "xformers", "split_attn", "max_train_steps", "max_train_epochs",
                "max_data_loader_n_workers", "persistent_data_loader_workers", "seed", "gradient_checkpointing",
                "gradient_checkpointing_cpu_offload", "gradient_accumulation_steps", "full_bf16", "full_fp16",
                "logging_dir", "log_with", "log_prefix", "log_tracker_name", "wandb_run_name", "log_tracker_config",
                "wandb_api_key", "log_config", "ddp_timeout", "ddp_gradient_as_bucket_view", "ddp_static_graph",
                # Sample generation settings (from TrainingSettings)
                "sample_every_n_steps", "sample_every_n_epochs", "sample_at_first", "sample_prompts", "sample_width",
                "sample_height", "sample_num_frames", "sample_steps", "sample_guidance_scale", "sample_seed",
                "sample_scheduler", "sample_negative_prompt",
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
                "output_dir", "output_name", "resume", "save_every_n_epochs", "save_every_n_steps", "save_last_n_epochs",
                "save_last_n_epochs_state", "save_last_n_steps", "save_last_n_steps_state", "save_state",
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
        # TODO: Implement actual training logic here
        log.info("Starting Wan model training...")
        if print_only:
            return "Training command printed"
        else:
            return "Training started"
    
    return "Unknown action"


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
        "network_dim", "max_train_epochs"  # NEW: 0 means use max_train_steps instead
        # Removed: "ddp_timeout" (0 = use default 30min timeout - VALID)
        # Removed: "save_last_n_epochs" (0 = keep all epochs - VALID)
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
        'timestep_boundary', 'num_frames'
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

    # Final validation before returning
    result_values = []
    for i, v in enumerate(values):
        if isinstance(v, list):
            # Get parameter name for this index
            param_name = "unknown"
            if i > 0 and i <= len(parameters):
                for j, (k, _) in enumerate(parameters):
                    if j == i-1:  # -1 because values[0] is file_path
                        param_name = k
                        break
            log.warning(f"[VALIDATION] Final validation found list value at index {i} (param: {param_name}): {v}")
        result_values.append(v)

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
        'timestep_boundary', 'num_frames'
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
                "num_cpu_threads_per_process",
                "num_processes",
                "num_machines",
                "multi_gpu",
                "main_process_port",
                "dynamo_backend",
                "dynamo_mode",
                "dynamo_use_fullgraph",
                "dynamo_use_dynamic",
                "extra_accelerate_launch_args",
                "caching_latent_device",
                "caching_latent_batch_size",
                "caching_latent_num_workers",
                "caching_latent_skip_existing",
                "caching_latent_keep_cache",
                "caching_latent_debug_mode",
                "caching_latent_console_width",
                "caching_latent_console_back",
                "caching_latent_console_num_images",
                # ALL Text Encoder Caching parameters removed from exclusion - they should all be saved!
                # "caching_teo_text_encoder",
                # "caching_teo_device",
                # "caching_teo_fp8_vl",
                # "caching_teo_batch_size",
                # "caching_teo_num_workers",
                # "caching_teo_skip_existing",
                # "caching_teo_keep_cache",
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

    # Setup Configuration Files Gradio
    with gr.Accordion("Configuration file Settings", open=True):
        # Show configuration status
        # Check if this is a default config by looking for Wan-specific values
        is_using_defaults = (hasattr(config, 'config') and 
                            config.config.get("task") == "t2v-14B" and 
                            config.config.get("num_frames") == 81 and
                            config.config.get("network_module") == "networks.lora_wan")
        
        if is_using_defaults:
            config_status = """
            [OK] **Wan Models Optimal Defaults Active**
            
            **Key Optimizations Applied:**
            - Task: t2v-14B (Text-to-Video 14B parameters)
            - Optimizer: adamw8bit (memory efficient, recommended)
            - Learning Rate: 1e-4 (recommended for Wan models)
            - Mixed Precision: bf16 (strongly recommended)
            - SDPA Attention: Enabled (fastest for Wan models)
            - Gradient Checkpointing: Enabled (memory savings)
            - Video Frames: 81 (standard for Wan models)
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
        text_encoder_caching = TextEncoderOutputsCaching(headless=headless, config=config)

    # Sample Generation Settings
    sample_accordion = gr.Accordion("Sample Generation Settings", open=False, elem_classes="samples_background")
    accordions.append(sample_accordion)
    with sample_accordion:
        with gr.Row():
            sample_every_n_steps = gr.Number(
                label="Sample Every N Steps",
                value=config.get("sample_every_n_steps", 0),
                minimum=0,
                step=1,
                info="Generate sample videos every N steps (0=disabled, 100-500 recommended)"
            )
            sample_every_n_epochs = gr.Number(
                label="Sample Every N Epochs", 
                value=config.get("sample_every_n_epochs", 0),
                minimum=0,
                step=1,
                info="Generate sample videos every N epochs (0=disabled, 1 recommended)"
            )
        
        with gr.Row():
            sample_at_first = gr.Checkbox(
                label="Sample at First",
                value=config.get("sample_at_first", False),
                info="Generate samples before training starts"
            )
            with gr.Column(scale=8):
                sample_prompts = gr.Textbox(
                    label="Sample Prompts File",
                    value=str(config.get("sample_prompts", "")),
                    info="Path to text file containing prompts for sample generation"
                )
            with gr.Column(scale=1):
                sample_prompts_button = gr.Button("📂", size="sm")
        
        # Sample generation parameters
        with gr.Row():
            sample_width = gr.Number(
                label="Sample Width",
                value=config.get("sample_width", 960),
                minimum=64,
                maximum=4096,
                step=64,
                info="Width for generated sample videos. Optimal resolutions: 960×960, 1280×720, 720×1280"
            )
            sample_height = gr.Number(
                label="Sample Height", 
                value=config.get("sample_height", 960),
                minimum=64,
                maximum=4096,
                step=64,
                info="Height for generated sample videos. Optimal resolutions: 960×960, 1280×720, 720×1280"
            )
        
        with gr.Row():
            sample_num_frames = gr.Number(
                label="Sample Number of Frames",
                value=config.get("sample_num_frames", 81),
                minimum=1,
                maximum=200,
                step=1,
                info="Number of frames in sample videos"
            )
            sample_steps = gr.Number(
                label="Sample Steps",
                value=config.get("sample_steps", 20),
                minimum=1,
                maximum=100,
                step=1,
                info="Number of inference steps for sample generation"
            )
        
        with gr.Row():
            sample_guidance_scale = gr.Number(
                label="Sample Guidance Scale",
                value=config.get("sample_guidance_scale", 7.0),
                minimum=1.0,
                maximum=20.0,
                step=0.1,
                info="Guidance scale for sample generation (higher = stronger prompt adherence)"
            )
            sample_seed = gr.Number(
                label="Sample Seed",
                value=config.get("sample_seed", 99),
                minimum=-1,
                step=1,
                info="Seed for sample generation (-1 = random each time)"
            )
        
        with gr.Row():
            sample_scheduler = gr.Dropdown(
                label="Sample Scheduler",
                choices=["unipc", "euler", "dpmpp_2m"],
                value=config.get("sample_scheduler", "unipc"),
                info="Scheduler for sample generation"
            )
            with gr.Column(scale=8):
                sample_negative_prompt = gr.Textbox(
                    label="Sample Negative Prompt",
                    value=str(config.get("sample_negative_prompt", "")),
                    info="Default negative prompt for all samples"
                )

        # Set up sample prompts file browser
        sample_prompts_button.click(fn=lambda: get_file_path(), outputs=[sample_prompts])

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
        wan_model_settings.text_encoder,
        wan_model_settings.clip_vision,
        wan_model_settings.dit_high_noise,
        wan_model_settings.timestep_boundary,
        wan_model_settings.offload_inactive_dit,
        wan_model_settings.dit_dtype,
        wan_model_settings.text_encoder_dtype,
        wan_model_settings.vae_dtype,
        wan_model_settings.clip_vision_dtype,
        wan_model_settings.fp8_base,
        wan_model_settings.fp8_scaled,
        wan_model_settings.fp8_vl,
        wan_model_settings.blocks_to_swap,
        wan_model_settings.vae_tiling,
        wan_model_settings.vae_chunk_size,
        wan_model_settings.num_frames,
        wan_model_settings.one_frame,

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

        # Sample generation settings (from TrainingSettings)
        training_settings.sample_every_n_steps,
        training_settings.sample_every_n_epochs,
        training_settings.sample_at_first,
        training_settings.sample_prompts,
        sample_width,
        sample_height,
        sample_num_frames,
        sample_steps,
        sample_guidance_scale,
        sample_seed,
        sample_scheduler,
        sample_negative_prompt,

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
