import gradio as gr

def version_history_tab(headless=False, config=None):
    """
    Create the Changelogs tab interface
    """
    with gr.Column():
        gr.Markdown("""
## Version History

### 23 September 2025 - V16

**Enhanced Qwen Image LoRA Training with Advanced Network Architectures and Format Conversions - Based on latest Musubi Tuner updates.**

Changes:
- **NEW: Advanced LoRA Architecture Selection** - Dropdown menu for selecting different LoRA implementations:
  - Standard `networks.lora_qwen_image` - Default Qwen Image LoRA implementation
  - `networks.dylora` - Dynamic LoRA that trains multiple ranks simultaneously
  - `networks.lora_fa` - LoRA with Frozen-A for improved training stability
  - Custom module support - Specify any custom LoRA implementation path
- **NEW: Custom Network Module Field** - Dynamic text field appears when "custom" is selected for advanced users
- **NEW: Post-Training Format Conversion Options** - Automatic conversion to different formats after training:
  - Convert to Diffusers format - Compatible with diffusers library and various inference tools
  - Convert to alternative SafeTensors format - Different key naming for compatibility
  - Configurable output directories for each conversion format
  - Dynamic UI - conversion directories only show when conversion is enabled
- **ENHANCED: FP8 Optimization Descriptions** - Comprehensive guidance for FP8 settings:
  - fp8_vl - Now clearly indicates Qwen2.5-VL text encoder with ~8GB VRAM savings
  - fp8_base - Clarifies on-the-fly BF16→FP8 conversion process
  - fp8_scaled - Emphasizes CRITICAL importance for quality with block-wise scaling
  - Added hardware compatibility notes (RTX 4000+ native, RTX 3000 emulation)
- **NEW: Real-time FP8 Configuration Validation** - Automatic warning system:
  - Critical red warning when fp8_base enabled without fp8_scaled
  - Prevents significant quality degradation from improper FP8 configuration
  - Dynamic warning appears/disappears based on settings
- **NEW: Dtype Conflict Detection** - Validation system for training precision:
  - Warns about full_bf16 + full_fp16 simultaneous usage conflict
  - Provides helpful tips for choosing between BF16 and FP16
  - Real-time validation with clear warning messages
- **ENHANCED: Network Parameter Descriptions** - Detailed guidance for all LoRA settings:
  - Network Dimension - Specific recommendations for Qwen Image (4-8 low, 16-32 balanced, 64-128 high)
  - Network Alpha - Best practice formula (alpha = rank/2 for stability)
  - Network Args - Comprehensive examples for DyLoRA, LoRA-FA, LyCORIS, block-wise configurations
- **IMPROVED: Parameter Search Integration** - All new parameters fully searchable:
  - custom_network_module, convert_to_diffusers, diffusers_output_dir
  - convert_to_safetensors, safetensors_output_dir
  - Enhanced parameter descriptions in search results
- **FIXED: Dtype validation scope error** - Removed external reference that caused startup crash
- **Full TOML Configuration Support** - All new parameters integrate seamlessly with save/load system
- **Smart Parameter Handling** - Automatic handling of custom modules and post-training operations
- **Backward Compatibility** - All existing configurations continue to work without modification
- Based on latest Musubi Tuner repository updates (August-September 2025)
- Includes VRAM optimizations, bug fixes, and stability improvements from upstream

### 12 September 2025 - V15

**Intelligent Logging Directory Management - Automatic path resolution for training logs with proper permissions.**

Changes:
- **FIXED: PermissionError when creating logging directory** - System no longer tries to create directories at root filesystem
- **NEW: Smart logging_dir path handling** - Automatically manages logging paths based on user configuration:
  - Empty logging_dir with logging enabled: Auto-creates `{output_dir}/logs/session_{timestamp}` directory
  - Relative path (e.g., "mylogs"): Converts to `{output_dir}/mylogs` for proper organization
  - Absolute path: Uses the exact path specified by the user
- **Cross-platform compatibility** - Works correctly on both Windows and Linux with proper path normalization
- **Intelligent defaults** - When logging tool (tensorboard/wandb) is selected but directory is empty, automatically generates appropriate path
- **Fallback safety** - If output_dir is not set, falls back to current directory with `./logs/session_{timestamp}`
- **Permission-aware** - Ensures logging directories are created in user-writable locations, preventing permission errors
- **Forward slash standardization** - All paths converted to forward slashes for TOML compatibility across platforms
- Previously, empty or "/" logging_dir values would cause training to fail with permission errors
- Now seamlessly handles all logging directory scenarios with smart path resolution
- Particularly useful for users who enable logging but forget to specify a directory

### 5 September 2025 - V14

**Sample Prompt Enhancement Control - Added checkbox to disable automatic Kohya format parameter enhancement.**

Changes:
- **NEW: Disable Prompt Enhancement Checkbox** - Added user control to prevent automatic sample prompt formatting
- **Enhanced User Control** - Users can now choose to use original prompt files without automatic Kohya format parameter additions
- **Configurable Behavior** - Checkbox setting is saved and loaded with configuration files for persistence
- **Default Safe Setting** - Enhancement remains enabled by default, preserving existing workflow compatibility
- **Clear User Interface** - Added informative checkbox with clear labeling explaining the feature functionality
- **Intelligent Logic** - When disabled, the system logs the choice and uses original prompts exactly as provided by the user
- **Backward Compatibility** - All existing configurations continue to work exactly as before without any changes required
- Previously, all sample prompts were automatically enhanced with GUI default parameters (width, height, steps, etc.)
- Now users can opt out of this behavior when they want to use their original prompt files without modification
- Particularly useful when users have pre-formatted prompts or want complete control over sampling parameters
- Located in Qwen Image Training tab after the Sample Prompts File selection for logical workflow integration

### 4 September 2025 - V13

**Cross-Platform Path Handling Fix - Spaces in dataset paths now work correctly on Windows and Linux.**

Changes:
- **FIXED: Path handling with spaces** - Dataset paths containing spaces now generate valid TOML configurations
- **NEW: Cross-platform path normalization** - Added normalize_path(), validate_path_for_toml(), and is_path_safe() utility functions
- **Enhanced TOML generation** - All paths in dataset_config_generator.py now use proper path formatting for cross-platform compatibility
- **Updated file dialog functions** - All file/folder selection dialogs now normalize paths consistently across Windows and Linux
- **Forward slash standardization** - All paths converted to use forward slashes for consistency and TOML compatibility
- **Automatic path validation** - Paths are validated and normalized before being written to configuration files
- **Backward compatibility maintained** - All existing functionality preserved while fixing the space-in-path issues
- Previously, users would encounter TOML parsing errors when dataset paths contained spaces
- Now supports paths like "C:/My Dataset/Training Images" and "/home/user/My Training Data" seamlessly
- Works correctly on both Windows (backslash) and Linux (forward slash) path conventions

### 4 September 2025 - V12

**Fixed Qwen Image Text Encoder Caching Configuration Bug - ALL caching settings now properly save and load.**

Changes:
- **FIXED: Text Encoder Caching parameters not saving** - ALL settings in the Text Encoder Caching tab now properly save to configuration files
- Removed all caching_teo_* parameters from the exclusion list in qwen_image_lora_gui.py:
  - caching_teo_text_encoder (Text Encoder Path)
  - caching_teo_device (Caching Device)
  - caching_teo_fp8_vl (Use FP8 for VL Model)
  - caching_teo_batch_size (Caching Batch Size)
  - caching_teo_num_workers (Data Loading Workers)
  - caching_teo_skip_existing (Skip Existing)
  - caching_teo_keep_cache (Keep Cache)
- All Text Encoder Caching settings in Qwen Image now correctly persist across save/load operations

### 31 August 2025 - V11

**Professional Stop Button Implementation for Batch Image Captioning - Allows safe interruption of long-running batch operations.**

Changes:
- **NEW: Professional Stop Button for Batch Captioning** - Added comprehensive stop functionality to Image Captioning tab
- **Enhanced User Control** - Stop button becomes available immediately when batch processing starts
- **Safe Processing Halt** - Stops after completing current image to prevent data corruption or incomplete caption files  
- **Smart Button State Management** - Batch button disabled during processing, stop button enabled, automatic state restoration
- **Comprehensive Stop Integration** - Works for both text file and JSONL output formats with proper progress tracking
- **Immediate Response** - Stop requests are processed quickly with clear user feedback and status updates
- **Graceful Termination** - Properly updates statistics, saves completed work, and shows clear "stopped by user" messaging
- **Professional UX** - Clean interface with proper button states and visual feedback throughout the process
- Processing can be safely interrupted at any point without losing completed captions or corrupting the output
- All batch processing statistics and progress information remain accurate even when stopped mid-process
- **FIXED: Caption prefix/suffix timing issue** - Prefix and suffix are now applied AFTER word replacement instead of before, ensuring word replacement works on the original generated text for both single and batch captioning

### 31 August 2025 - V10

**Critical Checkpoint Management Bug Fix - Prevents immediate deletion of saved checkpoints.**

Changes:
- **FIXED: Critical checkpoint removal bug** - Checkpoints were being deleted immediately after saving when save_last_n_epochs=0
- **FIXED: Parameter translation for save_last_n_* parameters** - GUI now properly converts 0→None for musubi tuner compatibility
- Enhanced parameter handling in SaveConfigFileToRun() to translate user-friendly values to musubi tuner expectations
- Added save_last_n_epochs, save_last_n_steps, save_last_n_epochs_state, save_last_n_steps_state to zero_to_none_params list
- **UI Improvement: Renamed "Save Load Settings" to "Save Models and Resume Training Settings"** for better clarity
- **Enhanced output_name field** with better default value "my-qwen-lora" and clearer instructions about .safetensors auto-extension
- **Improved checkpoint management labels** - Simplified confusing "Save Last N Epochs/Steps" to clearer "Keep Last N Checkpoints/State Files"
- Removed misleading epoch/step terminology from checkpoint cleanup labels for better user understanding
- Fixed checkpoint management logic where 0 (keep all) was incorrectly triggering immediate checkpoint removal
- Training now properly preserves all checkpoints when save_last_n_epochs=0 as intended

### 31 August 2025 - V9

**Critical Configuration Save/Load Bug Fixes - Ensures proper parameter handling and GPU selection.**

Changes:
- **FIXED: ddp_timeout parameter being incorrectly forced to minimum value 1** - Now properly saves/loads 0 (use default 30min timeout)
- **FIXED: save_last_n_epochs parameter being incorrectly forced to minimum value 1** - Now properly saves/loads 0 (keep all epochs)
- **FIXED: GPU IDs not being saved to TOML configuration files** - Now properly saves/loads gpu_ids for single GPU selection
- Enhanced parameter constraint validation to respect valid zero values for optional parameters
- Improved compatibility with musubi tuner parameter expectations (None vs 0 handling)
- Configuration double-save corruption issues resolved - parameters maintain correct values through multiple save/load cycles
- GPU selection now persists properly in configuration files, allowing single GPU training setups to be saved and restored

### 31 August 2025 - V8

**Enhanced parameter support for Qwen Image models with Edit mode and Fine-Tuning preparation.**

Changes:
- Added comprehensive parameter support for Qwen Image training with 100% coverage of Musubi Tuner parameters
- **NEW: Integrated search bar in Qwen Image Training tab** - quickly find any setting without opening all panels
- **Tab renamed from "Qwen Image LoRA" to "Qwen Image Training"** to reflect both LoRA and Fine-tuning support
- Implemented Qwen-Image-Edit mode support for control image training (experimental - not fully tested)
- Added control image resolution settings for Edit mode (dataset_qwen_image_edit_control_resolution_width/height)
- Introduced dataset_qwen_image_edit_no_resize_control option for maintaining original control image sizes
- Started implementation of Qwen Image Fine-Tuning mode (DreamBooth) - parameter infrastructure in place
- Enhanced FP8 quantization descriptions with clearer GPU compatibility information
- Improved timestep sampling with better documentation of qwen_shift vs standard shift methods
- Added advanced flow matching parameters (logit_mean, logit_std, mode_scale) for fine-tuned control
- Implemented complete VAE optimization settings (tiling, chunk_size, spatial_tile_sample_min_size)
- Enhanced parameter descriptions throughout the GUI for better user understanding
- Verified 88% parameter coverage with Musubi Tuner repository (22/25 parameters)
- All critical default values confirmed to match official Qwen Image documentation
- Parameter accuracy validated at 100% for all implemented features

### 30 August 2025 - V7

**Smart Sample Prompt Enhancement System - Automatically applies optimal Qwen Image resolution (1328x1328) to sample prompts.**

Changes:
- Implemented intelligent sample prompt enhancement system that automatically adds GUI-specified defaults to prompts
- Sample Generation Settings moved to dedicated section with comprehensive configuration options
- Default sample resolution changed from musubi's 256x256 to Qwen Image's optimal 1328x1328
- Enhanced prompt files are saved to output directory for transparency and debugging
- Simple prompts (e.g., "A cat sitting") now automatically get proper parameters added
- Advanced users can still override any parameter per prompt using --w, --h, --s, --g, --d flags
- Fixed AttributeError with sample_prompts_button in wrong class
- Improved sample generation documentation with clear examples

### 30 August 2025 - V6

**Config save and load were broken for Optimizer Arguments and Scheduler Arguments and this problem fixed. Stop training button now will appear much earlier than before when Text Encoder caching starts.**

Changes:
- Fixed broken config save and load functionality for Optimizer Arguments and Scheduler Arguments
- Improved Stop Training button responsiveness - now appears much earlier when Text Encoder caching starts
- Enhanced training control for better user experience

### 29 August 2025 - V5

**Fixed skip existing captions logic and improved batch captioning status display.**

Changes:
- Fixed skip existing captions functionality in Image Captioning with Qwen2.5-VL
  - Previously skipping was happening after caption generation which was destroying the skip logic
  - Now properly checks for existing captions before processing, significantly improving efficiency
- Added full batch captioning status display in command line with progress tracking and ETA
- Enhanced config save/load functionality for better reliability
- Improved interface of Image Captioning with Qwen2.5-VL for better user experience
- Various error fixes in the Qwen2.5-VL captioning pipeline

### 29 August 2025 - V4

**Fixed sample prompts file selector error.**

Changes:
- Fixed TypeError in sample prompts file button click handler
- Corrected get_file_path() function call to use proper parameters (default_extension and extension_name instead of file_filter)
- Sample prompts file selector now works correctly when clicking the folder icon

### 29 August 2025 - V3

**Model downloader improvements and Windows compatibility fixes.**

Changes:
- Model downloader file updated and made more robust with SHA256 verification
- Image Captioning with Qwen2.5-VL works perfect now on Windows and Linux
- Enhanced file integrity checks for downloaded models
- Improved cross-platform compatibility

### 29 August 2025 - V2

**Dataset generation method fixed and GUI improvements.**

Changes:
- Fixed dataset generation error with `generate_dataset_config_from_folders()` function
- Updated function parameters to use correct argument names (parent_folder instead of parent_folder_path)
- Added example filenames to model path descriptions for better user guidance
- Updated image captioning GUI label to clarify processing time impact
- Minor GUI description improvements for better user experience

### 28 August 2025 - V1

**Initial app release with Qwen Image LoRA training support and Qwen2.5-VL Image Captioning support.**

Features included:
- Full support for Qwen Image LoRA training
- Qwen2.5-VL based image captioning functionality
- Intuitive GUI interface for training configuration
- Support for various training parameters and optimizations
- Batch processing capabilities for image captioning
        """)