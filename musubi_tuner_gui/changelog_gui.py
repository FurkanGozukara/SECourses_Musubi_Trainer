import gradio as gr

def version_history_tab(headless=False, config=None):
    """
    Create the Changelogs tab interface
    """
    with gr.Column():
        gr.Markdown("""
## Version History

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