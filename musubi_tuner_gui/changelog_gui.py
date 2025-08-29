import gradio as gr

def version_history_tab(headless=False, config=None):
    """
    Create the Changelogs tab interface
    """
    with gr.Column():
        gr.Markdown("""
## Version History

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