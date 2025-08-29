import gradio as gr

def version_history_tab(headless=False, config=None):
    """
    Create the Changelogs tab interface
    """
    with gr.Column():
        gr.Markdown("""
## Version History

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