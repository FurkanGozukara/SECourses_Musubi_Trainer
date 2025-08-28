import gradio as gr

def version_history_tab(headless=False, config=None):
    """
    Create the Changelogs tab interface
    """
    with gr.Column():
        gr.Markdown("""
## Version History

### 28 August 2025 - V1

**Initial app release with Qwen Image LoRA training support and Qwen2.5-VL Image Captioning support.**

Features included:
- Full support for Qwen Image LoRA training
- Qwen2.5-VL based image captioning functionality
- Intuitive GUI interface for training configuration
- Support for various training parameters and optimizations
- Batch processing capabilities for image captioning
        """)