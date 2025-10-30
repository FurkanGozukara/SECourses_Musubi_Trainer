import gradio as gr
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

from .class_image_captioning import ImageCaptioning, DEFAULT_PROMPT
from .class_gui_config import GUIConfig
from .common_gui import (
    get_file_path,
    get_folder_path,
    get_saveasfile_path,
    get_file_path_or_save_as,
    SaveConfigFile,
)
from .custom_logging import setup_logging

log = setup_logging()


class ImageCaptioningTab:
    """Image Captioning Tab GUI implementation"""
    
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.captioning = ImageCaptioning(headless, config)
        self.initialize_ui_components()
    
    def initialize_ui_components(self) -> None:
        """Initialize all UI components"""
        
        # Main container
        with gr.Column():
            gr.Markdown("# Image Captioning with Qwen2.5-VL")
            
            # Single Image Captioning Section (moved to very top)
            gr.Markdown("## Single Image Captioning")
            with gr.Row():
                with gr.Column(scale=1):
                    self.single_image_input = gr.Image(
                        label="Upload Image",
                        type="filepath",
                        height=400,
                    )
                    
                    with gr.Row():
                        self.caption_single_button = gr.Button(
                            "Generate Caption",
                            variant="primary",
                            size="lg"
                        )
                
                with gr.Column(scale=1):
                    self.single_caption_output = gr.Textbox(
                        label="Generated Caption",
                        lines=10,
                        max_lines=20,
                        placeholder="Generated caption will appear here...",
                    )
                    
                    with gr.Row():
                        self.copy_caption_button = gr.Button("Copy Caption")
                        self.save_caption_button = gr.Button("Save as Text File")
                    with gr.Row():
                        gr.Markdown("You can also test and use our amazing Joy Caption App: https://www.patreon.com/posts/118827960")
                    
                    # Add unload button and auto-unload checkbox
                    with gr.Row():
                        self.unload_model_button = gr.Button(
                            "🗑️ Unload Model from VRAM",
                            variant="secondary",
                            size="sm"
                        )
                        self.auto_unload_after_caption = gr.Checkbox(
                            label="Auto-unload model after captioning",
                            info="Automatically unload model from VRAM after single/batch captioning completes",
                            value=self.config.get("image_captioning.auto_unload_after_caption", False),
                        )
            
            
            
            # Top Row: Model Configuration and Caption Configuration in left, Configuration and Custom Prompt in right
            with gr.Row():
                # Left Column: Configuration (moved up) and Model Configuration
                with gr.Column(scale=1):
                    # Configuration panel moved above Model Configuration
                    with gr.Accordion("Configuration", open=True):
                        with gr.Row():
                            self.config_file_path = gr.Textbox(
                                label="Configuration File",
                                placeholder="Path to save/load configuration",
                                value="",
                                scale=3
                            )
                            
                            self.config_file_button = gr.Button(
                                "📁",
                                size="sm",
                                scale=0
                            )
                        
                        with gr.Row():
                            self.save_config_button = gr.Button("Save Configuration")
                            self.load_config_button = gr.Button("Load Configuration")
                        
                        self.config_status = gr.Textbox(
                            label="Configuration Status",
                            value="",
                            interactive=False,
                        )
                    
                    with gr.Accordion("Model Configuration", open=True):
                        with gr.Row():
                            self.model_path = gr.Textbox(
                                label="Qwen2.5-VL Model Path",
                                info="Path to Qwen2.5-VL text encoder model (qwen_2.5_vl_7b.safetensors from Comfy-Org/Qwen-Image_ComfyUI)",
                                placeholder="e.g., /path/to/qwen_2.5_vl_7b.safetensors",
                                value=self.config.get("image_captioning.model_path", ""),
                                scale=3
                            )
                            
                            self.model_path_button = gr.Button(
                                "📁",
                                size="sm",
                                scale=0
                            )
                        
                        with gr.Row():
                            self.fp8_vl = gr.Checkbox(
                                label="Use FP8 Precision",
                                info="Enable FP8 quantization to reduce VRAM usage. Recommended for lower VRAM GPUs",
                                value=self.config.get("image_captioning.fp8_vl", True),
                            )
                            
                            self.max_size = gr.Number(
                                label="Max Image Size",
                                info="Maximum image size for processing. Higher values need more VRAM",
                                value=self.config.get("image_captioning.max_size", 1280),
                                minimum=256,
                                maximum=2048,
                                step=64,
                            )
                        
                        self.model_status = gr.Textbox(
                            label="Model Status",
                            value="Model will auto-load when captioning starts",
                            interactive=False,
                        )
                    
                    # Caption Configuration kept here under Model Configuration
                    with gr.Accordion("Caption Configuration", open=True):
                        with gr.Row():
                            self.max_new_tokens = gr.Number(
                                label="Max New Tokens - More Tokens Longer Processing Time",
                                info="Maximum number of tokens to generate for each caption",
                                value=self.config.get("image_captioning.max_new_tokens", 1024),
                                minimum=64,
                                maximum=4096,
                                step=64,
                            )
                        
                        # Add Generation Parameters section
                        gr.Markdown("### Generation Parameters")
                        with gr.Row():
                            self.do_sample = gr.Checkbox(
                                label="Do Sample",
                                info="Whether to use sampling for text generation (recommended: True)",
                                value=self.config.get("image_captioning.do_sample", True),
                            )
                            
                            self.temperature = gr.Slider(
                                label="Temperature",
                                info="Controls randomness (0.1=focused, 1.0=creative)",
                                value=self.config.get("image_captioning.temperature", 0.7),
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                            )
                        
                        with gr.Row():
                            self.top_k = gr.Number(
                                label="Top K",
                                info="Number of top tokens to consider (higher=more diverse)",
                                value=self.config.get("image_captioning.top_k", 50),
                                minimum=1,
                                maximum=100,
                                step=1,
                            )
                            
                            self.top_p = gr.Slider(
                                label="Top P",
                                info="Cumulative probability threshold (0.95 recommended)",
                                value=self.config.get("image_captioning.top_p", 0.95),
                                minimum=0.1,
                                maximum=1.0,
                                step=0.05,
                            )
                        
                        with gr.Row():
                            self.repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                info="Penalty for repeating tokens (1.0=no penalty, higher=less repetition)",
                                value=self.config.get("image_captioning.repetition_penalty", 1.05),
                                minimum=1.0,
                                maximum=2.0,
                                step=0.05,
                            )
                        
                
                # Right Column: Caption Modifications and Custom Prompt
                with gr.Column(scale=1):
                    # Caption Modifications panel (moved from Caption Configuration)
                    with gr.Accordion("Caption Modifications", open=True):
                        with gr.Row():
                            self.prefix = gr.Textbox(
                                label="Caption Prefix",
                                info="Text to add before each generated caption",
                                placeholder="e.g., 'A detailed image showing '",
                                value=self.config.get("image_captioning.prefix", ""),
                            )
                            
                            self.suffix = gr.Textbox(
                                label="Caption Suffix", 
                                info="Text to add after each generated caption",
                                placeholder="e.g., ' with high quality details.'",
                                value=self.config.get("image_captioning.suffix", ""),
                            )
                        
                        self.replace_words = gr.Textbox(
                            label="Replace Words",
                            info="Word pairs to replace in captions. Format: orgword;replaceword (each line is one replacement pair). Applied after prefix/suffix.",
                            placeholder="Line 1: man;ohwx man\nLine 2: person;ohwx person\nLine 3: he;ohwx man\nLine 4: woman;ohwx woman\nLine 5: girl;ohwx woman",
                            value=self.config.get("image_captioning.replace_words", ""),
                            lines=5,
                            max_lines=5,
                        )
                        
                        with gr.Row():
                            self.replace_case_insensitive = gr.Checkbox(
                                label="Case Insensitive Replace",
                                info="Replace words regardless of case (e.g., 'Man', 'man', 'MAN' all match)",
                                value=self.config.get("image_captioning.replace_case_insensitive", True)
                            )
                            
                            self.replace_whole_words_only = gr.Checkbox(
                                label="Replace Whole Words Only",
                                info="Only replace complete words (e.g., 'he' won't match 'her' or 'the')",
                                value=self.config.get("image_captioning.replace_whole_words_only", True)
                            )
                    
                    # Custom Prompt with buttons
                    self.custom_prompt = gr.Textbox(
                        label="Custom Prompt",
                        info="Custom prompt for caption generation. Use \\n for newlines. Leave empty to use default",
                        placeholder="Enter custom prompt or leave empty for default...",
                        value=self.config.get("image_captioning.custom_prompt", ""),
                        lines=8,
                        max_lines=12,
                    )
                    
                    with gr.Row():
                        self.show_default_prompt = gr.Button("Show Default Prompt")
                        self.clear_prompt = gr.Button("Clear Prompt")
                    
                    # Unified Batch Captioning panel
                    with gr.Accordion("Batch Captioning", open=True):
                        with gr.Row():
                            self.batch_image_dir = gr.Textbox(
                                label="Image Directory",
                                info="Directory containing images to caption",
                                placeholder="e.g., /path/to/images",
                                value=self.config.get("image_captioning.batch_image_dir", ""),
                                scale=3
                            )
                            
                            self.batch_image_dir_button = gr.Button(
                                "📁",
                                size="sm",
                                scale=0
                            )
                        
                        with gr.Row():
                            self.batch_output_folder = gr.Textbox(
                                label="Output Folder (optional)",
                                info="Where to save caption files. Leave empty to save alongside images",
                                placeholder="e.g., /path/to/output (optional)",
                                value=self.config.get("image_captioning.batch_output_folder", ""),
                                scale=3
                            )
                            
                            self.batch_output_folder_button = gr.Button(
                                "📁",
                                size="sm",
                                scale=0
                            )
                        
                        with gr.Row():
                            self.extension = gr.Textbox(
                                label="Caption File Extension",
                                info="Extension for saved caption files",
                                value=self.config.get("image_captioning.extension", ".txt"),
                            )
                            
                            self.output_format = gr.Radio(
                                label="Output Format",
                                choices=["text", "jsonl"],
                                value=self.config.get("image_captioning.output_format", "text"),
                                info="text: Creates .txt file for each image | jsonl: Creates single JSONL file with all captions",
                            )
                        
                        with gr.Row(visible=False) as self.jsonl_output_row:
                            self.jsonl_output_file = gr.Textbox(
                                label="JSONL Output File",
                                info="Path to save JSONL file (required when output format is JSONL)",
                                placeholder="e.g., /path/to/captions.jsonl",
                                value=self.config.get("image_captioning.jsonl_output_file", ""),
                                scale=3
                            )
                            
                            self.jsonl_output_button = gr.Button(
                                "📁",
                                size="sm",
                                scale=0
                            )
                        
                        with gr.Row():
                            self.scan_subfolders = gr.Checkbox(
                                label="Scan Subfolders",
                                info="Include images from all subfolders recursively",
                                value=self.config.get("image_captioning.scan_subfolders", False)
                            )
                            
                            self.overwrite_existing_captions = gr.Checkbox(
                                label="Overwrite Existing Captions",
                                info="Replace existing caption files. If unchecked, skips images that already have captions",
                                value=self.config.get("image_captioning.overwrite_existing_captions", False)
                            )
                        
                        with gr.Row():
                            self.copy_images = gr.Checkbox(
                                label="Copy Images",
                                info="Copy images to output folder (preserves folder structure when scanning subfolders)",
                                value=self.config.get("image_captioning.copy_images", False)
                            )
                        
                        with gr.Row():
                            self.batch_caption_button = gr.Button(
                                "Start Batch Captioning",
                                variant="primary",
                                size="lg",
                                scale=2
                            )
                            
                            self.stop_batch_button = gr.Button(
                                "⏹️ Stop Processing",
                                variant="stop",
                                size="lg",
                                interactive=False,
                                scale=1
                            )
                        
                        self.batch_status = gr.Textbox(
                            label="Batch Progress/Status",
                            value="Ready to start batch captioning",
                            interactive=False,
                            lines=8,
                            max_lines=12,
                        )
        
        # Event handlers
        self.setup_event_handlers()
    
    def setup_event_handlers(self) -> None:
        """Set up all event handlers"""
        
        # File/folder picker buttons
        self.model_path_button.click(
            fn=lambda: get_file_path(file_path="", default_extension=".safetensors"),
            outputs=self.model_path,
        )
        
        self.batch_image_dir_button.click(
            fn=get_folder_path,
            outputs=self.batch_image_dir,
        )
        
        self.batch_output_folder_button.click(
            fn=get_folder_path,
            outputs=self.batch_output_folder,
        )
        
        self.jsonl_output_button.click(
            fn=lambda: get_saveasfile_path(file_path="", defaultextension=".jsonl"),
            outputs=self.jsonl_output_file,
        )
        
        # Config file button - now handles both save and load with auto-load
        self.config_file_button.click(
            fn=self.handle_config_file_selection,
            inputs=self.config_file_path,
            outputs=[
                self.config_file_path,
                # Auto-load outputs if file exists
                self.model_path,
                self.fp8_vl,
                self.max_size,
                self.max_new_tokens,
                self.prefix,
                self.suffix,
                self.replace_words,
                self.replace_case_insensitive,
                self.replace_whole_words_only,
                self.custom_prompt,
                self.batch_image_dir,
                self.output_format,
                self.jsonl_output_file,
                self.batch_output_folder,
                self.scan_subfolders,
                self.copy_images,
                self.overwrite_existing_captions,
                # Generation parameters
                self.do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.auto_unload_after_caption,
                self.config_status,
            ],
        )
        
        
        # Prompt management
        self.show_default_prompt.click(
            fn=lambda: DEFAULT_PROMPT,
            outputs=self.custom_prompt,
        )
        
        self.clear_prompt.click(
            fn=lambda: "",
            outputs=self.custom_prompt,
        )
        
        # Single image captioning
        self.caption_single_button.click(
            fn=self.caption_single_image,
            inputs=[
                self.single_image_input,
                self.max_new_tokens,
                self.custom_prompt,
                self.max_size,
                self.fp8_vl,
                self.prefix,
                self.suffix,
                self.replace_words,
                self.replace_case_insensitive,
                self.replace_whole_words_only,
                self.model_path,
                # Generation parameters
                self.do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.auto_unload_after_caption,  # Add auto-unload setting
            ],
            outputs=[self.single_caption_output, self.model_status],
        )
        
        # Unload model button
        self.unload_model_button.click(
            fn=self.unload_model,
            outputs=self.model_status,
        )
        
        # Caption utilities
        self.copy_caption_button.click(
            fn=self.copy_caption_to_clipboard,
            inputs=self.single_caption_output,
            outputs=self.config_status,
        )
        
        self.save_caption_button.click(
            fn=self.save_caption_as_file,
            inputs=[self.single_caption_output, self.single_image_input],
            outputs=self.config_status,
        )
        
        # Output format change handler
        self.output_format.change(
            fn=self.update_output_format_visibility,
            inputs=self.output_format,
            outputs=self.jsonl_output_row,
        )
        
        # Batch captioning
        self.batch_caption_button.click(
            fn=self.batch_caption_wrapper,
            inputs=[
                self.batch_image_dir,
                self.output_format,
                self.jsonl_output_file,
                self.batch_output_folder,
                self.max_new_tokens,
                self.custom_prompt,
                self.max_size,
                self.fp8_vl,
                self.prefix,
                self.suffix,
                self.replace_words,
                self.replace_case_insensitive,
                self.replace_whole_words_only,
                self.scan_subfolders,
                self.copy_images,
                self.overwrite_existing_captions,
                self.model_path,
                # Generation parameters
                self.do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.auto_unload_after_caption,  # Add auto-unload setting
            ],
            outputs=[self.batch_status, self.model_status, self.batch_caption_button, self.stop_batch_button],
        )
        
        # Stop batch processing
        self.stop_batch_button.click(
            fn=self.stop_batch_processing,
            outputs=[self.batch_status, self.batch_caption_button, self.stop_batch_button],
        )
        
        # Configuration management
        self.save_config_button.click(
            fn=self.save_configuration,
            inputs=[
                self.config_file_path,
                self.model_path,
                self.fp8_vl,
                self.max_size,
                self.max_new_tokens,
                self.prefix,
                self.suffix,
                self.replace_words,
                self.replace_case_insensitive,
                self.replace_whole_words_only,
                self.custom_prompt,
                self.batch_image_dir,
                self.output_format,
                self.jsonl_output_file,
                self.batch_output_folder,
                self.scan_subfolders,
                self.copy_images,
                self.overwrite_existing_captions,
                # Generation parameters
                self.do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.auto_unload_after_caption,
            ],
            outputs=[self.config_file_path, self.config_status],
        )
        
        self.load_config_button.click(
            fn=self.load_configuration,
            inputs=self.config_file_path,
            outputs=[
                self.model_path,
                self.fp8_vl,
                self.max_size,
                self.max_new_tokens,
                self.prefix,
                self.suffix,
                self.replace_words,
                self.replace_case_insensitive,
                self.replace_whole_words_only,
                self.custom_prompt,
                self.batch_image_dir,
                self.output_format,
                self.jsonl_output_file,
                self.batch_output_folder,
                self.scan_subfolders,
                self.copy_images,
                self.overwrite_existing_captions,
                # Generation parameters
                self.do_sample,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                self.auto_unload_after_caption,
                self.config_status,
            ],
        )
    
    def ensure_model_loaded(self, model_path: str, max_size: int, fp8_vl: bool) -> Tuple[bool, str]:
        """Ensure model is loaded, loading it if necessary"""
        if not model_path:
            return False, "Please provide a model path"
        
        # Check if model is already loaded with same parameters
        if (self.captioning.model_loaded and 
            hasattr(self.captioning, '_last_model_path') and
            hasattr(self.captioning, '_last_max_size') and
            hasattr(self.captioning, '_last_fp8_vl') and
            self.captioning._last_model_path == model_path and
            self.captioning._last_max_size == max_size and
            self.captioning._last_fp8_vl == fp8_vl):
            return True, "Model already loaded"
        
        # Load the model
        success, message = self.captioning.load_model_and_processor(model_path, max_size, fp8_vl)
        
        if success:
            # Store parameters for future comparison
            self.captioning._last_model_path = model_path
            self.captioning._last_max_size = max_size
            self.captioning._last_fp8_vl = fp8_vl
        
        return success, message
    
    def unload_model(self) -> str:
        """Unload the model from VRAM"""
        success, message = self.captioning.unload_model()
        if success:
            return message
        else:
            return f"Error: {message}"
    
    def caption_single_image(
        self,
        image_path: Optional[str],
        max_new_tokens: int,
        custom_prompt: str,
        max_size: int,
        fp8_vl: bool,
        prefix: str,
        suffix: str,
        replace_words: str,
        replace_case_insensitive: bool,
        replace_whole_words_only: bool,
        model_path: str,
        # Generation parameters
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        auto_unload_after_caption: bool,
    ) -> Tuple[str, str]:
        """Generate caption for a single image with auto-loading"""
        if not image_path:
            return "Please upload an image first", "Ready"
        
        # Auto-load model if needed
        model_loaded, load_message = self.ensure_model_loaded(model_path, max_size, fp8_vl)
        if not model_loaded:
            return f"Failed to load model: {load_message}", f"Model load failed: {load_message}"
        
        prompt = custom_prompt if custom_prompt.strip() else DEFAULT_PROMPT
        
        success, caption = self.captioning.generate_caption(
            image_path, max_new_tokens, prompt, max_size, fp8_vl, prefix, suffix, replace_words,
            replace_case_insensitive, replace_whole_words_only, do_sample, temperature, top_k, top_p, repetition_penalty
        )
        
        if success:
            # Auto-unload if requested
            if auto_unload_after_caption:
                unload_success, unload_msg = self.captioning.unload_model()
                if unload_success:
                    return caption, "Caption generated. Model unloaded from VRAM"
                else:
                    return caption, f"Caption generated. Warning: {unload_msg}"
            else:
                return caption, "Model loaded and ready"
        else:
            return f"Error: {caption}", "Model loaded but caption generation failed"
    
    def copy_caption_to_clipboard(self, caption: str) -> str:
        """Copy caption to clipboard (placeholder functionality)"""
        if caption:
            # Note: Actual clipboard functionality would require additional libraries
            return "Caption copied to clipboard (simulated)"
        return "No caption to copy"
    
    def save_caption_as_file(self, caption: str, image_path: Optional[str]) -> str:
        """Save caption as a text file"""
        if not caption:
            return "No caption to save"
        
        if not image_path:
            return "No image selected"
        
        try:
            # Generate text file path based on image path
            image_path_obj = Path(image_path)
            text_file_path = image_path_obj.with_suffix(".txt")
            
            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(caption)
            
            return f"Caption saved to: {text_file_path}"
        except Exception as e:
            return f"Error saving caption: {str(e)}"
    
    def update_output_format_visibility(self, output_format: str) -> gr.update:
        """Update visibility of JSONL output row based on format selection"""
        return gr.update(visible=(output_format == "jsonl"))
    
    def handle_config_file_selection(self, current_path: str) -> Tuple:
        """Handle config file selection with auto-load if file exists"""
        # Get the file path using the dialog
        file_path = get_file_path_or_save_as(
            file_path=current_path if current_path else "",
            default_extension=".toml",
            extension_name="TOML files"
        )
        
        # If no file was selected, return current state (no changes)
        if not file_path:
            # Return None for all fields to keep current values (23 fields + status)
            return (current_path,) + (gr.update(),) * 23 + ("No file selected",)
        
        # Check if the selected file exists
        if os.path.isfile(file_path):
            # File exists - auto-load it
            log.info(f"Auto-loading existing configuration from: {file_path}")
            # Call the load_configuration method and return its results with the file path
            load_results = self.load_configuration(file_path)
            return (file_path,) + load_results
        else:
            # File doesn't exist - it's a new file path for saving
            log.info(f"New configuration file path selected: {file_path}")
            # Return the new path but keep all other values unchanged (23 fields + status)
            return (file_path,) + (gr.update(),) * 23 + (f"Ready to save configuration to: {os.path.basename(file_path)}",)
    
    def batch_caption_wrapper(
        self,
        image_dir: str,
        output_format: str,
        jsonl_output_file: str,
        output_folder: str,
        max_new_tokens: int,
        custom_prompt: str,
        max_size: int,
        fp8_vl: bool,
        prefix: str,
        suffix: str,
        replace_words: str,
        replace_case_insensitive: bool,
        replace_whole_words_only: bool,
        scan_subfolders: bool,
        copy_images: bool,
        overwrite_existing_captions: bool,
        model_path: str,
        # Generation parameters
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        auto_unload_after_caption: bool,
    ) -> Tuple[str, str, gr.update, gr.update]:
        """Wrapper for batch captioning that manages button states during processing"""
        
        # Set button states at start (disable batch, enable stop)
        yield (
            "🔄 Starting batch captioning...",
            "Preparing...",
            gr.update(interactive=False),  # Disable batch caption button
            gr.update(interactive=True)    # Enable stop button
        )
        
        # Call the actual batch processing method
        result = self.batch_caption_images(
            image_dir, output_format, jsonl_output_file, output_folder, max_new_tokens,
            custom_prompt, max_size, fp8_vl, prefix, suffix, replace_words,
            replace_case_insensitive, replace_whole_words_only, scan_subfolders,
            copy_images, overwrite_existing_captions, model_path,
            do_sample, temperature, top_k, top_p, repetition_penalty, auto_unload_after_caption
        )
        
        # Return final result with proper button states
        yield result
    
    def batch_caption_images(
        self,
        image_dir: str,
        output_format: str,
        jsonl_output_file: str,
        output_folder: str,
        max_new_tokens: int,
        custom_prompt: str,
        max_size: int,
        fp8_vl: bool,
        prefix: str,
        suffix: str,
        replace_words: str,
        replace_case_insensitive: bool,
        replace_whole_words_only: bool,
        scan_subfolders: bool,
        copy_images: bool,
        overwrite_existing_captions: bool,
        model_path: str,
        # Generation parameters
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        auto_unload_after_caption: bool,
    ) -> Tuple[str, str, gr.update, gr.update]:
        """Process batch captioning with auto-loading"""
        if not image_dir:
            return (
                "Please provide an image directory", 
                "Ready",
                gr.update(interactive=True),  # Keep batch caption button enabled
                gr.update(interactive=False)  # Keep stop button disabled
            )
        
        if output_format == "jsonl" and not jsonl_output_file:
            return (
                "JSONL output file is required when output format is JSONL", 
                "Ready",
                gr.update(interactive=True),  # Keep batch caption button enabled
                gr.update(interactive=False)  # Keep stop button disabled
            )
        
        # Auto-load model if needed
        model_loaded, load_message = self.ensure_model_loaded(model_path, max_size, fp8_vl)
        if not model_loaded:
            return (
                f"Failed to load model: {load_message}",
                f"Model load failed: {load_message}",
                gr.update(interactive=True),  # Keep batch caption button enabled
                gr.update(interactive=False)  # Keep stop button disabled
            )
        
        prompt = custom_prompt if custom_prompt.strip() else DEFAULT_PROMPT
        
        # Update button states at start of processing
        # Note: This won't show immediately as it's during processing, but sets the states
        
        # Create a progress object
        progress = gr.Progress()
        
        success, message = self.captioning.batch_caption_images(
            image_dir, output_format, jsonl_output_file, output_folder, max_new_tokens,
            prompt, max_size, fp8_vl, prefix, suffix, replace_words, replace_case_insensitive,
            replace_whole_words_only, scan_subfolders, copy_images, overwrite_existing_captions, progress,
            do_sample, temperature, top_k, top_p, repetition_penalty
        )
        
        # Auto-unload if requested
        if auto_unload_after_caption:
            unload_success, unload_msg = self.captioning.unload_model()
            if unload_success:
                return (
                    message,
                    "Batch captioning completed. Model unloaded from VRAM",
                    gr.update(interactive=True),   # Re-enable batch caption button
                    gr.update(interactive=False)   # Disable stop button
                )
            else:
                return (
                    message,
                    f"Batch captioning completed. Warning: {unload_msg}",
                    gr.update(interactive=True),   # Re-enable batch caption button
                    gr.update(interactive=False)   # Disable stop button
                )
        else:
            return (
                message,
                "Model loaded and ready",
                gr.update(interactive=True),   # Re-enable batch caption button
                gr.update(interactive=False)   # Disable stop button
            )
    
    def stop_batch_processing(self) -> Tuple[str, gr.update, gr.update]:
        """Stop the current batch processing operation"""
        self.captioning.stop_batch_processing()
        return (
            "⏹️ Stop requested - processing will halt after current image...",
            gr.update(interactive=True),  # Re-enable batch caption button
            gr.update(interactive=False)  # Disable stop button
        )
    
    def save_configuration(
        self,
        config_file_path: str,
        model_path: str,
        fp8_vl: bool,
        max_size: int,
        max_new_tokens: int,
        prefix: str,
        suffix: str,
        replace_words: str,
        replace_case_insensitive: bool,
        replace_whole_words_only: bool,
        custom_prompt: str,
        batch_image_dir: str,
        output_format: str,
        jsonl_output_file: str,
        batch_output_folder: str,
        scan_subfolders: bool,
        copy_images: bool,
        overwrite_existing_captions: bool,
        # Generation parameters
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        auto_unload_after_caption: bool,
    ) -> tuple:
        """Save current configuration to file"""
        if not config_file_path:
            return config_file_path, "Please provide a configuration file path"
        
        # Auto-append .toml extension if not present
        if config_file_path and not config_file_path.endswith('.toml'):
            config_file_path = config_file_path + '.toml'
            log.info(f"Auto-appending .toml extension: {config_file_path}")
        
        try:
            config_data = {
                "image_captioning": {
                    "model_path": model_path,
                    "fp8_vl": fp8_vl,
                    "max_size": max_size,
                    "max_new_tokens": max_new_tokens,
                    "prefix": prefix,
                    "suffix": suffix,
                    "replace_words": replace_words,
                    "replace_case_insensitive": replace_case_insensitive,
                    "replace_whole_words_only": replace_whole_words_only,
                    "custom_prompt": custom_prompt,
                    "batch_image_dir": batch_image_dir,
                    "output_format": output_format,
                    "jsonl_output_file": jsonl_output_file,
                    "batch_output_folder": batch_output_folder,
                    "scan_subfolders": scan_subfolders,
                    "copy_images": copy_images,
                    "overwrite_existing_captions": overwrite_existing_captions,
                    # Generation parameters
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "auto_unload_after_caption": auto_unload_after_caption,
                }
            }
            
            # Ensure directory exists
            Path(config_file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save using the common GUI function (expects list of tuples)
            SaveConfigFile(
                parameters=list(config_data.items()),
                file_path=config_file_path,
                exclusion=[""],
            )
            
            # Show success message with timestamp
            config_name = os.path.basename(config_file_path)
            save_time = datetime.now().strftime("%I:%M:%S %p")  # Format: 01:32:23 PM
            success_msg = f"✅ Configuration saved successfully to: {config_name} - Saved at {save_time}"
            log.info(success_msg)
            gr.Info(success_msg)
            
            return config_file_path, success_msg
        except Exception as e:
            error_msg = f"❌ Error saving configuration: {str(e)}"
            log.error(error_msg)
            gr.Error(error_msg)
            return config_file_path, error_msg
    
    def load_configuration(self, config_file_path: str) -> Tuple[str, bool, int, int, str, str, str, bool, bool, str, str, str, str, str, bool, bool, bool, bool, float, int, float, float, bool, str]:
        """Load configuration from file"""
        if not config_file_path:
            return "", False, 1280, 1024, "", "", "", True, True, "", "", "text", "", "", False, False, False, True, 0.7, 50, 0.95, 1.05, False, "Please provide a configuration file path"
        
        if not os.path.isfile(config_file_path):
            error_msg = f"❌ Configuration file does not exist: {config_file_path}"
            log.error(error_msg)
            gr.Error(error_msg)
            return "", False, 1280, 1024, "", "", "", True, True, "", "", "text", "", "", False, False, False, True, 0.7, 50, 0.95, 1.05, False, error_msg
        
        try:
            import toml
            
            with open(config_file_path, "r", encoding="utf-8") as f:
                config_data = toml.load(f)
            
            captioning_config = config_data.get("image_captioning", {})
            
            # Show success message with Gradio Info
            config_name = os.path.basename(config_file_path)
            success_msg = f"✅ Configuration loaded successfully from: {config_name}"
            log.info(success_msg)
            gr.Info(success_msg)
            
            return (
                captioning_config.get("model_path", ""),
                captioning_config.get("fp8_vl", True),
                captioning_config.get("max_size", 1280),
                captioning_config.get("max_new_tokens", 1024),
                captioning_config.get("prefix", ""),
                captioning_config.get("suffix", ""),
                captioning_config.get("replace_words", ""),
                captioning_config.get("replace_case_insensitive", True),
                captioning_config.get("replace_whole_words_only", True),
                captioning_config.get("custom_prompt", ""),
                captioning_config.get("batch_image_dir", ""),
                captioning_config.get("output_format", "text"),
                captioning_config.get("jsonl_output_file", ""),
                captioning_config.get("batch_output_folder", ""),
                captioning_config.get("scan_subfolders", False),
                captioning_config.get("copy_images", False),
                captioning_config.get("overwrite_existing_captions", False),
                # Generation parameters - load with proper defaults
                captioning_config.get("do_sample", True),
                captioning_config.get("temperature", 0.7),
                captioning_config.get("top_k", 50),
                captioning_config.get("top_p", 0.95),
                captioning_config.get("repetition_penalty", 1.05),
                captioning_config.get("auto_unload_after_caption", False),
                success_msg,
            )
        except Exception as e:
            error_msg = f"❌ Error loading configuration: {str(e)}"
            log.error(error_msg)
            gr.Error(error_msg)
            return "", False, 1280, 1024, "", "", "", True, True, "", "", "text", "", "", False, False, False, True, 0.7, 50, 0.95, 1.05, False, error_msg


def image_captioning_tab(headless: bool = False, config: GUIConfig = None) -> None:
    """Create and return the Image Captioning tab"""
    if config is None:
        config = GUIConfig()
    
    tab = ImageCaptioningTab(headless, config)
    return tab