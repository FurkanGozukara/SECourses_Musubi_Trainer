import gradio as gr
import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps
import cv2
import re

from .class_gui_config import GUIConfig
from .common_gui import (
    get_folder_path,
    folder_symbol,
)
from .custom_logging import setup_logging

log = setup_logging()

# Add musubi-tuner src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))

# Import the Kohya preprocessing function
from musubi_tuner.dataset.image_video_dataset import (
    resize_image_to_bucket,
    BucketSelector,
)


def image_preprocessing_tab(headless: bool, config: GUIConfig):
    """
    Create the Image Preprocessing tab for visualizing Kohya bucket processing
    
    Args:
        headless: Whether to run in headless mode
        config: Configuration object
    """
    
    def divisible_by(num: int, divisor: int) -> int:
        return num - num % divisor
    
    def natural_sort_key(path, regex=re.compile(r'(\d+)')):
        """Natural sort key function for cross-platform filename sorting"""
        return [int(text) if text.isdigit() else text.lower() for text in regex.split(str(path))]
    
    def process_images(
        input_folder: str,
        output_folder: str,
        architecture: str,
        enable_bucket: bool,
        fix_exif_orientation: bool,
        resolution_width: int,
        resolution_height: int,
        progress=gr.Progress()
    ) -> Tuple[str, list]:
        """
        Process images using Kohya's bucket logic
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save processed images
            architecture: Model architecture (hv, wan, fp, fk, qi, qie)
            enable_bucket: Whether to use bucket resolution selection
            fix_exif_orientation: Whether to correct EXIF orientation metadata
            resolution_width: Target resolution width
            resolution_height: Target resolution height
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (status message, list of processed files)
        """
        try:
            # Validate inputs
            if not input_folder or not os.path.exists(input_folder):
                return "Error: Input folder does not exist", []
            
            if not output_folder:
                return "Error: Output folder not specified", []
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Get list of image files
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(input_folder).glob(f'*{ext}'))
                image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
            
            # Deduplicate (Windows filesystem is case-insensitive, so glob patterns may match same files)
            # Use natural sort for cross-platform consistent ordering (e.g., file2.jpg before file10.jpg)
            image_files = sorted(set(image_files), key=natural_sort_key)
            
            if not image_files:
                return "Error: No image files found in input folder", []
            
            processed_files = []
            bucket_stats = {}
            
            # Initialize bucket selector if enabled
            bucket_selector = None
            if enable_bucket:
                bucket_selector = BucketSelector(
                    resolution=(resolution_width, resolution_height),
                    enable_bucket=True,
                    no_upscale=False,
                    architecture=architecture
                )
            
            total_images = len(image_files)
            
            for idx, image_path in enumerate(image_files):
                progress(idx / total_images, desc=f"Processing {image_path.name}")
                
                try:
                    # Load image
                    image = Image.open(image_path)
                    
                    # Apply EXIF orientation correction if enabled
                    if fix_exif_orientation:
                        image = ImageOps.exif_transpose(image)
                    
                    original_size = image.size  # (width, height)
                    
                    # Determine bucket resolution
                    if enable_bucket and bucket_selector:
                        bucket_reso = bucket_selector.get_bucket_resolution(original_size)
                    else:
                        bucket_reso = (resolution_width, resolution_height)
                    
                    # Track bucket usage
                    bucket_key = f"{bucket_reso[0]}x{bucket_reso[1]}"
                    bucket_stats[bucket_key] = bucket_stats.get(bucket_key, 0) + 1
                    
                    # Process image using Kohya's resize logic
                    processed_image_np = resize_image_to_bucket(np.array(image), bucket_reso)
                    
                    # Convert back to PIL Image
                    processed_image = Image.fromarray(processed_image_np)
                    
                    # Save processed image
                    output_path = os.path.join(output_folder, image_path.name)
                    processed_image.save(output_path, quality=95)
                    
                    processed_files.append({
                        'original': f"{original_size[0]}x{original_size[1]}",
                        'bucket': bucket_key,
                        'file': image_path.name
                    })
                    
                except Exception as e:
                    log.error(f"Error processing {image_path.name}: {e}")
                    continue
            
            # Generate status message
            status_parts = [
                f"✓ Processed {len(processed_files)} images",
                f"\n\nBucket Distribution:"
            ]
            
            for bucket, count in sorted(bucket_stats.items()):
                percentage = (count / len(processed_files)) * 100
                status_parts.append(f"  {bucket}: {count} images ({percentage:.1f}%)")
            
            status_message = "\n".join(status_parts)
            
            return status_message, processed_files
            
        except Exception as e:
            log.error(f"Error in process_images: {e}")
            return f"Error: {str(e)}", []
    
    with gr.Column():
        gr.Markdown(
            """
            # Image Preprocessing Tool
            
            This tool demonstrates how Kohya processes images with bucket resolution selection.
            It takes your images and processes them exactly as Kohya would during training.
            
            **How it works:**
            - If bucketing is enabled, images are assigned to the nearest bucket resolution based on aspect ratio
            - Images are resized (maintaining aspect ratio) then center-cropped to the bucket resolution
            - Processed images are saved to the output folder with the same filenames
            """
        )
        
        with gr.Row():
            input_folder = gr.Textbox(
                label="Input Images Folder",
                placeholder="Path to folder containing images",
                interactive=True,
            )
            input_folder_button = gr.Button(
                folder_symbol,
                elem_id="open_folder",
                elem_classes=["tool"],
            )
            
            output_folder = gr.Textbox(
                label="Output Folder",
                placeholder="Path to save processed images",
                interactive=True,
            )
            output_folder_button = gr.Button(
                folder_symbol,
                elem_id="open_folder_save",
                elem_classes=["tool"],
            )
        
        with gr.Row():
            architecture = gr.Dropdown(
                label="Architecture",
                choices=[
                    ("hv - Hunyuan Video", "hv"),
                    ("wan - WAN", "wan"),
                    ("fp - FramePack", "fp"),
                    ("fk - Flux Kontext", "fk"),
                    ("qi - Qwen Image", "qi"),
                    ("qie - Qwen Image Edit", "qie"),
                ],
                value="qi",
                info="Model architecture to use for bucket resolution steps"
            )
            
            enable_bucket = gr.Checkbox(
                label="Enable Bucketing",
                value=True,
                info="When enabled, images are assigned to nearest bucket resolution based on aspect ratio"
            )
            
            fix_exif_orientation = gr.Checkbox(
                label="Fix EXIF Orientation",
                value=False,
                info="Correct image orientation based on EXIF metadata (Kohya does NOT do this automatically)"
            )
            
            resolution_width = gr.Number(
                label="Resolution Width",
                value=1328,
                minimum=256,
                maximum=8192,
                step=16,
                info="Target resolution width (will be adjusted to nearest bucket if bucketing enabled)"
            )
            
            resolution_height = gr.Number(
                label="Resolution Height",
                value=1328,
                minimum=256,
                maximum=8192,
                step=16,
                info="Target resolution height (will be adjusted to nearest bucket if bucketing enabled)"
            )
        
        with gr.Row():
            process_button = gr.Button(
                value="Process Images",
                variant="primary",
            )
        
        with gr.Row():
            status_output = gr.Textbox(
                label="Status",
                value="",
                interactive=False,
                lines=10,
            )
        
        with gr.Row():
            file_list = gr.JSON(
                label="Processed Files",
                value=[],
            )
        
        # Button events
        input_folder_button.click(
            fn=lambda x: get_folder_path(x),
            inputs=[input_folder],
            outputs=[input_folder],
            show_progress=False,
        )
        
        output_folder_button.click(
            fn=lambda x: get_folder_path(x),
            inputs=[output_folder],
            outputs=[output_folder],
            show_progress=False,
        )
        
        process_button.click(
            fn=process_images,
            inputs=[
                input_folder,
                output_folder,
                architecture,
                enable_bucket,
                fix_exif_orientation,
                resolution_width,
                resolution_height,
            ],
            outputs=[status_output, file_list],
        )
