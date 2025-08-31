import os
import json
import math
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Union, Optional, Tuple
import gradio as gr

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from .class_gui_config import GUIConfig
from .custom_logging import setup_logging

log = setup_logging()

IMAGE_FACTOR = 28  # The image size must be divisible by this factor
DEFAULT_MAX_SIZE = 1280

DEFAULT_PROMPT = """# Image Annotator
You are a professional image annotator. Please complete the following task based on the input image.
## Create Image Caption
1. Write the caption using natural, descriptive text without structured formats or rich text.
2. Enrich caption details by including: object attributes, vision relations between objects, and environmental details.
3. Identify the text visible in the image, without translation or explanation, and highlight it in the caption with quotation marks.
4. Maintain authenticity and accuracy, avoid generalizations."""


class ImageCaptioning:
    """Core Image Captioning functionality using Qwen2.5-VL"""
    
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        self.model = None
        self.processor = None
        self.device = None
        self.model_loaded = False
        self.stop_processing = False  # Flag to stop batch processing
        
    def load_model_and_processor(self, model_path: str, max_size: int = DEFAULT_MAX_SIZE, fp8_vl: bool = False) -> Tuple[bool, str]:
        """Load Qwen2.5-VL model and processor"""
        try:
            if not os.path.exists(model_path):
                return False, f"Model path does not exist: {model_path}"
                
            log.info(f"Loading model from: {model_path}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"Using device: {self.device}")
            
            min_pixels = 256 * IMAGE_FACTOR * IMAGE_FACTOR  # this means 256x256 is the minimum input size
            max_pixels = max_size * IMAGE_FACTOR * IMAGE_FACTOR
            
            # We don't have configs in model_path, so we use defaults from Hugging Face
            self.processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                min_pixels=min_pixels, 
                max_pixels=max_pixels
            )
            
            # Import load_qwen2_5_vl function from the musubi tuner
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))
            from musubi_tuner.qwen_image.qwen_image_utils import load_qwen2_5_vl
            
            # Use load_qwen2_5_vl function from qwen_image_utils
            dtype = torch.float8_e4m3fn if fp8_vl else torch.bfloat16
            _, self.model = load_qwen2_5_vl(model_path, dtype=dtype, device=self.device, disable_mmap=False)
            
            self.model.eval()
            self.model_loaded = True
            
            log.info(f"Model loaded successfully on device: {self.model.device}")
            return True, "Model loaded successfully"
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            log.error(error_msg)
            return False, error_msg
    
    def stop_batch_processing(self) -> None:
        """Stop the current batch processing operation"""
        self.stop_processing = True
        log.info("Stop batch processing requested")
    
    def reset_stop_flag(self) -> None:
        """Reset the stop processing flag"""
        self.stop_processing = False
    
    def unload_model(self) -> Tuple[bool, str]:
        """Completely unload/destroy the model from VRAM"""
        try:
            if not self.model_loaded:
                return True, "No model loaded"
            
            log.info("Unloading model from VRAM...")
            
            # Delete model and processor
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            # Clear CUDA cache to free VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.model_loaded = False
            self.device = None
            
            # Clear any stored model parameters
            if hasattr(self, '_last_model_path'):
                del self._last_model_path
            if hasattr(self, '_last_max_size'):
                del self._last_max_size
            if hasattr(self, '_last_fp8_vl'):
                del self._last_fp8_vl
            
            log.info("Model unloaded successfully from VRAM")
            return True, "Model unloaded from VRAM"
            
        except Exception as e:
            error_msg = f"Error unloading model: {str(e)}"
            log.error(error_msg)
            return False, error_msg
    
    def resize_image(self, image: Image.Image, max_size: int = DEFAULT_MAX_SIZE) -> Image.Image:
        """Resize image to a suitable resolution"""
        # Import the dataset utilities
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))
        from musubi_tuner.dataset import image_video_dataset
        
        min_area = 256 * 256
        max_area = max_size * max_size
        width, height = image.size
        width_rounded = int((width / IMAGE_FACTOR) + 0.5) * IMAGE_FACTOR
        height_rounded = int((height / IMAGE_FACTOR) + 0.5) * IMAGE_FACTOR
        
        bucket_resos = []
        if width_rounded * height_rounded < min_area:
            # Scale up to min area
            scale_factor = math.sqrt(min_area / (width_rounded * height_rounded))
            new_width = math.ceil(width * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
            new_height = math.ceil(height * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
            
            # Add to bucket resolutions: default and slight variations for keeping aspect ratio
            bucket_resos.append((new_width, new_height))
            bucket_resos.append((new_width + IMAGE_FACTOR, new_height))
            bucket_resos.append((new_width, new_height + IMAGE_FACTOR))
        elif width_rounded * height_rounded > max_area:
            # Scale down to max area
            scale_factor = math.sqrt(max_area / (width_rounded * height_rounded))
            new_width = math.floor(width * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
            new_height = math.floor(height * scale_factor / IMAGE_FACTOR) * IMAGE_FACTOR
            
            # Add to bucket resolutions: default and slight variations for keeping aspect ratio
            bucket_resos.append((new_width, new_height))
            bucket_resos.append((new_width - IMAGE_FACTOR, new_height))
            bucket_resos.append((new_width, new_height - IMAGE_FACTOR))
        else:
            # Keep original resolution, but add slight variations for keeping aspect ratio
            bucket_resos.append((width_rounded, height_rounded))
            bucket_resos.append((width_rounded - IMAGE_FACTOR, height_rounded))
            bucket_resos.append((width_rounded, height_rounded - IMAGE_FACTOR))
            bucket_resos.append((width_rounded + IMAGE_FACTOR, height_rounded))
            bucket_resos.append((width_rounded, height_rounded + IMAGE_FACTOR))
        
        # Min/max area filtering
        bucket_resos = [(w, h) for w, h in bucket_resos if w * h >= min_area and w * h <= max_area]
        
        # Select bucket which has a nearest aspect ratio
        aspect_ratio = width / height
        bucket_resos.sort(key=lambda x: abs((x[0] / x[1]) - aspect_ratio))
        bucket_reso = bucket_resos[0]
        
        # Resize to bucket
        image_np = image_video_dataset.resize_image_to_bucket(image, bucket_reso)
        
        # Convert back to PIL
        image = Image.fromarray(image_np)
        return image
    
    def generate_caption(
        self, 
        image_path: str, 
        max_new_tokens: int = 1024,
        prompt: str = DEFAULT_PROMPT,
        max_size: int = DEFAULT_MAX_SIZE,
        fp8_vl: bool = False,
        prefix: str = "",
        suffix: str = "",
        replace_words: str = "",
        replace_case_insensitive: bool = True,
        replace_whole_words_only: bool = True,
        # Generation parameters
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05
    ) -> Tuple[bool, str]:
        """Generate caption for a single image"""
        try:
            # Model loading is now handled by the GUI layer
                
            if not os.path.exists(image_path):
                return False, f"Image file does not exist: {image_path}"
            
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Process custom prompt - replace \n with actual newlines
            processed_prompt = prompt.replace("\\n", "\n") if prompt else DEFAULT_PROMPT
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": processed_prompt},
                    ],
                }
            ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs = self.resize_image(image, max_size=max_size)
            inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Set proper generation parameters
            generation_params = {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }
            
            # Generate caption with fp8 support
            if fp8_vl:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    generated_ids = self.model.generate(**inputs, **generation_params)
            else:
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **generation_params)
            
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            caption = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # Get caption as string
            caption_text = caption[0] if caption else ""
            
            # Add prefix and suffix if provided
            if prefix:
                caption_text = prefix + caption_text
            if suffix:
                caption_text = caption_text + suffix
            
            # Apply word replacements if provided
            if replace_words:
                try:
                    import re
                    # Parse replace_words format: "word1:replacement1;word2:replacement2"
                    pairs = replace_words.split(";")
                    for pair in pairs:
                        if ":" in pair:
                            parts = pair.split(":", 1)  # Split only on first colon
                            if len(parts) == 2:
                                org_word = parts[0].strip()
                                replace_word = parts[1].strip()
                                if org_word:  # Only replace if org_word is not empty
                                    if replace_whole_words_only:
                                        # Use word boundaries for whole word matching
                                        if replace_case_insensitive:
                                            # Case insensitive whole word replacement
                                            pattern = r'\b' + re.escape(org_word) + r'\b'
                                            caption_text = re.sub(pattern, replace_word, caption_text, flags=re.IGNORECASE)
                                        else:
                                            # Case sensitive whole word replacement
                                            pattern = r'\b' + re.escape(org_word) + r'\b'
                                            caption_text = re.sub(pattern, replace_word, caption_text)
                                    else:
                                        # Partial word matching allowed
                                        if replace_case_insensitive:
                                            # Case insensitive partial replacement
                                            pattern = re.escape(org_word)
                                            caption_text = re.sub(pattern, replace_word, caption_text, flags=re.IGNORECASE)
                                        else:
                                            # Case sensitive partial replacement (original behavior)
                                            caption_text = caption_text.replace(org_word, replace_word)
                except Exception as e:
                    log.warning(f"Error applying word replacements: {str(e)}")
            
            return True, caption_text
            
        except Exception as e:
            error_msg = f"Error generating caption: {str(e)}"
            log.error(error_msg)
            return False, error_msg
    
    def batch_caption_images(
        self,
        image_dir: str,
        output_format: str = "text",
        output_file: str = "",
        output_folder: str = "",
        max_new_tokens: int = 1024,
        prompt: str = DEFAULT_PROMPT,
        max_size: int = DEFAULT_MAX_SIZE,
        fp8_vl: bool = False,
        prefix: str = "",
        suffix: str = "",
        replace_words: str = "",
        replace_case_insensitive: bool = True,
        replace_whole_words_only: bool = True,
        scan_subfolders: bool = False,
        copy_images: bool = False,
        overwrite_existing_captions: bool = False,
        progress: gr.Progress = None,
        # Generation parameters
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05
    ) -> Tuple[bool, str]:
        """Batch caption images in a directory"""
        try:
            # Reset stop flag at the beginning of batch processing
            self.reset_stop_flag()
            
            # Model loading is now handled by the GUI layer
                
            if not os.path.exists(image_dir):
                return False, f"Image directory does not exist: {image_dir}"
            
            # Validate arguments
            if output_format == "jsonl" and not output_file:
                return False, "Output file is required when output format is 'jsonl'"
            
            # Import image utilities
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))
            from musubi_tuner.dataset import image_video_dataset
            import shutil
            
            # Get image files
            if scan_subfolders:
                # Recursively find images in all subfolders
                image_files = []
                for root, dirs, files in os.walk(image_dir):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']):
                            image_files.append(os.path.join(root, file))
                image_files.sort()  # Sort for consistent ordering
            else:
                # Use the existing method for current directory only
                image_files = image_video_dataset.glob_images(image_dir)
            
            if not image_files:
                return False, f"No image files found in directory: {image_dir}"
            
            log.info(f"Found {len(image_files)} image files")
            
            # Log processing mode
            if scan_subfolders:
                log.info("Scanning subfolders recursively")
            if copy_images:
                log.info("Will copy images to output folder")
            
            # Create output directory if needed for JSONL format
            if output_format == "jsonl":
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # If copying images with JSONL, create image output folder
                if copy_images and output_folder and output_folder.strip():
                    jsonl_image_dir = Path(output_folder)
                    jsonl_image_dir.mkdir(parents=True, exist_ok=True)
            
            processed_count = 0
            error_count = 0
            skipped_count = 0
            start_time = time.time()
            processing_times = []  # To track average processing time per image
            
            # Process images and write results
            if output_format == "jsonl":
                # JSONL output format
                with open(output_file, "w", encoding="utf-8") as f:
                    for i, image_path in enumerate(image_files):
                        # Check if stop was requested
                        if self.stop_processing:
                            print(f"\n⏹️ Processing stopped by user at image {i+1}/{len(image_files)}")
                            log.info(f"Batch processing stopped by user at image {i+1}/{len(image_files)}")
                            break
                        
                        current_item = i + 1
                        total_items = len(image_files)
                        
                        # Calculate progress and ETA
                        elapsed_time = time.time() - start_time
                        items_done = processed_count + error_count + skipped_count
                        
                        # Calculate ETA based on actual processing
                        if processing_times:
                            avg_time_per_caption = sum(processing_times) / len(processing_times)
                            items_remaining = total_items - current_item
                            eta_seconds = items_remaining * avg_time_per_caption
                            eta_str = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                        else:
                            eta_str = "ETA: calculating..."
                        
                        # Format progress message for Gradio
                        progress_msg = (
                            f"[{current_item}/{total_items}] "
                            f"Processing: {os.path.basename(image_path)}\n"
                            f"Processed: {processed_count} | Errors: {error_count}\n"
                            f"{eta_str} | Elapsed: {int(elapsed_time//60)}m {int(elapsed_time%60)}s"
                        )
                        
                        if progress:
                            progress((current_item) / total_items, progress_msg)
                        
                        # Print to console
                        filename = os.path.basename(image_path)[:50]
                        print(f"\r[{current_item}/{total_items}] Processing: {filename:<50} | "
                              f"Done: {processed_count} | Err: {error_count} | {eta_str}    ", end='', flush=True)
                        
                        # Track time for caption generation
                        caption_start_time = time.time()
                        
                        success, caption = self.generate_caption(
                            image_path, max_new_tokens, prompt, max_size, fp8_vl, prefix, suffix, replace_words,
                            replace_case_insensitive, replace_whole_words_only, do_sample, temperature, top_k, top_p, repetition_penalty
                        )
                        
                        # Track processing time for ETA calculation
                        caption_time = time.time() - caption_start_time
                        processing_times.append(caption_time)
                        # Keep only last 10 times for moving average
                        if len(processing_times) > 10:
                            processing_times.pop(0)
                        
                        if success:
                            # Copy image if requested (for JSONL format)
                            if copy_images and output_folder and output_folder.strip():
                                image_path_obj = Path(image_path)
                                if scan_subfolders:
                                    # Preserve subfolder structure
                                    rel_path = os.path.relpath(image_path, image_dir)
                                    rel_dir = os.path.dirname(rel_path)
                                    output_subdir = jsonl_image_dir / rel_dir
                                    output_subdir.mkdir(parents=True, exist_ok=True)
                                    image_output_path = output_subdir / image_path_obj.name
                                else:
                                    # No subfolder structure
                                    image_output_path = jsonl_image_dir / image_path_obj.name
                                
                                if not image_output_path.exists():
                                    shutil.copy2(image_path, image_output_path)
                                
                                # Update entry to use relative path if images were copied
                                rel_image_path = os.path.relpath(str(image_output_path), str(output_path.parent))
                                entry = {"image_path": rel_image_path, "caption": caption}
                            else:
                                # Create JSONL entry with original path
                                entry = {"image_path": image_path, "caption": caption}
                            
                            # Write to file
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            f.flush()  # Ensure data is written immediately
                            processed_count += 1
                        else:
                            log.error(f"Failed to caption {image_path}: {caption}")
                            error_count += 1
                
                # Print newline after progress bar
                print()  # Move to new line after progress display
                
                # Calculate final statistics
                total_time = time.time() - start_time
                if processing_times:
                    avg_time = sum(processing_times) / len(processing_times)
                    avg_time_str = f"{avg_time:.2f}s"
                else:
                    avg_time_str = "N/A"
                
                # Check if processing was stopped
                if self.stop_processing:
                    status_msg = "⏹️ Caption generation stopped by user!"
                else:
                    status_msg = "✅ Caption generation completed!"
                
                result_msg = (
                    f"{status_msg}\n"
                    f"📊 Statistics:\n"
                    f"  • Total images: {len(image_files)}\n"
                    f"  • Processed: {processed_count}\n"
                )
                
                if error_count > 0:
                    result_msg += f"  • Errors: {error_count}\n"
                
                result_msg += (
                    f"  • Total time: {int(total_time//60)}m {int(total_time%60)}s\n"
                    f"  • Avg time per caption: {avg_time_str}\n"
                    f"\n📁 Output: JSONL file saved to {output_file}"
                )
                
                if copy_images and output_folder and output_folder.strip():
                    result_msg += f"\n📁 Images copied to: {output_folder}"
                    if scan_subfolders:
                        result_msg += " (preserving folder structure)"
                
                # Also print summary to console
                print(f"\n{result_msg}")
                
            else:
                # Text file output format
                # Determine output directory: use output_folder if provided, otherwise use input directory
                if output_folder and output_folder.strip():
                    output_dir = Path(output_folder)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    save_to_output_folder = True
                else:
                    output_dir = None
                    save_to_output_folder = False
                
                for i, image_path in enumerate(image_files):
                    # Check if stop was requested
                    if self.stop_processing:
                        print(f"\n⏹️ Processing stopped by user at image {i+1}/{len(image_files)}")
                        log.info(f"Batch processing stopped by user at image {i+1}/{len(image_files)}")
                        break
                    
                    current_item = i + 1
                    total_items = len(image_files)
                    
                    # Calculate progress and ETA
                    elapsed_time = time.time() - start_time
                    items_done = processed_count + error_count + skipped_count
                    
                    # Calculate ETA based on actual processing (not skipped)
                    if processing_times:
                        avg_time_per_caption = sum(processing_times) / len(processing_times)
                        items_remaining = total_items - current_item
                        # Estimate how many will actually be processed (not skipped)
                        if items_done > 0:
                            skip_ratio = skipped_count / items_done
                            estimated_to_process = items_remaining * (1 - skip_ratio)
                        else:
                            estimated_to_process = items_remaining
                        eta_seconds = estimated_to_process * avg_time_per_caption
                        eta_str = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                    else:
                        eta_str = "ETA: calculating..."
                    
                    # Format progress message
                    progress_msg = (
                        f"[{current_item}/{total_items}] "
                        f"Processing: {os.path.basename(image_path)}\n"
                        f"Processed: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}\n"
                        f"{eta_str} | Elapsed: {int(elapsed_time//60)}m {int(elapsed_time%60)}s"
                    )
                    
                    if progress:
                        progress((current_item) / total_items, progress_msg)
                    
                    # Print to console as well
                    filename = os.path.basename(image_path)[:50]
                    print(f"\r[{current_item}/{total_items}] Processing: {filename:<50} | "
                          f"Done: {processed_count} | Skip: {skipped_count} | Err: {error_count} | {eta_str}    ", end='', flush=True)
                    
                    image_path_obj = Path(image_path)
                    
                    # Determine where the caption file will be saved BEFORE generating caption
                    if save_to_output_folder:
                        # Save to specified output folder
                        if scan_subfolders:
                            # Preserve subfolder structure
                            rel_path = os.path.relpath(image_path, image_dir)
                            rel_dir = os.path.dirname(rel_path)
                            output_subdir = output_dir / rel_dir
                            output_subdir.mkdir(parents=True, exist_ok=True)
                            text_file_path = output_subdir / f"{image_path_obj.stem}.txt"
                        else:
                            # No subfolder structure to preserve
                            text_file_path = output_dir / f"{image_path_obj.stem}.txt"
                    else:
                        # Save alongside image (same directory as image)
                        text_file_path = image_path_obj.with_suffix(".txt")
                    
                    # Check if caption already exists and skip if not overwriting (BEFORE generating caption)
                    if text_file_path.exists() and not overwrite_existing_captions:
                        log.info(f"Skipping {image_path} - caption already exists at {text_file_path}")
                        skipped_count += 1
                        continue
                    
                    # Track time for actual caption generation
                    caption_start_time = time.time()
                    
                    # Generate caption only if needed
                    success, caption = self.generate_caption(
                        image_path, max_new_tokens, prompt, max_size, fp8_vl, prefix, suffix, replace_words,
                        replace_case_insensitive, replace_whole_words_only, do_sample, temperature, top_k, top_p, repetition_penalty
                    )
                    
                    # Track processing time for ETA calculation
                    caption_time = time.time() - caption_start_time
                    processing_times.append(caption_time)
                    # Keep only last 10 times for moving average
                    if len(processing_times) > 10:
                        processing_times.pop(0)
                    
                    if success:
                        # Copy image if requested and saving to output folder
                        if save_to_output_folder and copy_images:
                            if scan_subfolders:
                                image_output_path = output_subdir / image_path_obj.name
                            else:
                                image_output_path = output_dir / image_path_obj.name
                            
                            if not image_output_path.exists():
                                shutil.copy2(image_path, image_output_path)
                        
                        # Write caption to text file
                        with open(text_file_path, "w", encoding="utf-8") as f:
                            f.write(caption)
                        processed_count += 1
                    else:
                        log.error(f"Failed to caption {image_path}: {caption}")
                        error_count += 1
                
                # Print newline after progress bar
                print()  # Move to new line after progress display
                
                # Calculate final statistics
                total_time = time.time() - start_time
                if processing_times:
                    avg_time = sum(processing_times) / len(processing_times)
                    avg_time_str = f"{avg_time:.2f}s"
                else:
                    avg_time_str = "N/A"
                
                # Check if processing was stopped
                if self.stop_processing:
                    status_msg = "⏹️ Caption generation stopped by user!"
                else:
                    status_msg = "✅ Caption generation completed!"
                
                result_msg = (
                    f"{status_msg}\n"
                    f"📊 Statistics:\n"
                    f"  • Total images: {len(image_files)}\n"
                    f"  • Processed: {processed_count}\n"
                )
                
                if skipped_count > 0:
                    result_msg += f"  • Skipped: {skipped_count} (already had captions)\n"
                if error_count > 0:
                    result_msg += f"  • Errors: {error_count}\n"
                
                result_msg += (
                    f"  • Total time: {int(total_time//60)}m {int(total_time%60)}s\n"
                    f"  • Avg time per caption: {avg_time_str}\n"
                )
                
                if save_to_output_folder:
                    result_msg += f"\n📁 Output: Text files saved to {output_dir}"
                    if copy_images:
                        result_msg += " (images also copied"
                        if scan_subfolders:
                            result_msg += ", preserving folder structure"
                        result_msg += ")"
                else:
                    result_msg += "\n📁 Output: Text files saved alongside each image"
                
                # Also print summary to console
                print(f"\n{result_msg}")
            
            return True, result_msg
            
        except Exception as e:
            error_msg = f"Error in batch captioning: {str(e)}"
            log.error(error_msg)
            return False, error_msg
    
    def get_supported_image_extensions(self) -> List[str]:
        """Get list of supported image file extensions"""
        return [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    
    def validate_image_file(self, file_path: str) -> bool:
        """Validate if file is a supported image"""
        if not os.path.exists(file_path):
            return False
        
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.get_supported_image_extensions()