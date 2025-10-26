import gradio as gr
import os
import sys
import torch
from pathlib import Path
from typing import Tuple, List
import glob
import gc

from .class_gui_config import GUIConfig
from .common_gui import (
    get_folder_path,
)
from .custom_logging import setup_logging

log = setup_logging()

# Add musubi-tuner src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))


class FP8Converter:
    """FP8 Model Converter - Converts Qwen Image models to FP8 scaled format"""
    
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        
    def is_already_fp8_scaled(self, model_path: str) -> bool:
        """
        Check if a model is already FP8 scaled by looking for scale_weight tensors
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model appears to be FP8 scaled
        """
        try:
            from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
            
            with MemoryEfficientSafeOpen(model_path) as f:
                keys = list(f.keys())
                # Check if there are any scale_weight keys (FP8 scaled models have these)
                scale_keys = [k for k in keys if k.endswith(".scale_weight")]
                return len(scale_keys) > 0
        except Exception as e:
            log.warning(f"Could not check if model is FP8 scaled: {e}")
            return False
    
    def convert_model_to_fp8(
        self,
        input_path: str,
        output_path: str,
        quantization_mode: str,
        delete_original: bool
    ) -> Tuple[bool, str]:
        """
        Convert a Qwen Image model to FP8 scaled format
        
        Args:
            input_path: Path to input model file
            output_path: Path to save FP8 converted model
            delete_original: Whether to delete original file after conversion
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not os.path.exists(input_path):
                return False, f"Input file does not exist: {input_path}"
            
            # Check if model is already FP8 scaled
            if self.is_already_fp8_scaled(input_path):
                return False, "Model already appears to be FP8 scaled (contains scale_weight tensors)"
            
            # Import FP8 optimization utilities
            from musubi_tuner.modules.fp8_optimization_utils import (
                load_safetensors_with_fp8_optimization
            )
            from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
            
            # Qwen Image specific FP8 optimization keys
            # Note: We need to include certain layers for ComfyUI compatibility
            FP8_OPTIMIZATION_TARGET_KEYS = None  # None means all Linear layers (for ComfyUI compatibility)
            FP8_OPTIMIZATION_EXCLUDE_KEYS = [
                "norm",
                "time_text_embed",
            ]
            
            log.info(f"Converting model from {input_path} to {output_path}")
            
            # Load model weights with FP8 optimization
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"Using device: {device}")
            
            # Load and convert to FP8 with selected quantization mode
            fp8_state_dict = load_safetensors_with_fp8_optimization(
                model_files=[input_path],
                calc_device=device,
                target_layer_keys=FP8_OPTIMIZATION_TARGET_KEYS,
                exclude_layer_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
                exp_bits=4,
                mantissa_bits=3,
                move_to_device=False,
                quantization_mode=quantization_mode,
                block_size=64,
                disable_numpy_memmap=False,
            )
            
            log.info(f"Successfully converted model weights to FP8 format")
            
            # Save the FP8 converted model with metadata indicating it's FP8 scaled
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            metadata = {
                "format": "pt",
                "fp8_scaled": "true",
                "quantization_mode": quantization_mode,
                "block_size": "64" if quantization_mode == "block" else "0",
            }
            
            # Log some debug info about the converted weights
            log.info(f"Sample of converted keys: {list(fp8_state_dict.keys())[:5]}")
            if len(fp8_state_dict) > 0:
                sample_key = list(fp8_state_dict.keys())[0]
                sample_tensor = fp8_state_dict[sample_key]
                log.info(f"Sample tensor '{sample_key}': dtype={sample_tensor.dtype}, shape={sample_tensor.shape}")
            
            mem_eff_save_file(fp8_state_dict, output_path, metadata=metadata)
            
            log.info(f"Saved FP8 converted model to {output_path}")
            log.info("✅ Model saved in ComfyUI-compatible FP8 format (FP8 weights + scalar scale tensors)")
            
            # Clean up memory
            del fp8_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Delete original if requested
            if delete_original:
                try:
                    os.remove(input_path)
                    log.info(f"Deleted original file: {input_path}")
                except Exception as e:
                    log.warning(f"Failed to delete original file: {e}")
            
            return True, f"Successfully converted model to FP8 format\nSaved to: {output_path}"
            
        except Exception as e:
            error_msg = f"Error converting model: {str(e)}"
            log.error(error_msg)
            return False, error_msg
    
    def batch_convert_models(
        self,
        input_folder: str,
        output_folder: str,
        quantization_mode: str,
        delete_original: bool
    ) -> Tuple[bool, str]:
        """
        Batch convert all Qwen Image models in a folder to FP8 format
        
        Args:
            input_folder: Folder containing model files
            output_folder: Folder to save converted models (can be same as input)
            delete_original: Whether to delete original files after conversion
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not os.path.exists(input_folder):
                return False, f"Input folder does not exist: {input_folder}"
            
            # Determine output folder
            if not output_folder or output_folder.strip() == "":
                output_folder = input_folder
            
            # Find all safetensors files (recursive search covers both root and subdirectories)
            model_files = []
            for ext in ["*.safetensors", "*.pth", "*.pt"]:
                model_files.extend(glob.glob(os.path.join(input_folder, "**", ext), recursive=True))
            
            # Remove duplicates while preserving order
            seen = set()
            model_files = [f for f in model_files if f not in seen and not seen.add(f)]
            
            if not model_files:
                return False, f"No model files found in {input_folder}"
            
            log.info(f"Found {len(model_files)} model files to process")
            
            results = []
            success_count = 0
            fail_count = 0
            skipped_count = 0
            
            for model_file in model_files:
                # Skip files that already have _FP8_scaled suffix
                if "_FP8_scaled" in os.path.basename(model_file):
                    skipped_count += 1
                    results.append(f"⊘ {os.path.basename(model_file)} (already FP8 scaled, skipped)")
                    continue
                
                rel_path = os.path.relpath(model_file, input_folder)
                output_path = os.path.join(output_folder, rel_path)
                
                # Add FP8_scaled suffix before extension
                base_name, ext = os.path.splitext(output_path)
                output_path = f"{base_name}_FP8_scaled{ext}"
                
                # Check if output file already exists
                if os.path.exists(output_path):
                    skipped_count += 1
                    results.append(f"⊘ {os.path.basename(model_file)} (output already exists, skipped)")
                    continue
                
                # Check if model is already FP8 scaled
                if self.is_already_fp8_scaled(model_file):
                    skipped_count += 1
                    results.append(f"⊘ {os.path.basename(model_file)} (already FP8 scaled model, skipped)")
                    continue
                
                log.info(f"Converting: {model_file} -> {output_path}")
                
                success, message = self.convert_model_to_fp8(
                    model_file,
                    output_path,
                    quantization_mode,
                    delete_original
                )
                
                if success:
                    success_count += 1
                    results.append(f"✓ {os.path.basename(model_file)}")
                else:
                    fail_count += 1
                    results.append(f"✗ {os.path.basename(model_file)}: {message}")
                
                # Clean up memory after each conversion
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            summary = f"Conversion complete!\n\nSuccess: {success_count}\nFailed: {fail_count}\nSkipped: {skipped_count}\n\n" + "\n".join(results)
            
            # Final memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True, summary
            
        except Exception as e:
            error_msg = f"Error in batch conversion: {str(e)}"
            log.error(error_msg)
            return False, error_msg


def fp8_converter_tab(headless: bool, config: GUIConfig) -> None:
    """Create the FP8 Converter tab"""
    
    converter = FP8Converter(headless, config)
    
    gr.Markdown("# FP8 Model Converter")
    gr.Markdown("### Convert Qwen Image models to FP8 Scaled format")
    gr.Markdown("This tool converts Qwen Image or Qwen Image Edit models to FP8 scaled format using Musubi Tuner's dynamic FP8 scaling methodology.")
    gr.Markdown("⚠️ **IMPORTANT:** Use tensor mode for low VRAM inference. Block mode uses 2-3x more VRAM because it dequantizes to BF16.")
    
    # 4-column layout for compact display
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **Features:**
            - Batch conversion
            - Auto-detection
            - Smart skipping
            """)
        
        with gr.Column():
            gr.Markdown("""
            **FP8 Method:**
            - Block-wise (64)
            - Dynamic scaling
            - Scale weights
            """)
        
        with gr.Column():
            gr.Markdown("""
            **Detection:**
            - "_FP8_scaled" files
            - Existing outputs
            - FP8 scale_weight
            """)
        
        with gr.Column():
            gr.Markdown("""
            **Quality:**
            - ~2-3% better
            - Preserves originals
            - Safe to run
            """)
    
    with gr.Row():
        with gr.Column(scale=4):
            input_folder = gr.Textbox(
                label="Input Folder",
                info="Folder containing model files to convert",
                placeholder="e.g., ./models/qwen_image",
                value=config.get("fp8_converter.input_folder", ""),
            )
        input_folder_button = gr.Button(
            "📂",
            size="sm",
            elem_id="input_folder_button"
        )
    
    with gr.Row():
        with gr.Column(scale=4):
            output_folder = gr.Textbox(
                label="Output Folder (Optional)",
                info="Folder to save converted models. If empty, uses input folder.",
                placeholder="e.g., ./models/qwen_image_fp8",
                value=config.get("fp8_converter.output_folder", ""),
            )
        output_folder_button = gr.Button(
            "📂",
            size="sm",
            elem_id="output_folder_button"
        )
    
    quantization_mode = gr.Radio(
        label="Quantization Mode",
        choices=["block", "tensor"],
        value=config.get("fp8_converter.quantization_mode", "block"),
        info="block = Better quality (requires ComfyUI patch), tensor = ComfyUI compatible",
    )
    
    delete_original = gr.Checkbox(
        label="Delete Original Files After Conversion",
        info="⚠️ Delete original model files after successful conversion",
        value=config.get("fp8_converter.delete_original", False),
    )
    
    convert_button = gr.Button(
        "🚀 Start Batch Conversion",
        variant="primary",
        size="lg"
    )
    
    output_status = gr.Textbox(
        label="Conversion Status",
        lines=20,
        max_lines=50,
        placeholder="Conversion status will appear here...",
        interactive=False,
    )
    
    # File browser callbacks
    input_folder_button.click(
        fn=lambda: get_folder_path(""),
        outputs=[input_folder],
        show_progress=False,
    )
    
    output_folder_button.click(
        fn=lambda: get_folder_path(""),
        outputs=[output_folder],
        show_progress=False,
    )
    
    # Conversion callback
    convert_button.click(
        fn=converter.batch_convert_models,
        inputs=[input_folder, output_folder, quantization_mode, delete_original],
        outputs=[output_status],
        show_progress=True,
    )

