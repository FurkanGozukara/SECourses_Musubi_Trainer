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
    get_file_path,
)
from .custom_logging import setup_logging

log = setup_logging()

# Add musubi-tuner src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "musubi-tuner", "src"))


class FP8Converter:
    """FP8 Model Converter - Converts Qwen Image and Z Image models to FP8 scaled format"""
    
    # Model type constants
    MODEL_TYPE_QWEN = "Qwen Image"
    MODEL_TYPE_ZIMAGE = "Z Image"
    MODEL_TYPE_AUTO = "Auto-detect"
    
    # FP8 optimization settings for different model types
    FP8_SETTINGS = {
        "qwen_image": {
            "target_keys": None,  # All Linear layers
            "exclude_keys": [
                ".norm.",  # Normalization layers inside modules
                ".norm_q",  # Attention norm layers
                ".norm_k",  # Attention norm layers
                ".norm_added_q",  # Attention norm layers
                ".norm_added_k",  # Attention norm layers
                "txt_norm",  # Text normalization layer (RMSNorm)
            ],
        },
        "z_image": {
            "target_keys": None,  # All Linear layers
            "exclude_keys": [
                ".q_norm",  # Attention QK norm (RMSNorm in JointAttention)
                ".k_norm",  # Attention QK norm (RMSNorm in JointAttention)
                ".attention_norm1",  # RMSNorm in JointTransformerBlock
                ".attention_norm2",  # RMSNorm in JointTransformerBlock
                ".ffn_norm1",  # RMSNorm in JointTransformerBlock
                ".ffn_norm2",  # RMSNorm in JointTransformerBlock
                ".norm_final",  # Final RMSNorm/LayerNorm
                "cap_embedder.0",  # RMSNorm in cap_embedder
            ],
        },
    }
    
    def __init__(self, headless: bool, config: GUIConfig) -> None:
        self.config = config
        self.headless = headless
        
    def detect_model_type(self, model_path: str) -> str:
        """
        Auto-detect model type based on state dict keys
        
        Args:
            model_path: Path to model file
            
        Returns:
            Detected model type string ("qwen_image" or "z_image")
        """
        try:
            from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
            
            with MemoryEfficientSafeOpen(model_path) as f:
                keys = list(f.keys())[:100]  # Check first 100 keys for efficiency
                
                # Z Image detection: Look for Lumina2/NextDiT specific keys
                z_image_indicators = [
                    "cap_embedder.1.weight",  # Lumina2 caption embedder
                    "noise_refiner.",  # NextDiT noise refiner
                    "context_refiner.",  # NextDiT context refiner
                    "x_embedder.weight",  # NextDiT x embedder
                ]
                
                # Qwen Image detection: Look for Qwen specific keys
                qwen_indicators = [
                    "transformer_blocks.",  # Qwen transformer blocks
                    "time_text_embed.",  # Qwen time text embedding
                    "proj_out.",  # Qwen projection out
                ]
                
                z_score = sum(1 for k in keys for ind in z_image_indicators if ind in k)
                qwen_score = sum(1 for k in keys for ind in qwen_indicators if ind in k)
                
                if z_score > qwen_score:
                    log.info(f"Auto-detected model type: Z Image (score: {z_score} vs {qwen_score})")
                    return "z_image"
                else:
                    log.info(f"Auto-detected model type: Qwen Image (score: {qwen_score} vs {z_score})")
                    return "qwen_image"
                    
        except Exception as e:
            log.warning(f"Could not auto-detect model type, defaulting to Qwen Image: {e}")
            return "qwen_image"
        
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
        delete_original: bool,
        model_type: str = "auto"
    ) -> Tuple[bool, str]:
        """
        Convert a Qwen Image or Z Image model to FP8 scaled format
        
        Args:
            input_path: Path to input model file
            output_path: Path to save FP8 converted model
            delete_original: Whether to delete original file after conversion
            model_type: Model type ("qwen_image", "z_image", or "auto" for auto-detection)
            
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
            from musubi_tuner.utils.safetensors_utils import mem_eff_save_file, MemoryEfficientSafeOpen

            # Auto-detect model type if needed
            if model_type == "auto":
                model_type = self.detect_model_type(input_path)
            
            # Get FP8 settings for the detected/selected model type
            fp8_settings = self.FP8_SETTINGS.get(model_type, self.FP8_SETTINGS["qwen_image"])
            FP8_OPTIMIZATION_TARGET_KEYS = fp8_settings["target_keys"]
            FP8_OPTIMIZATION_EXCLUDE_KEYS = fp8_settings["exclude_keys"]
            
            log.info(f"Converting model as {model_type.replace('_', ' ').title()} type")
            log.info(f"Exclude patterns: {FP8_OPTIMIZATION_EXCLUDE_KEYS}")

            # Read existing metadata from the input model to preserve architecture and resolution info
            existing_metadata = {}
            try:
                with MemoryEfficientSafeOpen(input_path) as f:
                    existing_metadata = f.metadata()
                log.info(f"Preserved existing metadata: {list(existing_metadata.keys())}")
            except Exception as e:
                log.warning(f"Could not read existing metadata: {e}")

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
            
            # Add the scaled_fp8 flag tensor (required by ComfyUI to enable FP8 dequantization)
            # This is a dummy tensor that signals to ComfyUI that the model uses scaled FP8 format
            fp8_state_dict["scaled_fp8"] = torch.zeros(2, dtype=torch.float8_e4m3fn)
            
            # Save the FP8 converted model with metadata indicating it's FP8 scaled
            # Preserve existing metadata and add FP8 information
            output_dir = os.path.dirname(output_path)
            if output_dir:  # Only create directory if path has a directory component
                os.makedirs(output_dir, exist_ok=True)
            metadata = existing_metadata.copy()  # Start with existing metadata
            metadata.update({
                "format": "pt",
                "fp8_scaled": "true",
                "quantization_mode": quantization_mode,
                "block_size": "64" if quantization_mode == "block" else "0",
                "model_type": model_type,  # Store model type for reference
            })
            
            # Log some debug info about the converted weights
            log.info(f"Sample of converted keys: {list(fp8_state_dict.keys())[:5]}")
            if len(fp8_state_dict) > 0:
                sample_key = list(fp8_state_dict.keys())[0]
                sample_tensor = fp8_state_dict[sample_key]
                log.info(f"Sample tensor '{sample_key}': dtype={sample_tensor.dtype}, shape={sample_tensor.shape}")
            
            log.info("✅ Added scaled_fp8 flag tensor for ComfyUI compatibility")
            
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
        delete_original: bool,
        model_type: str = "auto"
    ) -> Tuple[bool, str]:
        """
        Batch convert all Qwen Image or Z Image models in a folder to FP8 format
        
        Args:
            input_folder: Folder containing model files
            output_folder: Folder to save converted models (can be same as input)
            delete_original: Whether to delete original files after conversion
            model_type: Model type ("qwen_image", "z_image", or "auto" for auto-detection)
            
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
                    delete_original,
                    model_type
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
    
    def convert_single_file(
        self,
        input_file: str,
        output_file: str,
        quantization_mode: str,
        delete_original: bool,
        model_type: str = "auto"
    ) -> Tuple[bool, str]:
        """
        Convert a single model file to FP8 format
        
        Args:
            input_file: Path to input model file
            output_file: Path to save converted model (if empty, auto-generated)
            quantization_mode: Quantization mode ("tensor" or "block")
            delete_original: Whether to delete original file after conversion
            model_type: Model type ("qwen_image", "z_image", or "auto")
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not input_file or input_file.strip() == "":
                return False, "Please select an input file"
            
            if not os.path.exists(input_file):
                return False, f"Input file does not exist: {input_file}"
            
            # Check if file is a valid model file
            valid_extensions = [".safetensors", ".pth", ".pt"]
            if not any(input_file.lower().endswith(ext) for ext in valid_extensions):
                return False, f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            
            # Skip files that already have _FP8_scaled suffix
            if "_FP8_scaled" in os.path.basename(input_file):
                return False, "File already has _FP8_scaled suffix, skipping"
            
            # Check if model is already FP8 scaled
            if self.is_already_fp8_scaled(input_file):
                return False, "Model already appears to be FP8 scaled (contains scale_weight tensors)"
            
            # Auto-generate output path if not provided
            if not output_file or output_file.strip() == "":
                base_name, ext = os.path.splitext(input_file)
                output_file = f"{base_name}_FP8_scaled{ext}"
            
            # Check if output file already exists
            if os.path.exists(output_file):
                return False, f"Output file already exists: {output_file}"
            
            log.info(f"Converting single file: {input_file} -> {output_file}")
            
            # Perform the conversion
            success, message = self.convert_model_to_fp8(
                input_file,
                output_file,
                quantization_mode,
                delete_original,
                model_type
            )
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return success, message
            
        except Exception as e:
            error_msg = f"Error converting file: {str(e)}"
            log.error(error_msg)
            return False, error_msg


def fp8_converter_tab(headless: bool, config: GUIConfig) -> None:
    """Create the FP8 Converter tab"""
    
    converter = FP8Converter(headless, config)
    
    gr.Markdown("# FP8 Model Converter")
    gr.Markdown("### Convert Qwen Image and Z Image models to FP8 Scaled format")
    gr.Markdown("This tool converts Qwen Image, Qwen Image Edit, and **Z Image** models to FP8 scaled format using Musubi Tuner's dynamic FP8 scaling methodology.")
    gr.Markdown("⚠️ **IMPORTANT:** Tensor mode (default) is ComfyUI compatible and uses less VRAM. Block mode provides better quality but requires ComfyUI patch and uses 2-3x more VRAM.")
    
    # 4-column layout for compact display
    with gr.Row():
        with gr.Column():
            gr.Markdown("""
            **Features:**
            - Single file & batch
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
    
    # Helper function to convert UI model type to internal model type
    def get_internal_model_type(model_type_ui):
        model_type_map = {
            FP8Converter.MODEL_TYPE_AUTO: "auto",
            FP8Converter.MODEL_TYPE_QWEN: "qwen_image",
            FP8Converter.MODEL_TYPE_ZIMAGE: "z_image",
        }
        return model_type_map.get(model_type_ui, "auto")
    
    # ========== CREATE ALL UI COMPONENTS FIRST ==========
    with gr.Tabs():
        # ========== SINGLE FILE TAB ==========
        with gr.TabItem("🔹 Single File"):
            gr.Markdown("#### Convert a single model file to FP8 format")
            
            with gr.Row():
                with gr.Column(scale=4):
                    single_input_file = gr.Textbox(
                        label="Input File",
                        info="Path to the model file to convert (.safetensors, .pth, .pt)",
                        placeholder="e.g., ./models/z_image_turbo_bf16.safetensors",
                        value=config.get("fp8_converter.single_input_file", ""),
                    )
                single_input_button = gr.Button(
                    "📄",
                    size="lg",
                    elem_id="single_input_button"
                )
            
            with gr.Row():
                with gr.Column(scale=4):
                    single_output_file = gr.Textbox(
                        label="Output File (Optional)",
                        info="Path for converted model. If empty, adds '_FP8_scaled' suffix to input filename.",
                        placeholder="e.g., ./models/z_image_turbo_bf16_FP8_scaled.safetensors",
                        value="",
                    )
                single_output_button = gr.Button(
                    "📄",
                    size="lg",
                    elem_id="single_output_button"
                )
            
            with gr.Row():
                single_model_type = gr.Dropdown(
                    label="Model Type",
                    choices=[
                        FP8Converter.MODEL_TYPE_AUTO,
                        FP8Converter.MODEL_TYPE_QWEN,
                        FP8Converter.MODEL_TYPE_ZIMAGE,
                    ],
                    value=config.get("fp8_converter.model_type", FP8Converter.MODEL_TYPE_AUTO),
                    info="Select model architecture or use auto-detection.",
                )
                
                single_quantization_mode = gr.Radio(
                    label="Quantization Mode",
                    choices=["tensor", "block"],
                    value=config.get("fp8_converter.quantization_mode", "tensor"),
                    info="tensor = ComfyUI compatible (recommended)",
                )
            
            single_delete_original = gr.Checkbox(
                label="Delete Original File After Conversion",
                info="⚠️ Delete original model file after successful conversion",
                value=config.get("fp8_converter.delete_original", False),
            )
            
            single_convert_button = gr.Button(
                "🚀 Convert Single File",
                variant="primary",
                size="lg"
            )
        
        # ========== BATCH CONVERSION TAB ==========
        with gr.TabItem("📁 Batch Conversion"):
            gr.Markdown("#### Convert all model files in a folder to FP8 format")
            
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
                    size="lg",
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
                    size="lg",
                    elem_id="output_folder_button"
                )
            
            with gr.Row():
                batch_model_type = gr.Dropdown(
                    label="Model Type",
                    choices=[
                        FP8Converter.MODEL_TYPE_AUTO,
                        FP8Converter.MODEL_TYPE_QWEN,
                        FP8Converter.MODEL_TYPE_ZIMAGE,
                    ],
                    value=config.get("fp8_converter.model_type", FP8Converter.MODEL_TYPE_AUTO),
                    info="Select model architecture or use auto-detection.",
                )
                
                batch_quantization_mode = gr.Radio(
                    label="Quantization Mode",
                    choices=["tensor", "block"],
                    value=config.get("fp8_converter.quantization_mode", "tensor"),
                    info="tensor = ComfyUI compatible (recommended)",
                )
            
            batch_delete_original = gr.Checkbox(
                label="Delete Original Files After Conversion",
                info="⚠️ Delete original model files after successful conversion",
                value=config.get("fp8_converter.delete_original", False),
            )
            
            batch_convert_button = gr.Button(
                "🚀 Start Batch Conversion",
                variant="primary",
                size="lg"
            )
    
    # Shared output status (at bottom, after tabs)
    output_status = gr.Textbox(
        label="Conversion Status",
        lines=15,
        max_lines=50,
        placeholder="Conversion status will appear here...",
        interactive=False,
    )
    
    # ========== SET UP EVENT HANDLERS AFTER ALL COMPONENTS ARE CREATED ==========
    
    # Single file browser callbacks
    single_input_button.click(
        fn=lambda: get_file_path("", ".safetensors", "Model files"),
        outputs=[single_input_file],
        show_progress=False,
    )
    
    single_output_button.click(
        fn=lambda: get_file_path("", ".safetensors", "Safetensors files"),
        outputs=[single_output_file],
        show_progress=False,
    )
    
    # Single file conversion callback
    def convert_single_with_model_type(input_file, output_file, quantization_mode, delete_original, model_type_ui):
        model_type = get_internal_model_type(model_type_ui)
        return converter.convert_single_file(
            input_file, output_file, quantization_mode, delete_original, model_type
        )
    
    single_convert_button.click(
        fn=convert_single_with_model_type,
        inputs=[single_input_file, single_output_file, single_quantization_mode, single_delete_original, single_model_type],
        outputs=[output_status],
        show_progress=True,
    )
    
    # Batch folder browser callbacks
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
    
    # Batch conversion callback
    def convert_batch_with_model_type(input_folder, output_folder, quantization_mode, delete_original, model_type_ui):
        model_type = get_internal_model_type(model_type_ui)
        return converter.batch_convert_models(
            input_folder, output_folder, quantization_mode, delete_original, model_type
        )
    
    batch_convert_button.click(
        fn=convert_batch_with_model_type,
        inputs=[input_folder, output_folder, batch_quantization_mode, batch_delete_original, batch_model_type],
        outputs=[output_status],
        show_progress=True,
    )

