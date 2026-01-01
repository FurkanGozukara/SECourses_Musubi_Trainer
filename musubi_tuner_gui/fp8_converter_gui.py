import gradio as gr
import os
import sys
import io
import random
import traceback
import contextlib
import torch
from pathlib import Path
from typing import Tuple, List, Optional
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
        Check if a model is already quantized/FP8-scaled by looking for known marker tensors.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if model appears to be quantized (legacy scaled_fp8 or comfy_quant)
        """
        try:
            from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen
            
            with MemoryEfficientSafeOpen(model_path) as f:
                keys = list(f.keys())

                # Legacy scaled_fp8 format: scale tensors + marker tensor
                if "scaled_fp8" in keys:
                    return True
                if any(k.endswith(".scale_weight") for k in keys):
                    return True

                # ComfyUI comfy_quant format: per-layer .comfy_quant tensors + scale tensors
                if any(k.endswith(".comfy_quant") for k in keys):
                    return True
                if any(k.endswith(".weight_scale") for k in keys):
                    return True

                return False
        except Exception as e:
            log.warning(f"Could not check if model is FP8 scaled: {e}")
            return False

    def _import_convert_to_quant(self):
        """
        Import convert_to_quant from the bundled workspace folder (or installed package).

        Returns:
            convert_to_fp8_scaled callable
        """
        # 1) If the user installed convert_to_quant, its top-level package exports convert_to_fp8_scaled
        try:
            import convert_to_quant as c2q  # type: ignore

            if hasattr(c2q, "convert_to_fp8_scaled"):
                return c2q.convert_to_fp8_scaled  # type: ignore[attr-defined]
        except Exception:
            pass

        # 2) Workspace layout here is nested: ./convert_to_quant/convert_to_quant/convert_to_quant.py
        #    which maps to module path: convert_to_quant.convert_to_quant.convert_to_quant
        try:
            from convert_to_quant.convert_to_quant.convert_to_quant import (  # type: ignore
                convert_to_fp8_scaled,
            )

            return convert_to_fp8_scaled
        except Exception:
            pass

        # 3) Fallback: add the nested project root to sys.path and re-import as a real package.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        c2q_root = os.path.join(repo_root, "convert_to_quant")
        if os.path.isdir(os.path.join(c2q_root, "convert_to_quant")) and c2q_root not in sys.path:
            sys.path.insert(0, c2q_root)

        # If we previously imported the outer folder as a namespace package, drop it so Python can
        # resolve the real package (__init__.py) from c2q_root.
        if "convert_to_quant" in sys.modules:
            mod = sys.modules["convert_to_quant"]
            if getattr(mod, "__file__", None) is None:
                del sys.modules["convert_to_quant"]

        from convert_to_quant import convert_to_fp8_scaled  # type: ignore

        return convert_to_fp8_scaled

    @staticmethod
    def _tail_text(text: str, max_chars: int = 25000) -> str:
        if len(text) <= max_chars:
            return text
        return (
            text[:2000]
            + "\n\n... (output truncated) ...\n\n"
            + text[-max_chars:]
        )

    class _TeeTextIO(io.TextIOBase):
        """
        A small tee stream so convert_to_quant prints + tqdm progress bars show in the terminal
        while we also capture output for the GUI textbox.
        """

        def __init__(self, *streams):
            super().__init__()
            self._streams = [s for s in streams if s is not None]

        def write(self, s):  # type: ignore[override]
            for st in self._streams:
                try:
                    st.write(s)
                except Exception:
                    # Best-effort: never break conversion due to logging output issues
                    pass
            return len(s)

        def flush(self):  # type: ignore[override]
            for st in self._streams:
                try:
                    st.flush()
                except Exception:
                    pass

        def isatty(self):  # type: ignore[override]
            # tqdm checks this to decide dynamic display. Proxy the real terminal if present.
            for st in self._streams:
                if hasattr(st, "isatty"):
                    try:
                        return bool(st.isatty())
                    except Exception:
                        continue
            return False

        @property
        def encoding(self):  # type: ignore[override]
            for st in self._streams:
                enc = getattr(st, "encoding", None)
                if enc:
                    return enc
            return "utf-8"

    def convert_model_to_fp8_improved(
        self,
        input_path: str,
        output_path: str,
        quantization_mode: str,
        delete_original: bool,
        model_type: str = "auto",
        comfy_quant: bool = False,
        save_both_formats: bool = False,
        full_precision_matrix_mult: bool = False,
        skip_inefficient_layers: bool = True,
        calib_samples: int = 256,
        seed: int = 0,
        optimizer: str = "original",
        num_iter: int = 200,
        lr: float = 8.077300000003e-3,
        lr_schedule: str = "adaptive",
        full_matrix: bool = False,
        top_p: float = 0.01,
        min_k: int = 1,
        max_k: int = 16,
        no_learned_rounding: bool = False,
    ) -> Tuple[bool, str]:
        """
        Convert using convert_to_quant's learned-rounding FP8 scaling (higher quality, slower).

        This keeps legacy scaled_fp8 format by default (scale_weight + scaled_fp8 marker),
        unless comfy_quant=True is selected.
        """
        try:
            if not os.path.exists(input_path):
                return False, f"Input file does not exist: {input_path}"

            # Check if model is already quantized/FP8 scaled
            if self.is_already_fp8_scaled(input_path):
                return False, "Model already appears to be quantized (scaled_fp8 / comfy_quant detected)"

            if model_type == "auto":
                model_type = self.detect_model_type(input_path)

            # Map UI modes to convert_to_quant scaling modes
            scaling_mode = "block" if quantization_mode == "block" else "tensor"
            block_size = 64

            # Seed handling: -1 means random
            if seed is None:
                seed = 0
            if int(seed) < 0:
                seed = random.randint(0, 2**31 - 1)

            # Defensive defaults for UI-cleared fields (Gradio can pass None)
            if calib_samples is None:
                calib_samples = 256
            if optimizer is None:
                optimizer = "original"
            if num_iter is None:
                num_iter = 200
            if lr is None:
                lr = 8.077300000003e-3
            if lr_schedule is None:
                lr_schedule = "adaptive"
            if top_p is None:
                top_p = 0.01
            if min_k is None:
                min_k = 1
            if max_k is None:
                max_k = 16

            convert_to_fp8_scaled = self._import_convert_to_quant()

            def _import_convert_fp8_scaled_to_comfy_quant():
                # convert_fp8_scaled_to_comfy_quant is not re-exported at package root
                try:
                    from convert_to_quant.convert_to_quant.convert_to_quant import (  # type: ignore
                        convert_fp8_scaled_to_comfy_quant,
                    )

                    return convert_fp8_scaled_to_comfy_quant
                except Exception:
                    # Ensure nested project root is importable
                    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    c2q_root = os.path.join(repo_root, "convert_to_quant")
                    if os.path.isdir(os.path.join(c2q_root, "convert_to_quant")) and c2q_root not in sys.path:
                        sys.path.insert(0, c2q_root)

                    if "convert_to_quant" in sys.modules:
                        mod = sys.modules["convert_to_quant"]
                        if getattr(mod, "__file__", None) is None:
                            del sys.modules["convert_to_quant"]

                    from convert_to_quant.convert_to_quant.convert_to_quant import (  # type: ignore
                        convert_fp8_scaled_to_comfy_quant,
                    )

                    return convert_fp8_scaled_to_comfy_quant

            # When saving both formats, we quantize ONCE into legacy fp8_scaled, then do a fast
            # format-conversion step to comfy_quant. This avoids running the expensive optimizer twice.
            primary_is_comfy = bool(comfy_quant)

            def _add_suffix(path: str, suffix: str) -> str:
                base, ext = os.path.splitext(path)
                return f"{base}{suffix}{ext}"

            def _derive_paths(primary_path: str, primary_is_comfy: bool) -> Tuple[str, str]:
                """
                Returns (legacy_fp8_scaled_path, comfy_quant_path).
                If primary_is_comfy and primary ends with '_comfy_quant', the legacy path is the sibling
                without that suffix. Otherwise a safe suffix is appended.
                """
                base, ext = os.path.splitext(primary_path)
                if primary_is_comfy:
                    comfy_path = primary_path
                    if base.endswith("_comfy_quant"):
                        legacy_path = base[: -len("_comfy_quant")] + ext
                    else:
                        legacy_path = base + "_legacy_fp8_scaled" + ext
                else:
                    legacy_path = primary_path
                    comfy_path = base + "_comfy_quant" + ext
                return legacy_path, comfy_path

            legacy_out, comfy_out = _derive_paths(output_path, primary_is_comfy)

            # convert_to_quant expects output_dir to exist; also its saver uses dirname(output_file)
            # Ensure dirname is non-empty for relative filenames.
            def _normalize_out_path(p: str) -> str:
                return p if os.path.dirname(p) else os.path.join(".", p)

            # If saving both formats, always quantize to legacy first (needed for format conversion)
            quantize_output_path = legacy_out if save_both_formats else output_path
            quantize_output_path_for_converter = _normalize_out_path(quantize_output_path)

            # Model preset flags (only the ones relevant to this repo)
            qwen_flag = model_type == "qwen_image"
            zimage_flag = model_type == "z_image"

            # Pre-check outputs for safety
            if save_both_formats:
                if os.path.exists(legacy_out):
                    return False, f"Legacy output already exists: {legacy_out}"
                if os.path.exists(comfy_out):
                    return False, f"Comfy_quant output already exists: {comfy_out}"
            else:
                if os.path.exists(output_path):
                    return False, f"Output file already exists: {output_path}"

            buf = io.StringIO()
            real_stdout = getattr(sys, "__stdout__", sys.stdout)
            real_stderr = getattr(sys, "__stderr__", sys.stderr)
            tee_out = self._TeeTextIO(real_stdout, buf)
            tee_err = self._TeeTextIO(real_stderr, buf)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                convert_to_fp8_scaled(
                    input_file=input_path,
                    output_file=quantize_output_path_for_converter,
                    # If saving both: always write legacy first (comfy_quant=False), then convert format
                    comfy_quant=False if save_both_formats else bool(comfy_quant),
                    # Model filter presets
                    t5xxl=False,
                    mistral=False,
                    visual=False,
                    flux2=False,
                    distillation_large=False,
                    distillation_small=False,
                    nerf_large=False,
                    nerf_small=False,
                    radiance=False,
                    wan=False,
                    qwen=bool(qwen_flag),
                    hunyuan=False,
                    zimage=bool(zimage_flag),
                    zimage_refiner=False,
                    # Bias correction / reproducibility
                    calib_samples=int(calib_samples),
                    seed=int(seed),
                    # Force FP8 (user requested improved FP8 only)
                    int8=False,
                    fallback=None,
                    custom_layers=None,
                    custom_type=None,
                    custom_block_size=None,
                    custom_scaling_mode=None,
                    custom_simple=False,
                    custom_heur=False,
                    fallback_block_size=None,
                    fallback_simple=False,
                    full_precision_matrix_mult=bool(full_precision_matrix_mult)
                    if (not save_both_formats and comfy_quant)
                    else False,
                    skip_inefficient_layers=bool(skip_inefficient_layers),
                    include_input_scale=False,
                    no_learned_rounding=bool(no_learned_rounding),
                    save_quant_metadata=False,
                    layer_config=None,
                    layer_config_fullmatch=False,
                    # Converter params (learned rounding)
                    optimizer=str(optimizer),
                    num_iter=int(num_iter),
                    lr=float(lr),
                    lr_schedule=str(lr_schedule),
                    scaling_mode=str(scaling_mode),
                    block_size=int(block_size),
                    full_matrix=bool(full_matrix),
                    top_p=float(top_p),
                    min_k=int(min_k),
                    max_k=int(max_k),
                )

            # If requested, create comfy_quant copy via fast format conversion (no re-quantization)
            if save_both_formats:
                convert_fp8_scaled_to_comfy_quant = _import_convert_fp8_scaled_to_comfy_quant()

                buf2 = io.StringIO()
                tee_out2 = self._TeeTextIO(real_stdout, buf2)
                tee_err2 = self._TeeTextIO(real_stderr, buf2)
                with contextlib.redirect_stdout(tee_out2), contextlib.redirect_stderr(tee_err2):
                    convert_fp8_scaled_to_comfy_quant(
                        input_file=_normalize_out_path(legacy_out),
                        output_file=_normalize_out_path(comfy_out),
                        hp_filter=None,
                        full_precision_mm=bool(full_precision_matrix_mult),
                        include_input_scale=False,
                        save_quant_metadata=False,
                    )

                buf.write("\n\n----- comfy_quant format conversion -----\n\n")
                buf.write(buf2.getvalue())

            # convert_to_fp8_scaled prints errors; also check output existence
            produced_primary = _normalize_out_path(output_path)
            produced_legacy = _normalize_out_path(legacy_out)
            produced_comfy = _normalize_out_path(comfy_out)

            if save_both_formats:
                if not os.path.exists(produced_legacy) or not os.path.exists(produced_comfy):
                    out = self._tail_text(buf.getvalue())
                    return False, "Improved FP8 conversion did not produce both output files.\n\n" + out
            else:
                if not os.path.exists(produced_primary):
                    out = self._tail_text(buf.getvalue())
                    return False, "Improved FP8 conversion did not produce an output file.\n\n" + out

            # Delete original if requested
            if delete_original:
                try:
                    os.remove(input_path)
                    log.info(f"Deleted original file: {input_path}")
                except Exception as e:
                    log.warning(f"Failed to delete original file: {e}")

            out = self._tail_text(buf.getvalue())
            header = (
                "✅ Improved FP8 conversion complete (convert_to_quant learned rounding)\n"
                f"- Input:  {input_path}\n"
                f"- Output: {produced_primary if not save_both_formats else (produced_comfy if primary_is_comfy else produced_legacy)}\n"
                f"- Mode:   {quantization_mode} (scaling_mode={scaling_mode}, block_size=64)\n"
                f"- Preset: {'Qwen' if qwen_flag else ('Z-Image' if zimage_flag else model_type)}\n"
                f"- Learned rounding: {'OFF' if no_learned_rounding else 'ON'} (optimizer={optimizer}, num_iter={num_iter}, lr={lr}, schedule={lr_schedule}, top_p={top_p}, k=[{min_k},{max_k}])\n"
                f"- Bias correction samples: {calib_samples} (seed={seed})\n"
                f"- Output format: {'comfy_quant' if comfy_quant else 'legacy scaled_fp8'}\n"
                f"- comfy_quant full_precision_matrix_mult: {bool(full_precision_matrix_mult)}\n"
                f"- Heuristics (skip inefficient layers): {skip_inefficient_layers}\n"
            )
            if save_both_formats:
                header += f"- Also saved: {produced_comfy if not primary_is_comfy else produced_legacy}\n"
            return True, header + "\n" + out

        except Exception:
            return False, "Error in improved FP8 conversion:\n\n" + traceback.format_exc()
    
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
        model_type: str = "auto",
        engine: str = "musubi",
        # Improved FP8 params (convert_to_quant)
        improved_comfy_quant: bool = False,
        improved_save_both_formats: bool = False,
        improved_full_precision_matrix_mult: bool = False,
        improved_skip_inefficient_layers: bool = True,
        improved_calib_samples: int = 256,
        improved_seed: int = 0,
        improved_optimizer: str = "original",
        improved_num_iter: int = 200,
        improved_lr: float = 8.077300000003e-3,
        improved_lr_schedule: str = "adaptive",
        improved_full_matrix: bool = False,
        improved_top_p: float = 0.01,
        improved_min_k: int = 1,
        improved_max_k: int = 16,
        improved_no_learned_rounding: bool = False,
    ) -> str:
        """
        Batch convert all Qwen Image or Z Image models in a folder to FP8 format
        
        Args:
            input_folder: Folder containing model files
            output_folder: Folder to save converted models (can be same as input)
            delete_original: Whether to delete original files after conversion
            model_type: Model type ("qwen_image", "z_image", or "auto" for auto-detection)
            
        Returns:
            Summary/status string
        """
        try:
            if not os.path.exists(input_folder):
                return f"Input folder does not exist: {input_folder}"
            
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
                return f"No model files found in {input_folder}"
            
            log.info(f"Found {len(model_files)} model files to process")
            
            results: List[str] = []
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

                if engine == "convert_to_quant":
                    success, message = self.convert_model_to_fp8_improved(
                        model_file,
                        output_path,
                        quantization_mode,
                        delete_original,
                        model_type,
                        comfy_quant=improved_comfy_quant,
                        save_both_formats=improved_save_both_formats,
                        full_precision_matrix_mult=improved_full_precision_matrix_mult,
                        skip_inefficient_layers=improved_skip_inefficient_layers,
                        calib_samples=improved_calib_samples,
                        seed=improved_seed,
                        optimizer=improved_optimizer,
                        num_iter=improved_num_iter,
                        lr=improved_lr,
                        lr_schedule=improved_lr_schedule,
                        full_matrix=improved_full_matrix,
                        top_p=improved_top_p,
                        min_k=improved_min_k,
                        max_k=improved_max_k,
                        no_learned_rounding=improved_no_learned_rounding,
                    )
                else:
                    success, message = self.convert_model_to_fp8(
                        model_file,
                        output_path,
                        quantization_mode,
                        delete_original,
                        model_type,
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
            
            return summary
            
        except Exception as e:
            error_msg = f"Error in batch conversion: {str(e)}"
            log.error(error_msg)
            return error_msg
    
    def convert_single_file(
        self,
        input_file: str,
        output_file: str,
        quantization_mode: str,
        delete_original: bool,
        model_type: str = "auto",
        engine: str = "musubi",
        # Improved FP8 params (convert_to_quant)
        improved_comfy_quant: bool = False,
        improved_save_both_formats: bool = False,
        improved_full_precision_matrix_mult: bool = False,
        improved_skip_inefficient_layers: bool = True,
        improved_calib_samples: int = 256,
        improved_seed: int = 0,
        improved_optimizer: str = "original",
        improved_num_iter: int = 200,
        improved_lr: float = 8.077300000003e-3,
        improved_lr_schedule: str = "adaptive",
        improved_full_matrix: bool = False,
        improved_top_p: float = 0.01,
        improved_min_k: int = 1,
        improved_max_k: int = 16,
        improved_no_learned_rounding: bool = False,
    ) -> str:
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
                return "Please select an input file"
            
            if not os.path.exists(input_file):
                return f"Input file does not exist: {input_file}"
            
            # Check if file is a valid model file
            valid_extensions = [".safetensors", ".pth", ".pt"]
            if not any(input_file.lower().endswith(ext) for ext in valid_extensions):
                return f"Invalid file type. Supported formats: {', '.join(valid_extensions)}"
            
            # Skip files that already have _FP8_scaled suffix
            if "_FP8_scaled" in os.path.basename(input_file):
                return "File already has _FP8_scaled suffix, skipping"
            
            # Check if model is already FP8 scaled
            if self.is_already_fp8_scaled(input_file):
                return "Model already appears to be quantized (scaled_fp8 / comfy_quant detected)"
            
            # Auto-generate output path if not provided
            if not output_file or output_file.strip() == "":
                base_name, ext = os.path.splitext(input_file)
                output_file = f"{base_name}_FP8_scaled{ext}"
            
            # Check if output file already exists
            if os.path.exists(output_file):
                return f"Output file already exists: {output_file}"
            
            log.info(f"Converting single file: {input_file} -> {output_file}")
            
            # Perform the conversion
            if engine == "convert_to_quant":
                success, message = self.convert_model_to_fp8_improved(
                    input_file,
                    output_file,
                    quantization_mode,
                    delete_original,
                    model_type,
                    comfy_quant=improved_comfy_quant,
                    save_both_formats=improved_save_both_formats,
                    full_precision_matrix_mult=improved_full_precision_matrix_mult,
                    skip_inefficient_layers=improved_skip_inefficient_layers,
                    calib_samples=improved_calib_samples,
                    seed=improved_seed,
                    optimizer=improved_optimizer,
                    num_iter=improved_num_iter,
                    lr=improved_lr,
                    lr_schedule=improved_lr_schedule,
                    full_matrix=improved_full_matrix,
                    top_p=improved_top_p,
                    min_k=improved_min_k,
                    max_k=improved_max_k,
                    no_learned_rounding=improved_no_learned_rounding,
                )
            else:
                success, message = self.convert_model_to_fp8(
                    input_file,
                    output_file,
                    quantization_mode,
                    delete_original,
                    model_type,
                )
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return message
            
        except Exception as e:
            error_msg = f"Error converting file: {str(e)}"
            log.error(error_msg)
            return error_msg


def fp8_converter_tab(headless: bool, config: GUIConfig) -> None:
    """Create the FP8 Converter tab"""
    
    converter = FP8Converter(headless, config)
    
    gr.Markdown("# FP8 Model Converter")
    gr.Markdown("### Convert Qwen Image and Z Image models to FP8 Scaled format")
    gr.Markdown(
        "This tool converts Qwen Image, Qwen Image Edit, and **Z Image** models to FP8 scaled format.\n\n"
        "- **Standard**: Musubi Tuner dynamic FP8 scaling (fast, current behavior)\n"
        "- **Improved**: convert_to_quant learned rounding + bias correction (slower, higher quality)"
    )
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

    # FP8 engine selection (keep Musubi as default to avoid breaking existing workflows)
    ENGINE_STANDARD_UI = "Standard (Musubi dynamic scaling)"
    ENGINE_IMPROVED_UI = "Improved (convert_to_quant learned rounding)"

    def get_internal_engine(engine_ui: str) -> str:
        return "convert_to_quant" if engine_ui == ENGINE_IMPROVED_UI else "musubi"

    def _as_int(value, default: int) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _as_float(value, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    with gr.Row():
        initial_engine_value = config.get("fp8_converter.engine", ENGINE_STANDARD_UI)
        fp8_engine = gr.Radio(
            label="FP8 Engine",
            choices=[ENGINE_STANDARD_UI, ENGINE_IMPROVED_UI],
            value=initial_engine_value,
            info="Standard = current behavior. Improved = slower, higher quality (learned rounding + bias correction).",
        )

    # Improved engine settings (hidden unless selected)
    with gr.Column(visible=(initial_engine_value == ENGINE_IMPROVED_UI)) as improved_settings_col:
        with gr.Accordion("Improved FP8 settings (convert_to_quant)", open=False):
            gr.Markdown(
                "These settings are only used when **FP8 Engine = Improved**.\n\n"
                "Tip: start with Tensor mode + ~200 iterations + 256 bias samples, then increase if you want more quality."
            )

            # Quick presets (primarily tuned for Qwen Image)
            PRESET_CUSTOM = "Custom (manual)"
            PRESET_A = "Preset A (balanced quality)"
            PRESET_B = "Preset B (high quality)"
            PRESET_C = "Preset C (max quality / slowest)"
            PRESET_D = "Preset D (your defaults)"
            improved_preset = gr.Dropdown(
                label="Quality Preset",
                choices=[PRESET_CUSTOM, PRESET_A, PRESET_B, PRESET_C, PRESET_D],
                value=config.get("fp8_converter.improved.preset", PRESET_CUSTOM),
                info="Applies a bundle of recommended settings. You can still tweak values after selecting a preset.",
            )

            improved_comfy_quant = gr.Checkbox(
                label="Output ComfyUI comfy_quant format (adds .comfy_quant tensors)",
                value=config.get("fp8_converter.improved.comfy_quant", False),
                info="OFF keeps legacy scaled_fp8 format (scale_weight + scaled_fp8), matching current Musubi behavior.",
            )
            improved_save_both_formats = gr.Checkbox(
                label="Save both formats (legacy scaled_fp8 + comfy_quant) in one run",
                value=config.get("fp8_converter.improved.save_both_formats", False),
                info="Runs the expensive quantization once (legacy), then creates a comfy_quant copy via fast format conversion.",
            )
            improved_full_precision_mm = gr.Checkbox(
                label="comfy_quant: Full precision matrix multiply (recommended if you get noise)",
                value=config.get("fp8_converter.improved.full_precision_matrix_mult", False),
                info="Writes full_precision_matrix_mult=true into .comfy_quant. This avoids FP8 activation matmul and behaves more like legacy scaled_fp8 (better quality, slower).",
            )
            improved_skip_inefficient_layers = gr.Checkbox(
                label="Skip inefficient layers (heuristics)",
                value=config.get("fp8_converter.improved.skip_inefficient_layers", True),
                info="Faster; keeps some layers high precision when they are poor candidates for quantization.",
            )

            with gr.Row():
                improved_num_iter = gr.Slider(
                    label="Iterations per tensor (learned rounding)",
                    minimum=0,
                    maximum=3000,
                    step=25,
                    value=config.get("fp8_converter.improved.num_iter", 200),
                    info="Higher = better quality, slower. 0 disables learned rounding.",
                )
                improved_calib_samples = gr.Slider(
                    label="Bias correction samples",
                    minimum=0,
                    maximum=6144,
                    step=64,
                    value=config.get("fp8_converter.improved.calib_samples", 256),
                    info="Higher = better bias correction, more VRAM/time.",
                )

            with gr.Row():
                improved_optimizer = gr.Dropdown(
                    label="Optimizer",
                    choices=["original", "adamw", "radam"],
                    value=config.get("fp8_converter.improved.optimizer", "original"),
                )
                improved_lr_schedule = gr.Dropdown(
                    label="LR Schedule (original optimizer)",
                    choices=["adaptive", "exponential", "plateau"],
                    value=config.get("fp8_converter.improved.lr_schedule", "adaptive"),
                )
                improved_lr = gr.Number(
                    label="Learning Rate",
                    value=config.get("fp8_converter.improved.lr", 8.077300000003e-3),
                    precision=12,
                )

            improved_seed = gr.Number(
                label="Seed (use -1 for random)",
                value=config.get("fp8_converter.improved.seed", 0),
                precision=0,
            )

            with gr.Accordion("Advanced SVD options", open=False):
                improved_top_p = gr.Number(
                    label="top_p (proportion of principal components)",
                    value=config.get("fp8_converter.improved.top_p", 0.01),
                    precision=6,
                )
                with gr.Row():
                    improved_min_k = gr.Number(
                        label="min_k",
                        value=config.get("fp8_converter.improved.min_k", 1),
                        precision=0,
                    )
                    improved_max_k = gr.Number(
                        label="max_k",
                        value=config.get("fp8_converter.improved.max_k", 16),
                        precision=0,
                    )

                improved_full_matrix = gr.Checkbox(
                    label="Use full SVD (slower, more memory)",
                    value=config.get("fp8_converter.improved.full_matrix", False),
                )
                improved_no_learned_rounding = gr.Checkbox(
                    label="Disable learned rounding (simple quantization)",
                    value=config.get("fp8_converter.improved.no_learned_rounding", False),
                )
    
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

    # Toggle improved settings visibility
    fp8_engine.change(
        fn=lambda e: gr.update(visible=(e == ENGINE_IMPROVED_UI)),
        inputs=[fp8_engine],
        outputs=[improved_settings_col],
        show_progress=False,
    )

    # Apply improved presets
    def _apply_improved_preset(preset_name: str):
        # Defaults: no change
        if preset_name == PRESET_A:
            return (
                gr.update(value="tensor"),  # single_quantization_mode
                gr.update(value="tensor"),  # batch_quantization_mode
                gr.update(),  # improved_comfy_quant
                gr.update(),  # improved_save_both_formats
                gr.update(value=True),  # improved_full_precision_mm
                gr.update(value=True),  # improved_skip_inefficient_layers
                gr.update(value=400),  # improved_num_iter
                gr.update(value=1024),  # improved_calib_samples
                gr.update(value="original"),  # improved_optimizer
                gr.update(value="adaptive"),  # improved_lr_schedule
                gr.update(value=8.077300000003e-3),  # improved_lr
                gr.update(value=0.02),  # improved_top_p
                gr.update(value=16),  # improved_min_k
                gr.update(value=64),  # improved_max_k
                gr.update(value=False),  # improved_full_matrix
                gr.update(value=False),  # improved_no_learned_rounding
            )
        if preset_name == PRESET_B:
            return (
                gr.update(value="tensor"),
                gr.update(value="tensor"),
                gr.update(),
                gr.update(),
                gr.update(value=True),
                gr.update(value=True),
                gr.update(value=800),
                gr.update(value=2048),
                gr.update(value="original"),
                gr.update(value="adaptive"),
                gr.update(value=8.077300000003e-3),
                gr.update(value=0.05),
                gr.update(value=32),
                gr.update(value=128),
                gr.update(value=False),
                gr.update(value=False),
            )
        if preset_name == PRESET_C:
            return (
                gr.update(value="tensor"),
                gr.update(value="tensor"),
                gr.update(),
                gr.update(),
                gr.update(value=True),
                gr.update(value=True),
                gr.update(value=1000),
                gr.update(value=4096),
                gr.update(value="original"),
                gr.update(value="adaptive"),
                gr.update(value=8.077300000003e-3),
                gr.update(value=0.1),
                gr.update(value=64),
                gr.update(value=256),
                gr.update(value=False),
                gr.update(value=False),
            )
        if preset_name == PRESET_D:
            # Matches the user-requested defaults (screenshot):
            # - comfy_quant ON
            # - full_precision_matrix_mult ON
            # - skip inefficient layers OFF
            # - num_iter 3000, calib_samples 4096
            # - top_p 0.1, k 64..256
            # - full SVD ON
            return (
                gr.update(value="tensor"),
                gr.update(value="tensor"),
                gr.update(value=True),   # improved_comfy_quant
                gr.update(value=False),  # improved_save_both_formats
                gr.update(value=True),   # improved_full_precision_mm
                gr.update(value=False),  # improved_skip_inefficient_layers
                gr.update(value=3000),
                gr.update(value=4096),
                gr.update(value="original"),
                gr.update(value="adaptive"),
                gr.update(value=8.077300000003e-3),
                gr.update(value=0.1),
                gr.update(value=64),
                gr.update(value=256),
                gr.update(value=True),   # improved_full_matrix
                gr.update(value=False),  # improved_no_learned_rounding
            )
        # Custom: no changes
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    improved_preset.change(
        fn=_apply_improved_preset,
        inputs=[improved_preset],
        outputs=[
            single_quantization_mode,
            batch_quantization_mode,
            improved_comfy_quant,
            improved_save_both_formats,
            improved_full_precision_mm,
            improved_skip_inefficient_layers,
            improved_num_iter,
            improved_calib_samples,
            improved_optimizer,
            improved_lr_schedule,
            improved_lr,
            improved_top_p,
            improved_min_k,
            improved_max_k,
            improved_full_matrix,
            improved_no_learned_rounding,
        ],
        show_progress=False,
    )
    
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
    def convert_single_with_model_type(
        input_file,
        output_file,
        engine_ui,
        quantization_mode,
        delete_original,
        model_type_ui,
        comfy_quant,
        save_both_formats,
        full_precision_mm,
        skip_inefficient_layers,
        calib_samples,
        seed,
        optimizer,
        num_iter,
        lr,
        lr_schedule,
        top_p,
        min_k,
        max_k,
        full_matrix,
        no_learned_rounding,
    ):
        model_type = get_internal_model_type(model_type_ui)
        engine = get_internal_engine(engine_ui)
        return converter.convert_single_file(
            input_file=input_file,
            output_file=output_file,
            quantization_mode=quantization_mode,
            delete_original=delete_original,
            model_type=model_type,
            engine=engine,
            improved_comfy_quant=comfy_quant,
            improved_save_both_formats=bool(save_both_formats),
            improved_full_precision_matrix_mult=bool(full_precision_mm),
            improved_skip_inefficient_layers=skip_inefficient_layers,
            improved_calib_samples=_as_int(calib_samples, 256),
            improved_seed=_as_int(seed, 0),
            improved_optimizer=str(optimizer),
            improved_num_iter=_as_int(num_iter, 200),
            improved_lr=_as_float(lr, 8.077300000003e-3),
            improved_lr_schedule=str(lr_schedule),
            improved_top_p=_as_float(top_p, 0.01),
            improved_min_k=_as_int(min_k, 1),
            improved_max_k=_as_int(max_k, 16),
            improved_full_matrix=bool(full_matrix),
            improved_no_learned_rounding=bool(no_learned_rounding),
        )
    
    single_convert_button.click(
        fn=convert_single_with_model_type,
        inputs=[
            single_input_file,
            single_output_file,
            fp8_engine,
            single_quantization_mode,
            single_delete_original,
            single_model_type,
            improved_comfy_quant,
            improved_save_both_formats,
            improved_full_precision_mm,
            improved_skip_inefficient_layers,
            improved_calib_samples,
            improved_seed,
            improved_optimizer,
            improved_num_iter,
            improved_lr,
            improved_lr_schedule,
            improved_top_p,
            improved_min_k,
            improved_max_k,
            improved_full_matrix,
            improved_no_learned_rounding,
        ],
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
    def convert_batch_with_model_type(
        input_folder,
        output_folder,
        engine_ui,
        quantization_mode,
        delete_original,
        model_type_ui,
        comfy_quant,
        save_both_formats,
        full_precision_mm,
        skip_inefficient_layers,
        calib_samples,
        seed,
        optimizer,
        num_iter,
        lr,
        lr_schedule,
        top_p,
        min_k,
        max_k,
        full_matrix,
        no_learned_rounding,
    ):
        model_type = get_internal_model_type(model_type_ui)
        engine = get_internal_engine(engine_ui)
        return converter.batch_convert_models(
            input_folder=input_folder,
            output_folder=output_folder,
            quantization_mode=quantization_mode,
            delete_original=delete_original,
            model_type=model_type,
            engine=engine,
            improved_comfy_quant=comfy_quant,
            improved_save_both_formats=bool(save_both_formats),
            improved_full_precision_matrix_mult=bool(full_precision_mm),
            improved_skip_inefficient_layers=skip_inefficient_layers,
            improved_calib_samples=_as_int(calib_samples, 256),
            improved_seed=_as_int(seed, 0),
            improved_optimizer=str(optimizer),
            improved_num_iter=_as_int(num_iter, 200),
            improved_lr=_as_float(lr, 8.077300000003e-3),
            improved_lr_schedule=str(lr_schedule),
            improved_top_p=_as_float(top_p, 0.01),
            improved_min_k=_as_int(min_k, 1),
            improved_max_k=_as_int(max_k, 16),
            improved_full_matrix=bool(full_matrix),
            improved_no_learned_rounding=bool(no_learned_rounding),
        )
    
    batch_convert_button.click(
        fn=convert_batch_with_model_type,
        inputs=[
            input_folder,
            output_folder,
            fp8_engine,
            batch_quantization_mode,
            batch_delete_original,
            batch_model_type,
            improved_comfy_quant,
            improved_save_both_formats,
            improved_full_precision_mm,
            improved_skip_inefficient_layers,
            improved_calib_samples,
            improved_seed,
            improved_optimizer,
            improved_num_iter,
            improved_lr,
            improved_lr_schedule,
            improved_top_p,
            improved_min_k,
            improved_max_k,
            improved_full_matrix,
            improved_no_learned_rounding,
        ],
        outputs=[output_status],
        show_progress=True,
    )

