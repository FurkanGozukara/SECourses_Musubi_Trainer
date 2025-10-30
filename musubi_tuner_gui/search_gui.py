import gradio as gr
import json
from typing import Dict, List, Tuple, Any

class ParameterSearcher:
    """Search functionality for finding parameters in the GUI"""
    
    def __init__(self):
        self.parameter_map = self._build_parameter_map()
    
    def _build_parameter_map(self) -> Dict[str, Dict[str, Any]]:
        """Build a searchable map of all parameters with their locations and descriptions"""
        
        parameter_map = {
            # Model Settings
            "dit": {
                "display_name": "DiT (Base Model) Checkpoint Path",
                "tab": "Qwen Image LoRA",
                "section": "Model Settings",
                "keywords": ["dit", "base model", "checkpoint", "model path", "diffusion transformer"],
                "description": "Path to DiT base model checkpoint (qwen_image_bf16.safetensors)"
            },
            "vae": {
                "display_name": "VAE Checkpoint Path",
                "tab": "Qwen Image LoRA",
                "section": "Model Settings",
                "keywords": ["vae", "variational autoencoder", "encoder", "decoder"],
                "description": "Path to VAE model (diffusion_pytorch_model.safetensors from Qwen/Qwen-Image)"
            },
            "text_encoder": {
                "display_name": "Text Encoder (Qwen2.5-VL) Path",
                "tab": "Qwen Image LoRA",
                "section": "Model Settings",
                "keywords": ["text encoder", "qwen2.5", "vl", "vision language", "clip"],
                "description": "Path to Qwen2.5-VL text encoder model"
            },
            "dit_in_channels": {
                "display_name": "DiT Input Channels",
                "tab": "Qwen Image LoRA",
                "section": "Model Settings",
                "keywords": ["channels", "latent", "vae channels", "16"],
                "description": "VAE latent channels (MUST be 16 for Qwen Image)"
            },
            "training_mode": {
                "display_name": "Training Mode",
                "tab": "Qwen Image LoRA",
                "section": "Model Settings",
                "keywords": ["lora", "dreambooth", "fine-tuning", "training mode"],
                "description": "LoRA (efficient) or DreamBooth (full fine-tuning)"
            },
            "edit": {
                "display_name": "Enable Qwen-Image-Edit Mode",
                "tab": "Qwen Image LoRA",
                "section": "Model Settings",
                "keywords": ["edit", "control", "image edit", "control image"],
                "description": "Enable image editing training with control images"
            },
            
            # FP8 Quantization
            "fp8_base": {
                "display_name": "FP8 for Base Model (DiT) (BF16 Model On The Fly Converted)",
                "tab": "Qwen Image LoRA",
                "section": "FP8 Quantization",
                "keywords": ["fp8", "quantization", "dit", "base", "memory", "vram"],
                "description": "Converts bf16 model to FP8 on-the-fly, reducing VRAM usage"
            },
            "fp8_scaled": {
                "display_name": "Scaled FP8 for Base Model (DiT) (BF16 Model On The Fly Converted - Better FP8 Precision)",
                "tab": "Qwen Image LoRA",
                "section": "FP8 Quantization",
                "keywords": ["fp8", "scaled", "quantization", "quality"],
                "description": "Required with fp8_base for best quality"
            },
            "fp8_vl": {
                "display_name": "Use FP8 for Text Encoder",
                "tab": "Qwen Image LoRA",
                "section": "FP8 Quantization",
                "keywords": ["fp8", "text encoder", "vl", "memory", "vram"],
                "description": "FP8 quantization for Qwen2.5-VL reduces VRAM usage"
            },
            "blocks_to_swap": {
                "display_name": "Blocks to Swap to CPU",
                "tab": "Qwen Image LoRA",
                "section": "FP8 Quantization",
                "keywords": ["blocks", "swap", "cpu", "offload", "memory"],
                "description": "Swap DiT blocks to CPU to save VRAM (max 59)"
            },
            
            # VAE Optimization
            "vae_tiling": {
                "display_name": "VAE Tiling",
                "tab": "Qwen Image LoRA",
                "section": "VAE Optimization",
                "keywords": ["vae", "tiling", "spatial", "memory", "vram"],
                "description": "Enable spatial tiling for VAE to reduce VRAM usage"
            },
            "vae_chunk_size": {
                "display_name": "VAE Chunk Size",
                "tab": "Qwen Image LoRA",
                "section": "VAE Optimization",
                "keywords": ["vae", "chunk", "causal", "conv3d"],
                "description": "Chunk size for CausalConv3d in VAE"
            },
            "vae_spatial_tile_sample_min_size": {
                "display_name": "VAE Spatial Tile Min Size",
                "tab": "Qwen Image LoRA",
                "section": "VAE Optimization",
                "keywords": ["vae", "spatial", "tile", "minimum"],
                "description": "Minimum spatial tile size for VAE"
            },
            
            # Flow Matching
            "timestep_sampling": {
                "display_name": "Timestep Sampling Method",
                "tab": "Qwen Image LoRA",
                "section": "Flow Matching",
                "keywords": ["timestep", "sampling", "qwen_shift", "shift", "sigma"],
                "description": "qwen_shift = dynamic shift per resolution (best for Qwen)"
            },
            "discrete_flow_shift": {
                "display_name": "Discrete Flow Shift",
                "tab": "Qwen Image LoRA",
                "section": "Flow Matching",
                "keywords": ["discrete", "flow", "shift", "2.2"],
                "description": "Only used with 'shift' method. Qwen Image optimal: 2.2"
            },
            "flow_shift": {
                "display_name": "Flow Shift (Advanced)",
                "tab": "Qwen Image LoRA",
                "section": "Flow Matching",
                "keywords": ["flow", "shift", "noise", "schedule", "7.0"],
                "description": "Controls noise schedule in flow matching. Default 7.0 is optimal"
            },
            "weighting_scheme": {
                "display_name": "Weighting Scheme",
                "tab": "Qwen Image LoRA",
                "section": "Flow Matching",
                "keywords": ["weighting", "scheme", "logit", "mode", "none"],
                "description": "'none' recommended for Qwen Image"
            },
            "guidance_scale": {
                "display_name": "Guidance Scale",
                "tab": "Qwen Image LoRA",
                "section": "Flow Matching",
                "keywords": ["guidance", "cfg", "classifier-free", "scale"],
                "description": "Classifier-free guidance scale. Default: 1.0"
            },
            
            # Training Settings
            "learning_rate": {
                "display_name": "Learning Rate",
                "tab": "Qwen Image LoRA",
                "section": "Optimizer Settings",
                "keywords": ["learning rate", "lr", "5e-5", "optimizer"],
                "description": "Learning rate for training (5e-5 recommended)"
            },
            "optimizer_type": {
                "display_name": "Optimizer Type",
                "tab": "Qwen Image LoRA",
                "section": "Optimizer Settings",
                "keywords": ["optimizer", "adamw", "8bit", "adam"],
                "description": "Optimizer algorithm (adamw8bit recommended)"
            },
            "max_train_epochs": {
                "display_name": "Max Training Epochs",
                "tab": "Qwen Image LoRA",
                "section": "Training Settings",
                "keywords": ["epochs", "training", "duration", "iterations"],
                "description": "Maximum training epochs (16 recommended)"
            },
            "max_train_steps": {
                "display_name": "Max Training Steps",
                "tab": "Qwen Image LoRA",
                "section": "Training Settings",
                "keywords": ["steps", "training", "iterations", "1600"],
                "description": "Total training steps (alternative to epochs)"
            },
            "gradient_checkpointing": {
                "display_name": "Gradient Checkpointing",
                "tab": "Qwen Image LoRA",
                "section": "Training Settings",
                "keywords": ["gradient", "checkpointing", "memory", "vram"],
                "description": "Trade compute for memory by recomputing activations"
            },
            "gradient_accumulation_steps": {
                "display_name": "Gradient Accumulation Steps",
                "tab": "Qwen Image LoRA",
                "section": "Training Settings",
                "keywords": ["gradient", "accumulation", "batch", "effective"],
                "description": "Accumulate gradients over multiple steps"
            },
            
            # Network/LoRA Settings
            "network_dim": {
                "display_name": "Network Dimension (Rank)",
                "tab": "Qwen Image LoRA",
                "section": "Network Settings",
                "keywords": ["network", "dim", "rank", "lora", "16"],
                "description": "LoRA rank/dimension (16 recommended)"
            },
            "network_alpha": {
                "display_name": "Network Alpha",
                "tab": "Qwen Image LoRA",
                "section": "Network Settings",
                "keywords": ["network", "alpha", "scale", "lora"],
                "description": "Alpha for scaling (should equal network_dim)"
            },
            "network_dropout": {
                "display_name": "Network Dropout",
                "tab": "Qwen Image LoRA",
                "section": "Network Settings",
                "keywords": ["network", "dropout", "regularization", "lora"],
                "description": "Dropout rate for LoRA layers"
            },
            "network_module": {
                "display_name": "Network Module",
                "tab": "Qwen Image LoRA",
                "section": "Network Settings",
                "keywords": ["network", "module", "lora", "qwen"],
                "description": "LoRA implementation module"
            },
            
            # Dataset Settings
            "dataset_config": {
                "display_name": "Dataset Config File",
                "tab": "Qwen Image LoRA",
                "section": "Dataset Settings",
                "keywords": ["dataset", "config", "toml", "data"],
                "description": "Path to TOML file for training dataset configuration"
            },
            "dataset_resolution_width": {
                "display_name": "Resolution Width",
                "tab": "Qwen Image LoRA",
                "section": "Dataset Settings",
                "keywords": ["resolution", "width", "size", "1328"],
                "description": "Width of training images"
            },
            "dataset_resolution_height": {
                "display_name": "Resolution Height",
                "tab": "Qwen Image LoRA",
                "section": "Dataset Settings",
                "keywords": ["resolution", "height", "size", "1328"],
                "description": "Height of training images"
            },
            "dataset_enable_bucket": {
                "display_name": "Enable Bucketing",
                "tab": "Qwen Image LoRA",
                "section": "Dataset Settings",
                "keywords": ["bucket", "aspect ratio", "variable", "size"],
                "description": "Enable aspect ratio bucketing"
            },
            
            # Attention Settings
            "sdpa": {
                "display_name": "Use SDPA for CrossAttention",
                "tab": "Qwen Image LoRA",
                "section": "Attention Settings",
                "keywords": ["sdpa", "attention", "pytorch", "scaled dot product"],
                "description": "PyTorch's Scaled Dot Product Attention (fastest)"
            },
            "flash_attn": {
                "display_name": "Use FlashAttention",
                "tab": "Qwen Image LoRA",
                "section": "Attention Settings",
                "keywords": ["flash", "attention", "memory", "efficient"],
                "description": "Memory-efficient attention implementation"
            },
            "xformers": {
                "display_name": "Use xformers",
                "tab": "Qwen Image LoRA",
                "section": "Attention Settings",
                "keywords": ["xformers", "attention", "memory"],
                "description": "Memory-efficient attention from xformers library"
            },
            
            # Output Settings
            "output_dir": {
                "display_name": "Output Directory",
                "tab": "Qwen Image LoRA",
                "section": "Output Settings",
                "keywords": ["output", "directory", "save", "folder"],
                "description": "Directory to save trained model"
            },
            "output_name": {
                "display_name": "Output Name",
                "tab": "Qwen Image LoRA",
                "section": "Output Settings",
                "keywords": ["output", "name", "model", "filename"],
                "description": "Name for the trained model file"
            },
            "save_every_n_epochs": {
                "display_name": "Save Every N Epochs",
                "tab": "Qwen Image LoRA",
                "section": "Output Settings",
                "keywords": ["save", "checkpoint", "epochs", "frequency"],
                "description": "Save checkpoint every N epochs"
            },
            
            # Sample Generation
            "sample_prompts": {
                "display_name": "Sample Prompts",
                "tab": "Qwen Image LoRA",
                "section": "Sample Generation",
                "keywords": ["sample", "prompts", "test", "generation"],
                "description": "Prompts for generating sample images during training"
            },
            "sample_every_n_epochs": {
                "display_name": "Sample Every N Epochs",
                "tab": "Qwen Image LoRA",
                "section": "Sample Generation",
                "keywords": ["sample", "frequency", "epochs", "generation"],
                "description": "Generate samples every N epochs"
            },
            "sample_width": {
                "display_name": "Sample Width",
                "tab": "Qwen Image LoRA",
                "section": "Sample Generation",
                "keywords": ["sample", "width", "resolution", "1328"],
                "description": "Width of generated sample images"
            },
            "sample_height": {
                "display_name": "Sample Height",
                "tab": "Qwen Image LoRA",
                "section": "Sample Generation",
                "keywords": ["sample", "height", "resolution", "1328"],
                "description": "Height of generated sample images"
            },
        }
        
        return parameter_map
    
    def search_parameters(self, query: str) -> List[Dict[str, Any]]:
        """Search for parameters matching the query"""
        if not query or len(query.strip()) < 2:
            return []
        
        query_lower = query.lower().strip()
        results = []
        
        for param_id, param_info in self.parameter_map.items():
            score = 0
            
            # Check parameter ID
            if query_lower in param_id.lower():
                score += 10
            
            # Check display name
            if query_lower in param_info["display_name"].lower():
                score += 8
            
            # Check keywords
            for keyword in param_info.get("keywords", []):
                if query_lower in keyword.lower():
                    score += 5
                    break
            
            # Check description
            if query_lower in param_info.get("description", "").lower():
                score += 3
            
            # Check section
            if query_lower in param_info.get("section", "").lower():
                score += 2
            
            if score > 0:
                results.append({
                    "id": param_id,
                    "score": score,
                    **param_info
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:20]  # Return top 20 results
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as HTML for display"""
        if not results:
            return "<div style='padding: 10px; color: #666;'>No parameters found matching your search.</div>"
        
        html = "<div style='padding: 10px;'>"
        html += f"<div style='margin-bottom: 10px; color: #666;'>Found {len(results)} parameter(s):</div>"
        
        for result in results:
            html += f"""
            <div style='border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin-bottom: 10px; background: #f9f9f9;'>
                <div style='font-weight: bold; color: #333; margin-bottom: 5px;'>
                    {result['display_name']}
                </div>
                <div style='color: #666; font-size: 0.9em; margin-bottom: 5px;'>
                    📍 {result['tab']} → {result['section']}
                </div>
                <div style='color: #555; font-size: 0.85em; margin-bottom: 5px;'>
                    {result['description']}
                </div>
                <div style='color: #888; font-size: 0.8em;'>
                    <strong>Parameter ID:</strong> {result['id']}
                </div>
            </div>
            """
        
        html += "</div>"
        return html

def search_parameters_interface(headless: bool = False):
    """Create the search interface for parameters"""
    searcher = ParameterSearcher()
    
    with gr.Column():
        gr.Markdown("""
        ## 🔍 Parameter Search
        
        Search for any parameter by name, keyword, or description. This helps you quickly find and understand specific settings.
        """)
        
        with gr.Row():
            search_input = gr.Textbox(
                label="Search Parameters",
                placeholder="Enter parameter name, keyword, or description (e.g., 'learning rate', 'vram', 'fp8', 'epochs')",
                lines=1,
                scale=4
            )
            search_button = gr.Button("Search", variant="primary", scale=1)
        
        search_results = gr.HTML(
            value="<div style='padding: 10px; color: #666;'>Enter a search term above to find parameters.</div>",
            label="Search Results"
        )
        
        gr.Markdown("""
        ### 💡 Quick Search Tips:
        - **Memory/VRAM**: Search for "vram", "memory", "fp8", "swap", "tiling"
        - **Training**: Search for "epochs", "steps", "learning rate", "optimizer"
        - **Model**: Search for "dit", "vae", "text encoder", "checkpoint"
        - **LoRA**: Search for "network", "dim", "alpha", "dropout", "lora"
        - **Dataset**: Search for "dataset", "resolution", "bucket", "caption"
        - **Samples**: Search for "sample", "prompts", "generation"
        """)
        
        def perform_search(query):
            results = searcher.search_parameters(query)
            return searcher.format_search_results(results)
        
        # Connect events
        search_button.click(
            fn=perform_search,
            inputs=[search_input],
            outputs=[search_results]
        )
        
        # Also search on Enter key
        search_input.submit(
            fn=perform_search,
            inputs=[search_input],
            outputs=[search_results]
        )