# Qwen Image Configuration Update Report

## Date: 2025-08-25

## Summary
Performed a comprehensive analysis and update of Qwen Image LoRA training configuration based on the latest musubi-tuner repository commits and source code.

## Key Changes Made

### 1. **qwen_image_defaults.toml Updates**

#### Learning Rate & Network Settings
- **learning_rate**: Verified at `5e-5` (recommended in commit 82c5ce1, changed from previous 1e-4)
- **network_dim**: Verified at `16` (recommended in commit 82c5ce1, changed from previous 32)
- **network_alpha**: Updated to `16.0` (set equal to network_dim for best results, was 1.0)

#### Documentation Improvements
- Added detailed comments for network_args parameter explaining supported options:
  - loraplus_lr_ratio: Enables LoRA+ optimization
  - exclude_patterns: Regex patterns to exclude modules (default excludes modulation layers)
  - include_patterns: Regex patterns to re-include excluded modules
  - verbose: Detailed logging of trained modules

#### Sample Generation
- Updated sample_every_n_epochs default to `1` (recommended for monitoring training progress)
- Added documentation for sample_prompts parameter

### 2. **qwen_image_lora_gui.py Script Path Fixes**

Fixed incorrect script paths in the GUI:
- `./musubi-tuner/qwen_image_cache_latents.py` → `./musubi-tuner/src/musubi_tuner/qwen_image_cache_latents.py`
- `./musubi-tuner/qwen_image_cache_text_encoder_outputs.py` → `./musubi-tuner/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py`
- `{scriptdir}/musubi-tuner/qwen_image_train_network.py` → `{scriptdir}/musubi-tuner/src/musubi_tuner/qwen_image_train_network.py`

## Verified Parameters Against Source Code

### Core Training Parameters (Confirmed Correct)
- **timestep_sampling**: "shift" (Qwen Image default, base code default is "sigma")
- **discrete_flow_shift**: 2.2 (optimal for Qwen Image)
- **weighting_scheme**: "none" (recommended for Qwen Image)
- **optimizer_type**: "adamw8bit" (recommended)
- **mixed_precision**: "bf16" (required for Qwen Image)
- **network_module**: "networks.lora_qwen_image" (required)

### Memory Optimization Parameters (Verified)
- **fp8_vl**: false (enable for <16GB VRAM)
- **fp8_base**: false (saves ~12GB VRAM when enabled)
- **fp8_scaled**: false (required when fp8_base=true)
- **blocks_to_swap**: 0 (16=save ~8GB, 45=save ~30GB VRAM)
- **gradient_checkpointing**: true (recommended)

### VAE Settings (Confirmed Supported)
- **vae_tiling**: Supported in Qwen Image (contrary to some outdated comments)
- **vae_chunk_size**: Available for optimization
- **vae_spatial_tile_sample_min_size**: Available (auto-enables tiling if set)

### Qwen-Image-Edit Support (Verified)
- **edit**: false (parameter present and functional)
- Control image processing fully supported

## Recommendations Based on Analysis

1. **For Standard Training (24GB VRAM)**:
   - Use default settings as configured
   - Enable gradient_checkpointing
   - Use adamw8bit optimizer

2. **For Limited VRAM (<16GB)**:
   - Enable fp8_vl=true
   - Consider fp8_base=true with fp8_scaled=true
   - Use blocks_to_swap=16 or higher

3. **For Best Quality**:
   - Keep learning_rate at 5e-5
   - Use network_dim=16 with network_alpha=16
   - Enable sample_every_n_epochs=1 for monitoring

## Source Verification
All changes based on:
- Last 30 commits from musubi-tuner repository
- Direct source code analysis of hv_train_network.py
- qwen_image_train_network.py specific parameters
- Official documentation in docs/qwen_image.md
- Commit 82c5ce1 (2025-08-21): Updated learning rate and network dimension recommendations

## Status
✅ All parameters verified and updated
✅ GUI script paths corrected
✅ Documentation enhanced
✅ Defaults align with latest recommendations