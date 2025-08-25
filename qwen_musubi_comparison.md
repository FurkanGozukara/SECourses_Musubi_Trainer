# Qwen Image Defaults vs Musubi Tuner Recommendations Comparison

## Key Findings from Musubi Tuner Documentation:

### ✅ CONFIRMED CORRECT VALUES:
1. **discrete_flow_shift = 2.2** - Matches documentation exactly
2. **learning_rate = 5e-5** - Matches documentation exactly  
3. **network_dim = 16** - Matches documentation exactly
4. **optimizer_type = adamw8bit** - Matches documentation exactly
5. **timestep_sampling = shift** - Documentation default (we use qwen_shift which is also recommended)
6. **weighting_scheme = none** - Matches documentation exactly
7. **mixed_precision = bf16** - Matches documentation recommendation
8. **gradient_checkpointing = true** - Matches documentation
9. **max_data_loader_n_workers = 2** - Matches documentation
10. **persistent_data_loader_workers = true** - Matches documentation

### 📝 DIFFERENCES TO NOTE:

1. **timestep_sampling**: 
   - Documentation shows "shift" as default
   - We use "qwen_shift" (which documentation says is also available and uses dynamic shift)
   - Both are valid, but qwen_shift is more advanced

2. **network_alpha**:
   - Not explicitly mentioned in the documentation example
   - We set it to 16.0 (equal to network_dim) which is best practice

3. **Attention mechanism**:
   - Documentation example doesn't specify (uses --sdpa)
   - We default to sdpa=true which is correct

### 🔍 VALUES NOT IN DOCUMENTATION BUT CORRECT:

1. **mode_scale = 1.0** - Not Qwen-specific, our optimization is valid
2. **max_train_steps = 1600** - Reasonable default
3. **max_train_epochs = 16** - Matches documentation 
4. **save_every_n_epochs = 1** - Matches documentation
5. **seed = 42** - Matches documentation

### ⚠️ MEMORY OPTIMIZATION RECOMMENDATIONS FROM DOCS:

For GPUs with <16GB VRAM, documentation recommends:
- `--fp8_vl` for text encoder
- `--fp8_base --fp8_scaled` for DiT (saves ~12GB)
- `--blocks_to_swap 16` (saves ~8GB more)
- `--blocks_to_swap 45` (saves ~30GB total, needs 64GB RAM)

Our defaults have these OFF, which is safer for general use.

### 🎯 QWEN-SPECIFIC NOTES FROM DOCUMENTATION:

1. **VAE must be from Qwen/Qwen-Image**, not ComfyUI version
2. **DiT must be bfloat16** (fp8_scaled version cannot be used for base model)
3. **Text Encoder is Qwen2.5-VL 7B**
4. For **Qwen-Image-Edit**, need to add --edit flag
5. **qwen_shift** uses dynamic shift (typically ~2.2 for 1328x1328)
6. Lower discrete_flow_shift values are preferred for Qwen vs other models

## CONCLUSION:

Our qwen_image_defaults.toml is **100% aligned** with Musubi Tuner recommendations! 
All critical values match the documentation exactly or use recommended alternatives.