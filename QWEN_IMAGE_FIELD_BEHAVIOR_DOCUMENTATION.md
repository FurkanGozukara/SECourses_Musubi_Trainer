# Qwen Image Training - Field Behavior Documentation

This document explains the behavior of all fields in the Qwen Image LoRA training configuration, specifically what happens when fields are set to 0, empty, or None.

## 🔴 CRITICAL REQUIRED FIELDS (Must be specified by user)

These fields MUST be set by the user or training will fail:

- **`dit`** - Path to DiT checkpoint file
  - ❌ Empty/None: Training fails with "DiT checkpoint path is required"
  - ✅ Valid path: Loads the base Qwen Image model

- **`vae`** - Path to VAE checkpoint file  
  - ❌ Empty/None: Training fails with "VAE checkpoint path is required"
  - ✅ Valid path: Loads VAE for encoding/decoding images

- **`text_encoder`** - Path to Qwen2.5-VL checkpoint
  - ❌ Empty/None: Uses default path or fails if not found
  - ✅ Valid path: Loads text encoder for processing captions

- **`dataset_config`** - Path to dataset TOML configuration
  - ❌ Empty/None: Training fails with "Dataset config file is required"
  - ✅ Valid path: Loads dataset configuration (images, captions, etc.)

- **`output_dir`** - Directory to save trained models
  - ❌ Empty/None: Training fails with "Output directory is required"
  - ✅ Valid path: Creates directory and saves models there

- **`output_name`** - Base filename for saved models
  - ❌ Empty/None: Training fails with "Output name is required"  
  - ✅ Valid name: Used as prefix for saved .safetensors files

## 🟡 NUMERIC FIELDS - Special Behavior When 0

### Training Control
- **`max_train_steps = 1600`**
  - `0`: Training would never run (invalid)
  - `> 0`: Number of training steps to perform
  - Note: Ignored if `max_train_epochs` is set

- **`max_train_epochs = 16`**  
  - `0`: Uses `max_train_steps` instead
  - `> 0`: Number of full dataset passes, overrides `max_train_steps`

- **`seed = 42`**
  - `0`: Uses random seed (different results each run)
  - `> 0`: Fixed seed for reproducible results

### Memory & Performance
- **`blocks_to_swap = 0`**
  - `0`: ✅ No CPU swapping (normal operation)
  - `16`: Saves ~8GB VRAM, requires 32GB+ RAM
  - `45`: Saves ~30GB VRAM, requires 64GB+ RAM

- **`gradient_accumulation_steps = 1`**
  - `0`: Invalid, would cause training failure
  - `1`: ✅ Normal operation, update every step
  - `> 1`: Accumulate gradients over N steps (simulates larger batch size)

- **`max_data_loader_n_workers = 2`**
  - `0`: Single-threaded data loading (slower but uses less RAM)
  - `> 0`: Parallel data loading workers

### Learning Rate & Optimization
- **`learning_rate = 1e-4`**
  - `0.0`: No learning occurs (weights never update)
  - `> 0`: Learning rate for optimizer

- **`max_grad_norm = 1.0`**
  - `0.0`: ✅ Gradient clipping disabled
  - `> 0.0`: Maximum gradient norm for clipping (prevents exploding gradients)

- **`lr_warmup_steps = 0`**
  - `0`: ✅ No warmup, starts at full learning rate
  - `> 0`: Gradually increase LR over N steps
  - `0.1`: If float <1, treats as ratio of total steps (10% of training)

- **`lr_decay_steps = 0`**
  - `0`: ✅ No decay, constant learning rate  
  - `> 0`: Decay learning rate over N steps
  - `0.2`: If float <1, treats as ratio of total steps (20% of training)

### Network Architecture
- **`network_dim = 32`**
  - `0`: Invalid, would create unusable LoRA
  - `> 0`: LoRA rank/dimension (higher = more capacity, larger files)

- **`network_alpha = 1.0`**
  - `0.0`: LoRA has no effect (scaling factor = 0)
  - `> 0.0`: LoRA scaling factor (alpha/rank = final scaling)

- **`network_dropout = 0.0`**
  - `0.0`: ✅ No dropout (normal training)
  - `> 0.0`: Dropout rate for regularization (0.1 = 10% dropout)

- **`scale_weight_norms = 0.0`**
  - `0.0`: ✅ No weight scaling (normal training)
  - `> 0.0`: Scale weights to prevent exploding gradients

### Save/Load Settings  
- **`save_every_n_epochs = 1`**
  - `0`: Only save final model at end
  - `> 0`: Save checkpoint every N epochs

- **`save_every_n_steps = 0`**
  - `0`: ✅ Step-based saving disabled
  - `> 0`: Save checkpoint every N steps (overrides epoch saving)

- **`save_last_n_epochs = 0`**
  - `0`: ✅ Keep all epoch checkpoints
  - `> 0`: Keep only last N epoch checkpoints (deletes older ones)

- **`save_last_n_steps = 0`**
  - `0`: ✅ Keep all step checkpoints  
  - `> 0`: Keep only last N step checkpoints

### Caching Settings
- **`caching_latent_batch_size = 4`**
  - `0`: Invalid, would cause errors
  - `> 0`: How many images to process simultaneously during caching

- **`caching_latent_num_workers = 8`**
  - `0`: Single-threaded image loading (slower)
  - `> 0`: Parallel workers for loading images

- **`caching_teo_batch_size = 16`**
  - `0`: Invalid, would cause errors  
  - `> 0`: Text encoder batch size during caching

### Flow Matching Parameters
- **`discrete_flow_shift = 3.0`**
  - `0.0`: Different flow matching behavior (not recommended)
  - `3.0`: ✅ Optimal for Qwen Image
  - Note: Ignored if `timestep_sampling = "qwen_shift"`

- **`logit_mean = 0.0`**
  - `0.0`: ✅ Balanced timestep sampling
  - `< 0.0`: Favor early timesteps  
  - `> 0.0`: Favor late timesteps

- **`logit_std = 1.0`**
  - `0.0`: Invalid (would cause division by zero)
  - `1.0`: ✅ Normal distribution
  - `< 1.0`: Concentrated sampling
  - `> 1.0`: Spread out sampling

### Sampling/Testing
- **`sample_every_n_steps = 0`**
  - `0`: ✅ No step-based sampling
  - `> 0`: Generate test images every N steps

- **`sample_every_n_epochs = 0`**
  - `0`: ✅ No epoch-based sampling
  - `> 0`: Generate test images every N epochs (overrides step sampling)

### Distributed Training
- **`ddp_timeout = 0`**
  - `0`: ✅ Use default timeout (30 minutes)
  - `> 0`: Custom timeout in minutes for distributed training

## 🟢 STRING FIELDS - Behavior When Empty

### Model Paths (Required)
- **`dit = ""`** - ❌ REQUIRED: Must specify DiT checkpoint path
- **`vae = ""`** - ❌ REQUIRED: Must specify VAE checkpoint path  
- **`text_encoder = ""`** - ❌ REQUIRED: Must specify Qwen2.5-VL path
- **`dataset_config = ""`** - ❌ REQUIRED: Must specify dataset config path

### Optional Model Paths
- **`network_weights = ""`**
  - Empty: ✅ Start training from scratch
  - Valid path: Continue training from existing LoRA

- **`base_weights = ""`**
  - Empty: ✅ No base weights to merge
  - Valid path: Merge existing LoRA into model before training

- **`resume = ""`**
  - Empty: ✅ Start fresh training
  - Valid path: Resume from training state file

### Logging & Tracking
- **`logging_dir = ""`**
  - Empty: ✅ No logging/TensorBoard output
  - Valid path: Enable logging to directory

- **`log_with = ""`**
  - Empty: ✅ Auto-detect ("tensorboard" if logging_dir set, else none)
  - "tensorboard": Use TensorBoard logging
  - "wandb": Use Weights & Biases
  - "all": Use both TensorBoard and WandB

- **`log_prefix = ""`**
  - Empty: ✅ No prefix for log directory names
  - String: Prefix added to timestamped log directories

- **`wandb_api_key = ""`**
  - Empty: ✅ Use environment variable WANDB_API_KEY
  - String: Use specified API key for WandB login

### Advanced Settings
- **`optimizer_args = ""`**
  - Empty: ✅ Use optimizer defaults
  - String: Extra parameters like "weight_decay=0.01 betas=0.9,0.999"

- **`network_args = ""`**
  - Empty: ✅ Use network defaults
  - String: LoRA parameters like "conv_dim=4 conv_alpha=1"

- **`additional_parameters = ""`**
  - Empty: ✅ No extra command line arguments
  - String: Custom parameters appended to training command

## 🔵 BOOLEAN FIELDS

All boolean fields default to `false` except where noted:

- **`fp8_vl = true`** - ✅ RECOMMENDED: Saves ~8GB VRAM
- **`fp8_base = false`** - Optional: FP8 for DiT (saves ~12GB VRAM)
- **`fp8_scaled = false`** - REQUIRED when fp8_base=true
- **`sdpa = true`** - ✅ RECOMMENDED: Fastest attention
- **`gradient_checkpointing = true`** - ✅ RECOMMENDED: Essential for VRAM savings
- **`save_state = false`** - Save optimizer states (large files)
- **`no_metadata = false`** - Exclude training metadata from LoRA file

## 🟠 DROPDOWN/CHOICE FIELDS

- **`dit_dtype = "bfloat16"`** - 🔒 HARDCODED: Must be bfloat16 for Qwen Image
- **`timestep_sampling = "shift"`**
  - "shift": Uses `discrete_flow_shift` value
  - "qwen_shift": ✅ Auto-calculates optimal shift per image resolution
- **`weighting_scheme = "none"`** - ✅ RECOMMENDED for Qwen Image
- **`optimizer_type = "adamw8bit"`** - ✅ RECOMMENDED: Memory efficient, confirmed in official Qwen Image examples

### Supported Optimizer Types

**Built-in Optimizers:**
- `adamw8bit` - ✅ **RECOMMENDED**: 8-bit AdamW from bitsandbytes, memory efficient
- `AdamW` - Standard AdamW optimizer from PyTorch
- `AdaFactor` - Adaptive learning rate optimizer from transformers

**Bitsandbytes Optimizers (via full path):**
- `bitsandbytes.optim.AdEMAMix8bit` - 8-bit AdEMAMix optimizer
- `bitsandbytes.optim.PagedAdEMAMix8bit` - Paged 8-bit AdEMAMix optimizer

**Standard PyTorch Optimizers (via full path):**
- `torch.optim.Adam` - Standard Adam optimizer  
- `torch.optim.SGD` - Stochastic Gradient Descent
- `torch.optim.RMSprop` - RMSprop optimizer
- Any other PyTorch optimizer via `torch.optim.OptimizerName`

**Custom Optimizers:**
- Any optimizer can be used by specifying the full module path (e.g., `some_package.CustomOptimizer`)
- Schedule-free optimizers are automatically detected by suffix `ScheduleFree`

**Why adamw8bit is recommended for Qwen Image:**
- Memory efficient (uses 8-bit quantization)
- Proven performance in official Qwen Image examples
- Works well with bfloat16 mixed precision
- Handles large model parameters effectively

- **`lr_scheduler = "constant"`** - ✅ RECOMMENDED: Constant learning rate

## 📋 Summary of Changes Made

1. **Added explicit 0 values** instead of comments for numeric fields
2. **Added required model paths** with empty defaults and warnings
3. **Clarified field behavior** with detailed comments
4. **Fixed inconsistent defaults** that could cause confusion
5. **Made all numeric fields explicit** so users understand the effect of 0

## 🔧 **TOML Generation & Command Line Behavior:**

**✅ FIXED: Critical Parameter Handling Issues**
- **0 values**: Now correctly preserved (e.g., `lr_warmup_steps = 0`, `blocks_to_swap = 0`)
- **False values**: Now correctly preserved (e.g., `fp8_base = false`, `split_attn = false`) 
- **Empty strings**: Now correctly preserved (allows user to specify empty paths when needed)
- **None values**: Only None values are excluded from TOML files (as intended)

**Command Line Generation:**
- Boolean flags (`action="store_true"`): Only included when True, omitted when False
- Numeric parameters: All values including 0 are preserved and passed to musubi tuner
- String parameters: Empty strings are preserved, None values are omitted
- Caching commands: Fixed to handle 0 values correctly (e.g., `num_workers = 0` is valid)

The configuration now clearly shows:
- Which fields are required vs optional
- What happens when fields are 0 vs positive values  
- When empty strings are valid vs when paths are required
- Optimal default values with explanations
- **Accurate TOML generation** that preserves all meaningful parameter values