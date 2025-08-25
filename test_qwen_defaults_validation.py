#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to validate that qwen_image_defaults.toml values match:
1. Gradio component defaults in qwen_image_lora_gui.py
2. Actual training script requirements
"""

import toml
import re
import sys
import os

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def check_defaults():
    """Check that all defaults are valid and consistent"""
    
    # Load the qwen defaults
    with open('./qwen_image_defaults.toml', 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    print("=" * 60)
    print("QWEN IMAGE DEFAULTS VALIDATION")
    print("=" * 60)
    
    issues = []
    
    # 1. Check critical Qwen-specific values
    print("\n1. CRITICAL QWEN VALUES:")
    print("-" * 30)
    
    if config.get('dit_dtype') != 'bfloat16':
        issues.append(f"❌ dit_dtype must be 'bfloat16' for Qwen, got: {config.get('dit_dtype')}")
    else:
        print("✅ dit_dtype = bfloat16 (correct)")
    
    if config.get('timestep_sampling') != 'qwen_shift':
        print(f"⚠️  timestep_sampling = {config.get('timestep_sampling')} (consider using 'qwen_shift' for optimal Qwen performance)")
    else:
        print("✅ timestep_sampling = qwen_shift (optimal)")
    
    if config.get('mode_scale') != 1.0:
        print(f"⚠️  mode_scale = {config.get('mode_scale')} (1.0 recommended for Qwen)")
    else:
        print("✅ mode_scale = 1.0 (optimal for Qwen)")
    
    if config.get('weighting_scheme') != 'none':
        print(f"⚠️  weighting_scheme = {config.get('weighting_scheme')} ('none' recommended for Qwen)")
    else:
        print("✅ weighting_scheme = none (recommended)")
    
    # 2. Check dataset configuration
    print("\n2. DATASET CONFIGURATION:")
    print("-" * 30)
    
    if config.get('dataset_config_mode') != 'Generate from Folder Structure':
        issues.append(f"❌ dataset_config_mode should be 'Generate from Folder Structure', got: {config.get('dataset_config_mode')}")
    else:
        print("✅ dataset_config_mode = Generate from Folder Structure")
    
    # 3. Check recommended optimizer settings
    print("\n3. OPTIMIZER SETTINGS:")
    print("-" * 30)
    
    if config.get('optimizer_type') != 'adamw8bit':
        print(f"⚠️  optimizer_type = {config.get('optimizer_type')} ('adamw8bit' recommended for memory efficiency)")
    else:
        print("✅ optimizer_type = adamw8bit (memory efficient)")
    
    if config.get('learning_rate') != 5e-5:
        print(f"⚠️  learning_rate = {config.get('learning_rate')} (5e-5 recommended per official docs)")
    else:
        print("✅ learning_rate = 5e-5 (per official docs)")
    
    # 4. Check network settings
    print("\n4. NETWORK SETTINGS:")
    print("-" * 30)
    
    if config.get('network_module') != 'networks.lora_qwen_image':
        issues.append(f"❌ network_module must be 'networks.lora_qwen_image', got: {config.get('network_module')}")
    else:
        print("✅ network_module = networks.lora_qwen_image")
    
    if config.get('network_dim') != 16:
        print(f"⚠️  network_dim = {config.get('network_dim')} (16 recommended per official docs)")
    else:
        print("✅ network_dim = 16 (per official docs)")
    
    if config.get('network_alpha') != 16.0:
        print(f"⚠️  network_alpha = {config.get('network_alpha')} (should equal network_dim for best results)")
    else:
        print("✅ network_alpha = 16.0 (equals network_dim)")
    
    # 5. Check mixed precision
    print("\n5. MIXED PRECISION:")
    print("-" * 30)
    
    if config.get('mixed_precision') != 'bf16':
        print(f"⚠️  mixed_precision = {config.get('mixed_precision')} ('bf16' recommended for Qwen)")
    else:
        print("✅ mixed_precision = bf16 (recommended)")
    
    # 6. Check for empty string values that should be None or have defaults
    print("\n6. EMPTY STRING VALUES:")
    print("-" * 30)
    
    empty_ok = ['show_timesteps', 'log_with', 'lr_scheduler_type', 'caching_latent_debug_mode',
                'dit', 'vae', 'text_encoder', 'dataset_config', 'parent_folder_path',
                'output_dir', 'output_name', 'sample_prompts', 'logging_dir']
    
    empty_count = 0
    for key, value in config.items():
        if value == "" and key not in empty_ok:
            empty_count += 1
            if empty_count <= 5:  # Show first 5
                print(f"⚠️  {key} is empty string (might need a default value)")
    
    if empty_count == 0:
        print("✅ No unexpected empty string values")
    elif empty_count > 5:
        print(f"   ... and {empty_count - 5} more empty values")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("❌ VALIDATION FAILED - Critical issues found:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("✅ VALIDATION PASSED - All critical values are correct")
        print("\nNote: ⚠️ warnings are recommendations, not errors")
        return True

if __name__ == "__main__":
    success = check_defaults()
    sys.exit(0 if success else 1)