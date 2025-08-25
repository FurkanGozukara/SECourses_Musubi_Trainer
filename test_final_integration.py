#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final integration test to verify all values work correctly
"""

import toml
import sys
import os

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def test_integration():
    """Test that all config values integrate properly with training code"""
    
    print("=" * 60)
    print("FINAL INTEGRATION TEST")
    print("=" * 60)
    
    # Load config
    with open('./qwen_image_defaults.toml', 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # Test critical integrations
    issues = []
    warnings = []
    
    print("\n1. CHECKING TRAINING SCRIPT COMPATIBILITY:")
    print("-" * 40)
    
    # Check timestep_sampling
    valid_timestep_sampling = ["shift", "qwen_shift", "sigma", "uniform", "sigmoid", 
                              "flux_shift", "logsnr", "qinglong_flux", "qinglong_qwen"]
    if config.get('timestep_sampling') in valid_timestep_sampling:
        print(f"✅ timestep_sampling = '{config.get('timestep_sampling')}' is valid")
    else:
        issues.append(f"timestep_sampling = '{config.get('timestep_sampling')}' not in valid choices")
    
    # Check weighting_scheme
    valid_weighting = ["none", "logit_normal", "mode", "cosmap", "sigma_sqrt"]
    if config.get('weighting_scheme') in valid_weighting:
        print(f"✅ weighting_scheme = '{config.get('weighting_scheme')}' is valid")
    else:
        issues.append(f"weighting_scheme = '{config.get('weighting_scheme')}' not in valid choices")
    
    # Check mode_scale range
    mode_scale = config.get('mode_scale', 1.29)
    if 0.1 <= mode_scale <= 5.0:
        print(f"✅ mode_scale = {mode_scale} is in valid range [0.1, 5.0]")
    else:
        issues.append(f"mode_scale = {mode_scale} out of range")
    
    print("\n2. CHECKING QWEN-SPECIFIC REQUIREMENTS:")
    print("-" * 40)
    
    # Check required Qwen fields
    if config.get('network_module') == 'networks.lora_qwen_image':
        print("✅ network_module correctly set for Qwen")
    else:
        issues.append(f"network_module must be 'networks.lora_qwen_image'")
    
    if config.get('dit_dtype') == 'bfloat16':
        print("✅ dit_dtype = bfloat16 (required for Qwen)")
    else:
        issues.append("dit_dtype must be 'bfloat16' for Qwen")
    
    print("\n3. CHECKING MEMORY OPTIMIZATION SETTINGS:")
    print("-" * 40)
    
    # Check FP8 settings consistency
    if config.get('fp8_base', False):
        if config.get('fp8_scaled', False):
            print("✅ fp8_scaled enabled with fp8_base (good)")
        else:
            warnings.append("fp8_base=true but fp8_scaled=false (should both be true)")
    else:
        print("✅ FP8 disabled (standard precision)")
    
    # Check blocks_to_swap
    blocks = config.get('blocks_to_swap', 0)
    if blocks == 0:
        print("✅ blocks_to_swap = 0 (no CPU offloading)")
    elif 1 <= blocks <= 45:
        print(f"✅ blocks_to_swap = {blocks} (will save ~{blocks * 0.7:.1f}GB VRAM)")
    else:
        warnings.append(f"blocks_to_swap = {blocks} (unusual value)")
    
    print("\n4. CHECKING DATASET CONFIGURATION:")
    print("-" * 40)
    
    if config.get('dataset_config_mode') == 'Generate from Folder Structure':
        print("✅ Using folder structure mode (user-friendly)")
        if not config.get('parent_folder_path'):
            warnings.append("parent_folder_path empty - user must set this")
    elif config.get('dataset_config_mode') == 'Use TOML File':
        if not config.get('dataset_config'):
            warnings.append("dataset_config empty - user must provide TOML file")
    
    print("\n5. CHECKING OPTIMIZER SETTINGS:")
    print("-" * 40)
    
    valid_optimizers = ["adamw8bit", "AdamW", "AdaFactor", "Adam", "SGDNesterov", 
                       "SGDNesterov8bit", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad",
                       "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "Lion", "Lion8bit",
                       "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit", "PagedLion8bit",
                       "Prodigy", "AdaEMAMix8bit", "StableAdamW", "Ranger", "Ademamix",
                       "Ademamix8bit", "AdEMAMix", "schedulefree", "schedulefreeadamw",
                       "schedulefreesgd"]
    
    opt = config.get('optimizer_type', 'adamw8bit')
    if opt.lower() in [o.lower() for o in valid_optimizers]:
        print(f"✅ optimizer_type = '{opt}' is valid")
    else:
        issues.append(f"optimizer_type = '{opt}' not recognized")
    
    # Check learning rate
    lr = config.get('learning_rate', 5e-5)
    if 1e-7 <= lr <= 1e-2:
        print(f"✅ learning_rate = {lr} is in reasonable range")
    else:
        warnings.append(f"learning_rate = {lr} is unusual")
    
    print("\n6. CHECKING ATTENTION SETTINGS:")
    print("-" * 40)
    
    attn_settings = {
        'sdpa': config.get('sdpa', True),
        'flash_attn': config.get('flash_attn', False),
        'sage_attn': config.get('sage_attn', False),
        'xformers': config.get('xformers', False),
        'split_attn': config.get('split_attn', False)
    }
    
    enabled = [k for k, v in attn_settings.items() if v]
    if len(enabled) == 1 and enabled[0] == 'sdpa':
        print("✅ Using SDPA (recommended default)")
    elif len(enabled) > 1:
        if 'split_attn' in enabled:
            print(f"✅ Using {enabled} (split_attn correctly enabled)")
        else:
            warnings.append(f"Multiple attention methods enabled but split_attn=False")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("❌ INTEGRATION FAILED - Critical issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ INTEGRATION TEST PASSED")
        
    if warnings:
        print("\n⚠️  Warnings (non-critical):")
        for warning in warnings:
            print(f"   - {warning}")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)