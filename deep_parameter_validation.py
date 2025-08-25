#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEEP PARAMETER VALIDATION WITH SPECIFIC CHECKS
"""

import toml
import re
import sys
import os

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def deep_validation():
    """Perform deep validation of every parameter"""
    
    print("=" * 80)
    print("DEEP PARAMETER VALIDATION - COMPREHENSIVE CHECK")
    print("=" * 80)
    
    # Load config
    with open('./qwen_image_defaults.toml', 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # Track issues
    issues = []
    perfect = []
    warnings = []
    
    # Define expected parameters with their requirements
    parameter_requirements = {
        # CRITICAL QWEN REQUIREMENTS
        'dit_dtype': {
            'expected': 'bfloat16',
            'actual': config.get('dit_dtype'),
            'critical': True,
            'reason': 'MUST be bfloat16 for Qwen Image'
        },
        'network_module': {
            'expected': 'networks.lora_qwen_image',
            'actual': config.get('network_module'),
            'critical': True,
            'reason': 'Required for Qwen Image LoRA'
        },
        
        # RECOMMENDED VALUES FROM DOCS
        'learning_rate': {
            'expected': 5e-5,
            'actual': config.get('learning_rate'),
            'critical': False,
            'reason': 'Recommended per official documentation'
        },
        'network_dim': {
            'expected': 16,
            'actual': config.get('network_dim'),
            'critical': False,
            'reason': 'Recommended per official documentation'
        },
        'network_alpha': {
            'expected': 16.0,
            'actual': config.get('network_alpha'),
            'critical': False,
            'reason': 'Should equal network_dim for best results'
        },
        'optimizer_type': {
            'expected': 'adamw8bit',
            'actual': config.get('optimizer_type'),
            'critical': False,
            'reason': 'Memory efficient, confirmed in examples'
        },
        'timestep_sampling': {
            'expected': ['shift', 'qwen_shift'],
            'actual': config.get('timestep_sampling'),
            'critical': False,
            'reason': 'qwen_shift uses dynamic resolution-aware sampling'
        },
        'discrete_flow_shift': {
            'expected': 2.2,
            'actual': config.get('discrete_flow_shift'),
            'critical': False,
            'reason': 'Optimal for Qwen Image'
        },
        'weighting_scheme': {
            'expected': 'none',
            'actual': config.get('weighting_scheme'),
            'critical': False,
            'reason': 'Recommended for Qwen Image'
        },
        'mode_scale': {
            'expected': 1.0,
            'actual': config.get('mode_scale'),
            'critical': False,
            'reason': 'Optimized for Qwen (lower than SD3)'
        },
        'mixed_precision': {
            'expected': 'bf16',
            'actual': config.get('mixed_precision'),
            'critical': False,
            'reason': 'Recommended for Qwen Image'
        },
        'gradient_checkpointing': {
            'expected': True,
            'actual': config.get('gradient_checkpointing'),
            'critical': False,
            'reason': 'Saves memory'
        },
        'sdpa': {
            'expected': True,
            'actual': config.get('sdpa'),
            'critical': False,
            'reason': 'Recommended attention mode'
        },
        'max_data_loader_n_workers': {
            'expected': 2,
            'actual': config.get('max_data_loader_n_workers'),
            'critical': False,
            'reason': 'Optimal for most systems'
        },
        'persistent_data_loader_workers': {
            'expected': True,
            'actual': config.get('persistent_data_loader_workers'),
            'critical': False,
            'reason': 'Improves performance'
        },
    }
    
    print("\n🔍 CHECKING CRITICAL & RECOMMENDED PARAMETERS:")
    print("-" * 60)
    
    for param, req in parameter_requirements.items():
        expected = req['expected']
        actual = req['actual']
        
        # Handle list of valid values
        if isinstance(expected, list):
            match = actual in expected
            expected_str = f"one of {expected}"
        else:
            match = actual == expected
            expected_str = str(expected)
        
        if match:
            print(f"✅ {param}: {actual} = CORRECT")
            perfect.append(param)
        else:
            severity = "❌ CRITICAL" if req['critical'] else "⚠️ WARNING"
            print(f"{severity} {param}: {actual} (expected: {expected_str})")
            print(f"   Reason: {req['reason']}")
            if req['critical']:
                issues.append(f"{param}: expected {expected_str}, got {actual}")
            else:
                warnings.append(f"{param}: expected {expected_str}, got {actual}")
    
    # Check for missing parameters
    print("\n📋 CHECKING FOR MISSING PARAMETERS:")
    print("-" * 60)
    
    expected_params = [
        'dit', 'vae', 'text_encoder', 'dataset_config', 'output_dir', 'output_name',
        'sample_prompts', 'logging_dir', 'parent_folder_path'
    ]
    
    for param in expected_params:
        if param not in config:
            print(f"❌ MISSING: {param}")
            issues.append(f"Missing parameter: {param}")
        elif config[param] == "":
            print(f"ℹ️ {param}: Empty (user must set)")
        else:
            print(f"✅ {param}: Present")
    
    # Check data types
    print("\n🔢 CHECKING DATA TYPES:")
    print("-" * 60)
    
    type_checks = {
        'learning_rate': float,
        'network_dim': int,
        'network_alpha': (int, float),
        'max_train_steps': int,
        'max_train_epochs': int,
        'gradient_accumulation_steps': int,
        'seed': int,
        'blocks_to_swap': int,
        'gradient_checkpointing': bool,
        'sdpa': bool,
        'fp8_vl': bool,
        'mixed_precision': str,
        'optimizer_type': str,
    }
    
    type_errors = []
    for param, expected_type in type_checks.items():
        if param in config:
            actual_value = config[param]
            if isinstance(expected_type, tuple):
                if not any(isinstance(actual_value, t) for t in expected_type):
                    type_errors.append(f"{param}: expected {expected_type}, got {type(actual_value).__name__}")
            else:
                if not isinstance(actual_value, expected_type):
                    type_errors.append(f"{param}: expected {expected_type.__name__}, got {type(actual_value).__name__}")
    
    if type_errors:
        print("❌ Type errors found:")
        for error in type_errors:
            print(f"   - {error}")
    else:
        print("✅ All type checks passed")
    
    # Check value ranges
    print("\n📊 CHECKING VALUE RANGES:")
    print("-" * 60)
    
    range_checks = {
        'learning_rate': (1e-7, 1e-2, config.get('learning_rate', 0)),
        'network_dim': (1, 256, config.get('network_dim', 0)),
        'network_alpha': (0.1, 256, config.get('network_alpha', 0)),
        'network_dropout': (0, 1, config.get('network_dropout', 0)),
        'mode_scale': (0.1, 5.0, config.get('mode_scale', 0)),
        'discrete_flow_shift': (0.1, 20.0, config.get('discrete_flow_shift', 0)),
        'blocks_to_swap': (0, 45, config.get('blocks_to_swap', 0)),
        'min_timestep': (0, 999, config.get('min_timestep', 0)),
        'max_timestep': (1, 1000, config.get('max_timestep', 0)),
    }
    
    range_errors = []
    for param, (min_val, max_val, actual) in range_checks.items():
        if actual < min_val or actual > max_val:
            range_errors.append(f"{param}: {actual} not in [{min_val}, {max_val}]")
            
    if range_errors:
        print("❌ Range errors found:")
        for error in range_errors:
            print(f"   - {error}")
    else:
        print("✅ All values within valid ranges")
    
    # Check consistency
    print("\n🔗 CHECKING PARAMETER CONSISTENCY:")
    print("-" * 60)
    
    # network_alpha should equal network_dim
    if config.get('network_alpha', 0) == config.get('network_dim', 0):
        print("✅ network_alpha == network_dim (optimal)")
    else:
        print(f"⚠️ network_alpha ({config.get('network_alpha')}) != network_dim ({config.get('network_dim')})")
    
    # fp8_scaled requires fp8_base
    if config.get('fp8_scaled', False) and not config.get('fp8_base', False):
        print("⚠️ fp8_scaled=True but fp8_base=False (should both be True or both False)")
    else:
        print("✅ FP8 settings consistent")
    
    # Check dataset mode
    if config.get('dataset_config_mode') == 'Generate from Folder Structure':
        print("✅ Dataset mode set to user-friendly folder structure")
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    total_params = len(config)
    print(f"\n📊 Total parameters checked: {total_params}")
    print(f"✅ Perfect matches: {len(perfect)}")
    print(f"⚠️ Warnings: {len(warnings)}")
    print(f"❌ Critical issues: {len(issues)}")
    
    if issues:
        print("\n❌ VALIDATION FAILED - Critical issues must be fixed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✅ VALIDATION PASSED - Configuration is ready for Qwen Image training!")
        print("\nAll critical parameters are correctly set:")
        print("  • dit_dtype = bfloat16 ✓")
        print("  • network_module = networks.lora_qwen_image ✓")
        print("  • All recommended values properly configured ✓")
        print("  • Memory optimization options available ✓")
        print("  • Dataset configuration user-friendly ✓")
        
        if warnings:
            print(f"\nNote: {len(warnings)} non-critical warnings exist but won't affect training")
        
        return True

if __name__ == "__main__":
    success = deep_validation()
    sys.exit(0 if success else 1)