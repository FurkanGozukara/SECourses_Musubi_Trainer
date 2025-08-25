#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EXHAUSTIVE PARAMETER VALIDATION
Performs deep cross-check of every single parameter
"""

import toml
import re
import sys
import os
import subprocess

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def get_all_parameters():
    """Extract ALL parameters from qwen_image_defaults.toml"""
    
    with open('./qwen_image_defaults.toml', 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    return config

def check_training_script_arguments():
    """Check which arguments the training script actually accepts"""
    
    # Get help output from the training script
    try:
        result = subprocess.run(
            ["./venv/Scripts/python.exe", "musubi-tuner/src/musubi_tuner/qwen_image_train_network.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        help_text = result.stdout + result.stderr
        
        # Extract all --argument patterns
        args = re.findall(r'--([a-z_]+)', help_text)
        return set(args)
    except:
        print("Warning: Could not get training script arguments")
        return set()

def check_gui_components():
    """Check which parameters exist in GUI"""
    
    gui_params = {}
    
    # Read the main GUI file
    with open('musubi_tuner_gui/qwen_image_lora_gui.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all self.param = gr.Component patterns
    component_patterns = [
        r'self\.(\w+)\s*=\s*gr\.Number\(',
        r'self\.(\w+)\s*=\s*gr\.Textbox\(',
        r'self\.(\w+)\s*=\s*gr\.Dropdown\(',
        r'self\.(\w+)\s*=\s*gr\.Checkbox\(',
        r'self\.(\w+)\s*=\s*gr\.Slider\(',
        r'self\.(\w+)\s*=\s*gr\.Radio\(',
    ]
    
    for pattern in component_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            gui_params[match] = 'qwen_image_lora_gui'
    
    # Check imported classes
    class_files = [
        'musubi_tuner_gui/class_accelerate_launch.py',
        'musubi_tuner_gui/class_advanced_training.py',
        'musubi_tuner_gui/class_training.py',
        'musubi_tuner_gui/class_latent_caching.py',
        'musubi_tuner_gui/class_text_encoder_outputs_caching.py',
        'musubi_tuner_gui/class_optimizer_and_scheduler.py',
        'musubi_tuner_gui/class_network.py',
        'musubi_tuner_gui/class_save_load.py',
    ]
    
    for file_path in class_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in component_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if match not in gui_params:
                        gui_params[match] = file_path.split('/')[-1]
    
    return gui_params

def perform_exhaustive_check():
    """Perform exhaustive parameter validation"""
    
    print("=" * 80)
    print("EXHAUSTIVE PARAMETER VALIDATION - EVERY SINGLE PARAMETER")
    print("=" * 80)
    
    # Get all data
    config = get_all_parameters()
    training_args = check_training_script_arguments()
    gui_params = check_gui_components()
    
    # Track issues
    critical_issues = []
    warnings = []
    info = []
    
    print(f"\n📊 STATISTICS:")
    print(f"  - Total parameters in config: {len(config)}")
    print(f"  - GUI components found: {len(gui_params)}")
    print(f"  - Training script args found: {len(training_args) if training_args else 'Could not extract'}")
    
    print("\n" + "=" * 80)
    print("CHECKING EVERY SINGLE PARAMETER:")
    print("=" * 80)
    
    # Sort parameters for organized output
    sorted_params = sorted(config.keys())
    
    # Categories for better organization
    categories = {
        'Model Paths': ['dit', 'vae', 'text_encoder', 'text_encoder_dtype', 'vae_dtype', 'dit_dtype'],
        'Dataset': ['dataset_config', 'dataset_config_mode', 'parent_folder_path', 'dataset_resolution_width', 
                   'dataset_resolution_height', 'dataset_caption_extension', 'create_missing_captions',
                   'caption_strategy', 'dataset_batch_size', 'dataset_enable_bucket', 'dataset_bucket_no_upscale',
                   'dataset_cache_directory', 'dataset_control_directory', 'dataset_qwen_image_edit_no_resize_control',
                   'generated_toml_path'],
        'Memory Optimization': ['fp8_vl', 'fp8_base', 'fp8_scaled', 'blocks_to_swap', 'vae_tiling', 
                               'vae_chunk_size', 'vae_spatial_tile_sample_min_size', 'gradient_checkpointing',
                               'img_in_txt_in_offloading'],
        'Training Core': ['learning_rate', 'optimizer_type', 'max_train_steps', 'max_train_epochs', 
                         'gradient_accumulation_steps', 'max_grad_norm', 'seed'],
        'Network (LoRA)': ['network_module', 'network_dim', 'network_alpha', 'network_dropout', 
                          'network_args', 'network_weights', 'no_metadata', 'dim_from_weights',
                          'scale_weight_norms', 'base_weights', 'base_weights_multiplier'],
        'Scheduler': ['lr_scheduler', 'lr_warmup_steps', 'lr_decay_steps', 'lr_scheduler_num_cycles',
                     'lr_scheduler_power', 'lr_scheduler_timescale', 'lr_scheduler_min_lr_ratio',
                     'lr_scheduler_type', 'lr_scheduler_args', 'optimizer_args'],
        'Timestep/Flow': ['timestep_sampling', 'discrete_flow_shift', 'weighting_scheme', 'mode_scale',
                         'logit_mean', 'logit_std', 'sigmoid_scale', 'min_timestep', 'max_timestep',
                         'preserve_distribution_shape', 'num_timestep_buckets', 'show_timesteps'],
        'Attention': ['sdpa', 'flash_attn', 'sage_attn', 'xformers', 'flash3', 'split_attn'],
        'Saving/Loading': ['output_dir', 'output_name', 'resume', 'save_every_n_epochs', 'save_every_n_steps',
                          'save_last_n_epochs', 'save_last_n_epochs_state', 'save_last_n_steps',
                          'save_last_n_steps_state', 'save_state', 'save_state_on_train_end'],
        'Caching': ['caching_latent_device', 'caching_latent_batch_size', 'caching_latent_num_workers',
                   'caching_latent_skip_existing', 'caching_latent_keep_cache', 'caching_latent_debug_mode',
                   'caching_latent_console_width', 'caching_latent_console_back', 'caching_latent_console_num_images',
                   'caching_teo_text_encoder', 'caching_teo_device', 'caching_teo_fp8_vl', 'caching_teo_batch_size',
                   'caching_teo_num_workers', 'caching_teo_skip_existing', 'caching_teo_keep_cache'],
        'Accelerate': ['mixed_precision', 'dynamo_backend', 'dynamo_mode', 'dynamo_fullgraph', 'dynamo_dynamic',
                      'multi_gpu', 'gpu_ids', 'num_processes', 'num_machines', 'num_cpu_threads_per_process',
                      'main_process_port', 'extra_accelerate_launch_args', 'ddp_timeout', 'ddp_gradient_as_bucket_view',
                      'ddp_static_graph'],
        'Logging': ['logging_dir', 'log_with', 'log_prefix', 'log_tracker_name', 'wandb_run_name',
                   'log_tracker_config', 'wandb_api_key', 'log_config'],
        'Sampling': ['sample_every_n_steps', 'sample_every_n_epochs', 'sample_at_first', 'sample_prompts'],
        'Metadata': ['metadata_author', 'metadata_description', 'metadata_license', 'metadata_tags', 
                    'metadata_title', 'training_comment'],
        'HuggingFace': ['huggingface_repo_id', 'huggingface_token', 'huggingface_repo_type',
                       'huggingface_repo_visibility', 'huggingface_path_in_repo', 'save_state_to_huggingface',
                       'resume_from_huggingface', 'async_upload'],
        'Other': ['guidance_scale', 'edit', 'max_data_loader_n_workers', 'persistent_data_loader_workers']
    }
    
    # Check each category
    for category, params in categories.items():
        print(f"\n{'='*60}")
        print(f"📁 {category}")
        print(f"{'='*60}")
        
        for param in params:
            if param in config:
                value = config[param]
                value_type = type(value).__name__
                
                # Check if in GUI
                gui_status = "✅" if param in gui_params else "❌"
                gui_location = gui_params.get(param, "NOT FOUND")
                
                # Check if in training args (if we have them)
                train_status = "?"
                if training_args:
                    train_status = "✅" if param.replace('_', '-') in training_args else "⚠️"
                
                # Value validation
                value_check = check_value_validity(param, value)
                
                # Format value for display
                if isinstance(value, str):
                    display_value = f'"{value}"' if value else '""(empty)'
                elif isinstance(value, bool):
                    display_value = str(value)
                elif isinstance(value, (int, float)):
                    display_value = str(value)
                elif isinstance(value, list):
                    display_value = f"[{len(value)} items]" if value else "[]"
                else:
                    display_value = str(value)
                
                print(f"\n  📌 {param}:")
                print(f"     Value: {display_value} ({value_type})")
                print(f"     GUI: {gui_status} {gui_location if gui_status == '✅' else ''}")
                print(f"     Training Script: {train_status}")
                if value_check != "OK":
                    print(f"     ⚠️ {value_check}")
                    if "CRITICAL" in value_check:
                        critical_issues.append(f"{param}: {value_check}")
                    else:
                        warnings.append(f"{param}: {value_check}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if critical_issues:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            print(f"   - {issue}")
    else:
        print("\n✅ NO CRITICAL ISSUES")
    
    if warnings:
        print("\n⚠️ WARNINGS:")
        for warning in warnings[:10]:  # Show first 10
            print(f"   - {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings)-10} more warnings")
    
    # Special checks
    print("\n" + "=" * 80)
    print("SPECIAL VALIDATION CHECKS")
    print("=" * 80)
    
    # Check Qwen-specific requirements
    print("\n🔍 Qwen-Specific Requirements:")
    if config.get('dit_dtype') == 'bfloat16':
        print("   ✅ dit_dtype = bfloat16 (CORRECT - Required for Qwen)")
    else:
        print(f"   ❌ dit_dtype = {config.get('dit_dtype')} (MUST be bfloat16)")
        critical_issues.append("dit_dtype must be bfloat16")
    
    if config.get('network_module') == 'networks.lora_qwen_image':
        print("   ✅ network_module = networks.lora_qwen_image (CORRECT)")
    else:
        print(f"   ❌ network_module = {config.get('network_module')} (MUST be networks.lora_qwen_image)")
        critical_issues.append("network_module must be networks.lora_qwen_image")
    
    # Check recommended values
    print("\n🎯 Recommended Values Check:")
    recommendations = {
        'learning_rate': (5e-5, "5e-5 per official docs"),
        'network_dim': (16, "16 per official docs"),
        'network_alpha': (16.0, "Should equal network_dim"),
        'optimizer_type': ('adamw8bit', "Memory efficient"),
        'discrete_flow_shift': (2.2, "Optimal for Qwen"),
        'timestep_sampling': ('qwen_shift', "Dynamic resolution-aware"),
        'weighting_scheme': ('none', "Recommended for Qwen"),
        'mode_scale': (1.0, "Optimized for Qwen"),
        'mixed_precision': ('bf16', "Recommended for Qwen"),
    }
    
    for param, (recommended, reason) in recommendations.items():
        actual = config.get(param)
        if actual == recommended:
            print(f"   ✅ {param} = {actual} ({reason})")
        else:
            print(f"   ⚠️ {param} = {actual} (Recommended: {recommended} - {reason})")
    
    return len(critical_issues) == 0

def check_value_validity(param, value):
    """Check if a parameter value is valid"""
    
    # Type checks
    expected_types = {
        # Numbers
        'learning_rate': (float, 1e-7, 1e-2),
        'network_dim': (int, 1, 256),
        'network_alpha': (float, 0.1, 256),
        'network_dropout': (float, 0, 1),
        'max_train_steps': (int, 1, 1000000),
        'max_train_epochs': (int, 1, 1000),
        'gradient_accumulation_steps': (int, 1, 128),
        'max_grad_norm': (float, 0, 10),
        'seed': (int, 0, 2**32),
        'blocks_to_swap': (int, 0, 45),
        'vae_chunk_size': (int, 0, 128),
        'mode_scale': (float, 0.1, 5.0),
        'discrete_flow_shift': (float, 0.1, 20.0),
        'sigmoid_scale': (float, 0.1, 10.0),
        'min_timestep': (int, 0, 999),
        'max_timestep': (int, 1, 1000),
        # Strings that must have specific values
        'dit_dtype': (str, ['bfloat16']),
        'network_module': (str, ['networks.lora_qwen_image']),
        'mixed_precision': (str, ['no', 'fp16', 'bf16']),
        'optimizer_type': (str, ['adamw8bit', 'AdamW', 'AdaFactor', 'Adam', 'SGDNesterov', 'Lion', 'Prodigy']),
        'lr_scheduler': (str, ['constant', 'linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant_with_warmup', 'adafactor']),
        'timestep_sampling': (str, ['shift', 'qwen_shift', 'sigma', 'uniform', 'sigmoid', 'flux_shift', 'logsnr']),
        'weighting_scheme': (str, ['none', 'logit_normal', 'mode', 'cosmap', 'sigma_sqrt']),
    }
    
    if param in expected_types:
        expected = expected_types[param]
        if len(expected) == 3:  # Numeric with range
            expected_type, min_val, max_val = expected
            if not isinstance(value, expected_type):
                return f"Type mismatch: expected {expected_type.__name__}, got {type(value).__name__}"
            if value < min_val or value > max_val:
                return f"Value out of range: {value} not in [{min_val}, {max_val}]"
        elif len(expected) == 2:  # String with choices
            expected_type, choices = expected
            if not isinstance(value, expected_type):
                return f"Type mismatch: expected {expected_type.__name__}, got {type(value).__name__}"
            if value and value not in choices:
                if param == 'dit_dtype':
                    return f"CRITICAL: Must be 'bfloat16' for Qwen, got '{value}'"
                elif param == 'network_module':
                    return f"CRITICAL: Must be 'networks.lora_qwen_image', got '{value}'"
                else:
                    return f"Value not in valid choices: '{value}'"
    
    # Check for empty required fields
    required_fields = ['network_module', 'dit_dtype']
    if param in required_fields and not value:
        return f"CRITICAL: Required field is empty"
    
    return "OK"

if __name__ == "__main__":
    success = perform_exhaustive_check()
    sys.exit(0 if success else 1)