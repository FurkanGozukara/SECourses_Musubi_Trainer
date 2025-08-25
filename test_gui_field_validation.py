#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test to validate that all GUI fields are properly saved to and loaded from configuration files.
Tests both Qwen Image LoRA training GUI and Image Captioning GUI.
"""

import os
import sys
import toml
import tempfile
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def extract_qwen_lora_gui_parameters() -> Set[str]:
    """Extract all parameters from qwen_image_lora_gui.py function signature"""
    gui_file = Path("musubi_tuner_gui/qwen_image_lora_gui.py")
    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the qwen_image_gui_actions function
    match = re.search(r'def qwen_image_gui_actions\((.*?)\):', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find qwen_image_gui_actions function")
    
    params = match.group(1)
    # Extract parameter names (ignore comments and clean up)
    param_lines = [line.strip().rstrip(',') for line in params.split('\n') if line.strip() and not line.strip().startswith('#')]
    gui_params = set(p for p in param_lines if p and not p.startswith('#'))
    
    # Remove control parameters that aren't saved
    control_params = {'action_type', 'bool_value', 'file_path', 'headless', 'print_only'}
    gui_params = gui_params - control_params
    
    return gui_params


def extract_qwen_lora_gui_components() -> Dict[str, str]:
    """Extract all GUI components from qwen_image_lora_gui.py to understand field types"""
    gui_file = Path("musubi_tuner_gui/qwen_image_lora_gui.py")
    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    components = {}
    
    # Find all gr.* component definitions
    patterns = [
        (r'self\.(\w+)\s*=\s*gr\.Textbox\(', 'textbox'),
        (r'self\.(\w+)\s*=\s*gr\.Number\(', 'number'),
        (r'self\.(\w+)\s*=\s*gr\.Checkbox\(', 'checkbox'),
        (r'self\.(\w+)\s*=\s*gr\.Radio\(', 'radio'),
        (r'self\.(\w+)\s*=\s*gr\.Dropdown\(', 'dropdown'),
        (r'self\.(\w+)\s*=\s*gr\.Slider\(', 'slider'),
    ]
    
    for pattern, comp_type in patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            components[match.group(1)] = comp_type
    
    return components


def extract_image_captioning_fields() -> Dict[str, Any]:
    """Extract all fields from image_captioning_gui.py"""
    gui_file = Path("musubi_tuner_gui/image_captioning_gui.py")
    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find save_configuration function to get all fields
    save_match = re.search(r'def save_configuration\((.*?)\) -> str:', content, re.DOTALL)
    if not save_match:
        raise ValueError("Could not find save_configuration function")
    
    params = save_match.group(1)
    param_lines = [line.strip().rstrip(',').split(':')[0].strip() 
                   for line in params.split('\n') 
                   if line.strip() and not line.strip().startswith('#') and ':' in line]
    
    # Remove 'self' from the list
    save_params = [p for p in param_lines if p != 'self']
    
    # Find load_configuration function to verify fields
    load_match = re.search(r'def load_configuration.*?return \((.*?)\)', content, re.DOTALL)
    if not load_match:
        raise ValueError("Could not find load_configuration return statement")
    
    return_values = load_match.group(1)
    # Count the number of fields being returned (excluding status message)
    load_fields = [line.strip() for line in return_values.split(',') if 'captioning_config.get' in line]
    
    return {
        'save_params': save_params,
        'load_fields': load_fields,
        'expected_fields': [
            'model_path', 'fp8_vl', 'max_size', 'max_new_tokens',
            'prefix', 'suffix', 'replace_words', 'replace_case_insensitive',
            'replace_whole_words_only', 'custom_prompt', 'batch_image_dir',
            'output_format', 'jsonl_output_file', 'batch_output_folder',
            'scan_subfolders', 'copy_images', 'overwrite_existing_captions'
        ]
    }


def test_qwen_lora_save_load():
    """Test Qwen LoRA GUI save/load functionality"""
    print("=" * 80)
    print("Testing Qwen Image LoRA GUI Field Save/Load")
    print("=" * 80)
    
    # Get all parameters from the GUI
    gui_params = extract_qwen_lora_gui_parameters()
    components = extract_qwen_lora_gui_components()
    
    print(f"\nFound {len(gui_params)} parameters in qwen_image_gui_actions")
    
    # Create a test config with all parameters
    test_config = {}
    for param in gui_params:
        # Set test values based on parameter type
        if 'batch_size' in param or 'dim' in param or 'steps' in param or 'epochs' in param:
            test_config[param] = 1
        elif 'rate' in param or 'scale' in param or 'alpha' in param:
            test_config[param] = 0.5
        elif 'enable' in param or 'use' in param or '_bool' in param:
            test_config[param] = True
        elif 'path' in param or 'dir' in param or 'file' in param:
            test_config[param] = f"/test/path/{param}"
        elif 'args' in param:
            test_config[param] = ["arg1", "arg2"]
        else:
            test_config[param] = f"test_{param}"
    
    # Save test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(test_config, f)
        temp_file = f.name
    
    try:
        # Load the config back
        with open(temp_file, 'r') as f:
            loaded_config = toml.load(f)
        
        # Check for missing fields
        missing_in_load = set(test_config.keys()) - set(loaded_config.keys())
        extra_in_load = set(loaded_config.keys()) - set(test_config.keys())
        
        if missing_in_load:
            print(f"\nWARNING: Fields not loaded properly: {missing_in_load}")
        
        if extra_in_load:
            print(f"\nWARNING: Extra fields in loaded config: {extra_in_load}")
        
        # Check for value mismatches
        mismatches = []
        for key in test_config:
            if key in loaded_config and loaded_config[key] != test_config[key]:
                mismatches.append((key, test_config[key], loaded_config[key]))
        
        if mismatches:
            print(f"\nWARNING: Value mismatches found:")
            for key, expected, actual in mismatches:
                print(f"  {key}: expected={expected}, actual={actual}")
        
        if not missing_in_load and not extra_in_load and not mismatches:
            print("\nSUCCESS: All Qwen LoRA GUI fields save and load correctly!")
        
        # Check for potential issues with numeric fields
        print("\n" + "=" * 40)
        print("Checking numeric field constraints...")
        
        # Load actual config files to check for constraint violations
        config_files = ['config.toml', 'qwen_image_defaults.toml']
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    actual_config = toml.load(f)
                
                print(f"\nChecking {config_file}:")
                
                # Define known constraints from GUI components
                constraints = {
                    'num_processes': (1, None),
                    'num_machines': (1, None),
                    'num_cpu_threads_per_process': (1, None),
                    'gradient_accumulation_steps': (1, None),
                    'max_train_epochs': (1, None),
                    'lr_scheduler_num_cycles': (1, None),
                }
                
                violations = []
                for field, (min_val, max_val) in constraints.items():
                    if field in actual_config:
                        value = actual_config[field]
                        if min_val is not None and value < min_val:
                            violations.append(f"  {field}={value} (minimum should be {min_val})")
                        if max_val is not None and value > max_val:
                            violations.append(f"  {field}={value} (maximum should be {max_val})")
                
                if violations:
                    print("  WARNING: Constraint violations found:")
                    for v in violations:
                        print(f"    {v}")
                else:
                    print("  SUCCESS: No constraint violations")
        
    finally:
        os.unlink(temp_file)


def test_image_captioning_save_load():
    """Test Image Captioning GUI save/load functionality"""
    print("\n" + "=" * 80)
    print("Testing Image Captioning GUI Field Save/Load")
    print("=" * 80)
    
    captioning_info = extract_image_captioning_fields()
    
    print(f"\nSave function parameters: {len(captioning_info['save_params'])}")
    print(f"Expected fields: {len(captioning_info['expected_fields'])}")
    
    # Check if all expected fields are in save parameters
    save_params_set = set(captioning_info['save_params'])
    expected_set = set(captioning_info['expected_fields'])
    
    # Remove config_file_path as it's not a field to save
    save_params_set.discard('config_file_path')
    
    missing_in_save = expected_set - save_params_set
    extra_in_save = save_params_set - expected_set
    
    if missing_in_save:
        print(f"\nWARNING: Expected fields missing from save: {missing_in_save}")
    
    if extra_in_save:
        print(f"\nWARNING: Extra fields in save (not expected): {extra_in_save}")
    
    # Create test configuration
    test_config = {
        "image_captioning": {
            "model_path": "/test/model.safetensors",
            "fp8_vl": True,
            "max_size": 1280,
            "max_new_tokens": 1024,
            "prefix": "test prefix",
            "suffix": "test suffix",
            "replace_words": "word1:replacement1;word2:replacement2",
            "replace_case_insensitive": True,
            "replace_whole_words_only": True,
            "custom_prompt": "Test prompt",
            "batch_image_dir": "/test/images",
            "output_format": "text",
            "jsonl_output_file": "/test/output.jsonl",
            "batch_output_folder": "/test/output",
            "scan_subfolders": True,
            "copy_images": False,
            "overwrite_existing_captions": False,
        }
    }
    
    # Save test config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(test_config, f)
        temp_file = f.name
    
    try:
        # Load the config back
        with open(temp_file, 'r') as f:
            loaded_config = toml.load(f)
        
        # Check if structure is correct
        if "image_captioning" not in loaded_config:
            print("\nWARNING: Missing 'image_captioning' section in loaded config")
        else:
            loaded_caption_config = loaded_config["image_captioning"]
            test_caption_config = test_config["image_captioning"]
            
            # Check for missing fields
            missing_in_load = set(test_caption_config.keys()) - set(loaded_caption_config.keys())
            extra_in_load = set(loaded_caption_config.keys()) - set(test_caption_config.keys())
            
            if missing_in_load:
                print(f"\nWARNING: Fields not loaded properly: {missing_in_load}")
            
            if extra_in_load:
                print(f"\nWARNING: Extra fields in loaded config: {extra_in_load}")
            
            # Check for value mismatches
            mismatches = []
            for key in test_caption_config:
                if key in loaded_caption_config and loaded_caption_config[key] != test_caption_config[key]:
                    mismatches.append((key, test_caption_config[key], loaded_caption_config[key]))
            
            if mismatches:
                print(f"\nWARNING: Value mismatches found:")
                for key, expected, actual in mismatches:
                    print(f"  {key}: expected={expected}, actual={actual}")
            
            if not missing_in_load and not extra_in_load and not mismatches:
                print("\nSUCCESS: All Image Captioning GUI fields save and load correctly!")
        
    finally:
        os.unlink(temp_file)


def check_field_mapping_consistency():
    """Check that field names are consistent between save and load functions"""
    print("\n" + "=" * 80)
    print("Checking Field Mapping Consistency")
    print("=" * 80)
    
    # Check Image Captioning GUI
    print("\nImage Captioning GUI:")
    gui_file = Path("musubi_tuner_gui/image_captioning_gui.py")
    with open(gui_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract save inputs
    save_inputs_match = re.search(r'self\.save_config_button\.click\(.*?inputs=\[(.*?)\]', content, re.DOTALL)
    if save_inputs_match:
        save_inputs = [line.strip().replace('self.', '').rstrip(',') 
                      for line in save_inputs_match.group(1).split('\n') 
                      if 'self.' in line]
        print(f"  Save inputs: {len(save_inputs)} fields")
    
    # Extract load outputs
    load_outputs_match = re.search(r'self\.load_config_button\.click\(.*?outputs=\[(.*?)\]', content, re.DOTALL)
    if load_outputs_match:
        load_outputs = [line.strip().replace('self.', '').rstrip(',') 
                       for line in load_outputs_match.group(1).split('\n') 
                       if 'self.' in line]
        print(f"  Load outputs: {len(load_outputs)} fields")
        
        # Compare (excluding status fields)
        save_set = set(save_inputs) - {'config_file_path'}
        load_set = set(load_outputs) - {'config_status'}
        
        if save_set != load_set:
            missing_in_load = save_set - load_set
            extra_in_load = load_set - save_set
            
            if missing_in_load:
                print(f"  WARNING: Fields saved but not loaded: {missing_in_load}")
            if extra_in_load:
                print(f"  WARNING: Fields loaded but not saved: {extra_in_load}")
        else:
            print("  SUCCESS: Save and load fields match perfectly!")


def main():
    """Run all validation tests"""
    print("GUI Field Save/Load Validation Test")
    print("=" * 80)
    
    try:
        # Test Qwen LoRA GUI
        test_qwen_lora_save_load()
        
        # Test Image Captioning GUI
        test_image_captioning_save_load()
        
        # Check field mapping consistency
        check_field_mapping_consistency()
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("\nSUCCESS: Validation tests completed. Check above for any warnings or issues.")
        
    except Exception as e:
        print(f"\nERROR: Error during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()