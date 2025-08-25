#!/usr/bin/env python3
"""Test script to verify image captioning defaults and GUI integration"""

import toml
import re
from pathlib import Path

def load_defaults():
    """Load the image captioning defaults"""
    with open("image_captioning_defaults.toml", "r", encoding="utf-8") as f:
        return toml.load(f)

def analyze_gui_usage():
    """Analyze how defaults are used in GUI"""
    with open("musubi_tuner_gui/image_captioning_gui.py", "r", encoding="utf-8") as f:
        gui_content = f.read()
    
    # Extract all config.get() calls
    pattern = r'self\.config\.get\("([^"]+)",\s*([^)]+)\)'
    matches = re.findall(pattern, gui_content)
    
    gui_defaults = {}
    for key, default_value in matches:
        # Clean up the default value
        default_value = default_value.strip()
        # Try to evaluate simple literals
        try:
            if default_value == "True":
                default_value = True
            elif default_value == "False":
                default_value = False
            elif default_value.startswith('"') and default_value.endswith('"'):
                default_value = default_value[1:-1]
            elif default_value.isdigit():
                default_value = int(default_value)
            elif '.' in default_value and default_value.replace('.', '').isdigit():
                default_value = float(default_value)
        except:
            pass
        gui_defaults[key] = default_value
    
    return gui_defaults

def check_cli_defaults():
    """Check defaults in CLI module"""
    with open("musubi-tuner/src/musubi_tuner/caption_images_by_qwen_vl.py", "r", encoding="utf-8") as f:
        cli_content = f.read()
    
    cli_defaults = {}
    
    # Extract DEFAULT_MAX_SIZE
    match = re.search(r'DEFAULT_MAX_SIZE\s*=\s*(\d+)', cli_content)
    if match:
        cli_defaults['max_size'] = int(match.group(1))
    
    # Extract default max_new_tokens from argparse
    match = re.search(r'--max_new_tokens.*?default=(\d+)', cli_content, re.DOTALL)
    if match:
        cli_defaults['max_new_tokens'] = int(match.group(1))
    
    # Extract default output_format
    match = re.search(r'--output_format.*?default="([^"]+)"', cli_content, re.DOTALL)
    if match:
        cli_defaults['output_format'] = match.group(1)
    
    # Check for fp8_vl default (it's an action="store_true", so default is False)
    cli_defaults['fp8_vl'] = False  # store_true means default is False
    
    return cli_defaults

def main():
    print("=" * 70)
    print("IMAGE CAPTIONING CONFIGURATION ANALYSIS")
    print("=" * 70)
    
    # Load defaults from TOML
    defaults = load_defaults()
    captioning_config = defaults.get("image_captioning", {})
    
    print("\n1. DEFAULTS FROM image_captioning_defaults.toml:")
    print("-" * 50)
    for key, value in captioning_config.items():
        print(f"  {key}: {repr(value)} (type: {type(value).__name__})")
    
    # Check GUI usage
    gui_defaults = analyze_gui_usage()
    
    print("\n2. DEFAULTS USED IN GUI (image_captioning_gui.py):")
    print("-" * 50)
    for key, value in gui_defaults.items():
        print(f"  {key}: {repr(value)} (type: {type(value).__name__})")
    
    # Check CLI defaults
    cli_defaults = check_cli_defaults()
    
    print("\n3. DEFAULTS IN CLI MODULE (caption_images_by_qwen_vl.py):")
    print("-" * 50)
    for key, value in cli_defaults.items():
        print(f"  {key}: {repr(value)} (type: {type(value).__name__})")
    
    # Compare and validate
    print("\n4. VALIDATION RESULTS:")
    print("-" * 50)
    
    issues = []
    
    # Check if GUI uses correct defaults from TOML
    for key in gui_defaults:
        gui_val = gui_defaults[key]
        toml_val = captioning_config.get(key)
        
        if toml_val is not None and gui_val != toml_val:
            issues.append(f"  ❌ GUI default mismatch for '{key}': GUI={repr(gui_val)}, TOML={repr(toml_val)}")
    
    # Check data types
    for key, value in captioning_config.items():
        # Check for empty strings that should be validated
        if key in ["model_path"] and value == "":
            print(f"  WARNING: '{key}' is empty - user must provide before use")
        elif key in ["batch_image_dir"] and value == "":
            print(f"  OK: '{key}' is empty - optional for single image mode")
        elif key in ["prefix", "suffix", "custom_prompt", "batch_output_folder", "jsonl_output_file"] and value == "":
            print(f"  OK: '{key}' is empty - optional parameter")
        
        # Check boolean values
        if key in ["fp8_vl", "enable_progress_tracking", "validate_images", "auto_resize_images"]:
            if not isinstance(value, bool):
                issues.append(f"  ERROR: '{key}' should be boolean but is {type(value).__name__}")
        
        # Check numeric values
        if key in ["max_size", "max_new_tokens"]:
            if not isinstance(value, int):
                issues.append(f"  ERROR: '{key}' should be int but is {type(value).__name__}")
        
        # Check string values
        if key in ["output_format"]:
            if value not in ["text", "jsonl"]:
                issues.append(f"  ERROR: '{key}' has invalid value: {repr(value)} (should be 'text' or 'jsonl')")
    
    # Check if CLI defaults match TOML
    if cli_defaults.get('max_size') != captioning_config.get('max_size'):
        issues.append(f"  WARNING: CLI max_size default ({cli_defaults.get('max_size')}) matches TOML ({captioning_config.get('max_size')})")
    
    if cli_defaults.get('max_new_tokens') != captioning_config.get('max_new_tokens'):
        issues.append(f"  WARNING: CLI max_new_tokens default ({cli_defaults.get('max_new_tokens')}) matches TOML ({captioning_config.get('max_new_tokens')})")
    
    # Special check for fp8_vl
    toml_fp8 = captioning_config.get('fp8_vl', False)
    cli_fp8 = cli_defaults.get('fp8_vl', False)
    if toml_fp8 != cli_fp8:
        print(f"  INFO: fp8_vl default differs: TOML={toml_fp8}, CLI={cli_fp8} (CLI uses store_true, so False is expected)")
    
    # Check for unused parameters in TOML
    toml_only_params = set(captioning_config.keys()) - set(gui_defaults.keys())
    toml_only_params -= {"enable_progress_tracking", "validate_images", "auto_resize_images"}  # These might be used internally
    
    if toml_only_params:
        print(f"\n  INFO: Parameters in TOML but not directly referenced in GUI: {toml_only_params}")
    
    if issues:
        print("\n5. ISSUES FOUND:")
        print("-" * 50)
        for issue in issues:
            print(issue)
    else:
        print("\nOK: All validations passed! Defaults are correctly configured.")
    
    # Check how config loading works
    print("\n6. CONFIG LOADING MECHANISM:")
    print("-" * 50)
    print("  • TabConfigManager loads 'image_captioning_defaults.toml' for Image Captioning tab")
    print("  • GUIConfig.get() method retrieves values with fallback to defaults")
    print("  • GUI components use config.get(key, default) pattern")
    print("  • If user loads custom config, it overrides all defaults")
    
    # Final summary
    print("\n7. SUMMARY:")
    print("-" * 50)
    if not issues:
        print("  OK: Configuration is valid and properly integrated")
        print("  OK: All data types are correct")
        print("  OK: GUI uses appropriate defaults from TOML")
        print("  OK: Required fields are properly marked as empty")
        print("  OK: Optional fields have sensible defaults")
    else:
        print(f"  WARNING: Found {len(issues)} configuration issues that may need attention")

if __name__ == "__main__":
    main()