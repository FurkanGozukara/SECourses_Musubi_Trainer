"""
Test script to verify the comprehensive fix for empty string path handling.
This script simulates saving a config file with empty path parameters.
"""

import sys
import os
sys.path.insert(0, 'E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer')
from musubi_tuner_gui.common_gui import SaveConfigFile, SaveConfigFileToRun
import toml

# Test parameters with various empty strings that should be excluded
test_params = [
    # Empty path parameters that should be excluded
    ("network_weights", ""),
    ("base_weights", ""),
    ("dit", "E:/SECourses_Improved_Trainer_v1/Qwen_Image_Training_Models/qwen_image_bf16.safetensors"),  # Valid path
    ("vae", "E:/SECourses_Improved_Trainer_v1/Qwen_Image_Training_Models/qwen_train_vae.safetensors"),  # Valid path
    ("text_encoder", ""),  # Empty, should be excluded
    ("sample_prompts", ""),  # Empty, should be excluded
    
    # Parameters that can be empty
    ("output_dir", "E:/SECourses_Improved_Trainer_v1/test_case"),
    ("output_name", "my_lora"),
    ("comment", ""),  # Can be empty
    ("metadata_author", ""),  # Can be empty
    
    # List parameters
    ("network_args", "[]"),  # Should be converted to empty list
    ("optimizer_args", "[]"),  # Should be converted to empty list
    
    # Numeric parameters
    ("learning_rate", 5e-5),
    ("network_dim", 16),
    ("network_alpha", 16),
    
    # Other parameters
    ("dataset_config", "E:/SECourses_Improved_Trainer_v1/test_case/dataset_config_20250828_142234.toml"),
    ("mixed_precision", "bf16"),
    ("network_module", "networks.lora_qwen_image"),
]

print("Testing SaveConfigFileToRun (for training)...")
print("-" * 50)

# Test with SaveConfigFileToRun (used for training)
test_file = "E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer/test_config_fixed.toml"
SaveConfigFileToRun(test_params, test_file)

# Load and verify
with open(test_file, 'r') as f:
    config = toml.load(f)

print(f"Saved {len(config)} parameters to {test_file}")
print()

# Check if empty path parameters were excluded
excluded_params = []
included_params = []

for key, value in test_params:
    if key in config:
        included_params.append(key)
        if isinstance(value, str) and value == "" and key in ["network_weights", "base_weights", "text_encoder", "sample_prompts"]:
            print(f"[ERROR]: {key} was included despite being empty path!")
    else:
        excluded_params.append(key)
        if isinstance(value, str) and value == "" and key in ["network_weights", "base_weights", "text_encoder", "sample_prompts"]:
            print(f"[CORRECT]: {key} was excluded (empty path)")

print()
print(f"Excluded parameters: {excluded_params}")
print(f"Included parameters: {included_params}")

# Check specific cases
print()
print("Checking specific parameter handling:")
print(f"  network_weights in config: {'network_weights' in config} (should be False)")
print(f"  base_weights in config: {'base_weights' in config} (should be False)")
print(f"  dit in config: {'dit' in config} (should be True)")
print(f"  vae in config: {'vae' in config} (should be True)")
print(f"  output_dir in config: {'output_dir' in config} (should be True)")
print(f"  comment in config: {'comment' in config} (should be True)")

# Check list conversions
if "network_args" in config:
    print(f"  network_args type: {type(config['network_args'])} value: {config['network_args']}")
if "optimizer_args" in config:
    print(f"  optimizer_args type: {type(config['optimizer_args'])} value: {config['optimizer_args']}")

# Clean up
os.remove(test_file)

print()
print("=" * 50)
print("SUMMARY:")
if "network_weights" not in config and "base_weights" not in config:
    print("[SUCCESS]: Empty path parameters are properly excluded!")
    print("The fix prevents FileNotFoundError from empty string paths.")
else:
    print("[FAILURE]: Empty path parameters were not excluded properly.")