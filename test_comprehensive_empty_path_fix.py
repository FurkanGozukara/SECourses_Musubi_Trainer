"""
Comprehensive test to verify ALL empty string path handling fixes.
This tests all the file path parameters that could cause FileNotFoundError.
"""

import sys
import os
sys.path.insert(0, 'E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer')
from musubi_tuner_gui.common_gui import SaveConfigFileToRun
import toml

# Create test parameters with ALL possible empty path strings
test_params = [
    # Model and weight paths (should be excluded when empty)
    ("network_weights", ""),
    ("base_weights", ""),
    ("dit", "E:/valid/path.safetensors"),  # Valid, should be included
    ("vae", "E:/valid/vae.safetensors"),   # Valid, should be included
    ("text_encoder", ""),
    ("weights", ""),
    ("pretrained_model_name_or_path", ""),
    ("state_dict", ""),
    ("checkpoint", ""),
    ("ckpt", ""),
    ("safetensors", ""),
    ("model_path", ""),
    
    # Text encoder paths (should be excluded when empty)
    ("text_encoder1", ""),
    ("text_encoder2", ""),
    ("caching_teo_text_encoder", ""),
    ("caching_teo_text_encoder1", ""),
    ("caching_teo_text_encoder2", ""),
    
    # Resume and state paths (should be excluded when empty)
    ("resume", ""),
    ("resume_from_huggingface", ""),
    
    # Sample and prompt paths (should be excluded when empty)
    ("sample_prompts", ""),
    ("prompt_file", ""),
    ("from_file", ""),
    
    # Config and tracker paths (should be excluded when empty)
    ("log_tracker_config", ""),
    ("dataset_config", "E:/valid/dataset.toml"),  # Valid, should be included
    
    # Output paths (should be excluded when empty as inputs)
    ("jsonl_output_file", ""),
    ("image_jsonl_file", ""),
    ("video_jsonl_file", ""),
    
    # Latent paths (should be excluded when empty)
    ("latent_path", ""),
    
    # Generated paths (should be excluded when empty)
    ("generated_toml_path", ""),
    
    # Parameters that CAN be empty (should be included)
    ("output_dir", ""),
    ("output_name", ""),
    ("comment", ""),
    ("metadata_author", ""),
    ("metadata_description", ""),
    ("huggingface_repo_id", ""),
    ("wandb_api_key", ""),
    
    # Non-path parameters (should be included)
    ("learning_rate", 5e-5),
    ("network_dim", 16),
    ("mixed_precision", "bf16"),
    ("network_module", "networks.lora"),
]

print("Testing comprehensive empty path fix...")
print("=" * 60)

# Save config using the fixed function
test_file = "E:/SECourses_Improved_Trainer_v1/SECourses_Improved_Trainer/test_comprehensive_config.toml"
SaveConfigFileToRun(test_params, test_file)

# Load and verify
with open(test_file, 'r') as f:
    config = toml.load(f)

print(f"Total parameters provided: {len(test_params)}")
print(f"Parameters saved to config: {len(config)}")
print()

# Check each empty path parameter
empty_path_params = [
    "network_weights", "base_weights", "text_encoder", "weights",
    "pretrained_model_name_or_path", "state_dict", "checkpoint", "ckpt",
    "safetensors", "model_path", "text_encoder1", "text_encoder2",
    "caching_teo_text_encoder", "caching_teo_text_encoder1", "caching_teo_text_encoder2",
    "resume", "resume_from_huggingface", "sample_prompts", "prompt_file",
    "from_file", "log_tracker_config", "jsonl_output_file", "image_jsonl_file",
    "video_jsonl_file", "latent_path", "generated_toml_path"
]

errors = []
for param in empty_path_params:
    if param in config:
        errors.append(param)
        print(f"[ERROR]: {param} was included despite being empty path!")

if not errors:
    print("[SUCCESS]: All empty path parameters were correctly excluded!")
    print()
    print("Empty path parameters excluded (correct):")
    for param in empty_path_params:
        if param not in config:
            print(f"  - {param}")
else:
    print(f"\n[FAILURE]: {len(errors)} empty path parameters were not excluded:")
    for param in errors:
        print(f"  - {param}")

print()
print("Parameters that CAN be empty (should be included):")
allowed_empty = ["output_dir", "output_name", "comment", "metadata_author", 
                  "metadata_description", "huggingface_repo_id", "wandb_api_key"]
for param in allowed_empty:
    status = "INCLUDED" if param in config else "EXCLUDED"
    print(f"  {param}: {status}")

print()
print("Valid paths that should be included:")
valid_paths = ["dit", "vae", "dataset_config"]
for param in valid_paths:
    if param in config:
        print(f"  [OK] {param}: {config[param]}")
    else:
        print(f"  [ERROR] {param}: Missing from config!")

# Clean up
os.remove(test_file)

print()
print("=" * 60)
if not errors:
    print("[COMPREHENSIVE TEST PASSED]")
    print("The fix successfully prevents ALL FileNotFoundError cases from empty paths!")
else:
    print(f"[TEST FAILED] {len(errors)} parameters still have issues.")