"""
Script to test the comprehensive fix for empty string path handling issues.
This will show what parameters need fixing.
"""

import os

# List of all parameters that represent file paths and should not be empty strings
FILE_PATH_PARAMETERS = [
    # Model paths
    "network_weights",
    "base_weights", 
    "dit",
    "vae",
    "text_encoder",
    "caching_teo_text_encoder1",
    "caching_teo_text_encoder2",
    
    # Dataset and output paths
    # NOTE: output_dir can be empty and will be generated if not specified
    # NOTE: dataset_config should already be validated in train function
    
    # Resume paths
    "resume",
    "resume_from_huggingface",
    
    # Sample prompts
    "sample_prompts",  # This is a file path
    
    # Any other model/checkpoint paths
    "weights",
    "pretrained_model_name_or_path",
    "state_dict",
]

# Parameters that can be empty strings (not file paths)
ALLOWED_EMPTY_STRINGS = [
    "output_dir",  # Can be empty, will be auto-generated
    "output_name",  # Can be empty, will use default
    "comment",  # Optional comment
    "metadata_author",
    "metadata_description", 
    "metadata_license",
    "metadata_tags",
    "metadata_title",
    "huggingface_repo_id",
    "huggingface_token",
    "huggingface_path_in_repo",
    "extra_accelerate_launch_args",
    "additional_parameters",
    "wandb_api_key",
    "tracker_name",
    "tracker_run_name",
    "log_tracker_name",
    "log_tracker_config",
]

def should_exclude_empty_string(name, value):
    """
    Determine if a parameter with empty string value should be excluded from config.
    
    Returns True if the parameter should be excluded (not saved to config).
    """
    # Skip None values (already handled)
    if value is None:
        return True
    
    # If it's not a string, include it
    if not isinstance(value, str):
        return False
    
    # If it's not empty, include it  
    if value != "":
        return False
    
    # Now we have an empty string - decide if we should exclude it
    
    # If it's a known file path parameter, exclude it
    if name in FILE_PATH_PARAMETERS:
        print(f"  Excluding empty file path parameter: {name}")
        return True
    
    # If it's in the allowed empty strings list, include it
    if name in ALLOWED_EMPTY_STRINGS:
        return False
    
    # For unknown parameters, check if it looks like a path
    if any(keyword in name.lower() for keyword in ["path", "file", "weights", "model", "checkpoint", "ckpt"]):
        print(f"  Excluding suspected empty file path parameter: {name}")
        return True
    
    # Otherwise include it
    return False

if __name__ == "__main__":
    # Test with some examples
    test_cases = [
        ("network_weights", ""),
        ("base_weights", ""),
        ("output_dir", ""),
        ("comment", ""),
        ("dit", ""),
        ("vae", ""),
        ("some_random_param", ""),
        ("model_path", ""),
        ("checkpoint_file", ""),
    ]
    
    print("Testing parameter exclusion logic:")
    print("-" * 50)
    for name, value in test_cases:
        should_exclude = should_exclude_empty_string(name, value)
        status = "EXCLUDE" if should_exclude else "INCLUDE"
        print(f"{name:20s} = '{value}' -> {status}")