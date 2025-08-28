#!/usr/bin/env python
"""
Comprehensive test script to verify all empty string parameter fixes
"""

import os
import sys
import toml

def test_empty_string_handling():
    """Test that all empty strings are properly handled in config loading"""
    
    # Create a test config with all empty string parameters
    test_config = {
        'network_weights': '',
        'base_weights': '', 
        'resume': '',
        'additional_parameters': '',
        'huggingface_path_in_repo': '',
        'huggingface_repo_id': '',
        'huggingface_repo_type': '',
        'huggingface_repo_visibility': '',
        'huggingface_token': '',
        'log_prefix': '',
        'log_tracker_config': '',
        'log_tracker_name': '',
        'logging_dir': '',
        'lr_scheduler_type': '',
        'metadata_author': '',
        'metadata_description': '',
        'metadata_license': '',
        'metadata_tags': '',
        'metadata_title': '',
        'resume_from_huggingface': '',
        'sample_prompts': '',
        'show_timesteps': '',
        'training_comment': '',
        'wandb_api_key': '',
        'wandb_run_name': '',
        # Add some non-empty values for required fields
        'output_dir': 'E:/test',
        'learning_rate': 5e-5,
        'config_file': 'test_config'
    }
    
    # Save test config
    test_config_path = 'test_all_empty_strings.toml'
    with open(test_config_path, 'w') as f:
        toml.dump(test_config, f)
    
    print("Created test config with empty strings")
    print(f"Total parameters with empty strings: {sum(1 for v in test_config.values() if v == '')}")
    
    # Test loading with the fixed code
    sys.path.insert(0, 'musubi-tuner/src')
    from musubi_tuner.hv_train_network import read_config_from_file
    import argparse
    
    # Create a comprehensive parser
    parser = argparse.ArgumentParser()
    
    # Add all arguments that might be affected
    params = [
        'config_file', 'network_weights', 'base_weights', 'resume',
        'additional_parameters', 'huggingface_path_in_repo', 'huggingface_repo_id',
        'huggingface_repo_type', 'huggingface_repo_visibility', 'huggingface_token',
        'log_prefix', 'log_tracker_config', 'log_tracker_name', 'logging_dir',
        'lr_scheduler_type', 'metadata_author', 'metadata_description',
        'metadata_license', 'metadata_tags', 'metadata_title', 'resume_from_huggingface',
        'sample_prompts', 'show_timesteps', 'training_comment', 'wandb_api_key',
        'wandb_run_name', 'output_dir'
    ]
    
    for param in params:
        if param in ['learning_rate']:
            parser.add_argument(f'--{param}', type=float, default=None)
        else:
            parser.add_argument(f'--{param}', type=str, default=None)
    
    # Add learning rate as float
    parser.add_argument('--learning_rate', type=float, default=None)
    
    # Create args with config file
    args = argparse.Namespace(config_file=test_config_path)
    
    # Load config with the fixed function
    loaded_args = read_config_from_file(args, parser)
    
    print("\nAfter loading with fixed code:")
    
    # Check all empty string parameters
    empty_string_params = [
        'network_weights', 'base_weights', 'resume', 'additional_parameters',
        'huggingface_path_in_repo', 'huggingface_repo_id', 'huggingface_repo_type',
        'huggingface_repo_visibility', 'huggingface_token', 'log_prefix',
        'log_tracker_config', 'log_tracker_name', 'logging_dir', 'lr_scheduler_type',
        'metadata_author', 'metadata_description', 'metadata_license', 'metadata_tags',
        'metadata_title', 'resume_from_huggingface', 'sample_prompts', 'show_timesteps',
        'training_comment', 'wandb_api_key', 'wandb_run_name'
    ]
    
    # Verify they are None, not empty strings
    success = True
    failed_params = []
    for param in empty_string_params:
        if hasattr(loaded_args, param):
            value = getattr(loaded_args, param)
            if value != None:
                print(f"ERROR: {param} should be None, got '{value}' (type: {type(value).__name__})")
                failed_params.append(param)
                success = False
        else:
            print(f"WARNING: {param} not found in loaded_args")
    
    # Clean up
    os.remove(test_config_path)
    
    if success:
        print(f"\n[SUCCESS] All {len(empty_string_params)} empty string parameters correctly converted to None")
        return True
    else:
        print(f"\n[FAILED] {len(failed_params)} parameters not correctly converted:")
        for param in failed_params:
            print(f"  - {param}")
        return False

if __name__ == "__main__":
    success = test_empty_string_handling()
    exit(0 if success else 1)