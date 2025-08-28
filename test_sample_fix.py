#!/usr/bin/env python
"""Test script to verify the fix for sample_every_n_steps ZeroDivisionError."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from musubi_tuner_gui.common_gui import SaveConfigFileToRun, SaveConfigFile
import toml

def test_sample_parameter_conversion():
    """Test that sample_every_n_steps and sample_every_n_epochs are converted from 0 to None."""
    
    print("Testing sample parameter conversion...")
    
    # Test parameters with 0 values for sampling
    test_params = [
        ("sample_every_n_steps", 0),
        ("sample_every_n_epochs", 0),
        ("output_dir", "test_output"),
        ("output_name", "test_model"),
        ("learning_rate", 5e-5),
    ]
    
    # Test SaveConfigFileToRun
    test_file_run = os.path.join(os.path.dirname(__file__), "test_config_run.toml")
    SaveConfigFileToRun(test_params, test_file_run)
    
    with open(test_file_run, 'r') as f:
        config_run = toml.load(f)
    
    print("\n=== SaveConfigFileToRun Results ===")
    print(f"sample_every_n_steps: {config_run.get('sample_every_n_steps')} (type: {type(config_run.get('sample_every_n_steps')).__name__})")
    print(f"sample_every_n_epochs: {config_run.get('sample_every_n_epochs')} (type: {type(config_run.get('sample_every_n_epochs')).__name__})")
    
    # Test SaveConfigFile  
    test_file = os.path.join(os.path.dirname(__file__), "test_config.toml")
    SaveConfigFile(test_params, test_file)
    
    with open(test_file, 'r') as f:
        config = toml.load(f)
    
    print("\n=== SaveConfigFile Results ===")
    print(f"sample_every_n_steps: {config.get('sample_every_n_steps')} (type: {type(config.get('sample_every_n_steps')).__name__})")
    print(f"sample_every_n_epochs: {config.get('sample_every_n_epochs')} (type: {type(config.get('sample_every_n_epochs')).__name__})")
    
    # Check if the values are None (not in the file) or properly handled
    success = True
    
    # Check SaveConfigFileToRun
    if 'sample_every_n_steps' in config_run and config_run['sample_every_n_steps'] == 0:
        print("\n[ERROR] SaveConfigFileToRun: sample_every_n_steps is still 0 (should be None or excluded)")
        success = False
    elif 'sample_every_n_steps' not in config_run:
        print("\n[OK] SaveConfigFileToRun: sample_every_n_steps excluded from TOML (treated as None)")
    else:
        print(f"\n[OK] SaveConfigFileToRun: sample_every_n_steps = {config_run['sample_every_n_steps']}")
    
    if 'sample_every_n_epochs' in config_run and config_run['sample_every_n_epochs'] == 0:
        print("[ERROR] SaveConfigFileToRun: sample_every_n_epochs is still 0 (should be None or excluded)")
        success = False
    elif 'sample_every_n_epochs' not in config_run:
        print("[OK] SaveConfigFileToRun: sample_every_n_epochs excluded from TOML (treated as None)")
    else:
        print(f"[OK] SaveConfigFileToRun: sample_every_n_epochs = {config_run['sample_every_n_epochs']}")
    
    # Check SaveConfigFile
    if 'sample_every_n_steps' in config and config['sample_every_n_steps'] == 0:
        print("\n[ERROR] SaveConfigFile: sample_every_n_steps is still 0 (should be None or excluded)")
        success = False
    elif 'sample_every_n_steps' not in config:
        print("\n[OK] SaveConfigFile: sample_every_n_steps excluded from TOML (treated as None)")
    else:
        print(f"\n[OK] SaveConfigFile: sample_every_n_steps = {config['sample_every_n_steps']}")
    
    if 'sample_every_n_epochs' in config and config['sample_every_n_epochs'] == 0:
        print("[ERROR] SaveConfigFile: sample_every_n_epochs is still 0 (should be None or excluded)")
        success = False
    elif 'sample_every_n_epochs' not in config:
        print("[OK] SaveConfigFile: sample_every_n_epochs excluded from TOML (treated as None)")
    else:
        print(f"[OK] SaveConfigFile: sample_every_n_epochs = {config['sample_every_n_epochs']}")
    
    # Cleanup
    if os.path.exists(test_file_run):
        os.remove(test_file_run)
    if os.path.exists(test_file):
        os.remove(test_file)
    
    if success:
        print("\n=== TEST PASSED ===")
        print("The fix correctly converts 0 to None for sample_every_n_steps and sample_every_n_epochs")
    else:
        print("\n=== TEST FAILED ===")
        print("The parameters are not being properly converted")
    
    return success

if __name__ == "__main__":
    test_sample_parameter_conversion()