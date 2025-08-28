#!/usr/bin/env python
"""Test to see why save_last_n_epochs is being removed."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from musubi_tuner_gui.common_gui import SaveConfigFile, SaveConfigFileToRun
import toml

def test_save_last_n_epochs():
    """Test that save_last_n_epochs=0 is preserved."""
    
    print("Testing save_last_n_epochs preservation...")
    
    # Test parameters including save_last_n_epochs = 0
    test_params = [
        ("save_last_n_epochs", 0),
        ("save_last_n_epochs_state", 0),
        ("save_every_n_epochs", 1),
        ("output_dir", "test_output"),
        ("output_name", "test_model"),
    ]
    
    # Test SaveConfigFile
    test_file = os.path.join(os.path.dirname(__file__), "test_save_last_n.toml")
    SaveConfigFile(test_params, test_file)
    
    with open(test_file, 'r') as f:
        config = toml.load(f)
    
    print("\n=== SaveConfigFile Results ===")
    print(f"Parameters in TOML: {list(config.keys())}")
    
    if "save_last_n_epochs" in config:
        print(f"save_last_n_epochs: {config['save_last_n_epochs']} (type: {type(config['save_last_n_epochs']).__name__})")
    else:
        print("ERROR: save_last_n_epochs is MISSING from the TOML!")
    
    if "save_last_n_epochs_state" in config:
        print(f"save_last_n_epochs_state: {config['save_last_n_epochs_state']} (type: {type(config['save_last_n_epochs_state']).__name__})")
    else:
        print("ERROR: save_last_n_epochs_state is MISSING from the TOML!")
    
    # Test SaveConfigFileToRun
    test_file_run = os.path.join(os.path.dirname(__file__), "test_save_last_n_run.toml")
    SaveConfigFileToRun(test_params, test_file_run)
    
    with open(test_file_run, 'r') as f:
        config_run = toml.load(f)
    
    print("\n=== SaveConfigFileToRun Results ===")
    print(f"Parameters in TOML: {list(config_run.keys())}")
    
    if "save_last_n_epochs" in config_run:
        print(f"save_last_n_epochs: {config_run['save_last_n_epochs']} (type: {type(config_run['save_last_n_epochs']).__name__})")
    else:
        print("ERROR: save_last_n_epochs is MISSING from the TOML!")
    
    if "save_last_n_epochs_state" in config_run:
        print(f"save_last_n_epochs_state: {config_run['save_last_n_epochs_state']} (type: {type(config_run['save_last_n_epochs_state']).__name__})")
    else:
        print("ERROR: save_last_n_epochs_state is MISSING from the TOML!")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists(test_file_run):
        os.remove(test_file_run)
    
    # Check if the values are preserved correctly
    success = True
    if "save_last_n_epochs" not in config or config["save_last_n_epochs"] != 0:
        success = False
    if "save_last_n_epochs_state" not in config_run or config_run["save_last_n_epochs_state"] != 0:
        success = False
    
    if success:
        print("\n=== TEST PASSED ===")
        print("save_last_n_epochs=0 is correctly preserved")
    else:
        print("\n=== TEST FAILED ===")
        print("save_last_n_epochs=0 is being removed or modified")
    
    return success

if __name__ == "__main__":
    test_save_last_n_epochs()