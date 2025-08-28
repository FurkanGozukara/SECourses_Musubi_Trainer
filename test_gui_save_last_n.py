#!/usr/bin/env python
"""Test what happens when save_last_n_epochs is 0 in the GUI save operation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from musubi_tuner_gui.common_gui import SaveConfigFile
import toml

def test_gui_save_scenario():
    """Simulate what the GUI does when saving with save_last_n_epochs=0."""
    
    print("Testing GUI save scenario with save_last_n_epochs=0...")
    
    # Simulate the parameters that would come from the GUI
    # This mimics what qwen_image_gui_actions receives
    test_params = []
    
    # Add various parameters as they would be from the GUI
    test_params.append(("output_dir", "E:/test_output"))
    test_params.append(("output_name", "my_model"))
    test_params.append(("save_every_n_epochs", 1))
    test_params.append(("save_last_n_epochs", 0))  # This is what GUI sends when value is 0
    test_params.append(("save_last_n_epochs_state", 0))
    test_params.append(("save_every_n_steps", 0))
    test_params.append(("sample_every_n_steps", 0))
    test_params.append(("learning_rate", 5e-5))
    
    # Test SaveConfigFile as GUI would use it
    test_file = os.path.join(os.path.dirname(__file__), "test_gui_save.toml")
    SaveConfigFile(test_params, test_file)
    
    # Load and check the saved file
    with open(test_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    print("\n=== Saved TOML Contents ===")
    print(f"Parameters in TOML: {list(config.keys())}")
    print("")
    
    # Check specific parameters
    params_to_check = ["save_last_n_epochs", "save_last_n_epochs_state", "save_every_n_steps", "sample_every_n_steps"]
    
    for param in params_to_check:
        if param in config:
            value = config[param]
            print(f"{param}: {value} (type: {type(value).__name__ if value is not None else 'None'})")
        else:
            print(f"{param}: MISSING FROM TOML!")
    
    # Show the actual TOML content
    print("\n=== Raw TOML File ===")
    with open(test_file, 'r', encoding='utf-8') as f:
        print(f.read())
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Summary
    print("\n=== Summary ===")
    if "save_last_n_epochs" in config:
        if config["save_last_n_epochs"] == 0:
            print("✓ save_last_n_epochs=0 is preserved correctly")
        elif config["save_last_n_epochs"] is None:
            print("✗ save_last_n_epochs was converted to None (should be 0)")
        else:
            print(f"✗ save_last_n_epochs has unexpected value: {config['save_last_n_epochs']}")
    else:
        print("✗ save_last_n_epochs is missing from TOML")

if __name__ == "__main__":
    test_gui_save_scenario()