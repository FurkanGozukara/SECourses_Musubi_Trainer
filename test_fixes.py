#!/usr/bin/env python
"""
Test script to verify the fixes for empty string parameters in training scripts
"""

import os
import sys
import toml

def test_empty_string_handling():
    """Test that empty strings are properly handled in config loading"""
    
    # Create a test config with empty strings
    test_config = {
        'network_weights': '',
        'base_weights': '', 
        'resume': '',
        'output_dir': 'E:/test',
        'learning_rate': 5e-5
    }
    
    # Save test config
    test_config_path = 'test_empty_strings.toml'
    with open(test_config_path, 'w') as f:
        toml.dump(test_config, f)
    
    print("Created test config with empty strings")
    print(f"  network_weights: '{test_config['network_weights']}'")
    print(f"  base_weights: '{test_config['base_weights']}'")
    print(f"  resume: '{test_config['resume']}'")
    
    # Test loading with the fixed code
    sys.path.insert(0, 'musubi-tuner/src')
    from musubi_tuner.hv_train_network import read_config_from_file
    import argparse
    
    # Create parser with minimal required args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--network_weights', type=str, default=None)
    parser.add_argument('--base_weights', type=str, default=None) 
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    
    # Create args with config file
    args = argparse.Namespace(config_file=test_config_path)
    
    # Load config with the fixed function
    loaded_args = read_config_from_file(args, parser)
    
    print("\nAfter loading with fixed code:")
    print(f"  network_weights: {loaded_args.network_weights} (type: {type(loaded_args.network_weights).__name__})")
    print(f"  base_weights: {loaded_args.base_weights} (type: {type(loaded_args.base_weights).__name__})")
    print(f"  resume: {loaded_args.resume} (type: {type(loaded_args.resume).__name__})")
    
    # Verify they are None, not empty strings
    success = True
    if loaded_args.network_weights != None:
        print(f"ERROR: network_weights should be None, got {loaded_args.network_weights}")
        success = False
    if loaded_args.base_weights != None:
        print(f"ERROR: base_weights should be None, got {loaded_args.base_weights}")
        success = False
    if loaded_args.resume != None:
        print(f"ERROR: resume should be None, got {loaded_args.resume}")
        success = False
        
    # Clean up
    os.remove(test_config_path)
    
    if success:
        print("\n[SUCCESS] All empty strings correctly converted to None")
        return True
    else:
        print("\n[FAILED] Some parameters not correctly converted")
        return False

if __name__ == "__main__":
    test_empty_string_handling()