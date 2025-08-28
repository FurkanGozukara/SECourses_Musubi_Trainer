"""Test script to verify network_args fix"""
import sys
import os
import toml
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from musubi_tuner_gui.common_gui import SaveConfigFile, SaveConfigFileToRun

def test_network_args_fix():
    print("Testing network_args fix...")
    
    # Test parameters with string representation of empty list
    test_params = [
        ("network_args", "[]"),
        ("optimizer_args", "[]"),
        ("lr_scheduler_args", "[]"),
        ("output_dir", "test_output"),
        ("learning_rate", 5e-5),
        ("max_train_steps", 100),
    ]
    
    # Test SaveConfigFile
    print("\n1. Testing SaveConfigFile...")
    test_file1 = os.path.join(os.path.dirname(__file__), "test_config_file.toml")
    SaveConfigFile(test_params, test_file1)
    
    # Load and check
    with open(test_file1, "r") as f:
        config1 = toml.load(f)
    
    print(f"   network_args type: {type(config1['network_args'])}")
    print(f"   network_args value: {config1['network_args']}")
    assert isinstance(config1['network_args'], list), "network_args should be a list!"
    assert config1['network_args'] == [], "network_args should be an empty list!"
    print("   ✓ SaveConfigFile correctly converts '[]' string to empty list")
    
    # Test SaveConfigFileToRun
    print("\n2. Testing SaveConfigFileToRun...")
    test_file2 = os.path.join(os.path.dirname(__file__), "test_config_file_to_run.toml")
    SaveConfigFileToRun(test_params, test_file2)
    
    # Load and check
    with open(test_file2, "r") as f:
        config2 = toml.load(f)
    
    print(f"   network_args type: {type(config2['network_args'])}")
    print(f"   network_args value: {config2['network_args']}")
    assert isinstance(config2['network_args'], list), "network_args should be a list!"
    assert config2['network_args'] == [], "network_args should be an empty list!"
    print("   ✓ SaveConfigFileToRun correctly converts '[]' string to empty list")
    
    # Test with actual arguments
    print("\n3. Testing with actual network arguments...")
    test_params_with_args = [
        ("network_args", "conv_dim=4 conv_alpha=1"),
        ("optimizer_args", "weight_decay=0.01 betas=0.9,0.999"),
        ("lr_scheduler_args", '["T_max=100", "eta_min=1e-6"]'),  # JSON format
        ("output_dir", "test_output"),
    ]
    
    test_file3 = os.path.join(os.path.dirname(__file__), "test_config_with_args.toml")
    SaveConfigFileToRun(test_params_with_args, test_file3)
    
    with open(test_file3, "r") as f:
        config3 = toml.load(f)
    
    print(f"   network_args: {config3['network_args']}")
    assert isinstance(config3['network_args'], list), "network_args should be a list!"
    assert config3['network_args'] == ["conv_dim=4", "conv_alpha=1"], "network_args not parsed correctly!"
    print("   ✓ Space-separated args parsed correctly")
    
    print(f"   optimizer_args: {config3['optimizer_args']}")
    assert isinstance(config3['optimizer_args'], list), "optimizer_args should be a list!"
    assert config3['optimizer_args'] == ["weight_decay=0.01", "betas=0.9,0.999"], "optimizer_args not parsed correctly!"
    print("   ✓ Optimizer args parsed correctly")
    
    print(f"   lr_scheduler_args: {config3['lr_scheduler_args']}")
    assert isinstance(config3['lr_scheduler_args'], list), "lr_scheduler_args should be a list!"
    assert config3['lr_scheduler_args'] == ["T_max=100", "eta_min=1e-6"], "lr_scheduler_args not parsed correctly!"
    print("   ✓ JSON format args parsed correctly")
    
    # Clean up test files
    for f in [test_file1, test_file2, test_file3]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n✅ All tests passed! The fix is working correctly.")
    print("\n📋 Summary:")
    print("- Empty string '[]' is correctly converted to empty list []")
    print("- Space-separated arguments are correctly parsed into list")
    print("- JSON format arguments are correctly parsed")
    print("- Both SaveConfigFile and SaveConfigFileToRun are fixed")

if __name__ == "__main__":
    test_network_args_fix()