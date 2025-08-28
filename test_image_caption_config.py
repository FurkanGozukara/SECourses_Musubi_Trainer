#!/usr/bin/env python3
"""Test script to verify image captioning configuration loading"""

import sys
import toml
from musubi_tuner_gui.class_gui_config import GUIConfig
from musubi_tuner_gui.class_tab_config_manager import TabConfigManager

def test_default_loading():
    """Test loading image_captioning_defaults.toml"""
    print("Testing default config loading...")
    
    # Load the defaults file directly
    with open("image_captioning_defaults.toml", "r") as f:
        defaults = toml.load(f)
    
    print("\n=== Direct TOML Load ===")
    print(f"Has image_captioning section: {'image_captioning' in defaults}")
    
    if 'image_captioning' in defaults:
        ic_config = defaults['image_captioning']
        print(f"Temperature: {ic_config.get('temperature', 'NOT FOUND')}")
        print(f"Top K: {ic_config.get('top_k', 'NOT FOUND')}")
        print(f"Top P: {ic_config.get('top_p', 'NOT FOUND')}")
        print(f"Do Sample: {ic_config.get('do_sample', 'NOT FOUND')}")
        print(f"Repetition Penalty: {ic_config.get('repetition_penalty', 'NOT FOUND')}")

def test_gui_config():
    """Test GUIConfig loading"""
    print("\n=== GUIConfig Test ===")
    
    # Test with image_captioning_defaults.toml
    config = GUIConfig("image_captioning_defaults.toml")
    
    # Test nested key access
    print(f"image_captioning.temperature: {config.get('image_captioning.temperature', 'NOT FOUND')}")
    print(f"image_captioning.top_k: {config.get('image_captioning.top_k', 'NOT FOUND')}")
    print(f"image_captioning.top_p: {config.get('image_captioning.top_p', 'NOT FOUND')}")
    print(f"image_captioning.do_sample: {config.get('image_captioning.do_sample', 'NOT FOUND')}")
    print(f"image_captioning.repetition_penalty: {config.get('image_captioning.repetition_penalty', 'NOT FOUND')}")

def test_tab_config_manager():
    """Test TabConfigManager loading"""
    print("\n=== TabConfigManager Test ===")
    
    # Create manager (will load from defaults)
    manager = TabConfigManager()
    
    # Get config for image_captioning tab
    config = manager.get_config_for_tab("image_captioning")
    
    print(f"Config loaded: {config is not None}")
    if config:
        # Debug: Check what's actually in the config
        print(f"Config data keys: {list(config.config.keys())}")
        if 'image_captioning' in config.config:
            print(f"image_captioning keys: {list(config.config['image_captioning'].keys())[:5]}...")
        
        print(f"image_captioning.temperature: {config.get('image_captioning.temperature', 'NOT FOUND')}")
        print(f"image_captioning.top_k: {config.get('image_captioning.top_k', 'NOT FOUND')}")
        print(f"image_captioning.top_p: {config.get('image_captioning.top_p', 'NOT FOUND')}")
        print(f"image_captioning.do_sample: {config.get('image_captioning.do_sample', 'NOT FOUND')}")
        print(f"image_captioning.repetition_penalty: {config.get('image_captioning.repetition_penalty', 'NOT FOUND')}")
        print(f"image_captioning.model_path: {config.get('image_captioning.model_path', 'NOT FOUND')}")
        print(f"image_captioning.fp8_vl: {config.get('image_captioning.fp8_vl', 'NOT FOUND')}")

def test_save_load():
    """Test save/load functionality"""
    print("\n=== Save/Load Test ===")
    
    # Create a test config
    test_config = {
        "image_captioning": {
            "model_path": "/test/path/model.safetensors",
            "temperature": 0.8,
            "top_k": 60,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "prefix": "Test prefix: ",
            "suffix": " (test suffix)"
        }
    }
    
    # Save it
    test_file = "test_caption_config.toml"
    with open(test_file, "w") as f:
        toml.dump(test_config, f)
    
    print(f"Saved test config to {test_file}")
    
    # Load it back with GUIConfig
    loaded_config = GUIConfig(test_file)
    
    print(f"Loaded temperature: {loaded_config.get('image_captioning.temperature', 'NOT FOUND')}")
    print(f"Loaded top_k: {loaded_config.get('image_captioning.top_k', 'NOT FOUND')}")
    print(f"Loaded prefix: {loaded_config.get('image_captioning.prefix', 'NOT FOUND')}")
    
    # Clean up
    import os
    os.remove(test_file)
    print(f"Cleaned up {test_file}")

def main():
    print("=" * 60)
    print("Image Captioning Configuration Test")
    print("=" * 60)
    
    try:
        test_default_loading()
    except Exception as e:
        print(f"Error in test_default_loading: {e}")
    
    try:
        test_gui_config()
    except Exception as e:
        print(f"Error in test_gui_config: {e}")
    
    try:
        test_tab_config_manager()
    except Exception as e:
        print(f"Error in test_tab_config_manager: {e}")
    
    try:
        test_save_load()
    except Exception as e:
        print(f"Error in test_save_load: {e}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()