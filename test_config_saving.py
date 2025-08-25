#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for config saving functionality
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path


def test_timestamp_format():
    """Test the timestamp format generation"""
    print("\n=== Testing Timestamp Format ===")
    
    # Generate multiple timestamps to ensure uniqueness
    timestamps = []
    for i in range(5):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        timestamps.append(timestamp)
        print(f"Timestamp {i+1}: {timestamp}")
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.01)
    
    # Check uniqueness
    if len(timestamps) == len(set(timestamps)):
        print("[OK] All timestamps are unique")
    else:
        print("[FAIL] Duplicate timestamps found")
    
    # Verify format
    for ts in timestamps:
        parts = ts.split('_')
        if len(parts) == 7:  # year, month, day, hour, minute, second, millisecond
            print(f"[OK] Format correct for {ts}")
        else:
            print(f"[FAIL] Format incorrect for {ts}")


def test_config_saving():
    """Test saving dataset TOML and JSON config"""
    print("\n=== Testing Config Saving ===")
    
    # Create a temporary output directory
    output_dir = tempfile.mkdtemp(prefix="test_output_")
    print(f"Test output directory: {output_dir}")
    
    try:
        # Create a sample dataset TOML
        dataset_toml_content = """
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
image_directory = "./test_images"
num_repeats = 1
cache_directory = "./test_images/cache"
"""
        
        # Create temporary dataset TOML file
        temp_toml = os.path.join(tempfile.gettempdir(), "test_dataset.toml")
        with open(temp_toml, 'w') as f:
            f.write(dataset_toml_content)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        
        # Copy dataset TOML to output directory
        dataset_toml_dest = os.path.join(output_dir, f"{timestamp}.toml")
        shutil.copy2(temp_toml, dataset_toml_dest)
        
        if os.path.exists(dataset_toml_dest):
            print(f"[OK] Dataset TOML saved to: {dataset_toml_dest}")
        else:
            print(f"[FAIL] Dataset TOML not saved")
        
        # Create sample JSON config
        config_dict = {
            "dataset_config": temp_toml,
            "output_dir": output_dir,
            "output_name": "test_model",
            "learning_rate": 0.0001,
            "max_train_epochs": 10,
            "batch_size": 1,
            "network_dim": 32,
            "network_alpha": 16,
            "dit": "/path/to/dit/model.safetensors",
            "vae": "/path/to/vae/model.safetensors",
            "text_encoder": "/path/to/text_encoder.safetensors",
            "timestamp_created": timestamp
        }
        
        # Save JSON config
        json_config_path = os.path.join(output_dir, f"{timestamp}.json")
        with open(json_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        
        if os.path.exists(json_config_path):
            print(f"[OK] JSON config saved to: {json_config_path}")
        else:
            print(f"[FAIL] JSON config not saved")
        
        # Verify contents
        with open(json_config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        if loaded_config["output_name"] == "test_model":
            print("[OK] JSON config content verified")
        else:
            print("[FAIL] JSON config content incorrect")
        
        # Check file naming consistency
        toml_name = os.path.basename(dataset_toml_dest).replace('.toml', '')
        json_name = os.path.basename(json_config_path).replace('.json', '')
        
        if toml_name == json_name:
            print(f"[OK] File names are consistent: {toml_name}")
        else:
            print(f"[FAIL] File names inconsistent: TOML={toml_name}, JSON={json_name}")
        
        # List all files in output directory
        print("\nFiles in output directory:")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size} bytes)")
        
        # Clean up temp TOML
        os.remove(temp_toml)
        
    finally:
        # Clean up
        print(f"\n[CLEANUP] Removing test directory: {output_dir}")
        shutil.rmtree(output_dir)


def test_multiple_training_sessions():
    """Test that multiple training sessions create unique files"""
    print("\n=== Testing Multiple Training Sessions ===")
    
    output_dir = tempfile.mkdtemp(prefix="test_multi_")
    print(f"Test output directory: {output_dir}")
    
    try:
        timestamps = []
        
        # Simulate 3 training sessions
        for i in range(3):
            print(f"\nSimulating training session {i+1}...")
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            timestamps.append(timestamp)
            
            # Save files
            toml_path = os.path.join(output_dir, f"{timestamp}.toml")
            json_path = os.path.join(output_dir, f"{timestamp}.json")
            
            with open(toml_path, 'w') as f:
                f.write(f"# Training session {i+1}")
            
            with open(json_path, 'w') as f:
                json.dump({"session": i+1, "timestamp": timestamp}, f)
            
            print(f"  Created: {os.path.basename(toml_path)}")
            print(f"  Created: {os.path.basename(json_path)}")
            
            # Small delay between sessions
            import time
            time.sleep(0.1)
        
        # Verify all files exist and are unique
        files = os.listdir(output_dir)
        expected_count = 6  # 3 sessions * 2 files each
        
        if len(files) == expected_count:
            print(f"\n[OK] All {expected_count} files created")
        else:
            print(f"\n[FAIL] Expected {expected_count} files, found {len(files)}")
        
        # Check uniqueness of timestamps
        if len(timestamps) == len(set(timestamps)):
            print("[OK] All timestamps are unique across sessions")
        else:
            print("[FAIL] Duplicate timestamps found across sessions")
        
    finally:
        print(f"\n[CLEANUP] Removing test directory: {output_dir}")
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Config Saving Test Suite")
    print("=" * 60)
    
    test_timestamp_format()
    test_config_saving()
    test_multiple_training_sessions()
    
    print("\n" + "=" * 60)
    print("[OK] All tests completed!")
    print("=" * 60)