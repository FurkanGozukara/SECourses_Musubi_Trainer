#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the dataset configuration generator
"""

import os
import tempfile
import shutil
from pathlib import Path
from musubi_tuner_gui.dataset_config_generator import (
    generate_dataset_config_from_folders,
    save_dataset_config,
    validate_dataset_config,
    extract_repeat_count,
    get_image_files,
    create_caption_files
)


def create_test_folder_structure():
    """Create a test folder structure with sample images"""
    temp_dir = tempfile.mkdtemp(prefix="test_dataset_")
    print(f"Created test directory: {temp_dir}")
    
    # Create subdirectories with different naming patterns
    subdirs = [
        "1_ohwx",
        "3_style_training",
        "regular_folder",
        "10_high_quality"
    ]
    
    for subdir in subdirs:
        subdir_path = os.path.join(temp_dir, subdir)
        os.makedirs(subdir_path)
        
        # Create dummy image files
        for i in range(3):
            img_path = os.path.join(subdir_path, f"image_{i:03d}.png")
            with open(img_path, 'w') as f:
                f.write("dummy image content")
        
        # Create some caption files for testing
        if subdir == "3_style_training":
            for i in range(2):  # Only create captions for 2 out of 3 images
                caption_path = os.path.join(subdir_path, f"image_{i:03d}.txt")
                with open(caption_path, 'w') as f:
                    f.write("existing caption content")
    
    # Create a subdirectory with nested folders (should be skipped)
    nested_dir = os.path.join(temp_dir, "nested_folder")
    os.makedirs(os.path.join(nested_dir, "subfolder"))
    
    return temp_dir


def test_extract_repeat_count():
    """Test the repeat count extraction function"""
    print("\n=== Testing Repeat Count Extraction ===")
    
    test_cases = [
        ("3_ohwx", (3, "ohwx")),
        ("10_my_dataset", (10, "my_dataset")),
        ("dataset", (1, "dataset")),
        ("1_single", (1, "single")),
        ("100_large_number", (100, "large_number"))
    ]
    
    for folder_name, expected in test_cases:
        result = extract_repeat_count(folder_name)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} '{folder_name}' -> {result} (expected: {expected})")


def test_dataset_generation():
    """Test the dataset configuration generation"""
    print("\n=== Testing Dataset Configuration Generation ===")
    
    # Create test folder structure
    test_dir = create_test_folder_structure()
    
    try:
        # Test with folder_name caption strategy
        config, messages = generate_dataset_config_from_folders(
            parent_folder=test_dir,
            resolution=(960, 544),
            caption_extension=".txt",
            create_missing_captions=True,
            caption_strategy="folder_name",
            batch_size=1,
            enable_bucket=False,
            bucket_no_upscale=False,
            cache_directory_name="cache_dir",
            control_directory_name="edit_images",
            qwen_image_edit_no_resize_control=False
        )
        
        print("\nGenerated Configuration:")
        print("=" * 50)
        
        # Print general settings
        print("General Settings:")
        for key, value in config["general"].items():
            print(f"  {key}: {value}")
        
        # Print datasets
        print(f"\nDatasets ({len(config['datasets'])} found):")
        for i, dataset in enumerate(config["datasets"]):
            print(f"\n  Dataset {i+1}:")
            for key, value in dataset.items():
                if key == "image_directory":
                    value = os.path.basename(value)  # Show only folder name for readability
                print(f"    {key}: {value}")
        
        # Print messages
        print("\nStatus Messages:")
        for msg in messages:
            print(f"  {msg}")
        
        # Save the configuration
        output_path = os.path.join(test_dir, "generated_config.toml")
        save_dataset_config(config, output_path)
        print(f"\n[OK] Configuration saved to: {output_path}")
        
        # Validate the saved configuration
        is_valid, validation_messages = validate_dataset_config(output_path)
        print(f"\n=== Validation Results ===")
        print(f"Valid: {'[OK]' if is_valid else '[FAIL]'}")
        for msg in validation_messages:
            print(f"  {msg}")
        
        # Test with empty caption strategy
        print("\n=== Testing with Empty Caption Strategy ===")
        config2, messages2 = generate_dataset_config_from_folders(
            parent_folder=test_dir,
            resolution=(1024, 1024),
            caption_extension=".txt",
            create_missing_captions=True,
            caption_strategy="empty",
            batch_size=2,
            enable_bucket=True,
            bucket_no_upscale=True,
            cache_directory_name="/absolute/cache/path",
            control_directory_name="control",
            qwen_image_edit_no_resize_control=True
        )
        
        print(f"Generated {len(config2['datasets'])} datasets with empty captions")
        print(f"Bucket settings: enable={config2['general']['enable_bucket']}, no_upscale={config2['general']['bucket_no_upscale']}")
        
    finally:
        # Clean up
        print(f"\n[CLEANUP] Cleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir)


def test_error_handling():
    """Test error handling"""
    print("\n=== Testing Error Handling ===")
    
    # Test with non-existent directory
    try:
        config, messages = generate_dataset_config_from_folders(
            parent_folder="/non/existent/path",
            resolution=(960, 544)
        )
        print("[FAIL] Should have raised an error for non-existent path")
    except ValueError as e:
        print(f"[OK] Correctly raised error: {e}")
    
    # Test with empty directory
    temp_dir = tempfile.mkdtemp(prefix="empty_test_")
    try:
        config, messages = generate_dataset_config_from_folders(
            parent_folder=temp_dir,
            resolution=(960, 544)
        )
        print("[FAIL] Should have raised an error for empty directory")
    except ValueError as e:
        print(f"[OK] Correctly raised error: {e}")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Configuration Generator Test Suite")
    print("=" * 60)
    
    test_extract_repeat_count()
    test_dataset_generation()
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("[OK] All tests completed!")
    print("=" * 60)