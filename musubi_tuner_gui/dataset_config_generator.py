import os
import re
import toml
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .common_gui import validate_path_for_toml, normalize_path, is_path_safe


def extract_repeat_count(folder_name: str) -> Tuple[int, str]:
    """
    Extract repeat count from folder name.
    Examples:
        "3_ohwx" -> (3, "ohwx")
        "10_my_dataset" -> (10, "my_dataset")
        "dataset" -> (1, "dataset")
    """
    match = re.match(r'^(\d+)_(.+)$', folder_name)
    if match:
        return int(match.group(1)), match.group(2)
    return 1, folder_name


def get_image_files(directory: str) -> List[str]:
    """Get all image files in a directory (non-recursive)."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        image_files.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    
    return sorted(image_files)


def create_caption_files(
    image_files: List[str],
    caption_extension: str,
    caption_content: str,
    overwrite: bool = False
) -> int:
    """
    Create caption files for images that don't have them.
    Returns the number of caption files created.
    """
    created_count = 0
    
    for image_file in image_files:
        caption_file = os.path.splitext(image_file)[0] + caption_extension
        
        if not os.path.exists(caption_file) or overwrite:
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption_content)
            created_count += 1
    
    return created_count


def generate_dataset_config_from_folders(
    parent_folder: str,
    resolution: Tuple[int, int],
    caption_extension: str = ".txt",
    create_missing_captions: bool = True,
    caption_strategy: str = "folder_name",  # "folder_name" or "empty"
    batch_size: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    cache_directory_name: str = "cache_dir",
    control_directory_name: str = "edit_images",
    qwen_image_edit_no_resize_control: bool = False
) -> Tuple[Dict, List[str]]:
    """
    Generate dataset configuration from folder structure.
    
    Returns:
        Tuple of (config_dict, messages_list)
        config_dict: The generated TOML configuration as a dictionary
        messages_list: List of status/warning messages
    """
    messages = []
    
    # Normalize parent folder path
    parent_folder = normalize_path(parent_folder)
    
    if not os.path.exists(parent_folder):
        raise ValueError(f"Parent folder does not exist: {parent_folder}")
    
    if not os.path.isdir(parent_folder):
        raise ValueError(f"Path is not a directory: {parent_folder}")
    
    # Get all subdirectories (excluding hidden ones)
    subdirs = [d for d in os.listdir(parent_folder) 
               if os.path.isdir(os.path.join(parent_folder, d)) 
               and not d.startswith('.')]
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in: {parent_folder}")
    
    # Sort subdirectories for consistent ordering
    subdirs.sort()
    
    # Build configuration
    config = {
        "general": {
            "resolution": list(resolution),
            "caption_extension": caption_extension,
            "batch_size": batch_size,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale
        },
        "datasets": []
    }
    
    for subdir in subdirs:
        subdir_path = os.path.join(parent_folder, subdir)
        
        # Check if this directory has subdirectories (which we don't want to scan)
        has_subdirs = any(os.path.isdir(os.path.join(subdir_path, item)) 
                         for item in os.listdir(subdir_path)
                         if not item.startswith('.') and item not in [cache_directory_name, control_directory_name])
        
        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct image files are supported)")
            continue
        
        # Get image files
        image_files = get_image_files(subdir_path)
        
        if not image_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No image files found")
            continue
        
        # Extract repeat count and clean name
        repeat_count, clean_name = extract_repeat_count(subdir)
        
        # Create caption files if requested
        if create_missing_captions:
            caption_content = ""
            if caption_strategy == "folder_name":
                caption_content = clean_name
            
            created = create_caption_files(
                image_files, 
                caption_extension, 
                caption_content
            )
            
            if created > 0:
                messages.append(f"[OK] Created {created} caption files for '{subdir}' with content: '{caption_content}'")
        
        # Check if all images have captions
        missing_captions = []
        for img in image_files:
            caption_file = os.path.splitext(img)[0] + caption_extension
            if not os.path.exists(caption_file):
                missing_captions.append(os.path.basename(img))
        
        if missing_captions:
            messages.append(f"[WARNING] '{subdir}': {len(missing_captions)} images missing caption files")
        
        # Build dataset entry with normalized paths
        dataset_entry = {
            "image_directory": validate_path_for_toml(subdir_path),
            "num_repeats": repeat_count
        }
        
        # Set cache directory - MUST be unique per dataset (musubi-tuner requirement)
        if cache_directory_name:
            if os.path.isabs(cache_directory_name):
                # Absolute path provided - append subdir name to make it unique
                cache_path = os.path.join(cache_directory_name, subdir)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)
            else:
                # Relative path - put inside subdirectory (each dataset gets its own)
                cache_path = os.path.join(subdir_path, cache_directory_name)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)
        
        # Check for control directory
        control_dir_path = os.path.join(subdir_path, control_directory_name)
        if os.path.exists(control_dir_path) and os.path.isdir(control_dir_path):
            dataset_entry["control_directory"] = validate_path_for_toml(control_dir_path)
            
            # Add Qwen Image Edit specific settings if control directory exists
            if qwen_image_edit_no_resize_control:
                dataset_entry["qwen_image_edit_no_resize_control"] = True
            
            messages.append(f"[OK] Found control directory for '{subdir}'")
        
        config["datasets"].append(dataset_entry)
    
    if not config["datasets"]:
        raise ValueError("No valid datasets found in the provided folder structure")
    
    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")
    
    return config, messages


def save_dataset_config(config: Dict, output_path: str) -> None:
    """Save the dataset configuration to a TOML file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        toml.dump(config, f)


def validate_dataset_config(config_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a dataset configuration file.
    Returns (is_valid, messages)
    """
    messages = []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = toml.load(f)
        
        # Check for required sections
        if "datasets" not in config or not config["datasets"]:
            messages.append("[ERROR] No datasets defined in configuration")
            return False, messages
        
        # Validate each dataset
        for i, dataset in enumerate(config["datasets"]):
            if "image_directory" not in dataset:
                messages.append(f"[ERROR] Dataset {i+1}: Missing image_directory")
                continue
            
            if not os.path.exists(dataset["image_directory"]):
                messages.append(f"[WARNING] Dataset {i+1}: Image directory does not exist: {dataset['image_directory']}")
            
            if "control_directory" in dataset and not os.path.exists(dataset["control_directory"]):
                messages.append(f"[WARNING] Dataset {i+1}: Control directory does not exist: {dataset['control_directory']}")
        
        messages.append(f"[OK] Configuration validated: {len(config['datasets'])} datasets")
        return True, messages
        
    except Exception as e:
        messages.append(f"[ERROR] Error validating configuration: {str(e)}")
        return False, messages