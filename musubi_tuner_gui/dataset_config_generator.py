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


def _get_files_by_extensions(directory: str, extensions: List[str]) -> List[str]:
    """Collect files matching the extensions, deduplicated for case-insensitive filesystems."""
    found = {}
    for ext in extensions:
        for pattern in (f'*{ext}', f'*{ext.upper()}'):
            for path in glob.glob(os.path.join(directory, pattern)):
                found[os.path.normcase(os.path.abspath(path))] = path
    return sorted(found.values())


def get_image_files(directory: str) -> List[str]:
    """Get all image files in a directory (non-recursive)."""
    return _get_files_by_extensions(directory, ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'])


def get_video_files(directory: str) -> List[str]:
    """Get all video files in a directory (non-recursive)."""
    return _get_files_by_extensions(directory, ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv'])


def get_media_files(directory: str) -> Tuple[List[str], List[str]]:
    """Get all image and video files in a directory (non-recursive)."""
    image_files = get_image_files(directory)
    video_files = get_video_files(directory)
    return image_files, video_files


def create_caption_files(
    media_files: List[str],
    caption_extension: str,
    caption_content: str,
    overwrite: bool = False
) -> int:
    """
    Create caption files for media files (images/videos) that don't have them.
    Returns the number of caption files created.
    """
    created_count = 0

    for media_file in media_files:
        caption_file = os.path.splitext(media_file)[0] + caption_extension

        if not os.path.exists(caption_file) or overwrite:
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption_content)
            created_count += 1

    return created_count


def generate_wan_dataset_config_from_folders(
    parent_folder: str,
    resolution: Tuple[int, int],
    caption_extension: str = ".txt",
    create_missing_captions: bool = True,
    caption_strategy: str = "folder_name",  # "folder_name" or "empty"
    batch_size: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    cache_directory_name: str = "cache_dir",
    num_frames: int = 1,
    frame_extraction: str = "head",
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 129,
    source_fps: float = None
) -> Tuple[Dict, List[str]]:
    """
    Generate WAN dataset configuration from folder structure.
    Handles both images and videos for WAN training.

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
                         if not item.startswith('.') and item not in [cache_directory_name])

        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct media files are supported)")
            continue

        # Get media files (both images and videos)
        image_files, video_files = get_media_files(subdir_path)
        all_media_files = image_files + video_files

        if not all_media_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No image or video files found")
            continue

        # Extract repeat count and clean name
        repeat_count, clean_name = extract_repeat_count(subdir)

        # Determine dataset type based on content
        has_videos = len(video_files) > 0
        has_images = len(image_files) > 0

        # For WAN, use video_directory if videos are present, otherwise image_directory
        directory_type = "video_directory" if has_videos else "image_directory"
        media_type = "videos" if has_videos else "images"

        # Create caption files if requested
        if create_missing_captions:
            caption_content = ""
            if caption_strategy == "folder_name":
                caption_content = clean_name

            created = create_caption_files(
                all_media_files,
                caption_extension,
                caption_content
            )

            if created > 0:
                messages.append(f"[OK] Created {created} caption files for '{subdir}' with content: '{caption_content}'")

        # Check if all media files have captions
        missing_captions = []
        for media_file in all_media_files:
            caption_file = os.path.splitext(media_file)[0] + caption_extension
            if not os.path.exists(caption_file):
                missing_captions.append(os.path.basename(media_file))

        if missing_captions:
            messages.append(f"[WARNING] '{subdir}': {len(missing_captions)} media files missing caption files")

        # Build dataset entry with normalized paths
        dataset_entry = {
            directory_type: validate_path_for_toml(subdir_path),
            "num_repeats": repeat_count
        }
        
        # Add video-specific parameters if this is a video dataset
        if has_videos:
            dataset_entry["target_frames"] = [num_frames]
            dataset_entry["frame_extraction"] = frame_extraction
            dataset_entry["frame_stride"] = frame_stride
            dataset_entry["frame_sample"] = frame_sample
            dataset_entry["max_frames"] = max_frames
            if source_fps is not None and source_fps > 0:
                dataset_entry["source_fps"] = source_fps

        # Set cache directory - MUST be unique per dataset
        if cache_directory_name:
            if os.path.isabs(cache_directory_name):
                # Absolute path provided - append subdir name to make it unique
                cache_path = os.path.join(cache_directory_name, subdir)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)
            else:
                # Relative path - put inside subdirectory (each dataset gets its own)
                cache_path = os.path.join(subdir_path, cache_directory_name)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)

        config["datasets"].append(dataset_entry)

        # Add info about media types found
        media_info = []
        if has_images:
            media_info.append(f"{len(image_files)} images")
        if has_videos:
            media_info.append(f"{len(video_files)} videos")
        messages.append(f"[OK] Added {subdir} ({', '.join(media_info)}) as {media_type} dataset")

    if not config["datasets"]:
        raise ValueError("No valid datasets found in the provided folder structure")

    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")

    return config, messages


def round_frames_to_ltx2(frames: int) -> int:
    """Round a frame count DOWN to the nearest valid LTX-2 value (8*k + 1).

    The LTX-2 video VAE encodes the first frame alone and the rest in groups
    of 8, so valid frame counts are 1, 9, 17, 25, 33, 41, 49, ...
    """
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        return 1
    if frames <= 1:
        return 1
    return ((frames - 1) // 8) * 8 + 1


def generate_ltx2_dataset_config_from_folders(
    parent_folder: str,
    resolution: Tuple[int, int],
    caption_extension: str = ".txt",
    create_missing_captions: bool = True,
    caption_strategy: str = "folder_name",  # "folder_name" or "empty"
    batch_size: int = 1,
    enable_bucket: bool = False,
    bucket_no_upscale: bool = False,
    cache_directory_name: str = "cache_dir",
    num_frames: int = 33,
    frame_extraction: str = "head",
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 129,
    source_fps: float = None,
    target_fps: float = 25.0,
) -> Tuple[Dict, List[str]]:
    """
    Generate LTX-2 dataset configuration from folder structure.
    Handles both images (single-frame) and videos.

    Differences from the WAN generator:
    - target_frames are rounded DOWN to the nearest 8*k + 1 (LTX-2 VAE rule)
    - target_fps is written (LTX-2 resamples videos to this rate; default 25)

    Returns:
        Tuple of (config_dict, messages_list)
    """
    messages = []

    parent_folder = normalize_path(parent_folder)

    if not os.path.exists(parent_folder):
        raise ValueError(f"Parent folder does not exist: {parent_folder}")

    if not os.path.isdir(parent_folder):
        raise ValueError(f"Path is not a directory: {parent_folder}")

    subdirs = [d for d in os.listdir(parent_folder)
               if os.path.isdir(os.path.join(parent_folder, d))
               and not d.startswith('.')]

    if not subdirs:
        raise ValueError(f"No subdirectories found in: {parent_folder}")

    subdirs.sort()

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

    normalized_frames = round_frames_to_ltx2(num_frames)
    if normalized_frames != num_frames:
        messages.append(
            f"[INFO] Target frames adjusted from {num_frames} to {normalized_frames} (LTX-2 requires 8*k+1: 1, 9, 17, 25, 33, 41, 49, ...)"
        )

    for subdir in subdirs:
        subdir_path = os.path.join(parent_folder, subdir)

        has_subdirs = any(os.path.isdir(os.path.join(subdir_path, item))
                         for item in os.listdir(subdir_path)
                         if not item.startswith('.') and item not in [cache_directory_name])

        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct media files are supported)")
            continue

        image_files, video_files = get_media_files(subdir_path)
        all_media_files = image_files + video_files

        if not all_media_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No image or video files found")
            continue

        repeat_count, clean_name = extract_repeat_count(subdir)

        has_videos = len(video_files) > 0
        has_images = len(image_files) > 0

        directory_type = "video_directory" if has_videos else "image_directory"
        media_type = "videos" if has_videos else "images"

        if create_missing_captions:
            caption_content = ""
            if caption_strategy == "folder_name":
                caption_content = clean_name

            created = create_caption_files(
                all_media_files,
                caption_extension,
                caption_content
            )

            if created > 0:
                messages.append(f"[OK] Created {created} caption files for '{subdir}' with content: '{caption_content}'")

        missing_captions = []
        for media_file in all_media_files:
            caption_file = os.path.splitext(media_file)[0] + caption_extension
            if not os.path.exists(caption_file):
                missing_captions.append(os.path.basename(media_file))

        if missing_captions:
            messages.append(f"[WARNING] '{subdir}': {len(missing_captions)} media files missing caption files")

        dataset_entry = {
            directory_type: validate_path_for_toml(subdir_path),
            "num_repeats": repeat_count
        }

        if has_videos:
            dataset_entry["target_frames"] = [normalized_frames]
            dataset_entry["frame_extraction"] = frame_extraction
            if frame_extraction == "slide":
                dataset_entry["frame_stride"] = frame_stride
            if frame_extraction == "uniform":
                dataset_entry["frame_sample"] = frame_sample
            dataset_entry["max_frames"] = max_frames
            if source_fps is not None and source_fps > 0:
                dataset_entry["source_fps"] = source_fps
            if target_fps is not None and target_fps > 0:
                dataset_entry["target_fps"] = target_fps

        if cache_directory_name:
            if os.path.isabs(cache_directory_name):
                cache_path = os.path.join(cache_directory_name, subdir)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)
            else:
                cache_path = os.path.join(subdir_path, cache_directory_name)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)

        config["datasets"].append(dataset_entry)

        media_info = []
        if has_images:
            media_info.append(f"{len(image_files)} images")
        if has_videos:
            media_info.append(f"{len(video_files)} videos")
        messages.append(f"[OK] Added {subdir} ({', '.join(media_info)}) as {media_type} dataset")

    if not config["datasets"]:
        raise ValueError("No valid datasets found in the provided folder structure")

    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")

    return config, messages


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
    qwen_image_edit_no_resize_control: bool = False,
    no_resize_control: bool = False,
    control_resolution: Optional[Tuple[int, int]] = None,
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
            
            # Control image resizing options (shared by FLUX.2, FLUX.1 Kontext, Qwen-Image-Edit, etc.)
            if qwen_image_edit_no_resize_control or no_resize_control:
                dataset_entry["no_resize_control"] = True

            if control_resolution is not None:
                try:
                    cw, ch = int(control_resolution[0]), int(control_resolution[1])
                    if cw > 0 and ch > 0:
                        dataset_entry["control_resolution"] = [cw, ch]
                except Exception:
                    # Keep generation resilient: invalid values should not crash dataset generation.
                    pass
            
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
    
    # Remove trailing commas from arrays (cosmetic improvement for TOML spec compliance)
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace trailing commas in arrays: [ item, ] -> [ item]
    content = re.sub(r',(\s*])', r'\1', content)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


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
        
        # Validate each dataset (image, video, or audio datasets are all valid)
        for i, dataset in enumerate(config["datasets"]):
            media_key = next(
                (key for key in ("image_directory", "video_directory", "audio_directory") if key in dataset),
                None,
            )
            if media_key is None:
                if any(key in dataset for key in ("image_jsonl_file", "video_jsonl_file", "audio_jsonl_file")):
                    continue
                messages.append(f"[ERROR] Dataset {i+1}: Missing image_directory / video_directory / audio_directory")
                continue

            if not os.path.exists(dataset[media_key]):
                messages.append(f"[WARNING] Dataset {i+1}: {media_key} does not exist: {dataset[media_key]}")

            if "control_directory" in dataset and not os.path.exists(dataset["control_directory"]):
                messages.append(f"[WARNING] Dataset {i+1}: Control directory does not exist: {dataset['control_directory']}")
        
        messages.append(f"[OK] Configuration validated: {len(config['datasets'])} datasets")
        return True, messages
        
    except Exception as e:
        messages.append(f"[ERROR] Error validating configuration: {str(e)}")
        return False, messages
