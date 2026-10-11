import os
import re
import toml
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .common_gui import validate_path_for_toml, normalize_path, is_path_safe


# Kohya-style dataset folders carry the repeat count as a numeric prefix:
# "<repeats>_<name>" (e.g. "5_ohwx" = train images of "ohwx" 5 times per epoch).
REPEAT_PREFIX_PATTERN = re.compile(r'^(\d+)_(.+)$')


def parse_repeat_count(folder_name: str) -> Tuple[int, str, Optional[str]]:
    """
    Parse the repeat-count prefix from a dataset folder name.

    Dataset folders should be named "<repeats>_<name>" (e.g. "5_ohwx").
    This parser is deliberately forgiving: a folder that does not follow the
    convention still trains with num_repeats defaulting to 1, and a
    user-facing notice explains what happened so the omission is never silent.

    Returns:
        Tuple of (repeat_count, clean_name, notice)
        repeat_count: parsed repeat count (always >= 1)
        clean_name: folder name with the prefix stripped (used for captions)
        notice: None for a well-formed name, otherwise a message to show
                to the user

    Examples:
        "3_ohwx"        -> (3, "ohwx", None)
        "10_my_dataset" -> (10, "my_dataset", None)
        "dataset"       -> (1, "dataset", "[INFO] ...")
        "0_ohwx"        -> (1, "ohwx", "[WARNING] ...")
    """
    name = str(folder_name) if folder_name is not None else ""
    match = REPEAT_PREFIX_PATTERN.match(name)

    if match:
        repeat_count = int(match.group(1))
        clean_name = match.group(2)
        if repeat_count < 1:
            return 1, clean_name, (
                f"[WARNING] Folder '{name}': repeat count 0 is invalid - a dataset with 0 repeats "
                f"would be skipped entirely during training. It was counted as 1 repeat instead. "
                f"Rename the folder to 'N_{clean_name}' with N >= 1 (e.g. '5_{clean_name}') to set a real repeat count."
            )
        return repeat_count, clean_name, None

    # Number-only names ("5") or a prefix with nothing after it ("5_") are
    # ambiguous - treat the whole name as the dataset name with 1 repeat.
    if re.fullmatch(r'\d+_?', name):
        return 1, name, (
            f"[INFO] Folder '{name}': name looks like a repeat count without a dataset name. "
            f"Expected format is 'N_name' (e.g. '{name.rstrip('_')}_myconcept'), so no repeat count "
            f"was applied - the folder was counted as 1 repeat (num_repeats = 1)."
        )

    return 1, name, (
        f"[INFO] Folder '{name}': you forgot the repeat-count prefix (expected format 'N_name', "
        f"e.g. '5_{name}'), so it was counted as 1 repeat (num_repeats = 1). Rename the folder to "
        f"e.g. '1_{name}' to set the repeat count explicitly and remove this message."
    )


def extract_repeat_count(folder_name: str) -> Tuple[int, str]:
    """
    Extract repeat count from folder name.

    Backward-compatible wrapper around parse_repeat_count(); use
    parse_repeat_count() when the user-facing notice is also needed.
    Examples:
        "3_ohwx" -> (3, "ohwx")
        "10_my_dataset" -> (10, "my_dataset")
        "dataset" -> (1, "dataset")
    """
    repeat_count, clean_name, _ = parse_repeat_count(folder_name)
    return repeat_count, clean_name


def owned_subfolder_names(cache_directory_name, *others) -> set:
    """Sub-folders the trainer itself creates or expects inside a dataset folder: the relative cache folder, the
    separate image cache of a mixed image/video folder ("<cache>_images") and e.g. the control-image folder.
    They must not make the folder look like a nested dataset ("Contains subdirectories") on the next run."""
    names = set()
    if cache_directory_name and not os.path.isabs(str(cache_directory_name)):
        name = os.path.basename(os.path.normpath(str(cache_directory_name)))
        names.update({name, name + "_images"})
    names.update(str(other) for other in others if other)
    return names


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
                         if not item.startswith('.') and item not in owned_subfolder_names(cache_directory_name))

        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct media files are supported)")
            continue

        # Get media files (both images and videos)
        image_files, video_files = get_media_files(subdir_path)
        all_media_files = image_files + video_files

        if not all_media_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No image or video files found")
            continue

        # Extract repeat count and clean name, informing the user when the
        # folder name deviates from the "N_name" convention
        repeat_count, clean_name, repeat_notice = parse_repeat_count(subdir)
        if repeat_notice:
            messages.append(repeat_notice)

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
        messages.append(f"[OK] Added {subdir} ({', '.join(media_info)}) as {media_type} dataset with num_repeats={repeat_count}")

        if has_videos and has_images:
            # a video dataset reads only the videos of its folder: give the images their own dataset, with its
            # own cache folder (training loads every cache file of a folder, so the two must not share one)
            image_entry = {"image_directory": validate_path_for_toml(subdir_path), "num_repeats": repeat_count}
            if "cache_directory" in dataset_entry:
                image_entry["cache_directory"] = validate_path_for_toml(cache_path + "_images")
            config["datasets"].append(image_entry)
            messages.append(f"[OK] Added the {len(image_files)} images of {subdir} as a separate image dataset")

    if not config["datasets"]:
        raise ValueError(_no_valid_datasets_message(parent_folder, messages))

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
                         if not item.startswith('.') and item not in owned_subfolder_names(cache_directory_name))

        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct media files are supported)")
            continue

        image_files, video_files = get_media_files(subdir_path)
        all_media_files = image_files + video_files

        if not all_media_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No image or video files found")
            continue

        repeat_count, clean_name, repeat_notice = parse_repeat_count(subdir)
        if repeat_notice:
            messages.append(repeat_notice)

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
        messages.append(f"[OK] Added {subdir} ({', '.join(media_info)}) as {media_type} dataset with num_repeats={repeat_count}")
        if has_videos and has_images:
            # audio-video training must stay video-only, so images in a video folder are not added here
            messages.append(
                f"[WARNING] '{subdir}': its {len(image_files)} images are not used (the folder also holds videos). "
                "Put images in a folder of their own to train on them in video-only mode."
            )

    if not config["datasets"]:
        raise ValueError(_no_valid_datasets_message(parent_folder, messages))

    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")

    return config, messages


def round_frames_to_minimax_h3(frames: int) -> int:
    """Round a frame count DOWN to the nearest valid MiniMax H3 value (17*n + 5).

    The MiniMax H3 packing requires frame counts of 5, 22, 39, 56, ..., and the
    released duration range is 124-345 frames (5-15 seconds at 24 fps).
    """
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        return 124
    if frames <= 5:
        return 5
    return ((frames - 5) // 17) * 17 + 5


MINIMAX_H3_RELEASED_MIN_FRAMES = 124
MINIMAX_H3_RELEASED_MAX_FRAMES = 345


def generate_minimax_h3_dataset_config_from_folders(
    parent_folder: str,
    resolution: Tuple[int, int],
    caption_extension: str = ".txt",
    create_missing_captions: bool = True,
    caption_strategy: str = "folder_name",  # "folder_name" or "empty"
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
    cache_directory_name: str = "cache_dir",
    num_frames: int = 124,
    frame_extraction: str = "head",
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = MINIMAX_H3_RELEASED_MAX_FRAMES,
    allow_experimental_duration: bool = False,
) -> Tuple[Dict, List[str]]:
    """
    Generate a MiniMax H3 dataset configuration from folder structure (videos only).

    Differences from the LTX-2/WAN generators:
    - target_frames are rounded DOWN to the nearest 17*n + 5 (H3 packing rule)
    - batch_size is always 1 (hard requirement of the architecture)
    - no source_fps / target_fps: H3 always normalizes videos to 24 fps from timestamps
    - image-only folders are skipped (H3 trains on video targets)
    - resolution must be a multiple of 32 in both dimensions

    Returns:
        Tuple of (config_dict, messages_list)
    """
    messages = []

    parent_folder = normalize_path(parent_folder)

    if not os.path.exists(parent_folder):
        raise ValueError(f"Parent folder does not exist: {parent_folder}")

    if not os.path.isdir(parent_folder):
        raise ValueError(f"Path is not a directory: {parent_folder}")

    width, height = int(resolution[0]), int(resolution[1])
    if width % 32 or height % 32 or width <= 0 or height <= 0:
        raise ValueError(f"MiniMax H3 resolution must be positive multiples of 32, got {width}x{height}")

    subdirs = [d for d in os.listdir(parent_folder)
               if os.path.isdir(os.path.join(parent_folder, d))
               and not d.startswith('.')]

    if not subdirs:
        raise ValueError(f"No subdirectories found in: {parent_folder}")

    subdirs.sort()

    config = {
        "general": {
            "resolution": [width, height],
            "caption_extension": caption_extension,
            "batch_size": 1,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale
        },
        "datasets": []
    }

    normalized_frames = round_frames_to_minimax_h3(num_frames)
    if normalized_frames != num_frames:
        messages.append(
            f"[INFO] Target frames adjusted from {num_frames} to {normalized_frames} (MiniMax H3 requires 17*n+5: 5, 22, 39, ..., 124, 141, ...)"
        )
    if normalized_frames < MINIMAX_H3_RELEASED_MIN_FRAMES:
        if allow_experimental_duration:
            messages.append(
                f"[INFO] Target frames {normalized_frames} is below the released minimum {MINIMAX_H3_RELEASED_MIN_FRAMES} "
                f"(5 s at 24 fps); latent caching will run with --allow_experimental_duration."
            )
        else:
            messages.append(
                f"[WARNING] Target frames {normalized_frames} is below the released minimum {MINIMAX_H3_RELEASED_MIN_FRAMES} "
                f"(5 s at 24 fps). Enable 'Allow Experimental Duration' or caching will fail."
            )
    if normalized_frames > MINIMAX_H3_RELEASED_MAX_FRAMES:
        messages.append(
            f"[WARNING] Target frames {normalized_frames} exceeds the released maximum {MINIMAX_H3_RELEASED_MAX_FRAMES} (15 s at 24 fps)."
        )

    for subdir in subdirs:
        subdir_path = os.path.join(parent_folder, subdir)

        has_subdirs = any(os.path.isdir(os.path.join(subdir_path, item))
                         for item in os.listdir(subdir_path)
                         if not item.startswith('.') and item not in owned_subfolder_names(cache_directory_name))

        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct video files are supported)")
            continue

        image_files, video_files = get_media_files(subdir_path)

        if not video_files:
            if image_files:
                messages.append(
                    f"[WARNING] Skipping '{subdir}': Only images found - MiniMax H3 trains on video targets "
                    f"(frame counts 17*n+5); single images cannot satisfy the packing rule yet."
                )
            else:
                messages.append(f"[WARNING] Skipping '{subdir}': No video files found")
            continue

        repeat_count, clean_name, repeat_notice = parse_repeat_count(subdir)
        if repeat_notice:
            messages.append(repeat_notice)

        if create_missing_captions:
            caption_content = ""
            if caption_strategy == "folder_name":
                caption_content = clean_name

            created = create_caption_files(
                video_files,
                caption_extension,
                caption_content
            )

            if created > 0:
                messages.append(f"[OK] Created {created} caption files for '{subdir}' with content: '{caption_content}'")

        missing_captions = []
        for media_file in video_files:
            caption_file = os.path.splitext(media_file)[0] + caption_extension
            if not os.path.exists(caption_file):
                missing_captions.append(os.path.basename(media_file))

        if missing_captions:
            messages.append(f"[WARNING] '{subdir}': {len(missing_captions)} videos missing caption files")

        dataset_entry = {
            "video_directory": validate_path_for_toml(subdir_path),
            "num_repeats": repeat_count,
            "target_frames": [normalized_frames],
            "frame_extraction": frame_extraction,
        }
        if frame_extraction == "slide":
            dataset_entry["frame_stride"] = frame_stride
        if frame_extraction == "uniform":
            dataset_entry["frame_sample"] = frame_sample
        dataset_entry["max_frames"] = max_frames

        if cache_directory_name:
            if os.path.isabs(cache_directory_name):
                cache_path = os.path.join(cache_directory_name, subdir)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)
            else:
                cache_path = os.path.join(subdir_path, cache_directory_name)
                dataset_entry["cache_directory"] = validate_path_for_toml(cache_path)

        config["datasets"].append(dataset_entry)

        messages.append(f"[OK] Added {subdir} ({len(video_files)} videos) as video dataset with num_repeats={repeat_count}")

    if not config["datasets"]:
        raise ValueError(_no_valid_datasets_message(parent_folder, messages))

    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")

    return config, messages


def round_frames_to_4n_plus_1(frames: int) -> int:
    """Round a frame count DOWN to the nearest 4*n + 1 (Kandinsky 6 / HunyuanVideo VAE grid)."""
    try:
        frames = int(frames)
    except (TypeError, ValueError):
        return 121
    if frames <= 1:
        return 1
    return ((frames - 1) // 4) * 4 + 1


def generate_kandinsky6_dataset_config_from_folders(
    parent_folder: str,
    resolution: Tuple[int, int],
    caption_extension: str = ".txt",
    create_missing_captions: bool = True,
    caption_strategy: str = "folder_name",  # "folder_name" or "empty"
    enable_bucket: bool = True,
    bucket_no_upscale: bool = False,
    cache_directory_name: str = "cache_dir",
    num_frames: int = 121,
    frame_extraction: str = "head",
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 241,
) -> Tuple[Dict, List[str]]:
    """Kandinsky 6 dataset from repeat-prefixed folders.

    Video folders become video datasets with ``target_frames`` rounded down to 4*n + 1 (24 fps,
    121 frames is about five seconds); audio comes from a same-stem sidecar or the video
    container. Image-only folders become image datasets, which Kandinsky 6 caches as one-frame
    videos with masked silence (appearance / identity LoRAs). ``batch_size`` is always 1.
    """
    messages: List[str] = []
    parent_folder = normalize_path(parent_folder)
    if not os.path.isdir(parent_folder):
        raise ValueError(f"Parent folder does not exist or is not a directory: {parent_folder}")
    width, height = int(resolution[0]), int(resolution[1])
    if width % 16 or height % 16 or width <= 0 or height <= 0:
        raise ValueError(f"Kandinsky 6 resolution must be positive multiples of 16, got {width}x{height}")

    subdirs = sorted(
        d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d)) and not d.startswith(".")
    )
    if not subdirs:
        raise ValueError(f"No subdirectories found in: {parent_folder}")

    config = {
        "general": {
            "resolution": [width, height],
            "caption_extension": caption_extension,
            "batch_size": 1,
            "enable_bucket": enable_bucket,
            "bucket_no_upscale": bucket_no_upscale,
        },
        "datasets": [],
    }
    frames = round_frames_to_4n_plus_1(num_frames)
    if frames != int(num_frames or 0):
        messages.append(f"[INFO] Target frames adjusted from {num_frames} to {frames} (Kandinsky 6 needs 4*n+1: 81, 97, 121, ...)")

    for subdir in subdirs:
        subdir_path = os.path.join(parent_folder, subdir)
        has_subdirs = any(
            os.path.isdir(os.path.join(subdir_path, item))
            for item in os.listdir(subdir_path)
            if not item.startswith(".") and item not in owned_subfolder_names(cache_directory_name)
        )
        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct media files are supported)")
            continue
        image_files, video_files = get_media_files(subdir_path)
        if not image_files and not video_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No images or videos found")
            continue
        repeat_count, clean_name, repeat_notice = parse_repeat_count(subdir)
        if repeat_notice:
            messages.append(repeat_notice)
        media_files = video_files if video_files else image_files
        if create_missing_captions:
            created = create_caption_files(media_files, caption_extension, clean_name if caption_strategy == "folder_name" else "")
            if created:
                messages.append(f"[OK] Created {created} caption files for '{subdir}'")
        missing = [f for f in media_files if not os.path.exists(os.path.splitext(f)[0] + caption_extension)]
        if missing:
            messages.append(f"[WARNING] '{subdir}': {len(missing)} files missing caption files")

        if video_files:
            if image_files:
                messages.append(f"[INFO] '{subdir}': using the {len(video_files)} videos; images in the same folder are ignored")
            entry = {
                "video_directory": validate_path_for_toml(subdir_path),
                "num_repeats": repeat_count,
                "target_frames": [frames],
                "frame_extraction": frame_extraction,
                "max_frames": max(int(max_frames or frames), frames),
            }
            if frame_extraction == "slide":
                entry["frame_stride"] = frame_stride
            if frame_extraction == "uniform":
                entry["frame_sample"] = frame_sample
            kind = f"{len(video_files)} videos"
        else:
            entry = {"image_directory": validate_path_for_toml(subdir_path), "num_repeats": repeat_count}
            kind = f"{len(image_files)} images (one-frame, silent)"
        if cache_directory_name:
            base = cache_directory_name if os.path.isabs(cache_directory_name) else subdir_path
            cache_path = os.path.join(base, subdir) if os.path.isabs(cache_directory_name) else os.path.join(base, cache_directory_name)
            entry["cache_directory"] = validate_path_for_toml(cache_path)
        config["datasets"].append(entry)
        messages.append(f"[OK] Added {subdir} ({kind}) with num_repeats={repeat_count}")

    if not config["datasets"]:
        raise ValueError(_no_valid_datasets_message(parent_folder, messages))
    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")
    return config, messages


def _no_valid_datasets_message(parent_folder: str, messages: list) -> str:
    """Explain *why* nothing usable was found (the per-folder warnings are otherwise never shown)."""
    reasons = [message for message in messages if message.startswith("[WARNING]")]
    text = f"No valid datasets found in {parent_folder}. Put your images (or videos) directly inside sub-folders such as 1_ohwx."
    if reasons:
        text += " Skipped:" + "".join(f"\n- {reason[len('[WARNING] '):]}" for reason in reasons[:10])
    return text


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
                         if not item.startswith('.') and item not in owned_subfolder_names(cache_directory_name, control_directory_name))
        
        if has_subdirs:
            messages.append(f"[WARNING] Skipping '{subdir}': Contains subdirectories (only direct image files are supported)")
            continue
        
        # Get image files
        image_files = get_image_files(subdir_path)
        
        if not image_files:
            messages.append(f"[WARNING] Skipping '{subdir}': No image files found")
            continue
        
        # Extract repeat count and clean name, informing the user when the
        # folder name deviates from the "N_name" convention
        repeat_count, clean_name, repeat_notice = parse_repeat_count(subdir)
        if repeat_notice:
            messages.append(repeat_notice)

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
        messages.append(f"[OK] Added {subdir} ({len(image_files)} images) with num_repeats={repeat_count}")

    if not config["datasets"]:
        raise ValueError(_no_valid_datasets_message(parent_folder, messages))
    
    messages.append(f"[OK] Generated configuration for {len(config['datasets'])} datasets")
    
    return config, messages


def dataset_toml_matches(toml_path: str, parent_folder: str, resolution=None) -> bool:
    """Whether a generated dataset TOML still describes this dataset folder (and resolution), so Start may reuse it.

    Start used to reuse the last generated TOML whenever the file existed, so after choosing another dataset folder
    or resolution without pressing Generate, the previous dataset trained. Hand-written TOMLs whose datasets all
    live inside the folder (e.g. audio datasets) are still reused.
    """
    try:
        data = toml.load(toml_path)
    except Exception:
        return False
    parent = normalize_path(parent_folder) if parent_folder else ""
    if not parent:
        return False
    if resolution:
        configured = (data.get("general") or {}).get("resolution")
        try:
            wanted = [int(value) for value in resolution]
            if configured is not None:
                have = [int(value) for value in (configured if isinstance(configured, (list, tuple)) else [configured, configured])]
                if have != wanted:
                    return False
        except (TypeError, ValueError):
            pass
    directories = [
        entry.get(key)
        for entry in data.get("datasets") or []
        for key in ("image_directory", "video_directory", "audio_directory")
        if entry.get(key)
    ]
    if not directories:
        return False
    parent_key = os.path.normcase(parent)
    for directory in directories:
        try:
            if os.path.commonpath([os.path.normcase(normalize_path(directory)), parent_key]) != parent_key:
                return False
        except ValueError:  # another drive
            return False
    return True


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
