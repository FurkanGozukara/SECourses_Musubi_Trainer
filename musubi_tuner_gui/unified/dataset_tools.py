"""Dataset helpers for the unified UI.

* ``prepare_dataset`` makes "Generate from Folder Structure" work on Start for every trainer. Most
  trainers build the dataset TOML themselves when training starts; the classic Qwen Image and Wan
  tabs need a manual "Generate Dataset Configuration" click first, so that generation (ported from
  those tabs' handlers) runs here before their train function is called.
  Every train function reuses ``generated_toml_path`` when it exists, so the TOML made here (with the
  trainer's own generator, for its frame / fps / bucket rules) is the one the trainer uses. On top:
  - a folder with both images and videos also trains its images (Wan, LTX 2.3 video mode, Kandinsky 6):
    the classic Wan generator does this itself, the LTX and Kandinsky 6 generators keep only the
    videos (LTX cannot know the mode; AV mode must stay video-only). MiniMax H3 trains on videos only.
  - LTX 2.3 audio mode gets ``audio_directory`` datasets (the classic LTX generator only knows image
    and video folders), and videos with an audio track are used too: their audio is extracted to WAV
    in the cache folder (sub-folders elsewhere would make the video generators skip the folder).
* ``scan_dataset`` summarises a parent dataset folder for the "Check dataset" button.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from ..common_gui import validate_path_for_toml
from ..dataset_config_generator import (
    extract_repeat_count,
    generate_dataset_config_from_folders,
    generate_wan_dataset_config_from_folders,
    owned_subfolder_names,
    save_dataset_config,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv", ".wmv", ".gif")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma")  # musubi's audio loader list
FOLDER_MODE = "Generate from Folder Structure"


def _subfolders(parent: str) -> list[str]:
    return sorted(os.path.join(parent, name) for name in os.listdir(parent) if os.path.isdir(os.path.join(parent, name)))


def _write_missing_captions(parent: str, extension: str, strategy: str, media_extensions: tuple) -> int:
    created = 0
    for folder in _subfolders(parent):
        _repeats, name = extract_repeat_count(os.path.basename(folder))
        text = name if strategy == "folder_name" else ""
        for file_name in os.listdir(folder):
            path = os.path.join(folder, file_name)
            if os.path.isfile(path) and file_name.lower().endswith(media_extensions):
                caption = os.path.splitext(path)[0] + extension
                if not os.path.exists(caption):
                    with open(caption, "w", encoding="utf-8") as handle:
                        handle.write(text)
                    created += 1
    return created


def _output_path(values: dict, prefix: str) -> str:
    from ..common_gui import scriptdir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(values.get("output_dir") or "").strip()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, f"{prefix}_{stamp}.toml")
    folder = os.path.join(scriptdir, "dataset_tomls")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{prefix}_{stamp}.toml")


def _files(folder: str, extensions: tuple) -> list[str]:
    return sorted(name for name in os.listdir(folder) if name.lower().endswith(extensions) and os.path.isfile(os.path.join(folder, name)))


def _same_folder(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def add_images_of_mixed_folders(config: dict, messages: list[str]) -> int:
    """Video datasets whose folder also holds images: add an image dataset for those images.

    The classic generators make one dataset per folder and keep only the videos of a mixed folder.
    Image and video datasources only read their own file types, so both can point at the same
    folder; the image dataset gets its own cache folder.
    """
    added = 0
    datasets = config.get("datasets") or []
    for entry in list(datasets):
        folder = entry.get("video_directory")
        if not folder or not os.path.isdir(folder):
            continue
        if any(other.get("image_directory") and _same_folder(other["image_directory"], folder) for other in datasets):
            continue
        images = _files(folder, IMAGE_EXTENSIONS)
        if not images:
            continue
        image_entry = {"image_directory": entry["video_directory"], "num_repeats": entry.get("num_repeats", 1)}
        if entry.get("cache_directory"):
            image_entry["cache_directory"] = str(entry["cache_directory"]).rstrip("/\\") + "_images"
        datasets.append(image_entry)
        added += 1
        messages.append(f"[OK] '{os.path.basename(os.path.normpath(folder))}': its {len(images)} images train too (image dataset beside the videos)")
    return added


def _rewrite_dataset_toml(path: str, edit) -> list[str]:
    """Load a generated dataset TOML, let ``edit(config, messages)`` change it, save and validate it."""
    import toml

    from ..dataset_config_generator import validate_dataset_config

    with open(path, "r", encoding="utf-8") as handle:
        config = toml.load(handle)
    messages: list[str] = []
    if edit(config, messages):
        save_dataset_config(config, path)
        valid, notes = validate_dataset_config(path)
        if not valid:
            raise ValueError("The dataset TOML is not valid:\n" + "\n".join(notes))
    return messages


def _qwen_dataset(values: dict) -> tuple[str, list[str]]:
    parent = str(values.get("parent_folder_path") or "").strip()
    extension = str(values.get("dataset_caption_extension") or ".txt")
    messages = []
    if values.get("create_missing_captions"):
        created = _write_missing_captions(parent, extension, str(values.get("caption_strategy") or "folder_name"), IMAGE_EXTENSIONS)
        if created:
            messages.append(f"[OK] Created {created} missing caption file(s)")
    control_dir = qwen_control_directory(values)
    width_c = int(values.get("dataset_qwen_image_edit_control_resolution_width") or 0)
    height_c = int(values.get("dataset_qwen_image_edit_control_resolution_height") or 0)
    if values.get("auto_generate_black_control_images"):
        from PIL import Image

        for folder in _subfolders(parent):
            control_folder = os.path.join(folder, control_dir)
            os.makedirs(control_folder, exist_ok=True)
            for file_name in os.listdir(folder):
                path = os.path.join(folder, file_name)
                if os.path.isfile(path) and file_name.lower().endswith(IMAGE_EXTENSIONS):
                    target = os.path.join(control_folder, f"{os.path.splitext(file_name)[0]}_0.png")
                    if not os.path.exists(target):
                        Image.new("RGB", (width_c or 1024, height_c or 1024), (0, 0, 0)).save(target, "PNG")
    no_resize = bool(values.get("dataset_qwen_image_edit_no_resize_control"))
    config, notes = generate_dataset_config_from_folders(
        parent_folder=parent,
        resolution=(int(values.get("dataset_resolution_width") or 1024), int(values.get("dataset_resolution_height") or 1024)),
        caption_extension=extension,
        create_missing_captions=False,
        caption_strategy="folder_name",
        batch_size=int(values.get("dataset_batch_size") or 1),
        enable_bucket=bool(values.get("dataset_enable_bucket")),
        bucket_no_upscale=bool(values.get("dataset_bucket_no_upscale")),
        cache_directory_name=str(values.get("dataset_cache_directory") or "cache_dir"),
        control_directory_name=control_dir,
        qwen_image_edit_no_resize_control=no_resize,
    )
    if not config or not config.get("datasets"):
        raise ValueError("Could not build a dataset from the parent folder. Check the folder structure.\n" + "\n".join(notes))
    if not no_resize and width_c > 0 and height_c > 0:
        for dataset in config.get("datasets", []):
            if "control_directory" in dataset:
                dataset["control_resolution"] = [width_c, height_c]
    path = _output_path(values, "dataset_config")
    save_dataset_config(config, path)
    return path, messages + list(notes)


def _wan_dataset(values: dict) -> tuple[str, list[str]]:
    from ..wan_lora_gui import find_min_video_frames_in_wan_dataset, round_frames_to_n4plus1

    parent = str(values.get("parent_folder_path") or "").strip()
    extension = str(values.get("dataset_caption_extension") or ".txt")
    cache_dir = str(values.get("dataset_cache_directory") or "cache_dir")
    messages = []
    requested = max(1, int(values.get("target_frames") or 81))
    frames = round_frames_to_n4plus1(requested)
    if frames != requested:
        messages.append(f"[INFO] Target frames rounded to Wan's 4n+1 rule: {requested} -> {frames}")
    source_fps = float(values.get("source_fps") or 0) or None
    if values.get("auto_normalize_target_frames"):
        shortest, shortest_path, scanned = find_min_video_frames_in_wan_dataset(
            parent_folder=parent, cache_directory_name=cache_dir, source_fps=source_fps, early_stop_at=frames
        )
        if scanned and shortest is not None and round_frames_to_n4plus1(shortest) < frames:
            frames = round_frames_to_n4plus1(shortest)
            messages.append(f"[OK] Target frames lowered to {frames} to fit the shortest clip ({os.path.basename(shortest_path or '')})")
    if values.get("create_missing_captions"):
        created = _write_missing_captions(parent, extension, str(values.get("caption_strategy") or "folder_name"), IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)
        if created:
            messages.append(f"[OK] Created {created} missing caption file(s)")
    config, notes = generate_wan_dataset_config_from_folders(
        parent_folder=parent,
        resolution=(int(values.get("dataset_resolution_width") or 960), int(values.get("dataset_resolution_height") or 544)),
        caption_extension=extension,
        create_missing_captions=False,
        caption_strategy="folder_name",
        batch_size=int(values.get("dataset_batch_size") or 1),
        enable_bucket=bool(values.get("dataset_enable_bucket")),
        bucket_no_upscale=bool(values.get("dataset_bucket_no_upscale")),
        cache_directory_name=cache_dir,
        num_frames=int(frames),
        frame_extraction=str(values.get("frame_extraction") or "head"),
        frame_stride=int(values.get("frame_stride") or 1),
        frame_sample=int(values.get("frame_sample") or 1),
        max_frames=int(values.get("max_frames") or 129),
        source_fps=source_fps,
    )
    if not config or not config.get("datasets"):
        raise ValueError("Could not build a Wan dataset from the parent folder. Check the folder structure.\n" + "\n".join(notes))
    notes = list(notes)
    add_images_of_mixed_folders(config, notes)
    path = _output_path(values, "wan_dataset_config")
    save_dataset_config(config, path)
    values["target_frames"] = int(frames)
    return path, messages + list(notes)


EXTRACTED_AUDIO_DIRNAME = "extracted_audio"
EXTRACTED_SOURCES_FILENAME = "sources.json"  # size and modification time of the video each WAV came from


def has_audio_track(video_path: str) -> bool:
    import av

    try:
        with av.open(video_path) as container:
            return any(stream.type == "audio" for stream in container.streams)
    except (OSError, av.FFmpegError, ValueError):
        return False


def extract_audio_track(video_path: str, target_path: str) -> bool:
    """Write the first audio track of ``video_path`` as 16-bit PCM WAV (original rate and channels).
    False when the video has no audio track. Written under a temporary name that is not an audio file
    name (a failed extraction must never become a training clip), then renamed; removed on failure."""
    import av

    partial = target_path + ".partial"
    try:
        with av.open(video_path) as source:
            stream = next((item for item in source.streams if item.type == "audio"), None)
            if stream is None:
                return False
            rate = stream.rate or stream.codec_context.sample_rate or 48000
            layout = stream.layout.name if stream.layout else ("mono" if stream.channels == 1 else "stereo")
            resampler = av.AudioResampler(format="s16", layout=layout, rate=rate)
            with av.open(partial, "w", format="wav") as target:
                output = target.add_stream("pcm_s16le", rate=rate, layout=layout)
                for frame in source.decode(stream):
                    for converted in resampler.resample(frame):
                        for packet in output.encode(converted):
                            target.mux(packet)
                for converted in resampler.resample(None):
                    for packet in output.encode(converted):
                        target.mux(packet)
                for packet in output.encode(None):
                    target.mux(packet)
        os.replace(partial, target_path)
        return True
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass


def _cache_root(folder: str, cache_name: str) -> str:
    return os.path.join(cache_name, os.path.basename(folder)) if os.path.isabs(cache_name) else os.path.join(folder, cache_name)


def _read_sources(folder: str) -> dict:
    try:
        with open(os.path.join(folder, EXTRACTED_SOURCES_FILENAME), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_sources(folder: str, sources: dict) -> None:
    path = os.path.join(folder, EXTRACTED_SOURCES_FILENAME)
    with open(path + ".partial", "w", encoding="utf-8") as handle:
        json.dump(sources, handle, indent=1, ensure_ascii=False)
    os.replace(path + ".partial", path)


def _write_if_changed(path: str, content: bytes) -> None:
    try:
        with open(path, "rb") as handle:
            if handle.read() == content:
                return
    except OSError:
        pass
    with open(path, "wb") as handle:
        handle.write(content)


def _prune(folder: str, keep: set) -> None:
    """Delete the files of ``folder`` (our own extraction folder) that this run does not train on."""
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name not in keep and entry.is_file(follow_symlinks=False) and not entry.is_symlink():
                try:
                    os.remove(entry.path)
                except OSError:
                    pass


def audio_from_videos(folder: str, cache_name: str, extension: str, caption_text: str | None, messages: list[str]) -> tuple[str | None, int]:
    """Extract the audio of every captioned video in ``folder`` into ``<cache>/extracted_audio``, with
    the video's caption (``caption_text`` for videos without one, when missing captions are created).

    Each WAV is named after the whole video file name (``clip.mp4`` and ``clip.mov`` stay apart) and
    is reused while its video keeps the same size and modification time. Files of removed, replaced
    or uncaptioned videos are deleted, so they never train again. Returns (folder, clip count)."""
    from .runner import RUNNER

    videos = _files(folder, VIDEO_EXTENSIONS)
    target_dir = os.path.join(_cache_root(folder, cache_name), EXTRACTED_AUDIO_DIRNAME)
    if not videos and not os.path.isdir(target_dir):
        return None, 0
    label = os.path.basename(folder)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        messages.append(f"[WARNING] '{label}': cannot create the folder for the extracted audio ({target_dir}): {exc}")
        return None, 0
    sources = _read_sources(target_dir)
    kept: dict = {}
    silent, failed, uncaptioned = [], [], []
    for name in videos:
        RUNNER.raise_if_cancelled()  # extraction can take minutes: Stop ends the run here
        video = os.path.join(folder, name)
        caption_source = os.path.splitext(video)[0] + extension
        wav_name = name + ".wav"
        try:
            if os.path.isfile(caption_source):
                with open(caption_source, "rb") as handle:
                    caption = handle.read()
            elif caption_text is not None:
                caption = caption_text.encode("utf-8")
            else:
                uncaptioned.append(name)  # musubi reads a caption for every clip
                continue
            stat = os.stat(video)
            signature = [stat.st_size, stat.st_mtime_ns]
            wav = os.path.join(target_dir, wav_name)
            if not (os.path.isfile(wav) and sources.get(wav_name) == signature):
                if not extract_audio_track(video, wav):
                    silent.append(name)
                    continue
            _write_if_changed(os.path.join(target_dir, name + extension), caption)
        except Exception as exc:  # unreadable or exotic container: skip that clip, keep the rest
            failed.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        kept[wav_name] = signature
    _prune(target_dir, set(kept) | {wav_name[: -len(".wav")] + extension for wav_name in kept} | {EXTRACTED_SOURCES_FILENAME})
    _write_sources(target_dir, kept)
    extracted = len(kept)
    if extracted:
        messages.append(f"[OK] '{label}': using the audio track of {extracted} video(s) (extracted to {target_dir})")
    if silent:
        messages.append(f"[WARNING] '{label}': {len(silent)} video(s) have no audio track and are skipped: {', '.join(silent[:5])}")
    if uncaptioned:
        messages.append(
            f"[WARNING] '{label}': {len(uncaptioned)} video(s) have no {extension} caption and are skipped "
            f"(turn on Create Missing Captions to use them): {', '.join(uncaptioned[:5])}"
        )
    if failed:
        messages.append(f"[WARNING] '{label}': could not read the audio of {len(failed)} video(s), skipped: {'; '.join(failed[:3])}")
    return (target_dir if extracted else None), extracted


def _ltx2_audio_dataset(values: dict) -> tuple[str, list[str]]:
    """Audio-only datasets: audio clips as they are, plus the audio tracks of videos."""
    parent = str(values.get("parent_folder_path") or "").strip()
    extension = str(values.get("dataset_caption_extension") or ".txt")
    cache_name = str(values.get("dataset_cache_directory") or "cache_dir")
    create_missing = bool(values.get("create_missing_captions"))
    strategy = str(values.get("caption_strategy") or "folder_name")
    messages = []
    if create_missing:
        created = _write_missing_captions(parent, extension, strategy, AUDIO_EXTENSIONS)
        if created:
            messages.append(f"[OK] Created {created} missing caption file(s)")
    datasets, uncaptioned = [], []
    for folder in _subfolders(parent):
        name = os.path.basename(folder)
        if name.startswith(".") or name == cache_name:
            continue
        repeats, clean = extract_repeat_count(name)
        cache = _cache_root(folder, cache_name)
        clips = _files(folder, AUDIO_EXTENSIONS)
        if clips:
            datasets.append({"audio_directory": validate_path_for_toml(folder), "cache_directory": validate_path_for_toml(cache), "num_repeats": repeats})
            missing = [clip for clip in clips if not os.path.exists(os.path.splitext(os.path.join(folder, clip))[0] + extension)]
            if missing:  # musubi reads a caption for every clip of an audio folder and stops at the first missing one
                uncaptioned.append(f"'{name}': {', '.join(missing[:5])}{' …' if len(missing) > 5 else ''}")
            messages.append(f"[OK] Added {name} ({len(clips)} audio clips) as an audio dataset with num_repeats={repeats}")
        caption_text = (clean if strategy == "folder_name" else "") if create_missing else None
        extracted_dir, count = audio_from_videos(folder, cache_name, extension, caption_text, messages)
        if extracted_dir:
            datasets.append(
                {
                    "audio_directory": validate_path_for_toml(extracted_dir),
                    "cache_directory": validate_path_for_toml(os.path.join(cache, EXTRACTED_AUDIO_DIRNAME + "_cache")),
                    "num_repeats": repeats,
                }
            )
        if not clips and not count:
            messages.append(f"[WARNING] Skipping '{name}': no audio files and no videos with an audio track")
    if uncaptioned:
        raise ValueError(
            f"These audio clips have no {extension} caption file, which musubi needs for every clip: "
            + "; ".join(uncaptioned)
            + ". Add the captions, or turn on Create Missing Captions in 📁 Dataset."
        )
    if not datasets:
        problems = [line for line in messages if line.startswith("[WARNING]")]
        raise ValueError(
            f"No audio found in the sub-folders of {parent}. Put audio clips ({', '.join(AUDIO_EXTENSIONS)}) or videos with "
            "sound, each with a same-name caption file, in sub-folders such as 1_my_voice."
            + ("\n" + "\n".join(problems) if problems else "")
        )
    config = {
        "general": {"caption_extension": extension, "batch_size": int(values.get("dataset_batch_size") or 1)},
        "datasets": datasets,
    }
    path = _output_path(values, "ltx2_audio_dataset")
    save_dataset_config(config, path)
    return path, messages


def _ltx2_dataset(values: dict) -> tuple[str, list[str]]:
    mode = str(values.get("ltx2_mode") or "video").strip().lower()
    if mode == "audio":
        return _ltx2_audio_dataset(values)
    from ..ltx2_lora_gui import _create_ltx2_dataset_config

    path, messages = _create_ltx2_dataset_config(values)
    messages = list(messages)
    if mode == "video":
        # the classic generator does not know the mode and warns that images next to videos are unused;
        # in video mode they are trained (image batches would only dilute audio learning in AV mode)
        messages = [line for line in messages if "images are not used" not in line]
        messages += _rewrite_dataset_toml(path, add_images_of_mixed_folders)
    return path, messages


def _kandinsky6_dataset(values: dict) -> tuple[str, list[str]]:
    from ..image_models_gui import create_dataset_config, get_spec

    path, messages = create_dataset_config(get_spec("kandinsky6"), values)
    messages = [line for line in messages if "images in the same folder are ignored" not in line]
    messages += _rewrite_dataset_toml(path, add_images_of_mixed_folders)
    return path, messages


def qwen_control_directory(values: dict) -> str:
    """The Qwen Image control folder Start pairs with the targets (one rule for Start and the dataset check)."""
    return str(values.get("dataset_control_directory") or "control_images")


GENERATORS = {
    "qwen_image": _qwen_dataset,
    "wan": _wan_dataset,
    "ltx2": _ltx2_dataset,
    "kandinsky6": _kandinsky6_dataset,
}  # Z-Image builds its dataset TOML at Start itself (classic tab fix of 2026-10-11)


def _generator_for(backend: str, values: dict):
    return GENERATORS.get(backend)


def prepare_dataset(backend: str, values: dict) -> list[str]:
    """Generate the dataset TOML for trainers that do not do it on Start themselves (mutates values)."""
    generator = _generator_for(backend, values)
    if generator is None or str(values.get("dataset_config_mode") or "") != FOLDER_MODE:
        return []
    parent = str(values.get("parent_folder_path") or "").strip()
    if not parent:
        raise ValueError("Set the Parent Dataset Folder (📁 Dataset section) or switch the dataset mode to 'Use TOML File'.")
    if not os.path.isdir(parent):
        raise ValueError(f"The parent dataset folder does not exist: {parent}")
    path, messages = generator(values)
    values["dataset_config"] = path
    values["generated_toml_path"] = path
    return [f"[OK] Dataset TOML generated: {path}"] + messages


def dataset_profile(backend: str, values: dict) -> str:
    """What the selected trainer accepts: image, video (images + videos), video_only, av or audio."""
    if backend == "ltx2":
        mode = str(values.get("ltx2_mode") or "video").strip().lower()
        return {"audio": "audio", "av": "av"}.get(mode, "video")
    if backend in ("wan", "kandinsky6"):
        return "video"
    if backend == "minimax_h3":
        return "video_only"
    return "image"


def scan_dataset(
    parent: str,
    caption_extension: str = ".txt",
    control_directory: str = "",
    profile: str = "image",
    cache_directory: str = "cache_dir",
    allowed_folders: tuple = (),
    probe_audio: bool = True,
) -> str:
    """Markdown summary of a parent dataset folder (sub-folders, repeats, media and captions), with
    what the selected model will do with it. ``control_directory``: control images are on and every
    folder needs one; ``allowed_folders``: inner folders the trainer's generator accepts anyway;
    ``probe_audio``: open every video to tell whether it has sound (audio mode; the dataset check
    measures that itself, within its time limit)."""
    from .values import clean_path

    parent = clean_path(parent)
    if not parent:
        return "⚠️ Set the **Parent Dataset Folder** first."
    if not os.path.isdir(parent):
        return f"❌ Folder not found: `{parent}`"
    folders = [folder for folder in _subfolders(parent) if not os.path.basename(folder).startswith(".")]
    if not folders:
        return (
            f"⚠️ `{parent}` has no sub-folders. Put your media in sub-folders such as `1_ohwx man` "
            "(the optional number prefix is the repeat count)."
        )
    rows = ["| Folder | Repeats | Images | Videos | Audio | Captions | Missing captions |", "|---|---:|---:|---:|---:|---:|---:|"]
    totals = [0, 0, 0, 0, 0]
    mixed, image_only, nested, with_sound, silent = [], [], [], 0, 0
    owned = owned_subfolder_names(clean_path(cache_directory) or "cache_dir", control_directory, *allowed_folders)
    for folder in folders:
        repeats, _name = extract_repeat_count(os.path.basename(folder))
        # the image / video generators skip a folder that holds other folders (audio mode reads only the clips)
        inner = sorted(name for name in os.listdir(folder) if not name.startswith(".") and name not in owned and os.path.isdir(os.path.join(folder, name)))
        if inner and profile != "audio":
            nested.append(f"`{os.path.basename(folder)}` (has `{inner[0]}`{' …' if len(inner) > 1 else ''})")
            rows.append(f"| `{os.path.basename(folder)}` ⏭️ skipped | {repeats} | | | | | |")
            continue
        images = videos = audio = captions = missing = 0
        for file_name in os.listdir(folder):
            path = os.path.join(folder, file_name)
            if not os.path.isfile(path):
                continue
            lower = file_name.lower()
            is_image, is_video, is_audio = lower.endswith(IMAGE_EXTENSIONS), lower.endswith(VIDEO_EXTENSIONS), lower.endswith(AUDIO_EXTENSIONS)
            images += is_image
            videos += is_video
            audio += is_audio
            if lower.endswith(caption_extension.lower()):
                captions += 1
            if (is_image or is_video or is_audio) and not os.path.exists(os.path.splitext(path)[0] + caption_extension):
                missing += 1
            if is_video and profile == "audio" and probe_audio:
                if has_audio_track(path):
                    with_sound += 1
                else:
                    silent += 1
        if images and videos:
            mixed.append(os.path.basename(folder))
        if images and not videos:
            image_only.append(os.path.basename(folder))
        for index, value in enumerate((images, videos, audio, captions, missing)):
            totals[index] += value
        flag = " ⚠️" if missing else ""
        rows.append(f"| `{os.path.basename(folder)}` | {repeats} | {images} | {videos} | {audio} | {captions} | {missing}{flag} |")
    rows.append(f"| **Total** | | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** | **{totals[3]}** | **{totals[4]}** |")
    notes = []
    if nested:
        notes.append(
            f"⚠️ Start skips folders that contain other folders: {', '.join(nested[:6])}{' …' if len(nested) > 6 else ''}. "
            "Move those inner folders out of the dataset (only the cache and control-image folders may stay)."
        )
    if totals[4]:
        notes.append(f"⚠️ {totals[4]} file(s) have no `{caption_extension}` caption. Enable **Create Missing Captions** or caption them in 🧰 Tools → Image Captioning.")
    if control_directory:
        missing_control = [os.path.basename(folder) for folder in folders if not os.path.isdir(os.path.join(folder, control_directory))]
        if missing_control:
            notes.append(f"⚠️ No `{control_directory}` sub-folder in: " + ", ".join(f"`{name}`" for name in missing_control))
    names = lambda items: ", ".join(f"`{item}`" for item in items[:6]) + (" …" if len(items) > 6 else "")  # noqa: E731
    usable = totals[0]
    if profile == "image":
        if totals[1] or totals[2]:
            notes.append(f"ℹ️ This model trains on images: the {totals[1]} video(s) and {totals[2]} audio file(s) are not used.")
    elif profile == "video":
        usable = totals[0] + totals[1]
        if mixed:
            notes.append(f"✅ {names(mixed)} mix images and videos: both are trained (an image dataset is added beside the videos).")
    elif profile == "video_only":
        usable = totals[1]
        if image_only:
            notes.append(f"⚠️ This model trains on videos only; folders with only images are skipped: {names(image_only)}.")
    elif profile == "av":
        usable = totals[0] + totals[1]
        if mixed:
            notes.append(f"ℹ️ Audio + Video mode uses the videos of {names(mixed)}; the images next to them are not used.")
    elif profile == "audio":
        usable = totals[2] + (with_sound if probe_audio else totals[1])
        if with_sound:
            notes.append(f"🔊 {with_sound} video(s) have sound: Start extracts their audio (WAV, in the cache folder) and trains on it.")
        if silent:
            notes.append(f"ℹ️ {silent} video(s) have no audio track and are skipped.")
        if totals[0]:
            notes.append(f"ℹ️ Audio training does not use the {totals[0]} image(s).")
    if not usable:
        wanted = {"image": "images", "video": "images or videos", "video_only": "videos", "av": "videos with sound", "audio": "audio clips or videos with sound"}[profile]
        notes.append(f"❌ Nothing this model can train on was found: add {wanted}.")
    if not any(note.startswith(("⚠️", "❌")) for note in notes):
        notes.append("✅ Dataset looks good.")
    return "\n".join(rows) + "\n\n" + "\n\n".join(notes)
