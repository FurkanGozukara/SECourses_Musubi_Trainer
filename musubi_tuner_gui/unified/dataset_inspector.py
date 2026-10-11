"""Dataset inspector for the unified workspaces: statistics, problems and sample previews.

``inspect_dataset`` reproduces what the trainer will read and returns a ``DatasetReport``:
* musubi folder mode: the dataset generators' ``<repeats>_<name>`` folders (a folder holding another
  folder is skipped), then musubi's own file selection: its extension lists, no dot files, an image
  is trained only when a caption with exactly its stem exists, a video or audio clip without one
  stops caching, control images paired by musubi's rule (``photo.png`` -> ``photo.png``,
  ``photo_0.png``, ``photo_1.png``; any captioned target without one stops caching);
* a dataset TOML: its ``[[datasets]]`` (image, video or audio directory, control directory, repeats,
  caption extension, resolution) or kohya's ``[[datasets.subsets]]``;
* kohya folder mode: sd-scripts' rule (a folder counts when its name starts with a number >= 1 before
  the first ``_``; a file without caption takes the folder's class name; regularization images
  double the steps of an epoch).
``report_html`` renders stat cards, problems with example files, a folder table and distributions;
``sample_previews`` makes gallery thumbnails in a temporary folder (an edit pair side by side, an
image as it trains, a video's first frame, an audio waveform), each with its caption.
"""

from __future__ import annotations

import hashlib
import html
import math
import os
import random
import re
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache

from ..dataset_config_generator import extract_repeat_count, owned_subfolder_names
from .dataset_tools import AUDIO_EXTENSIONS, _subfolders, scan_dataset
from .values import clean_path

PREVIEW_DIRNAME = "secourses_dataset_previews"
MAX_PREVIEW_FILES = 400
PROBE_SECONDS = 6.0  # opening every file of a big dataset (size, length, readable to the end) stops after this
PREVIEW_SECONDS = 6.0
CAPTION_READ_LIMIT = 20000
EXAMPLES = 4
PROFILE_ITEMS = {  # media each profile trains on in folder mode (dataset_tools.dataset_profile)
    "image": ("image",),
    "video": ("video", "image"),
    "video_only": ("video",),
    "av": ("video",),
    "audio": ("audio", "video"),
}
FALLBACK_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".AVIF")
FALLBACK_VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg")


@lru_cache(maxsize=1)
def media_extensions() -> tuple:
    """musubi's own image and video extension lists (JPEG-XL only when its plugin is installed)."""
    try:
        from musubi_tuner.dataset.media_utils import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

        return tuple(IMAGE_EXTENSIONS), tuple(VIDEO_EXTENSIONS)
    except Exception:  # noqa: BLE001 - the lists below are a copy
        videos = FALLBACK_VIDEO_EXTENSIONS + tuple(extension.upper() for extension in FALLBACK_VIDEO_EXTENSIONS)
        return FALLBACK_IMAGE_EXTENSIONS, videos


def _ends_with(name: str, extensions) -> bool:
    """glob's matching: case-insensitive on Windows, exactly the listed spellings elsewhere."""
    if os.name == "nt":
        return name.lower().endswith(tuple(extension.lower() for extension in extensions))
    return name.endswith(tuple(extensions))


def _kind(name: str) -> str:
    images, videos = media_extensions()
    if _ends_with(name, images):
        return "image"
    if _ends_with(name, videos):
        return "video"
    if _ends_with(name, AUDIO_EXTENSIONS):
        return "audio"
    return ""


@dataclass
class MediaItem:
    path: str
    kind: str  # image | video | audio
    folder: str  # training folder name
    caption: str | None  # None: no caption file
    controls: list = field(default_factory=list)  # control image paths (edit datasets), in musubi's order
    used: bool = True  # the trainer reads it
    size: tuple | None = None  # (width, height) as stored (musubi ignores EXIF rotation)
    rotated: bool = False  # EXIF says the photo should be shown rotated
    duration: float | None = None  # seconds
    fps: float | None = None
    frames: int | None = None
    has_sound: bool | None = None
    error: str = ""


@dataclass
class FolderSummary:
    name: str
    path: str
    repeats: int
    images: int = 0
    videos: int = 0
    audio: int = 0
    captions: int = 0
    missing_captions: int = 0
    control_folder: bool = False
    paired: int = 0
    unpaired: int = 0
    unused_controls: int = 0
    regularization: bool = False
    skipped: str = ""  # why the trainer leaves this folder out


@dataclass
class DatasetReport:
    parent: str
    profile: str
    caption_extension: str
    control_name: str
    control_required: bool
    folders: list = field(default_factory=list)
    items: list = field(default_factory=list)
    problems: list = field(default_factory=list)  # (level "error" | "warn" | "info", text, example names)
    notes_markdown: str = ""  # what the model trains on (scan_dataset's notes in folder mode)
    probed: int = 0
    probe_total: int = 0
    batch_size: int = 1
    resolution: tuple | None = None
    kohya: bool = False
    unknown_sound: int = 0  # audio mode: videos the time limit left unchecked (counted as having sound)
    error: str = ""  # the folder itself is unusable (not set, missing)

    @property
    def used(self) -> list:
        return [item for item in self.items if item.used]

    @property
    def regularization_images(self) -> int:
        return sum(folder.images for folder in self.folders if folder.regularization and not folder.skipped)


@dataclass
class _Source:
    """One training folder: a <repeats>_<name> sub-folder, or one dataset of a dataset TOML."""

    summary: FolderSummary
    kinds: tuple  # media the trainer reads from this folder
    control_dir: str = ""  # its control-image folder ("" = none)
    control_named: bool = False  # a dataset TOML names that folder (it must exist)
    caption_extension: str = ".txt"
    class_captions: bool = False  # kohya: a file without caption takes the folder's class name


# ------------------------------------------------------------------------------------------------ control images
def match_controls(targets: list, controls: list) -> tuple[dict, list, list]:
    """musubi's pairing (image_video_dataset.ImageDirectoryDatasource): targets in order of decreasing file
    name length each take every remaining control named ``<stem>.`` or ``<stem>_``; after a ``_`` suffix only
    digits are allowed (``photo_0.png``; ``photo_a_0.png`` is fine too: only the text after the last ``_``
    counts) and ``pic.1.png`` stops caching. Returns (target -> controls in musubi's order, controls no
    target takes, controls musubi refuses). Linear in the number of files."""
    rank = {name: position for position, name in enumerate(sorted(targets, key=len, reverse=True))}
    by_stem: dict = {}
    for name in targets:
        by_stem.setdefault(os.path.splitext(name)[0], []).append(name)
    groups: dict = {}
    unused = []
    for control in controls:
        owner = None
        for index, char in enumerate(control):
            if char in "._" and index:
                for target in by_stem.get(control[:index], ()):
                    if owner is None or rank[target] < rank[owner]:
                        owner = target
        if owner is None:
            unused.append(control)
        else:
            groups.setdefault(owner, []).append(control)
    matches, rejected = {}, []
    for target, found in groups.items():
        stem = os.path.splitext(target)[0]
        ordered = []
        for control in found:
            control_stem = os.path.splitext(control)[0]
            if control_stem == stem:
                ordered.append((0, control))
                continue
            suffix = control_stem.rsplit("_", 1)[-1]
            if suffix.isdigit() and "_" in control_stem:
                ordered.append((int(suffix) + 1, control))
            else:
                rejected.append(control)
        if ordered:
            matches[target] = [control for _key, control in sorted(ordered)]
    return matches, sorted(unused), sorted(rejected)


def control_settings(backend: str, values: dict) -> tuple[str, bool]:
    """(control folder name, whether every image needs one) the way the selected model's dataset generator uses
    control images. "" when it takes none: the generators then treat a control folder like any other inner
    folder and skip the training folder that holds it (Krea 2, Ideogram 4, Z-Image pass a dummy name)."""
    from .dataset_tools import qwen_control_directory

    if backend == "qwen_image":  # paired whenever the folder exists; the Edit versions train with them
        return qwen_control_directory(values), str(values.get("model_version") or "").startswith("edit")
    if backend == "flux":  # FLUX.2: reference images whenever the folder exists
        return str(values.get("control_directory_name") or "control_images"), False
    if "use_control_images" in values:  # Qwen-Image 2.1, Mage-Flow (Edit), HiDream-O1: only with Use Control Images on
        if str(values.get("use_control_images")).strip().lower() not in ("true", "1", "yes", "on"):
            return "", False
        return str(values.get("control_directory_name") or "control_images"), True
    return "", False


# ------------------------------------------------------------------------------------------------ probing
def _read_caption(path: str) -> tuple[str | None, str]:
    """(caption, problem). musubi reads captions as strict UTF-8."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read(CAPTION_READ_LIMIT).strip(), ""
    except UnicodeDecodeError:
        return "", "is not UTF-8 text"
    except OSError as exc:
        return None, f"cannot be read ({exc.strerror or exc})"


def _probe_image(item: MediaItem) -> None:
    """Size as stored, EXIF rotation, and whether the file decodes (a cut-off JPEG stops musubi)."""
    from PIL import Image

    try:
        with Image.open(item.path) as image:
            item.size = image.size
            item.rotated = image.getexif().get(0x0112) in (5, 6, 7, 8)
            image_format = image.format
            if image_format == "JPEG":
                image.draft("RGB", (max(1, image.width // 8), max(1, image.height // 8)))  # decode at 1/8: fast
                image.load()
            elif image_format != "PNG":
                image.load()
        if image_format == "PNG":
            with Image.open(item.path) as image:
                image.verify()  # checks every chunk without decoding pixels (only right after opening)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is reported, never fatal
        item.error = f"cannot be read ({type(exc).__name__}: {str(exc)[:80]})"


def _probe_av(item: MediaItem) -> None:
    """Length, frame rate, frames, sound; a video must decode at its start and near its end."""
    try:
        import av

        with av.open(item.path) as container:
            video = next((stream for stream in container.streams if stream.type == "video"), None)
            audio = next((stream for stream in container.streams if stream.type == "audio"), None)
            item.has_sound = audio is not None
            if container.duration:
                item.duration = container.duration / 1_000_000
            elif audio is not None and audio.duration and audio.time_base:
                item.duration = float(audio.duration * audio.time_base)
            if item.kind == "audio" and audio is None:
                item.error = "has no audio stream"
                return
            if item.kind != "video":
                return
            if video is None:
                item.error = "has no video stream"
                return
            item.size = (video.codec_context.width, video.codec_context.height)
            if video.average_rate:
                item.fps = float(video.average_rate)
            item.frames = video.frames or (round(item.duration * item.fps) if item.duration and item.fps else None)
            next(container.decode(video=0))
            if item.duration and item.duration > 0.5:
                container.seek(int(item.duration * 0.9 * 1_000_000))
                next(container.decode(video=0))
    except StopIteration:
        item.error = "has no frame to decode near its end (cut off?)"
    except Exception as exc:  # noqa: BLE001
        item.error = f"cannot be read ({type(exc).__name__}: {str(exc)[:80]})"


# ------------------------------------------------------------------------------------------------ inspection
def inspect_dataset(
    parent: str,
    *,
    profile: str = "image",
    caption_extension: str = ".txt",
    control_name: str = "control_images",
    control_required: bool = False,
    cache_directory: str = "cache_dir",
    create_missing_captions: bool = False,
    batch_size=1,
    resolution: tuple | None = None,
    dataset_toml: str = "",
    kohya: bool = False,
    regularization_folder: str = "",
    probe_seconds: float = PROBE_SECONDS,
) -> DatasetReport:
    """Folder mode (``parent``: <repeats>_<name> sub-folders), a dataset TOML (``dataset_toml``: musubi
    ``[[datasets]]`` or kohya ``[[datasets.subsets]]``) or kohya's folder rule (``kohya``)."""
    report = DatasetReport(clean_path(dataset_toml or parent), profile, str(caption_extension or ""), control_name, control_required)
    report.kohya = kohya
    try:
        report.batch_size = max(1, int(float(batch_size or 1)))
    except (TypeError, ValueError):
        report.batch_size = 1
    report.resolution = resolution
    if report.caption_extension and not report.caption_extension.startswith("."):
        report.problems.append(("error", f"Caption Extension '{report.caption_extension}' has no leading dot: the trainer would look for 'photo{report.caption_extension}'. Use '.{report.caption_extension}'.", []))
    if dataset_toml:
        sources = _toml_sources(report)
    elif kohya:
        sources = _kohya_sources(report, regularization_folder)
    else:
        sources = _folder_sources(report, cache_directory)
    if report.error:
        return report
    _inspect_sources(report, sources, create_missing_captions, probe_seconds)
    return report


def _folder_sources(report: DatasetReport, cache_directory: str) -> list:
    if not report.parent:
        report.error = "Set the Parent Dataset Folder first."
        return []
    if not os.path.isdir(report.parent):
        report.error = f"Folder not found: {report.parent}"
        return []
    profile, control_name = report.profile, report.control_name
    pairs_possible = profile == "image" and bool(control_name)
    # what the selected model trains on and which folders Start skips: one rule set (scan_dataset). Sound,
    # control folders and missing captions are measured below, with example files.
    try:
        report.notes_markdown = scan_dataset(
            report.parent,
            report.caption_extension,
            "",
            profile,
            cache_directory=cache_directory,
            allowed_folders=(control_name,) if pairs_possible else (),
            probe_audio=False,
        ).split("\n\n", 1)[-1]
    except OSError as exc:
        report.problems.append(("warn", f"Part of the dataset folder cannot be read: {exc.strerror or exc}", []))
    owned = owned_subfolder_names(clean_path(cache_directory) or "cache_dir", control_name if pairs_possible else "")
    try:
        folders = _subfolders(report.parent)
    except OSError as exc:
        report.error = f"The dataset folder cannot be read: {exc.strerror or exc}"
        return []
    sources = []
    for folder in folders:
        name = os.path.basename(folder)
        if name.startswith("."):
            continue
        summary = FolderSummary(name, folder, extract_repeat_count(name)[0])
        report.folders.append(summary)
        try:
            inner = [entry for entry in os.listdir(folder) if not entry.startswith(".") and entry not in owned and os.path.isdir(os.path.join(folder, entry))]
        except OSError as exc:
            summary.skipped = f"cannot be read ({exc.strerror or exc})"
            continue
        if inner and profile != "audio":  # the generators skip a folder that holds other folders
            summary.skipped = f"contains the folder {sorted(inner)[0]}"
            continue
        kinds = PROFILE_ITEMS.get(profile, ("image",))
        sources.append(_Source(summary, kinds, os.path.join(folder, control_name) if pairs_possible else "", caption_extension=report.caption_extension))
    return sources


def _kohya_repeats(name: str) -> int:
    """sd-scripts (config_util.extract_dreambooth_params): the number before the first "_", else ignored."""
    try:
        return int(name.split("_")[0])
    except ValueError:
        return 0


def _kohya_sources(report: DatasetReport, regularization_folder: str = "") -> list:
    """kohya sd-scripts reads each folder whose name starts with a number >= 1 before the first "_" (10_ohwx man,
    also 10); folders inside a training folder do not matter (its images are read without recursion)."""
    if not report.parent:
        report.error = "Set the Training Images Folder first."
        return []
    if not os.path.isdir(report.parent):
        report.error = f"Folder not found: {report.parent}"
        return []
    report.notes_markdown = (
        "ℹ️ kohya trains the images of every `<repeats>_<name>` folder (for example `10_ohwx man`); a file without "
        "caption takes the folder's class name (`ohwx man`)."
    )
    sources = []
    for root, regularization in ((report.parent, False), (clean_path(regularization_folder), True)):
        if not root:
            continue
        if not os.path.isdir(root):
            report.problems.append(("error", f"Regularization Images Folder not found: {root}", []))
            continue
        try:
            folders = _subfolders(root)
        except OSError as exc:
            report.problems.append(("warn", f"{root} cannot be read: {exc.strerror or exc}", []))
            continue
        for folder in folders:
            name = os.path.basename(folder)
            if name.startswith("."):
                continue
            repeats = _kohya_repeats(name)
            summary = FolderSummary(("reg: " if regularization else "") + name, folder, repeats, regularization=regularization)
            report.folders.append(summary)
            if repeats < 1:
                summary.skipped = "ignored by kohya: the name must start with the repeat count, e.g. 10_ohwx man"
                continue
            sources.append(_Source(summary, ("image",), caption_extension=report.caption_extension, class_captions=True))
    return sources


def _toml_sources(report: DatasetReport) -> list:
    import toml

    path = report.parent
    if not path:
        report.error = "Choose the dataset TOML file first."
        return []
    try:
        from .backends import read_preset_file

        data = read_preset_file(path)
    except FileNotFoundError:
        report.error = f"Dataset TOML not found: {path}"
        return []
    except (OSError, ValueError, toml.TomlDecodeError) as exc:
        report.error = f"The dataset TOML cannot be read: {exc}"
        return []
    if not isinstance(data, dict):
        report.error = f"{os.path.basename(path)} holds no settings table."
        return []
    general = data.get("general") if isinstance(data.get("general"), dict) else {}
    default_extension = str(general.get("caption_extension") or report.caption_extension)
    if general.get("batch_size"):
        try:
            report.batch_size = max(1, int(general["batch_size"]))
        except (TypeError, ValueError):
            pass
    report.resolution = _toml_resolution(general.get("resolution")) or report.resolution  # the TOML's own wins
    entries = []
    for dataset in data.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets")
        if isinstance(subsets, list):  # kohya: [[datasets.subsets]] image_dir
            entries.extend((dict(dataset, **subset), True) for subset in subsets if isinstance(subset, dict))
        else:
            entries.append((dataset, False))
    sources, jsonl, base = [], 0, os.path.dirname(path)
    for index, (entry, kohya) in enumerate(entries, start=1):
        kind_key = next((key for key in ("image_directory", "video_directory", "audio_directory", "image_dir") if entry.get(key)), None)
        if kind_key is None:
            jsonl += any(str(key).endswith("jsonl_file") for key in entry)
            continue
        kinds = {"video_directory": ("video",), "audio_directory": ("audio",)}.get(kind_key, ("image",))
        folder = _toml_path(entry[kind_key], base)
        try:
            repeats = int(entry.get("num_repeats") or 1)
        except (TypeError, ValueError):
            repeats = 1
            report.problems.append(("error", f"Dataset {index}: num_repeats '{entry.get('num_repeats')}' is not a whole number.", []))
        name = f"{index}: {os.path.basename(folder.rstrip(os.sep)) or folder} ({kinds[0]}s)"
        summary = FolderSummary(name, folder, repeats)
        report.folders.append(summary)
        if not os.path.isdir(folder):
            summary.skipped = f"folder not found: {folder}"
            continue
        control = entry.get("control_directory")
        extension = str(entry.get("caption_extension") or default_extension)
        sources.append(_Source(summary, kinds, _toml_path(control, base) if control else "", bool(control), extension, class_captions=kohya))
        if entry.get("resolution") and not report.resolution:
            report.resolution = _toml_resolution(entry.get("resolution"))
    if jsonl:
        report.problems.append(("info", f"{jsonl} dataset(s) list their files in a JSONL file; those are not inspected here.", []))
    report.notes_markdown = f"ℹ️ {len(sources)} dataset folder(s) from `{os.path.basename(path)}`."
    if not entries:
        report.error = f"{os.path.basename(path)} lists no datasets ([[datasets]] with image_directory / video_directory)."
    return sources


def _toml_path(value, base: str) -> str:
    folder = clean_path(str(value))
    return folder if os.path.isabs(folder) else os.path.normpath(os.path.join(base, folder))


def _toml_resolution(value) -> tuple | None:
    if isinstance(value, (list, tuple)) and value:
        return tuple(value)
    if isinstance(value, (int, float)) and value:
        return (value, value)
    return None


def _inspect_sources(report: DatasetReport, sources: list, create_missing_captions: bool, probe_seconds: float) -> None:
    deadline = time.monotonic() + probe_seconds
    found = {key: [] for key in ("unpaired", "unused", "rejected", "unreadable", "empty", "missing_images", "missing_media", "class_named", "not_utf8", "rotated", "no_control", "silent", "video_no_caption")}
    for source in sources:
        summary, name = source.summary, source.summary.name
        try:
            files = sorted(entry for entry in os.listdir(summary.path) if not entry.startswith(".") and os.path.isfile(os.path.join(summary.path, entry)))
        except OSError as exc:
            summary.skipped = f"cannot be read ({exc.strerror or exc})"
            continue
        extension = source.caption_extension
        caption_stems = {os.path.splitext(entry)[0] for entry in files if extension and _ends_with(entry, (extension,))}
        summary.captions = len(caption_stems)
        images = [entry for entry in files if _kind(entry) == "image"]
        # musubi trains an image only when a caption with exactly its stem exists (glob_images); Start writes the
        # missing ones first when Create Missing Captions is on
        targets = [entry for entry in images if os.path.splitext(entry)[0] in caption_stems or create_missing_captions or source.class_captions]
        matches = {}
        if source.control_dir and "image" in source.kinds:
            if os.path.isdir(source.control_dir):
                try:
                    controls = [entry for entry in os.listdir(source.control_dir) if not entry.startswith(".") and _kind(entry) == "image"]
                except OSError as exc:
                    controls = []
                    report.problems.append(("error", f"{name}: the control folder cannot be read ({exc.strerror or exc}).", []))
                summary.control_folder = True
                label = os.path.basename(source.control_dir.rstrip(os.sep))
                matches, unused, rejected = match_controls(targets, controls)
                found["unused"].extend(f"{name}/{label}/{entry}" for entry in unused)
                found["rejected"].extend(f"{name}/{label}/{entry}" for entry in rejected)
                summary.unused_controls = len(unused)
            elif source.control_named or (report.control_required and images):
                found["no_control"].append(name + (f" ({source.control_dir})" if source.control_named else ""))
        elif report.control_required and images and "image" in source.kinds:
            found["no_control"].append(name)
        for entry in files:
            kind = _kind(entry)
            if not kind:
                continue
            path = os.path.join(summary.path, entry)
            stem = os.path.splitext(entry)[0]
            counter = {"image": "images", "video": "videos", "audio": "audio"}[kind]
            setattr(summary, counter, getattr(summary, counter) + 1)
            used = kind in source.kinds
            # images: an exact-stem caption (musubi compares names); videos / audio: musubi opens <stem><ext>
            has_caption = stem in caption_stems if kind == "image" else bool(extension) and os.path.isfile(os.path.join(summary.path, stem + extension))
            caption, caption_problem = (None, "")
            if has_caption:
                caption, caption_problem = _read_caption(os.path.join(summary.path, stem + extension))
                if caption_problem:
                    found["not_utf8" if "UTF-8" in caption_problem else "unreadable"].append(f"{name}/{stem}{extension} {caption_problem}")
            item = MediaItem(path, kind, name, caption if has_caption else None, controls=[os.path.join(source.control_dir, control) for control in matches.get(entry, [])])
            if used and not has_caption:
                summary.missing_captions += 1
                if source.class_captions:
                    found["class_named"].append(f"{name}/{entry}")
                elif create_missing_captions:
                    found["missing_media" if kind != "image" else "missing_images"].append(f"{name}/{entry}")
                elif kind == "image":
                    used = False  # dropped by musubi
                    found["missing_images"].append(f"{name}/{entry}")
                elif kind == "video" and report.profile == "audio":
                    used = False  # audio extraction skips a video without caption
                    found["video_no_caption"].append(f"{name}/{entry}")
                else:
                    found["missing_media"].append(f"{name}/{entry}")
            elif used and caption is not None and not caption and not caption_problem:
                found["empty"].append(f"{name}/{entry}")
            item.used = used
            if used and summary.control_folder and kind == "image":
                if item.controls:
                    summary.paired += 1
                else:
                    summary.unpaired += 1
                    found["unpaired"].append(f"{name}/{entry}")
            report.probe_total += 1
            if time.monotonic() < deadline:
                (_probe_image if kind == "image" else _probe_av)(item)
                report.probed += 1
                if item.error:
                    found["unreadable"].append(f"{name}/{entry} {item.error}")
                if item.rotated and used:
                    found["rotated"].append(f"{name}/{entry}")
            if kind == "video" and report.profile == "audio" and "video" in source.kinds:
                if item.has_sound is False:
                    item.used = False
                    found["silent"].append(f"{name}/{entry}")
                elif item.has_sound is None:
                    report.unknown_sound += 1
            report.items.append(item)
    _collect_problems(report, found, create_missing_captions)


def _examples(names: list) -> list:
    return names[:EXAMPLES] + ([f"… and {len(names) - EXAMPLES} more"] if len(names) > EXAMPLES else [])


def _collect_problems(report: DatasetReport, found: dict, create_missing_captions: bool) -> None:
    add = report.problems.append
    control = report.control_name or "control_images"
    if found["rejected"]:
        add(("error", f"{len(found['rejected'])} control image name(s) stop musubi's caching: after the last `_` only digits are allowed (photo_0.png, photo_1.png).", _examples(found["rejected"])))
    if found["no_control"]:
        add(("error", f"Control images are missing: no `{control}` folder (or a control_directory that does not exist) for {len(found['no_control'])} training folder(s).", _examples(found["no_control"])))
    if found["unpaired"]:
        add(("error", f"{len(found['unpaired'])} target image(s) have no control image: musubi stops caching (\"Could not find matching control images\"). Name each one like its target: photo.png → `{control}/photo.png` or `photo_0.png`.", _examples(found["unpaired"])))
    if found["unused"]:
        add(("warn", f"{len(found['unused'])} control image(s) match no trained target image and are not used.", _examples(found["unused"])))
    if found["unreadable"]:
        add(("error", f"{len(found['unreadable'])} file(s) cannot be read and would stop caching.", _examples(found["unreadable"])))
    if found["not_utf8"]:
        add(("error", f"{len(found['not_utf8'])} caption file(s) are not UTF-8 text: musubi stops when it reads them. Save them as UTF-8.", _examples(found["not_utf8"])))
    if found["missing_media"]:
        if create_missing_captions:
            add(("info", f"{len(found['missing_media'])} video / audio file(s) have no `{report.caption_extension}` caption: Start creates them (Create Missing Captions is on).", _examples(found["missing_media"])))
        else:
            add(("error", f"{len(found['missing_media'])} video / audio file(s) have no `{report.caption_extension}` caption: musubi stops caching. Turn on Create Missing Captions or caption them.", _examples(found["missing_media"])))
    if found["missing_images"]:
        if create_missing_captions:
            add(("info", f"{len(found['missing_images'])} image(s) have no `{report.caption_extension}` caption: Start creates them (Create Missing Captions is on).", _examples(found["missing_images"])))
        else:
            add(("warn", f"{len(found['missing_images'])} image(s) have no `{report.caption_extension}` caption with exactly their name and are NOT trained (musubi skips them). Turn on Create Missing Captions or caption them.", _examples(found["missing_images"])))
    if found["video_no_caption"]:
        add(("warn", f"{len(found['video_no_caption'])} video(s) have no `{report.caption_extension}` caption: Start skips them when it extracts the audio.", _examples(found["video_no_caption"])))
    if found["class_named"]:
        add(("info", f"{len(found['class_named'])} image(s) have no caption file: kohya uses the folder's class name as their caption.", _examples(found["class_named"])))
    if found["empty"]:
        add(("warn", f"{len(found['empty'])} caption file(s) are empty: those files train without any text.", _examples(found["empty"])))
    if found["rotated"]:
        add(("warn", f"{len(found['rotated'])} photo(s) carry an EXIF rotation: the trainer reads them as stored, so they train sideways (the previews show them that way). Save them upright.", _examples(found["rotated"])))
    if found["silent"]:
        add(("info", f"{len(found['silent'])} video(s) have no audio track and are skipped.", _examples(found["silent"])))
    sound = [item for item in report.items if item.kind == "video" and item.used and item.has_sound]
    if report.profile == "audio" and sound:
        add(("info", f"🔊 {len(sound)} video(s) have sound: Start extracts their audio (WAV, in the cache folder) and trains on it.", []))
    if report.unknown_sound:
        add(("info", f"The sound of {report.unknown_sound} video(s) was not checked (time limit); they are counted as having sound.", []))
    if report.resolution:
        try:
            target = min(int(float(value)) for value in report.resolution if value)
        except (TypeError, ValueError):
            target = 0
        small = [f"{item.folder}/{os.path.basename(item.path)} ({item.size[0]}×{item.size[1]})" for item in report.used if item.size and min(item.size) < target * 0.75]
        if small:
            add(("info", f"{len(small)} file(s) are much smaller than the training resolution ({target}px): they are upscaled, which softens them.", _examples(small)))
    extreme = [f"{item.folder}/{os.path.basename(item.path)}" for item in report.used if item.size and min(item.size) and max(item.size) / min(item.size) > 3]
    if extreme:
        add(("info", f"{len(extreme)} file(s) have an extreme aspect ratio (over 3:1); buckets crop or squeeze them.", _examples(extreme)))
    stems = {}
    for item in report.items:
        stems.setdefault((item.folder, os.path.splitext(os.path.basename(item.path))[0].lower()), []).append(os.path.basename(item.path))
    duplicates = [f"{folder}/{' + '.join(names)}" for (folder, _stem), names in stems.items() if len(names) > 1]
    if duplicates:
        add(("warn", f"{len(duplicates)} file name(s) exist twice with different extensions and share one caption file.", _examples(duplicates)))
    if report.probed < report.probe_total:
        add(("info", f"Files were opened and checked for {report.probed} of {report.probe_total} files (time limit); counts and pairs cover every file.", []))


# ------------------------------------------------------------------------------------------------ rendering
def _bucket_bars(title: str, counts: dict) -> str:
    total = sum(counts.values())
    if not total:
        return ""
    peak = max(counts.values())
    rows = "".join(
        f'<div class="ds-bar"><span class="ds-bar-label">{html.escape(label)}</span>'
        f'<span class="ds-bar-track"><span class="ds-bar-fill" style="width:{max(2, round(count * 100 / peak))}%"></span></span>'
        f'<span class="ds-bar-count">{count}</span></div>'
        for label, count in counts.items()
        if count
    )
    return f'<div class="ds-chart"><div class="ds-chart-title">{html.escape(title)}</div>{rows}</div>'


def _buckets(values: list, edges: list, labels: list) -> dict:
    counts = dict.fromkeys(labels, 0)
    for value in values:
        index = next((i for i, edge in enumerate(edges) if value < edge), len(edges))
        counts[labels[index]] += 1
    return counts


ASPECTS = (("9:21", 9 / 21), ("9:16", 9 / 16), ("2:3", 2 / 3), ("3:4", 3 / 4), ("4:5", 4 / 5), ("1:1", 1.0), ("5:4", 5 / 4), ("4:3", 4 / 3), ("3:2", 3 / 2), ("16:9", 16 / 9), ("21:9", 21 / 9))


def _aspect_label(width: int, height: int) -> str:
    """The nearest common aspect ratio (768x512 is 3:2, 1080x1920 is 9:16)."""
    if not width or not height:
        return "unknown"
    ratio = width / height
    return min(ASPECTS, key=lambda aspect: abs(math.log(ratio / aspect[1])))[0]


def _seconds(value: float) -> str:
    value = float(value or 0)
    if value >= 3600:
        return f"{value / 3600:.1f} h"
    if value >= 60:
        return f"{value / 60:.1f} min"
    return f"{value:.1f} s"


def markdown_inline(text: str) -> str:
    """The small Markdown subset scan_dataset writes (`code`, **bold**) as HTML."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)


def steps_per_epoch(report: DatasetReport) -> float:
    """Steps of one epoch: every trained file times its folder's repeats, divided by the batch size; kohya adds
    the regularization images (repeated up to the number of training images)."""
    repeats = {folder.name: folder.repeats for folder in report.folders}
    trained = sum(repeats.get(item.folder, 1) for item in report.used if not item.folder.startswith("reg: "))
    if report.kohya and report.regularization_images:
        trained *= 2
    return trained / report.batch_size


def report_html(report: DatasetReport, model_name: str = "") -> str:
    if report.error:
        return f'<div class="ds-report"><div class="ds-problem ds-error">❌ {html.escape(report.error)}</div></div>'
    used = report.used
    active = [folder for folder in report.folders if not folder.skipped]
    images = sum(folder.images for folder in active)
    videos = sum(folder.videos for folder in active)
    audio = sum(folder.audio for folder in active)
    media = images + videos + audio
    captioned = sum(1 for item in report.items if item.caption)
    steps = steps_per_epoch(report)
    cards = [("📁", "Training folders", f"{len(active)}" + (f" <small>({len(report.folders) - len(active)} skipped)</small>" if len(active) < len(report.folders) else ""))]
    if images:
        cards.append(("🖼️", "Images", f"{images:,}"))
    if videos:
        cards.append(("🎬", "Videos", f"{videos:,}"))
    if audio:
        cards.append(("🔊", "Audio clips", f"{audio:,}"))
    if media:
        cards.append(("📝", "Captioned", f"{captioned:,} / {media:,} <small>({captioned * 100 // max(1, media)}%)</small>"))
    paired_folders = [folder for folder in active if folder.control_folder]
    if paired_folders:
        paired = sum(folder.paired for folder in paired_folders)
        targets = paired + sum(folder.unpaired for folder in paired_folders)
        cards.append(("🧩", "Edit pairs", f"{paired:,} / {targets:,}"))
    trained = len([item for item in used if not item.folder.startswith("reg: ")])
    if used:
        cards.append(("✅", "Trained files", f"{trained:,}" + (f" <small>of {media:,}</small>" if trained != media else "")))
        cards.append(("🔁", "Steps / epoch", f"≈ {steps:,.0f} <small>(batch {report.batch_size}{', with regularization' if report.kohya and report.regularization_images else ''})</small>"))
    sizes = [item.size for item in used if item.size]
    if sizes:
        short = sorted(min(size) for size in sizes)
        cards.append(("📐", "Short side", f"{short[0]}–{short[-1]} px <small>(median {int(statistics.median(short))})</small>"))
    durations = [item.duration for item in used if item.kind in ("video", "audio") and item.duration]
    if durations:
        cards.append(("⏱️", "Total length", f"{_seconds(sum(durations))} <small>({_seconds(min(durations))}–{_seconds(max(durations))})</small>"))
    parts = ['<div class="ds-report">']
    title = f"Dataset check{' for ' + html.escape(model_name) if model_name else ''}"
    parts.append(f'<div class="ds-title">🔍 {title} <span>{html.escape(report.parent)}</span></div>')
    parts.append('<div class="ds-cards">' + "".join(f'<div class="ds-card"><span>{icon} {label}</span><b>{value}</b></div>' for icon, label, value in cards) + "</div>")
    # what the model trains on (one note per line), then the deeper problems with example files; missing
    # captions are listed below with example files
    notes = [line for line in report.notes_markdown.split("\n\n") if line.strip() and "caption. Enable **Create Missing Captions**" not in line]
    for note in notes:
        level = "error" if note.startswith("❌") else "warn" if note.startswith("⚠️") else "ok" if note.startswith("✅") else "info"
        if level == "ok" and report.problems and any(problem[0] in ("error", "warn") for problem in report.problems):
            continue  # "looks good" only when the deeper check agrees
        parts.append(f'<div class="ds-problem ds-{level}">{markdown_inline(note)}</div>')
    icons = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}
    for level, text, examples in sorted(report.problems, key=lambda problem: ("error", "warn", "info").index(problem[0])):
        details = "".join(f"<li>{html.escape(example)}</li>" for example in examples)
        parts.append(f'<div class="ds-problem ds-{level}">{icons[level]} {markdown_inline(text)}' + (f"<ul>{details}</ul>" if details else "") + "</div>")
    # per-folder table
    head = "<tr><th>Folder</th><th>Repeats</th><th>Images</th><th>Videos</th><th>Audio</th><th>Missing captions</th>" + ("<th>Pairs</th>" if paired_folders else "") + "</tr>"
    rows = []
    for folder in report.folders:
        if folder.skipped:
            rows.append(f'<tr class="ds-skipped"><td>⏭️ {html.escape(folder.name)}</td><td colspan="{6 if paired_folders else 5}">skipped: {html.escape(folder.skipped)}</td></tr>')
            continue
        pairs = ""
        if paired_folders:
            pairs = f"<td>{folder.paired}/{folder.paired + folder.unpaired}{' ⚠️' if folder.unpaired else ''}</td>" if folder.control_folder else "<td>—</td>"
        rows.append(
            f"<tr><td>{html.escape(folder.name)}</td><td>{folder.repeats}</td><td>{folder.images}</td><td>{folder.videos}</td><td>{folder.audio}</td>"
            f"<td>{folder.missing_captions}{' ⚠️' if folder.missing_captions else ''}</td>{pairs}</tr>"
        )
    parts.append(f'<div class="ds-table-wrap"><table class="ds-table">{head}{"".join(rows)}</table></div>')
    # distributions
    charts = []
    if sizes:
        aspect = {}
        for width, height in sizes:
            label = _aspect_label(width, height)
            aspect[label] = aspect.get(label, 0) + 1
        charts.append(_bucket_bars("Aspect ratio", {label: aspect.get(label, 0) for label in [*[name for name, _ratio in ASPECTS], "unknown"]}))
        charts.append(_bucket_bars("Short side (px)", _buckets([min(size) for size in sizes], [512, 768, 1024, 1536, 2048], ["< 512", "512–767", "768–1023", "1024–1535", "1536–2047", "≥ 2048"])))
    words = [len(item.caption.split()) for item in used if item.caption is not None]
    if words:
        charts.append(_bucket_bars("Caption length (words)", _buckets(words, [1, 6, 21, 61, 151], ["empty", "1–5", "6–20", "21–60", "61–150", "> 150"])))
    if durations:
        charts.append(_bucket_bars("Clip length", _buckets(durations, [1, 3, 6, 10, 20], ["< 1 s", "1–3 s", "3–6 s", "6–10 s", "10–20 s", "> 20 s"])))
    fps = [round(item.fps) for item in used if item.fps]
    if fps:
        counts = {}
        for value in sorted(fps):
            counts[f"{value} fps"] = counts.get(f"{value} fps", 0) + 1
        charts.append(_bucket_bars("Frame rate", counts))
    charts = [chart for chart in charts if chart]
    if charts:
        parts.append('<div class="ds-charts">' + "".join(charts) + "</div>")
    parts.append("</div>")
    return "".join(parts)


# ------------------------------------------------------------------------------------------------ previews
def preview_dir() -> str:
    folder = os.path.join(tempfile.gettempdir(), PREVIEW_DIRNAME)
    os.makedirs(folder, exist_ok=True)
    return folder


def _prune_previews(folder: str) -> None:
    """Keep the MAX_PREVIEW_FILES most recently used previews (a reused preview is touched)."""
    try:
        names = [os.path.join(folder, name) for name in os.listdir(folder) if name.endswith(".jpg")]
        if len(names) <= MAX_PREVIEW_FILES:
            return
        for path in sorted(names, key=os.path.getmtime)[: len(names) - MAX_PREVIEW_FILES]:
            try:
                os.remove(path)  # our own preview thumbnails only
            except OSError:
                pass
    except OSError:
        pass


def _signature(paths: list, extra: str) -> str:
    parts = [extra]
    for path in paths:
        try:
            stat = os.stat(path)
            parts.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
        except OSError:
            parts.append(path)
    return hashlib.sha1("\n".join(parts).encode("utf-8", "replace")).hexdigest()[:24]


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1
        return ImageFont.load_default()


def _open_rgb(path: str, height: int):
    """An image as the trainer reads it (stored orientation: musubi ignores EXIF rotation), at one height."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = height / max(1, image.height)
        return image.resize((max(1, round(image.width * scale)), height), Image.LANCZOS)


def _label(draw, xy, text: str, size: int = 22) -> None:
    font = _font(size)
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), radius=6, fill=(15, 23, 42, 200))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def _save(image, target: str) -> None:
    """Write next to the final name, then rename: another browser never gets a half-written preview."""
    handle, partial = tempfile.mkstemp(suffix=".part", dir=os.path.dirname(target))
    os.close(handle)
    try:
        image.save(partial, "JPEG", quality=86)
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _pair_preview(item: MediaItem, target: str) -> None:
    """Controls, an arrow, then the target image, at one height."""
    from PIL import Image, ImageDraw

    height = 360
    panels = [(_open_rgb(control, height), f"control {index + 1}" if len(item.controls) > 1 else "control") for index, control in enumerate(item.controls[:3])]
    panels.append((_open_rgb(item.path, height), "target"))
    gap, arrow = 14, 46
    width = sum(panel.width for panel, _label_text in panels) + gap * (len(panels) - 1) + arrow
    canvas = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(canvas, "RGBA")
    x = 0
    for index, (panel, text) in enumerate(panels):
        if index == len(panels) - 1:  # the arrow before the target
            middle = height // 2
            draw.polygon([(x + 8, middle - 16), (x + arrow - 10, middle), (x + 8, middle + 16)], fill=(99, 102, 241, 255))
            x += arrow
        canvas.paste(panel, (x, 0))
        _label(draw, (x + 10, 10), text)
        x += panel.width + gap
    canvas.thumbnail((1400, height))
    _save(canvas, target)


def _fit(image, longest: int = 640, shortest_longest: int = 480):
    """At most ``longest`` px; a tiny image is enlarged so its labels stay readable."""
    from PIL import Image

    image.thumbnail((longest, longest), Image.LANCZOS)
    if max(image.size) < shortest_longest:
        scale = shortest_longest / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BICUBIC)
    return image


def _image_preview(item: MediaItem, target: str) -> None:
    from PIL import Image

    with Image.open(item.path) as image:
        image = _fit(image.convert("RGB"))  # as stored: the trainer ignores EXIF rotation
        _save(image, target)


def _video_preview(item: MediaItem, target: str) -> None:
    """The first frame with a play badge and the clip's length."""
    import av
    from PIL import ImageDraw

    with av.open(item.path) as container:
        frame = next(container.decode(video=0))
        image = frame.to_image().convert("RGB")
    image = _fit(image)
    draw = ImageDraw.Draw(image, "RGBA")
    middle_x, middle_y, radius = image.width // 2, image.height // 2, max(18, min(image.size) // 9)
    draw.ellipse((middle_x - radius, middle_y - radius, middle_x + radius, middle_y + radius), fill=(15, 23, 42, 150))
    draw.polygon([(middle_x - radius // 3, middle_y - radius // 2), (middle_x + radius // 2, middle_y), (middle_x - radius // 3, middle_y + radius // 2)], fill=(255, 255, 255, 235))
    facts = " · ".join(part for part in (_seconds(item.duration) if item.duration else "", f"{item.fps:g} fps" if item.fps else "", f"{item.frames} frames" if item.frames else "", "sound" if item.has_sound else "") if part)
    if facts:
        _label(draw, (10, image.height - 40), facts)
    _save(image, target)


def _audio_preview(item: MediaItem, target: str) -> None:
    """A waveform of the first minute with the clip's length."""
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    width, height, rate = 720, 200, 8000
    chunks = []
    with av.open(item.path) as container:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
            if sum(len(chunk) for chunk in chunks) >= rate * 60:
                break
    samples = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    columns = np.array_split(samples, width)
    peaks = np.array([float(np.abs(column).max()) if len(column) else 0.0 for column in columns])
    peaks = peaks / max(1e-6, float(peaks.max()))
    image = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(image, "RGBA")
    middle = height // 2
    for x, peak in enumerate(peaks):
        extent = max(1, int(peak * (middle - 14)))
        draw.line((x, middle - extent, x, middle + extent), fill=(99, 102, 241, 255))
    _label(draw, (10, 10), f"{_seconds(item.duration) if item.duration else ''} audio".strip())
    _save(image, target)


def _caption_text(item: MediaItem) -> str:
    text = item.caption if item.caption is not None else "(no caption file)"
    text = text if len(text) <= 220 else text[:217].rstrip() + "…"
    return f"{item.folder} / {os.path.basename(item.path)}\n{text}"


def sample_previews(report: DatasetReport, count: int = 12, seed: int = 0, seconds: float = PREVIEW_SECONDS) -> list:
    """Gallery items (preview file, caption) for up to ``count`` files the model trains on, spread over the
    training folders. Previews are cached by file version (asking again is instant); a file that cannot be
    previewed is added to the report's problems."""
    pool = [item for item in report.used if not item.error]
    if not pool:
        return []
    rng = random.Random(seed)
    by_folder = {}
    for item in pool:
        by_folder.setdefault(item.folder, []).append(item)
    for items in by_folder.values():
        rng.shuffle(items)
    picked, round_index = [], 0
    folders = list(by_folder)
    rng.shuffle(folders)
    while len(picked) < count and any(round_index < len(items) for items in by_folder.values()):
        for folder in folders:
            items = by_folder[folder]
            if round_index < len(items) and len(picked) < count:
                picked.append(items[round_index])
        round_index += 1
    folder = preview_dir()
    gallery, failed = [], []
    deadline = time.monotonic() + seconds
    for item in picked:
        sources = [item.path, *item.controls]
        target = os.path.join(folder, _signature(sources, f"{item.kind}-{bool(item.controls)}-stored") + ".jpg")
        if os.path.isfile(target):
            try:
                os.utime(target)  # recently used: kept by the pruning
            except OSError:
                pass
        else:
            if time.monotonic() > deadline:
                break
            try:
                if item.controls:
                    _pair_preview(item, target)
                elif item.kind == "image":
                    _image_preview(item, target)
                elif item.kind == "video":
                    _video_preview(item, target)
                else:
                    _audio_preview(item, target)
            except Exception as exc:  # noqa: BLE001 - reported below
                failed.append(f"{item.folder}/{os.path.basename(item.path)} ({type(exc).__name__})")
                continue
        gallery.append((target, _caption_text(item)))
    if failed:
        report.problems.append(("error", f"{len(failed)} file(s) could not be opened for a preview; the trainer will likely fail on them too.", _examples(failed)))
    _prune_previews(folder)
    return gallery
