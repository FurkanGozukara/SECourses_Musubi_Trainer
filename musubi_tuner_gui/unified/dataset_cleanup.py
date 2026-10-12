"""Inner folders of training folders, and safe removal of old cache folders (🧹 Clean old caches).

The trainers read only the files directly inside a training folder, so any folder inside it (an old
``cache_dir_<W>x<H>`` from another resolution, a backup, a control folder of another model) is ignored.
``inner_folders`` tells what each one holds; ``remove_cache_folders`` deletes only folders that hold
nothing but cache files, after checking every target again:

* the target is an absolute path strictly inside the dataset folder (path-aware containment, never a
  string prefix), never a drive / share root, the home folder or the app folders;
* no symbolic link, junction or other reparse point is followed or deleted (a folder that holds one
  is refused);
* the folder still holds only cache files when it is deleted (it is classified again first);
* files are removed bottom-up with Python's own file APIs and the result is verified.
"""

from __future__ import annotations

import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

CACHE_NAME_RE = re.compile(r"(^|[_\-. ])(cache|latents?)([_\-. ]|$)|^cache", re.IGNORECASE)
CACHE_EXTENSIONS = (".safetensors", ".npz", ".npy", ".pt", ".pth", ".json")
# LTX 2.3 audio mode extracts the sound of videos to <cache>/extracted_audio (WAV + a copy of the caption)
EXTRACTED_AUDIO_EXTENSIONS = (".wav", ".txt")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl", ".gif", ".tif", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma")
SCAN_SECONDS = 4.0


@dataclass
class InnerFolder:
    training_folder: str  # the training folder's name (1_ohwx)
    name: str  # the inner folder's name
    path: str
    kind: str  # "cache" | "media" | "other" | "empty" | "linked"
    files: int = 0
    size: int = 0
    media: dict = field(default_factory=dict)  # image / video / audio counts (outside extracted_audio)
    complete: bool = True  # False: the time budget stopped the scan (never deleted then)

    @property
    def label(self) -> str:
        return f"{self.training_folder}/{self.name}"


def _is_link(entry_or_path) -> bool:
    """A symbolic link, a junction or any other reparse point (Windows), checked without following it."""
    try:
        if isinstance(entry_or_path, os.DirEntry):
            if entry_or_path.is_symlink() or (hasattr(entry_or_path, "is_junction") and entry_or_path.is_junction()):
                return True
            info = entry_or_path.stat(follow_symlinks=False)
        else:
            path = str(entry_or_path)
            if os.path.islink(path) or (hasattr(os.path, "isjunction") and os.path.isjunction(path)):
                return True
            info = os.lstat(path)
        attributes = getattr(info, "st_file_attributes", 0)
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    except OSError:
        return True  # unreadable: treated as a link, so it is never deleted


def _media_kind(name: str) -> str:
    lower = name.lower()
    if lower.endswith(IMAGE_EXTENSIONS):
        return "image"
    if lower.endswith(VIDEO_EXTENSIONS):
        return "video"
    if lower.endswith(AUDIO_EXTENSIONS):
        return "audio"
    return ""


def classify_folder(path: str, training_folder: str = "", deadline: float | None = None) -> InnerFolder:
    """What one inner folder holds (recursively, never following links). Only a folder whose name says cache
    (``cache_dir``, ``cache_dir_lab_1024x1024``, ``my_latent_cache``) and that holds nothing but cache files
    (and LTX 2.3's extracted audio) is a "cache" folder: a ``loras`` folder full of .safetensors is not."""
    name = os.path.basename(os.path.normpath(path))
    item = InnerFolder(training_folder, name, path, "other")
    if _is_link(path):
        item.kind = "linked"
        return item
    cache_named = bool(CACHE_NAME_RE.search(name))
    only_cache_files = True
    linked = False
    stack = [(path, False)]
    while stack:
        folder, in_extracted_audio = stack.pop()
        if deadline is not None and time.monotonic() > deadline:
            item.complete = False
            break
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if _is_link(entry):
                        linked = True
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((entry.path, in_extracted_audio or entry.name.lower() == "extracted_audio"))
                        continue
                    item.files += 1
                    try:
                        item.size += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
                    lower = entry.name.lower()
                    kind = _media_kind(lower)
                    extracted = in_extracted_audio and lower.endswith(EXTRACTED_AUDIO_EXTENSIONS)
                    if kind and not extracted:
                        item.media[kind] = item.media.get(kind, 0) + 1
                    if not (lower.endswith(CACHE_EXTENSIONS) or extracted):
                        only_cache_files = False
        except OSError:
            only_cache_files = False
            item.complete = False
    if linked:
        item.kind = "linked"
    elif any(item.media.values()):
        item.kind = "media"
    elif item.files == 0:
        item.kind = "empty"
    elif cache_named and only_cache_files and item.complete:
        item.kind = "cache"
    return item


def inner_folders(training_folder: str, owned: set, deadline: float | None = None) -> list[InnerFolder]:
    """Every inner folder of ``training_folder`` the trainer ignores (``owned``: the cache / control folders the
    current settings use, which are not listed)."""
    found = []
    display = os.path.basename(os.path.normpath(training_folder))
    try:
        names = sorted(os.listdir(training_folder))
    except OSError:
        return found
    for name in names:
        path = os.path.join(training_folder, name)
        if name.startswith(".") or name in owned or not os.path.isdir(path):
            continue
        found.append(classify_folder(path, display, deadline))
    return found


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# ------------------------------------------------------------------------------------------------ deletion
def _resolved(path: str) -> Path:
    return Path(os.path.abspath(path)).resolve(strict=False)


def _protected_roots() -> list[Path]:
    from .schema import APP_DIR, ROOT_DIR

    roots = [Path.home(), Path(ROOT_DIR), Path(APP_DIR)]
    return [_resolved(str(root)) for root in roots]


def validate_target(target: str, allowed_root: str) -> str:
    """'' when ``target`` may be deleted as a cache folder directly inside ``allowed_root`` (its training folder),
    else the reason it may not."""
    if not target or not str(target).strip() or not allowed_root or not str(allowed_root).strip():
        return "empty path"
    if not os.path.isabs(target) or not os.path.isabs(allowed_root):
        return "not an absolute path"
    if _is_link(target):
        return "it is a link or junction"
    target_path, root_path = _resolved(target), _resolved(allowed_root)
    if target_path == Path(target_path.anchor) or root_path == Path(root_path.anchor):
        return "a drive or share root"
    if target_path in _protected_roots():
        return "a protected folder"
    try:
        relative = target_path.relative_to(root_path)
    except ValueError:
        return "outside the dataset folder"
    if not relative.parts:
        return "the dataset folder itself"
    if len(relative.parts) != 1:
        return "not directly inside its training folder"
    if not target_path.is_dir():
        return "not a folder"
    return ""


def remove_cache_folders(targets: list[tuple[str, str]]) -> tuple[list[str], list[str], int]:
    """Delete the listed cache folders, each given with the training folder it must be inside:
    (removed labels, problems, bytes freed). Each one is validated and classified again right before
    deletion; anything that no longer qualifies is left alone and reported."""
    removed, problems, freed = [], [], 0
    for target, allowed_root in targets:
        reason = validate_target(target, allowed_root)
        if not reason and _resolved(allowed_root) in _protected_roots():
            reason = "its training folder is a protected folder"
        if reason:
            problems.append(f"{target}: not deleted ({reason})")
            continue
        current = classify_folder(target, os.path.basename(os.path.dirname(target)))
        if current.kind != "cache":
            problems.append(f"{target}: not deleted (it no longer holds only cache files: {current.kind})")
            continue
        try:
            for folder, dirnames, filenames in os.walk(target, topdown=False, followlinks=False):
                for filename in filenames:
                    path = os.path.join(folder, filename)
                    if _is_link(path):
                        raise OSError(f"link found while deleting: {path}")
                    os.remove(path)
                for dirname in dirnames:
                    path = os.path.join(folder, dirname)
                    if _is_link(path):
                        raise OSError(f"link found while deleting: {path}")
                    os.rmdir(path)
            os.rmdir(target)
        except OSError as exc:
            problems.append(f"{target}: deletion stopped ({exc})")
            continue
        if os.path.exists(target):
            problems.append(f"{target}: still exists after deletion")
            continue
        removed.append(f"{current.training_folder}/{current.name}")
        freed += current.size
    return removed, problems, freed
