"""Preset discovery, guardrails and remembered UI state for the unified workspaces.

* Shipped presets: ``<install root>/Training_Presets/<Model>_Training_Configs/{LoRA_Training,
  Fine_Tuning_Training}/<tier>_GB_GPUs_<peak>_GB*.toml`` (via ``preset_library``; read-only).
* User presets: ``<install root>/user_presets/<base model>/<name>.toml``. Saving with a plain name
  (``my preset`` or ``my preset.toml``) writes there; a path (absolute, or containing a folder
  separator) writes exactly there.
* Every preset the unified UI saves starts with ``preset_*`` keys (category, base model, training
  type). Loading checks them — or, for older files, the preset folder and the file contents — so a
  preset made for another model or category is never applied silently.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from .. import preset_library
from ..preset_library import ALL_TIERS, PresetEntry
from .param_spec import FULL, LORA
from .registry import CATEGORY_TITLES, ENTRIES, ENTRY_BY_KEY, MODE_LABELS, ModelEntry
from .schema import ROOT_DIR, load_schema
from .values import clean_path, strip_quotes

USER_PRESETS_DIRNAME = "user_presets"
UI_STATE_FILENAME = "ui_state.json"
PRESET_FORMAT = 1
AUTO_TIER = "auto"
MODE_FOLDER_LABEL = {LORA: "LoRA", FULL: "Fine-Tuning"}
SHIPPED_ICON = "📦"
USER_ICON = "⭐"  # high contrast in both themes (a silhouette emoji vanishes on dark backgrounds)


def user_presets_root() -> str:
    override = os.environ.get("MUSUBI_USER_PRESETS_DIR", "").strip()
    return os.path.abspath(override) if override else os.path.join(ROOT_DIR, USER_PRESETS_DIRNAME)


def ui_state_path() -> str:
    return os.path.join(user_presets_root(), UI_STATE_FILENAME)


# ------------------------------------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------------------------------------
def detected_gpu() -> tuple[str, float] | None:
    """(short name, total GB) of GPU 0, or None."""
    gpus = preset_library.detect_gpus()
    if not gpus:
        return None
    name, vram = gpus[0]
    return preset_library.gpu_label(name), vram


def gpu_text() -> str:
    gpu = detected_gpu()
    return f"{gpu[0]} · {gpu[1]:g} GB" if gpu else "No NVIDIA GPU detected"


# ------------------------------------------------------------------------------------------------
# Listing
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PresetOption:
    path: str
    label: str
    shipped: bool
    tier_gb: float | None = None
    used_gb: float | None = None
    quality: int | None = None
    description: str = ""


def _entry_folders(entry: ModelEntry) -> list[str]:
    return list(entry.preset_folders)


def shipped_presets(entry: ModelEntry, mode: str) -> list[PresetEntry]:
    wanted = MODE_FOLDER_LABEL[mode]
    entries = preset_library.list_presets(_entry_folders(entry), mode=wanted)
    if entry.preset_subfolders:
        allowed = tuple(part.lower() for part in entry.preset_subfolders)
        entries = [item for item in entries if any(f"/{part}/" in item.path.replace("\\", "/").lower() for part in allowed)]
    return entries


def tier_choices(entry: ModelEntry, mode: str) -> list[tuple[str, str]]:
    tiers = sorted({item.tier_gb for item in shipped_presets(entry, mode) if item.tier_gb is not None})
    gpu = detected_gpu()
    auto_label = f"🎯 Auto · {gpu[1]:.0f} GB" if gpu else "🎯 Auto (no GPU)"
    return [(auto_label, AUTO_TIER), ("All GPUs", ALL_TIERS)] + [(f"{tier:g} GB GPUs", f"{tier:g}") for tier in tiers]


def offered_tier(entry: ModelEntry, mode: str, tier: str | None) -> str:
    """``tier`` when this model / training type offers it, else auto (a tier picked for another model
    must not hide every preset of this one)."""
    if tier and tier in {value for _label, value in tier_choices(entry, mode)}:
        return tier
    return AUTO_TIER


def resolve_tier(entry: ModelEntry, mode: str, tier: str | None) -> str:
    """Concrete tier value ("24", "all") for the tier selector value (which may be "auto")."""
    tier = offered_tier(entry, mode, tier)
    if tier != AUTO_TIER:
        return tier
    gpu = detected_gpu()
    return preset_library.best_tier_for(shipped_presets(entry, mode), gpu[1] if gpu else None)


def _shipped_label(item: PresetEntry) -> str:
    parts = []
    if item.tier_gb is not None:
        parts.append(f"{item.tier_gb:g} GB GPUs")
    if item.used_gb is not None:
        parts.append(f"peak {item.used_gb:g} GB")
    name = os.path.splitext(os.path.basename(item.path))[0]
    match = preset_library.TIER_PATTERN.search(name)
    details = name[match.end():] if match else name
    details = details.replace("_", " ").strip()
    if details:
        parts.append(details)
    subgroup = item.family.split(" · ", 1)[1] if " · " in item.family else ""
    if subgroup:
        parts.insert(0, subgroup)
    return f"{SHIPPED_ICON} " + " · ".join(parts)


def user_presets(entry: ModelEntry, mode: str) -> list[PresetOption]:
    folder = os.path.join(user_presets_root(), entry.key)
    options: list[PresetOption] = []
    if not os.path.isdir(folder):
        return options
    for name in sorted(os.listdir(folder), key=str.lower):
        if not name.lower().endswith((".toml", ".json")) or _is_partial_file(name):
            continue
        path = os.path.join(folder, name)
        meta = read_metadata(path)
        preset_mode = meta.get("preset_training_type")
        if preset_mode in (LORA, FULL) and preset_mode != mode:
            continue
        options.append(PresetOption(path=os.path.abspath(path), label=f"{USER_ICON} {os.path.splitext(name)[0]}", shipped=False))
    return options


def preset_options(entry: ModelEntry, mode: str, tier: str | None) -> list[PresetOption]:
    concrete = resolve_tier(entry, mode, tier)
    shipped = preset_library.filter_presets(shipped_presets(entry, mode), tier=concrete)
    ranked = sorted(shipped, key=lambda item: (item.tier_gb if item.tier_gb is not None else 999, item.preference))
    options = [
        PresetOption(item.path, _shipped_label(item), True, item.tier_gb, item.used_gb, item.quality, item.description)
        for item in ranked
    ]
    return user_presets(entry, mode) + options


MULTI_GPU_RE = re.compile(r"(\d+)x_GPUs?", re.I)


def _gpus_needed(path: str) -> int:
    match = MULTI_GPU_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else 1


def recommended_preset(entry: ModelEntry, mode: str, tier: str | None) -> str | None:
    concrete = resolve_tier(entry, mode, tier)
    candidates = preset_library.filter_presets(shipped_presets(entry, mode), tier=concrete)
    # multi-GPU presets ("..._4x_GPU_Config") are only auto-picked when that many GPUs exist
    gpu_count = max(1, len(preset_library.detect_gpus()))
    single = [item for item in candidates if _gpus_needed(item.path) <= gpu_count]
    pool = single or candidates
    for preferred in entry.preset_preference:  # e.g. Wan 2.2: text-to-video presets before text-to-image ones
        matching = [item for item in pool if preferred.lower() in item.path.replace("\\", "/").lower()]
        if matching:
            pool = matching
            break
    gpu = detected_gpu()
    best = preset_library.default_preset(pool, gpu[1] if gpu else None)
    return best.path if best else None


def describe(entry: ModelEntry, path: str | None) -> str:
    if not path:
        return ""
    for item in shipped_presets(entry, LORA) + shipped_presets(entry, FULL):
        if os.path.normcase(item.path) == os.path.normcase(os.path.abspath(path)):
            text = item.description or ""
            return text
    meta = read_metadata(path)
    return str(meta.get("preset_description") or "")


# ------------------------------------------------------------------------------------------------
# Metadata + identification
# ------------------------------------------------------------------------------------------------
def _is_partial_file(name: str) -> bool:
    """Files a save is still writing (or that an interrupted save left behind), never presets."""
    lowered = name.lower()
    return lowered.startswith(".tmp_") or lowered.endswith((".saving.toml", ".partial.toml"))


def read_metadata(path: str) -> dict:
    """``preset_*`` keys of a preset, read cheaply from its top lines (TOML) or the JSON object."""
    meta: dict = {}
    try:
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8-sig") as handle:
                data = json.load(handle)
            return {key: value for key, value in data.items() if key.startswith("preset_")} if isinstance(data, dict) else {}
        with open(path, "r", encoding="utf-8-sig") as handle:
            for _ in range(40):
                line = handle.readline()
                if not line:
                    break
                match = re.match(r'^\s*(preset_\w+)\s*=\s*(.+?)\s*$', line)
                if match:
                    raw = match.group(2)
                    if raw.startswith('"') and raw.endswith('"'):
                        meta[match.group(1)] = raw[1:-1]
                    elif raw.isdigit():
                        meta[match.group(1)] = int(raw)
                    else:
                        meta[match.group(1)] = raw
    except Exception:  # unreadable, not UTF-8, malformed JSON: no metadata (loading reports the real problem)
        pass
    return meta


@dataclass(frozen=True)
class PresetIdentity:
    category: str | None
    entry_key: str | None
    mode: str | None
    source: str  # "metadata" | "folder" | "content" | "unknown"

    @property
    def entry(self) -> ModelEntry | None:
        return ENTRY_BY_KEY.get(self.entry_key) if self.entry_key else None

    def describe(self) -> str:
        entry = self.entry
        parts = []
        if self.category:
            parts.append(CATEGORY_TITLES.get(self.category, self.category))
        if entry:
            parts.append(f"{entry.icon} {entry.name}")
        if self.mode:
            parts.append(MODE_LABELS[self.mode])
        return " → ".join(parts) if parts else "an unknown model"


@lru_cache(maxsize=1)
def _folder_owner() -> dict[str, str]:
    owners = {}
    for entry in ENTRIES:
        for folder in entry.preset_folders:
            owners[folder.lower()] = entry.key
    return owners


@lru_cache(maxsize=1)
def _backend_key_sets() -> dict[str, frozenset]:
    sets = {name: frozenset(data["keys"]) for name, data in load_schema()["backends"].items()}
    try:
        from .kohya_backend import kohya_param_specs

        for model in ("flux1_dev", "sdxl"):
            sets[f"kohya:{model}"] = frozenset(spec.key for spec in kohya_param_specs(model))
    except Exception:
        pass
    return sets


NETWORK_MODULE_BACKENDS = {
    "networks.lora_krea2": "krea2",
    "networks.lora_ideogram4": "ideogram4",
    "networks.lora_qwen_image_21": "qwen_image_21",
    "networks.lora_qwen_image": "qwen_image",
    "networks.lora_flux_2": "flux",
    "networks.lora_flux_3": "flux3_klein",
    "networks.lora_moss_image": "moss_image",
    "networks.lora_zimage": "zimage",
    "networks.lora_ernie_image": "ernie_image",
    "networks.lora_mage_flow": "mage_flow",
    "musubi_tuner.networks.lora_mage_flow": "mage_flow",
    "networks.lora_hidream_o1": "hidream_o1",
    "networks.lora_wan": "wan",
    "networks.lora_ltx2": "ltx2",
    "networks.lora_minimax_h3": "minimax_h3",
    "networks.lora_kandinsky6": "kandinsky6",
}


def _entry_for_backend(backend: str, data: dict) -> str | None:
    if backend == "flux":
        family = str(data.get("model_family") or "")
        version = str(data.get("model_version") or "").lower()
        if "klein" in family.lower() or "klein" in version:
            return "flux2_klein_4b" if "4b" in version else "flux2_klein_9b"
        return "flux2_dev"
    if backend == "zimage":
        return "zimage_turbo" if str(data.get("zimage_variant") or "").lower() == "turbo" else "zimage_base"
    if backend == "mage_flow":
        return "mage_flow_edit" if bool(data.get("is_edit")) else "mage_flow"
    if backend == "flux3_klein":
        return "flux3_klein_edit" if bool(data.get("use_control_images")) else "flux3_klein"
    if backend == "wan":
        task = str(data.get("task") or "")
        return "wan22" if "A14B" in task else "wan21"
    if backend == "ltx2":
        return "ltx23_audio" if str(data.get("ltx2_mode") or "") == "audio" else "ltx23"
    if backend.startswith("kohya:"):
        return backend.split(":", 1)[1]
    candidates = [entry.key for entry in ENTRIES if entry.backend == backend]
    return candidates[0] if candidates else None


def _mode_from_data(data: dict) -> str | None:
    if "training_mode" in data:
        value = str(data.get("training_mode") or "").lower()
        if "fine" in value or "dreambooth" in value or "full" in value:
            return FULL
        if "lora" in value:
            return LORA
    if "LoRA_type" in data:
        return LORA
    if data.get("network_module"):
        return LORA
    if "network_module" in data or "network_dim" in data:
        return LORA
    return None


def _kohya_json_backend(data: dict) -> str | None:
    if "pretrained_model_name_or_path" not in data:
        return None
    if data.get("flux1_checkbox") or "ae" in data and data.get("ae") or "t5xxl" in data and data.get("t5xxl"):
        return "kohya:flux1_dev"
    if data.get("sdxl") or "sdxl_no_half_vae" in data:
        return "kohya:sdxl"
    return None


def identify_preset(path: str, data: dict | None = None) -> PresetIdentity:
    """Which category / base model / training type a preset file was made for."""
    meta = read_metadata(path)
    entry_key = meta.get("preset_model")
    if entry_key in ENTRY_BY_KEY:
        entry = ENTRY_BY_KEY[entry_key]
        mode = meta.get("preset_training_type") if meta.get("preset_training_type") in (LORA, FULL) else None
        return PresetIdentity(entry.category, entry.key, mode, "metadata")

    if data is None:
        from .backends import read_preset_file

        try:
            data = read_preset_file(path)
        except Exception:
            return PresetIdentity(None, None, None, "unknown")

    if not isinstance(data, dict):
        return PresetIdentity(None, None, None, "unknown")
    mode = _mode_from_data(data)
    # shipped preset folders name their model; only the folders and file name below that one can tell the
    # training type (the install path itself may well contain "DreamBooth" or "Fine_Tuning")
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    for index, part in enumerate(parts):
        owner = _folder_owner().get(part.lower())
        if owner:
            entry = ENTRY_BY_KEY[owner]
            below = "/".join(parts[index + 1:]).lower()
            if "fine_tuning" in below or "dreambooth" in below:
                mode = FULL
            elif "lora" in below:
                mode = LORA
            return PresetIdentity(entry.category, entry.key, mode or _kohya_mode(entry.backend, data), "folder")

    backend = _kohya_json_backend(data) if path.lower().endswith(".json") else None
    if backend is None:
        module = str(data.get("network_module") or "").strip()
        backend = NETWORK_MODULE_BACKENDS.get(module)
    if backend is None:
        keys = {key for key in data if not key.startswith("preset_")}
        sets = _backend_key_sets()
        scores = []
        for name, known in sets.items():
            if not keys:
                break
            overlap = len(keys & known) / len(keys)
            unique_hits = len(keys & (known - frozenset().union(*(other for other_name, other in sets.items() if other_name != name))))
            scores.append((overlap + 0.05 * unique_hits, name))
        scores.sort(reverse=True)
        # a clear winner only: a few common settings (learning rate, epochs) fit every trainer equally
        if scores and scores[0][0] >= 0.6 and (len(scores) == 1 or scores[0][0] - scores[1][0] > 1e-9):
            backend = scores[0][1]
        elif scores and scores[0][0] >= 0.6:
            # trainers with the same settings (Krea 2 / Ideogram 4 share one tab module): the file or folder name decides
            tied = {name for score, name in scores if scores[0][0] - score <= 1e-9}
            hinted = _name_hint(path)
            if hinted and ENTRY_BY_KEY[hinted].backend in tied:
                backend = ENTRY_BY_KEY[hinted].backend
    if backend is None:
        return PresetIdentity(None, None, mode, "unknown")
    entry_key = _entry_for_backend(backend, data)
    entry = ENTRY_BY_KEY.get(entry_key) if entry_key else None
    return PresetIdentity(entry.category if entry else None, entry.key if entry else None, mode or _kohya_mode(backend, data), "content")


# (pattern on the normalized file / folder names, entry key): most specific first
_NAME_HINTS = (
    (r"mage.?flow.?edit", "mage_flow_edit"),
    (r"mage.?flow", "mage_flow"),
    (r"flux.?3.?(?:klein.?)?edit", "flux3_klein_edit"),
    (r"flux.?3", "flux3_klein"),
    (r"moss", "moss_image"),
    (r"klein.?4b|4b.?klein", "flux2_klein_4b"),
    (r"klein", "flux2_klein_9b"),
    (r"flux.?2", "flux2_dev"),
    (r"flux.?1|flux.?dev", "flux1_dev"),
    (r"qwen.?image.?2.?1|qwen.?2.?1", "qwen_image_21"),
    (r"qwen", "qwen_image"),
    (r"z.?image.?turbo", "zimage_turbo"),
    (r"z.?image", "zimage_base"),
    (r"krea", "krea2"),
    (r"ideogram", "ideogram4"),
    (r"hidream", "hidream_o1"),
    (r"ernie", "ernie_image"),
    (r"kandinsky", "kandinsky6"),
    (r"ltx.*audio", "ltx23_audio"),
    (r"ltx", "ltx23"),
    (r"minimax|(^|[^a-z])h3([^a-z0-9]|$)", "minimax_h3"),
    (r"wan.?2.?2|wan22", "wan22"),
    (r"wan", "wan21"),
    (r"sdxl", "sdxl"),
)


def _name_hint(path: str) -> str | None:
    """The base model a preset's file name (then its folder names) mentions, e.g. Krea_2_Fine_Tuning_Demo.toml."""
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    for part in [parts[-1], *reversed(parts[-4:-1])]:
        text = part.lower()
        for pattern, entry_key in _NAME_HINTS:
            if entry_key in ENTRY_BY_KEY and re.search(pattern, text):
                return entry_key
    return None


KOHYA_LORA_KEYS = ("LoRA_type", "network_dim", "network_alpha", "network_module")


def _kohya_mode(backend: str, data: dict) -> str | None:
    """kohya_ss DreamBooth / fine-tuning configs carry no LoRA settings at all (the LoRA tab always saves them)."""
    if backend.startswith("kohya:") and not any(key in data for key in KOHYA_LORA_KEYS):
        return FULL
    return None


# ------------------------------------------------------------------------------------------------
# Name-or-path handling
# ------------------------------------------------------------------------------------------------
INVALID_NAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def looks_like_path(text: str) -> bool:
    """A folder separator, a drive or ``~`` / ``%VAR%`` makes a path; ``my preset.toml`` is still a name."""
    text = strip_quotes(text or "")
    return bool(text) and (os.path.isabs(text) or "/" in text or "\\" in text or bool(re.match(r"^[A-Za-z]:|^~|^%", text)))


def preset_name(text: str) -> str:
    """The file name (without extension) a typed preset name is saved under."""
    name = strip_quotes(text or "")
    if name.lower().endswith((".toml", ".json")):
        name = name[:-5]
    return INVALID_NAME_CHARS.sub("_", name).strip(" .")


def save_target(text: str, entry: ModelEntry) -> str:
    """Absolute file the Save action writes for a name or a path."""
    text = strip_quotes(text or "")
    if not text:
        raise ValueError("Type a preset name (saved under user_presets) or a full file path first.")
    if looks_like_path(text):
        path = os.path.abspath(clean_path(text))
        if not path.lower().endswith((".toml", ".json")):
            path += ".toml"
        if path.lower().endswith(".json"):
            path = path[:-5] + ".toml"
        return path
    name = preset_name(text)
    if not name:
        raise ValueError("That preset name has no usable characters.")
    return os.path.join(user_presets_root(), entry.key, f"{name}.toml")


def find_preset(text: str, entry: ModelEntry, mode: str) -> str:
    """Resolve a typed preset name or path to an existing file (user presets first)."""
    text = strip_quotes(text or "")
    if not text:
        raise ValueError("Type a preset name or a full path, or pick a preset from the list.")
    if looks_like_path(text):
        path = os.path.abspath(clean_path(text))
        candidates = [path] if path.lower().endswith((".toml", ".json")) else [path, path + ".toml", path + ".json"]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"No preset file at {path}")
    wanted = preset_name(text).lower()  # the same cleaning Save applies: "my:preset" finds my_preset.toml
    if not wanted:
        raise FileNotFoundError(f"No preset named '{text}'.")
    search_dirs = [os.path.join(user_presets_root(), entry.key), user_presets_root()]
    for folder in search_dirs:
        for extension in (".toml", ".json"):
            candidate = os.path.join(folder, wanted + extension)
            if os.path.isfile(candidate):
                return candidate
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if os.path.splitext(name)[0].lower() == wanted and name.lower().endswith((".toml", ".json")):
                    return os.path.join(folder, name)
    for item in shipped_presets(entry, mode) + shipped_presets(entry, FULL if mode == LORA else LORA):
        if os.path.splitext(os.path.basename(item.path))[0].lower() == wanted:
            return item.path
    # other models' user presets (the guardrail will then explain the mismatch)
    root = user_presets_root()
    if os.path.isdir(root):
        for folder in sorted(os.listdir(root)):
            full = os.path.join(root, folder)
            if os.path.isdir(full):
                for name in os.listdir(full):
                    if os.path.splitext(name)[0].lower() == wanted and name.lower().endswith((".toml", ".json")):
                        return os.path.join(full, name)
    raise FileNotFoundError(f"No preset named '{text}' in {os.path.join(user_presets_root(), entry.key)} or the shipped presets.")


def write_metadata_header(path: str, entry: ModelEntry, mode: str, description: str = "", extra: dict | None = None) -> None:
    """Prepend the ``preset_*`` identity keys (top-level TOML keys must precede any table); ``extra``: more
    ``preset_*`` string values (the best checkpoint finder's settings), one line each."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        body = handle.read()
    body = "\n".join(line for line in body.splitlines() if not re.match(r"^\s*preset_\w+\s*=", line))
    header = [
        f"# SECourses Musubi Trainer preset · {entry.name} · {MODE_FOLDER_LABEL[mode]} · saved {time.strftime('%Y-%m-%d %H:%M')}",
        f"preset_format = {PRESET_FORMAT}",
        f'preset_category = "{entry.category}"',
        f'preset_model = "{entry.key}"',
        f'preset_training_type = "{mode}"',
    ]
    if description:
        header.append("preset_description = " + json.dumps(description))
    for key, value in (extra or {}).items():
        if key.startswith("preset_") and isinstance(value, str):
            header.append(f"{key} = " + json.dumps(value, ensure_ascii=False))  # a JSON string is a valid TOML basic string
    text = "\n".join(header) + "\n\n" + body.lstrip()
    if not text.endswith("\n"):
        text += "\n"
    _atomic_write(path, text)


def _atomic_write(path: str, text: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    handle, temp = tempfile.mkstemp(prefix=".tmp_", suffix=".toml", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        for attempt in range(5):
            try:
                os.replace(temp, path)
                break
            except PermissionError:  # Windows: another program (antivirus, an editor) has the file open
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(temp):
            try:
                os.remove(temp)
            except OSError:
                pass


def is_read_only_preset(path: str) -> bool:
    """Inside a shipped ``*_Training_Configs`` folder (in any letter case on Windows)."""
    return preset_library.is_shipped_preset(path)


# ------------------------------------------------------------------------------------------------
# Remembered UI state (last base model / training type / preset per category)
# ------------------------------------------------------------------------------------------------
_state_lock = threading.RLock()


def load_ui_state() -> dict:
    with _state_lock:  # never read while this process replaces the file (Windows refuses the replace)
        try:
            with open(ui_state_path(), "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}


def update_ui_state(**changes) -> None:
    """Merge ``changes`` (nested dicts merge one level deep) into ui_state.json atomically."""
    with _state_lock:
        state = load_ui_state()
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(state.get(key), dict):
                merged = dict(state[key])
                for inner_key, inner_value in value.items():
                    if isinstance(inner_value, dict) and isinstance(merged.get(inner_key), dict):
                        merged[inner_key] = {**merged[inner_key], **inner_value}
                    else:
                        merged[inner_key] = inner_value
                state[key] = merged
            else:
                state[key] = value
        try:
            _atomic_write(ui_state_path(), json.dumps(state, indent=2, ensure_ascii=False))
        except OSError:
            pass


def remembered(category: str) -> dict:
    """The category's remembered selection; anything of the wrong shape (a hand-edited or damaged
    ui_state.json) is ignored rather than stopping the app from starting."""
    data = load_ui_state().get(category)
    if not isinstance(data, dict):
        return {}
    data = dict(data)
    for key in ("entry", "mode", "tier"):
        if key in data and not isinstance(data[key], str):
            data.pop(key)
    for key in ("presets", "modes"):
        if key in data and not isinstance(data[key], dict):
            data.pop(key)
    return data


def remember_selection(category: str, entry_key: str, mode: str, tier: str | None, preset_path: str | None) -> None:
    # not the open tab: every workspace re-applies its selection when a page opens (the tab click records that)
    update_ui_state(
        **{
            category: {
                "entry": entry_key,
                "mode": mode,
                "tier": tier or AUTO_TIER,
                "presets": {f"{entry_key}:{mode}": preset_path or ""},
                "modes": {entry_key: mode},
            }
        },
    )


def remembered_preset(category: str, entry_key: str, mode: str) -> str | None:
    path = (remembered(category).get("presets") or {}).get(f"{entry_key}:{mode}")
    return path if isinstance(path, str) and path and os.path.isfile(path) else None


def remembered_mode(category: str, entry_key: str) -> str | None:
    mode = (remembered(category).get("modes") or {}).get(entry_key)
    return mode if mode in (LORA, FULL) else None
