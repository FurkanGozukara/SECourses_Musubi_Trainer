"""Ready-made GPU tier training presets shipped next to the application.

Presets live in ``<install root>/Training_Presets/<Model>_Training_Configs/`` folders (older
releases kept the ``*_Training_Configs`` folders directly in ``<install root>``; both are found). File names follow
``<GPU>_GB_GPUs_<measured>_GB_<details>.toml`` so the tier and the measured peak VRAM
can be shown without opening the file. ``MUSUBI_PRESETS_DIR`` overrides the root.

A tab can list several preset folders (e.g. FLUX.2 Dev + Klein 9B + Klein 4B); each folder, and
each ``*_Configs`` sub-folder directly inside it (Wan 2.2 T2I / T2V / I2V), is one *model family*
in the picker. The picker filters by family, training type and GPU tier, and pre-selects the
best preset for the detected GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import gradio as gr

from .common_gui import scriptdir
from .custom_logging import setup_logging

log = setup_logging()

PRESET_FOLDER_SUFFIX = "_Training_Configs"
TIER_PATTERN = re.compile(r"(?P<tier>\d+(?:\.\d+)?)_GB_GPUs?_(?P<used>\d+(?:\.\d+)?)_GB", re.IGNORECASE)
LEGACY_MB_PATTERN = re.compile(r"(?P<used>\d+)_MB", re.IGNORECASE)
QUALITY_PATTERN = re.compile(r"Quality[_ ](?P<quality>\d+)", re.IGNORECASE)
# older hand-made presets (Qwen_Training_Configs): Tier1 = best quality, then the measured MB
LEGACY_QUALITY_PATTERN = re.compile(r"(?:^|_)Tier(?P<quality>\d+)_", re.IGNORECASE)
LEGACY_FREE_VRAM_GB = 1.0  # older presets are named by the VRAM they were measured at on their target GPUs
MODE_FOLDERS = {
    "lora": "LoRA",
    "lora_training": "LoRA",
    "fine_tuning": "Fine-Tuning",
    "fine_tuning_training": "Fine-Tuning",
    "fine_tuning_training-dreambooth": "Fine-Tuning",
    "full_fine_tuning": "Fine-Tuning",
    "dreambooth": "Fine-Tuning",
}
FAMILY_NAMES = {
    "FLUX 2": "FLUX.2 Dev",
    "FLUX Klein 9B": "FLUX.2 Klein 9B",
    "FLUX Klein 4B": "FLUX.2 Klein 4B",
    "FLUX 3 Klein": "FLUX.3 Klein",
    "FLUX 3 Klein Edit": "FLUX.3 Klein Edit",
    "MOSS Image": "MOSS-Image",
    "Wan 21": "Wan 2.1",
    "Wan22": "Wan 2.2",
    "Z Image Base": "Z-Image Base",
    "Z Image Turbo": "Z-Image Turbo",
    "Qwen": "Qwen Image",
    "Qwen Image 2.1": "Qwen-Image 2.1",
    "HiDream O1": "HiDream-O1",
    "Mage Flow": "Mage-Flow",
    "Mage Flow Edit": "Mage-Flow-Edit",
    "ERNIE Image": "ERNIE-Image",
}
ALL_TIERS = "all"


@dataclass(frozen=True)
class PresetEntry:
    path: str
    label: str
    mode: str
    tier_gb: float | None
    used_gb: float | None
    description: str
    family: str = ""
    quality: int | None = None
    low_ram: bool = False

    @property
    def sort_key(self):
        return (self.mode != "LoRA", self.mode, self.tier_gb if self.tier_gb is not None else 999.0, self.label)

    @property
    def preference(self):
        """Lower is better: best quality first, regular before Low-RAM variants, then the faster (bigger) preset."""
        return (self.quality if self.quality is not None else 99, self.low_ram, -(self.used_gb or 0.0), self.label)


TRAINING_PRESETS_DIRNAME = "Training_Presets"


def preset_roots() -> list[str]:
    """Folders that hold ``*_Training_Configs`` preset folders, searched in order.

    ``<install root>/Training_Presets`` first, then ``<install root>`` itself (older releases kept
    the preset folders there). ``MUSUBI_PRESETS_DIR`` replaces both.
    """
    override = os.environ.get("MUSUBI_PRESETS_DIR", "").strip()
    if override:
        return [os.path.abspath(override)]
    install_root = os.path.abspath(os.path.join(scriptdir, ".."))
    roots = [os.path.join(install_root, TRAINING_PRESETS_DIRNAME), install_root]
    existing = [root for root in roots if os.path.isdir(root)]
    return existing or [install_root]


def preset_root() -> str:
    return preset_roots()[0]


def preset_folder_paths(folders: list[str] | tuple[str, ...]) -> list[str]:
    """Absolute path of each preset folder: the first root that has it (else the main root)."""
    roots = preset_roots()
    paths = []
    for folder in folders:
        found = next((os.path.join(root, folder) for root in roots if os.path.isdir(os.path.join(root, folder))), None)
        paths.append(found or os.path.join(roots[0], folder))
    return paths


def is_shipped_preset(path: str | os.PathLike | None) -> bool:
    """True for files inside a shipped ``*_Training_Configs`` folder (treated as read-only)."""
    if not path:
        return False
    # normcase: Windows paths are case-insensitive, so "g:/..._training_configs/x.toml" is the same shipped file
    absolute = os.path.normcase(os.path.abspath(str(path)))
    for root in preset_roots():
        try:
            relative = os.path.relpath(absolute, os.path.normcase(root))
        except ValueError:  # different drive on Windows
            continue
        if relative.startswith(".."):
            continue
        first = relative.replace("\\", "/").split("/", 1)[0]
        if first.lower().endswith(PRESET_FOLDER_SUFFIX.lower()):
            return True
    return False


def _read_description(path: str) -> str:
    """First ``# description:`` style comment (or first comment line) of a preset."""
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            for _ in range(8):
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                text = stripped.lstrip("#").strip()
                if text.lower().startswith("description:"):
                    return text.split(":", 1)[1].strip()
                return text
    except OSError:
        return ""
    return ""


def _mode_for(relative_parts: list[str]) -> str:
    for part in relative_parts:
        mode = MODE_FOLDERS.get(part.strip().lower())
        if mode:
            return mode
    joined = " ".join(relative_parts).lower()
    if "fine" in joined or "dreambooth" in joined:
        return "Fine-Tuning"
    return "LoRA"


def _pretty(name: str) -> str:
    for suffix in (PRESET_FOLDER_SUFFIX, "_Configs"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.replace("_", " ").strip()
    return FAMILY_NAMES.get(name, name)


def family_for(folder: str, relative_parts: list[str]) -> str:
    """Display name of the model family a preset belongs to."""
    top = os.path.basename(os.path.normpath(folder))
    family = _pretty(top)
    if relative_parts and MODE_FOLDERS.get(relative_parts[0].lower()) is None and relative_parts[0].endswith("_Configs"):
        sub = relative_parts[0][: -len("_Configs")].replace("_", " ").strip()
        raw_top = top[: -len(PRESET_FOLDER_SUFFIX)] if top.endswith(PRESET_FOLDER_SUFFIX) else top
        prefix = raw_top.replace("_", " ").strip()
        if sub.lower().startswith(prefix.lower() + " "):
            sub = sub[len(prefix) + 1 :]
        family = f"{family} · {sub}"
    return family


def _entry_for(folder: str, path: str) -> PresetEntry:
    relative = os.path.relpath(path, folder)
    parts = relative.replace("\\", "/").split("/")
    filename = os.path.splitext(parts[-1])[0]
    mode = _mode_for(parts[:-1])
    family = family_for(folder, parts[:-1])
    match = TIER_PATTERN.search(filename)
    tier = used = None
    if match:
        tier = float(match.group("tier"))
        used = float(match.group("used"))
    else:
        legacy = LEGACY_MB_PATTERN.search(filename)
        if legacy:
            used = round(int(legacy.group("used")) / 1024, 2)
    details = filename
    if match:
        details = filename[match.end() :].strip("_ ")
    details = details.replace("_", " ").strip()
    family_parts = 1 if " · " in family else 0
    subgroup = " / ".join(
        part.replace("_", " ") for index, part in enumerate(parts[:-1]) if MODE_FOLDERS.get(part.lower()) is None and index >= family_parts
    )
    quality_match = QUALITY_PATTERN.search(filename) or LEGACY_QUALITY_PATTERN.search(filename)
    tier_text = f"{tier:g} GB GPUs" if tier is not None else filename.replace("_", " ")
    used_text = f" · peak {used:g} GB" if used is not None else ""
    pieces = [mode, tier_text + used_text]
    if subgroup:
        pieces.append(subgroup)
    if details and match:
        pieces.append(details)
    return PresetEntry(
        path=os.path.abspath(path),
        label=" · ".join(pieces),
        mode=mode,
        tier_gb=tier,
        used_gb=used,
        description=_read_description(path),
        family=family,
        quality=int(quality_match.group("quality")) if quality_match else None,
        low_ram="low_ram" in filename.lower(),
    )


def _is_training_preset(name: str, directory_parts: list[str]) -> bool:
    """Training presets only: example datasets shipped beside them hold dataset_config*.toml files."""
    if not name.lower().endswith(".toml") or name.lower().startswith("dataset_config"):
        return False
    return not any(part.lower().endswith("_dataset") for part in directory_parts)


@lru_cache(maxsize=64)
def _scan(folder: str, mtime_key: float) -> tuple[PresetEntry, ...]:
    entries: list[PresetEntry] = []
    for current, dirs, files in os.walk(folder):
        dirs.sort()
        relative_dir = os.path.relpath(current, folder)
        directory_parts = [] if relative_dir == "." else relative_dir.replace("\\", "/").split("/")
        for name in sorted(files):
            if _is_training_preset(name, directory_parts):
                entries.append(_entry_for(folder, os.path.join(current, name)))
    return tuple(sorted(entries, key=lambda entry: entry.sort_key))


def _folder_mtime(folder: str) -> float:
    """Newest mtime of the folder tree, files included (an edited preset description must show up too)."""
    latest = 0.0
    for current, _dirs, files in os.walk(folder):
        for path in [current] + [os.path.join(current, name) for name in files if name.lower().endswith(".toml")]:
            try:
                latest = max(latest, os.path.getmtime(path))
            except OSError:
                continue
    return latest


def list_presets(folders: list[str] | tuple[str, ...], mode: str | None = None) -> list[PresetEntry]:
    entries: list[PresetEntry] = []
    for folder in preset_folder_paths(folders):
        if os.path.isdir(folder):
            entries.extend(_scan(folder, _folder_mtime(folder)))
    if mode:
        entries = [entry for entry in entries if entry.mode == mode]
    return entries


def filter_presets(entries: list[PresetEntry], family: str | None = None, mode: str | None = None, tier: str | None = None) -> list[PresetEntry]:
    result = entries
    if family:
        result = [entry for entry in result if entry.family == family]
    if mode and mode != "All":
        result = [entry for entry in result if entry.mode == mode]
    if tier and tier != ALL_TIERS:
        try:
            wanted = float(tier)
        except ValueError:
            wanted = None
        if wanted is not None:
            result = [entry for entry in result if entry.tier_gb is not None and abs(entry.tier_gb - wanted) < 0.01]
    return result


def preset_choices(folders, mode: str | None = None) -> list[tuple[str, str]]:
    return [(entry.label, entry.path) for entry in list_presets(folders, mode)]


@lru_cache(maxsize=1)
def detect_gpus() -> tuple[tuple[str, float], ...]:
    """(name, total GB) per GPU via nvidia-smi - never creates a CUDA context in the GUI process."""
    import subprocess

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            creationflags=0x08000000 if os.name == "nt" else 0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                gpus.append((parts[0], round(float(parts[1]) / 1024, 1)))
            except ValueError:
                continue
    return tuple(gpus)


def best_tier_for(entries: list[PresetEntry], vram_gb: float | None) -> str:
    """Largest available tier that fits the GPU (string choice value), or ``all``."""
    tiers = sorted({entry.tier_gb for entry in entries if entry.tier_gb is not None})
    if not tiers or vram_gb is None:
        return ALL_TIERS
    fitting = [tier for tier in tiers if tier <= vram_gb + 0.5]
    return f"{(fitting[-1] if fitting else tiers[0]):g}"


def best_preset(entries: list[PresetEntry]) -> PresetEntry | None:
    measured = [entry for entry in entries if entry.tier_gb is not None]
    return min(measured, key=lambda entry: entry.preference) if measured else None


def default_preset(entries: list[PresetEntry], vram_gb: float | None) -> PresetEntry | None:
    """Pre-selected preset: the best tier preset, else (older names without a GPU tier, e.g.
    ``Tier2_14750_MB``) the best-quality one whose measured VRAM leaves 1 GB of this GPU free, else
    the only preset there is (e.g. the unmeasured Ideogram 4 BF16 fine-tuning preset)."""
    best = best_preset(entries)
    if best is not None:
        return best
    fitting = []
    if vram_gb is not None:
        fitting = [entry for entry in entries if entry.used_gb is not None and entry.used_gb <= vram_gb - LEGACY_FREE_VRAM_GB]
    if fitting:
        return min(fitting, key=lambda entry: entry.preference)
    return entries[0] if len(entries) == 1 else None


def gpu_label(name: str) -> str:
    """'NVIDIA GeForce RTX 5090' -> 'RTX 5090' (keeps the picker's info line short)."""
    for prefix in ("NVIDIA ", "GeForce "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.strip()


def recommended_preset(folders, gpu_index: int = 0, mode: str = "LoRA", family: str | None = None) -> PresetEntry | None:
    """Best preset of ``mode`` (and ``family``) for the largest tier that fits the GPU."""
    gpus = detect_gpus()
    if not gpus or gpu_index >= len(gpus):
        return None
    entries = filter_presets(list_presets(folders), family=family, mode=mode)
    tier = best_tier_for(entries, gpus[gpu_index][1])
    if tier == ALL_TIERS:
        return None
    return best_preset(filter_presets(entries, tier=tier))


def describe_preset(folders, path: str | None) -> str:
    if not path:
        return ""
    for entry in list_presets(folders):
        if os.path.normcase(entry.path) == os.path.normcase(os.path.abspath(path)):
            parts = [f"**{entry.family} — {entry.label}**" if entry.family else f"**{entry.label}**"]
            if entry.description:
                parts.append(entry.description)
            parts.append(f"`{os.path.relpath(entry.path, preset_root())}`")
            return "  \n".join(parts)
    return f"`{path}`"


def _tier_choices(entries: list[PresetEntry]) -> list[tuple[str, str]]:
    tiers = sorted({entry.tier_gb for entry in entries if entry.tier_gb is not None})
    return [("All GPUs", ALL_TIERS)] + [(f"{tier:g} GB", f"{tier:g}") for tier in tiers]


def _families(entries: list[PresetEntry]) -> list[str]:
    return list(dict.fromkeys(entry.family for entry in entries))


def _modes(entries: list[PresetEntry]) -> list[str]:
    return sorted({entry.mode for entry in entries}, key=lambda mode: (mode != "LoRA", mode))


def _default_mode(modes: list[str]) -> str:
    return "LoRA" if "LoRA" in modes else "All"


class PresetLibrary:
    """Preset picker shown at the top of a trainer tab.

    ``bind()`` wires *Apply Preset* to the tab's own Load handler: the preset path is first
    written into the tab's config-file dropdown, then the same load function runs, so presets
    behave exactly like loading that TOML through the 📥 Load button.
    """

    def __init__(self, folders: list[str] | tuple[str, ...], *, model_name: str, headless: bool = False):
        self.folders = tuple(folders)
        self.model_name = model_name
        entries = list_presets(self.folders)
        has_presets = bool(entries)
        families = _families(entries)
        modes = _modes(entries)
        gpus = detect_gpus()
        vram = gpus[0][1] if gpus else None

        default_family = families[0] if families else None
        default_mode = _default_mode(modes)
        scoped = filter_presets(entries, family=default_family, mode=default_mode)
        default_tier = best_tier_for(scoped, vram)
        visible = filter_presets(scoped, tier=default_tier)
        recommended = default_preset(visible, vram)
        folder_text = ", ".join(f"`{folder}`" for folder in self.folders)
        gpu_info = f"Detected: {gpu_label(gpus[0][0])} ({vram:g} GB)" if gpus else "No NVIDIA GPU detected"

        with gr.Group(elem_classes=["preset-library"]):
            # the filters only appear when there is something to choose between
            with gr.Row(equal_height=True, visible=len(families) > 1 or len(modes) > 1) as self.filter_row:
                self.family = gr.Dropdown(
                    label="Model",
                    choices=families or [""],
                    value=default_family,
                    visible=len(families) > 1,
                    interactive=True,
                    scale=3,
                    min_width=200,
                )
                self.mode_filter = gr.Radio(
                    label="Training Type",
                    choices=["All"] + modes,
                    value=default_mode,
                    visible=len(modes) > 1,
                    scale=3,
                    min_width=240,
                    interactive=True,
                )
            with gr.Row(equal_height=False, elem_classes=["preset-row"]):
                # older presets without GPU tiers (Qwen Image) have nothing to filter by
                self.tier = gr.Dropdown(
                    label="Your GPU VRAM",
                    choices=_tier_choices(scoped),
                    value=default_tier,
                    visible=len(_tier_choices(scoped)) > 1,
                    scale=2,
                    min_width=180,
                    interactive=True,
                    info=gpu_info,
                )
                self.preset = gr.Dropdown(
                    label=f"⚡ {model_name} GPU Presets",
                    choices=[(entry.label, entry.path) for entry in visible],
                    value=recommended.path if recommended else None,
                    scale=6,
                    min_width=280,
                    interactive=True,
                    allow_custom_value=False,
                    info=(
                        "Measured presets: every one keeps at least 1.5 GB of VRAM free. Apply loads it like the 📥 Load button."
                        if any(entry.tier_gb is not None for entry in entries)
                        else "Researched presets, named by the VRAM they use. Apply loads it like the 📥 Load button."
                        if has_presets
                        else f"No presets found yet in {folder_text} beside the app folder."
                    ),
                )
                self.apply_button = gr.Button(
                    "⚡ Apply Preset",
                    variant="primary",
                    scale=0,
                    min_width=170,
                    elem_classes=["mbtn", "mbtn-emerald", "preset-action"],
                    interactive=True,
                )
                self.refresh_button = gr.Button(
                    "🔄",
                    scale=0,
                    min_width=48,
                    elem_classes=["mbtn", "mbtn-slate", "preset-action"],
                )
            initial_details = describe_preset(self.folders, recommended.path) if recommended else ""
            self.details = gr.Markdown(value=initial_details, visible=bool(initial_details), elem_classes=["preset-details"])

        def scope(family, mode, tier, *, keep_tier: bool):
            all_entries = list_presets(self.folders)
            scoped_entries = filter_presets(all_entries, family=family or None, mode=mode)
            tier_value = tier if keep_tier else best_tier_for(scoped_entries, vram)
            valid_tiers = {value for _label, value in _tier_choices(scoped_entries)}
            if tier_value not in valid_tiers:
                tier_value = best_tier_for(scoped_entries, vram)
            shown = filter_presets(scoped_entries, tier=tier_value)
            best = default_preset(shown, vram)
            tier_choices = _tier_choices(scoped_entries)
            return (
                gr.Dropdown(choices=tier_choices, value=tier_value, visible=len(tier_choices) > 1),
                gr.Dropdown(choices=[(entry.label, entry.path) for entry in shown], value=best.path if best else None),
                gr.Markdown(value=describe_preset(self.folders, best.path) if best else "", visible=bool(best)),
            )

        def refresh(family, mode, tier):
            # presets added or edited while the app runs: rebuild the filters as well
            all_entries = list_presets(self.folders)
            current_families, current_modes = _families(all_entries), _modes(all_entries)
            if family not in current_families:
                family = current_families[0] if current_families else None
            if mode not in ["All"] + current_modes:
                mode = _default_mode(current_modes)
            return (
                gr.Row(visible=len(current_families) > 1 or len(current_modes) > 1),
                gr.Dropdown(choices=current_families or [""], value=family, visible=len(current_families) > 1),
                gr.Radio(choices=["All"] + current_modes, value=mode, visible=len(current_modes) > 1),
                *scope(family, mode, tier, keep_tier=True),
            )

        self._scope = scope
        filters = [self.family, self.mode_filter, self.tier]
        outputs = [self.tier, self.preset, self.details]
        self.family.input(lambda f, m, t: scope(f, m, t, keep_tier=False), inputs=filters, outputs=outputs, show_progress=False)
        self.mode_filter.input(lambda f, m, t: scope(f, m, t, keep_tier=False), inputs=filters, outputs=outputs, show_progress=False)
        self.tier.input(lambda f, m, t: scope(f, m, t, keep_tier=True), inputs=filters, outputs=outputs, show_progress=False)
        self.refresh_button.click(
            refresh, inputs=filters, outputs=[self.filter_row, self.family, self.mode_filter, *outputs], show_progress=False
        )
        self.preset.input(
            lambda path: gr.Markdown(value=describe_preset(self.folders, path), visible=bool(path)),
            inputs=[self.preset],
            outputs=[self.details],
            show_progress=False,
        )

    def follow_training_mode(self, training_mode):
        """Keep the Training Type filter on the mode the user picks in the tab's own Training Mode radio.

        Only user changes (``.input``) are followed, so applying a preset never re-filters the list it came from.
        """
        return training_mode.input(
            self.follow_mode,
            inputs=[training_mode, self.family, self.tier],
            outputs=[self.mode_filter, self.tier, self.preset, self.details],
            show_progress=False,
        )

    def follow_mode(self, mode, family, tier):
        """Updates for (Training Type, GPU VRAM, preset, details) after the tab's Training Mode changed."""
        wanted = "Fine-Tuning" if "fine" in str(mode or "").lower() else "LoRA"
        if wanted not in _modes(filter_presets(list_presets(self.folders), family=family or None)):
            return (gr.update(),) * 4  # no presets of that kind: leave the picker as it is
        return (gr.Radio(value=wanted), *self._scope(family, wanted, tier, keep_tier=True))

    def bind(self, config_file_name, load_fn=None, inputs=None, outputs=None, then_fn=None, then_inputs=None, then_outputs=None):
        """Wire *Apply Preset*. With ``load_fn=None`` only the config dropdown is set (for tabs that
        already auto-load on ``config_file_name.change``)."""

        def select(path):
            if not path:
                raise gr.Error("Select a preset first.", duration=6)
            if not os.path.isfile(path):
                raise gr.Error(f"Preset file not found: {path}", duration=8)
            return path

        event = self.apply_button.click(select, inputs=[self.preset], outputs=[config_file_name], queue=False)
        if load_fn is None:
            return event
        event = event.success(load_fn, inputs=inputs, outputs=outputs, show_progress=False)
        if then_fn is not None:
            event = event.then(then_fn, inputs=then_inputs or [], outputs=then_outputs or [], show_progress=False)
        return event
