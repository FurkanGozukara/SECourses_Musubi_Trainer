"""One training workspace (Image / Video / Audio): base model → training type → GPU preset → settings.

Every setting of every base model in the category exists once as a shared field.

Speed design (measured on Gradio 6.30): the browser looks up every updated component with a walk
of the whole layout tree, and an update to a component hidden with ``visible=False`` additionally
re-walks the whole app tree twice. So fields are never hidden through Gradio. Each field carries
static CSS classes (which trainers use it, LoRA-only / fine-tune-only, advanced, locked, dependent
on a controller) and generated CSS rules show or hide it from ``data-*`` attributes on the
workspace root. Switching model, training type, "show advanced" or a controller such as the
dataset mode only changes those attributes (instantly, in the browser); Python sends nothing but
the labels, choices and values that really differ (``emit`` diffs against a per-session mirror of
what the browser shows and against the values it just sent).
"""

from __future__ import annotations

import html
import itertools
import json
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import gradio as gr

from ..best_checkpoint import prompts as BP
from ..best_checkpoint.samples import index_samples
from ..custom_logging import setup_logging
from . import checkpoint_finder as CF
from . import links as L
from . import presets as P
from .backends import Backend, get_backend, read_preset_file
from .dataset_cleanup import remove_cache_folders
from .dataset_inspector import (
    DatasetReport,
    clean_plan,
    clean_question_html,
    clean_result_html,
    control_settings,
    inspect_dataset,
    report_html,
    sample_previews,
)
from .dataset_tools import dataset_profile
from .downloads import DOWNLOADS
from .param_spec import FULL, LORA, OPEN_SECTIONS, SECTIONS, ParamSpec
from .registry import (
    CATEGORY_TITLES,
    DEFAULT_ENTRY,
    MODE_DESCRIPTIONS,
    MODE_LABELS,
    ModelEntry,
    entries_for,
    get_entry,
)
from .schema import APP_DIR, HIDDEN_KEYS, ROOT_DIR
from .values import clean_path
from .vram_advice import WATCHED_KEYS as VRAM_KEYS
from .vram_advice import advice_html, advise

log = setup_logging()

FIELD_CLASS = "ws-field"
HOST_CLASS = "vh"  # element whose display the CSS rules toggle (a component, or a path row)
NORMAL_MIN_WIDTH = 260
CHECK_MIN_WIDTH = 230
MAX_SESSIONS = 64
_MISSING = object()


def token(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", str(name))


# ------------------------------------------------------------------------------------------------
# Dependent fields: (rule id, controller key, dependent keys, Python predicate, JS predicate on v).
# A rule only applies when the active trainer has the controller key.
# ------------------------------------------------------------------------------------------------
DATASET_FOLDER_KEYS = (
    "parent_folder_path",
    "dataset_resolution_width",
    "dataset_resolution_height",
    "dataset_caption_extension",
    "create_missing_captions",
    "caption_strategy",
    "dataset_batch_size",
    "dataset_enable_bucket",
    "dataset_bucket_no_upscale",
    "dataset_cache_directory",
    "target_frames",
    "auto_normalize_target_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
    "source_fps",
    "dataset_control_directory",
    "dataset_qwen_image_edit_no_resize_control",
    "dataset_qwen_image_edit_control_resolution_width",
    "dataset_qwen_image_edit_control_resolution_height",
    "auto_generate_black_control_images",
)
CLIP_TASKS = ("i2v-14B", "flf2v-14B", "i2v-14B-FC")
# timestep samplings that use --sigmoid_scale (musubi training/timestep_sampling.py, sd-scripts flux_train_utils)
SIGMOID_SCALE_SAMPLINGS = frozenset({"sigmoid", "shift", "flux_shift", "flux2_shift", "qwen_shift", "krea2_shift", "qinglong_flux", "qinglong_qwen"})


def _as_float(value) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


RULES: tuple = (
    ("dsfolder", "dataset_config_mode", DATASET_FOLDER_KEYS, lambda v: str(v or "") != "Use TOML File", "String(v) !== 'Use TOML File'"),
    ("dstoml", "dataset_config_mode", ("dataset_config",), lambda v: str(v or "") == "Use TOML File", "String(v) === 'Use TOML File'"),
    ("captions", "create_missing_captions", ("caption_strategy",), bool, "!!v"),
    ("a14b", "task", ("dit_high_noise", "timestep_boundary", "offload_inactive_dit"), lambda v: "A14B" in str(v or ""), "String(v).includes('A14B')"),
    ("clip", "task", ("clip",), lambda v: str(v or "") in CLIP_TASKS, "['i2v-14B','flf2v-14B','i2v-14B-FC'].includes(String(v))"),
    ("netcustom", "network_module", ("custom_network_module",), lambda v: str(v or "") == "custom", "String(v) === 'custom'"),
    ("diffusers", "convert_to_diffusers", ("diffusers_output_dir",), bool, "!!v"),
    ("safet", "convert_to_safetensors", ("safetensors_output_dir",), bool, "!!v"),
    (
        "compile",
        "compile",
        ("compile_backend", "compile_mode", "compile_dynamic", "compile_fullgraph", "compile_cache_size_limit", "compile_resident_blocks_only"),
        bool,
        "!!v",
    ),
    ("wandb", "log_with", ("wandb_run_name", "wandb_api_key"), lambda v: str(v or "") in ("wandb", "all"), "['wandb','all'].includes(String(v))"),
    ("swap", "blocks_to_swap", ("use_pinned_memory_for_block_swap", "block_swap_h2d_only", "block_swap_ring_size"), lambda v: _as_float(v) > 0, "Number(v) > 0"),
    ("fp8", "fp8_base", ("fp8_scaled",), bool, "!!v"),
    ("control", "use_control_images", ("control_directory_name", "no_resize_control", "control_resolution_width", "control_resolution_height"), bool, "!!v"),
    # Ideogram 4: warning instead of failing only applies while captions are validated (the save turns it off)
    ("capwarn", "validate_caption_structure", ("warn_on_caption_issues",), bool, "!!v"),
    # settings the trainers only read (or only write to the runtime TOML) in some cases (tools/audit_parameter_passthrough.py)
    (
        "sigscale",
        "timestep_sampling",
        ("sigmoid_scale",),
        lambda v: str(v or "") in SIGMOID_SCALE_SAMPLINGS,
        "['sigmoid','shift','flux_shift','flux2_shift','qwen_shift','krea2_shift','qinglong_flux','qinglong_qwen'].includes(String(v))",
    ),
    ("modescale", "weighting_scheme", ("mode_scale",), lambda v: str(v or "") == "mode", "String(v) === 'mode'"),
    ("bweights", "base_weights", ("base_weights_multiplier",), lambda v: bool(str(v or "").strip()), "String(v || '').trim() !== ''"),
    ("workers", "max_data_loader_n_workers", ("persistent_data_loader_workers",), lambda v: _as_float(v) > 0, "Number(v) > 0"),
)
RULES_BY_DEPENDENT: dict[str, list[str]] = {}
for _rule_id, _controller, _dependents, _predicate, _js in RULES:
    for _key in _dependents:
        RULES_BY_DEPENDENT.setdefault(_key, []).append(_rule_id)
CONTROLLER_KEYS = tuple(dict.fromkeys(rule[1] for rule in RULES))
DEPENDENT_KEYS = frozenset(RULES_BY_DEPENDENT)


def rule_states(values: dict, backend_keys: set) -> dict[str, str]:
    """data-r-<id> attribute values: "0" hides the rule's dependents."""
    states = {}
    for rule_id, controller, _dependents, predicate, _js in RULES:
        if controller not in backend_keys:
            states[rule_id] = "1"
            continue
        try:
            states[rule_id] = "1" if predicate(values.get(controller)) else "0"
        except Exception:
            states[rule_id] = "1"
    return states


def rules_allow(key: str, values: dict, backend_keys: set) -> bool:
    if key not in DEPENDENT_KEYS:
        return True
    states = rule_states(values, backend_keys)
    return all(states.get(rule_id, "1") == "1" for rule_id in RULES_BY_DEPENDENT[key])


# ------------------------------------------------------------------------------------------------
# Lenient validation for unified fields: a field hidden for the current model may hold another
# model's value; the trainer code validates what it uses, so Gradio must not reject the event.
# ------------------------------------------------------------------------------------------------
_GUARD_INSTALLED = False


def install_lenient_validation() -> None:
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    _GUARD_INSTALLED = True

    def is_unified(component) -> bool:
        classes = getattr(component, "elem_classes", None) or []
        return FIELD_CLASS in (classes if isinstance(classes, (list, tuple)) else [classes])

    for component_cls in (gr.Dropdown, gr.Radio, gr.Number, gr.Slider):
        original = component_cls.preprocess

        def lenient(self, payload, _original=original):
            if payload is None or not is_unified(self):
                return _original(self, payload)
            try:
                return _original(self, payload)
            except gr.Error:
                return payload

        component_cls.preprocess = lenient


# ------------------------------------------------------------------------------------------------
# Field model
# ------------------------------------------------------------------------------------------------
def _spec_family(spec: ParamSpec) -> str:
    family = spec.extras.get("family") if spec.extras else None
    if family:
        return family
    if spec.kind == "checkbox":
        return "bool"
    if spec.kind in ("number", "slider"):
        return "number"
    if spec.multiselect:
        return "list"
    return "str"


@dataclass
class FieldDef:
    uid: str
    key: str
    family: str
    kind: str
    section: str
    priority: int
    order: int
    allow_custom: bool = False
    per_backend: dict = field(default_factory=dict)  # backend name -> ParamSpec
    host_classes: list = field(default_factory=list)
    bounds: tuple = (None, None, 1)  # canonical (minimum, maximum, step) for plain number fields
    label: str = ""  # canonical label / info shared by every trainer ...
    info: str = ""
    text_overrides: dict = field(default_factory=dict)  # ... except backend -> (label, info) where it is model-specific
    component: object = None
    container: object = None  # path row
    browse: object = None


@dataclass
class Loaded:
    """Settings for one model / training type, from its defaults and (optionally) a preset."""

    values: dict
    warnings: list
    defaults: dict  # the model's own defaults (a shipped preset's machine paths fall back to these)
    error: str = ""  # the preset could not be read at all: nothing of it was applied


def _signature(path: str | None):
    """What identifies one version of a preset file (a re-saved preset loads again on page reload)."""
    try:
        return (os.path.getmtime(path), os.path.getsize(path)) if path else None
    except OSError:
        return None


def _unified_kind(specs: list[ParamSpec], family: str, key: str) -> tuple[str, bool]:
    kinds = {spec.kind for spec in specs}
    allow_custom = any(spec.allow_custom_value for spec in specs)
    if family == "bool":
        return "checkbox", False
    if family == "number":
        return ("slider" if (kinds == {"slider"} or key == "blocks_to_swap") else "number"), False
    if family == "list":
        return "dropdown", True
    path = next((kind for kind in ("folder", "file") if kind in kinds), None)
    if path:
        return path, False
    if "dropdown" in kinds:
        return "dropdown", allow_custom or bool(kinds & {"text", "textarea"})
    if kinds == {"radio"}:
        return "radio", False
    if "radio" in kinds:
        return "dropdown", allow_custom
    if "textarea" in kinds:
        return "textarea", False
    return "text", False


def _choices_with(choices, value) -> list:
    """Choices list that contains ``value`` (added as a custom entry when missing)."""
    normalized = [list(choice) if isinstance(choice, (list, tuple)) else [str(choice), choice] for choice in (choices or ())]
    values = [choice[1] for choice in normalized]
    for item in value if isinstance(value, list) else [value]:
        if item not in (None, "") and item not in values:
            normalized.append([f"{item} (custom)", item])
            values.append(item)
    return normalized


def _same(a, b) -> bool:
    if a is b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if a in (None, "") and b in (None, ""):
        return True
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# Settings whose label / info name the model (model files, block-swap limits, variant selectors):
# these keep each trainer's own text. Everything else shows one canonical text, so switching models
# does not have to re-send hundreds of labels.
PER_BACKEND_TEXT_KEYS = frozenset(
    {
        "dit", "dit_high_noise", "vae", "video_vae", "audio_vae", "vocoder", "text_encoder", "text_encoder_qwen",
        "text_encoder_clip", "t5", "clip", "ltx2_checkpoint", "gemma_root", "gemma_safetensors", "training_adapter_path",
        "turbo_dit", "unconditional_dit", "blocks_to_swap", "task", "model_type", "model_version", "model_variant",
        "ltx2_mode", "model_family", "zimage_variant", "pretrained_model_name_or_path", "ae", "clip_l", "t5xxl",
    }
)
MODEL_NAME_RE = re.compile(r"\b(krea|qwen|wan|flux|klein|z-?image|ideogram|hidream|ernie|mage-?flow|kandinsky|ltx|minimax|h3|sdxl|gemma)\b", re.I)


def _is_raw_key_label(label: str, key: str) -> bool:
    return label.strip().lower() == key.lower() or ("_" in label and " " not in label)


def _field_priority(key: str, section: str, items: list) -> int:
    """Position of a shared field: the canonical section order, else the musubi trainers' order (the kohya
    backend numbers its settings on its own scale, which would pull shared fields to the top)."""
    from .schema import SECTION_KEYS

    ordered = SECTION_KEYS.get(section, [])
    if key in ordered:
        return ordered.index(key) * 10
    musubi = [spec.priority for name, spec, _order in items if not name.startswith("kohya:")]
    return min(musubi or [spec.priority for _name, spec, _order in items])


def canonical_text(key: str, specs: dict) -> tuple[str, str, dict]:
    """(label, info, overrides) for one shared field; overrides: backend -> (label, info)."""
    from collections import Counter

    labels = Counter(spec.label for spec in specs.values() if spec.label and not _is_raw_key_label(spec.label, key))
    label = labels.most_common(1)[0][0] if labels else next(iter(specs.values())).label or key.replace("_", " ").title()
    # the musubi trainers' texts first: a kohya text can describe kohya's own presets ("Off (presets)")
    candidates = [spec for name, spec in specs.items() if not name.startswith("kohya:")] or list(specs.values())
    generic_infos = Counter(spec.info for spec in candidates if spec.info and not MODEL_NAME_RE.search(spec.info))
    info = ""
    if generic_infos:
        top = generic_infos.most_common()
        best_count = top[0][1]
        info = max((text for text, count in top if count == best_count), key=len)
    overrides = {}
    for name, spec in specs.items():
        if key in PER_BACKEND_TEXT_KEYS:
            own_label = spec.label if spec.label and not _is_raw_key_label(spec.label, key) else label
            overrides[name] = (own_label, spec.info or info)
        elif spec.info and MODEL_NAME_RE.search(spec.info):
            overrides[name] = (label, spec.info)
    return label, info, overrides


def _model_file_exists(path: str) -> bool:
    """A model path as training reads it: a relative one (the presets' Training_Models_*/... convention) is also found
    under the app and install folders. A preset loaded before its models were downloaded keeps its relative paths, and
    ⬇️ Download puts the files there."""
    if os.path.exists(path):
        return True
    if os.path.isabs(os.path.expandvars(os.path.expanduser(path))):
        return False
    return any(os.path.exists(os.path.join(base, path)) for base in (APP_DIR, ROOT_DIR))


class Workspace:
    def __init__(self, category: str, *, headless: bool = False):
        self.category = category
        self.headless = headless
        self.root_id = f"ws-{category}"
        self.backends: dict[str, Backend] = {}
        for entry in entries_for(category):
            if entry.backend not in self.backends:
                try:
                    backend = get_backend(entry.backend)
                    backend.specs  # noqa: B018 - load now so a broken backend is skipped at start
                    self.backends[entry.backend] = backend
                except Exception as exc:
                    log.error(f"{category}: backend {entry.backend} unavailable: {exc}")
        self.entries: list[ModelEntry] = [entry for entry in entries_for(category) if entry.backend in self.backends]
        self.entry_keys = [entry.key for entry in self.entries]
        self.fields: "OrderedDict[str, FieldDef]" = OrderedDict()
        self.backend_uid: dict[str, dict[str, str]] = {name: {} for name in self.backends}
        self._collect_fields()
        # settings the classic tabs change when another one changes (links.py), by the trigger's field
        self.link_groups: dict[str, list[L.Link]] = {}
        for link in L.LINKS:
            uid = self.backend_uid.get(link.backend, {}).get(link.trigger)
            if uid is not None:
                self.link_groups.setdefault(uid, []).append(link)
        self.sections: dict = {}
        self.section_fields: dict[str, list[FieldDef]] = {section_id: [] for section_id, _t, _d in SECTIONS}
        for item in self.fields.values():
            self.section_fields.setdefault(item.section, []).append(item)
        for items in self.section_fields.values():
            items.sort(key=lambda item: (item.priority, item.order))
        self.baseline: dict[int, dict] = {}
        self._mirrors: "OrderedDict[str, dict]" = OrderedDict()
        self._session_locks: "OrderedDict[str, threading.RLock]" = OrderedDict()
        self._reports: "OrderedDict[str, tuple]" = OrderedDict()  # last dataset check per session (New samples)
        self._clean_plans: "OrderedDict[str, dict]" = OrderedDict()  # what 🧹 Clean asked about, per session
        self._vram_base: "OrderedDict[str, tuple]" = OrderedDict()  # what each session loaded last (VRAM panel)
        self._mirror_lock = threading.Lock()
        self._seq = itertools.count(1)
        self.finder = CF.available(category)  # 🏆 Best Checkpoint Finder: image and video models
        self.finder_inputs: list = []

    # ------------------------------------------------------------------ field union + CSS classes
    def _collect_fields(self) -> None:
        by_key: "OrderedDict[str, OrderedDict]" = OrderedDict()
        order = 0
        for name, backend in self.backends.items():
            for spec in backend.specs:
                by_key.setdefault(spec.key, OrderedDict()).setdefault(_spec_family(spec), []).append((name, spec, order))
                order += 1
        for key, families in by_key.items():
            conflicting = len(families) > 1
            for family, items in families.items():
                uid = f"{key}#{family}" if conflicting else key
                specs = [spec for _name, spec, _order in items]
                kind, allow_custom = _unified_kind(specs, family, key)
                sections = [spec.section for spec in specs]
                section = max(set(sections), key=sections.count)
                definition = FieldDef(
                    uid=uid,
                    key=key,
                    family=family,
                    kind=kind,
                    section=section,
                    priority=_field_priority(key, section, items),
                    order=min(item_order for _name, _spec, item_order in items),
                    allow_custom=allow_custom or any(spec.allow_custom_value for spec in specs),
                )
                for name, spec, _order in items:
                    definition.per_backend[name] = spec
                    self.backend_uid[name][key] = uid
                definition.label, definition.info, definition.text_overrides = canonical_text(key, definition.per_backend)
                if kind == "number":
                    minimums = [spec.minimum for spec in specs]
                    maximums = [spec.maximum for spec in specs]
                    steps = [spec.step for spec in specs if spec.step]
                    # no step anywhere (kohya's decimals such as Text Encoder LR): the browser's default 1 marks 5e-5
                    # invalid and its arrows jump by 1
                    decimal = any(not self.backends[name].whole_number(key) for name, _spec, _order in items)
                    definition.bounds = (
                        None if any(value is None for value in minimums) else min(minimums),
                        None if any(value is None for value in maximums) else max(maximums),
                        min(steps) if steps else (1e-6 if decimal else 1),
                    )
                definition.host_classes = self._host_classes(definition)
                self.fields[uid] = definition

    def _host_classes(self, item: FieldDef) -> list[str]:
        classes = [HOST_CLASS]
        for name, spec in item.per_backend.items():
            b = token(name)
            classes.append(f"b-{b}")
            if spec.modes == frozenset({LORA}):
                classes.append(f"lo-{b}")
            elif spec.modes == frozenset({FULL}):
                classes.append(f"fo-{b}")
            if spec.advanced:
                classes.append(f"adv-{b}")
            if not spec.visible or item.key in HIDDEN_KEYS:  # e.g. training_mode: step 2 sets it for every engine
                classes.append(f"hid-{b}")
        for entry in entries_for(self.category):
            if item.key in entry.locked_keys and entry.backend in item.per_backend:
                classes.append(f"lk-{token(entry.key)}")
        for rule_id in RULES_BY_DEPENDENT.get(item.key, ()):
            classes.append(f"r-{rule_id}")
        return classes

    def generated_css(self) -> str:
        root = f"#{self.root_id}"
        rules = [
            f"{root}:not([data-be]) .{HOST_CLASS} {{display:none!important}}",
            f"{root}.ws-busy .ws-section {{opacity:.55;transition:opacity .12s}}",
            f'{root}:not([data-dl="1"]) .download-row {{display:none!important}}',
        ]
        for name in self.backends:
            b = token(name)
            rules.append(f'{root}[data-be="{b}"] .{HOST_CLASS}:not(.b-{b}) {{display:none!important}}')
            rules.append(f'{root}[data-be="{b}"][data-mode="full"] .lo-{b} {{display:none!important}}')
            rules.append(f'{root}[data-be="{b}"][data-mode="lora"] .fo-{b} {{display:none!important}}')
            rules.append(f'{root}[data-be="{b}"][data-adv="0"] .adv-{b} {{display:none!important}}')
            rules.append(f'{root}[data-be="{b}"] .hid-{b} {{display:none!important}}')
            for mode in (LORA, FULL):
                for adv in ("0", "1"):
                    rules.append(f'{root}[data-be="{b}"][data-mode="{mode}"][data-adv="{adv}"] .ws-section:not(.sv-{b}-{mode}-{adv}) {{display:none!important}}')
        for entry in self.entries:
            if entry.locked_keys:
                rules.append(f'{root}[data-entry="{token(entry.key)}"] .lk-{token(entry.key)} {{display:none!important}}')
        for rule_id, *_rest in RULES:
            rules.append(f'{root}[data-r-{rule_id}="0"] .r-{rule_id} {{display:none!important}}')
        return "\n".join(rules)

    def _section_classes(self, section_id: str) -> list[str]:
        classes = ["ws-section", f"sec-{section_id}"]
        items = self.section_fields.get(section_id, [])
        for name in self.backends:
            b = token(name)
            for mode in (LORA, FULL):
                for adv in (False, True):
                    shown = section_id in ("dataset", "model", "bcf") or any(
                        (spec := item.per_backend.get(name)) is not None and spec.visible and mode in spec.modes and (adv or not spec.advanced)
                        for item in items
                    )
                    if shown:
                        classes.append(f"sv-{b}-{mode}-{int(adv)}")
        return classes

    # ------------------------------------------------------------------ helpers
    def entry(self, key: str | None) -> ModelEntry:
        if key in self.entry_keys:
            return get_entry(key)
        default = DEFAULT_ENTRY[self.category]
        return get_entry(default) if default in self.entry_keys else self.entries[0]

    def backend_of(self, entry: ModelEntry) -> Backend:
        return self.backends[entry.backend]

    @property
    def field_components(self) -> list:
        return [item.component for item in self.fields.values()]

    def client_values(self, args) -> dict:
        return dict(zip(self.field_components, args))

    def values_for(self, entry: ModelEntry, client: dict) -> dict:
        """Trainer key -> value for ``entry``'s backend from the browser's field values."""
        backend = self.backend_of(entry)
        values = {}
        for spec in backend.specs:
            item = self.fields.get(self.backend_uid[backend.name].get(spec.key))
            values[spec.key] = client.get(item.component, spec.default) if item else spec.default
        values.update(entry.fixed)
        return values

    def _ui_value(self, item: FieldDef, spec: ParamSpec, value):
        kind = item.kind
        if kind == "checkbox":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if kind in ("number", "slider"):
            if value in (None, ""):
                return None if kind == "number" else (spec.minimum if spec.minimum is not None else 0)
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None if kind == "number" else (spec.minimum or 0)
            return int(number) if number.is_integer() else number
        if kind == "dropdown" and spec.multiselect:
            if value in (None, ""):
                return []
            return list(value) if isinstance(value, (list, tuple)) else [value]
        if kind in ("dropdown", "radio"):
            return " ".join(str(item) for item in value) if isinstance(value, list) else value
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            if kind in ("file", "folder"):  # several paths: the classic syntax, quotes around paths with spaces
                return " ".join(f'"{item}"' if any(char.isspace() for char in str(item)) else str(item) for item in value)
            return " ".join(str(item) for item in value)
        return str(value)

    def hidden_advanced(self, entry: ModelEntry, mode: str) -> int:
        backend = self.backend_of(entry)
        return sum(
            1
            for item in self.fields.values()
            if (spec := item.per_backend.get(backend.name)) is not None
            and spec.advanced
            and spec.visible
            and mode in spec.modes
            and spec.key not in entry.locked_keys
        )

    def adv_label(self, entry: ModelEntry, mode: str) -> str:
        count = self.hidden_advanced(entry, mode)
        return f"⚙️ Advanced settings (+{count})" if count else "⚙️ Advanced settings"

    def attrs(self, entry: ModelEntry, mode: str, show_advanced: bool, values: dict) -> dict:
        backend = self.backend_of(entry)
        attributes = {"be": token(backend.name), "mode": mode, "entry": token(entry.key), "adv": "1" if show_advanced else "0"}
        attributes["dl"] = "1" if entry.download_bundles else "0"  # the Download row (CSS)
        for rule_id, state in rule_states(values, set(backend.keys())).items():
            attributes[f"r-{rule_id}"] = state
        return attributes

    # ------------------------------------------------------------------ session mirror + diff
    def _mirror(self, request) -> dict:
        key = getattr(request, "session_hash", None) or "local"
        with self._mirror_lock:
            mirror = self._mirrors.get(key)
            if mirror is None:
                mirror = {component_id: dict(props) for component_id, props in self.baseline.items()}
                self._mirrors[key] = mirror
                while len(self._mirrors) > MAX_SESSIONS:
                    self._mirrors.popitem(last=False)
            else:
                self._mirrors.move_to_end(key)
            return mirror

    def session(self, request) -> threading.RLock:
        """One browser session's events run one at a time (a quick model click while a preset loads, an
        edit recorded during a switch): each one diffs against, and updates, that session's mirror."""
        key = getattr(request, "session_hash", None) or "local"
        with self._mirror_lock:
            lock = self._session_locks.get(key)
            if lock is None:
                lock = self._session_locks[key] = threading.RLock()
                while len(self._session_locks) > MAX_SESSIONS:
                    self._session_locks.popitem(last=False)
            else:
                self._session_locks.move_to_end(key)
            return lock

    def forget(self, request) -> None:
        key = getattr(request, "session_hash", None) or "local"
        with self._mirror_lock:
            self._mirrors.pop(key, None)

    def mirror_values(self, request, entry: ModelEntry) -> dict:
        """Trainer key -> the value this browser session shows, for ``entry``'s trainer."""
        mirror = self._mirror(request)
        backend = self.backend_of(entry)
        values = {}
        for spec in backend.specs:
            item = self.fields.get(self.backend_uid[backend.name].get(spec.key))
            known = mirror.get(item.component._id, {}) if item else {}
            values[spec.key] = known.get("value", spec.default)
        values.update(entry.fixed)
        return values

    def _track(self, component, **props) -> None:
        self.baseline[component._id] = props

    def link_outputs(self, item: FieldDef) -> list:
        """The fields the links of ``item`` may change, for every trainer that shares the field."""
        outputs = []
        for link in self.link_groups.get(item.uid, []):
            for key in link.outputs:
                target = self.fields.get(self.backend_uid[link.backend].get(key, ""))
                if target is not None and target.component not in outputs:
                    outputs.append(target.component)
        return outputs

    def follow_links(self, request, entry_key: str, item: FieldDef) -> dict:
        """After the user changed ``item`` (already in the mirror): the settings the classic tab changes
        with it (links.py), then the VRAM panel, which may follow either."""
        entry = self.entry(entry_key)
        backend = self.backend_of(entry)
        links = [link for link in self.link_groups.get(item.uid, []) if link.backend == backend.name]
        updates: dict = {}
        if links:
            mirror = self._mirror(request)

            def user_set(key: str) -> bool:
                target = self.fields.get(self.backend_uid[backend.name].get(key, ""))
                return target is not None and bool(mirror.get(target.component._id, {}).get("user"))

            desired = {}
            changed = L.follow(links, self.mirror_values(request, entry), user_set)
            for key, value in changed.items():
                target = self.fields.get(self.backend_uid[backend.name].get(key, ""))
                if target is not None:
                    desired[target.component] = {"value": self._ui_value(target, target.per_backend[backend.name], value)}
            if desired:
                updates.update(self.emit(request, desired))
            if any(target.kind in ("file", "folder") for key in changed if (target := self.fields.get(self.backend_uid[backend.name].get(key, "")))):
                # a model file the link switched (HiDream dev, Mage-Flow Edit): its ✅ / ⚠️ chip follows
                updates.update(self.emit(request, {self.files_html: {"value": self.files_status(entry, self.mirror_values(request, entry))}}))
        updates.update(self.emit(request, {self.vram_html: {"value": self.vram_note(request)}}))
        return updates

    def emit(self, request, desired: dict, client: dict | None = None, *, user: bool = False) -> dict:
        """Updates for only the props/values that differ from what this browser shows.

        ``user`` marks the written values as the user's own (their edits or their own preset), which
        decides whether a dataset/output path survives a later model or preset switch.
        """
        mirror = self._mirror(request)
        client = client or {}
        out = {}
        for component, props in desired.items():
            known = mirror.setdefault(component._id, {})
            changed = {}
            for name, value in props.items():
                current = client[component] if name == "value" and component in client else known.get(name, _MISSING)
                if current is _MISSING or not _same(current, value):
                    changed[name] = value
                if name == "value" and (name in changed or "user" not in known):
                    known["user"] = user
                known[name] = value
            if not changed:
                continue
            # A bare value is cheaper for Gradio than an update dict (no component re-creation).
            out[component] = changed["value"] if list(changed) == ["value"] and not isinstance(changed["value"], dict) else gr.update(**changed)
        if out:
            return out
        # "no change" must still name one of the event's own outputs (Gradio rejects any other component)
        return {next(iter(desired)): gr.skip()} if desired else self.nothing()

    def nothing(self) -> dict:
        """A valid "no change" result for handlers whose outputs include the workspace state."""
        return {self.state_html: gr.skip()}

    # ------------------------------------------------------------------ desired state
    def field_props(self, item: FieldDef, spec: ParamSpec, entry: ModelEntry, value) -> dict:
        ui = self._ui_value(item, spec, value)
        backend_name = next((name for name, candidate in item.per_backend.items() if candidate is spec), None)
        label, info = item.text_overrides.get(backend_name, (item.label, item.info))
        props = {"label": label, "info": info or None, "value": ui}
        if item.kind in ("dropdown", "radio"):
            props["choices"] = _choices_with(entry.choice_overrides.get(spec.key) or spec.choices or (), ui)
        if item.kind in ("number", "slider"):
            minimum, maximum = (spec.minimum, spec.maximum) if item.kind == "slider" else item.bounds[:2]
            if isinstance(ui, (int, float)):
                if minimum is not None and ui < minimum:
                    minimum = ui
                if maximum is not None and ui > maximum:
                    maximum = ui
            if item.kind == "slider":
                minimum = 0 if minimum is None else minimum
                maximum = max(minimum + 1, 100 if maximum is None else maximum)
            props["minimum"] = minimum
            props["maximum"] = maximum
            props["step"] = (spec.step if spec.step is not None else 1) if item.kind == "slider" else item.bounds[2]
        interactive = spec.extras.get("interactive") if spec.extras else None
        props["interactive"] = interactive is not False
        return props

    def desired_fields(self, entry: ModelEntry, values: dict) -> dict:
        backend = self.backend_of(entry)
        desired = {}
        for item in self.fields.values():
            spec = item.per_backend.get(backend.name)
            if spec is not None:
                desired[item.component] = self.field_props(item, spec, entry, values.get(spec.key, spec.default))
        return desired

    def load_values(self, entry: ModelEntry, mode: str, preset_path: str | None) -> Loaded:
        backend = self.backend_of(entry)
        values = backend.defaults(mode)
        values.update(entry.default_overrides)
        values.update(entry.fixed)
        values.update(L.on_load(backend.name, entry.fixed, entry.default_overrides, values, preset_loaded=False))
        if "output_dir" in values and not str(values.get("output_dir") or "").strip() and entry.model_folder.startswith("Training_Models"):
            # never start without an output folder: the shipped presets' place (../Training_Outputs_<Model>)
            values["output_dir"] = os.path.join(ROOT_DIR, "Training_Outputs" + entry.model_folder[len("Training_Models"):])
        defaults = dict(values)
        warnings: list[str] = []
        error = ""
        if preset_path:
            name = os.path.basename(preset_path)
            try:
                if not os.path.isfile(preset_path):
                    raise FileNotFoundError("the file no longer exists")
                if not isinstance(read_preset_file(preset_path), dict):  # a damaged file is refused, never half-applied
                    raise ValueError("it does not contain settings")
                result = backend.load(preset_path, values)
                if result.failed:
                    raise ValueError("; ".join(result.warnings) or "the trainer could not read it")
                values = result.values
                warnings.extend(result.warnings)
                if result.file_keys and result.matched_keys / max(1, result.file_keys) < 0.5:
                    warnings.append(f"Only {result.matched_keys} of {result.file_keys} settings in this file belong to {entry.name}; the rest were ignored.")
            except Exception as exc:
                log.warning(f"Preset {preset_path} could not be loaded: {exc}")
                line = getattr(exc, "lineno", None)  # toml.TomlDecodeError: the file is not valid TOML
                reason = f"it is not a valid TOML file (problem on line {line})" if line else str(exc)
                error = f"{name} could not be loaded: {reason}"
                values, warnings = dict(defaults), []
        values.update(entry.fixed)
        # the Mage-Flow Edit checkpoint for its fixed is_edit, Wan 2.2's compile residency for its default task
        values.update(L.on_load(backend.name, entry.fixed, entry.default_overrides, values, preset_loaded=bool(preset_path) and not error))
        if "training_mode" in values:
            values["training_mode"] = backend.mode_value(mode)
        return Loaded(values, warnings, defaults, error)

    def presets_state(self, entry: ModelEntry, mode: str, tier: str, selected: str | None) -> dict:
        options = P.preset_options(entry, mode, tier)
        choices = [[option.label, option.path] for option in options]
        if selected and selected not in [path for _label, path in choices] and os.path.isfile(selected):
            choices = [[f"📄 {os.path.basename(selected)}", selected]] + choices
        tiers = [list(choice) for choice in P.tier_choices(entry, mode)]
        if tier not in [value for _label, value in tiers]:
            tier = P.AUTO_TIER
        description = P.describe(entry, selected) if selected else ""
        if not options:
            info = "No ready-made presets for this model and training type yet — the defaults are loaded. Save your own below."
        elif tier == P.AUTO_TIER:
            info = f"{len(options)} preset(s) for {P.gpu_text()} · picking one loads it instantly"
        else:
            info = f"{len(options)} preset(s) · picking one loads it instantly"
        return {
            "tier": {"choices": tiers, "value": tier},
            "preset": {"choices": choices, "value": selected if selected in [p for _l, p in choices] else None, "info": info},
            "desc": {"value": f'<div class="ud-preset-desc">{html.escape(description)}</div>' if description else ""},
        }

    def desired_presets(self, entry: ModelEntry, mode: str, tier: str, selected: str | None) -> dict:
        state = self.presets_state(entry, mode, tier, selected)
        return {self.tier_dd: state["tier"], self.preset_dd: state["preset"], self.preset_desc: state["desc"]}

    def model_card(self, entry: ModelEntry, mode: str) -> str:
        from .model_tips import tips_for

        hints = tips_for(entry, mode)
        badges = "".join(f'<span class="ud-badge">{html.escape(badge)}</span>' for badge in entry.badges)
        modes = "".join(f'<span class="ud-mode-chip{" on" if item == mode else ""}">{html.escape(MODE_LABELS[item])}</span>' for item in entry.modes)
        if len(entry.modes) == 1:
            modes += '<span class="ud-mode-desc">LoRA training only for this model</span>'
        tips = "".join(f"<li>{html.escape(tip)}</li>" for tip in hints.get("tips", []))
        facts = []
        if hints.get("dataset"):
            facts.append(f'<div class="ud-fact"><span>📁 Dataset</span>{html.escape(hints["dataset"])}</div>')
        if hints.get("vram"):
            facts.append(f'<div class="ud-fact"><span>⚡ VRAM</span>{html.escape(hints["vram"])}</div>')
        about = hints.get("about") or entry.tagline
        tips_html = (
            f'<details class="ud-tips" open><summary>💡 Tips for {html.escape(entry.name)} · {html.escape(MODE_LABELS[mode])}</summary><ul>{tips}</ul>{"".join(facts)}</details>'
            if tips or facts
            else ""
        )
        return (
            '<div class="ud-model-card">'
            f'<div class="ud-model-icon">{entry.icon}</div>'
            '<div class="ud-model-body">'
            f'<div class="ud-model-title">{html.escape(entry.name)} {badges}</div>'
            f'<div class="ud-model-about">{html.escape(about)}</div>'
            f'<div class="ud-model-modes">{modes}</div>'
            f"{tips_html}</div></div>"
        )

    @staticmethod
    def banner(lines: list[str], warnings: list[str] | None = None) -> str:
        warnings = warnings or []
        parts = []
        if lines:
            parts.append('<div class="ud-banner ud-banner-ok">' + "<br>".join(html.escape(line) for line in lines) + "</div>")
        if warnings:
            parts.append('<div class="ud-banner ud-banner-warn">⚠️ ' + "<br>⚠️ ".join(html.escape(line) for line in warnings) + "</div>")
        return "".join(parts)

    def files_status(self, entry: ModelEntry, values: dict) -> str:
        backend = self.backend_of(entry)
        backend_keys = set(backend.keys())
        rows = []
        for spec in backend.specs:
            if spec.section != "model" or spec.kind not in ("file", "folder") or not spec.visible or spec.key in entry.locked_keys:
                continue
            if not rules_allow(spec.key, values, backend_keys):
                continue
            path = clean_path(values.get(spec.key))
            if path and _model_file_exists(path):
                rows.append(f'<span class="ud-file ok">✅ {html.escape(spec.label)}</span>')
            elif path:
                rows.append(f'<span class="ud-file bad" title="{html.escape(path)}">⚠️ {html.escape(spec.label)} — not found</span>')
            else:
                rows.append(f'<span class="ud-file empty">⬜ {html.escape(spec.label)} — not set</span>')
        return '<div class="ud-files">' + "".join(rows) + "</div>" if rows else ""

    # ------------------------------------------------------------------ full state application
    def apply(
        self,
        request,
        client: dict,
        entry_key: str,
        mode: str | None,
        tier: str | None,
        preset_path: str | None,
        show_advanced: bool,
        *,
        messages=None,
        warnings=None,
        keep_user_paths: bool = True,
        refuse_unreadable: bool = False,
        finder_from_preset: bool = False,
    ) -> dict:
        """Show ``entry`` / ``mode`` with ``preset_path`` loaded. A preset that cannot be read is refused
        (``refuse_unreadable``: the user picked it) or replaced by the defaults (remembered / automatic)."""
        entry = self.entry(entry_key)
        if mode not in entry.modes:
            mode = entry.modes[0]
        tier = P.offered_tier(entry, mode, tier)
        loaded = self.load_values(entry, mode, preset_path)
        warnings = list(warnings or [])
        if loaded.error:
            if refuse_unreadable:
                return self.refuse(request, client, f"Not loaded: {loaded.error}")
            warnings.append(f"{loaded.error} — the defaults are loaded instead.")
            preset_path = None
        values, load_warnings, defaults = loaded.values, loaded.warnings, loaded.defaults
        lines = list(messages or [])
        kept = []
        user_preset = bool(preset_path) and not P.is_read_only_preset(preset_path)
        backend = self.backend_of(entry)
        if not user_preset:
            mirror = self._mirror(request)
            for key in STICKY_KEYS:
                item = self.fields.get(self.backend_uid[backend.name].get(key, ""))
                if item is None:
                    continue
                known = mirror.get(item.component._id, {})
                current = known.get("value")
                if keep_user_paths and known.get("user") and current not in (None, ""):
                    if not _same(current, values.get(key)):
                        values[key] = current
                        if key in STICKY_LABELS:
                            kept.append(STICKY_LABELS[key])
                elif preset_path and _foreign_path(values.get(key)):
                    # a shipped preset carries its author's machine paths: use this PC's default
                    values[key] = defaults.get(key, "")
        name = os.path.splitext(os.path.basename(preset_path))[0] if preset_path else ""
        if preset_path and not load_warnings:
            lines.append(f"✅ Loaded {'your preset' if user_preset else 'preset'} {name} · {entry.name} · {MODE_LABELS[mode]}")
        elif not preset_path:
            lines.append(f"✅ {entry.name} · {MODE_LABELS[mode]} defaults loaded")
        if kept:
            lines.append("📌 Kept your " + " and ".join(kept))
        desired = self.desired_fields(entry, values)
        desired.update(self.desired_presets(entry, mode, tier, preset_path))
        if finder_from_preset and user_preset and self.finder_inputs:
            finder = CF.from_metadata(P.read_metadata(preset_path), self.category)
            if finder is not None:
                CF.remember(self.category, finder)
                desired.update(self.desired_finder(finder))
        state = self.attrs(entry, mode, show_advanced, values)
        # always re-sent: a dependent-field rule the browser switched on its own (a dataset-mode change)
        # must follow the loaded values even when the server's last state reads the same
        state["seq"] = next(self._seq)
        desired[self.state_html] = {"value": json.dumps(state, sort_keys=True)}
        desired[self.adv_toggle] = {"label": self.adv_label(entry, mode)}
        desired[self.model_radio] = {"value": entry.key}
        desired[self.mode_radio] = {"choices": [[MODE_LABELS[item], item] for item in entry.modes], "value": mode}
        desired[self.mode_note] = {"value": MODE_DESCRIPTIONS[mode]}
        desired[self.model_card_html] = {"value": self.model_card(entry, mode)}
        banner = self.banner(lines, warnings + load_warnings)
        desired[self.banner_html] = {"value": banner}
        desired[self.files_html] = {"value": self.files_status(entry, values)}
        bundles = [list(bundle) for bundle in entry.download_bundles]
        if bundles:
            desired[self.bundle_dd] = {"choices": bundles, "value": bundles[0][1]}
        # the VRAM panel compares later edits with what was just loaded
        self._store(self._vram_base, request, (entry.key, mode, dict(values), preset_path))
        desired[self.vram_html] = {"value": ""}
        P.remember_selection(self.category, entry.key, mode, tier, preset_path)
        updates = self.emit(request, desired, client, user=user_preset)
        if user_preset:
            # a value of your own preset that is only this model's default (its output folder) is no choice of
            # yours: it must not follow you to the next model
            mirror = self._mirror(request)
            for key in STICKY_KEYS:
                item = self.fields.get(self.backend_uid[backend.name].get(key, ""))
                if item is not None and _same(values.get(key), defaults.get(key)):
                    mirror.setdefault(item.component._id, {})["user"] = False
        return updates

    def refuse(self, request, client: dict, message: str) -> dict:
        """Leave the workspace as it is, the preset selector included, and say why."""
        gr.Warning(message, duration=10)
        desired = {self.banner_html: {"value": self.banner([], [message])}}
        shown = self._mirror(request).get(self.preset_dd._id, {}).get("value")
        if self.preset_dd in client and client[self.preset_dd] != shown:
            desired[self.preset_dd] = {"value": shown}  # back to the preset that is really loaded
        return self.emit(request, desired, client)

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        install_lenient_validation()
        entry, mode, tier, preset = self.remembered_selection()
        loaded = self.load_values(entry, mode, preset)
        if loaded.error:  # the page shows the defaults; loading the page then explains (restore)
            log.warning(loaded.error)
            preset = None
        self.initial = (entry.key, mode, tier, preset, _signature(preset))
        values = loaded.values
        backend = self.backend_of(entry)

        with gr.Column(elem_classes=["ws", f"ws-{self.category}"], elem_id=self.root_id):
            gr.HTML(f"<style>{self.generated_css()}</style>", elem_classes=["ws-css"])
            # Gradio 6 re-sends a loading status to every component that was ever a progress target,
            # on every event. All events here report progress to this one hidden element instead of
            # their (hundreds of) outputs, which keeps every click and the status timer cheap.
            self.progress_sink = gr.HTML("", elem_classes=["ws-sink"])
            state_value = json.dumps(self.attrs(entry, mode, False, values), sort_keys=True)
            self.state_html = gr.HTML(state_value, js_on_load=STATE_JS.replace("__ROOT__", self.root_id), elem_classes=["ws-state"])
            self._track(self.state_html, value=state_value)
            # ---------------------------------------------------------- step 1: base model
            with gr.Group(elem_classes=["ws-card", "ws-step"]):
                gr.HTML(f'<div class="ws-step-title"><span class="ws-step-num">1</span>Choose the base model<span class="ws-step-hint">{len(self.entries)} model{"s" if len(self.entries) != 1 else ""}</span></div>')
                choices = []
                for item in self.entries:
                    tags = [badge for badge in item.badges if not badge.startswith("🔊")]
                    choices.append((item.card_label + (" · " + " · ".join(tags) if tags else ""), item.key))
                self.model_radio = gr.Radio(choices=choices, value=entry.key, label="Base model", show_label=False, container=False, elem_classes=["model-picker"])
                self._track(self.model_radio, value=entry.key)
            with gr.Row(equal_height=True, elem_classes=["ws-row"]):
                # ------------------------------------------------------ step 2: training type
                with gr.Column(scale=1, min_width=260, elem_classes=["ws-col"]), gr.Group(elem_classes=["ws-card", "ws-step", "ws-step-mode"]):
                    gr.HTML('<div class="ws-step-title"><span class="ws-step-num">2</span>Training type</div>')
                    mode_choices = [[MODE_LABELS[item], item] for item in entry.modes]
                    self.mode_radio = gr.Radio(choices=mode_choices, value=mode, label="Training type", show_label=False, container=False, elem_classes=["mode-picker"])
                    self._track(self.mode_radio, choices=mode_choices, value=mode)
                    self.mode_note = gr.Markdown(MODE_DESCRIPTIONS[mode], elem_classes=["mode-note"])
                    self._track(self.mode_note, value=MODE_DESCRIPTIONS[mode])
                # ------------------------------------------------------ step 3: preset
                with gr.Column(scale=3, min_width=320, elem_classes=["ws-col"]), gr.Group(elem_classes=["ws-card", "ws-step", "ws-step-preset"]):
                    gr.HTML(f'<div class="ws-step-title"><span class="ws-step-num">3</span>GPU preset<span class="ws-step-hint">{html.escape(P.gpu_text())}</span></div>')
                    preset_state = self.presets_state(entry, mode, tier, preset)
                    with gr.Row(equal_height=True, elem_classes=["preset-row"]):
                        self.tier_dd = gr.Dropdown(label="GPU VRAM", scale=2, min_width=170, elem_classes=["tier-dd"], **preset_state["tier"])
                        self._track(self.tier_dd, **preset_state["tier"])
                        self.preset_dd = gr.Dropdown(label="Preset", scale=6, min_width=260, elem_classes=["preset-dd"], **preset_state["preset"])
                        self._track(self.preset_dd, **preset_state["preset"])
                        self.refresh_btn = gr.Button("🔄", scale=0, min_width=46, elem_classes=["ws-icon-btn"])
                    self.preset_desc = gr.HTML(elem_classes=["preset-desc"], **preset_state["desc"])
                    self._track(self.preset_desc, **preset_state["desc"])
                    with gr.Row(equal_height=True, elem_classes=["preset-io"]):
                        self.name_tb = gr.Textbox(
                            label="Preset name or path",
                            show_label=False,
                            lines=1,
                            max_lines=1,
                            placeholder="Preset name (saved in user_presets) or a full .toml / .json path",
                            scale=6,
                            min_width=220,
                            container=False,
                            elem_classes=["preset-name"],
                        )
                        self.save_btn = gr.Button("💾 Save", scale=0, min_width=124, elem_classes=["ws-btn", "ws-btn-save"])
                        self.load_btn = gr.Button("📂 Load", scale=0, min_width=124, elem_classes=["ws-btn", "ws-btn-load"])
            self.banner_html = gr.HTML("", elem_classes=["ws-banner"])  # CSS hides it while empty
            self._track(self.banner_html, value="")
            card = self.model_card(entry, mode)
            self.model_card_html = gr.HTML(card, elem_classes=["ws-model-card"])
            self._track(self.model_card_html, value=card)

            # ---------------------------------------------------------- settings toolbar
            with gr.Row(equal_height=True, elem_classes=["ws-toolbar"]):
                self.search_tb = gr.Textbox(
                    label="Search settings",
                    show_label=False,
                    lines=1,
                    max_lines=1,
                    placeholder="🔎 Search settings (e.g. learning rate, block swap, resolution)",
                    container=False,
                    scale=6,
                    min_width=220,
                    elem_classes=["ws-search"],
                )
                adv_label = self.adv_label(entry, mode)
                self.adv_toggle = gr.Checkbox(label=adv_label, value=False, container=False, scale=0, min_width=250, elem_classes=["ws-adv-toggle"])
                self._track(self.adv_toggle, label=adv_label)
                self.expand_btn = gr.Button("⬇ Expand all", scale=0, min_width=136, elem_classes=["ws-btn", "ws-btn-ghost"])
                self.collapse_btn = gr.Button("⬆ Collapse all", scale=0, min_width=146, elem_classes=["ws-btn", "ws-btn-ghost"])

            # ---------------------------------------------------------- sections
            for section_id, title, description in SECTIONS:
                if section_id == "bcf" and not self.finder:
                    continue
                with gr.Accordion(title, open=section_id in OPEN_SECTIONS, elem_classes=self._section_classes(section_id)) as accordion:
                    self.sections[section_id] = accordion
                    gr.HTML(f'<div class="ws-section-desc">{html.escape(description)}</div>', elem_classes=["ws-section-desc-wrap"])
                    if section_id == "model":
                        self._build_model_tools(entry, values)
                    self._build_section_fields(section_id, backend, entry, values)
                    if section_id == "dataset":
                        self._build_dataset_tools()
                    if section_id == "bcf":
                        self._build_finder()

            # ---------------------------------------------------------- VRAM advice (sticks to the bottom while scrolling)
            self.vram_html = gr.HTML("", elem_classes=["ws-vram"])
            self._track(self.vram_html, value="")

            # ---------------------------------------------------------- actions + monitor
            with gr.Group(elem_classes=["ws-card", "ws-actions"]):
                with gr.Row(equal_height=True, elem_classes=["action-row"]):
                    self.start_btn = gr.Button("▶ Start Training", variant="primary", scale=4, min_width=200, elem_classes=["ws-start"])
                    # Stop and its confirmation are always mounted; CSS shows them from the root's
                    # data-running / data-stopask attributes, so they appear instantly and cost no update.
                    self.stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1, min_width=120, elem_classes=["ws-stop"])
                    self.print_btn = gr.Button("📋 Print Command", scale=1, min_width=160, elem_classes=["ws-btn", "ws-btn-ghost"])
                with gr.Row(equal_height=True, elem_classes=["stop-confirm"]) as self.stop_confirm_row:
                    gr.HTML('<div class="ud-stop-q">⚠️ Stop the current training? Progress since the last saved checkpoint is lost.</div>')
                    self.stop_yes = gr.Button("Yes, stop training", variant="stop", scale=0, min_width=170, elem_classes=["ws-stop-yes"])
                    self.stop_no = gr.Button("Keep training", scale=0, min_width=140, elem_classes=["ws-stop-no"])
                self.command_code = gr.Code(label="Training command preview", language="shell", visible=False, interactive=False, elem_classes=["ws-command"])
            from .monitor import build_monitor

            self.monitor = build_monitor(self.root_id)
            # clicked by the monitor when a model download finished (CSS-hidden, outside the collapsible sections)
            self.files_refresh_btn = gr.Button("files", elem_classes=["ws-files-refresh"])
        self._wire()

    def _build_model_tools(self, entry: ModelEntry, values: dict) -> None:
        status = self.files_status(entry, values)
        self.files_html = gr.HTML(status, elem_classes=["ws-files"])
        self._track(self.files_html, value=status)
        bundles = [list(bundle) for bundle in entry.download_bundles]
        with gr.Row(equal_height=True, elem_classes=["download-row"]) as self.download_row:  # shown by data-dl (CSS)
            self.bundle_dd = gr.Dropdown(label="Download model files", choices=bundles or [["", ""]], value=bundles[0][1] if bundles else None, scale=4, min_width=220)
            self.download_btn = gr.Button("⬇️ Download", scale=0, min_width=165, elem_classes=["ws-btn", "ws-btn-load"])
            self.download_stop_btn = gr.Button("✖ Stop", scale=0, min_width=110, elem_classes=["ws-btn", "ws-btn-ghost"])
        self._track(self.bundle_dd, choices=bundles or [["", ""]], value=bundles[0][1] if bundles else None)
        self.download_log = gr.HTML('<pre class="dl-log"></pre>', elem_classes=["ws-download-log"])

    def _build_dataset_tools(self) -> None:
        with gr.Row(equal_height=True, elem_classes=["dataset-tools"]):
            self.check_dataset_btn = gr.Button("🔍 Check dataset", scale=0, min_width=215, elem_classes=["ws-btn", "ws-btn-ghost"])
            self.samples_btn = gr.Button("🎲 New samples", scale=0, min_width=205, elem_classes=["ws-btn", "ws-btn-ghost"])
            self.clean_btn = gr.Button("🧹 Clean old caches", scale=0, min_width=255, elem_classes=["ws-btn", "ws-btn-ghost"])
            gr.HTML(
                '<div class="ds-hint">Statistics, repeats and steps, problems with example files and sample previews (edit models: '
                "each target image beside its control images) of the dataset the selected model will train on.</div>"
            )
        # 🧹 asks first; the question is always mounted and shown from data-cleanask (CSS), like Stop's
        with gr.Row(equal_height=True, elem_classes=["clean-confirm"]):
            self.clean_question = gr.HTML("", elem_classes=["clean-question"])
            with gr.Column(scale=0, min_width=300):
                self.clean_current = gr.Checkbox(
                    label="Also delete the current cache folder",
                    info="Everything is cached again on the next Start",
                    value=False,
                )
                with gr.Row():
                    self.clean_yes = gr.Button("🗑️ Delete", scale=0, min_width=130, elem_classes=["ws-clean-yes"])
                    self.clean_no = gr.Button("Cancel", scale=0, min_width=110, elem_classes=["ws-clean-no"])
        self.clean_result = gr.HTML("", elem_classes=["clean-result"])
        # "<1|0>:<seq>": shows / hides the question (a new seq re-applies it after Cancel hid it in the browser)
        self.clean_state = gr.HTML("0:0", js_on_load=CLEAN_STATE_JS.replace("__ROOT__", self.root_id), elem_classes=["ws-state"])
        # shown once, by the first check: hidden components cost nothing until then
        self.dataset_report = gr.HTML("", visible=False, elem_classes=["dataset-report"])
        self.dataset_gallery = gr.Gallery(
            label="Dataset samples · click one to enlarge",
            visible=False,
            columns=4,
            height="auto",
            object_fit="contain",
            allow_preview=True,
            interactive=False,
            elem_classes=["dataset-gallery"],
        )

    def _build_finder(self) -> None:
        settings = CF.remembered(self.category)
        gr.HTML(FINDER_INTRO, elem_classes=["bcf-intro-wrap"])
        self.bcf_enable = gr.Checkbox(
            label="🏆 Find the best checkpoint when training ends",
            info="Off by default. Samples your test prompts with the base model and at every checkpoint save, then scores them.",
            value=settings["enabled"],
            elem_classes=["bcf-enable"],
        )
        with gr.Row(equal_height=False, elem_classes=["bcf-row", "bcf-subject-row"]):
            self.bcf_subject = gr.Textbox(
                label="Subject (trigger word and class)",
                info="As your captions name it. A new subject is written into the prompts.",
                value=settings["subject"],
                placeholder="ohwx man",
                lines=1,
                max_lines=1,
                scale=5,
                min_width=260,
            )
            self.bcf_from_dataset = gr.Button("✨ From dataset folder", scale=0, min_width=280, elem_classes=["ws-btn", "ws-btn-ghost", "bcf-from-ds"])
        with gr.Row(equal_height=True, elem_classes=["bcf-row"]):
            self.bcf_seeds = gr.Radio(
                choices=[1, 2, 3, 4],
                value=settings["seeds"],
                label="Seeds per prompt",
                info="More seeds: steadier scores, longer sampling",
                scale=2,
                min_width=260,
            )
            self.bcf_leakage = gr.Checkbox(
                label="Leakage check",
                info="Adds a prompt without the trigger word: your subject should not appear in it",
                value=settings["leakage"],
                scale=3,
                min_width=280,
            )
        self.bcf_prompts = gr.Textbox(
            label="Test prompts · one per line (blank lines are ignored)",
            info="Describe your subject in different scenes. Every checkpoint renders the same prompts and seeds, so only the weights differ.",
            value=CF.prompts_text(settings),
            lines=8,
            max_lines=24,
            elem_classes=["bcf-prompts"],
        )
        with gr.Row(equal_height=True, elem_classes=["bcf-row"]):
            self.bcf_defaults_btn = gr.Button("↺ Default prompts", scale=0, min_width=200, elem_classes=["ws-btn", "ws-btn-ghost"])
            summary = finder_summary_html(settings)
            self.bcf_summary = gr.HTML(summary, elem_classes=["bcf-summary"])
        with gr.Row(equal_height=True, elem_classes=["bcf-actions"]):
            self.bcf_show_btn = gr.Button("📊 Show results", scale=0, min_width=190, elem_classes=["ws-btn", "ws-btn-load", "bcf-show-btn"])
            self.bcf_analyze_btn = gr.Button("🔍 Analyze folder", scale=0, min_width=210, elem_classes=["ws-btn", "ws-btn-ghost"])
            self.bcf_open_btn = gr.Button("🌐 Open full report", scale=0, min_width=220, elem_classes=["ws-btn", "ws-btn-ghost"], visible=not self.headless)
            gr.HTML(
                '<div class="ds-hint">For the Output Directory and Output Name set above. 🔍 Analyze also scores a run made without the '
                "finder, from the samples it saved.</div>"
            )
        # shown by the first 📊: hidden components cost nothing until then
        self.bcf_result = gr.HTML("", visible=False, elem_classes=["bcf-result-wrap"])
        self.bcf_gallery = gr.Gallery(
            label="Base model · 🏆 best · last checkpoint for each test prompt · click one to enlarge",
            visible=False,
            columns=3,
            height="auto",
            object_fit="contain",
            allow_preview=True,
            interactive=False,
            elem_classes=["bcf-gallery"],
        )
        self.finder_inputs = [self.bcf_enable, self.bcf_subject, self.bcf_prompts, self.bcf_seeds, self.bcf_leakage]
        for component, value in zip(self.finder_inputs, finder_values(settings)):
            self._track(component, value=value)
        self._track(self.bcf_summary, value=summary)

    def finder_settings(self, args) -> dict | None:
        """The finder's settings from its inputs' values (None: this category has no finder)."""
        return CF.from_ui(self.category, *args) if self.finder_inputs else None

    def desired_finder(self, settings: dict) -> dict:
        if not self.finder_inputs:
            return {}
        desired = {component: {"value": value} for component, value in zip(self.finder_inputs, finder_values(settings))}
        desired[self.bcf_summary] = {"value": finder_summary_html(settings)}
        return desired

    def finder_unavailable(self, entry: ModelEntry, values: dict) -> str:
        """Why the finder cannot judge this run ('' when it can)."""
        _control_name, control_required = control_settings(self.backend_of(entry).name, values)
        if control_required:
            return "🏆 Best Checkpoint Finder skipped: edit-model samples need control images, which its test prompts do not have."
        return ""

    def finder_references(self, entry: ModelEntry, values: dict) -> list[str]:
        """The training images and videos the run reads (what the samples are compared with)."""
        report = self.inspect(entry, values, probe_seconds=0)
        regularization = {folder.name for folder in report.folders if folder.regularization}  # kohya: not the subject
        return [item.path for item in report.used if item.kind in ("image", "video") and item.folder not in regularization]

    def finder_run(self, entry: ModelEntry, values: dict, settings: dict | None) -> tuple[dict, object, str]:
        """(run values, after-step or None, note) for Start and Print Command."""
        if not settings or not settings["enabled"]:
            return values, None, ""
        reason = self.finder_unavailable(entry, values)
        if reason:
            return values, None, reason
        size = None
        if self.backend_of(entry).name.startswith("kohya:"):
            # sd-scripts samples a line without --w / --h at 512 x 512 (the musubi tabs add their own sample size):
            # FLUX.1 renders poorer faces there and the wide shots' faces get too small to compare
            try:
                size = (int(float(values["dataset_resolution_width"])), int(float(values["dataset_resolution_height"])))
            except (KeyError, TypeError, ValueError):
                size = None
        try:
            run, after = CF.prepare_run(values, settings, self.finder_references(entry, values), sample_size=size)
        except CF.FinderError as exc:
            raise gr.Error(str(exc), title="Training could not start", duration=None)
        return run, after, ""

    def _make_component(self, item: FieldDef, spec: ParamSpec, entry: ModelEntry, value):
        props = self.field_props(item, spec, entry, value)
        classes = [FIELD_CLASS, f"f-{item.key}", *item.host_classes]
        common = dict(label=props["label"], info=props["info"], value=props["value"], interactive=props["interactive"], elem_classes=classes)
        kind = item.kind
        if kind == "checkbox":
            component = gr.Checkbox(min_width=CHECK_MIN_WIDTH, **common)
        elif kind == "number":
            if all(other.default is None for other in item.per_backend.values()):
                common["elem_classes"] = [*classes, "ws-optional-number"]  # blank is a real choice here (STATE_JS keeps it)
            component = gr.Number(minimum=props["minimum"], maximum=props["maximum"], step=props["step"], min_width=NORMAL_MIN_WIDTH, **common)
        elif kind == "slider":
            component = gr.Slider(minimum=props["minimum"], maximum=props["maximum"], step=props["step"], min_width=NORMAL_MIN_WIDTH, **common)
        elif kind == "radio":
            component = gr.Radio(choices=props["choices"], min_width=NORMAL_MIN_WIDTH, **common)
        elif kind == "dropdown":
            component = gr.Dropdown(
                choices=props["choices"],
                multiselect=spec.multiselect or None,
                allow_custom_value=item.allow_custom,
                min_width=NORMAL_MIN_WIDTH,
                **common,
            )
        elif kind == "textarea":
            component = gr.Textbox(lines=2, max_lines=6, min_width=NORMAL_MIN_WIDTH, placeholder=spec.placeholder or None, **common)
        else:
            component = gr.Textbox(lines=1, min_width=NORMAL_MIN_WIDTH, placeholder=spec.placeholder or None, **common)
        self._track(component, **props)
        return component

    def _build_section_fields(self, section_id: str, backend: Backend, entry: ModelEntry, values: dict) -> None:
        items = self.section_fields.get(section_id, [])
        regular = [item for item in items if item.kind != "checkbox"]
        checks = [item for item in items if item.kind == "checkbox"]

        def spec_for(item: FieldDef) -> ParamSpec:
            return item.per_backend.get(backend.name) or next(iter(item.per_backend.values()))

        def initial_value(item: FieldDef, spec: ParamSpec):
            return values.get(spec.key, spec.default) if backend.name in item.per_backend else spec.default

        row_items: list[FieldDef] = []

        def flush_row():
            if not row_items:
                return
            with gr.Row(equal_height=False, elem_classes=["field-grid"]):
                for row_item in row_items:
                    spec = spec_for(row_item)
                    row_item.component = self._make_component(row_item, spec, entry, initial_value(row_item, spec))
            row_items.clear()

        for item in regular:
            if item.kind in ("file", "folder"):
                flush_row()
                spec = spec_for(item)
                props = self.field_props(item, spec, entry, initial_value(item, spec))
                # one path per field: Enter never adds a line break to it. A trainer that also takes text
                # here (kohya's Sample Prompts: a prompt file or the prompts themselves) gets a growing box.
                text_too = any(other.lines > 1 for other in item.per_backend.values())
                with gr.Row(equal_height=True, elem_classes=["path-row", *item.host_classes]) as row:
                    item.component = gr.Textbox(
                        value=props["value"],
                        label=props["label"],
                        info=props["info"],
                        interactive=props["interactive"],
                        lines=1,
                        max_lines=8 if text_too else 1,
                        scale=12,
                        min_width=240,
                        elem_classes=[FIELD_CLASS, f"f-{item.key}", "path-field"],
                    )
                    if not self.headless:  # a server without a display has no file dialog to open
                        item.browse = gr.Button("📂" if item.kind == "folder" else "📄", scale=0, min_width=48, elem_classes=["ws-icon-btn", "browse-btn"])
                item.container = row
                self._track(item.component, **props)
            elif item.kind == "textarea":
                flush_row()
                spec = spec_for(item)
                with gr.Row(equal_height=False, elem_classes=["field-grid", "field-wide"]):
                    item.component = self._make_component(item, spec, entry, initial_value(item, spec))
            else:
                row_items.append(item)
        flush_row()
        if checks:
            with gr.Row(equal_height=False, elem_classes=["check-grid"]):
                for item in checks:
                    spec = spec_for(item)
                    item.component = self._make_component(item, spec, entry, initial_value(item, spec))

    # ------------------------------------------------------------------ events
    def _wire(self) -> None:
        fields = self.field_components
        tracked_outputs = list(
            dict.fromkeys(
                [
                    self.state_html,
                    self.model_radio,
                    self.mode_radio,
                    self.mode_note,
                    self.tier_dd,
                    self.preset_dd,
                    self.preset_desc,
                    self.model_card_html,
                    self.banner_html,
                    self.files_html,
                    self.bundle_dd,
                    self.adv_toggle,
                    self.vram_html,
                    *fields,
                    *self.finder_inputs,
                    *([self.bcf_summary] if self.finder_inputs else []),
                ]
            )
        )
        self.apply_outputs = tracked_outputs
        finder_inputs = self.finder_inputs
        selection_inputs = [self.model_radio, self.mode_radio, self.tier_dd, self.preset_dd, self.adv_toggle]
        quiet = dict(show_progress="hidden", show_progress_on=[self.progress_sink])
        busy_js = BUSY_JS.replace("__ROOT__", self.root_id)
        unbusy_js = UNBUSY_JS.replace("__ROOT__", self.root_id)
        poke_js = POKE_JS.replace("__ROOT__", self.root_id)

        def selection(args):
            client = dict(zip(selection_inputs, args[:5]))
            return list(args[:5]), client

        # ------------------------------------------------------------ edit tracking
        # The server keeps a mirror of every field's value per browser session, so model / preset
        # switches need no field inputs (each listed input costs browser-side bookkeeping). Every
        # user edit updates the mirror: blur for typed fields, input for choices, change for sliders.
        def recorder(component):
            component_id = component._id

            def record(value, request: gr.Request = None):
                with self.session(request):
                    known = self._mirror(request).setdefault(component_id, {})
                    if not _same(known.get("value", _MISSING), value):
                        known["user"] = True
                    known["value"] = value

            return record

        def path_recorder(component):
            record = recorder(component)

            def record_path(value, request: gr.Request = None):
                # Show exactly what the trainer will get: no quotes or padding, native separators.
                cleaned = clean_path(value)
                record(cleaned, request)
                return cleaned if cleaned != (value or "") else gr.skip()

            return record_path

        def vram_recorder(component):
            record = recorder(component)

            def record_and_advise(value, request: gr.Request = None):
                record(value, request)
                with self.session(request):
                    return self.emit(request, {self.vram_html: {"value": self.vram_note(request)}})

            return record_and_advise

        def link_recorder(item: FieldDef):
            record = recorder(item.component)

            def record_and_follow(entry_key, value, request: gr.Request = None):
                record(value, request)
                with self.session(request):
                    return self.follow_links(request, entry_key, item)

            return record_and_follow

        for item in self.fields.values():
            if item.kind in ("file", "folder"):
                item.component.blur(path_recorder(item.component), inputs=[item.component], outputs=[item.component], queue=False, **quiet)
                continue
            extra = {}
            if item.kind in ("checkbox", "dropdown", "radio"):
                trigger = item.component.input
            elif item.kind == "slider":
                # input, not change: change also fires for every value a model switch sends, which costs a
                # request per slider right while the switch renders; always_last keeps the final drag position
                trigger, extra = item.component.input, {"trigger_mode": "always_last"}
            else:
                trigger = item.component.blur
            if item.uid in self.link_groups:  # the classic tab changes other settings with this one (links.py)
                outputs = [*self.link_outputs(item), self.files_html, self.vram_html]
                trigger(link_recorder(item), inputs=[self.model_radio, item.component], outputs=outputs, queue=False, **quiet, **extra)
            elif item.key in VRAM_KEYS:  # settings that move VRAM: the panel follows them
                trigger(vram_recorder(item.component), inputs=[item.component], outputs=[self.vram_html], queue=False, **quiet, **extra)
            else:
                trigger(recorder(item.component), inputs=[item.component], queue=False, **quiet, **extra)

        # ------------------------------------------------------------ model / training type / preset
        def on_model(request: gr.Request, *args):
            (entry_key, mode, tier, _preset, show_adv), client = selection(args)
            with self.session(request):
                entry = self.entry(entry_key)
                mode = P.remembered_mode(self.category, entry.key) or mode
                if mode not in entry.modes:
                    mode = entry.modes[0]
                tier = P.offered_tier(entry, mode, tier)
                preset = P.remembered_preset(self.category, entry.key, mode) or P.recommended_preset(entry, mode, tier)
                return self.apply(request, client, entry.key, mode, tier, preset, show_adv)

        def on_mode(request: gr.Request, *args):
            (entry_key, mode, tier, _preset, show_adv), client = selection(args)
            with self.session(request):
                entry = self.entry(entry_key)
                if mode not in entry.modes:
                    mode = entry.modes[0]
                tier = P.offered_tier(entry, mode, tier)
                preset = P.remembered_preset(self.category, entry.key, mode) or P.recommended_preset(entry, mode, tier)
                return self.apply(request, client, entry.key, mode, tier, preset, show_adv)

        def on_tier(request: gr.Request, *args):
            (entry_key, mode, tier, _preset, show_adv), client = selection(args)
            with self.session(request):
                entry = self.entry(entry_key)
                tier = P.offered_tier(entry, mode, tier)
                recommended = P.recommended_preset(entry, mode, tier)
                if recommended is None:
                    desired = self.desired_presets(entry, mode, tier, None)
                    banner = self.banner([], ["No ready-made preset for this GPU tier — pick another tier or 'All GPUs'."])
                    desired[self.banner_html] = {"value": banner}
                    return self.emit(request, desired, client)
                return self.apply(request, client, entry.key, mode, tier, recommended, show_adv)

        def on_preset(request: gr.Request, *args):
            (entry_key, mode, tier, preset, show_adv), client = selection(args)
            if not preset:
                return self.nothing()
            return self.load_with_guardrails(request, client, entry_key, mode, tier, preset, show_adv)

        def on_refresh(request: gr.Request, *args):
            (entry_key, mode, tier, preset, _show_adv), client = selection(args)
            with self.session(request):
                return self.emit(request, self.desired_presets(self.entry(entry_key), mode, tier, preset), client)

        for component, handler in ((self.model_radio, on_model), (self.mode_radio, on_mode), (self.tier_dd, on_tier), (self.preset_dd, on_preset)):
            component.input(None, js=busy_js)
            # always_last: a second quick click waits for the running switch and then applies (the default
            # "once" would drop it, leaving the selector on one model and the settings on another)
            component.input(handler, inputs=selection_inputs, outputs=tracked_outputs, queue=False, trigger_mode="always_last", **quiet).then(None, js=unbusy_js)
        self.refresh_btn.click(on_refresh, inputs=selection_inputs, outputs=[self.tier_dd, self.preset_dd, self.preset_desc], queue=False, **quiet)

        # ------------------------------------------------------------ save / load by name or path
        def on_save(request: gr.Request, *args):
            entry_key, mode, tier, current_preset, _show_adv = args[:5]
            name = args[5]
            finder = self.finder_settings(args[6 + len(fields):])
            client = self.client_values(args[6 : 6 + len(fields)])
            entry = self.entry(entry_key)
            try:
                path = P.save_target(name, entry)
            except ValueError as exc:
                raise gr.Error(str(exc), duration=8)
            if P.is_read_only_preset(path):
                raise gr.Error("Shipped presets are read-only. Type a new name to save your own copy.", duration=8)
            self._sync_values(request, client)
            backend = self.backend_of(entry)
            values = self.values_for(entry, client)
            if "training_mode" in values:
                values["training_mode"] = backend.mode_value(mode)
            # a classic save would store such a value changed (a fraction cut to a whole number) or not at all
            raise_problems(backend.check_values(values, mode), "Not saved")
            existed = os.path.isfile(path)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                backend.save(path, values)
                P.write_metadata_header(path, entry, mode, extra=finder_header(finder))
            except Exception as exc:
                log.exception("Saving preset failed")
                raise gr.Error(f"Could not save the preset: {exc}", duration=10)
            P.remember_selection(self.category, entry.key, mode, P.offered_tier(entry, mode, tier), path)
            gr.Info(f"{'Updated' if existed else 'Saved'} preset: {path}")
            with self.session(request):
                desired = self.desired_presets(entry, mode, tier, path)
                banner = self.banner([f"💾 {'Updated' if existed else 'Saved'} {path}"])
                desired[self.banner_html] = {"value": banner}
                updates = self.emit(request, desired, {self.preset_dd: current_preset, self.tier_dd: tier})
            updates[self.name_tb] = gr.update(value="")
            return updates

        self.save_btn.click(
            on_save,
            inputs=selection_inputs + [self.name_tb] + fields + finder_inputs,
            outputs=[self.tier_dd, self.preset_dd, self.preset_desc, self.banner_html, self.name_tb],
            **quiet,
        )

        def on_load(request: gr.Request, *args):
            (entry_key, mode, tier, _preset, show_adv), client = selection(args)
            name = args[5]
            entry = self.entry(entry_key)
            try:
                path = P.find_preset(name, entry, mode)
            except (ValueError, FileNotFoundError) as exc:
                raise gr.Error(str(exc), duration=8)
            updates, applied = self._guarded_load(request, client, entry_key, mode, tier, path, show_adv)
            if applied:
                updates[self.name_tb] = gr.update(value="")  # a refused name stays, ready to correct
            return updates

        for trigger in (self.load_btn.click, self.name_tb.submit):
            trigger(None, js=busy_js)
            trigger(on_load, inputs=selection_inputs + [self.name_tb], outputs=tracked_outputs + [self.name_tb], **quiet).then(None, js=unbusy_js)

        # ------------------------------------------------------------ instant client-side visibility
        root = self.root_id
        self.adv_toggle.input(None, inputs=[self.adv_toggle], js=ATTR_JS.replace("__ROOT__", root).replace("__ATTR__", "adv").replace("__EXPR__", "!!v"))
        for rule_id, controller, _dependents, _predicate, js_predicate in RULES:
            for item in self.fields.values():
                if item.key == controller:
                    item.component.input(
                        None,
                        inputs=[item.component],
                        js=ATTR_JS.replace("__ROOT__", root).replace("__ATTR__", f"r-{rule_id}").replace("__EXPR__", js_predicate),
                    )
        self.expand_btn.click(None, js=SECTIONS_JS.replace("__ROOT__", root).replace("__OPEN__", "true"))
        self.collapse_btn.click(None, js=SECTIONS_JS.replace("__ROOT__", root).replace("__OPEN__", "false"))
        self.search_tb.input(None, inputs=[self.search_tb], js=SEARCH_JS.replace("__ROOT__", root))

        # ------------------------------------------------------------ browse buttons
        from ..common_gui import get_file_path, get_folder_path

        def browser(item: FieldDef):
            if item.kind == "folder":

                def ask(current):
                    return get_folder_path(current or "")

            else:
                extension = ".toml" if item.key == "dataset_config" else (".txt" if item.key == "sample_prompts" else ".safetensors")
                label = "TOML files" if extension == ".toml" else ("Text files" if extension == ".txt" else "Model files")

                def ask(current):
                    return get_file_path(current or "", default_extension=extension, extension_name=label)

            def browse(current, request: gr.Request = None):
                chosen = ask(current)  # the dialog stays open as long as the user likes: not inside the session lock
                with self.session(request):
                    return self.emit(request, {item.component: {"value": chosen}}, {item.component: current}, user=True)

            return browse

        for item in self.fields.values():
            if item.browse is not None:
                item.browse.click(browser(item), inputs=[item.component], outputs=[item.component], **quiet)

        # ------------------------------------------------------------ model files
        path_items = [item for item in self.fields.values() if item.kind in ("file", "folder") and item.section == "model"]

        def on_paths(request: gr.Request, entry_key, *path_values):
            with self.session(request):
                entry = self.entry(entry_key)
                backend_name = self.backend_of(entry).name
                # every other setting as this browser shows it: a rule such as Wan's task decides which
                # model files are listed (the A14B high-noise DiT, the CLIP of the i2v tasks)
                values = self.mirror_values(request, entry)
                values.update({item.key: value for item, value in zip(path_items, path_values) if backend_name in item.per_backend})
                return self.emit(request, {self.files_html: {"value": self.files_status(entry, values)}})

        for item in path_items:
            item.component.blur(on_paths, inputs=[self.model_radio, *[p.component for p in path_items]], outputs=[self.files_html], queue=False, **quiet)

        def on_download(bundle):
            label = next((text for text, value in self._all_bundles() if value == bundle), bundle)
            gr.Info(DOWNLOADS.start(bundle, label))

        self.download_btn.click(on_download, inputs=[self.bundle_dd], **quiet).then(None, js=poke_js)
        self.download_stop_btn.click(lambda: gr.Info(DOWNLOADS.stop()), **quiet).then(None, js=poke_js)

        def on_downloaded(request: gr.Request, entry_key):
            with self.session(request):
                entry = self.entry(entry_key)
                return self.emit(request, {self.files_html: {"value": self.files_status(entry, self.mirror_values(request, entry))}})

        self.files_refresh_btn.click(on_downloaded, inputs=[self.model_radio], outputs=[self.files_html], queue=False, **quiet)

        # ------------------------------------------------------------ dataset check
        def on_check(request: gr.Request, entry_key, *args):
            entry = self.entry(entry_key)
            client = self.client_values(args)
            self._sync_values(request, client)
            values = self.values_for(entry, client)
            report = self.inspect(entry, values)
            self._remember_report(request, report, 0, self.inspection_key(entry, values))
            samples = sample_previews(report, seed=0)  # first: a file that cannot be previewed joins the problems
            return gr.update(value=report_html(report, entry.name), visible=True), gr.update(value=samples, visible=bool(samples))

        def on_samples(request: gr.Request, entry_key, *args):
            key = getattr(request, "session_hash", None) or "local"
            with self._mirror_lock:
                remembered = self._reports.get(key)
            entry = self.entry(entry_key)
            values = self.values_for(entry, self.client_values(args))
            # another dataset, model or dataset setting since the last check: check again
            if remembered is None or remembered[2] != self.inspection_key(entry, values):
                return on_check(request, entry_key, *args)
            report, seed, inspection = remembered
            self._remember_report(request, report, seed + 1, inspection)
            samples = sample_previews(report, seed=seed + 1)
            return gr.skip(), gr.update(value=samples, visible=bool(samples))

        dataset_outputs = [self.dataset_report, self.dataset_gallery]
        checking_js = DATASET_BUSY_JS.replace("__ROOT__", self.root_id)
        done_js = DATASET_DONE_JS.replace("__ROOT__", self.root_id)
        for button, handler in ((self.check_dataset_btn, on_check), (self.samples_btn, on_samples)):
            button.click(None, js=checking_js)
            button.click(handler, inputs=[self.model_radio, *fields], outputs=dataset_outputs, **quiet).then(None, js=done_js)

        # ------------------------------------------------------------ 🧹 clean old caches (asks first)
        def on_clean(request: gr.Request, entry_key, *args):
            entry = self.entry(entry_key)
            client = self.client_values(args)
            self._sync_values(request, client)
            values = self.values_for(entry, client)
            report = self.inspect(entry, values, probe_seconds=0)  # the folders only: no file is opened
            plan = clean_plan(report)
            self._store(self._clean_plans, request, plan)
            question, result = clean_question_html(report, plan)
            return question, result, f"{1 if question else 0}:{next(self._seq)}"

        def on_clean_yes(request: gr.Request, include_current):
            key = getattr(request, "session_hash", None) or "local"
            with self._mirror_lock:
                plan = self._clean_plans.pop(key, None)
            if not plan:
                return clean_result_html([], ["Nothing to delete: press 🧹 Clean old caches again."], 0), f"0:{next(self._seq)}"
            targets = list(plan["old"]) + (list(plan["current"]) if include_current else [])
            removed, problems, freed = remove_cache_folders(targets)
            if removed:
                log.info(f"Clean old caches: removed {len(removed)} folder(s), {freed} bytes")
            return clean_result_html(removed, problems, freed), f"0:{next(self._seq)}"

        clean_outputs = [self.clean_question, self.clean_result, self.clean_state]
        self.clean_btn.click(None, js=checking_js)
        self.clean_btn.click(on_clean, inputs=[self.model_radio, *fields], outputs=clean_outputs, **quiet).then(None, js=done_js)
        self.clean_no.click(None, js=ATTR_JS.replace("__ROOT__", self.root_id).replace("__ATTR__", "cleanask").replace("__EXPR__", "false"))
        self.clean_yes.click(
            on_clean_yes, inputs=[self.clean_current], outputs=[self.clean_result, self.clean_state], **quiet
        ).then(  # the check shows the dataset as it is now
            on_check, inputs=[self.model_radio, *fields], outputs=dataset_outputs, **quiet
        )

        # ------------------------------------------------------------ training actions
        def on_start(request: gr.Request, entry_key, mode, tier, preset, *args):
            from .runner import RUNNER

            if RUNNER.is_active():  # before the last-run settings are overwritten
                raise gr.Error("A training run is already in progress. Stop it first or wait until it finishes.", title="Training could not start", duration=None)
            entry = self.entry(entry_key)
            backend = self.backend_of(entry)
            finder = self.finder_settings(args[len(fields):])
            args = args[: len(fields)]
            client = self.client_values(args)
            self._sync_values(request, client)
            values = self.values_for(entry, client)
            if "training_mode" in values:
                values["training_mode"] = backend.mode_value(mode)
            preflight(values, backend, mode)
            preset_name = os.path.splitext(os.path.basename(preset))[0].replace("_", " ") if preset else ""
            title = f"{entry.name} · {MODE_LABELS[mode]}"
            meta = {
                "model": f"{entry.icon} {entry.name}",
                "mode": MODE_LABELS[mode],
                "preset": preset_name,
                "category": self.category,
                "entry": entry.key,
                "gpu_ids": str(values.get("gpu_ids") or ""),
            }
            try:
                autosave = os.path.join(P.user_presets_root(), entry.key, f"_last_run_{mode}.toml")
                os.makedirs(os.path.dirname(autosave), exist_ok=True)
                backend.save(autosave, values)
                P.write_metadata_header(autosave, entry, mode, "Settings of the last training run (saved automatically on Start).", extra=finder_header(finder))
            except Exception as exc:
                log.debug(f"Could not auto-save the last run settings: {exc}")
            user_values = values
            values, after, note = self.finder_run(entry, values, finder)  # after the autosave: it keeps your own sampling
            if note:
                gr.Warning(note, duration=10)
            started, message = backend.train(values, title=title, meta=meta, after=after)
            if not started:
                raise gr.Error(message, title="Training could not start", duration=None)
            extra = ""
            if after is not None:
                samples = CF.sample_count(user_values, finder)
                extra = " · 🏆 the best checkpoint is found when it ends" + (f" ({samples} finder samples)" if samples else "")
            gr.Info(f"Training started: {title}{extra}")

        self.start_btn.click(
            on_start,
            inputs=[self.model_radio, self.mode_radio, self.tier_dd, self.preset_dd, *fields, *finder_inputs],
            concurrency_limit=1,
            **quiet,
        ).then(None, js=poke_js)

        def on_print(request: gr.Request, entry_key, mode, *args):
            entry = self.entry(entry_key)
            backend = self.backend_of(entry)
            finder = self.finder_settings(args[len(fields):])
            args = args[: len(fields)]
            client = self.client_values(args)
            self._sync_values(request, client)
            values = self.values_for(entry, client)
            if "training_mode" in values:
                values["training_mode"] = backend.mode_value(mode)
            preflight(values, backend, mode)
            values, after, note = self.finder_run(entry, values, finder)
            try:
                text = backend.print_command(values)
            except gr.Error:
                raise
            except Exception as exc:
                raise gr.Error(f"{type(exc).__name__}: {exc}", title="Command preview failed", duration=None)
            text = text or "(The trainer printed nothing.)"
            if after is not None:
                text += "\n\n# 🏆 when training ends, the best checkpoint finder runs:\n# " + " ".join(after.cmd)
            elif note:
                text += f"\n\n# {note}"
            return gr.update(value=text, visible=True)

        self.print_btn.click(on_print, inputs=[self.model_radio, self.mode_radio, *fields, *finder_inputs], outputs=[self.command_code], **quiet)
        if self.finder_inputs:
            self._wire_finder(quiet, poke_js)

        ask_js = ATTR_JS.replace("__ROOT__", self.root_id).replace("__ATTR__", "stopask")
        self.stop_btn.click(None, js=ask_js.replace("__EXPR__", "true"))
        self.stop_no.click(None, js=ask_js.replace("__EXPR__", "false"))
        self.stop_yes.click(None, js=ask_js.replace("__EXPR__", "false"))

        def on_stop():
            from .runner import RUNNER

            gr.Warning(RUNNER.stop())

        # queued (unlike the instant UI events) so its toast reaches the browser; never waits behind other events
        self.stop_yes.click(on_stop, concurrency_limit=None, **quiet).then(None, js=poke_js)

    def _wire_finder(self, quiet: dict, poke_js: str) -> None:
        inputs = self.finder_inputs

        def changed(request: gr.Request, *args):
            settings = self.finder_settings(args)
            CF.remember(self.category, settings)
            with self.session(request):
                self._sync_values(request, dict(zip(inputs, args)))
                return self.emit(request, {self.bcf_summary: {"value": finder_summary_html(settings)}})

        for component in (self.bcf_enable, self.bcf_seeds, self.bcf_leakage):
            component.input(changed, inputs=inputs, outputs=[self.bcf_summary], queue=False, **quiet)
        self.bcf_prompts.blur(changed, inputs=inputs, outputs=[self.bcf_summary], queue=False, **quiet)

        def subject_changed(request: gr.Request, *args):
            settings = self.finder_settings(args)
            previous = CF.remembered(self.category)["subject"]
            text = CF.replace_subject(CF.prompts_text(settings), previous, settings["subject"])
            settings = CF.normalize({**settings, "prompts": text}, self.category)
            CF.remember(self.category, settings)
            with self.session(request):
                self._sync_values(request, dict(zip(inputs, args)))
                return self.emit(request, self.desired_finder(settings), dict(zip(inputs, args)))

        for trigger in (self.bcf_subject.blur, self.bcf_subject.submit):
            trigger(subject_changed, inputs=inputs, outputs=[*inputs, self.bcf_summary], queue=False, **quiet)

        def default_prompts(request: gr.Request, *args):
            settings = self.finder_settings(args)
            settings["prompts"] = BP.apply_subject(CF.defaults(self.category)["prompts"], settings["subject"])
            CF.remember(self.category, settings)
            with self.session(request):
                return self.emit(request, self.desired_finder(settings), dict(zip(inputs, args)))

        self.bcf_defaults_btn.click(default_prompts, inputs=inputs, outputs=[*inputs, self.bcf_summary], **quiet)

        def from_dataset(request: gr.Request, entry_key, *args):
            settings = self.finder_settings(args)
            entry = self.entry(entry_key)
            values = self.mirror_values(request, entry)
            report = self.inspect(entry, values, probe_seconds=0)
            subject = CF.subject_from_folders([folder.name for folder in report.folders if not folder.regularization], settings["subject"])
            if not subject:
                gr.Warning("No training folder named '<repeats>_<trigger class>' (such as 10_ohwx man): type the subject.", duration=8)
                return {self.bcf_summary: gr.skip()}
            text = CF.replace_subject(CF.prompts_text(settings), settings["subject"], subject)
            settings = CF.normalize({**settings, "subject": subject, "prompts": text}, self.category)
            CF.remember(self.category, settings)
            gr.Info(f"Subject: {subject}")
            with self.session(request):
                return self.emit(request, self.desired_finder(settings), dict(zip(inputs, args)))

        self.bcf_from_dataset.click(from_dataset, inputs=[self.model_radio, *inputs], outputs=[*inputs, self.bcf_summary], **quiet)

        def run_paths(request, entry_key) -> tuple[ModelEntry, dict, str, str]:
            entry = self.entry(entry_key)
            values = self.mirror_values(request, entry)
            output_dir = os.path.abspath(str(values.get("output_dir") or "").strip() or ".")
            return entry, values, output_dir, str(values.get("output_name") or "").strip()

        def show(request: gr.Request, entry_key):
            _entry, _values, output_dir, output_name = run_paths(request, entry_key)
            report = CF.load_report(output_dir, output_name)
            page = CF.results_html(report, output_dir, output_name)
            if report is None:
                status = CF.run_status(output_dir, output_name)
                if status:
                    page = page.replace("</div>", f" {html.escape(status)}</div>", 1)
            items = CF.comparison_gallery(report)
            columns = max(1, len(CF.comparison_points(report)))  # one row per prompt: base · best · last
            return gr.update(value=page, visible=True), gr.update(value=items, visible=bool(items), columns=columns)

        self.bcf_show_btn.click(show, inputs=[self.model_radio], outputs=[self.bcf_result, self.bcf_gallery], **quiet)

        def analyze(request: gr.Request, entry_key, *args):
            from .runner import RUNNER

            if RUNNER.is_active():
                raise gr.Error("A training run is in progress. Analyze when it has finished.", duration=8)
            entry, values, output_dir, output_name = run_paths(request, entry_key)
            if not output_name or not index_samples(output_dir, output_name):
                raise gr.Error(f"No samples of '{output_name}' in {os.path.join(output_dir, 'sample')}. The finder scores the samples a run saved.", duration=10)
            settings = self.finder_settings(args)
            # a run made with the finder (or analyzed before) names its training images in its own settings.json
            earlier = CF.run_settings(output_dir, output_name)
            references = [] if earlier else self.finder_references(entry, values)
            if not earlier and not references:
                raise gr.Error("No training images found to compare the samples with: check the dataset settings.", duration=10)
            settings_path = CF.analysis_settings(values, settings, references)
            after = CF.after_step(output_dir, output_name, settings_path, values.get("gpu_ids"))
            meta = {"model": f"{entry.icon} {entry.name}", "mode": "🏆 Best checkpoint", "category": self.category, "entry": entry.key,
                    "gpu_ids": str(values.get("gpu_ids") or ""), "finder": "1"}
            started, message = RUNNER.start_after_step(after, title=f"{entry.name} · Best checkpoint", meta=meta)
            if not started:
                raise gr.Error(message, duration=10)
            gr.Info("Finding the best checkpoint: progress in the monitor below. 📊 Show results when it is done.")

        self.bcf_analyze_btn.click(analyze, inputs=[self.model_radio, *inputs], concurrency_limit=1, **quiet).then(None, js=poke_js)

        def open_report(request: gr.Request, entry_key):
            _entry, _values, output_dir, output_name = run_paths(request, entry_key)
            path = os.path.join(CF.run_folder(output_dir, output_name), "report.html")
            if not os.path.isfile(path):
                raise gr.Error(f"No report yet: {path}", duration=8)
            import pathlib
            import webbrowser

            webbrowser.open(pathlib.Path(path).as_uri())

        self.bcf_open_btn.click(open_report, inputs=[self.model_radio], **quiet)

    def inspection_key(self, entry: ModelEntry, values: dict) -> str:
        """Everything a dataset check depends on: the model and its dataset settings."""
        backend = self.backend_of(entry)
        keys = ("dataset_config_mode", "parent_folder_path", "dataset_config", "dataset_caption_extension", "create_missing_captions",
                "dataset_cache_directory", "dataset_batch_size", "dataset_resolution_width", "dataset_resolution_height", "reg_data_dir",
                "use_control_images", "control_directory_name", "dataset_control_directory", "model_version", "task", "ltx2_mode",
                "max_train_epochs", "max_train_steps", "gradient_accumulation_steps")
        return json.dumps([entry.key, backend.name, *[str(values.get(key, "")) for key in keys]])

    def inspect(self, entry: ModelEntry, values: dict, probe_seconds: float | None = None) -> DatasetReport:
        """The dataset the selected model would train on: the folder structure, or the dataset TOML."""
        backend = self.backend_of(entry)
        profile = dataset_profile(backend.name, values)
        kohya = backend.engine == "kohya"  # sd-scripts' own folder rules, not the musubi generators'
        control_name, control_required = ("", False) if kohya else control_settings(backend.name, values)
        toml_mode = str(values.get("dataset_config_mode") or "") == "Use TOML File"
        try:  # what Start passes: ".txt" also when "txt" was typed
            extension = backend.trainer_value("dataset_caption_extension", values.get("dataset_caption_extension") or ".txt")
        except Exception:  # noqa: BLE001
            extension = str(values.get("dataset_caption_extension") or ".txt")
        return inspect_dataset(
            str(values.get("parent_folder_path") or ""),
            profile=profile,
            caption_extension=str(extension or ".txt"),
            control_name=control_name,
            control_required=control_required,
            cache_directory=str(values.get("dataset_cache_directory") or "cache_dir"),
            create_missing_captions=bool(values.get("create_missing_captions")),
            batch_size=values.get("dataset_batch_size") or 1,
            resolution=(values.get("dataset_resolution_width"), values.get("dataset_resolution_height")) if values.get("dataset_resolution_width") else None,
            dataset_toml=str(values.get("dataset_config") or "") if toml_mode else "",
            kohya=kohya,
            regularization_folder=str(values.get("reg_data_dir") or "") if kohya else "",
            epochs=values.get("max_train_epochs"),
            max_steps=values.get("max_train_steps"),
            gradient_accumulation=values.get("gradient_accumulation_steps") or 1,
            **({"probe_seconds": probe_seconds} if probe_seconds is not None else {}),
        )

    def _remember_report(self, request, report: DatasetReport, seed: int, inspection: str) -> None:
        self._store(self._reports, request, (report, seed, inspection))

    def _store(self, store: OrderedDict, request, value) -> None:
        """One value per browser session, at most MAX_SESSIONS (least recently used dropped)."""
        key = getattr(request, "session_hash", None) or "local"
        with self._mirror_lock:
            store[key] = value
            store.move_to_end(key)
            while len(store) > MAX_SESSIONS:
                store.popitem(last=False)

    def vram_note(self, request) -> str:
        """The VRAM panel for this session: what the edits since the last load do to VRAM."""
        key = getattr(request, "session_hash", None) or "local"
        with self._mirror_lock:
            base = self._vram_base.get(key)
        if base is None:
            return ""
        entry_key, mode, baseline, preset_path = base
        entry = self.entry(entry_key)
        changes = advise(self.backend_of(entry).name, baseline, self.mirror_values(request, entry), mode)
        return advice_html(changes, preset_path, P.detected_gpu())

    def _sync_values(self, request, client: dict) -> None:
        """Ground-truth field values from an event that sent them: refresh the session mirror."""
        with self.session(request):
            mirror = self._mirror(request)
            for component, value in client.items():
                known = mirror.setdefault(component._id, {})
                if "value" in known and not _same(known["value"], value):
                    known["user"] = True
                known["value"] = value

    def _all_bundles(self) -> list[tuple[str, str]]:
        bundles = []
        for entry in self.entries:
            bundles.extend(entry.download_bundles)
        return bundles

    # ------------------------------------------------------------------ guardrails
    def load_with_guardrails(self, request, client: dict, entry_key: str, mode: str, tier: str, path: str, show_adv: bool) -> dict:
        return self._guarded_load(request, client, entry_key, mode, tier, path, show_adv)[0]

    def _guarded_load(self, request, client: dict, entry_key: str, mode: str, tier: str, path: str, show_adv: bool) -> tuple[dict, bool]:
        """Load a preset the user picked or named: (updates, whether it was applied)."""
        with self.session(request):
            entry = self.entry(entry_key)
            name = os.path.basename(path)
            identity = P.identify_preset(path)
            if identity.category and identity.category != self.category:
                target = CATEGORY_TITLES.get(identity.category, identity.category)
                message = f"Not loaded: {name} is a preset for {identity.describe()}, not for {CATEGORY_TITLES[self.category]}. Open the {target} tab and load it there."
                return self.refuse(request, client, message), False
            if identity.entry_key and identity.entry_key not in self.entry_keys and identity.category == self.category:
                message = f"Not loaded: {name} is a preset for {identity.entry.name}, which is not available in this install (its trainer failed to load; see the log)."
                return self.refuse(request, client, message), False
            messages = []
            target_entry = entry
            if identity.entry_key and identity.entry_key != entry.key:
                target_entry = self.entry(identity.entry_key)
                messages.append(f"🔁 Switched base model to {target_entry.name} to match the preset.")
            target_mode = mode
            if identity.mode and identity.mode != mode:
                if identity.mode in target_entry.modes:
                    target_mode = identity.mode
                    messages.append(f"🔁 Switched training type to {MODE_LABELS[target_mode]} to match the preset.")
                else:
                    message = f"Not loaded: {name} is a {MODE_LABELS[identity.mode]} preset, which {target_entry.name} does not support."
                    return self.refuse(request, client, message), False
            warnings = []
            if identity.source == "unknown":
                warnings.append("Could not tell which model this file was made for; only the settings that match the selected model were applied.")
            if target_mode not in target_entry.modes:
                target_mode = target_entry.modes[0]
            updates = self.apply(
                request, client, target_entry.key, target_mode, tier, path, show_adv, messages=messages, warnings=warnings, refuse_unreadable=True,
                finder_from_preset=True,
            )
            applied = self.state_html in updates  # an applied preset always re-sends the state (its seq); a refusal never
            if applied and messages:
                gr.Info(" ".join(messages))
            return updates, applied

    # ------------------------------------------------------------------ page load
    def remembered_selection(self) -> tuple[ModelEntry, str, str, str | None]:
        remembered = P.remembered(self.category)
        entry = self.entry(remembered.get("entry"))
        mode = remembered.get("mode") if remembered.get("mode") in entry.modes else entry.modes[0]
        tier = P.offered_tier(entry, mode, remembered.get("tier"))
        preset = P.remembered_preset(self.category, entry.key, mode) or P.recommended_preset(entry, mode, tier)
        return entry, mode, tier, preset

    def restore(self, request) -> dict:
        """Re-apply the remembered selection when a browser opens the page. When it is the build-time
        selection and the preset file is unchanged, only the model-file status is refreshed (files may
        have been downloaded since the app started)."""
        with self.session(request):
            entry, mode, tier, preset = self.remembered_selection()
            self.forget(request)  # a (re)loaded page shows the build-time state
            finder = self.desired_finder(CF.remembered(self.category))
            if (entry.key, mode, tier, preset, _signature(preset)) == self.initial:
                values = self.mirror_values(request, entry)
                self._store(self._vram_base, request, (entry.key, mode, dict(values), preset))  # the VRAM panel's baseline
                return self.emit(request, {self.files_html: {"value": self.files_status(entry, values)}, **finder})
            updates = self.apply(request, {}, entry.key, mode, tier, preset, False, keep_user_paths=False)
            if finder:
                updates.update(self.emit(request, finder))
            return updates


def finder_values(settings: dict) -> list:
    """The finder's input values in ``Workspace.finder_inputs`` order."""
    return [settings["enabled"], settings["subject"], CF.prompts_text(settings), settings["seeds"], settings["leakage"]]


def finder_header(settings: dict | None) -> dict | None:
    return {CF.HEADER_KEY: CF.header_value(settings)} if settings else None


def finder_summary_html(settings: dict) -> str:
    state = "on" if settings["enabled"] else "off"
    return (
        f'<div class="bcf-summary-line bcf-{state}"><span class="bcf-dot"></span><b>{"On" if settings["enabled"] else "Off"}</b> · '
        f"{html.escape(CF.summary_line(settings))}</div>"
    )


FINDER_INTRO = """<div class="bcf-intro">
<div class="bcf-step"><span>1</span><div><b>Sample</b><small>Your test prompts with the base model before training and at every checkpoint save (same seeds).</small></div></div>
<div class="bcf-step"><span>2</span><div><b>Score</b><small>Likeness to your training photos, prompt following, variety and over-training signs, with local models.</small></div></div>
<div class="bcf-step"><span>3</span><div><b>Recommend</b><small>The checkpoint where likeness levels off before over-training, with the reasons and a side-by-side report.</small></div></div>
</div>"""


# Your own data/output locations survive a model switch or a shipped preset (a shipped preset holds
# the preset author's paths). Loading one of your own presets restores everything it contains.
STICKY_KEYS = ("dataset_config_mode", "parent_folder_path", "dataset_config", "output_dir", "sample_prompts", "logging_dir")
STICKY_LABELS = {"parent_folder_path": "dataset folder", "output_dir": "output folder", "dataset_config": "dataset TOML"}


def _foreign_path(value) -> bool:
    """An absolute path that does not exist on this PC (a shipped preset author's machine path)."""
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    absolute = os.path.isabs(text) or text.startswith("/") or (len(text) > 2 and text[1] == ":")
    return absolute and not os.path.exists(text)

FOLDER_DATASET_MODE = "Generate from Folder Structure"


def preflight(values: dict, backend: Backend | None = None, mode: str | None = None) -> None:
    """Plain-language checks before Start / Print Command. The trainers' own errors name internal keys
    and stop at the first problem; this lists every problem at once, with the setting's own label."""
    problems: list[str] = []
    dataset_mode = str(values.get("dataset_config_mode") or "")
    own_checks = backend is not None and backend.engine == "kohya"  # kohya validates these itself, with its own labels
    if own_checks:
        dataset_mode = ""
    if dataset_mode == FOLDER_DATASET_MODE and "parent_folder_path" in values:
        parent = clean_path(values.get("parent_folder_path"))
        if not parent:
            problems.append("Set the Parent Dataset Folder in 📁 Dataset, or switch Dataset Configuration Method to 'Use TOML File'.")
        elif not os.path.isdir(parent):
            problems.append(f"The Parent Dataset Folder does not exist: {parent}")
    elif dataset_mode and "dataset_config" in values:
        dataset_toml = clean_path(values.get("dataset_config"))
        if not dataset_toml:
            problems.append("Choose the dataset TOML file in 📁 Dataset, or switch Dataset Configuration Method to 'Generate from Folder Structure'.")
        elif not os.path.isfile(dataset_toml):
            problems.append(f"The dataset TOML file does not exist: {dataset_toml}")
    if not own_checks and "output_dir" in values and not clean_path(values.get("output_dir")):
        problems.append("Set the Output Folder in 💾 Output & Saving.")
    if backend is not None:
        problems.extend(backend.check(values, mode))
    raise_problems(problems)


def raise_problems(problems: list[str], prefix: str = "") -> None:
    """One toast listing every problem, one per line (the toast renders HTML; the text is escaped: it can
    echo what was typed)."""
    if not problems:
        return
    shown = problems[:8] + ([f"…and {len(problems) - 8} more."] if len(problems) > 8 else [])
    title = "Setting needs attention" if len(problems) == 1 else f"{len(problems)} settings need attention"
    raise gr.Error("<br>".join(f"• {html.escape(problem)}" for problem in shown), title=f"{prefix}: {title.lower()}" if prefix else title, duration=None)


STATE_JS = """
const apply = () => {
  const root = document.getElementById('__ROOT__');
  if (!root) return;
  let state = {};
  try { state = JSON.parse(props.value || '{}'); } catch (e) { return; }
  for (const [key, value] of Object.entries(state)) root.setAttribute('data-' + key, String(value));
  root.classList.remove('ws-busy');
  // After "Expand all" / "Collapse all", sections that a model / training type switch newly reveals
  // follow that choice; sections that stayed visible keep whatever the user did with them.
  requestAnimationFrame(() => root.querySelectorAll('.ws-section').forEach((section) => {
    const visible = getComputedStyle(section).display !== 'none';
    const revealed = visible && section.dataset.wasVisible === '0';
    section.dataset.wasVisible = visible ? '1' : '0';
    if (!revealed || !(root.dataset.expanded === '1' || root.dataset.expanded === '0')) return;
    const button = section.querySelector('button.label-wrap');
    if (button && button.classList.contains('open') !== (root.dataset.expanded === '1')) button.click();
  }));
};
apply();
watch('value', apply);
// Number fields: an emptied required number (or text the browser cannot read as a number) would only
// fail at Start, so when the field loses focus the previous number comes back, for the screen and for
// Gradio alike (an input event updates the component), with a short flash. Optional numbers, whose
// blank means "not set", stay blank.
const root = document.getElementById('__ROOT__');
if (root && !root.dataset.numberGuard) {
  root.dataset.numberGuard = '1';
  const isNumber = (el) => el && el.matches && el.matches('input[type="number"]');
  const remember = (el) => { if (el.value !== '' && !el.validity.badInput) el.dataset.lastValid = el.value; };
  root.addEventListener('focusin', (event) => { if (isNumber(event.target)) remember(event.target); }, true);
  root.addEventListener('input', (event) => { if (isNumber(event.target)) remember(event.target); }, true);
  root.addEventListener('focusout', (event) => {
    const el = event.target;
    if (!isNumber(el) || !(el.value === '' || el.validity.badInput)) return;
    const optional = !!el.closest('.ws-optional-number');
    if (optional || el.dataset.lastValid === undefined) {
      if (el.validity.badInput) { el.value = ''; el.dispatchEvent(new Event('input', {bubbles: true})); }
      return;
    }
    el.value = el.dataset.lastValid;
    el.dispatchEvent(new Event('input', {bubbles: true}));
    el.classList.add('ws-restored');
    el.title = 'Empty or not a number: the previous value is kept.';
    setTimeout(() => el.classList.remove('ws-restored'), 1600);
  }, true);
}
"""

BUSY_JS = """() => { const r = document.getElementById('__ROOT__'); if (r) { r.classList.add('ws-busy'); setTimeout(() => r.classList.remove('ws-busy'), 4000); } }"""

SECTIONS_JS = """() => {
  const root = document.getElementById('__ROOT__');
  if (!root) return;
  root.dataset.expanded = __OPEN__ ? '1' : '0';  // STATE_JS opens sections that appear later too
  root.querySelectorAll('.ws-section').forEach((section) => {
    if (getComputedStyle(section).display === 'none') return;
    const button = section.querySelector('button.label-wrap');
    if (button && button.classList.contains('open') !== __OPEN__) button.click();
  });
}"""

POKE_JS = """() => { const poll = window.__udPoll && window.__udPoll['__ROOT__']; if (poll) poll(); }"""

ATTR_JS = """(v) => { const r = document.getElementById('__ROOT__'); if (r) r.setAttribute('data-__ATTR__', (__EXPR__) ? '1' : '0'); }"""

CLEAN_STATE_JS = """
const apply = () => {
  const root = document.getElementById('__ROOT__');
  if (root) root.setAttribute('data-cleanask', String(props.value || '0').startsWith('1') ? '1' : '0');
};
apply();
watch('value', apply);
"""

# the dataset check can take seconds on a big dataset: show that it runs (CSS .ds-checking)
DATASET_BUSY_JS = """() => { const r = document.querySelector('#__ROOT__ .dataset-tools'); if (r) r.classList.add('ds-checking'); }"""
DATASET_DONE_JS = """() => { const r = document.querySelector('#__ROOT__ .dataset-tools'); if (r) r.classList.remove('ds-checking'); }"""

UNBUSY_JS = """() => { const r = document.getElementById('__ROOT__'); if (r) r.classList.remove('ws-busy'); }"""

SEARCH_JS = """async (query) => {
  const root = document.getElementById('__ROOT__');
  if (!root) return;
  const q = (query || '').trim().toLowerCase();
  const sections = Array.from(root.querySelectorAll('.ws-section')).filter((s) => getComputedStyle(s).display !== 'none' || s.classList.contains('ws-hide'));
  if (q) {
    // closed sections are not rendered yet: open them once so their settings can be searched
    let opened = false;
    for (const section of sections) {
      const button = section.querySelector('button.label-wrap');
      if (button && !button.classList.contains('open')) { button.click(); opened = true; section.dataset.searchOpened = '1'; }
    }
    if (opened) await new Promise((resolve) => setTimeout(resolve, 80));
  }
  root.querySelectorAll('.vh').forEach((host) => {
    if (!q) { host.classList.remove('ws-hide'); return; }
    const words = (host.innerText || '') + ' ' + Array.from(host.querySelectorAll('[class*="f-"]')).concat([host]).map((n) => Array.from(n.classList).filter((c) => c.startsWith('f-')).join(' ')).join(' ').replace(/f-/g, ' ').replace(/_/g, ' ');
    host.classList.toggle('ws-hide', !words.toLowerCase().includes(q));
  });
  for (const section of sections) {
    const button = section.querySelector('button.label-wrap');
    if (!q) {
      section.classList.remove('ws-hide');
      if (section.dataset.searchOpened === '1' && button && button.classList.contains('open')) button.click();
      delete section.dataset.searchOpened;
      continue;
    }
    const hit = Array.from(section.querySelectorAll('.vh:not(.ws-hide)')).some((el) => getComputedStyle(el).display !== 'none');
    section.classList.toggle('ws-hide', !hit);
  }
}"""
