"""Uniform adapter over the trainer backends used by the unified UI.

``LegacyMusubiBackend`` drives one of the existing musubi trainer modules through its own
``*_gui_actions`` dispatcher, so loading (with portable model-path relocation and training-mode
inference), saving, validation, runtime-TOML writing and command building stay exactly the code the
classic tabs run. ``KohyaBackend`` drives the SDXL / FLUX.1 sd-scripts backend.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import gradio as gr
import toml

from ..custom_logging import setup_logging
from .adapter import ADAPTER_KEY, NETWORK_KEY, adapter_spec, fold_adapter, is_synthetic, unfold_adapter
from .arguments import ARGUMENT_KEYS, argument_problems, normalize_items
from .arguments import LITERAL_KEYS as LITERAL_ARGUMENT_KEYS
from .legacy_tabs import LEGACY_BACKENDS, actions_callable, legacy_module
from .param_spec import FULL, LORA, ParamSpec
from .schema import PATH_KINDS, PATH_TEXT_KEYS, WHOLE_NUMBER_KEYS, ZERO_MEANS_UNSET, arg_type, build_param_specs, resolve_placeholders
from .values import ValueProblem, json_like_number, parse_bool, parse_number, to_trainer_value

log = setup_logging()

FOLDER_DATASET_MODE = "Generate from Folder Structure"


def _tidy_path(key: str, value):
    """Absolute default folders/files with native separators (legacy tabs build them from a '/' app dir)."""
    if key in PATH_KINDS and isinstance(value, str) and value and os.path.isabs(value):
        return os.path.normpath(value)
    return value


def _is_path(key: str, spec: ParamSpec | None = None) -> bool:
    return key in PATH_KINDS or key in PATH_TEXT_KEYS or (spec is not None and spec.kind in ("file", "folder"))


def _local_model_files(payload: dict) -> dict:
    """Model paths as this PC has the files, for Start and Print Command: a preset loaded before its models were
    downloaded keeps the path it was written with (the presets' Training_Models_*/... convention, or another machine's
    G:/.../Training_Models_X/file), and ⬇️ Download then puts the files in this install's Training_Models_* folders. The
    trainers run from the app folder, so such a path is resolved here (common_gui.resolve_portable_model_value: existing
    paths and Hugging Face ids stay as they are, a missing file stays as written and the trainer names it)."""
    from ..common_gui import PORTABLE_MODEL_PATH_KEYS, resolve_portable_model_value

    for key in PORTABLE_MODEL_PATH_KEYS & payload.keys():
        if payload[key]:
            payload[key] = resolve_portable_model_value(key, payload[key])
    return payload


def _fresh_dataset(payload: dict) -> dict:
    """Folder mode builds the dataset TOML from the current folder on every Start. The classic train code
    reuses ``generated_toml_path`` (and FLUX.2 ``dataset_config``) whenever that file exists, so a TOML a
    preset carries, or one made for another folder or model, is never passed on."""
    if str(payload.get("dataset_config_mode") or "") == FOLDER_DATASET_MODE:
        for key in ("generated_toml_path", "dataset_config"):
            if key in payload:
                payload[key] = ""
    return payload


def _hand_over_dataset(payload: dict, notes: list) -> None:
    """After prepare_dataset built this run's dataset TOML (with audio clips, mixed-folder images), the classic
    train must use it as it is: in folder mode it compares the TOML with the folder and may generate its own
    (dropping an LTX audio dataset whose clips sit in the cache folder, or Kandinsky 6 mixed-folder images)."""
    if notes and payload.get("dataset_config"):
        payload["dataset_config_mode"] = "Use TOML File"


def _remove_quietly(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass

LORA_MODE_FALLBACK = "LoRA Training"
FULL_MODE_FALLBACK = "Full Fine-Tuning"

# ------------------------------------------------------------------------------------------------
# Quiet legacy toasts: the trainer modules call gr.Info / gr.Warning while loading presets. The
# unified UI reports loads itself, so those calls are collected instead of shown while ``quiet()``
# is active on the calling thread.
# ------------------------------------------------------------------------------------------------
_quiet = threading.local()
_ORIGINAL_INFO = gr.Info
_ORIGINAL_WARNING = gr.Warning


def _quiet_info(message="", *args, **kwargs):
    sink = getattr(_quiet, "sink", None)
    if sink is not None:
        sink.append(("info", str(message)))
        return None
    return _ORIGINAL_INFO(message, *args, **kwargs)


def _quiet_warning(message="", *args, **kwargs):
    sink = getattr(_quiet, "sink", None)
    if sink is not None:
        sink.append(("warning", str(message)))
        return None
    return _ORIGINAL_WARNING(message, *args, **kwargs)


def install_quiet_toasts() -> None:
    if gr.Info is not _quiet_info:
        gr.Info = _quiet_info
        gr.Warning = _quiet_warning
    _install_dialog_guard()


DIALOG_FUNCTIONS = ("get_file_path_or_save_as", "get_saveasfile_path", "get_file_path", "get_folder_path", "get_saveasfilename_path")


def _install_dialog_guard() -> None:
    """Never open a native file dialog from a trainer's load/save/train code run by the unified UI.

    Inside ``quiet()`` the trainer functions receive the path the unified UI already chose; their
    dialog helpers return that path unchanged. The UI's own browse buttons run outside ``quiet()``.
    """
    from .. import common_gui

    modules = [common_gui] + [legacy_module(backend) for backend in LEGACY_BACKENDS.values()]
    for module in modules:
        for name in DIALOG_FUNCTIONS:
            original = getattr(module, name, None)
            if original is None or getattr(original, "_unified_guard", False):
                continue

            def guarded(*args, _original=original, **kwargs):
                if getattr(_quiet, "sink", None) is not None:
                    return args[0] if args else kwargs.get("file_path", kwargs.get("folder_path", ""))
                return _original(*args, **kwargs)

            guarded._unified_guard = True
            setattr(module, name, guarded)


@contextlib.contextmanager
def quiet():
    previous = getattr(_quiet, "sink", None)
    sink: list[tuple[str, str]] = []
    _quiet.sink = sink
    try:
        yield sink
    finally:
        _quiet.sink = previous


def _is_update(value) -> bool:
    return isinstance(value, dict) and value.get("__type__") == "update"


@dataclass
class LoadResult:
    values: dict
    messages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    matched_keys: int = 0
    file_keys: int = 0
    failed: bool = False  # the trainer reported that it could not read the file


# ------------------------------------------------------------------------------------------------
# Runtime TOML type gate
# ------------------------------------------------------------------------------------------------
_ACTIVE_TRAINER = threading.local()


@contextlib.contextmanager
def trainer_scope(name: str):
    """Mark the calling thread as running ``name``'s train action for the unified UI."""
    previous = getattr(_ACTIVE_TRAINER, "name", None)
    _ACTIVE_TRAINER.name = name
    try:
        yield
    finally:
        _ACTIVE_TRAINER.name = previous


def enforce_trainer_types(name: str, parameters):
    """(key, value) pairs with every value in the type the trainer's argparse option declares.

    The last step before the runtime TOML is written, so quirks of the classic tab code (for example
    Qwen Image collapsing list options to their first item) cannot change what the trainer receives.
    Values that cannot be converted are left unchanged; Start already reported them.
    """
    fixed = []
    for key, value in parameters:
        arg = arg_type(name, key)
        if arg is not None:
            try:
                value = _literal_arguments(key, to_trainer_value(value, label=key, kind="", arg=arg, is_path=_is_path(key)))
            except ValueProblem:
                pass
        fixed.append((key, value))
    return fixed


def _literal_arguments(key: str, value):
    """Optimizer / scheduler arguments are read with ast.literal_eval by the trainers: true -> True, 0,01 -> 0.01."""
    if key in LITERAL_ARGUMENT_KEYS and isinstance(value, list):
        return normalize_items(value, literal=True)
    return value


def install_toml_type_gate() -> None:
    """Wrap common_gui.SaveConfigFileToRun (every musubi trainer tab writes its runtime TOML with it).
    The gate only acts inside ``trainer_scope``, so the classic UI keeps its exact behaviour."""
    from .. import common_gui

    original = common_gui.SaveConfigFileToRun
    if getattr(original, "_unified_gate", False):
        return

    def gated(parameters, *args, **kwargs):
        name = getattr(_ACTIVE_TRAINER, "name", None)
        if name:
            parameters = enforce_trainer_types(name, list(parameters))
        return original(parameters, *args, **kwargs)

    gated._unified_gate = True
    common_gui.SaveConfigFileToRun = gated
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("musubi_tuner_gui.") and getattr(module, "SaveConfigFileToRun", None) is original:
            module.SaveConfigFileToRun = gated


class Backend:
    name: str = ""
    engine: str = "musubi"

    @property
    def specs(self) -> list[ParamSpec]:
        raise NotImplementedError

    def keys(self) -> list[str]:
        return [spec.key for spec in self.specs]

    def spec_map(self) -> dict[str, ParamSpec]:
        specs = self.specs
        cached = getattr(self, "_spec_map_cache", None)
        if cached is None or cached[0] is not specs:
            cached = (specs, {spec.key: spec for spec in specs})
            self._spec_map_cache = cached
        return cached[1]

    def mode_value(self, mode: str) -> str:
        return LORA_MODE_FALLBACK if mode == LORA else FULL_MODE_FALLBACK

    def mode_of_value(self, value: Any) -> str:
        from ..full_finetune_gui import is_full_fine_tuning

        try:
            return FULL if is_full_fine_tuning(value) else LORA
        except ValueError:
            return LORA

    def defaults(self, mode: str) -> dict:
        raise NotImplementedError

    def trainer_value(self, key: str, value: Any) -> Any:
        """``value`` as the trainer must receive it (raises ValueProblem for unusable input)."""
        spec = self.spec_map().get(key)
        if spec is None:
            return value
        return _literal_arguments(
            key,
            to_trainer_value(
                value,
                label=spec.label or key,
                kind=spec.kind,
                arg=arg_type(self.name, key),
                is_path=_is_path(key, spec),
                whole=self.whole_number(key),
            ),
        )

    def whole_number(self, key: str) -> bool:
        """The trainer, or the classic code that runs it, takes only whole numbers for this setting."""
        return False

    def hub_model_id(self, key: str, value) -> bool:
        """``value`` of this model-file setting is a Hugging Face id the trainer downloads itself (no local file)."""
        return False

    def check_values(self, values: dict, mode: str | None = None) -> list[str]:
        """Plain-language problems with the values the user entered (empty list: all usable). Save refuses
        these too: a classic save would store such a value changed or not at all."""
        problems = []
        for spec in self.specs:
            if mode is not None and mode not in spec.modes:
                continue
            value = values.get(spec.key, spec.default)
            try:
                converted = self.trainer_value(spec.key, value)
            except ValueProblem as exc:
                problems.append(str(exc))
                continue
            # blank is "not set" where 0 is the default or the trainer's own default is None (the classic loaders
            # read a saved 0 back as blank there: max epochs 0 = use max steps, max timestep 0 = full range)
            if converted is None and spec.kind in ("number", "slider") and spec.visible and spec.default not in (None, 0):
                if (arg_type(self.name, spec.key) or {"default": 0}).get("default") is not None:
                    problems.append(f"{spec.label or spec.key} is empty. Enter a value (default: {spec.default}).")
            elif isinstance(converted, float) and not converted.is_integer() and self.whole_number(spec.key):
                problems.append(f"{spec.label or spec.key} must be a whole number (got {value}).")
            elif _is_path(spec.key, spec) and isinstance(converted, str) and "\n" in converted and spec.kind != "textarea":
                if not (self.engine == "kohya" and spec.key == "sample_prompts"):  # kohya also takes the prompts themselves
                    problems.append(f"{spec.label or spec.key}: one path per field (it contains line breaks).")
            elif spec.key in ARGUMENT_KEYS and self.engine != "kohya":  # kohya_backend checks its own (sd-scripts rules)
                problems.extend(argument_problems(spec.label or spec.key, value, literal=spec.key in LITERAL_ARGUMENT_KEYS))
            elif spec.key == "gpu_ids":
                from ..class_accelerate_launch import normalize_gpu_ids

                try:
                    normalize_gpu_ids(value)
                except ValueError as exc:
                    problems.append(f"{spec.label or spec.key}: {exc}")
        # Wan 2.2 high + low noise DoRA trains one magnitude per expert (<name>.safetensors for the low-noise model,
        # <name>_high_noise.safetensors for the high-noise model): nothing to refuse
        return problems

    def check(self, values: dict, mode: str | None = None) -> list[str]:
        """Everything Start / Print Command must refuse before they run."""
        return self.check_values(values, mode)

    def load(self, path: str, values: dict) -> LoadResult:
        raise NotImplementedError

    def save(self, path: str, values: dict) -> None:
        raise NotImplementedError

    def print_command(self, values: dict) -> str:
        raise NotImplementedError

    def train(self, values: dict, *, title: str, meta: dict, after=None) -> tuple[bool, str]:
        raise NotImplementedError


class LegacyMusubiBackend(Backend):
    def __init__(self, name: str):
        self.name = name
        self.legacy = LEGACY_BACKENDS[name]
        self._specs: list[ParamSpec] | None = None
        self._actions = None
        self._int_keys: frozenset | None = None

    def whole_number(self, key: str) -> bool:
        spec = self.spec_map().get(key)
        if spec is None or spec.kind not in ("number", "slider"):
            return False
        if self._int_keys is None:  # LTX2_INT_KEYS, MINIMAX_H3_INT_KEYS: converted with int() before training
            module = legacy_module(self.legacy)
            sets = [getattr(module, name) for name in dir(module) if name.endswith("_INT_KEYS")]
            self._int_keys = frozenset(item for keys in sets if isinstance(keys, (set, frozenset, list, tuple)) for item in keys)
        extras = spec.extras or {}
        if (arg_type(self.name, key) or {}).get("type") == "int" or extras.get("precision") == 0:
            return True
        return key in self._int_keys or (key in WHOLE_NUMBER_KEYS and arg_type(self.name, key) is None)

    @property
    def specs(self) -> list[ParamSpec]:
        if self._specs is None:
            specs = build_param_specs(self.name)
            if any(spec.key == NETWORK_KEY and LORA in spec.modes for spec in specs):
                specs.append(adapter_spec())  # last: the classic trainer's positional arguments end before it
            self._specs = specs
        return self._specs

    @property
    def actions(self):
        if self._actions is None:
            self._actions = actions_callable(self.legacy)
        return self._actions

    def mode_value(self, mode: str) -> str:
        spec = self.spec_map().get("training_mode")
        choices = spec.choice_values() if spec and spec.choices else []
        if mode == LORA:
            return next((choice for choice in choices if "lora" in str(choice).lower()), LORA_MODE_FALLBACK)
        return next((choice for choice in choices if "fine" in str(choice).lower()), FULL_MODE_FALLBACK)

    # ---------------------------------------------------------------- values
    def coerce(self, key: str, value: Any) -> Any:
        """The value as the trainer must receive it.

        Settings the training script reads get exactly the type its argparse option declares (the
        runtime TOML bypasses argparse conversion: ``1.0`` or ``"1e-4"`` would reach the trainer as
        is), paths are cleaned (values.py). GUI-only settings keep the types the classic tabs get
        through Gradio's JSON round trip (whole numbers as int).
        """
        spec = self.spec_map().get(key)
        if spec is None:
            return value
        family = spec.extras.get("family")
        if family == "list":
            if value is None:
                return []
            return list(value) if isinstance(value, (list, tuple)) else [value]
        label = spec.label or key
        if arg_type(self.name, key) is None:
            if family == "bool":
                return parse_bool(value, label=label)
            if family == "number":
                number = parse_number(value, label=label, whole=self.whole_number(key))
                if number is None:
                    return None
                precision = spec.extras.get("precision")
                if precision == 0:
                    return int(round(number))
                if isinstance(precision, int) and precision > 0:
                    number = round(number, precision)
                return json_like_number(number)
        converted = self.trainer_value(key, value)
        if converted is None and spec.kind in ("text", "textarea", "file", "folder"):
            return ""
        if key in ZERO_MEANS_UNSET and converted == 0:
            return None  # Max Timestep 0 would train without noise; the classic tabs read it as "not set"
        return converted

    def ordered_args(self, values: dict) -> list:
        """The classic trainer's positional arguments (the LoRA / DoRA switch folded into Network Arguments)."""
        if self.mode_of_value(values.get("training_mode", self.mode_value(LORA))) == LORA:
            values = fold_adapter(values)
        return [self.coerce(spec.key, values.get(spec.key, spec.default)) for spec in self.specs if not is_synthetic(spec)]

    def normalized(self, values: dict) -> dict:
        """All values converted for the trainer (cleaned paths, argparse types); keys it does not know are kept."""
        payload = dict(values)
        for spec in self.specs:
            if spec.key in payload:
                payload[spec.key] = self.coerce(spec.key, payload[spec.key])
        return payload

    def defaults(self, mode: str) -> dict:
        values = {spec.key: spec.default for spec in self.specs}
        values.update(self._runtime_defaults(mode))
        if "training_mode" in values:
            values["training_mode"] = self.mode_value(mode)
        values.update(self._mode_recipe(mode, values))
        return values

    def _mode_recipe(self, mode: str, values: dict) -> dict:
        """What the classic tab changes when the user switches its Training Mode radio (presets keep their own)."""
        module = legacy_module(self.legacy)
        changes: dict = {}
        if self.legacy.module.endswith("ltx2_lora_gui") and mode == FULL:
            changes.update(module.LTX2_FULL_FINE_TUNING_DEFAULTS)
            changes["block_swap_h2d_only"] = False
            # full fine-tuning trains the dense BF16 weights: base quantization is rejected at Start
            changes.update({key: False for key in module.LTX2_BASE_QUANTIZATION_KEYS if key in values})
        elif self.legacy.module.endswith("modern_image_lora_gui"):
            spec = module.get_architecture(self.legacy.spec_key)
            maximum = spec.max_blocks_to_swap - 1 if mode == FULL and spec.is_ideogram else spec.max_blocks_to_swap
            try:
                blocks = int(values.get("blocks_to_swap") or 0)
            except (TypeError, ValueError):
                blocks = 0
            if blocks > maximum:
                changes["blocks_to_swap"] = maximum
        return {key: value for key, value in changes.items() if key in values}

    def _runtime_defaults(self, mode: str) -> dict:
        """Defaults the original tab computes at build time (architecture recipes, local model files)."""
        module = legacy_module(self.legacy)
        result: dict = {}
        try:
            if self.legacy.module.endswith("image_models_gui"):
                spec = module.get_spec(self.legacy.spec_key)
                result.update(module.architecture_defaults(spec, full_finetune=(mode == FULL)))
                result.update({k: v for k, v in module._default_model_paths(spec).items() if v})
            elif self.legacy.module.endswith("modern_image_lora_gui"):
                spec = module.get_architecture(self.legacy.spec_key)
                result.update(module._architecture_defaults(spec))
                result.update({k: v for k, v in module._default_model_paths(spec).items() if v})
            elif hasattr(module, "_default_model_paths"):
                result.update({k: v for k, v in module._default_model_paths().items() if v})
        except Exception as exc:  # defaults are a convenience; never block the UI
            log.debug(f"{self.name}: runtime defaults unavailable: {exc}")
        known = set(self.keys())
        return {key: _tidy_path(key, resolve_placeholders(value)) for key, value in result.items() if key in known}

    # ---------------------------------------------------------------- load / save
    def load(self, path: str, values: dict) -> LoadResult:
        args = self.ordered_args(values)
        # Wan's "open_configuration" always asks for a file; its dialog-free load is "reload_configuration".
        action = "reload_configuration" if self.legacy.prefix == 4 else "open_configuration"
        with quiet() as messages:
            result = self.actions(action, False, path, True, False, *args)
        try:
            saved = read_preset_file(path)
        except Exception:
            saved = {}
        result = list(result or [])
        loaded = dict(values)
        returned = result[2:]
        for spec, value in zip(self.specs, returned):
            if _is_update(value):
                if "value" not in value:
                    continue
                value = value["value"]
            stored = saved.get(spec.key)
            if isinstance(stored, list) and len(stored) > 1 and _is_path(spec.key, spec) and isinstance(value, str):
                value = stored  # "../my loras/a.safetensors ../b.safetensors" cannot be split back reliably
            loaded[spec.key] = value
        if ADAPTER_KEY in self.spec_map():
            loaded = unfold_adapter(loaded)
        status = result[1] if len(result) > 1 else None
        status_text = status.get("value") if _is_update(status) else (status if isinstance(status, str) else "")
        warnings = [text for level, text in messages if level == "warning"]
        probe = str(status_text or "")
        for name in sorted({path, os.path.abspath(path), os.path.basename(path)}, key=len, reverse=True):
            probe = probe.replace(name, "")  # a file named "failsafe.toml" must not read as a failure
        failed = bool(re.search(r"fail|does not exist|not found|could not|cannot", probe, re.I))
        if failed:
            warnings.append(str(status_text))
        file_keys, matched = _count_matching_keys(path, set(self.keys()))
        return LoadResult(
            values=loaded,
            messages=[text for level, text in messages if level == "info"],
            warnings=warnings,
            matched_keys=matched,
            file_keys=file_keys,
            failed=failed,
        )

    def save(self, path: str, values: dict) -> None:
        """Save through the classic tab's own save action. It writes a temporary file that replaces the
        preset only when it exists: a refused save never leaves an older file looking freshly saved."""
        if "generated_toml_path" in values:
            values = {**values, "generated_toml_path": ""}  # made for one run's folder: never part of a preset
        args = self.ordered_args(values)
        partial = os.path.splitext(path)[0] + ".saving.toml"
        try:
            with quiet() as messages:
                result = self.actions("save_configuration", False, partial, True, False, *args)
            if not os.path.isfile(partial):
                problem = _status_text(result) or next((text for level, text in messages if level == "warning"), "")
                raise RuntimeError(problem or f"The trainer did not write {path}")
            self._keep_lists(partial, dict(zip(self.keys(), args)))
            os.replace(partial, path)
        finally:
            _remove_quietly(partial)

    def _keep_lists(self, path: str, sent: dict) -> None:
        """Wan's save keeps only the first item of a number list (base-weight multipliers); write the
        whole list back so the preset holds every value that was typed."""
        saved = read_preset_file(path)
        fixes = {
            key: value
            for key, value in sent.items()
            if isinstance(value, list) and len(value) > 1 and saved.get(key) in (value[0], [value[0]])
        }
        if fixes:
            saved.update(fixes)
            with open(path, "w", encoding="utf-8") as handle:
                toml.dump(saved, handle)

    # ---------------------------------------------------------------- train
    def print_command(self, values: dict) -> str:
        from .dataset_tools import prepare_dataset

        payload = _local_model_files(_fresh_dataset(self.normalized(values)))
        notes = prepare_dataset(self.name, payload)
        _hand_over_dataset(payload, notes)
        args = self.ordered_args(payload)
        install_toml_type_gate()
        with quiet(), capture_output() as buffer, trainer_scope(self.name):
            self.actions("train_model", False, "", True, True, *args)
        return clean_preview("\n".join(notes) + "\n" + buffer.getvalue())

    def train(self, values: dict, *, title: str, meta: dict, after=None) -> tuple[bool, str]:
        from .runner import RUNNER

        from .dataset_tools import prepare_dataset

        payload = _local_model_files(_fresh_dataset(self.normalized(values)))
        meta = {**meta, "engine": self.engine}
        install_toml_type_gate()

        def target():
            notes = prepare_dataset(self.name, payload)
            for note in notes:
                log.info(note)
            _hand_over_dataset(payload, notes)
            args = self.ordered_args(payload)
            with quiet(), trainer_scope(self.name):  # the job thread writes the runtime TOML
                self.actions("train_model", False, "", True, False, *args)

        return RUNNER.start_job(target, title=title, subtitle=meta.get("subtitle", ""), meta=meta, after=after)


def _status_text(result) -> str:
    """The message a classic action returned in its status box (gr.update(value=...) or plain text)."""
    for item in list(result or [])[1:]:
        text = item.get("value") if _is_update(item) else item if isinstance(item, str) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _count_matching_keys(path: str, known: set[str]) -> tuple[int, int]:
    try:
        data = read_preset_file(path)
    except Exception:
        return 0, 0
    keys = {key for key in data if not key.startswith("preset_")}
    return len(keys), len(keys & known)


def read_preset_file(path: str) -> dict:
    """A preset's settings (TOML or JSON). toml_compat, installed by the package, also reads TOMLs whose
    Windows paths an older toml writer left raw."""
    with open(path, "r", encoding="utf-8-sig") as handle:
        if path.lower().endswith(".json"):
            import json

            return json.load(handle)
        return toml.load(handle)


class KohyaBackend(Backend):
    engine = "kohya"

    def __init__(self, model_key: str):
        self.name = f"kohya:{model_key}"
        self.model_key = model_key
        self._specs: list[ParamSpec] | None = None

    @property
    def module(self):
        from . import kohya_backend

        return kohya_backend

    @property
    def specs(self) -> list[ParamSpec]:
        if self._specs is None:
            specs = list(self.module.kohya_param_specs(self.model_key))
            if any(spec.key == NETWORK_KEY and LORA in spec.modes for spec in specs):
                specs.append(adapter_spec())
            self._specs = specs
        return self._specs

    def _for_trainer(self, values: dict) -> dict:
        """The values sd-scripts and the preset file get: the LoRA / DoRA switch folded into Network Arguments."""
        if ADAPTER_KEY not in self.spec_map():
            return values
        if self.mode_of_value(values.get("training_mode")) == LORA:
            values = fold_adapter(values)
        return {key: value for key, value in values.items() if key != ADAPTER_KEY}

    def trainer_value(self, key: str, value: Any) -> Any:
        """Clean what the user typed by field kind only. kohya_backend builds the sd-scripts config itself
        (its fields do not always map one to one: Resume From Hugging Face holds the repo location), and
        the runtime TOML it writes is tested against sd-scripts' argparse types."""
        spec = self.spec_map().get(key)
        if spec is None:
            return value
        if key == "sample_prompts":
            return self._sample_prompts(value)
        return to_trainer_value(value, label=spec.label or key, kind=spec.kind, arg=None, is_path=_is_path(key, spec), whole=self.whole_number(key))

    @staticmethod
    def _sample_prompts(value: Any) -> str:
        """Sample Prompts takes a prompt file or the prompts themselves: an existing file or a path (drive, UNC, ~, ./,
        file:) gets the path cleaning of every file field, prompt text stays as typed (path cleaning would turn '\\('
        into '/(' and '//' into '/', and expand $NAME)."""
        from .values import clean_path, strip_quotes

        text = "" if value is None else str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text or "\n" in text:
            return text
        path = clean_path(text)
        if os.path.isfile(path) or re.match(r"^(?:[A-Za-z]:[\\/]|\\\\|//|~[\\/]|\.{1,2}[\\/]|file:)", strip_quotes(text), re.I):
            return path
        return text

    def hub_model_id(self, key: str, value) -> bool:
        """sd-scripts takes ``owner/name`` Hugging Face ids for the SDXL base model and the VAE (the presets' VAE is
        stabilityai/sdxl-vae); FLUX.1 needs its files."""
        return self.model_key == "sdxl" and key in ("pretrained_model_name_or_path", "vae") and self.module.looks_like_hf_repo_id(value)

    def whole_number(self, key: str) -> bool:
        """sd-scripts (or accelerate launch) reads this setting as an int (kohya_backend.kohya_whole_number): rank,
        epochs, steps, batch size, block swap ... Float options and the warmup (below 1 a ratio) keep fractions."""
        spec = self.spec_map().get(key)
        return spec is not None and spec.kind in ("number", "slider") and self.module.kohya_whole_number(self.model_key, key)

    def check(self, values: dict, mode: str | None = None) -> list[str]:
        """Unusable input, then everything kohya_backend's own validation would refuse at Start (missing
        model files, the dataset folder layout, setting combinations sd-scripts rejects). A field the input
        check refused is judged at its default there, so one problem gives one message."""
        problems = super().check(values, mode)
        merged = dict(values)
        if mode is not None:
            merged["training_mode"] = self.mode_value(mode)
        named = self._fields_named(problems)
        if named:
            defaults = self.defaults(mode) if mode is not None else {spec.key: spec.default for spec in self.specs}
            merged.update({key: defaults.get(key) for key in named})
        try:
            refused = self.module.kohya_validation_problems(self.model_key, merged)
        except Exception as exc:  # unusable input already listed above; Start reports anything else
            log.debug(f"{self.name}: validation preview failed: {exc}")
            refused = []
        return problems + [problem for problem in refused if problem not in problems]

    def _fields_named(self, problems: list[str]) -> set[str]:
        """Keys of the fields input-check problems are about ('<label> is empty…', '<label>: '…' is not a number.',
        '<label> must be a whole number…'): the longest label a problem starts with."""
        specs = sorted(self.specs, key=lambda spec: len(spec.label or spec.key), reverse=True)
        keys = set()
        for problem in problems:
            for spec in specs:
                label = spec.label or spec.key
                if problem.startswith((f"{label} ", f"{label}:")):
                    keys.add(spec.key)
                    break
        return keys

    def mode_value(self, mode: str) -> str:
        from ..full_finetune_gui import FULL_FINE_TUNING_MODE, LORA_TRAINING_MODE

        return LORA_TRAINING_MODE if mode == LORA else FULL_FINE_TUNING_MODE

    def defaults(self, mode: str) -> dict:
        values = dict(self.module.kohya_defaults(self.model_key, mode))
        values["training_mode"] = self.mode_value(mode)
        return values

    def load(self, path: str, values: dict) -> LoadResult:
        with quiet() as messages:
            loaded, notes = self.module.load_kohya_preset(path, self.model_key, dict(values))
        if ADAPTER_KEY in self.spec_map():
            loaded = unfold_adapter(loaded)
        file_keys, matched = _count_matching_keys(path, set(self.keys()))
        if path.lower().endswith(".json"):
            matched = file_keys  # raw kohya_ss GUI configs use their own key names; conversion reports leftovers
        return LoadResult(
            values=loaded,
            messages=[text for level, text in messages if level == "info"],
            warnings=[text for level, text in messages if level == "warning"] + [note for note in notes if note],
            matched_keys=matched,
            file_keys=file_keys,
        )

    def save(self, path: str, values: dict) -> None:
        values = self._for_trainer(values)
        known = self.spec_map()
        data = {}
        for key, value in sorted(values.items()):
            if key in known and value is not None:
                value = self.trainer_value(key, value)  # cleaned paths, trainer types (raises ValueProblem)
                if value is not None:
                    data[key] = value
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        partial = os.path.splitext(path)[0] + ".saving.toml"
        try:
            with open(partial, "w", encoding="utf-8") as handle:
                toml.dump(data, handle)
            os.replace(partial, path)
        finally:
            _remove_quietly(partial)

    def print_command(self, values: dict) -> str:
        from .runner import RUNNER

        with quiet():
            return str(self.module.train_kohya_model(self.model_key, self._for_trainer(dict(values)), print_only=True, runner=RUNNER) or "")

    def train(self, values: dict, *, title: str, meta: dict, after=None) -> tuple[bool, str]:
        from .runner import RUNNER

        meta = {**meta, "engine": self.engine}
        payload = self._for_trainer(dict(values))

        def target():
            with quiet():
                self.module.train_kohya_model(self.model_key, payload, print_only=False, runner=RUNNER)

        return RUNNER.start_job(target, title=title, subtitle=meta.get("subtitle", ""), meta=meta, after=after)


class _StdoutRouter:
    """sys.stdout while any capture_output() is active: writes from a capturing thread go to that thread's
    buffer, every other thread keeps the real stdout. One router serves overlapping captures (two Print
    Command clicks at once), so no capture can leave sys.stdout pointing at another one's buffer."""

    def __init__(self, original):
        self.original = original
        self.buffers: dict[int, io.StringIO] = {}

    def write(self, text: str) -> int:
        buffer = self.buffers.get(threading.get_ident())
        if buffer is not None:
            return buffer.write(text)
        return self.original.write(text) if self.original is not None else len(text)

    def flush(self) -> None:
        if threading.get_ident() not in self.buffers and self.original is not None:
            self.original.flush()

    def __getattr__(self, name):  # encoding, isatty, fileno, ... of the real stdout
        return getattr(self.original, name)


_capture_lock = threading.Lock()
_capture_state: dict = {"router": None, "users": 0, "consoles": []}


@contextlib.contextmanager
def capture_output():
    """Collect what the calling thread prints or logs (plain text). The GUI's Rich console keeps writing
    to the real terminal, and a training run's output on other threads is never swallowed."""
    buffer = io.StringIO()
    thread_id = threading.get_ident()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(lambda record: record.thread == thread_id)
    root = logging.getLogger()
    targets = [root] + [lg for lg in (log,) if lg is not None and not lg.propagate]
    with _capture_lock:
        state = _capture_state
        if state["users"] == 0:
            state["router"] = _StdoutRouter(sys.stdout)
            sys.stdout = state["router"]
            consoles = [h.console for lg in (log, root) if lg is not None for h in lg.handlers if hasattr(h, "console")]
            state["consoles"] = [(console, getattr(console, "_file", None)) for console in consoles]
            for console, _file in state["consoles"]:
                console._file = state["router"].original  # Rich would otherwise follow sys.stdout into a buffer
        state["users"] += 1
        router = state["router"]
        previous = router.buffers.get(thread_id)
        router.buffers[thread_id] = buffer
    for target in targets:
        target.addHandler(handler)
    try:
        yield buffer
    finally:
        for target in targets:
            target.removeHandler(handler)
        with _capture_lock:
            if previous is None:
                router.buffers.pop(thread_id, None)
            else:
                router.buffers[thread_id] = previous
            state["users"] -= 1
            if state["users"] == 0:
                for console, file in state["consoles"]:
                    console._file = file
                state["consoles"] = []
                if sys.stdout is router:
                    sys.stdout = router.original
                state["router"] = None


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def clean_preview(text: str) -> str:
    """Command preview without colour codes or the copies the trainers both print and log."""
    paragraphs, current = [], []
    for line in ANSI_RE.sub("", text).splitlines():
        if line.strip():
            current.append(line.rstrip())
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    kept: list[list[str]] = []
    for paragraph in paragraphs:
        lines = [line for index, line in enumerate(paragraph) if index == 0 or line != paragraph[index - 1]]
        if kept and lines == kept[-1]:
            continue
        if kept and len(lines) == 1 and lines[0] == kept[-1][-1]:
            continue
        kept.append(lines)
    return "\n\n".join("\n".join(lines) for lines in kept).strip()


@lru_cache(maxsize=None)
def get_backend(name: str) -> Backend:
    if name.startswith("kohya:"):
        return KohyaBackend(name.split(":", 1)[1])
    return LegacyMusubiBackend(name)
