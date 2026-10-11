"""Settings the classic tabs change when the user changes another one (value coupling).

The unified UI reuses the classic modules' load / save / train code but not their Gradio events, so the
couplings those events implement live here. The workspace runs a link when its trigger field changes, and
at load time when the model entry fixes the trigger (Mage-Flow Edit's ``is_edit``) or, without a preset,
sets it (Wan 2.2's ``task``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .values import ValueProblem, parse_number


@dataclass(frozen=True)
class Link:
    backend: str
    trigger: str
    outputs: tuple[str, ...]  # every key ``apply`` may change
    apply: Callable[[dict], dict]  # the current values (with the trigger's new value) -> {key: new value}
    on_load: bool = False  # also when the entry fixes the trigger, or (no preset) sets it by default
    keep_user: tuple[str, ...] = ()  # outputs the user set themselves stay as they are


def _true(value) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in ("1", "true", "yes", "on")
    return bool(value)


def _model_dir(spec_key: str) -> str:
    """Where the classic tab looks for this model's default files (``image_models_gui`` specs)."""
    from .. import image_models_gui as module

    spec = module.get_spec(spec_key)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(module.__file__)), "..", spec.model_folder))


def _variant_file(current, model_dir: str, names: tuple[str, ...], wanted: int) -> str | None:
    """The classic model-file switch between a model's variants. An empty path or another variant's file
    (by its file name: a preset can point to another install's model folder) becomes this variant's file,
    next to it or in the model folder; a file the user picked, or this variant's already, stays (None).
    A missing target clears the path, so the model file status and Start name it, instead of training
    the other variant's checkpoint."""
    text = str(current or "").strip()
    lowered = [name.lower() for name in names]
    name = os.path.basename(text).lower() if text else ""
    if text and (name not in lowered or name == lowered[wanted]):
        return None
    for folder in ([os.path.dirname(text)] if text else []) + [model_dir]:
        target = os.path.normpath(os.path.join(folder, names[wanted]))
        if os.path.exists(target):
            return target
    return ""


HIDREAM_FILES = ("hidream_o1_image_bf16.safetensors", "hidream_o1_image_dev_bf16.safetensors")
MAGE_FLOW_FILES = ("mage_flow_base_bf16.safetensors", "mage_flow_edit_base_bf16.safetensors")


def _hidream_model_type(values: dict) -> dict:
    """image_models_gui switch_hidream_type: the dev checkpoint trains with its own noise schedule (and
    samples without CFG)."""
    dev = str(values.get("model_type") or "") == "dev"
    start, end, clip = (7.5, 7.5, 2.5) if dev else (8.0, 8.0, 0.0)
    changes = {"noise_scale_start": start, "noise_scale_end": end, "noise_clip_std": clip, "sample_cfg_scale": 0.0 if dev else 5.0}
    dit = _variant_file(values.get("dit"), _model_dir("hidream_o1"), HIDREAM_FILES, int(dev))
    if dit is not None:
        changes["dit"] = dit
    return changes


def _hidream_task(values: dict) -> dict:
    """i2i trains with reference images (the trainer refuses i2i without them)."""
    return {"use_control_images": str(values.get("task") or "") == "i2i"}


def _mage_flow_dit(values: dict) -> dict:
    """image_models_gui switch_mage_variant: Mage-Flow Edit trains its own checkpoint."""
    dit = _variant_file(values.get("dit"), _model_dir("mage_flow"), MAGE_FLOW_FILES, int(_true(values.get("is_edit"))))
    return {} if dit is None else {"dit": dit}


def _wan_fp8(values: dict) -> dict:
    """wan_lora_gui: FP8 Scaled follows FP8 Base (scaled needs the FP8 base; a plain FP8 cast loses quality)."""
    return {"fp8_scaled": _true(values.get("fp8_base"))}


def _wan_compile_residency(values: dict) -> dict:
    from ..wan_lora_gui import wan_compile_resident_blocks_default

    return {"compile_resident_blocks_only": bool(wan_compile_resident_blocks_default(values.get("task")))}


def _klein_sampling(values: dict) -> dict:
    """flux_lora_gui update_klein_sampling_defaults: distilled Klein samples in 4 steps without guidance."""
    distilled = str(values.get("model_version") or "").strip().lower() in {"klein-4b", "klein-9b"}
    return {"sample_steps": 4 if distilled else 50, "sample_guidance_scale": 1.0 if distilled else 4.0}


def _rounded(key: str, rounder_name: str) -> Callable[[dict], dict]:
    """A frame count rounded down to the grid the model's VAE needs, as the classic tab does on blur."""

    def apply(values: dict) -> dict:
        try:
            number = parse_number(values.get(key), whole=True)
        except (ValueProblem, TypeError, ValueError):
            return {}
        if number is None:
            return {}  # blank: Start names it
        from .. import dataset_config_generator as generator

        rounded = getattr(generator, rounder_name)(int(number))
        return {} if rounded == number else {key: rounded}

    return apply


LINKS: tuple[Link, ...] = (
    Link("hidream_o1", "model_type", ("dit", "noise_scale_start", "noise_scale_end", "noise_clip_std", "sample_cfg_scale"), _hidream_model_type),
    Link("hidream_o1", "task", ("use_control_images",), _hidream_task),
    Link("mage_flow", "is_edit", ("dit",), _mage_flow_dit, on_load=True),
    Link("wan", "fp8_base", ("fp8_scaled",), _wan_fp8),
    Link("wan", "task", ("compile_resident_blocks_only",), _wan_compile_residency, on_load=True, keep_user=("compile_resident_blocks_only",)),
    Link("flux", "model_version", ("sample_steps", "sample_guidance_scale"), _klein_sampling),
    Link("kandinsky6", "num_frames", ("num_frames",), _rounded("num_frames", "round_frames_to_4n_plus_1")),
    Link("ltx2", "dataset_num_frames", ("dataset_num_frames",), _rounded("dataset_num_frames", "round_frames_to_ltx2")),
    Link("minimax_h3", "dataset_num_frames", ("dataset_num_frames",), _rounded("dataset_num_frames", "round_frames_to_minimax_h3")),
    Link("minimax_h3", "sample_num_frames", ("sample_num_frames",), _rounded("sample_num_frames", "round_frames_to_minimax_h3")),
)


def follow(links, values: dict, user_set: Callable[[str], bool] = lambda key: False) -> dict:
    """What ``links`` change in ``values`` (only keys the values hold, never one the user set where the link
    keeps the user's choice)."""
    changes: dict = {}
    for link in links:
        for key, value in link.apply({**values, **changes}).items():
            if key in values and key in link.outputs and not (key in link.keep_user and user_set(key)):
                changes[key] = value
    return changes


def on_load(backend: str, fixed: dict, defaults: dict, values: dict, preset_loaded: bool) -> dict:
    """Links whose trigger the model entry decides: fixed (always) or set by default (without a preset)."""
    links = [
        link
        for link in LINKS
        if link.backend == backend and link.on_load and (link.trigger in fixed or (not preset_loaded and link.trigger in defaults))
    ]
    return follow(links, values)
