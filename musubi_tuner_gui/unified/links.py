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


QWEN21_DIT_FILES = ("qwen_image_2.1_bf16.safetensors", "qwen_image_2.1_turbo_bf16.safetensors")
QWEN21_TURBO_ADAPTER = "qwen_image_2_turbo_training_adapter_v1.safetensors"
KREA2_DIT_FILES = ("Krea_2_Raw_Base.safetensors", "Krea_2_Turbo_BF16.safetensors")
KREA2_TURBO_ADAPTER = "krea2_turbo_training_adapter_v1.safetensors"
# Turbo samples: 8 steps without CFG (Qwen/Qwen-Image-2.1-Turbo and krea/Krea-2-Turbo, ComfyUI's Turbo templates, ai-toolkit)
TURBO_SAMPLING = {"sample_steps": 8, "sample_cfg_scale": 1.0}


def _turbo_variant(values: dict, model_dir: str, dit_files: tuple[str, str], adapter_file: str, base_sampling: dict) -> dict:
    """The DiT variant switch of Qwen-Image 2.1 / Krea 2: the variant's own checkpoint (a missing file clears the path),
    the Turbo training adapter with Turbo (cleared otherwise: merged into the base model it would train the wrong
    weights), and Turbo's 8 steps without CFG; back to the base model, sample settings still at Turbo's return to the
    base model's defaults (a user's own settings stay)."""
    turbo = str(values.get("dit_variant") or "").strip().lower() == "turbo"
    changes: dict = {}
    dit = _variant_file(values.get("dit"), model_dir, dit_files, int(turbo))
    if dit is not None:
        changes["dit"] = dit
    if turbo:
        adapter = str(values.get("training_adapter_path") or "").strip()
        if not adapter or not os.path.isfile(adapter):
            target = os.path.join(model_dir, adapter_file)
            changes["training_adapter_path"] = target if os.path.isfile(target) else ""
        changes.update(TURBO_SAMPLING)
    else:
        changes["training_adapter_path"] = ""
        try:
            at_turbo = all(float(values.get(key)) == float(value) for key, value in TURBO_SAMPLING.items())
        except (TypeError, ValueError):
            at_turbo = False
        if at_turbo:
            changes.update(base_sampling)
    return changes


def _qwen21_variant(values: dict) -> dict:
    """image_models_gui Qwen-Image 2.1 DiT Variant (base / turbo)."""
    return _turbo_variant(values, _model_dir("qwen_image_21"), QWEN21_DIT_FILES, QWEN21_TURBO_ADAPTER, {"sample_steps": 30, "sample_cfg_scale": 4.0})


def _modern_model_dir(spec_key: str) -> str:
    """Where the Ideogram 4 / Krea 2 tab looks for its default files (``modern_image_lora_gui`` architectures)."""
    from .. import modern_image_lora_gui as module

    spec = module.get_architecture(spec_key)
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(module.__file__)), "..", spec.model_folder))


def _krea2_variant(values: dict) -> dict:
    """modern_image_lora_gui Krea 2 DiT Variant (raw / turbo)."""
    return _turbo_variant(values, _modern_model_dir("krea2"), KREA2_DIT_FILES, KREA2_TURBO_ADAPTER, {"sample_steps": 28, "sample_cfg_scale": 5.5})


ZIMAGE_DIT_FILES = ("Z_Image_BF16.safetensors", "Z_Image_Turbo_BF16.safetensors")
ZIMAGE_SHARED_FILES = {"vae": "Z_Image_Train_VAE.safetensors", "text_encoder": "Z_Image_Training_Text_Encoder.safetensors"}


def _zimage_model_dir() -> str:
    """Training_Models_Z_Image in the install folder, where both Z-Image bundles of Download_Train_Models.py go."""
    from .schema import ROOT_DIR

    return os.path.join(ROOT_DIR, "Training_Models_Z_Image")


def _zimage_variant(values: dict) -> dict:
    """Z-Image Base / Turbo: the variant's own DiT, and the bundles' VAE and text encoder where a path is empty. The
    defaults leave all three empty, so the Turbo Fine-Tuning card, which has no preset, started without model files
    although they were downloaded. The training adapter is a preset's choice and stays as it is."""
    model_dir = _zimage_model_dir()
    turbo = str(values.get("zimage_variant") or "").strip().lower() == "turbo"
    changes: dict = {}
    dit = _variant_file(values.get("dit"), model_dir, ZIMAGE_DIT_FILES, int(turbo))
    if dit is not None:
        changes["dit"] = dit
    for key, name in ZIMAGE_SHARED_FILES.items():
        if not str(values.get(key) or "").strip():
            target = os.path.join(model_dir, name)
            if os.path.isfile(target):
                changes[key] = target
    return changes


# Download_Train_Models.py bundles 1-4 (Training_Models_Qwen): "original" trains Qwen-Image and 2512 alike
QWEN_DIT_FILES = {
    "original": ("qwen_image_bf16.safetensors", "Qwen_Image_2512_BF16.safetensors"),
    "edit-2509": ("Qwen_Image_Edit_Plus_2509_bf16.safetensors",),
    "edit-2511": ("Qwen_Image_Edit_2511_BF16.safetensors",),
}


def _qwen_model_dir() -> str:
    from .schema import ROOT_DIR

    return os.path.join(ROOT_DIR, "Training_Models_Qwen")


def _qwen_model_version(values: dict) -> dict:
    """Qwen-Image Model Version: Edit 2509 / 2511 train their own checkpoints. The classic tab left the DiT on the
    text-to-image file, so picking an edit version trained the edit LoRA on the wrong model. Another version's file
    (by its name) becomes this version's, next to it or in Training_Models_Qwen; a file the user picked stays, and a
    missing target clears the path so the model file status and Start name it."""
    from ..qwen_image_lora_gui import normalize_qwen_image_model_version

    wanted = QWEN_DIT_FILES.get(normalize_qwen_image_model_version(values.get("model_version")))
    if not wanted:
        return {}
    known = {name.lower() for names in QWEN_DIT_FILES.values() for name in names}
    text = str(values.get("dit") or "").strip()
    name = os.path.basename(text).lower() if text else ""
    if text and (name not in known or name in {item.lower() for item in wanted}):
        return {}
    for folder in ([os.path.dirname(text)] if text else []) + [_qwen_model_dir()]:
        for item in wanted:
            target = os.path.normpath(os.path.join(folder, item))
            if os.path.exists(target):
                return {"dit": target}
    return {"dit": ""}


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


def _ltx_version_files(values: dict) -> dict:
    """LTX 2.5 is its own split pack (transformer, Gemma 4 text encoder, video VAE, audio VAE in Training_Models_LTX_2_5)
    and samples the trained dev transformer with its full schedule; LTX 2.3 / 2.0 use the single-file checkpoint. A path
    that names the other version's file switches to this version's file, or is cleared when that file is missing."""
    from .. import ltx2_lora_gui as module

    version = str(values.get("ltx_version") or "2.3").strip()
    v23_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(module.__file__)), "..", module.LTX2_MODEL_FOLDER))
    v25_dir = module.ltx25_model_dir()
    checkpoints = (module.LTX2_CHECKPOINT_FILENAME, module.LTX25_TRANSFORMER_FILENAME, module.LTX25_PT_TRANSFORMER_FILENAME)
    changes: dict = {}
    if version == "2.5":
        # either LTX 2.5 transformer (Dev or Pre-Trained) stays: only the Transformer Variant switches between them
        wanted = module.ltx25_transformer_filename(values.get("ltx25_variant"))
        target = _variant_file(values.get("ltx2_checkpoint"), v25_dir, (module.LTX2_CHECKPOINT_FILENAME, wanted), 1)
        if target is not None:
            changes["ltx2_checkpoint"] = target
        for key, name in module.ltx25_default_files().items():
            if key != "ltx2_checkpoint" and not str(values.get(key) or "").strip():
                path = os.path.join(v25_dir, name)
                if os.path.isfile(path):
                    changes[key] = path
        if str(values.get("sample_sampling_preset") or "defaults").strip() in ("", "defaults"):
            changes["sample_sampling_preset"] = "ltx25_full"
        # the DiffVAE decodes with a wider receptive field than the conv VAE (fork docs/ltx_2.md)
        for key, minimum in (("sample_vae_tile_overlap", 160), ("sample_vae_temporal_tile_overlap", 40)):
            try:
                current = int(parse_number(values.get(key), whole=True) or 0)
            except (ValueProblem, TypeError, ValueError):
                current = 0
            if current < minimum:
                changes[key] = minimum
    else:
        target = _variant_file(values.get("ltx2_checkpoint"), v23_dir, checkpoints, 0)
        if target is not None and str(values.get("ltx2_checkpoint") or "").strip():
            changes["ltx2_checkpoint"] = target
    return changes


def _ltx25_variant_file(values: dict) -> dict:
    """ltx2_lora_gui's Transformer Variant switch (Dev / Pre-Trained LTX 2.5 transformer), for LTX 2.5 only."""
    from .. import ltx2_lora_gui as module

    if str(values.get("ltx_version") or "").strip() != "2.5":
        return {}
    target = module.switch_ltx25_variant_checkpoint(values.get("ltx25_variant"), values.get("ltx2_checkpoint"))
    return {} if target is None else {"ltx2_checkpoint": target}


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
    Link("qwen_image_21", "dit_variant", ("dit", "training_adapter_path", "sample_steps", "sample_cfg_scale"), _qwen21_variant, on_load=True),
    Link("krea2", "dit_variant", ("dit", "training_adapter_path", "sample_steps", "sample_cfg_scale"), _krea2_variant, on_load=True),
    Link("zimage", "zimage_variant", ("dit", "vae", "text_encoder"), _zimage_variant, on_load=True),
    Link("qwen_image", "model_version", ("dit",), _qwen_model_version),
    Link("wan", "fp8_base", ("fp8_scaled",), _wan_fp8),
    Link("wan", "task", ("compile_resident_blocks_only",), _wan_compile_residency, on_load=True, keep_user=("compile_resident_blocks_only",)),
    Link("flux", "model_version", ("sample_steps", "sample_guidance_scale"), _klein_sampling),
    Link("kandinsky6", "num_frames", ("num_frames",), _rounded("num_frames", "round_frames_to_4n_plus_1")),
    Link("ltx2", "dataset_num_frames", ("dataset_num_frames",), _rounded("dataset_num_frames", "round_frames_to_ltx2")),
    Link("ltx2", "ltx_version", ("ltx2_checkpoint", "ltx2_text_encoder_checkpoint", "ltx2_video_vae", "ltx2_audio_vae", "sample_sampling_preset",
                                 "sample_vae_tile_overlap", "sample_vae_temporal_tile_overlap"), _ltx_version_files, on_load=True),
    Link("ltx2", "ltx25_variant", ("ltx2_checkpoint",), _ltx25_variant_file),
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
