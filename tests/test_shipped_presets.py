"""Every shipped GPU-tier preset must be self-describing, keep 1.5 GB of VRAM free and load in its tab."""

from __future__ import annotations

import os
import re

import pytest
import toml

from musubi_tuner_gui import image_models_gui as image_models
from musubi_tuner_gui import ltx2_lora_gui as ltx2
from musubi_tuner_gui.full_finetune_gui import FULL_FINE_TUNING_MODE, LORA_TRAINING_MODE, normalize_training_mode
from musubi_tuner_gui.preset_library import TIER_PATTERN, preset_root

# folder -> (tab kind, spec key)
FOLDERS = {
    "Qwen_Image_2.1_Training_Configs": ("image_models", "qwen_image_21"),
    "ERNIE_Image_Training_Configs": ("image_models", "ernie_image"),
    "Mage_Flow_Training_Configs": ("image_models", "mage_flow"),
    "Mage_Flow_Edit_Training_Configs": ("image_models", "mage_flow"),
    "FLUX_3_Klein_Training_Configs": ("image_models", "flux3_klein"),
    "MOSS_Image_Training_Configs": ("image_models", "moss_image"),
    "HiDream_O1_Training_Configs": ("image_models", "hidream_o1"),
    "Kandinsky_6_Training_Configs": ("image_models", "kandinsky6"),
    "LTX_2.3_Training_Configs": ("ltx2", None),
    "LTX_2.5_Training_Configs": ("ltx2", None),
    "FLUX_2_Training_Configs": ("other", None),
    "FLUX_Klein_9B_Training_Configs": ("other", None),
    "FLUX_Klein_4B_Training_Configs": ("other", None),
    "Z_Image_Base_Training_Configs": ("other", None),
    "Z_Image_Turbo_Training_Configs": ("other", None),
    "Krea_2_Training_Configs": ("other", None),
    "Ideogram_4_Training_Configs": ("other", None),
    "Wan_21_Training_Configs": ("other", None),
    "MiniMax_H3_Training_Configs": ("other", None),
}


def _presets():
    root = preset_root()
    found = []
    for folder, (kind, spec_key) in FOLDERS.items():
        base = os.path.join(root, folder)
        if not os.path.isdir(base):
            continue
        for current, _dirs, files in os.walk(base):
            for name in files:
                if name.endswith(".toml") and TIER_PATTERN.search(name):
                    found.append(pytest.param(os.path.join(current, name), kind, spec_key, id=f"{folder}/{name}"))
    return found


PRESETS = _presets()


@pytest.mark.skipif(not PRESETS, reason="no shipped tier presets next to the app")
@pytest.mark.parametrize("path,kind,spec_key", PRESETS)
def test_shipped_preset(path, kind, spec_key):
    name = os.path.basename(path)
    match = TIER_PATTERN.search(name)
    tier, peak = float(match.group("tier")), float(match.group("used"))
    assert tier - peak >= 1.5 - 1e-6, f"{name}: measured peak {peak} GB leaves less than 1.5 GB of a {tier:g} GB GPU"

    with open(path, "r", encoding="utf-8") as handle:
        first_line = handle.readline()
    assert first_line.startswith("# description:"), f"{name}: missing '# description:' header"
    data = toml.loads(open(path, "r", encoding="utf-8").read())

    folder_mode = FULL_FINE_TUNING_MODE if "Fine_Tuning" in path.replace("\\", "/") else LORA_TRAINING_MODE
    if "training_mode" in data:
        assert normalize_training_mode(data["training_mode"]) == folder_mode, f"{name}: training_mode does not match its folder"
    assert not data.get("parent_folder_path"), f"{name}: ships a machine-specific dataset folder"
    assert not str(data.get("dataset_config") or "").strip(), f"{name}: ships a machine-specific dataset TOML"

    if kind == "image_models":
        spec = image_models.get_spec(spec_key)
        keys = image_models.param_keys(spec)
        defaults = image_models.architecture_defaults(spec)
        parameters = [(key, data.get(key, defaults.get(key))) for key in keys]
        image_models.normalize_parameters(spec, parameters, validate_paths=False)
    elif kind == "ltx2":
        parameters = [(key, data.get(key, ltx2.LTX2_DEFAULTS[key])) for key in ltx2.LTX2_PARAM_KEYS]
        ltx2._normalize_ltx2_parameters(parameters, validate_paths=False)
    else:
        _load_through_tab(path)


def _load_through_tab(path):
    """The owning tab's Open handler must accept the file (smoke test of the real load path)."""
    folder = os.path.relpath(path, preset_root()).replace("\\", "/").split("/")[0]
    if folder.startswith(("FLUX_2_", "FLUX_Klein_")):
        from musubi_tuner_gui import flux_lora_gui as tab

        result = tab.open_flux_configuration(False, path, [(key, None) for key in tab.FLUX_PARAM_KEYS])
    elif folder.startswith("Z_Image_"):
        from musubi_tuner_gui import zimage_lora_gui as tab

        result = tab.open_zimage_configuration(False, path, [(key, None) for key in tab.ZIMAGE_PARAM_KEYS])
    elif folder.startswith(("Krea_2_", "Ideogram_4_")):
        from musubi_tuner_gui import modern_image_lora_gui as tab

        spec_key = "krea2" if folder.startswith("Krea_2_") else "ideogram4"
        result = tab.open_modern_configuration(spec_key, False, path, [(key, None) for key in tab.MODERN_IMAGE_PARAM_KEYS])
    elif folder.startswith("MiniMax_H3_"):
        from musubi_tuner_gui import minimax_h3_lora_gui as tab

        result = tab.open_minimax_h3_configuration(False, path, [(key, None) for key in tab.MINIMAX_H3_PARAM_KEYS])
    else:
        return  # Wan builds its parameter list inside the tab; covered by the generic checks above
    assert isinstance(result, tuple) and result, f"{path}: the tab returned nothing"


def _all_preset_files():
    root = preset_root()
    if not os.path.isdir(root):
        return []
    found = []
    for folder in sorted(os.listdir(root)):
        if folder.endswith("_Training_Configs") and os.path.isdir(os.path.join(root, folder)):
            for current, _dirs, files in os.walk(os.path.join(root, folder)):
                found.extend(os.path.join(current, name) for name in sorted(files) if name.endswith(".toml"))
    return found


ALL_PRESET_FILES = _all_preset_files()


@pytest.mark.skipif(not ALL_PRESET_FILES, reason="no shipped presets next to the app")
def test_no_shipped_preset_names_a_machine_specific_model_path():
    """Model files are named Training_Models_*/... (relative to the install folder), never by the measuring machine's
    path: G:/SECourses_Musubi_Trainer_v29/Training_Models_X/... showed that folder on every PC, and after ⬇️ Download the
    fields still named it (2026-10-25; dont_upload/vram_lab/preset_paths.py rewrites them)."""
    from musubi_tuner_gui.common_gui import PORTABLE_MODEL_PATH_KEYS
    from musubi_tuner_gui.unified.kohya_backend import MODEL_PATH_KEYS

    keys = set(PORTABLE_MODEL_PATH_KEYS) | set(MODEL_PATH_KEYS)
    bad = []
    for path in ALL_PRESET_FILES:
        data = toml.loads(open(path, "r", encoding="utf-8-sig").read())
        for key in sorted(keys & data.keys()):
            for value in data[key] if isinstance(data[key], list) else [data[key]]:
                text = str(value or "").strip()
                if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith(("/", "\\", "~")):
                    bad.append(f"{os.path.relpath(path, preset_root())}: {key} = {text}")
    assert not bad, f"{len(bad)} machine-specific model paths, e.g. " + "; ".join(bad[:5])


def test_tier_pattern_matches_shipped_naming():
    assert TIER_PATTERN.search("24_GB_GPUs_21.3_GB_Quality_1.toml")
    assert re.search(r"Quality_\d", "8_GB_GPUs_6.4_GB_Quality_3.toml")
