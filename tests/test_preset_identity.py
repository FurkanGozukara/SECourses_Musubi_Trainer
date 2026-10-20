"""Older presets without preset_* metadata: trainers that share every setting (Krea 2 / Ideogram 4) are told apart by
the file or folder name (presets._name_hint), never against the file's own content."""

from __future__ import annotations

import os
import sys

import pytest
import toml

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from musubi_tuner_gui.unified import presets as P  # noqa: E402
from musubi_tuner_gui.unified.schema import load_schema  # noqa: E402


@pytest.fixture()
def shared_settings():
    backends = load_schema()["backends"]
    keys = sorted(set(backends["krea2"]["keys"]) & set(backends["ideogram4"]["keys"]) - {"network_module", "training_mode"})
    return {key: backends["krea2"]["params"][key].get("value") for key in keys if backends["krea2"]["params"][key].get("value") is not None}


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Krea_2_Fine_Tuning_Demo.toml", "krea2"),
        ("Ideogram_4_Fine_Tuning_Demo.toml", "ideogram4"),
        ("my_settings.toml", None),
    ],
)
def test_a_tie_between_identical_trainers_follows_the_name(tmp_path, shared_settings, name, expected):
    path = tmp_path / "Demo_Training_Configs" / name
    path.parent.mkdir()
    with open(path, "w", encoding="utf-8") as handle:
        toml.dump(shared_settings, handle)
    identity = P.identify_preset(str(path))
    assert identity.entry_key == expected


def test_content_wins_over_the_name(tmp_path):
    path = tmp_path / "Krea_2_copy.toml"
    with open(path, "w", encoding="utf-8") as handle:
        toml.dump({"network_module": "networks.lora_ideogram4", "learning_rate": 1e-4}, handle)
    assert P.identify_preset(str(path)).entry_key == "ideogram4"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("FLUX_Klein_4B_LoRA_Demo.toml", "flux2_klein_4b"),
        ("FLUX_2_LoRA_Demo.toml", "flux2_dev"),
        ("Qwen_Image_2.1_x.toml", "qwen_image_21"),
        ("Z_Image_Turbo_LoRA_Demo.toml", "zimage_turbo"),
        ("Wan_2.2_t2v.toml", "wan22"),
        ("MiniMax_H3_LoRA.toml", "minimax_h3"),
        ("Mage_Flow_Edit_x.toml", "mage_flow_edit"),
        ("FLUX_3_Klein_Edit_x.toml", "flux3_klein_edit"),
        ("FLUX_3_Klein_x.toml", "flux3_klein"),
    ],
)
def test_name_hints(name, expected):
    assert P._name_hint(os.path.join("C:/anything", name)) == expected


@pytest.mark.parametrize("controls, expected", [(True, "flux3_klein_edit"), (False, "flux3_klein")])
def test_flux3_reference_images_pick_the_edit_card(tmp_path, controls, expected):
    """FLUX.3 Klein editing has its own card (presets measured with a reference per image): a FLUX.3 preset with control
    images on is the Edit card's, whatever its file is called."""
    path = tmp_path / "my_run.toml"
    with open(path, "w", encoding="utf-8") as handle:
        toml.dump({"network_module": "networks.lora_flux_3", "use_control_images": controls, "learning_rate": 6e-5}, handle)
    assert P.identify_preset(str(path)).entry_key == expected


def test_the_shipped_flux3_edit_presets_belong_to_the_edit_card():
    folder = os.path.join(os.path.dirname(APP_DIR), "Training_Presets", "FLUX_3_Klein_Edit_Training_Configs", "LoRA_Training")
    if not os.path.isdir(folder):
        pytest.skip("no Training_Presets in this checkout")
    names = sorted(name for name in os.listdir(folder) if name.endswith(".toml"))
    assert names and all(P.identify_preset(os.path.join(folder, name)).entry_key == "flux3_klein_edit" for name in names)
