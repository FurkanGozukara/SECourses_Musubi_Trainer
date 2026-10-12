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
    ],
)
def test_name_hints(name, expected):
    assert P._name_hint(os.path.join("C:/anything", name)) == expected
