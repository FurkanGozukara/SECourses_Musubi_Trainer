"""The VRAM panel: settings changes that move VRAM, compared with the preset they started from. CPU only."""

from __future__ import annotations

import os
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from musubi_tuner_gui.unified.vram_advice import MORE, LESS, advice_html, advise, preset_vram  # noqa: E402


def _texts(changes, direction):
    return [text for found, _size, text in changes if found == direction]


def test_resolution_frames_batch_and_control_images_raise_vram():
    base = {"dataset_resolution_width": 1024, "dataset_resolution_height": 1024, "dataset_batch_size": 1, "target_frames": "1, 25, 45", "use_control_images": False, "gradient_checkpointing": True, "fp8_base": True}
    now = dict(base, dataset_resolution_width="1328", dataset_resolution_height=1328.0, dataset_batch_size=2, target_frames="1, 25, 81", use_control_images=True, gradient_checkpointing=False, fp8_base="false")
    raised = " | ".join(_texts(advise("wan", base, now), MORE))
    assert "Resolution 1024×1024 → 1328×1328: ≈1.7× image tokens" in raised
    assert "Batch size 1 → 2" in raised and "Frames 45 → 81: ≈1.8× video tokens" in raised
    assert "Control images on" in raised and "Gradient checkpointing off" in raised and "no longer quantized" in raised


def test_less_vram_and_unchanged_settings():
    base = {"blocks_to_swap": 0, "dataset_resolution_width": 1328, "dataset_resolution_height": 1328}
    now = dict(base, blocks_to_swap=20, dataset_resolution_width=1024, dataset_resolution_height=1024)
    changes = advise("krea2", base, now)
    assert not _texts(changes, MORE)
    assert any("20 more blocks wait in CPU RAM" in text for text in _texts(changes, LESS))
    assert advise("krea2", base, dict(base)) == [] and advice_html([], None, None) == ""


def test_edit_versions_and_fps():
    raised = _texts(advise("qwen_image", {"model_version": "original"}, {"model_version": "edit-2511"}), MORE)
    assert raised and "control images join every step" in raised[0]
    fps = advise("ltx2", {"dataset_target_fps": 25}, {"dataset_target_fps": "50"})
    assert fps and fps[0][0] == MORE and "fps" in fps[0][2]


def test_the_panel_warns_when_the_preset_already_filled_the_gpu():
    path = "C:/x/Training_Presets/Krea_2_Training_Configs/LoRA_Training/24_GB_GPUs_22.4_GB_Quality_1.toml"
    assert preset_vram(path) == (24.0, 22.4) and preset_vram("my run.toml") == (None, None)
    changes = advise("krea2", {"dataset_batch_size": 1}, {"dataset_batch_size": 4})
    panel = advice_html(changes, path, ("RTX 4090", 24.0))
    assert "vram-bad" in panel and "will likely run out of memory" in panel and "22.4 GB" in panel
    calm = advice_html(changes, path, ("RTX PRO 6000", 96.0))
    assert "vram-warn" in calm and "73.6 GB to spare" in calm


@pytest.fixture(scope="module")
def image_workspace(tmp_path_factory):
    import gradio as gr

    from musubi_tuner_gui.unified.backends import install_quiet_toasts
    from musubi_tuner_gui.unified.workspace import Workspace

    install_quiet_toasts()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        with gr.Blocks():
            workspace = Workspace("image", headless=True)
            workspace.build()
        yield workspace


class _Request:
    def __init__(self, session):
        self.session_hash = session


def test_the_workspace_panel_follows_edits_and_resets_on_load(image_workspace):
    from musubi_tuner_gui.unified import presets as P

    workspace = image_workspace
    request = _Request("vram")
    workspace.forget(request)
    workspace.apply(request, {}, "krea2", "lora", P.AUTO_TIER, None, False)
    assert workspace.vram_note(request) == ""
    item = workspace.fields[workspace.backend_uid["krea2"]["dataset_resolution_width"]]
    workspace._mirror(request)[item.component._id]["value"] = 2048
    panel = workspace.vram_note(request)
    assert "Resolution" in panel
    workspace.emit(request, {workspace.vram_html: {"value": panel}})  # what the field's event sends
    updates = workspace.apply(request, {}, "krea2", "lora", P.AUTO_TIER, None, False)  # a new load is the new baseline
    assert workspace.vram_note(request) == "" and workspace.vram_html in updates


def test_optimizer_advice_follows_the_size_of_its_state():
    """Adafactor keeps almost no state; AdamW8bit 2 bytes per parameter; AdamW 8."""
    raised = advise("kohya:flux1_dev", {"optimizer_type": "Adafactor"}, {"optimizer_type": "AdamW8bit"}, "full")
    assert raised and raised[0][0] == MORE and raised[0][1] == "big"
    lowered = advise("krea2", {"optimizer_type": "AdamW"}, {"optimizer_type": "AdamW8bit"}, "lora")
    assert lowered and lowered[0][0] == LESS and lowered[0][1] == "small"
    assert advise("krea2", {"optimizer_type": "AdamW"}, {"optimizer_type": "automagic"}) == []  # unknown: no guess


def test_swapping_one_quantization_for_another_is_not_an_alarm():
    base = {"fp8_base": True, "convrot_int8": False}
    assert advise("krea2", base, {"fp8_base": False, "convrot_int8": True}) == []
    assert advise("krea2", base, {"fp8_base": False, "convrot_int8": False})[0][0] == MORE


def test_only_trained_frames_count():
    """Wan's num_frames is only for samples; max_frames matters only with full extraction."""
    assert advise("wan", {"num_frames": 81}, {"num_frames": 121}) == []
    head = {"max_frames": 129, "frame_extraction": "head"}
    assert advise("wan", head, dict(head, max_frames=257)) == []
    full = {"max_frames": 129, "frame_extraction": "full"}
    assert advise("wan", full, dict(full, max_frames=257))[0][0] == MORE


def test_small_changes_never_raise_the_out_of_memory_alarm():
    path = "C:/x/Krea_2_Training_Configs/LoRA_Training/24_GB_GPUs_22.4_GB_Quality_1.toml"
    changes = advise("krea2", {"img_in_txt_in_offloading": True}, {"img_in_txt_in_offloading": False})
    panel = advice_html(changes, path, ("RTX 4090", 24.0))
    assert "run out of memory" not in panel and "vram-info" in panel


def test_panel_wording():
    changes = advise("krea2", {"dataset_resolution_width": 1024, "dataset_resolution_height": 1024}, {"dataset_resolution_width": 1000, "dataset_resolution_height": 1000})
    assert "−5%" in changes[0][2]
    assert "compared with the defaults" in advice_html(changes, None, ("GPU", 24.0))
    big = "C:/x/Krea_2_Training_Configs/LoRA_Training/48_GB_GPUs_40_GB_Quality_1.toml"
    raised = advise("krea2", {"dataset_batch_size": 1}, {"dataset_batch_size": 2})
    panel = advice_html(raised, big, ("RTX 4090", 24.0))
    assert "made for 48 GB GPUs" in panel and "left)" not in panel
    lowered = advise("krea2", {"dataset_batch_size": 4}, {"dataset_batch_size": 1})
    assert "4.0× less activation memory" in lowered[0][2]
