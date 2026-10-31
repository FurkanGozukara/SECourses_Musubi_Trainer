"""Settings the classic tabs change together with another one (unified/links.py). CPU only."""

from __future__ import annotations

import os
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from musubi_tuner_gui.unified import links as L  # noqa: E402


class _Request:
    def __init__(self, session):
        self.session_hash = session


@pytest.fixture(scope="module")
def workspaces(tmp_path_factory):
    import gradio as gr

    from musubi_tuner_gui.unified.backends import install_quiet_toasts
    from musubi_tuner_gui.unified.workspace import Workspace

    install_quiet_toasts()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        built = {}
        with gr.Blocks():
            for category in ("image", "video"):
                built[category] = Workspace(category, headless=True)
                built[category].build()
        yield built


def _links(backend, trigger):
    return [link for link in L.LINKS if link.backend == backend and link.trigger == trigger]


def test_every_link_names_settings_its_trainer_has():
    from musubi_tuner_gui.unified.backends import get_backend

    for link in L.LINKS:
        keys = set(get_backend(link.backend).keys())
        assert link.trigger in keys and set(link.outputs) <= keys, link


def test_hidream_dev_trains_its_own_checkpoint_and_noise(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_model_dir", lambda _key: str(tmp_path))
    full, dev = (tmp_path / name for name in L.HIDREAM_FILES)
    full.write_bytes(b"")
    values = {"model_type": "dev", "dit": str(full), "noise_scale_start": 8.0, "noise_scale_end": 8.0, "noise_clip_std": 0.0, "sample_cfg_scale": 5.0}
    # the dev file is not there: the full checkpoint must not stay behind with the dev settings
    assert L.follow(_links("hidream_o1", "model_type"), values) == {
        "noise_scale_start": 7.5, "noise_scale_end": 7.5, "noise_clip_std": 2.5, "sample_cfg_scale": 0.0, "dit": ""
    }
    dev.write_bytes(b"")
    assert L.follow(_links("hidream_o1", "model_type"), values)["dit"] == str(dev)
    back = L.follow(_links("hidream_o1", "model_type"), dict(values, model_type="full", dit=str(dev).replace("\\", "/")))
    assert back["dit"] == str(full) and back["noise_scale_start"] == 8.0 and back["sample_cfg_scale"] == 5.0
    own = dict(values, dit="D:/my models/hidream_finetune.safetensors")
    assert "dit" not in L.follow(_links("hidream_o1", "model_type"), own)  # a file the user picked stays
    other = tmp_path / "other install"  # a preset pointing to another install's model folder
    other.mkdir()
    for name in L.HIDREAM_FILES:
        (other / name).write_bytes(b"")
    switched = L.follow(_links("hidream_o1", "model_type"), dict(values, dit=str(other / L.HIDREAM_FILES[0])))
    assert switched["dit"] == str(other / L.HIDREAM_FILES[1])
    assert L.follow(_links("hidream_o1", "task"), {"task": "i2i", "use_control_images": False}) == {"use_control_images": True}


def test_mage_flow_edit_loads_the_edit_checkpoint(workspaces, tmp_path, monkeypatch):
    from musubi_tuner_gui.unified.registry import get_entry

    from musubi_tuner_gui import image_models_gui

    monkeypatch.setattr(L, "_model_dir", lambda _key: str(tmp_path))
    monkeypatch.setattr(image_models_gui, "_default_model_paths", lambda _spec: {})  # independent of downloaded models
    image = workspaces["image"]
    assert image.load_values(get_entry("mage_flow_edit"), "lora", None).values["dit"] == ""  # not downloaded: no wrong file
    for name in L.MAGE_FLOW_FILES:
        (tmp_path / name).write_bytes(b"")
    assert image.load_values(get_entry("mage_flow_edit"), "lora", None).values["dit"] == str(tmp_path / L.MAGE_FLOW_FILES[1])
    assert image.load_values(get_entry("mage_flow"), "lora", None).values["dit"] == str(tmp_path / L.MAGE_FLOW_FILES[0])


def test_zimage_cards_without_a_preset_find_the_downloaded_model_files(workspaces, tmp_path, monkeypatch):
    """Z-Image Turbo Fine-Tuning has no preset: its defaults left DiT, VAE and text encoder empty (V39)."""
    from musubi_tuner_gui.unified.registry import get_entry

    monkeypatch.setattr(L, "_zimage_model_dir", lambda: str(tmp_path))
    image = workspaces["image"]
    values = image.load_values(get_entry("zimage_turbo"), "full", None).values
    assert values["dit"] == "" and values["vae"] == "" and values["text_encoder"] == ""  # not downloaded: nothing wrong is filled
    for name in L.ZIMAGE_DIT_FILES + tuple(L.ZIMAGE_SHARED_FILES.values()):
        (tmp_path / name).write_bytes(b"")
    values = image.load_values(get_entry("zimage_turbo"), "full", None).values
    assert values["dit"] == str(tmp_path / L.ZIMAGE_DIT_FILES[1])
    assert values["vae"] == str(tmp_path / L.ZIMAGE_SHARED_FILES["vae"])
    assert values["text_encoder"] == str(tmp_path / L.ZIMAGE_SHARED_FILES["text_encoder"])
    assert image.load_values(get_entry("zimage_base"), "full", None).values["dit"] == str(tmp_path / L.ZIMAGE_DIT_FILES[0])
    # a file the user picked stays; the other variant's file becomes this variant's
    picked = {"zimage_variant": "turbo", "dit": "D:/my_models/z_finetune.safetensors", "vae": "D:/my.vae", "text_encoder": "D:/te"}
    assert L.follow(_links("zimage", "zimage_variant"), picked) == {}
    other = dict(picked, dit=str(tmp_path / L.ZIMAGE_DIT_FILES[0]))
    assert L.follow(_links("zimage", "zimage_variant"), other) == {"dit": str(tmp_path / L.ZIMAGE_DIT_FILES[1])}


def test_qwen_image_model_version_switches_to_the_edit_checkpoint(tmp_path, monkeypatch):
    """V39: Model Version Edit 2509 / 2511 kept the text-to-image DiT, so the edit LoRA trained on the wrong model."""
    monkeypatch.setattr(L, "_qwen_model_dir", lambda: str(tmp_path))
    base = str(tmp_path / L.QWEN_DIT_FILES["original"][0])
    follow = lambda **values: L.follow(_links("qwen_image", "model_version"), {"dit": base, **values})
    assert follow(model_version="edit-2511") == {"dit": ""}  # not downloaded: cleared, never the wrong model
    for names in L.QWEN_DIT_FILES.values():
        for name in names:
            (tmp_path / name).write_bytes(b"")
    assert follow(model_version="edit-2511") == {"dit": str(tmp_path / "Qwen_Image_Edit_2511_BF16.safetensors")}
    assert follow(model_version="edit-2509") == {"dit": str(tmp_path / "Qwen_Image_Edit_Plus_2509_bf16.safetensors")}
    assert follow(model_version="original") == {}  # Qwen-Image or 2512: both train as original
    edit = str(tmp_path / "Qwen_Image_Edit_2511_BF16.safetensors")
    assert L.follow(_links("qwen_image", "model_version"), {"dit": edit, "model_version": "original"}) == {"dit": base}
    assert L.follow(_links("qwen_image", "model_version"), {"dit": "D:/my/qwen_finetune.safetensors", "model_version": "edit-2511"}) == {}


def test_a_switched_model_file_updates_its_status_chip(workspaces, tmp_path, monkeypatch):
    from musubi_tuner_gui.unified import presets as P

    monkeypatch.setattr(L, "_model_dir", lambda _key: str(tmp_path))
    full = tmp_path / L.HIDREAM_FILES[0]
    full.write_bytes(b"")
    image, request = workspaces["image"], _Request("links-files")
    image.forget(request)
    image.apply(request, {}, "hidream_o1", "lora", P.AUTO_TIER, None, False)
    mirror = image._mirror(request)
    model_type = image.fields[image.backend_uid["hidream_o1"]["model_type"]]
    dit = image.fields[image.backend_uid["hidream_o1"]["dit"]]
    mirror[dit.component._id]["value"] = str(full)
    entry = image.entry("hidream_o1")
    image.emit(request, {image.files_html: {"value": image.files_status(entry, image.mirror_values(request, entry))}})  # ✅ shown
    mirror[model_type.component._id]["value"] = "dev"
    updates = image.follow_links(request, "hidream_o1", model_type)
    assert updates[dit.component] == "" and "not set" in updates[image.files_html]
    assert image.files_html in [*image.link_outputs(model_type), image.files_html]


def test_wan_fp8_scaled_follows_fp8_base_in_the_workspace(workspaces):
    from musubi_tuner_gui.unified import presets as P

    video, request = workspaces["video"], _Request("links-fp8")
    video.forget(request)
    video.apply(request, {}, "wan22", "lora", P.AUTO_TIER, None, False)
    base = video.fields[video.backend_uid["wan"]["fp8_base"]]
    scaled = video.fields[video.backend_uid["wan"]["fp8_scaled"]]
    for value in (True, False, True):
        video._mirror(request)[base.component._id]["value"] = value
        video.follow_links(request, "wan22", base)
        assert video._mirror(request)[scaled.component._id]["value"] is value
    assert scaled.component in video.link_outputs(base)


def test_wan_compile_residency_follows_the_task_unless_the_user_set_it(workspaces):
    from musubi_tuner_gui.unified import presets as P

    video, request = workspaces["video"], _Request("links-compile")
    video.forget(request)
    video.apply(request, {}, "wan22", "lora", P.AUTO_TIER, None, False)
    task = video.fields[video.backend_uid["wan"]["task"]]
    resident = video.fields[video.backend_uid["wan"]["compile_resident_blocks_only"]]
    mirror = video._mirror(request)
    assert mirror[resident.component._id]["value"] is True  # Wan 2.2's default task
    mirror[task.component._id]["value"] = "t2v-14B"
    video.follow_links(request, "wan22", task)
    assert mirror[resident.component._id]["value"] is False
    mirror[resident.component._id].update(value=True, user=True)  # the user's own choice
    mirror[task.component._id]["value"] = "t2v-A14B"
    video.follow_links(request, "wan22", task)
    mirror[task.component._id]["value"] = "t2v-14B"
    video.follow_links(request, "wan22", task)
    assert mirror[resident.component._id]["value"] is True
    video.apply(request, {}, "wan21", "lora", P.AUTO_TIER, None, False)
    assert mirror[resident.component._id]["value"] is False  # Wan 2.1 compiles every block


def test_frame_counts_are_rounded_to_the_models_grid(workspaces):
    from musubi_tuner_gui.unified import presets as P

    video, request = workspaces["video"], _Request("links-frames")
    video.forget(request)
    video.apply(request, {}, "kandinsky6", "lora", P.AUTO_TIER, None, False)
    frames = video.fields[video.backend_uid["kandinsky6"]["num_frames"]]
    video._mirror(request)[frames.component._id]["value"] = 120
    updates = video.follow_links(request, "kandinsky6", frames)
    assert updates[frames.component] == 117 and video._mirror(request)[frames.component._id]["value"] == 117
    video._mirror(request)[frames.component._id]["value"] = None
    assert frames.component not in video.follow_links(request, "kandinsky6", frames)  # blank: Start names it
    rounded = L.follow(_links("minimax_h3", "dataset_num_frames"), {"dataset_num_frames": "130"})
    assert rounded == {"dataset_num_frames": 124}
    assert L.follow(_links("ltx2", "dataset_num_frames"), {"dataset_num_frames": 121}) == {}


def test_klein_sampling_follows_distilled_or_base():
    assert L.follow(_links("flux", "model_version"), {"model_version": "klein-9b", "sample_steps": 50, "sample_guidance_scale": 4.0}) == {"sample_steps": 4, "sample_guidance_scale": 1.0}
    assert L.follow(_links("flux", "model_version"), {"model_version": "klein-base-4b", "sample_steps": 4, "sample_guidance_scale": 1.0}) == {"sample_steps": 50, "sample_guidance_scale": 4.0}
