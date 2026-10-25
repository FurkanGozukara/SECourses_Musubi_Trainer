"""Model paths of a preset made on another machine find the same files in this install's Training_Models* folders
(common_gui.resolve_portable_model_path): any root (Windows drive, Linux home, relative), any slash style, quotes,
letter case of the Training_Models folder, Diffusers snapshot sub-folders. Existing paths, Hugging Face repo ids and
ambiguous file names are never changed."""

from __future__ import annotations

import os
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from musubi_tuner_gui.common_gui import resolve_portable_model_path, resolve_portable_model_value  # noqa: E402


@pytest.fixture()
def models(tmp_path, monkeypatch):
    root = tmp_path / "install"
    folder = root / "Training_Models_UnitTest"
    (folder / "Snapshot-Diffusers" / "vae").mkdir(parents=True)
    (folder / "Snapshot-Diffusers" / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    (folder / "unit_dit_7c1f.safetensors").write_bytes(b"x")
    (folder / "sub").mkdir()
    (folder / "sub" / "unit_te_7c1f.safetensors").write_bytes(b"x")
    other = root / "Training_Models_Other"
    (other / "a").mkdir(parents=True)
    (other / "b").mkdir(parents=True)
    (other / "a" / "twin_7c1f.safetensors").write_bytes(b"x")
    (other / "b" / "twin_7c1f.safetensors").write_bytes(b"x")
    monkeypatch.setenv("MUSUBI_TRAINING_MODELS_DIR", str(root))
    return folder


def _same(path: str, expected) -> bool:
    return os.path.normcase(os.path.normpath(path)) == os.path.normcase(os.path.normpath(str(expected)))


@pytest.mark.parametrize(
    "written",
    [
        "/home/Ubuntu/Documents/SECourses_Musubi_Trainer_v35/Training_Models_UnitTest/unit_dit_7c1f.safetensors",
        "D:\\Apps\\SECourses_Musubi_Trainer_v40\\Training_Models_UnitTest\\unit_dit_7c1f.safetensors",
        '"D:\\My Apps\\Training_Models_UnitTest\\unit_dit_7c1f.safetensors"',
        "  'G:/Old Install/training_models_unittest/unit_dit_7c1f.safetensors'  ",
        "../Training_Models_UnitTest/unit_dit_7c1f.safetensors",
        "Training_Models_UnitTest/unit_dit_7c1f.safetensors",
        "/workspace/unit_dit_7c1f.safetensors",  # no Training_Models folder in it: found by its unique name
    ],
)
def test_a_foreign_model_path_finds_the_local_file(models, written):
    assert _same(resolve_portable_model_path(written), models / "unit_dit_7c1f.safetensors")


def test_sub_folders_and_snapshot_folders_keep_their_place(models):
    linux = "/root/Training_Models_UnitTest/sub/unit_te_7c1f.safetensors"
    assert _same(resolve_portable_model_path(linux), models / "sub" / "unit_te_7c1f.safetensors")
    snapshot = "C:\\x\\Training_Models_UnitTest\\Snapshot-Diffusers\\vae"
    assert _same(resolve_portable_model_path(snapshot), models / "Snapshot-Diffusers" / "vae")


def test_what_must_never_change(models, tmp_path):
    existing = tmp_path / "mine.safetensors"
    existing.write_bytes(b"x")
    assert resolve_portable_model_path(str(existing)) == str(existing)
    assert resolve_portable_model_path("Qwen/Qwen-Image") == "Qwen/Qwen-Image"
    ambiguous = "/elsewhere/twin_7c1f.safetensors"  # two local files share the name: never guessed
    assert resolve_portable_model_path(ambiguous) == ambiguous
    assert resolve_portable_model_path("") == ""
    missing = "/elsewhere/Training_Models_UnitTest/not_downloaded_7c1f.safetensors"
    assert resolve_portable_model_path(missing) == missing  # shown as "not found" until it is downloaded


def test_list_values_and_non_model_keys(models):
    value = ["/a/Training_Models_UnitTest/unit_dit_7c1f.safetensors", "/b/Training_Models_UnitTest/sub/unit_te_7c1f.safetensors"]
    resolved = resolve_portable_model_value("base_weights", value)
    assert _same(resolved[0], models / "unit_dit_7c1f.safetensors") and _same(resolved[1], models / "sub" / "unit_te_7c1f.safetensors")
    assert resolve_portable_model_value("output_dir", "/a/Training_Models_UnitTest/unit_dit_7c1f.safetensors").startswith("/a/")


def test_start_uses_this_installs_files_for_a_presets_paths(models):
    """A preset loaded before its models were downloaded keeps the paths it was written with; ⬇️ Download then puts the
    files in this install, and Start / Print Command (unified backends._local_model_files) use them: the trainers run
    from the app folder, where neither another PC's path nor a Training_Models_*/... path would be found."""
    from musubi_tuner_gui.unified.backends import _local_model_files

    payload = {
        "dit": "G:/SECourses_Musubi_Trainer_v29/Training_Models_UnitTest/unit_dit_7c1f.safetensors",
        "text_encoder": "Training_Models_UnitTest/sub/unit_te_7c1f.safetensors",
        "vae": "/elsewhere/Training_Models_UnitTest/not_downloaded_7c1f.safetensors",
        "output_dir": "G:/x/Training_Models_UnitTest/unit_dit_7c1f.safetensors",
        "network_weights": "",
    }
    resolved = _local_model_files(dict(payload))
    assert _same(resolved["dit"], models / "unit_dit_7c1f.safetensors")
    assert _same(resolved["text_encoder"], models / "sub" / "unit_te_7c1f.safetensors")
    assert resolved["vae"] == payload["vae"]  # not downloaded: the trainer names the missing file as written
    assert resolved["output_dir"] == payload["output_dir"]  # not a model path
    assert resolved["network_weights"] == ""


def test_model_files_status_finds_a_foreign_path_in_this_install(tmp_path, monkeypatch):
    from musubi_tuner_gui.unified import workspace

    root = tmp_path / "install"
    (root / "Training_Models_UnitTest" / "sub").mkdir(parents=True)
    (root / "Training_Models_UnitTest" / "sub" / "m_7c1f.safetensors").write_bytes(b"x")
    monkeypatch.setattr(workspace, "ROOT_DIR", str(root))
    monkeypatch.setattr(workspace, "APP_DIR", str(root / "SECourses_Musubi_Trainer"))
    for written in (
        "G:/SECourses_Musubi_Trainer_v29/Training_Models_UnitTest/sub/m_7c1f.safetensors",
        "/home/Ubuntu/Documents/SECourses_Musubi_Trainer_v35/Training_Models_UnitTest/sub/m_7c1f.safetensors",
        "D:\Apps\Training_Models_UnitTest\sub\m_7c1f.safetensors",
        "Training_Models_UnitTest/sub/m_7c1f.safetensors",
    ):
        assert workspace._model_file_exists(written), written
    assert not workspace._model_file_exists("G:/a/Training_Models_UnitTest/sub/missing_7c1f.safetensors")
    assert not workspace._model_file_exists("G:/a/sub/m_7c1f.safetensors")  # no Training_Models folder in it
