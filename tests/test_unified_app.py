"""The whole unified app: a broken trainer install, the remembered tab, model downloads. CPU only."""

from __future__ import annotations

import os
import sys
import time

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401


@pytest.fixture(scope="module")
def broken_ltx_app(tmp_path_factory):
    """The app built while the LTX 2.3 trainer fails to import (the only Audio trainer)."""
    import gradio as gr

    from musubi_tuner_gui.unified import workspace as W
    from musubi_tuner_gui.unified.app import build_app

    real = W.get_backend

    def broken(name):
        if name == "ltx2":
            raise ImportError("simulated broken LTX 2.3 install")
        return real(name)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        patch.setattr(W, "get_backend", broken)
        patch.chdir(APP_DIR)
        demo = build_app(headless=True)
        yield demo, gr


def _tabs(demo, gr):
    return next(block for block in demo.blocks.values() if isinstance(block, gr.Tabs) and block.elem_id == "main_tabs")


def test_one_broken_trainer_leaves_the_other_tabs_working(broken_ltx_app):
    demo, gr = broken_ltx_app
    texts = [str(block.value) for block in demo.blocks.values() if isinstance(block, gr.HTML)]
    assert any("could not be loaded" in text and "Audio" in text for text in texts)
    radios = [block for block in demo.blocks.values() if isinstance(block, gr.Radio) and "model-picker" in (block.elem_classes or [])]
    keys = {value for radio in radios for _label, value in radio.choices}
    assert "krea2" in keys and "wan22" in keys and "ltx23" not in keys and "ltx23_audio" not in keys


def test_a_reload_opens_the_tab_used_last(broken_ltx_app):
    from musubi_tuner_gui.unified import presets as P

    demo, gr = broken_ltx_app
    tabs = _tabs(demo, gr)
    restore = next(fn for fn in demo.fns.values() if any(event == "load" for _id, event in fn.targets) and fn.outputs and fn.outputs[0] is tabs)
    selects = [fn for fn in demo.fns.values() if any(event == "select" for _id, event in fn.targets) and fn.js and "userActivation" in fn.js]
    assert len(selects) == 5  # Image, Video, Audio, Tools, Version History
    tools = next(fn for fn in selects if fn.fn.__closure__ and "tools" in [cell.cell_contents for cell in fn.fn.__closure__])
    tools.fn("load")  # the select Gradio fires on its own when a page opens: ignored
    assert P.load_ui_state().get("last_category") != "tools"
    tools.fn("user")
    assert P.load_ui_state().get("last_category") == "tools"
    assert restore.fn().selected == "tools"
    P.update_ui_state(last_category="no such tab")
    assert restore.fn().selected == "image"


def test_a_workspace_selection_does_not_move_the_open_tab(tmp_path, monkeypatch):
    from musubi_tuner_gui.unified import presets as P

    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path))
    P.update_ui_state(last_category="tools")
    P.remember_selection("video", "wan22", "lora", None, None)  # every workspace restores on page load
    assert P.load_ui_state()["last_category"] == "tools"
    assert P.remembered("video")["entry"] == "wan22"


def test_a_stopped_download_reads_as_stopped(tmp_path, monkeypatch):
    from musubi_tuner_gui.unified import downloads as D

    script = tmp_path / "fake_downloader.py"
    script.write_text("import time\nprint('fetching', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
    monkeypatch.setattr(D, "DOWNLOADER", str(script))
    manager = D.DownloadManager()
    assert manager.start("bundle", "Fake models").startswith("Downloading")
    process = manager.process
    deadline = time.time() + 20
    while time.time() < deadline and "fetching" not in manager.snapshot()[2]:
        time.sleep(0.05)
    assert manager.stop().startswith("Download stopped")
    process.wait(timeout=20)
    deadline = time.time() + 20
    while time.time() < deadline and "===" not in manager.snapshot()[2].splitlines()[-1]:
        time.sleep(0.05)
    _version, status, text = manager.snapshot()
    assert status == "stopped" and "stopped" in text.splitlines()[-1] and "exit code" not in text


def test_every_download_button_names_a_bundle_of_the_downloader():
    # Download_Train_Models.py ships in the install root, next to this repository
    import ast

    from musubi_tuner_gui.unified import downloads as D
    from musubi_tuner_gui.unified.registry import ENTRIES

    if not os.path.isfile(D.DOWNLOADER):
        pytest.skip("Download_Train_Models.py is not installed next to the app")
    tree = ast.parse(open(D.DOWNLOADER, encoding="utf-8").read())
    configs = next(node.value for node in tree.body if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "MODEL_CONFIGS" for t in node.targets))
    known = {key.value for key in configs.keys}
    named = {bundle for entry in ENTRIES for _label, bundle in entry.download_bundles}
    assert named and named <= known, sorted(named - known)
    assert {"flux1_dev", "sdxl_realvis"} <= named  # the kohya models download like the others


def test_the_downloader_saves_next_to_itself_wherever_it_is_started(tmp_path, monkeypatch):
    """Every bundle's Training_Models_* folder is made next to Download_Train_Models.py (the install root, where the app
    looks for its models), not in the folder the console happens to be in; an explicit --dir is taken from that folder
    (user report 2026-10-25: models went to another drive)."""
    import importlib.util

    from musubi_tuner_gui.unified import downloads as D

    if not os.path.isfile(D.DOWNLOADER):
        pytest.skip("Download_Train_Models.py is not installed next to the app")
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "300"))  # the module sets it
    spec = importlib.util.spec_from_file_location("download_train_models_under_test", D.DOWNLOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)

    def real(path) -> str:
        return os.path.normcase(os.path.realpath(str(path)))

    root = real(os.path.dirname(D.DOWNLOADER))
    for config in module.MODEL_CONFIGS.values():
        target = real(module.default_download_dir(config))
        assert target == real(os.path.join(root, config["default_dir"])), target
    # resolved on both sides: a Training_Models_* folder may be a link to another drive (the server's dev install)
    expected = {real(os.path.join(root, config["default_dir"])) for config in module.MODEL_CONFIGS.values()}
    assert {real(folder) for folder in module.get_default_model_directories()} == expected
    assert real(module.normalize_path("my_models")) == real(tmp_path / "my_models")  # --dir: the console's folder
    assert real(module.normalize_path('"' + str(tmp_path / "with space") + '"')) == real(tmp_path / "with space")
    assert not any(name.startswith("Training_Models") for name in os.listdir(tmp_path))
