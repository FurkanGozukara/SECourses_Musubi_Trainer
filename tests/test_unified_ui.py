"""Tests for the unified Image / Video / Audio workspaces (musubi_tuner_gui/unified).

Everything here runs on the CPU in a few seconds; nothing starts a real training run.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
import gradio as gr  # noqa: E402

from musubi_tuner_gui.unified import presets as P  # noqa: E402
from musubi_tuner_gui.unified.dashboard import render_dashboard  # noqa: E402
from musubi_tuner_gui.unified.param_spec import FULL, LORA, SECTION_IDS  # noqa: E402
from musubi_tuner_gui.unified.registry import CATEGORIES, DEFAULT_ENTRY, ENTRIES, ENTRY_BY_KEY, entries_for  # noqa: E402
from musubi_tuner_gui.unified.runner import DecimatedSeries, TrainingRunner, parse_tqdm  # noqa: E402
from musubi_tuner_gui.unified.schema import build_param_specs, legacy_keys, load_schema  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _private_state(tmp_path_factory):
    """Keep every test away from the real user_presets folder, its ui_state.json and the run logs."""
    from musubi_tuner_gui.unified import runner

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        patch.setattr(runner, "RUN_LOG_DIR", str(tmp_path_factory.mktemp("training_runs")))
        yield


# --------------------------------------------------------------------------------------------- schema
def test_schema_matches_the_trainer_modules():
    """legacy_schema.json must list each trainer's settings in the trainer's own positional order."""
    from musubi_tuner_gui import flux_lora_gui, image_models_gui, ltx2_lora_gui, minimax_h3_lora_gui, modern_image_lora_gui, qwen_image_lora_gui, zimage_lora_gui

    expected = {
        "flux": list(flux_lora_gui.FLUX_PARAM_KEYS),
        "zimage": list(zimage_lora_gui.ZIMAGE_PARAM_KEYS),
        "krea2": list(modern_image_lora_gui.MODERN_IMAGE_PARAM_KEYS),
        "ideogram4": list(modern_image_lora_gui.MODERN_IMAGE_PARAM_KEYS),
        "ltx2": list(ltx2_lora_gui.LTX2_PARAM_KEYS),
        "minimax_h3": list(minimax_h3_lora_gui.MINIMAX_H3_PARAM_KEYS),
        "qwen_image": list(inspect.signature(qwen_image_lora_gui.qwen_image_gui_actions).parameters)[5:],
    }
    for spec_key in ("qwen_image_21", "ernie_image", "mage_flow", "flux3_klein", "moss_image", "hidream_o1", "kandinsky6"):
        expected[spec_key] = image_models_gui.param_keys(image_models_gui.ARCHITECTURES[spec_key])
    for name, keys in expected.items():
        assert legacy_keys(name) == keys, f"{name}: legacy_schema.json is stale — run tools/generate_unified_schema.py"


def test_param_specs_are_valid_for_every_backend():
    for name in load_schema()["backends"]:
        specs = build_param_specs(name)
        assert len({spec.key for spec in specs}) == len(specs)
        for spec in specs:
            assert spec.section in SECTION_IDS, (name, spec.key, spec.section)
            assert spec.modes and spec.modes <= {LORA, FULL}


# --------------------------------------------------------------------------------------------- registry
def test_registry_entries_are_consistent():
    keys = [entry.key for entry in ENTRIES]
    assert len(keys) == len(set(keys))
    assert ENTRY_BY_KEY["krea2"].category == "image" and DEFAULT_ENTRY["image"] == "krea2"
    backends = set(load_schema()["backends"])
    for entry in ENTRIES:
        assert entry.category in CATEGORIES
        assert entry.modes and set(entry.modes) <= {LORA, FULL}
        assert entry.is_kohya or entry.backend in backends, entry.key
        for folder in entry.preset_folders:
            assert folder.endswith("_Training_Configs")
    for category in CATEGORIES:
        assert DEFAULT_ENTRY[category] in [entry.key for entry in entries_for(category)]


# --------------------------------------------------------------------------------------------- workspace
@pytest.fixture(scope="module")
def workspaces():
    from musubi_tuner_gui.unified.backends import install_quiet_toasts
    from musubi_tuner_gui.unified.workspace import Workspace

    install_quiet_toasts()
    built = {}
    with gr.Blocks():
        for category in CATEGORIES:
            workspace = Workspace(category, headless=True)
            workspace.build()
            built[category] = workspace
    return built


class _Request:
    def __init__(self, session):
        self.session_hash = session


def _value(update):
    """The value an event result sets (a bare value, or an update dict)."""
    return update["value"] if isinstance(update, dict) else update


def test_workspaces_build_with_css_rules_for_every_backend(workspaces):
    for category, workspace in workspaces.items():
        assert workspace.fields, category
        css = workspace.generated_css()
        for name in workspace.backends:
            assert f'data-be="{name.replace(":", "-")}"' in css
        # every field carries the classes the CSS rules key on
        for item in workspace.fields.values():
            assert "vh" in item.host_classes
            assert any(cls.startswith("b-") for cls in item.host_classes)


@pytest.mark.parametrize("entry_key", [entry.key for entry in ENTRIES])
def test_every_model_and_training_type_applies(workspaces, entry_key):
    entry = ENTRY_BY_KEY[entry_key]
    workspace = workspaces[entry.category]
    if entry.key not in workspace.entry_keys:
        pytest.skip(f"{entry.key}: backend unavailable in this environment")
    for mode in entry.modes:
        request = _Request(f"apply-{entry_key}-{mode}")
        workspace.forget(request)
        preset = P.recommended_preset(entry, mode, P.AUTO_TIER)
        updates = workspace.apply(request, {}, entry.key, mode, P.AUTO_TIER, preset, False)
        assert updates, entry_key
        state = json.loads(workspace._mirror(request)[workspace.state_html._id]["value"])
        assert state["entry"] == entry.key.replace(":", "-") and state["mode"] == mode
        # fixed variant values (e.g. FLUX family, Mage-Flow edit) reach the trainer values
        loaded = workspace.load_values(entry, mode, preset)
        assert not loaded.error, loaded.error
        values = loaded.values
        for key, value in entry.fixed.items():
            assert values.get(key) == value
        backend = workspace.backend_of(entry)
        if "training_mode" in backend.keys():
            assert values["training_mode"] == backend.mode_value(mode)


def test_switch_sends_only_changes(workspaces):
    workspace = workspaces["image"]
    request = _Request("diff")
    workspace.forget(request)
    first = workspace.apply(request, {}, "flux2_dev", LORA, P.AUTO_TIER, None, False)
    again = workspace.apply(request, {}, "flux2_dev", LORA, P.AUTO_TIER, None, False)
    assert len(again) <= 2 < len(first)  # nothing but the banner (and maybe the state) on a repeat


def test_user_paths_survive_switches_but_shipped_machine_paths_do_not(workspaces):
    from musubi_tuner_gui.unified.workspace import _foreign_path

    workspace = workspaces["image"]
    request = _Request("sticky")
    workspace.forget(request)
    workspace.apply(request, {}, "krea2", LORA, P.AUTO_TIER, None, False)
    item = workspace.fields["output_dir"]
    known = workspace._mirror(request)[item.component._id]
    known["value"], known["user"] = "D:/my_outputs", True
    workspace.apply(request, {}, "hidream_o1", LORA, P.AUTO_TIER, P.recommended_preset(ENTRY_BY_KEY["hidream_o1"], LORA, P.AUTO_TIER), False)
    assert workspace._mirror(request)[item.component._id]["value"] == "D:/my_outputs"
    assert _foreign_path("/home/Ubuntu/Downloads/training_imgs")
    assert not _foreign_path("../Training_Outputs_Krea_2") and not _foreign_path("")


# --------------------------------------------------------------------------------------------- presets
def test_preset_name_or_path_targets(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))
    entry = ENTRY_BY_KEY["krea2"]
    assert P.save_target("my best run", entry) == str(tmp_path / "user_presets" / "krea2" / "my best run.toml")
    absolute = str(tmp_path / "elsewhere" / "custom")
    assert P.save_target(absolute, entry) == absolute + ".toml"
    assert P.save_target(absolute + ".json", entry).endswith("custom.toml")
    with pytest.raises(ValueError):
        P.save_target("   ", entry)


def test_saved_preset_identity_round_trip_and_guardrail(tmp_path, monkeypatch, workspaces):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))
    workspace = workspaces["image"]
    entry = ENTRY_BY_KEY["qwen_image_21"]
    backend = workspace.backend_of(entry)
    path = P.save_target("guard test", entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    values = backend.defaults(FULL)
    values["learning_rate"] = 1.234e-5
    backend.save(path, values)
    P.write_metadata_header(path, entry, FULL)
    identity = P.identify_preset(path)
    assert (identity.category, identity.entry_key, identity.mode, identity.source) == ("image", "qwen_image_21", FULL, "metadata")
    assert P.find_preset("guard test", entry, LORA) == path
    loaded = backend.load(path, backend.defaults(LORA))
    assert abs(float(loaded.values["learning_rate"]) - 1.234e-5) < 1e-12
    # the video workspace refuses an image preset
    video = workspaces["video"]
    request = _Request("guard")
    video.forget(request)
    updates = video.load_with_guardrails(request, {}, "wan22", LORA, P.AUTO_TIER, path, False)
    banner = updates.get(video.banner_html)
    assert banner is not None and "Not loaded" in _value(banner)


def test_shipped_presets_are_identified_from_their_folder():
    for entry in ENTRIES:
        for mode in entry.modes:
            for option in P.preset_options(entry, mode, "all")[:2]:
                if not option.shipped:
                    continue
                identity = P.identify_preset(option.path)
                assert identity.entry_key == entry.key or (identity.entry_key or "").split("_")[0] == entry.key.split("_")[0], option.path
                assert P.is_read_only_preset(option.path)


def test_multi_gpu_presets_are_not_auto_picked_on_one_gpu(monkeypatch):
    from musubi_tuner_gui import preset_library

    monkeypatch.setattr(preset_library, "detect_gpus", lambda: (("RTX 5090", 31.8),))
    picked = P.recommended_preset(ENTRY_BY_KEY["flux1_dev"], LORA, P.AUTO_TIER)
    if picked:
        assert "x_GPU" not in os.path.basename(picked)


def test_numbers_reach_the_trainers_like_from_the_classic_tabs():
    """The classic tabs get numbers through JSON, so whole numbers arrive as int; the runtime TOML keeps the type."""
    from musubi_tuner_gui.unified.backends import clean_preview, get_backend

    backend = get_backend("krea2")
    steps = backend.coerce("gradient_accumulation_steps", 1.0)
    assert steps == 1 and isinstance(steps, int)
    assert backend.coerce("learning_rate", 6e-5) == 6e-5
    assert isinstance(backend.coerce("discrete_flow_shift", 2.5), float)
    assert backend.coerce("seed", "") is None
    preview = clean_preview("\x1b[33mWARNING\x1b[0m run:\n\nrun:\ncmd --a 1\n\ncmd --a 1\n\nnext")
    assert preview == "WARNING run:\n\nrun:\ncmd --a 1\n\nnext"


def test_preflight_names_the_missing_setting(tmp_path):
    from musubi_tuner_gui.unified.workspace import FOLDER_DATASET_MODE, preflight

    base = {"dataset_config_mode": FOLDER_DATASET_MODE, "parent_folder_path": "", "dataset_config": "", "output_dir": str(tmp_path)}
    with pytest.raises(gr.Error, match="Parent Dataset Folder"):
        preflight(base)
    with pytest.raises(gr.Error, match="does not exist"):
        preflight(dict(base, parent_folder_path=str(tmp_path / "missing")))
    preflight(dict(base, parent_folder_path=f'"{tmp_path}"'))  # pasted quotes are fine
    with pytest.raises(gr.Error, match="dataset TOML"):
        preflight(dict(base, dataset_config_mode="Use TOML File"))
    with pytest.raises(gr.Error, match="Output Folder"):
        preflight(dict(base, parent_folder_path=str(tmp_path), output_dir=" "))


def test_no_change_results_name_the_events_own_outputs(workspaces):
    """Gradio rejects a result for a component the event does not output (Refresh, Browse, file status)."""
    workspace = workspaces["image"]
    request = _Request("no-change")
    workspace.forget(request)
    desired = workspace.desired_presets(workspace.entry("krea2"), LORA, P.AUTO_TIER, None)
    workspace.emit(request, desired)
    assert list(workspace.emit(request, desired)) == [workspace.tier_dd]
    workspace.emit(request, {workspace.files_html: {"value": "x"}})
    assert list(workspace.emit(request, {workspace.files_html: {"value": "x"}})) == [workspace.files_html]


def test_every_switch_resends_the_rule_state(workspaces):
    """A dependent-field rule the browser switched itself (dataset mode) follows the loaded values again."""
    workspace = workspaces["image"]
    request = _Request("seq")
    workspace.forget(request)
    first = workspace.apply(request, {}, "krea2", LORA, P.AUTO_TIER, None, False)
    second = workspace.apply(request, {}, "krea2", LORA, P.AUTO_TIER, None, False)
    assert workspace.state_html in first and workspace.state_html in second


def test_unreadable_presets_are_refused_or_replaced_by_the_defaults(tmp_path, monkeypatch, workspaces):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))
    workspace = workspaces["image"]
    request = _Request("unreadable")
    workspace.forget(request)
    workspace.apply(request, {}, "krea2", LORA, P.AUTO_TIER, None, False)
    shown = workspace._mirror(request)[workspace.preset_dd._id].get("value")
    broken = tmp_path / "user_presets" / "krea2" / "broken.toml"
    broken.parent.mkdir(parents=True)
    broken.write_text("learning_rate = = 1\n[[[", encoding="utf-8")
    # picked by the user: refused, nothing applied, and the selector returns to the preset really loaded
    client = {workspace.preset_dd: str(broken)}
    updates = workspace.load_with_guardrails(request, client, "krea2", LORA, P.AUTO_TIER, str(broken), False)
    assert "Not loaded" in _value(updates[workspace.banner_html]) and workspace.state_html not in updates
    assert _value(updates[workspace.preset_dd]) == shown
    # remembered from an earlier session: the defaults load, the reason is shown, and it is forgotten
    P.remember_selection("image", "krea2", LORA, P.AUTO_TIER, str(broken))
    updates = workspace.apply(request, {}, "krea2", LORA, P.AUTO_TIER, str(broken), False)
    assert "could not be loaded" in _value(updates[workspace.banner_html])
    assert P.remembered_preset("image", "krea2", LORA) is None
    # a JSON file that is not a settings object never breaks the preset list
    (broken.parent / "listed.json").write_text("[1, 2]", encoding="utf-8")
    assert P.read_metadata(str(broken.parent / "listed.json")) == {}
    assert len(P.user_presets(ENTRY_BY_KEY["krea2"], LORA)) == 2


def test_page_reload_picks_up_a_preset_saved_again(tmp_path, monkeypatch, workspaces):
    from musubi_tuner_gui.unified.workspace import _signature

    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))
    workspace = workspaces["image"]
    entry = ENTRY_BY_KEY["krea2"]
    backend = workspace.backend_of(entry)
    path = P.save_target("reload test", entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backend.save(path, backend.defaults(LORA))
    P.write_metadata_header(path, entry, LORA)
    P.remember_selection("image", "krea2", LORA, P.AUTO_TIER, path)
    monkeypatch.setattr(workspace, "initial", ("krea2", LORA, P.AUTO_TIER, path, _signature(path)))
    assert workspace.state_html not in workspace.restore(_Request("reload-1"))  # unchanged: the built page is right
    backend.save(path, dict(backend.defaults(LORA), learning_rate=3e-5))
    P.write_metadata_header(path, entry, LORA)
    stamp = os.path.getmtime(path) + 5
    os.utime(path, (stamp, stamp))
    assert workspace.state_html in workspace.restore(_Request("reload-2"))


def test_a_gpu_tier_this_model_does_not_offer_falls_back_to_auto():
    wan21 = ENTRY_BY_KEY["wan21"]
    offered = {value for _label, value in P.tier_choices(wan21, LORA)}
    assert "1234" not in offered
    assert P.offered_tier(wan21, LORA, "1234") == P.AUTO_TIER
    assert P.recommended_preset(wan21, LORA, "1234") == P.recommended_preset(wan21, LORA, P.AUTO_TIER) is not None
    assert P.preset_options(wan21, LORA, "1234")


def test_training_type_comes_from_the_preset_folders_not_the_install_path(tmp_path):
    import shutil

    source = next(option.path for option in P.preset_options(ENTRY_BY_KEY["wan21"], LORA, "all") if option.shipped)
    folder = tmp_path / "DreamBooth_Apps" / "Fine_Tuning" / "Training_Presets" / "Wan_21_Training_Configs" / "LoRA_Training"
    folder.mkdir(parents=True)
    copy = folder / os.path.basename(source)
    shutil.copy(source, copy)
    identity = P.identify_preset(str(copy))
    assert (identity.entry_key, identity.mode) == ("wan21", LORA)
    # a few settings every trainer has: no model is guessed
    partial = tmp_path / "my_lr.toml"
    partial.write_text("learning_rate = 5e-5\nmax_train_epochs = 10\n", encoding="utf-8")
    assert P.identify_preset(str(partial)).source == "unknown"
    # kohya_ss configs: DreamBooth / fine-tuning files carry no LoRA settings at all
    kohya = {"pretrained_model_name_or_path": "flux1-dev.safetensors", "flux1_checkbox": True, "ae": "ae.safetensors", "learning_rate": 4e-6}
    (tmp_path / "db.json").write_text(json.dumps(kohya), encoding="utf-8")
    (tmp_path / "lora.json").write_text(json.dumps(dict(kohya, LoRA_type="Flux1", network_dim=16)), encoding="utf-8")
    assert (P.identify_preset(str(tmp_path / "db.json")).entry_key, P.identify_preset(str(tmp_path / "db.json")).mode) == ("flux1_dev", FULL)
    assert P.identify_preset(str(tmp_path / "lora.json")).mode == LORA


def test_preset_names_with_an_extension_or_odd_characters(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))
    entry = ENTRY_BY_KEY["krea2"]
    root = tmp_path / "user_presets" / "krea2"
    assert P.save_target("my run.toml", entry) == str(root / "my run.toml")  # a name, not a file in the app folder
    assert P.save_target("my:preset", entry) == str(root / "my_preset.toml")
    root.mkdir(parents=True)
    (root / "my_preset.toml").write_text("", encoding="utf-8")
    assert P.find_preset("my:preset", entry, LORA) == str(root / "my_preset.toml")  # found by the name it was saved with
    assert P.find_preset("MY_PRESET.toml", entry, LORA) == str(root / "my_preset.toml")
    # files a save is still writing (or that an interrupted save left) never show up as presets
    (root / ".tmp_abc123.toml").write_text("", encoding="utf-8")
    (root / "mine.saving.toml").write_text("", encoding="utf-8")
    assert [option.label for option in P.user_presets(entry, LORA)] == [f"{P.USER_ICON} my_preset"]


def test_shipped_presets_are_read_only_in_any_letter_case():
    path = next(option.path for option in P.preset_options(ENTRY_BY_KEY["qwen_image_21"], LORA, "all") if option.shipped)
    assert P.is_read_only_preset(path)
    if os.name == "nt":
        assert P.is_read_only_preset(path.lower()) and P.is_read_only_preset(path.upper())
    assert not P.is_read_only_preset(P.save_target("mine", ENTRY_BY_KEY["qwen_image_21"]))


def test_a_damaged_ui_state_never_stops_the_app(tmp_path, monkeypatch):
    from musubi_tuner_gui.unified.workspace import Workspace

    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path))
    (tmp_path / "ui_state.json").write_text(json.dumps({"image": "krea2", "audio": {"entry": 5, "presets": [1], "modes": "x"}}), encoding="utf-8")
    assert P.remembered("image") == {} and P.remembered("audio") == {}
    assert P.remembered_preset("audio", "ltx23_audio", LORA) is None and P.remembered_mode("audio", "ltx23_audio") is None
    with gr.Blocks():
        Workspace("audio", headless=True).build()


def test_a_preset_for_a_model_whose_trainer_failed_to_load_is_refused(monkeypatch, workspaces):
    workspace = workspaces["image"]
    entry = ENTRY_BY_KEY["hidream_o1"]
    path = P.recommended_preset(entry, LORA, P.AUTO_TIER)
    if not path:
        pytest.skip("no shipped HiDream-O1 preset")
    monkeypatch.setattr(workspace, "entry_keys", [key for key in workspace.entry_keys if key != entry.key])
    request = _Request("unavailable")
    workspace.forget(request)
    updates = workspace.load_with_guardrails(request, {}, "krea2", LORA, P.AUTO_TIER, path, False)
    assert "not available in this install" in _value(updates[workspace.banner_html])


def test_model_file_status_follows_the_whole_form(workspaces):
    """After a path edit the file list still knows the other settings (Wan 2.2's task adds the high-noise DiT)."""
    import html

    video = workspaces["video"]
    entry = video.entry("wan22")
    request = _Request("files")
    video.forget(request)
    video.apply(request, {}, entry.key, LORA, P.AUTO_TIER, None, False)
    values = video.mirror_values(request, entry)
    assert "A14B" in str(values.get("task"))
    label = video.backend_of(entry).spec_map()["dit_high_noise"].label
    assert html.escape(label) in video.files_status(entry, values)


def test_every_default_fits_its_number_fields_step(workspaces):
    """The browser marks a number off its field's step invalid and its arrows jump by the step: kohya's Text Encoder
    LR had no step (Gradio's 1) for 5e-5, Qwen-Image Edit's control resolution step 64 for its 1328 default."""
    off = []
    for category, workspace in workspaces.items():
        for item in workspace.fields.values():
            if item.kind != "number":
                continue
            low, _high, step = item.bounds
            for name, spec in item.per_backend.items():
                for mode in spec.modes:
                    value = workspace.backends[name].defaults(mode).get(spec.key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    count = (value - (low or 0)) / step
                    if abs(count - round(count)) > 1e-6 * max(1.0, abs(count)):
                        off.append((category, item.uid, name, mode, value, step))
    assert not off, off


def test_a_shared_field_shows_each_models_own_rule(workspaces):
    """LTX 2.3 and MiniMax H3 share Target Frames: LTX showed H3's "17n+5 ... 124 = released minimum" (the longer
    text won the tie) although its frames are 8k+1; a text that states numbers is each model's own."""
    video = workspaces["video"]
    frames = video.fields["dataset_num_frames"]

    def shown(item, backend):
        return item.text_overrides.get(backend, (item.label, item.info))[1]

    assert "8k+1" in shown(frames, "ltx2") and "17n+5" not in shown(frames, "ltx2")
    assert "17n+5" in shown(frames, "minimax_h3")
    width = next(item for item in video.fields.values() if item.key == "dataset_resolution_width" and "kandinsky6" in item.per_backend)
    assert shown(width, "kandinsky6") == video.backends["kandinsky6"].spec_map()["dataset_resolution_width"].info


def test_a_hugging_face_id_is_not_reported_missing(workspaces):
    """The SDXL presets' VAE is stabilityai/sdxl-vae: sd-scripts downloads it, so the file list must not say "not found"."""
    image = workspaces["image"]
    entry = image.entry("sdxl")
    values = image.backend_of(entry).defaults(FULL)
    values["vae"] = "stabilityai/sdxl-vae"
    status = image.files_status(entry, values)
    assert "Hugging Face" in status and "VAE (optional) — not found" not in status
    flux = image.entry("flux1_dev")
    flux_values = image.backend_of(flux).defaults(LORA)
    flux_values["pretrained_model_name_or_path"] = "black-forest-labs/FLUX.1-dev"  # FLUX.1 needs its file
    assert "Hugging Face" not in image.files_status(flux, flux_values)


def test_a_model_downloaded_after_the_preset_loaded_is_found(tmp_path, monkeypatch, workspaces):
    """A FLUX.1 preset loaded before ⬇️ Download keeps its relative Training_Models_FLUX_1/... paths (kohya relocates
    only files that exist); training finds them under the install folder, and so must the file list."""
    from musubi_tuner_gui.unified import workspace as W

    image = workspaces["image"]
    entry = image.entry("flux1_dev")
    values = image.backend_of(entry).defaults(LORA)
    values["ae"] = "Training_Models_FLUX_1/ae.safetensors"
    monkeypatch.setattr(W, "ROOT_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path / "..")
    assert "FLUX VAE (AE) — not found" in image.files_status(entry, values)
    (tmp_path / "Training_Models_FLUX_1").mkdir()
    (tmp_path / "Training_Models_FLUX_1" / "ae.safetensors").write_bytes(b"0")  # the download
    status = image.files_status(entry, values)
    assert "✅ FLUX VAE (AE)" in status and "FLUX VAE (AE) — not found" not in status


def test_fixed_settings_are_never_shown_as_editable():
    """A setting the base model fixes (Mage-Flow Edit's control images) must not show a control that does nothing."""
    from musubi_tuner_gui.unified.backends import get_backend
    from musubi_tuner_gui.unified.schema import HIDDEN_KEYS

    for entry in ENTRIES:
        specs = get_backend(entry.backend).spec_map()
        for key in entry.fixed:
            spec = specs.get(key)
            if spec is not None and spec.visible and key not in HIDDEN_KEYS:
                assert key in entry.locked_keys, (entry.key, key)


# --------------------------------------------------------------------------------------------- datasets
def test_ltx_audio_folders_become_audio_datasets(tmp_path):
    import toml

    from musubi_tuner.dataset.config_utils import ConfigSanitizer

    from musubi_tuner_gui.unified.dataset_tools import FOLDER_MODE, prepare_dataset

    clips = tmp_path / "data" / "3_my voice"
    clips.mkdir(parents=True)
    (clips / "a.wav").write_bytes(b"RIFF")
    (clips / "b.flac").write_bytes(b"fLaC")
    (clips / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "data" / "no audio").mkdir()
    values = {
        "dataset_config_mode": FOLDER_MODE,
        "parent_folder_path": str(tmp_path / "data"),
        "ltx2_mode": "audio",
        "output_dir": str(tmp_path / "out"),
        "create_missing_captions": True,
        "caption_strategy": "folder_name",
    }
    notes = prepare_dataset("ltx2", values)
    config = toml.load(values["dataset_config"])
    assert values["generated_toml_path"] == values["dataset_config"]
    assert [(os.path.normpath(d["audio_directory"]), d["num_repeats"]) for d in config["datasets"]] == [(str(clips), 3)]
    assert (clips / "b.txt").read_text(encoding="utf-8") == "my voice"
    assert any("no audio" in note for note in notes)
    ConfigSanitizer().sanitize_user_config(config)  # musubi's own dataset schema accepts it
    # video and audio-video runs use the trainer's own image / video generator, which finds nothing here
    with pytest.raises(ValueError, match="No valid datasets"):
        prepare_dataset("ltx2", dict(values, ltx2_mode="av"))


# --------------------------------------------------------------------------------------------- runner
def test_tqdm_parser_and_bounded_series():
    bar = parse_tqdm("steps:  99%|█████████▉| 99000/100000 [52:10:03<00:31:37, 1.90s/it, avr_loss=0.0412, lr=1e-05]")
    assert bar["n"] == 99000 and bar["total"] == 100000 and bar["elapsed"] == 187803 and bar["postfix"]["lr"] == "1e-05"
    assert parse_tqdm("no progress here") is None
    series = DecimatedSeries(480)
    for step in range(1, 100_001):
        series.add(step, 1.0 / step)
    assert len(series.points) <= 480


def _python(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", code]


def _wait(runner: TrainingRunner, seconds: float = 20.0) -> None:
    deadline = time.time() + seconds
    while runner.is_active() and time.time() < deadline:
        time.sleep(0.05)


def test_runner_tracks_a_run_to_completion():
    runner = TrainingRunner()
    script = (
        "import sys, time\n"
        "print('caching latents', flush=True)\n"
        "print('epoch 1/2', flush=True)\n"
        "for i in range(1, 51):\n"
        "    sys.stderr.write(f'\\rsteps:  {i*2}%|#| {i}/50 [00:01<00:01, 25.00it/s, avr_loss={1/i:.4f}, lr=1e-04]')\n"
        "    sys.stderr.flush()\n"
        "    time.sleep(0.005)\n"
        "print('', flush=True)\n"
        "print('saving checkpoint: out/model-000001.safetensors', flush=True)\n"
    )
    started, _ = runner.start_job(lambda: runner.execute_command(_python(script)), title="unit test", meta={})
    assert started
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "finished", snapshot["message"]
    assert snapshot["step"] == 50 and snapshot["total_steps"] == 50 and snapshot["epoch"] == 1
    assert snapshot["lr"] == pytest.approx(1e-4) and snapshot["checkpoints"][-1]["name"] == "model-000001.safetensors"
    assert all(stage["state"] in ("done", "skipped") for stage in snapshot["stages"])
    assert "saving checkpoint" in runner.log_tail(100)
    assert render_dashboard(snapshot)


def test_runner_reports_failures_and_stops_runs():
    runner = TrainingRunner()
    runner.start_job(lambda: runner.execute_command(_python("raise SystemExit('torch.OutOfMemoryError: CUDA out of memory')")), title="fail", meta={})
    _wait(runner)
    failed = runner.snapshot()
    assert failed["status"] == "failed" and "out of memory" in failed["error"].lower()

    runner.start_job(lambda: runner.execute_command(_python("import time\nwhile True: time.sleep(0.1)")), title="stop", meta={})
    time.sleep(0.5)
    assert runner.is_active()
    runner.stop()
    _wait(runner)
    assert runner.snapshot()["status"] == "stopped" and not runner.is_running()


def test_dashboard_escapes_untrusted_text():
    runner = TrainingRunner()
    snapshot = runner.snapshot()
    snapshot.update(status="failed", job_id=1, error="<script>alert(1)</script>", title="x", started_at=time.time(), ended_at=time.time())
    html = render_dashboard(snapshot)
    assert "<script>" not in html and "&lt;script&gt;" in html


# --------------------------------------------------------------------------------------------- live LR hook
def test_metrics_hook_adds_learning_rate_to_steps_bar():
    import io

    import torch
    from tqdm import tqdm

    from musubi_tuner_gui.unified import metrics_hook

    metrics_hook.install()
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=3e-4)
    parameter.grad = torch.ones(1)
    optimizer.step()
    stream = io.StringIO()
    bar = tqdm(total=10, desc="steps", file=stream, mininterval=0)
    bar.set_postfix(avr_loss=0.1)
    bar.update(1)
    bar.close()
    assert "lr=0.0003" in stream.getvalue()


def test_best_checkpoint_finder_settings_follow_your_presets_and_reach_the_run(tmp_path, workspaces):
    from musubi_tuner_gui.unified import checkpoint_finder as CF

    image = workspaces["image"]
    assert image.finder_inputs and workspaces["video"].finder_inputs and not workspaces["audio"].finder_inputs
    entry = ENTRY_BY_KEY["krea2"]
    backend = image.backend_of(entry)
    mine = CF.normalize({"enabled": True, "subject": "sks woman", "prompts": ["photo of sks woman"], "seeds": 3, "leakage": False}, "image")
    path = P.save_target("finder test", entry)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backend.save(path, backend.defaults(LORA))
    P.write_metadata_header(path, entry, LORA, extra={CF.HEADER_KEY: CF.header_value(mine)})
    CF.remember("image", CF.defaults("image"))
    request = _Request("finder")
    image.forget(request)
    updates = image.load_with_guardrails(request, {}, "krea2", LORA, P.AUTO_TIER, path, False)
    assert _value(updates[image.bcf_subject]) == "sks woman" and _value(updates[image.bcf_prompts]) == "photo of sks woman"
    assert _value(updates[image.bcf_seeds]) == 3 and CF.remembered("image") == mine  # a loaded preset's finder settings are remembered
    # a shipped preset keeps them
    shipped = P.recommended_preset(entry, LORA, P.AUTO_TIER)
    if shipped:
        updates = image.load_with_guardrails(request, {}, "krea2", LORA, P.AUTO_TIER, shipped, False)
        assert image.bcf_subject not in updates or _value(updates[image.bcf_subject]) == "sks woman"
    # Start: the finder's sampling and the after-step
    data = tmp_path / "data" / "5_sks woman"
    data.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (64, 64)).save(data / "a.png")
    (data / "a.txt").write_text("sks woman", encoding="utf-8")
    values = backend.defaults(LORA)
    values.update(parent_folder_path=str(tmp_path / "data"), dataset_config_mode="Generate from Folder Structure", output_dir=str(tmp_path / "out"),
                  output_name="mine", save_every_n_epochs=5)
    run, after, note = image.finder_run(entry, values, mine)
    assert not note and after is not None and run["sample_every_n_epochs"] == 5 and run["sample_at_first"]
    assert run["sample_prompts"].endswith("sample_prompts.txt") and values.get("sample_prompts") != run["sample_prompts"]
    assert "--w " not in open(run["sample_prompts"], encoding="utf-8").read()  # the musubi tabs add their own sample size
    # sd-scripts samples a line without a size at 512 x 512: the finder writes the training resolution for kohya runs
    flux = ENTRY_BY_KEY["flux1_dev"]
    kohya_values = image.backend_of(flux).defaults(LORA)
    kohya_values.update(parent_folder_path=str(tmp_path / "data"), output_dir=str(tmp_path / "kohya"), output_name="k", save_every_n_epochs=5,
                        dataset_resolution_width=1024, dataset_resolution_height=768)
    kohya_run, kohya_after, _note = image.finder_run(flux, kohya_values, mine)
    lines = open(kohya_run["sample_prompts"], encoding="utf-8").read().splitlines()
    assert kohya_after is not None and lines and all(line.endswith(" --w 1024 --h 768 --d " + line.rsplit(" ", 1)[1]) for line in lines[:3])
    assert image.finder_run(entry, values, {**mine, "enabled": False})[1] is None
    edit = ENTRY_BY_KEY["mage_flow_edit"]
    edit_values = {**image.backend_of(edit).defaults(LORA), **edit.fixed, "output_dir": str(tmp_path / "out"), "output_name": "e", "save_every_n_epochs": 5}
    _run, edit_after, edit_note = image.finder_run(edit, edit_values, mine)
    assert edit_after is None and "control images" in edit_note
