"""The unified UI's LoRA / DoRA switch: folded into Network Arguments for the trainers, read back from them on load."""

from __future__ import annotations

import os
import sys

import pytest
import toml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from media_fixtures import caption, write_image  # noqa: E402
from test_unified_runtime_toml import _runtime_toml  # noqa: E402

from musubi_tuner_gui.unified.adapter import (  # noqa: E402
    ADAPTER_KEY,
    DORA_ADAPTER,
    LORA_ADAPTER,
    fold_adapter,
    is_synthetic,
    unfold_adapter,
)
from musubi_tuner_gui.unified.backends import get_backend, install_quiet_toasts  # noqa: E402
from musubi_tuner_gui.unified.legacy_tabs import LEGACY_BACKENDS  # noqa: E402
from musubi_tuner_gui.unified.registry import ENTRY_BY_KEY  # noqa: E402
from musubi_tuner_gui.unified.runner import install_runner  # noqa: E402
from musubi_tuner_gui.unified.schema import load_schema  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _app_like(tmp_path_factory):
    install_quiet_toasts()
    install_runner()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        patch.chdir(tmp_path_factory.mktemp("cwd"))
        yield


@pytest.mark.parametrize(
    ("adapter", "network_args", "expected"),
    [
        (DORA_ADAPTER, "", "use_dora=True"),
        (DORA_ADAPTER, "conv_dim=4  conv_alpha=1", "use_dora=True conv_dim=4  conv_alpha=1"),
        (DORA_ADAPTER, "use_dora=false, conv_dim=4", "use_dora=True conv_dim=4"),
        (DORA_ADAPTER, "dora_axis=output", "use_dora=True dora_axis=output"),
        (LORA_ADAPTER, "use_dora=True dora_axis=input conv_dim=4", "conv_dim=4"),
        (LORA_ADAPTER, "conv_dim=4", "conv_dim=4"),
        (" dora ", "", "use_dora=True"),
    ],
)
def test_fold_writes_the_switch_into_network_arguments(adapter, network_args, expected):
    folded = fold_adapter({ADAPTER_KEY: adapter, "network_args": network_args})
    assert folded["network_args"] == expected


@pytest.mark.parametrize("typed", ["conv_dim=8, conv_alpha=4", "conv_dim=8\nconv_alpha=4", "", "a=1;b=2"])
def test_switching_to_dora_and_back_keeps_the_text_as_typed(typed):
    folded = fold_adapter({ADAPTER_KEY: DORA_ADAPTER, "network_args": typed})
    unfolded = unfold_adapter({"network_args": folded["network_args"]})
    assert unfolded[ADAPTER_KEY] == DORA_ADAPTER
    assert unfolded["network_args"] == typed


def test_fold_keeps_list_values_lists():
    folded = fold_adapter({ADAPTER_KEY: DORA_ADAPTER, "network_args": ["conv_dim=4"]})
    assert folded["network_args"] == ["use_dora=True", "conv_dim=4"]


@pytest.mark.parametrize(
    ("network_args", "adapter", "remaining"),
    [
        ("use_dora=True", DORA_ADAPTER, ""),
        (["use_dora=true", "dora_axis=input", "conv_dim=4"], DORA_ADAPTER, ["conv_dim=4"]),
        ("use_dora=1 dora_axis=output", DORA_ADAPTER, "dora_axis=output"),
        ("use_dora=False conv_dim=4", LORA_ADAPTER, "conv_dim=4"),
        ("conv_dim=4", LORA_ADAPTER, "conv_dim=4"),
        ("", LORA_ADAPTER, ""),
    ],
)
def test_unfold_reads_the_switch_back(network_args, adapter, remaining):
    unfolded = unfold_adapter({"network_args": network_args})
    assert unfolded[ADAPTER_KEY] == adapter
    assert unfolded["network_args"] == remaining


@pytest.mark.parametrize("name", sorted(LEGACY_BACKENDS))
def test_the_switch_never_shifts_the_classic_positional_arguments(name):
    backend = get_backend(name)
    classic_keys = load_schema()["backends"][name]["keys"]
    if "network_args" not in classic_keys:
        pytest.skip("no network arguments")
    assert backend.specs[-1].key == ADAPTER_KEY and is_synthetic(backend.specs[-1])
    values = backend.defaults("lora")
    values[ADAPTER_KEY] = DORA_ADAPTER
    args = backend.ordered_args(values)
    assert len(args) == len(classic_keys)
    network_args = args[classic_keys.index("network_args")]
    assert "use_dora=True" in network_args
    if "full" in {mode for entry in ENTRY_BY_KEY.values() if entry.backend == name for mode in entry.modes}:
        values = backend.defaults("full")
        values[ADAPTER_KEY] = DORA_ADAPTER
        args = backend.ordered_args(values)
        assert "use_dora=True" not in (args[classic_keys.index("network_args")] or [])


@pytest.mark.parametrize("entry_key", ["qwen_image", "krea2", "ltx23", "wan21"])
def test_dora_survives_save_load_and_reaches_the_runtime_toml(entry_key, tmp_path):
    entry = ENTRY_BY_KEY[entry_key]
    backend = get_backend(entry.backend)
    values = backend.defaults("lora")
    values.update(entry.fixed)
    values[ADAPTER_KEY] = DORA_ADAPTER
    values["network_args"] = "conv_dim=4 conv_alpha=1" if entry_key == "qwen_image" else ""
    preset = tmp_path / "dora preset.toml"
    backend.save(str(preset), values)
    saved = toml.load(preset)
    assert "use_dora=True" in saved["network_args"]

    loaded = backend.load(str(preset), backend.defaults("lora")).values
    assert loaded[ADAPTER_KEY] == DORA_ADAPTER
    assert "use_dora" not in str(loaded["network_args"])

    loaded[ADAPTER_KEY] = LORA_ADAPTER
    backend.save(str(preset), loaded)
    assert "use_dora" not in str(toml.load(preset).get("network_args", ""))

    models = tmp_path / "models"
    models.mkdir()
    for spec in backend.specs:
        if spec.kind == "file" and spec.section == "model":
            (models / f"{spec.key}.safetensors").touch()
            loaded[spec.key] = str(models / f"{spec.key}.safetensors")
        elif spec.kind == "folder" and spec.section == "model":
            (models / spec.key).mkdir()
            loaded[spec.key] = str(models / spec.key)
    for key in ("turbo_dit", "unconditional_dit", "training_adapter_path", "gemma_safetensors", "network_weights", "base_weights", "dit_high_noise", "resume"):
        if key in loaded:
            loaded[key] = ""
    images = tmp_path / "images" / "1_ohwx"
    images.mkdir(parents=True)
    caption(write_image(str(images / "a.png"), size=(512, 512)), "ohwx")
    out = tmp_path / "out"
    out.mkdir()
    loaded.update(
        adapter_type=DORA_ADAPTER,
        dataset_config_mode="Generate from Folder Structure",
        parent_folder_path=str(images.parent),
        output_dir=str(out),
        sample_prompts="",
    )
    if "training_mode" in loaded:
        loaded["training_mode"] = backend.mode_value("lora")
    data = _runtime_toml(backend, loaded, out)
    assert "use_dora=True" in data["network_args"]


@pytest.mark.parametrize("model_key", ["flux1_dev", "sdxl"])
def test_kohya_dora_reaches_sd_scripts_and_the_preset(model_key, tmp_path):
    from musubi_tuner_gui.unified import kohya_backend

    backend = get_backend(f"kohya:{model_key}")
    assert backend.specs[-1].key == ADAPTER_KEY
    values = backend.defaults("lora")
    values[ADAPTER_KEY] = DORA_ADAPTER
    values["network_args"] = "conv_dim=4 conv_alpha=1"
    sent = backend._for_trainer(values)
    assert ADAPTER_KEY not in sent
    config = dict(kohya_backend._build_sd_scripts_config(model_key, sent, "lora"))
    assert config["network_args"][0] == "use_dora=True" and "conv_dim=4" in config["network_args"]

    preset = tmp_path / "kohya dora.toml"
    backend.save(str(preset), values)
    saved = toml.load(preset)
    assert ADAPTER_KEY not in saved and "use_dora=True" in str(saved["network_args"])
    loaded = backend.load(str(preset), backend.defaults("lora")).values
    assert loaded[ADAPTER_KEY] == DORA_ADAPTER and "use_dora" not in str(loaded["network_args"])


def test_dora_trains_wan22_high_and_low_noise_together():
    """The trainer keeps one DoRA magnitude per expert (<name>.safetensors for the low-noise model,
    <name>_high_noise.safetensors for the high-noise model), so the UI refuses neither adapter with both DiTs."""
    backend = get_backend("wan")
    values = backend.defaults("lora")
    values["dit_high_noise"] = "C:/models/wan22_high.safetensors"
    for adapter in (DORA_ADAPTER, LORA_ADAPTER):
        values[ADAPTER_KEY] = adapter
        assert not any("high and low noise" in problem for problem in backend.check_values(values, "lora")), adapter
