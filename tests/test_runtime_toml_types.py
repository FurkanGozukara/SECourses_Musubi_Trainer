"""Runtime TOML values reach argparse unconverted, so the GUI must write them in the backend's types.

read_config_from_file builds the namespace straight from the TOML, so a base_weights string made the backend
iterate over its characters, and a quoted path (Explorer's "Copy as path") was looked up with its quotes.
"""

import importlib
import os

import pytest
import toml

from musubi_tuner_gui import image_models_gui, wan_lora_gui
from musubi_tuner_gui.common_gui import SaveConfigFileToRun, split_float_list, split_path_list, strip_path_text


def test_base_weights_are_written_as_lists(tmp_path):
    lora = tmp_path / "my lora.safetensors"
    lora.write_bytes(b"")
    config = tmp_path / "runtime.toml"

    SaveConfigFileToRun([("base_weights", f'"{lora}"'), ("base_weights_multiplier", 1), ("dit", "dit.safetensors")], str(config))
    data = toml.load(config)
    assert data["base_weights"] == [str(lora)]
    assert data["base_weights_multiplier"] == [1.0]

    SaveConfigFileToRun([("base_weights", ""), ("base_weights_multiplier", 1.0), ("dit", "dit.safetensors")], str(config))
    data = toml.load(config)
    assert "base_weights" not in data and "base_weights_multiplier" not in data


def test_path_lists_and_numbers_are_split_like_a_user_types_them(tmp_path):
    existing = tmp_path / "with space.safetensors"
    existing.write_bytes(b"")
    assert split_path_list(str(existing)) == [str(existing)]  # one existing path keeps its spaces
    assert split_path_list('"C:/a b/x.safetensors" "C:/c/y.safetensors"') == ["C:/a b/x.safetensors", "C:/c/y.safetensors"]
    assert split_path_list("C:/a/x.safetensors\nC:/b/y.safetensors") == ["C:/a/x.safetensors", "C:/b/y.safetensors"]
    assert split_float_list("1 0.5") == split_float_list("1,0.5") == split_float_list("1;0.5") == split_float_list([1, "0.5"]) == [1.0, 0.5]
    assert split_float_list("0,5") == [0.5]  # decimal comma (European number settings)
    assert split_float_list("0,5 1") == [0.5, 1.0]
    assert strip_path_text('  "C:\\data\\set"  ') == "C:\\data\\set"


@pytest.mark.parametrize(
    "module_name, opener",
    [
        ("flux_lora_gui", "open_flux_configuration"),
        ("flux2_lora_gui", "open_flux2_configuration"),
        ("zimage_lora_gui", "open_zimage_configuration"),
        ("wan_lora_gui", "open_wan_configuration"),
    ],
)
def test_loaders_show_base_weight_lists_in_the_form_they_parse_back(tmp_path, monkeypatch, module_name, opener):
    # runtime TOMLs store both as lists; the FLUX loaders put the raw lists into their textboxes ("['C:/a', ...]")
    module = importlib.import_module(f"musubi_tuner_gui.{module_name}")
    monkeypatch.setattr(module.gr, "Info", lambda *_args, **_kwargs: None)
    paths = ["C:/a/one.safetensors", "C:/b c/two.safetensors"]
    config = tmp_path / "run.toml"
    config.write_text(toml.dumps({"base_weights": paths, "base_weights_multiplier": [1.0, 0.5]}), encoding="utf-8")
    result = getattr(module, opener)(False, str(config), [("base_weights", ""), ("base_weights_multiplier", "")])
    weights, multipliers = result[-2], result[-1]
    assert weights == "C:/a/one.safetensors; C:/b c/two.safetensors" and multipliers == "1 0.5"
    assert split_path_list(weights) == paths and split_float_list(multipliers) == [1.0, 0.5]


@pytest.mark.parametrize("task, maximum", [("t2v-1.3B", 28), ("t2v-1.3B-FC", 28), ("t2v-14B", 38), ("i2v-A14B", 38)])
def test_wan_refuses_more_block_swap_than_the_model_has_before_caching(task, maximum):
    # the field accepts up to 59; the trainer only failed after the latents had been cached
    wan_lora_gui.validate_wan_block_swap_options({"task": task, "blocks_to_swap": maximum})
    with pytest.raises(ValueError, match=f"at most {maximum} blocks"):
        wan_lora_gui.validate_wan_block_swap_options({"task": task, "blocks_to_swap": maximum + 1})


def test_wan_save_keeps_every_base_weight_multiplier(tmp_path):
    config = tmp_path / "wan.toml"
    wan_lora_gui.save_wan_configuration(False, str(config), [("base_weights_multiplier", [1.0, 0.5]), ("base_weights", ["C:/a.safetensors", "C:/b.safetensors"])])
    data = toml.load(config)
    assert data["base_weights_multiplier"] == [1.0, 0.5] and data["base_weights"] == ["C:/a.safetensors", "C:/b.safetensors"]


def test_quoted_paths_are_cleaned_in_the_runtime_toml_and_dataset_folder(tmp_path):
    config = tmp_path / "runtime.toml"
    SaveConfigFileToRun([("dit", ' "C:/models/dit.safetensors" '), ("output_dir", f'"{tmp_path}"')], str(config))
    data = toml.load(config)
    assert data["dit"] == "C:/models/dit.safetensors" and data["output_dir"] == str(tmp_path)

    dataset = tmp_path / "dataset"
    (dataset / "1_ohwx").mkdir(parents=True)
    (dataset / "1_ohwx" / "a.png").write_bytes(b"")
    spec = image_models_gui.get_spec("qwen_image_21")
    path, _messages = image_models_gui.create_dataset_config(
        spec, {"parent_folder_path": f'"{dataset}"', "output_dir": str(tmp_path), "use_control_images": False}
    )
    assert os.path.isfile(path)
