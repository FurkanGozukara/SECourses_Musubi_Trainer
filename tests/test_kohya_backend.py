"""kohya sd-scripts backend (FLUX.1 Dev / SDXL) of the unified UI: specs, presets, runtime TOML and launch command.

Pure Python, no GPU. The last test runs the real sd-scripts argument parsing and dataset setup in a subprocess with
CUDA hidden (no model is loaded); it is skipped when the tokenizers are not in the Hugging Face cache.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import toml

from musubi_tuner_gui.full_finetune_gui import FULL_FINE_TUNING_MODE, LORA_TRAINING_MODE
from musubi_tuner_gui.preset_library import TIER_PATTERN
from musubi_tuner_gui.unified import kohya_backend as kb
from musubi_tuner_gui.unified.param_spec import ALL_MODES, KINDS, SECTION_IDS

MODELS = ("flux1_dev", "sdxl")
COMBOS = [(model, kind) for model in MODELS for kind in ("lora", "full")]
SCRIPTS = {
    ("flux1_dev", "lora"): "flux_train_network.py",
    ("flux1_dev", "full"): "flux_train.py",
    ("sdxl", "lora"): "sdxl_train_network.py",
    ("sdxl", "full"): "sdxl_train.py",
}
KOHYA_APP = Path("F:/Kohya_FLUX_DreamBooth_LoRA_v31")
JSON_FOLDERS = {
    "FLUX_LoRA_Training_LoRA_Tab_Configs": ("flux1_dev", "lora"),
    "FLUX_DreamBooth_Tab_Fine_Tuning_Training_Configs": ("flux1_dev", "full"),
    "SDXL_DreamBooth_Fine_Tuning_Tab_Configs": ("sdxl", "full"),
}
PRESET_FOLDERS = {
    "FLUX_1_Dev_Training_Configs": "flux1_dev",
    "SDXL_Training_Configs": "sdxl",
}


class _FakeRunner:
    def __init__(self):
        self.calls = []

    def execute_command(self, run_cmd, **kwargs):
        self.calls.append({"run_cmd": run_cmd, **kwargs})


def _write_fake_flux_checkpoint(path: Path) -> None:
    """A tiny safetensors file with the FLUX.1-dev key layout (sd-scripts reads the header to detect dev/schnell)."""
    keys = [f"double_blocks.{i}.img_attn.proj.bias" for i in range(19)]
    keys += [f"single_blocks.{i}.modulation.lin.bias" for i in range(38)]
    keys.append("guidance_in.in_layer.bias")
    header, offset = {}, 0
    for key in keys:
        header[key] = {"dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]}
        offset += 4
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * offset)


def _make_dataset(root: Path, folder: str = "1_ohwx man", count: int = 2) -> Path:
    from PIL import Image

    images = root / folder
    images.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (64, 64 + 32 * index), (40 * index, 90, 140)).save(images / f"img{index}.png")
        (images / f"img{index}.txt").write_text("ohwx man", encoding="utf-8")
    return root


def _values(tmp_path: Path, model: str, kind: str, **overrides) -> dict:
    values = kb.kohya_defaults(model, kind)
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    dataset = _make_dataset(tmp_path / "dataset")
    if model == "flux1_dev":
        _write_fake_flux_checkpoint(models / "flux1-dev.safetensors")
        for name in ("ae.safetensors", "clip_l.safetensors", "t5xxl_fp16.safetensors"):
            (models / name).write_bytes(b"x")
        values.update(
            pretrained_model_name_or_path=str(models / "flux1-dev.safetensors"),
            ae=str(models / "ae.safetensors"),
            clip_l=str(models / "clip_l.safetensors"),
            t5xxl=str(models / "t5xxl_fp16.safetensors"),
        )
    else:
        (models / "sdxl.safetensors").write_bytes(b"x")
        values.update(pretrained_model_name_or_path=str(models / "sdxl.safetensors"))
    values.update(parent_folder_path=str(dataset), output_dir=str(tmp_path / f"out_{model}_{kind}"), output_name=f"test_{model}_{kind}")
    values.update(overrides)
    return values


def _runtime_toml(text: str) -> tuple[str, dict]:
    path = re.search(r"# runtime config: (.+)", text).group(1).strip()
    return path, toml.load(path)


def _command_line(text: str) -> str:
    return text.splitlines()[1]


# ----------------------------------------------------------------------------------------------------------------------
# specs and defaults
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_specs_are_valid(model):
    specs = kb.kohya_param_specs(model)
    keys = [spec.key for spec in specs]
    assert len(keys) == len(set(keys)), "duplicate keys"
    order = {section: index for index, section in enumerate(SECTION_IDS)}
    assert [(order[s.section], s.priority) for s in specs] == sorted((order[s.section], s.priority) for s in specs)
    for spec in specs:
        assert spec.section in SECTION_IDS, spec.key
        assert spec.kind in KINDS, spec.key
        assert spec.label.strip(), spec.key
        assert spec.modes and spec.modes <= ALL_MODES, spec.key
        if spec.choices:
            assert spec.default in spec.choice_values(), (spec.key, spec.default)
        if spec.kind == "file" and spec.key in kb.MODEL_PATH_KEYS:
            assert spec.file_types, spec.key
        if spec.kind in ("number", "slider") and spec.minimum is not None and spec.maximum is not None:
            assert spec.minimum <= spec.default <= spec.maximum, (spec.key, spec.default)
    for kind in ("lora", "full"):
        defaults = kb.kohya_defaults(model, kind)
        assert list(defaults) == keys, "defaults must cover every key (in spec order)"
        assert all(value is not None for value in defaults.values())
        for spec in specs:
            if spec.choices:
                assert defaults[spec.key] in spec.choice_values(), (kind, spec.key, defaults[spec.key])


@pytest.mark.parametrize("model", MODELS)
def test_shared_and_kohya_key_names(model):
    keys = {spec.key for spec in kb.kohya_param_specs(model)}
    shared = {
        "training_mode", "output_dir", "output_name", "learning_rate", "optimizer_type", "optimizer_args", "lr_scheduler",
        "lr_warmup_steps", "max_grad_norm", "max_train_epochs", "max_train_steps", "save_every_n_epochs", "save_precision",
        "seed", "gradient_checkpointing", "gradient_accumulation_steps", "mixed_precision", "full_bf16", "blocks_to_swap",
        "network_dim", "network_alpha", "network_args", "sample_prompts", "parent_folder_path", "dataset_resolution_width",
        "dataset_resolution_height", "dataset_batch_size", "dataset_caption_extension", "dataset_enable_bucket",
        "cache_latents", "cache_text_encoder_outputs", "gpu_ids", "num_processes", "log_with", "logging_dir",
        "additional_parameters", "compile",
    }
    assert shared <= keys, shared - keys
    assert {"pretrained_model_name_or_path", "reg_data_dir", "unet_lr", "text_encoder_lr", "loss_type", "huber_c"} <= keys
    if model == "flux1_dev":
        assert {"ae", "clip_l", "t5xxl", "guidance_scale", "model_prediction_type", "apply_t5_attn_mask", "cpu_offload_checkpointing", "fused_backward_pass"} <= keys
        assert not {"sdpa", "xformers", "vae"} & keys  # no effect on FLUX.1 in sd-scripts
    else:
        assert {"vae", "no_half_vae", "learning_rate_te1", "learning_rate_te2", "noise_offset", "sdpa", "xformers"} <= keys
    by_key = {spec.key: spec for spec in kb.kohya_param_specs(model)}
    assert by_key["network_dim"].modes == frozenset({"lora"})
    assert by_key["fused_backward_pass"].modes == frozenset({"full"})
    assert by_key["blocks_to_swap"].maximum == kb.KOHYA_MODELS[model]["max_blocks_to_swap"]


def test_models_and_default_paths():
    assert set(kb.KOHYA_MODELS) == {"flux1_dev", "sdxl"}
    flux = kb.kohya_defaults("flux1_dev", "lora")
    root = kb.ROOT_DIR.as_posix()
    assert flux["pretrained_model_name_or_path"] == f"{root}/Training_Models_FLUX_1/flux1-dev.safetensors"
    assert flux["t5xxl"].endswith("Training_Models_FLUX_1/t5xxl_fp16.safetensors")
    sdxl = kb.kohya_defaults("sdxl", "full")
    assert sdxl["pretrained_model_name_or_path"] == f"{root}/Training_Models_SDXL/SDXL_RealVis_v5_FP32.safetensors"
    assert sdxl["vae"] == "stabilityai/sdxl-vae"
    assert kb.KOHYA_MODELS["flux1_dev"]["train_scripts"] == {"lora": "flux_train_network.py", "full": "flux_train.py"}
    assert kb.KOHYA_MODELS["sdxl"]["train_scripts"] == {"lora": "sdxl_train_network.py", "full": "sdxl_train.py"}
    # values of the measured 24 GB presets
    assert (flux["learning_rate"], flux["blocks_to_swap"], flux["network_dim"], flux["max_grad_norm"]) == (5e-05, 16, 128, 1.0)
    assert flux["block_swap_h2d_only"] is True  # the 24 GB LoRA preset streams its 16 blocks H2D-only
    flux_full = kb.kohya_defaults("flux1_dev", "full")
    assert (flux_full["learning_rate"], flux_full["blocks_to_swap"], flux_full["fused_backward_pass"]) == (4e-06, 10, True)
    assert flux_full["block_swap_h2d_only"] is False
    assert (sdxl["learning_rate"], sdxl["learning_rate_te1"], sdxl["max_grad_norm"]) == (8e-06, 3e-06, 0.0)
    assert kb.kohya_defaults("sdxl", "lora")["text_encoder_lr"] == 0.0  # U-Net only, so TE outputs can be cached


def test_hints_cover_every_model_and_type():
    for model in MODELS:
        for kind in ("lora", "full"):
            tips = kb.KOHYA_HINTS[model][kind]
            assert 4 <= len(tips) <= 8 and all(tip.strip() for tip in tips)


# ----------------------------------------------------------------------------------------------------------------------
# kohya_ss JSON conversion and shipped presets
# ----------------------------------------------------------------------------------------------------------------------


def _kohya_jsons():
    found = []
    for folder, (model, kind) in JSON_FOLDERS.items():
        for path in sorted((KOHYA_APP / folder).glob("*.json")):
            found.append(pytest.param(path, model, kind, id=f"{folder}/{path.name}"))
    return found


KOHYA_JSONS = _kohya_jsons()


@pytest.mark.skipif(not KOHYA_JSONS, reason="kohya_ss JSON presets not found next to this machine")
@pytest.mark.parametrize("path,model,kind", KOHYA_JSONS)
def test_every_kohya_json_preset_converts(path, model, kind):
    data = json.loads(path.read_text(encoding="utf-8"))
    values, notes = kb.convert_kohya_gui_json(data, model, kind)
    specs = {spec.key: spec for spec in kb.kohya_param_specs(model)}
    assert set(values) <= set(specs)
    assert values["training_mode"] == (LORA_TRAINING_MODE if kind == "lora" else FULL_FINE_TUNING_MODE)
    for key, value in values.items():
        spec = specs[key]
        if spec.choices and not spec.allow_custom_value:
            assert value in spec.choice_values(), (key, value)
        if spec.kind == "checkbox":
            assert isinstance(value, bool), key
    assert all(isinstance(note, str) for note in notes)
    # the kohya_ss GUI ran the steps of `epoch` epochs on every GPU: N GPUs trained epoch x N epochs of the dataset
    assert values["max_train_epochs"] == data["epoch"] * max(1, int(data.get("num_processes") or 1))
    assert kb.kohya_preset_file_name(path.name).endswith(".toml")
    assert TIER_PATTERN.search(kb.kohya_preset_file_name(path.name))
    if kind == "lora":
        assert values["network_module"] == "networks.lora_flux"
        assert values["max_grad_norm"] == 1.0  # the LoRA tab dropped 0 -> sd-scripts default 1.0
        assert values["text_encoder_lr"] == data["text_encoder_lr"]
    if model == "sdxl":
        assert values["max_grad_norm"] == 0.0 and values["no_half_vae"] is True  # folded from additional_parameters
        assert values["additional_parameters"] == ""
        assert values["blocks_to_swap"] == data["sdxl_blocks_to_swap"]


def test_preset_file_names():
    assert kb.kohya_preset_file_name("24GB_GPUs_21.61GB_Quality_1_LoRA.json") == "24_GB_GPUs_21.61_GB_Quality_1.toml"
    assert kb.kohya_preset_file_name("10GB_GPUs_8.75GB_FP8_Scaled_Quality_2_LoRA.json") == "10_GB_GPUs_8.75_GB_FP8_Scaled_Quality_2.toml"
    assert kb.kohya_preset_file_name("48GB_GPUs_40.63GB_VRAM_Batch_Size_7_Quality_2.json") == "48_GB_GPUs_40.63_GB_Batch_Size_7_Quality_2.toml"
    assert kb.kohya_preset_file_name("32GB_GPUs_29.33GB_Quality_2_LoRA_4x_GPU_Config.json") == "32_GB_GPUs_29.33_GB_Quality_2_4x_GPU_Config.toml"


def _shipped_presets():
    found = []
    root = kb.ROOT_DIR / "Training_Presets"
    for folder, model in PRESET_FOLDERS.items():
        for path in sorted(glob.glob(str(root / folder / "**" / "*.toml"), recursive=True)):
            found.append(pytest.param(path, model, id=f"{folder}/{Path(path).parent.name}/{Path(path).name}"))
    return found


SHIPPED = _shipped_presets()


@pytest.mark.skipif(not SHIPPED, reason="kohya presets not converted yet")
@pytest.mark.parametrize("path,model", SHIPPED)
def test_shipped_kohya_presets_load(path, model):
    name = os.path.basename(path)
    assert TIER_PATTERN.search(name), name
    with open(path, "r", encoding="utf-8") as handle:
        assert handle.readline().startswith("# description:")
    data = toml.load(path)
    assert data["preset_category"] == "image" and data["preset_model"] == model and data["preset_format"] == 1
    keys = [spec.key for spec in kb.kohya_param_specs(model)]
    assert set(keys) <= set(data)
    assert not data["parent_folder_path"] and not data["dataset_config"] and not data["reg_data_dir"]
    folder_mode = FULL_FINE_TUNING_MODE if "Fine_Tuning" in path.replace("\\", "/") else LORA_TRAINING_MODE
    assert data["training_mode"] == folder_mode
    assert data["preset_training_type"] == ("full" if folder_mode == FULL_FINE_TUNING_MODE else "lora")
    other = kb.kohya_defaults(model, "full" if folder_mode == LORA_TRAINING_MODE else "lora")
    values, notes = kb.load_kohya_preset(path, model, other)
    assert values["training_mode"] == folder_mode
    assert set(values) == set(keys)


# ----------------------------------------------------------------------------------------------------------------------
# training command and runtime TOML
# ----------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model,kind", COMBOS)
def test_print_only_command(tmp_path, model, kind):
    values = _values(tmp_path, model, kind)
    text = kb.train_kohya_model(model, values, print_only=True, runner=None)
    command = _command_line(text)
    assert "accelerate" in command and " launch " in f" {command} "
    script = (kb.SD_SCRIPTS_DIR / SCRIPTS[(model, kind)]).as_posix()
    assert script in command
    assert "--config_file" in command
    path, data = _runtime_toml(text)
    assert Path(path).parent == Path(values["output_dir"])
    assert command.split("--config_file", 1)[1].strip().startswith(path)
    assert data["training_mode"] == (LORA_TRAINING_MODE if kind == "lora" else FULL_FINE_TUNING_MODE)
    assert data["train_data_dir"] == Path(values["parent_folder_path"]).as_posix()
    assert data["resolution"] == "1024,1024"
    assert data["output_name"] == values["output_name"]
    assert data["console_log_simple"] is True  # plain log lines for the training status panel
    assert data["enable_multiline_caption"] is True  # whole caption files, as the musubi trainers read them
    if kind == "lora":
        assert data["network_module"] == kb.KOHYA_MODELS[model]["network_module"]
        assert data["network_dim"] == values["network_dim"]
    else:
        assert "network_module" not in data and "network_dim" not in data
    if model == "flux1_dev":
        assert {"ae", "clip_l", "t5xxl"} <= set(data) and data["guidance_scale"] == 1.0
        assert "--mixed_precision bf16" in command
    if (model, kind) == ("flux1_dev", "lora"):
        assert data["text_encoder_lr"] == [5e-05, 5e-05] and "network_train_unet_only" not in data
        assert data["save_precision"] == "float"
    if (model, kind) == ("sdxl", "lora"):
        assert data["network_train_unet_only"] is True and data["cache_text_encoder_outputs"] is True
    if (model, kind) == ("sdxl", "full"):
        assert data["train_text_encoder"] is True and data["learning_rate_te1"] == 3e-06 and data["max_grad_norm"] == 0.0
        assert data["vae"] == "stabilityai/sdxl-vae" and data["no_half_vae"] is True


def test_dataset_toml_and_regularization(tmp_path):
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.write_text("[[datasets]]\n[[datasets.subsets]]\nimage_dir = 'x'\n", encoding="utf-8")
    values = _values(tmp_path, "sdxl", "full", dataset_config_mode=kb.TOML_DATASET_MODE, dataset_config=str(dataset_toml))
    _path, data = _runtime_toml(kb.train_kohya_model("sdxl", values, print_only=True))
    assert data["dataset_config"] == dataset_toml.as_posix() and "train_data_dir" not in data
    assert "enable_multiline_caption" not in data  # the user's dataset TOML decides it per subset

    reg = _make_dataset(tmp_path / "reg", "1_man")
    values = _values(tmp_path, "flux1_dev", "full", reg_data_dir=str(reg))
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["reg_data_dir"] == reg.as_posix() and data["prior_loss_weight"] == 1.0


def test_logging_paths_follow_guardrail_1(tmp_path):
    values = _values(tmp_path, "flux1_dev", "lora")
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert "logging_dir" not in data and "log_with" not in data

    for logging_dir in ("", "C:/", "/"):
        values = _values(tmp_path, "flux1_dev", "lora", log_with="tensorboard", logging_dir=logging_dir)
        _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
        assert data["log_with"] == "tensorboard"
        assert Path(data["logging_dir"]) == Path(values["output_dir"]) / "logs"

    values = _values(tmp_path, "sdxl", "lora", log_with="", additional_parameters="--log_with tensorboard --max_token_length 225")
    text = kb.train_kohya_model("sdxl", values, print_only=True)
    _path, data = _runtime_toml(text)
    assert data["log_with"] == "tensorboard" and Path(data["logging_dir"]) == Path(values["output_dir"]) / "logs"
    command = _command_line(text)
    assert "--log_with" not in command and command.endswith("--max_token_length 225")

    values = _values(tmp_path, "sdxl", "lora", log_with="wandb", logging_dir="runs")
    _path, data = _runtime_toml(kb.train_kohya_model("sdxl", values, print_only=True))
    assert Path(data["logging_dir"]) == Path(values["output_dir"]) / "runs"
    assert data["wandb_run_name"] == values["output_name"]


@pytest.mark.parametrize("model", MODELS)
def test_training_mode_round_trip(tmp_path, model):
    """Guardrail 2: the runtime TOML keeps training_mode, and a TOML without it is inferred from network_module."""
    lora_current = kb.kohya_defaults(model, "lora")
    full_current = kb.kohya_defaults(model, "full")

    values = _values(tmp_path, model, "full")
    path, data = _runtime_toml(kb.train_kohya_model(model, values, print_only=True))
    assert data["training_mode"] == FULL_FINE_TUNING_MODE and "network_module" not in data
    loaded, _notes = kb.load_kohya_preset(path, model, lora_current)
    assert loaded["training_mode"] == FULL_FINE_TUNING_MODE
    assert loaded["parent_folder_path"] == values["parent_folder_path"].replace("\\", "/")
    assert loaded["learning_rate"] == values["learning_rate"]

    legacy = tmp_path / "legacy_full.toml"
    legacy.write_text("\n".join(line for line in Path(path).read_text(encoding="utf-8").splitlines() if not line.startswith("training_mode")), encoding="utf-8")
    assert "training_mode" not in toml.load(legacy)
    loaded, _notes = kb.load_kohya_preset(str(legacy), model, lora_current)
    assert loaded["training_mode"] == FULL_FINE_TUNING_MODE

    values = _values(tmp_path, model, "lora")
    path, data = _runtime_toml(kb.train_kohya_model(model, values, print_only=True))
    lora_legacy = tmp_path / "legacy_lora.toml"
    lora_legacy.write_text("\n".join(line for line in Path(path).read_text(encoding="utf-8").splitlines() if not line.startswith("training_mode")), encoding="utf-8")
    loaded, _notes = kb.load_kohya_preset(str(lora_legacy), model, full_current)
    assert loaded["training_mode"] == LORA_TRAINING_MODE
    assert loaded["network_dim"] == values["network_dim"]
    assert (loaded["dataset_resolution_width"], loaded["dataset_resolution_height"]) == (1024, 1024)


def test_reload_runtime_toml_restores_omitted_options(tmp_path):
    """Runtime TOMLs leave disabled options out: reloading one must not keep stale on-screen values for them."""
    values = _values(tmp_path, "flux1_dev", "lora", fp8_scaled=False, compile=False, blocks_to_swap=0, seed=7)
    path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert "fp8_scaled" not in data and "compile" not in data and "blocks_to_swap" not in data
    stale = kb.kohya_defaults("flux1_dev", "lora")
    stale.update(fp8_scaled=True, compile=True, blocks_to_swap=20, gradient_checkpointing=False, sample_at_first=True)
    loaded, _notes = kb.load_kohya_preset(path, "flux1_dev", stale)
    assert (loaded["fp8_scaled"], loaded["compile"], loaded["blocks_to_swap"]) == (False, False, 0)
    assert loaded["gradient_checkpointing"] is True and loaded["sample_at_first"] is False and loaded["seed"] == 7

    # a kohya_ss GUI runtime TOML (LoRA tab): max_grad_norm 0 was dropped -> sd-scripts default 1.0
    kohya_toml = tmp_path / "config_lora-20251124-013120.toml"
    kohya_toml.write_text(
        'network_module = "networks.lora_flux"\nnetwork_dim = 64\nnetwork_alpha = 64\nlearning_rate = 5e-5\n'
        'text_encoder_lr = [ 5e-5, 5e-5,]\nresolution = "768,768"\ntrain_data_dir = "C:/data"\nmax_train_steps = 280\n'
        'save_precision = "float"\nnetwork_args = [ "train_double_block_indices=0-9", "train_single_block_indices=all",]\n',
        encoding="utf-8",
    )
    loaded, notes = kb.load_kohya_preset(str(kohya_toml), "flux1_dev", stale)
    assert loaded["training_mode"] == LORA_TRAINING_MODE
    assert (loaded["dataset_resolution_width"], loaded["max_train_epochs"], loaded["max_train_steps"]) == (768, 0, 280)
    assert loaded["max_grad_norm"] == 1.0 and loaded["text_encoder_lr"] == 5e-05 and loaded["save_precision"] == "fp32"
    assert loaded["train_double_block_indices"] == "0-9" and loaded["network_args"] == ""
    assert loaded["parent_folder_path"] == "C:/data" and loaded["dataset_config_mode"] == kb.FOLDER_DATASET_MODE


def test_load_raw_kohya_json_and_missing_keys_keep_current(tmp_path):
    data = {"LoRA_type": "Flux1", "learning_rate": 1e-4, "unet_lr": 1e-4, "text_encoder_lr": 0, "network_dim": 32, "epoch": 20,
            "max_resolution": "768,512", "optimizer": "AdamW8bit", "save_precision": "bf16", "xformers": "sdpa", "max_grad_norm": 0}
    path = tmp_path / "my_kohya.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    current = kb.kohya_defaults("flux1_dev", "full")
    current["seed"] = 1234
    values, notes = kb.load_kohya_preset(str(path), "flux1_dev", current)
    assert values["training_mode"] == LORA_TRAINING_MODE
    assert (values["network_dim"], values["max_train_epochs"], values["optimizer_type"]) == (32, 20, "AdamW8bit")
    assert (values["dataset_resolution_width"], values["dataset_resolution_height"]) == (768, 512)
    assert values["unet_lr"] == 0.0 and values["text_encoder_lr"] == 0.0 and values["max_grad_norm"] == 1.0
    assert values["seed"] == 1234  # not in the file
    assert any("xformers" in note for note in notes)


def test_relocate_model_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(kb, "APP_DIR", tmp_path / "SECourses_Musubi_Trainer")
    monkeypatch.setattr(kb, "_training_model_directories", lambda: [])
    flux_dir = tmp_path / "Training_Models_FLUX_1"
    flux_dir.mkdir()
    (flux_dir / "flux1-dev.safetensors").write_bytes(b"x")
    legacy = tmp_path / "SDXL_Training_Models"
    legacy.mkdir()
    (legacy / "SDXL_RealVis_v5_FP32.safetensors").write_bytes(b"x")
    for folder in (flux_dir, legacy):
        (folder / "ae.safetensors").write_bytes(b"x")  # ambiguous name
    values = {
        # a path of the machine the preset was made on (never present here, also not on the Linux server)
        "pretrained_model_name_or_path": "/no/such/machine/Downloads/flux1-dev.safetensors",
        "ae": "/no/such/machine/Downloads/ae.safetensors",
        "clip_l": "/somewhere/else/clip_l.safetensors",
        "vae": "stabilityai/sdxl-vae",
        "learning_rate": 1e-4,
    }
    result = kb.relocate_model_paths(values)
    assert result["pretrained_model_name_or_path"] == (flux_dir / "flux1-dev.safetensors").resolve().as_posix()
    assert result["ae"] == values["ae"]  # two matches: never guess
    assert result["clip_l"] == values["clip_l"]  # nothing found
    assert result["vae"] == "stabilityai/sdxl-vae" and result["learning_rate"] == 1e-4
    relative = kb.relocate_model_paths({"pretrained_model_name_or_path": "Training_Models_SDXL/SDXL_RealVis_v5_FP32.safetensors"})
    assert relative["pretrained_model_name_or_path"] == (legacy / "SDXL_RealVis_v5_FP32.safetensors").resolve().as_posix()


def test_runner_launch_is_non_blocking(tmp_path):
    runner = _FakeRunner()
    values = _values(tmp_path, "flux1_dev", "lora", compile=False)
    status = kb.train_kohya_model("flux1_dev", values, print_only=False, runner=runner)
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert isinstance(call["run_cmd"], list) and isinstance(call["env"], dict)
    assert any(part.endswith("flux_train_network.py") for part in call["run_cmd"])
    config_path = call["run_cmd"][call["run_cmd"].index("--config_file") + 1]
    assert os.path.isfile(config_path) and "training_mode" in toml.load(config_path)
    assert "sd-scripts" in call["env"]["PYTHONPATH"]
    assert "flux_train_network.py" in status and config_path in status

    with pytest.raises(ValueError, match="runner"):
        kb.train_kohya_model("flux1_dev", values, print_only=False, runner=None)


def test_sample_prompt_text_is_written_to_a_file(tmp_path):
    values = _values(tmp_path, "sdxl", "lora", sample_prompts="a photo of ohwx man --w 1024 --h 1024", sample_every_n_epochs=5)
    _path, data = _runtime_toml(kb.train_kohya_model("sdxl", values, print_only=True))
    prompt_file = Path(data["sample_prompts"])
    assert prompt_file == Path(values["output_dir"]) / "sample" / "prompt.txt"
    assert "ohwx man" in prompt_file.read_text(encoding="utf-8") and data["sample_every_n_epochs"] == 5


@pytest.mark.parametrize(
    "model,kind,overrides,message",
    [
        ("flux1_dev", "lora", {"parent_folder_path": "Z:/does/not/exist"}, "Training Images Folder does not exist"),
        ("flux1_dev", "lora", {"t5xxl": "Z:/missing/t5xxl_fp16.safetensors"}, "T5-XXL file not found"),
        ("flux1_dev", "full", {"optimizer_type": "AdamW"}, "Fused Backward Pass works only with the Adafactor"),
        ("flux1_dev", "lora", {"blocks_to_swap": 13, "cpu_offload_checkpointing": True}, "CPU Offload Checkpointing"),
        ("flux1_dev", "lora", {"blocks_to_swap": 36}, "Blocks to Swap must be between 0 and 35"),
        ("flux1_dev", "lora", {"blocks_to_swap": 13, "block_swap_h2d_only": True, "gradient_checkpointing": False}, "H2D-only Block Swap needs Gradient Checkpointing"),
        ("sdxl", "lora", {"text_encoder_lr": 1e-5}, "Text Encoder LR = 0"),
        ("sdxl", "full", {"cache_text_encoder_outputs": True}, "cannot be trained while their outputs are cached"),
        ("sdxl", "full", {"full_bf16": True, "mixed_precision": "fp16"}, "Full BF16 Training needs Mixed Precision 'bf16'"),
        ("sdxl", "lora", {"max_train_epochs": 0, "max_train_steps": 0}, "Max Training Epochs or Max Training Steps"),
        ("sdxl", "lora", {"shuffle_caption": True}, "Caption shuffle"),
        ("sdxl", "lora", {"dataset_resolution_width": 1000}, "multiple of 32"),
        ("sdxl", "lora", {"sample_every_n_steps": 100, "sample_prompts": ""}, "Sample Prompts is empty"),
        ("flux1_dev", "lora", {"network_alpha": -1}, "Network Alpha cannot be negative"),
        # sd-scripts builds the 'adafactor' scheduler only for Adafactor with relative_step (IndexError otherwise)
        ("flux1_dev", "lora", {"lr_scheduler": "adafactor"}, "needs relative_step=True"),
        ("sdxl", "lora", {"lr_scheduler": "adafactor", "optimizer_type": "AdamW", "optimizer_args": ""}, "works only with the Adafactor optimizer"),
        # what sd-scripts fails on only after loading the models (the kohya_ss GUI refused malformed arguments first);
        # "weight_decay = 0.01", "betas=0.9, 0.999" and "conv_dim = 4" are read as meant (test_argument_lists_are_cleaned)
        ("flux1_dev", "lora", {"optimizer_args": "foo='a=b'"}, "one '=' each"),
        ("flux1_dev", "lora", {"optimizer_args": "weight_decay=0.01 oops"}, "Not usable: oops"),
        ("flux1_dev", "lora", {"lr_scheduler_type": "CyclicLR", "lr_scheduler_args": "mode=triangular2"}, "Not readable: mode=triangular2"),
        ("sdxl", "lora", {"network_args": "conv_dim=4 =3"}, "Network Arguments: write key=value items"),
        ("flux1_dev", "lora", {"lr_warmup_steps": 10}, "LR Warmup Steps must be 0 with the constant LR Scheduler"),
        ("flux1_dev", "lora", {"lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 10, "optimizer_args": ""}, "Adafactor's relative_step"),
        ("flux1_dev", "lora", {"lr_scheduler_type": "CosineAnnealingLR", "lr_scheduler_args": "T_max=100", "lr_warmup_steps": 0.05}, "Custom Scheduler Class"),
        ("sdxl", "full", {"lr_scheduler": "cosine_with_min_lr"}, "needs one minimum"),
        ("sdxl", "full", {"lr_scheduler": "cosine_with_min_lr", "lr_scheduler_args": "min_lr_rate=0.1"}, "min_lr_rate clashes"),
        ("sdxl", "full", {"lr_scheduler": "piecewise_constant"}, "needs step_rules"),
        ("sdxl", "full", {"lr_scheduler": "piecewise_constant", "lr_scheduler_args": "step_rules=0.5"}, "step_rules must be text in quotes"),
        ("flux1_dev", "lora", {"resume": "Z:/nope/x-state"}, "Resume From State Folder not found"),
        ("flux1_dev", "lora", {"resume": "Z:/nope/x-state", "resume_from_huggingface": "me/repo/state"}, "not both"),
        ("flux1_dev", "lora", {"resume_from_huggingface": "True"}, "Resume From Hugging Face must be repo_id/path_in_repo"),
        ("flux1_dev", "lora", {"network_weights": "Z:/nope/x.safetensors"}, "Continue From LoRA file not found"),
        ("flux1_dev", "lora", {"dim_from_weights": True}, "Rank From LoRA File needs"),
        ("sdxl", "full", {"log_tracker_config": "Z:/nope/tracker.toml"}, "Tracker Config file not found"),
        ("flux1_dev", "lora", {"text_encoder_lr": 0, "additional_parameters": "--network_train_text_encoder_only"}, "trains only the text encoder"),
        ("flux1_dev", "lora", {"network_module": "networks.oft_flux", "network_alpha": 0}, "Network Alpha 0 trains nothing with OFT"),
        ("sdxl", "lora", {"network_module": "networks.oft", "network_alpha": 0}, "Network Alpha 0 trains nothing with OFT"),
        ("flux1_dev", "lora", {"network_args": "train_t5xxl=True"}, "trains T5-XXL, so its outputs cannot be cached"),
        ("flux1_dev", "lora", {"t5xxl_lr": 1e-5, "text_encoder_lr": 0}, "T5-XXL LR needs a Text Encoder LR"),
        ("flux1_dev", "lora", {"sample_prompts": "Z:/nope/prompts.txt", "sample_every_n_epochs": 5}, "Sample prompts file not found"),
        ("flux1_dev", "lora", {"network_dim": "16.5"}, "Network Rank (Dim) must be a whole number (got 16.5)."),
        ("sdxl", "full", {"max_train_epochs": "ten"}, "Max Training Epochs: 'ten' is not a number."),
    ],
)
def test_validation_errors(tmp_path, model, kind, overrides, message):
    values = _values(tmp_path, model, kind, **overrides)
    with pytest.raises(ValueError, match=re.escape(message)):
        kb.train_kohya_model(model, values, print_only=True)
    assert any(message in problem for problem in kb.kohya_validation_problems(model, values))


def test_argument_lists_are_cleaned(tmp_path):
    """Spaces around '=', commas, new lines and lowercase booleans reach sd-scripts as the key=value items it reads."""
    values = _values(
        tmp_path, "flux1_dev", "lora",
        optimizer_type="AdamW",
        optimizer_args="weight_decay = 0.01,\nbetas=0.9, 0.999  amsgrad=false",
        network_args="conv_dim = 4; train_blocks=single",
    )
    assert kb.kohya_validation_problems("flux1_dev", values) == []
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["optimizer_args"] == ["weight_decay=0.01", "betas=0.9,0.999", "amsgrad=False"]
    assert "conv_dim=4" in data["network_args"] and "train_blocks=single" in data["network_args"]


def test_valid_settings_that_look_unusual(tmp_path):
    """Network Alpha 0 is sd-scripts' "alpha = rank"; the 'adafactor' scheduler is fine with relative_step=True."""
    values = _values(tmp_path, "flux1_dev", "lora", network_alpha=0)
    assert kb.kohya_validation_problems("flux1_dev", values) == []
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["network_alpha"] == 0.0
    values = _values(tmp_path, "flux1_dev", "lora", lr_scheduler="adafactor", optimizer_args="scale_parameter=True relative_step=True warmup_init=False")
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["lr_scheduler"] == "adafactor"
    assert kb._adafactor_relative_step("relative_step=False warmup_init=True") and not kb._adafactor_relative_step("relative_step=False")


def test_argument_lists_keep_commas_inside_values(tmp_path):
    """sd-scripts reads every key=value with ast.literal_eval: betas=0.9,0.999 must stay one (tuple) argument."""
    split = kb._split_key_value_args
    assert split("weight_decay=0.01 betas=0.9,0.999 eps=1e-8") == ["weight_decay=0.01", "betas=0.9,0.999", "eps=1e-8"]
    assert split("weight_decay=0.01, betas=0.9,0.999,eps=1e-8") == ["weight_decay=0.01", "betas=0.9,0.999", "eps=1e-8"]
    assert split("T_max=100, eta_min=1e-7,") == ["T_max=100", "eta_min=1e-7"]
    assert split("train_double_block_indices=0,1,2 loraplus_lr_ratio=4") == ["train_double_block_indices=0,1,2", "loraplus_lr_ratio=4"]
    values = _values(
        tmp_path,
        "flux1_dev",
        "lora",
        optimizer_type="AdamW",
        optimizer_args="weight_decay=0.01, betas=0.9,0.999 eps=1e-8",
        lr_scheduler_args="foo=1, bar=2",
        network_args="loraplus_lr_ratio=4, verbose=True",
    )
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["optimizer_args"] == ["weight_decay=0.01", "betas=0.9,0.999", "eps=1e-8"]
    assert data["lr_scheduler_args"] == ["foo=1", "bar=2"]
    assert data["network_args"] == ["loraplus_lr_ratio=4", "verbose=True"]


def test_quoted_and_url_paths_reach_sd_scripts_clean(tmp_path):
    """Explorer's "Copy as path", curly quotes, backticks and file:/// URLs: the same cleaning as every other path field."""
    dataset = _make_dataset(tmp_path / "my data set")
    url = "file:///" + dataset.as_posix().replace(" ", "%20")
    for typed in (f'"{dataset}"', f"\u201c{dataset}\u201d", f"`{dataset}`", url):
        values = _values(tmp_path, "flux1_dev", "lora", parent_folder_path=typed, output_dir=f"\u2018{tmp_path / 'out dir'}\u2019")
        assert kb.kohya_validation_problems("flux1_dev", values) == []
        _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
        assert data["train_data_dir"] == dataset.as_posix()
        assert data["output_dir"] == (tmp_path / "out dir").as_posix()


def test_kohya_lora_tab_learning_rates_train_the_same_parts():
    """lora_gui.py: unet_lr and text_encoder_lr both 0 train both parts at learning_rate; only unet_lr 0 trains only the
    text encoder. Here Text Encoder LR 0 means transformer only, and text-encoder-only training needs the sd-scripts flag."""
    base = {"LoRA_type": "Flux1", "network_dim": 16, "epoch": 10, "optimizer": "AdamW", "max_resolution": "1024,1024"}
    cases = [
        ((1e-4, 0, 0), 1e-4, ""),  # both parts at learning_rate
        ((1e-4, 0, 5e-5), 5e-5, "--network_train_text_encoder_only"),
        ((0, 0, 5e-5), 5e-5, "--network_train_text_encoder_only"),
        ((1e-4, 1e-4, 0), 0.0, ""),  # transformer only
        ((1e-4, 1e-4, 5e-5), 5e-5, ""),  # both, own rates
    ]
    for (learning_rate, unet_lr, text_encoder_lr), expected_te, expected_args in cases:
        data = dict(base, learning_rate=learning_rate, unet_lr=unet_lr, text_encoder_lr=text_encoder_lr)
        values, _notes = kb.convert_kohya_gui_json(data, "flux1_dev", "lora")
        assert values["text_encoder_lr"] == expected_te, data
        assert values["additional_parameters"] == expected_args, data


def test_reload_kohya_runtime_toml_that_trained_only_the_text_encoder(tmp_path):
    kohya_toml = tmp_path / "config_lora-20260101-000000.toml"
    kohya_toml.write_text(
        'network_module = "networks.lora_flux"\nnetwork_dim = 16\nnetwork_alpha = 16\nlearning_rate = 1e-4\n'
        'text_encoder_lr = [ 5e-5, 5e-5,]\nnetwork_train_text_encoder_only = true\nresolution = "1024,1024"\n'
        'train_data_dir = "C:/data"\nmax_train_steps = 280\n',
        encoding="utf-8",
    )
    current = kb.kohya_defaults("flux1_dev", "lora")
    current["additional_parameters"] = "--max_token_length 225"
    loaded, notes = kb.load_kohya_preset(str(kohya_toml), "flux1_dev", current)
    assert loaded["text_encoder_lr"] == 5e-05
    assert loaded["additional_parameters"] == "--max_token_length 225 --network_train_text_encoder_only"
    assert any("only the text encoder" in note for note in notes)


def test_multi_gpu_epochs_match_the_kohya_gui():
    """The kohya_ss GUI turned `epoch` into steps for one GPU and ran them on every GPU (lora_gui.py / dreambooth_gui.py
    max_train_steps); sd-scripts' epochs count passes over the whole dataset, so 4 GPUs x epoch 50 = 200 epochs."""
    base = {"LoRA_type": "Flux1", "network_dim": 16, "learning_rate": 1e-4, "unet_lr": 1e-4, "text_encoder_lr": 0, "epoch": 50}
    values, notes = kb.convert_kohya_gui_json(dict(base, num_processes=4), "flux1_dev", "lora")
    assert (values["max_train_epochs"], values["max_train_steps"]) == (200, 0) and any("4 GPUs" in note for note in notes)
    values, _notes = kb.convert_kohya_gui_json(dict(base, num_processes=1), "flux1_dev", "lora")
    assert (values["max_train_epochs"], values["max_train_steps"]) == (50, 0)
    # no steps were computed for a dataset TOML (or epoch 0): sd-scripts trained its default 1600 steps
    values, notes = kb.convert_kohya_gui_json(dict(base, num_processes=4, dataset_config="C:/data/dataset.toml"), "flux1_dev", "lora")
    assert (values["max_train_epochs"], values["max_train_steps"]) == (0, 1600) and any("1600" in note for note in notes)
    values, _notes = kb.convert_kohya_gui_json(dict(base, epoch=0, max_train_steps=500), "flux1_dev", "lora")
    assert (values["max_train_epochs"], values["max_train_steps"]) == (0, 500)
    for path in glob.glob(str(kb.ROOT_DIR / "Training_Presets" / "FLUX_1_Dev_Training_Configs" / "**" / "*4x_GPU*.toml"), recursive=True):
        data = toml.load(path)
        assert data["num_processes"] == 4 and data["max_train_epochs"] == 200, path


def test_dataset_folder_needs_kohya_subfolders(tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "img.png").write_bytes(b"x")
    values = _values(tmp_path, "flux1_dev", "lora", parent_folder_path=str(flat))
    with pytest.raises(ValueError, match="<repeats>_<name>"):
        kb.train_kohya_model("flux1_dev", values, print_only=True)


def test_dataset_folders_follow_sd_scripts_repeats(tmp_path):
    """sd-scripts reads int() of the text before the first '_' (config_util): '10' and '1_' train, '0_x' and '_1' do not."""
    for name in ("10", "1_", "2_ohwx_man"):
        root = _make_dataset(tmp_path / f"ok {name}", name)
        assert kb._kohya_subfolders(str(root)) == ([name], 2)
    for name in ("0_x", "_1", "x_10"):
        root = _make_dataset(tmp_path / f"bad {name}", name)
        assert kb._kohya_subfolders(str(root)) == ([], 0)
    values = _values(tmp_path, "flux1_dev", "lora", parent_folder_path=str(_make_dataset(tmp_path / "plain", "10")))
    assert kb.kohya_validation_problems("flux1_dev", values) == []


# ----------------------------------------------------------------------------------------------------------------------
# second review: arguments as sd-scripts reads them, preflight, conversion and reload
# ----------------------------------------------------------------------------------------------------------------------


def test_argument_items_keep_what_sd_scripts_reads():
    """network_reg_lrs patterns stay one item (sd-scripts' own example), the quotes ast.literal_eval needs stay, and one
    layer of quotes around a whole item goes. Network arguments are plain text to the modules: double quotes go."""
    split = kb._split_key_value_args
    reg_lrs = r"network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3"
    assert split(f"{reg_lrs} loraplus_lr_ratio=4") == [reg_lrs, "loraplus_lr_ratio=4"]
    assert split("network_reg_dims=img_attn=8,txt_attn=4 verbose=True") == ["network_reg_dims=img_attn=8,txt_attn=4", "verbose=True"]
    assert split("lr=1,foo='a,b=c',x=2") == ["lr=1", "foo='a,b=c'", "x=2"]
    assert split("step_rules='1:10,0.1:20,0.01' mode=\"triangular2\"") == ["step_rules='1:10,0.1:20,0.01'", 'mode="triangular2"']
    assert split('"betas=(0.9, 0.999)" eps=1e-8') == ["betas=(0.9, 0.999)", "eps=1e-8"]
    assert split('conv_dim="8" conv_alpha="4",train_t5xxl="True"', raw_values=True) == ["conv_dim=8", "conv_alpha=4", "train_t5xxl=True"]
    assert kb._join_args(["betas=(0.9, 0.999)", "eps=1e-8", 'mode="a b"']) == "\"betas=(0.9, 0.999)\" eps=1e-8 'mode=\"a b\"'"
    assert split(kb._join_args(["betas=(0.9, 0.999)", "eps=1e-8", 'mode="a b"'])) == ["betas=(0.9, 0.999)", "eps=1e-8", 'mode="a b"']


def test_runtime_toml_keeps_argument_quotes_and_reloads_items_with_spaces(tmp_path):
    values = _values(
        tmp_path,
        "flux1_dev",
        "lora",
        optimizer_type="AdamW",
        optimizer_args='"betas=(0.9, 0.999)" weight_decay=0.01',
        lr_scheduler="piecewise_constant",
        lr_scheduler_args="step_rules='1:10,0.1:20,0.01'",
        network_args=r'network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3 conv_dim="4"',
    )
    path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["optimizer_args"] == ["betas=(0.9, 0.999)", "weight_decay=0.01"]
    assert data["lr_scheduler_args"] == ["step_rules='1:10,0.1:20,0.01'"]
    assert data["network_args"] == [r"network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3", "conv_dim=4"]
    loaded, _notes = kb.load_kohya_preset(path, "flux1_dev", kb.kohya_defaults("flux1_dev", "lora"))
    assert loaded["optimizer_args"] == '"betas=(0.9, 0.999)" weight_decay=0.01'
    again = dict(values, **{key: loaded[key] for key in ("optimizer_args", "lr_scheduler_args", "network_args")})
    _path, reloaded = _runtime_toml(kb.train_kohya_model("flux1_dev", again, print_only=True))
    assert [reloaded[key] for key in ("optimizer_args", "lr_scheduler_args", "network_args")] == [data[key] for key in ("optimizer_args", "lr_scheduler_args", "network_args")]


def test_settings_sd_scripts_accepts_pass_preflight(tmp_path):
    cases = [
        ("flux1_dev", "lora", {"sample_prompts": "a photo of ohwx man reading notes.txt", "sample_every_n_epochs": 5}),
        ("flux1_dev", "lora", {"network_module": "networks.oft_flux", "network_alpha": 1e-3}),
        ("flux1_dev", "lora", {"lr_scheduler": "constant_with_warmup", "lr_warmup_steps": 10}),
        ("sdxl", "full", {"lr_scheduler": "cosine_with_min_lr", "lr_scheduler_args": "min_lr=1e-7"}),
        ("sdxl", "full", {"lr_scheduler": "cosine_with_min_lr", "additional_parameters": "--lr_scheduler_min_lr_ratio 0.1"}),
        ("sdxl", "full", {"lr_scheduler": "piecewise_constant", "lr_scheduler_args": "step_rules='1:10,0.1:20,0.01'"}),
        ("flux1_dev", "lora", {"lr_scheduler": "adafactor", "optimizer_args": "relative_step=True"}),
        (
            "flux1_dev",
            "lora",
            {
                "optimizer_type": "AdamW",
                "optimizer_args": '"betas=(0.9, 0.999)" weight_decay=0.01',
                "lr_scheduler_type": "CyclicLR",
                "lr_scheduler_args": "base_lr=1e-6 max_lr=1e-4 mode='triangular2' cycle_momentum=False",
            },
        ),
        ("flux1_dev", "lora", {"network_args": "train_t5xxl=True", "t5xxl_lr": 1e-5, "cache_text_encoder_outputs": False, "cache_text_encoder_outputs_to_disk": False}),
        ("flux1_dev", "lora", {"additional_parameters": "--network_train_text_encoder_only"}),
        ("flux1_dev", "lora", {"resume_from_huggingface": "me/my-lora/state:main:model"}),
        ("sdxl", "lora", {"network_args": r"network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3 conv_dim=4 conv_alpha=1"}),
    ]
    for model, kind, overrides in cases:
        values = _values(tmp_path, model, kind, **overrides)
        assert kb.kohya_validation_problems(model, values) == [], overrides


def test_text_encoder_only_is_recorded_in_the_runtime_toml(tmp_path):
    """--network_train_text_encoder_only goes into the runtime TOML (sd-scripts reads it there, the kohya_ss GUI wrote it
    there too), so reloading the run keeps it; reloading a run that trained both parts drops one left on screen."""
    values = _values(tmp_path, "flux1_dev", "lora", additional_parameters="--network_train_text_encoder_only --max_token_length 225")
    text = kb.train_kohya_model("flux1_dev", values, print_only=True)
    path, data = _runtime_toml(text)
    assert data["network_train_text_encoder_only"] is True and "network_train_unet_only" not in data
    assert "--network_train_text_encoder_only" not in _command_line(text) and _command_line(text).endswith("--max_token_length 225")
    current = kb.kohya_defaults("flux1_dev", "lora")
    loaded, _notes = kb.load_kohya_preset(path, "flux1_dev", current)
    assert "--network_train_text_encoder_only" in loaded["additional_parameters"]

    stale = dict(current, additional_parameters="--max_token_length 225 --network_train_text_encoder_only")
    both = _values(tmp_path, "flux1_dev", "lora", output_name="both_parts")
    path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", both, print_only=True))
    assert "network_train_text_encoder_only" not in data
    loaded, _notes = kb.load_kohya_preset(path, "flux1_dev", stale)
    assert loaded["additional_parameters"] == "--max_token_length 225"
    kohya_toml = tmp_path / "config_lora-20260101-000001.toml"  # the kohya_ss GUI never records additional_parameters
    kohya_toml.write_text('network_module = "networks.lora_flux"\nnetwork_dim = 16\nlearning_rate = 1e-4\nresolution = "1024,1024"\n', encoding="utf-8")
    loaded, _notes = kb.load_kohya_preset(str(kohya_toml), "flux1_dev", stale)
    assert loaded["additional_parameters"] == "--max_token_length 225"


def test_t5xxl_learning_rate_reaches_sd_scripts_and_reloads(tmp_path):
    """The kohya_ss GUI wrote text_encoder_lr = [CLIP-L, T5-XXL]; the T5-XXL LoRA exists only with train_t5xxl=True."""
    values = _values(tmp_path, "flux1_dev", "lora", t5xxl_lr=1e-5, network_args="train_t5xxl=True", cache_text_encoder_outputs=False, cache_text_encoder_outputs_to_disk=False)
    path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["text_encoder_lr"] == [5e-05, 1e-05] and data["network_args"] == ["train_t5xxl=True"]
    loaded, _notes = kb.load_kohya_preset(path, "flux1_dev", kb.kohya_defaults("flux1_dev", "lora"))
    assert (loaded["text_encoder_lr"], loaded["t5xxl_lr"]) == (5e-05, 1e-05)
    data = {"LoRA_type": "Flux1", "network_dim": 16, "learning_rate": 5e-5, "unet_lr": 5e-5, "text_encoder_lr": 5e-5, "t5xxl_lr": 1e-5, "train_t5xxl": True, "epoch": 10}
    converted, notes = kb.convert_kohya_gui_json(data, "flux1_dev", "lora")
    assert converted["t5xxl_lr"] == 1e-05 and "train_t5xxl=True" in converted["network_args"]
    assert not any("never trained" in note for note in notes)


def test_kohya_json_conversion_follows_what_the_kohya_gui_ran():
    base = {"LoRA_type": "Flux1", "network_dim": 16, "learning_rate": 5e-5, "unet_lr": 5e-5, "text_encoder_lr": 5e-5, "epoch": 10, "max_resolution": "1024,1024"}
    # split_mode: the GUI forced train_blocks=single, sd-scripts swapped 18 blocks unless blocks_to_swap was set
    values, _notes = kb.convert_kohya_gui_json(dict(base, split_mode=True, blocks_to_swap=0), "flux1_dev", "lora")
    assert (values["train_blocks"], values["blocks_to_swap"]) == ("single", 18)
    values, _notes = kb.convert_kohya_gui_json(dict(base, split_mode=True, blocks_to_swap=10), "flux1_dev", "lora")
    assert (values["train_blocks"], values["blocks_to_swap"]) == ("single", 10)
    # logging_dir alone made sd-scripts log to TensorBoard
    values, _notes = kb.convert_kohya_gui_json(dict(base, logging_dir="C:/logs", log_with=""), "flux1_dev", "lora")
    assert (values["log_with"], values["logging_dir"]) == ("tensorboard", "C:/logs")
    # Hugging Face resume: the location in resume, the text "True" in resume_from_huggingface
    values, _notes = kb.convert_kohya_gui_json(dict(base, resume="me/repo/state:main:model", resume_from_huggingface="True"), "flux1_dev", "lora")
    assert (values["resume"], values["resume_from_huggingface"]) == ("", "me/repo/state:main:model")
    values, _notes = kb.convert_kohya_gui_json(dict(base, resume="C:/runs/x-state", resume_from_huggingface=""), "flux1_dev", "lora")
    assert (values["resume"], values["resume_from_huggingface"]) == ("C:/runs/x-state", "")
    # warmup: int(lr_warmup_steps) when above 0, else lr_warmup %; with a dataset TOML lr_warmup % of max_train_steps
    for json_values, expected in (
        ({"lr_warmup_steps": 0.05}, 0),
        ({"lr_warmup_steps": 12.7, "lr_warmup": 10}, 12),
        ({"lr_warmup": 10}, 0.1),
        ({"dataset_config": "C:/d.toml", "max_train_steps": 500, "lr_warmup": 10, "lr_warmup_steps": 3}, 50),
        ({"dataset_config": "C:/d.toml", "lr_warmup_steps": 3}, 0),
    ):
        values, _notes = kb.convert_kohya_gui_json(dict(base, **json_values), "flux1_dev", "lora")
        assert values["lr_warmup_steps"] == expected, json_values


def test_hugging_face_resume_reaches_sd_scripts_as_location_and_flag(tmp_path):
    values = _values(tmp_path, "flux1_dev", "lora", resume_from_huggingface="me/repo/state:main:model")
    path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert (data["resume"], data["resume_from_huggingface"]) == ("me/repo/state:main:model", True)
    loaded, _notes = kb.load_kohya_preset(path, "flux1_dev", kb.kohya_defaults("flux1_dev", "lora"))
    assert (loaded["resume"], loaded["resume_from_huggingface"]) == ("", "me/repo/state:main:model")
    kohya_toml = tmp_path / "config_lora-20260101-000002.toml"  # the kohya_ss GUI stored the text "True"
    kohya_toml.write_text('network_module = "networks.lora_flux"\nresolution = "1024,1024"\nresume = "me/repo/state:main:model"\nresume_from_huggingface = "True"\n', encoding="utf-8")
    loaded, _notes = kb.load_kohya_preset(str(kohya_toml), "flux1_dev", kb.kohya_defaults("flux1_dev", "lora"))
    assert (loaded["resume"], loaded["resume_from_huggingface"]) == ("", "me/repo/state:main:model")


def test_reload_kohya_lora_tab_toml_uses_sd_scripts_defaults_for_dropped_zeros(tmp_path):
    """The kohya_ss LoRA tab drops every 0 from its runtime TOML, so sd-scripts used its own defaults (network_alpha 1)."""
    kohya_toml = tmp_path / "config_lora-20260101-000003.toml"
    kohya_toml.write_text(
        'network_module = "networks.lora_flux"\nnetwork_dim = 16\nlearning_rate = 1e-4\nresolution = "1024,1024"\n'
        'train_data_dir = "C:/data"\nmax_train_steps = 300\nlogging_dir = "C:/logs"\n',
        encoding="utf-8",
    )
    stale = kb.kohya_defaults("flux1_dev", "lora")
    stale.update(network_alpha=128.0, huber_c=0.5, huber_scale=2.0, loss_type="huber", lr_scheduler_num_cycles=3, lr_scheduler_power=2.0)
    stale.update(bucket_reso_steps=32, min_bucket_reso=512, max_bucket_reso=2048, max_data_loader_n_workers=0)
    loaded, _notes = kb.load_kohya_preset(str(kohya_toml), "flux1_dev", stale)
    assert loaded["network_alpha"] == 1.0 and (loaded["huber_c"], loaded["huber_scale"], loaded["loss_type"]) == (0.1, 1.0, "l2")
    assert (loaded["lr_scheduler_num_cycles"], loaded["lr_scheduler_power"], loaded["max_data_loader_n_workers"]) == (1, 1.0, 8)
    assert (loaded["bucket_reso_steps"], loaded["min_bucket_reso"], loaded["max_bucket_reso"]) == (64, 256, 1024)
    assert (loaded["log_with"], loaded["logging_dir"]) == ("tensorboard", "C:/logs")  # logging_dir alone: TensorBoard
    # a fine-tuning run's TOML has no network keys: the LoRA fields keep their values
    full_toml = tmp_path / "config_dreambooth-20260101-000004.toml"
    full_toml.write_text('learning_rate = 4e-6\nresolution = "1024,1024"\ntrain_data_dir = "C:/data"\nmax_train_steps = 300\n', encoding="utf-8")
    loaded, _notes = kb.load_kohya_preset(str(full_toml), "flux1_dev", stale)
    assert loaded["training_mode"] == FULL_FINE_TUNING_MODE and loaded["network_alpha"] == 128.0


def test_hand_edited_numbers_read_like_the_field_check(tmp_path):
    preset = tmp_path / "hand.toml"
    preset.write_text('preset_format = 1\nlearning_rate = "0,0001"\nmax_train_steps = "10,000"\nnetwork_alpha = " 16 "\nseed = "abc"\n', encoding="utf-8")
    current = kb.kohya_defaults("flux1_dev", "lora")
    loaded, _notes = kb.load_kohya_preset(str(preset), "flux1_dev", current)
    assert (loaded["learning_rate"], loaded["max_train_steps"], loaded["network_alpha"]) == (0.0001, 10000, 16.0)
    assert loaded["seed"] == 1  # not a number: the default
    values = _values(tmp_path, "flux1_dev", "lora", learning_rate="0,0001")
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["learning_rate"] == 0.0001


def test_kohya_epochs_note_the_step_difference():
    """The kohya_ss GUI trained ceil(images x repeats / batch / accumulation x epoch) steps; sd-scripts rounds every epoch up."""
    base = {"LoRA_type": "Flux1", "network_dim": 16, "learning_rate": 1e-4, "unet_lr": 1e-4, "text_encoder_lr": 0}
    _converted, notes = kb.convert_kohya_gui_json(dict(base, epoch=50, num_processes=4), "flux1_dev", "lora")
    assert any("15 images: 800 here, 750 in the kohya_ss app" in note for note in notes)
    _converted, notes = kb.convert_kohya_gui_json(dict(base, epoch=200, train_batch_size=7), "flux1_dev", "lora")
    assert any("15 images: 600 here, 429 in the kohya_ss app" in note for note in notes)
    _converted, notes = kb.convert_kohya_gui_json(dict(base, epoch=200), "flux1_dev", "lora")
    assert not any(note.startswith("Steps:") for note in notes)
    assert kb.kohya_steps_example(100, 1, 4, 1) == (15, 400, 375) and kb.kohya_steps_example(100, 1, 1, 1) is None
    readme = kb.ROOT_DIR / "Training_Presets" / "FLUX_1_Dev_Training_Configs" / "LoRA_Training" / "README.md"
    if readme.is_file():
        assert "15 images → 800 steps here, 750 in the kohya_ss app" in readme.read_text(encoding="utf-8")


def test_single_line_prompt_text_ending_in_txt_is_a_prompt(tmp_path):
    text = "a photo of ohwx man reading notes.txt"
    values = _values(tmp_path, "flux1_dev", "lora", sample_prompts=text, sample_every_n_epochs=5)
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert Path(data["sample_prompts"]).read_text(encoding="utf-8").strip() == text
    assert kb._looks_like_prompt_file("prompts.txt") and kb._looks_like_prompt_file('"C:/my prompts/p.txt"')
    assert not kb._looks_like_prompt_file("photo, cat/dog notes.txt")


def test_kohya_backend_whole_numbers_prompts_and_one_message_per_field(tmp_path):
    from musubi_tuner_gui.unified.backends import get_backend

    backend = get_backend("kohya:flux1_dev")
    whole = {key for key in backend.keys() if backend.whole_number(key)}
    assert {"network_dim", "max_train_epochs", "dataset_batch_size", "blocks_to_swap", "dataset_resolution_width", "num_processes", "seed"} <= whole
    assert not {"network_alpha", "max_grad_norm", "lr_warmup_steps", "learning_rate", "guidance_scale", "t5xxl_lr"} & whole
    sdxl = get_backend("kohya:sdxl")
    assert sdxl.whole_number("blocks_to_swap") and sdxl.whole_number("max_timestep") and not sdxl.whole_number("noise_offset")
    assert backend.trainer_value("max_train_steps", "10,000") == 10000

    # a blank or fractional field gives one message (the field check's), not a second one from kohya_backend
    values = _values(tmp_path, "flux1_dev", "lora")
    assert backend.check(values, "lora") == []
    for key, bad, message in (
        ("network_dim", 16.5, "Network Rank (Dim) must be a whole number (got 16.5)."),
        ("max_train_epochs", "10.5", "Max Training Epochs must be a whole number (got 10.5)."),
        ("blocks_to_swap", 13.7, "Blocks to Swap must be a whole number (got 13.7)."),
        ("dataset_batch_size", "1.5", "Batch Size must be a whole number (got 1.5)."),
        ("dataset_resolution_width", None, "Resolution Width is empty. Enter a value (default: 1024)."),
        ("num_processes", None, "Number of Processes is empty. Enter a value (default: 1)."),
        ("gradient_accumulation_steps", None, "Gradient Accumulation Steps is empty. Enter a value (default: 1)."),
    ):
        assert backend.check(dict(values, **{key: bad}), "lora") == [message], key

    # Sample Prompts: prompt text stays as typed; only an existing prompt file gets path cleaning
    text = r"a photo of ohwx man \(smiling\) //wide angle $USERNAME --w 1024"
    assert backend.trainer_value("sample_prompts", text) == text
    prompt_file = tmp_path / "my prompts.txt"
    prompt_file.write_text("a photo", encoding="utf-8")
    assert backend.trainer_value("sample_prompts", f'"{prompt_file}"') == os.path.normpath(str(prompt_file))
    preset = tmp_path / "prompts preset.toml"
    backend.save(str(preset), dict(backend.defaults("lora"), sample_prompts=text))
    assert backend.load(str(preset), backend.defaults("lora")).values["sample_prompts"] == text


# ----------------------------------------------------------------------------------------------------------------------
# the runtime TOMLs are accepted by the real sd-scripts code (CPU only, stops before loading a model)
# ----------------------------------------------------------------------------------------------------------------------

_DRY_RUN_CHILD = r"""
import argparse, json, os, sys, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
jobs = json.loads(sys.argv[2])
import importlib
import torch
from library import train_util, strategy_base
from accelerate.state import AcceleratorState

class Stop(Exception):
    pass

def stop(*args, **kwargs):
    raise Stop()

for name, script, toml_path in jobs:
    module = importlib.import_module(script)
    for cls in (strategy_base.TokenizeStrategy, strategy_base.LatentsCachingStrategy, strategy_base.TextEncodingStrategy, strategy_base.TextEncoderOutputsCachingStrategy):
        cls._strategy = None
    AcceleratorState._reset_state(True)
    sys.argv = [script + ".py", "--config_file", toml_path]
    parser = module.setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    # the optimizer and LR scheduler exactly as sd-scripts builds them (on a copy: Adafactor's relative_step edits args)
    probe = argparse.Namespace(**vars(args))
    _name, _arguments, optimizer = train_util.get_optimizer(probe, [torch.nn.Parameter(torch.zeros(1))])
    scheduler = train_util.get_scheduler_fix(probe, optimizer, 1)
    print("DRYRUN OPTIMIZER", name, type(optimizer).__name__, type(scheduler).__name__, flush=True)
    if script == "flux_train_network" and args.network_args:
        # the network arguments split as train_network.py splits them, read by networks.lora_flux (no model needed)
        from networks import lora_flux
        kwargs = dict(arg.split("=", 1) for arg in args.network_args)
        network = lora_flux.create_network(1.0, args.network_dim, args.network_alpha, None, [torch.nn.Module(), torch.nn.Module()], torch.nn.Module(), neuron_dropout=args.network_dropout, **kwargs)
        print("DRYRUN NETWORK", name, json.dumps({"reg_lrs": network.reg_lrs, "train_t5xxl": network.train_t5xxl, "text_encoder_lr": args.text_encoder_lr}), flush=True)
    try:
        if script == "flux_train_network":
            trainer = module.FluxNetworkTrainer(); trainer.load_target_model = stop; trainer.train(args)
        elif script == "sdxl_train_network":
            trainer = module.SdxlNetworkTrainer(); trainer.load_target_model = stop; trainer.train(args)
        elif script == "flux_train":
            module.flux_utils.load_ae = stop; module.flux_utils.load_flow_model = stop
            module.train(args)
        else:
            module.sdxl_train_util.load_target_model = stop
            module.train(args)
        print("DRYRUN NOSTOP", name, flush=True)
    except Stop:
        print("DRYRUN OK", name, flush=True)
"""


def _tokenizers_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    needed = [
        ("openai/clip-vit-large-patch14", "tokenizer_config.json"),
        ("google/t5-v1_1-xxl", "spiece.model"),
        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer_config.json"),
    ]
    return all(isinstance(try_to_load_from_cache(repo, name), str) for repo, name in needed)


@pytest.mark.skipif(not (kb.SD_SCRIPTS_DIR / "flux_train_network.py").is_file(), reason="sd-scripts not installed")
@pytest.mark.skipif(not _tokenizers_cached(), reason="CLIP/T5 tokenizers are not in the Hugging Face cache")
def test_runtime_tomls_pass_sd_scripts_setup_on_cpu(tmp_path):
    jobs = []
    for model, kind in COMBOS:
        base = tmp_path / f"{model}_{kind}"
        base.mkdir()
        values = _values(base, model, kind)
        if model == "sdxl":
            values["reg_data_dir"] = str(_make_dataset(base / "reg", "1_man"))
        path, _data = _runtime_toml(kb.train_kohya_model(model, values, print_only=True))
        jobs.append((f"{model}_{kind}", SCRIPTS[(model, kind)][:-3], path))
    # arguments sd-scripts reads with ast.literal_eval when it builds the optimizer and scheduler (tuples, quoted text),
    # the schedulers that need arguments, and network arguments with '=' inside a value
    reg_lrs = r"network_reg_lrs=single_blocks_(\d|10)_=1e-3,double_blocks=2e-3"
    expected = {
        "adamw_betas": ({"optimizer_type": "AdamW", "optimizer_args": "weight_decay=0.01, betas=0.9,0.999 eps=1e-8"}, "AdamW", "LambdaLR"),
        "adafactor_schedule": ({"lr_scheduler": "adafactor", "optimizer_args": "scale_parameter=True relative_step=True warmup_init=False"}, "Adafactor", "AdafactorSchedule"),
        "quoted_items": (
            {
                "optimizer_type": "AdamW",
                "optimizer_args": '"betas=(0.9, 0.999)" weight_decay=0.01',
                "lr_scheduler_type": "CyclicLR",
                "lr_scheduler_args": "base_lr=1e-6 max_lr=1e-4 mode='triangular2' cycle_momentum=False",
                "network_args": f"{reg_lrs} train_t5xxl=True",
                "t5xxl_lr": 1e-5,
                "cache_text_encoder_outputs": False,
                "cache_text_encoder_outputs_to_disk": False,
            },
            "AdamW",
            "CyclicLR",
        ),
        "piecewise": ({"lr_scheduler": "piecewise_constant", "lr_scheduler_args": "step_rules='1:10,0.1:20,0.01'"}, "Adafactor", "LambdaLR"),
        "min_lr": ({"lr_scheduler": "cosine_with_min_lr", "lr_scheduler_args": "min_lr=1e-7", "lr_warmup_steps": 0.05}, "Adafactor", "LambdaLR"),
    }
    for name, (overrides, _optimizer, _scheduler) in expected.items():
        base = tmp_path / name
        base.mkdir()
        values = _values(base, "flux1_dev", "lora", **overrides)
        path, _data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
        jobs.append((name, "flux_train_network", path))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(kb.SD_SCRIPTS_DIR), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", _DRY_RUN_CHILD, str(kb.SD_SCRIPTS_DIR), json.dumps(jobs)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    oks = [line for line in result.stdout.splitlines() if line.startswith("DRYRUN OK")]
    optimizers = {line.split()[2]: line.split()[3:] for line in result.stdout.splitlines() if line.startswith("DRYRUN OPTIMIZER")}
    assert len(oks) == len(optimizers) == len(jobs), f"{result.stdout[-4000:]}\n{result.stderr[-6000:]}"
    for name, (_overrides, optimizer, scheduler) in expected.items():
        assert optimizers[name] == [optimizer, scheduler], (name, optimizers[name])
    networks = {line.split()[2]: json.loads(line.split(" ", 3)[3]) for line in result.stdout.splitlines() if line.startswith("DRYRUN NETWORK")}
    assert networks["quoted_items"] == {"reg_lrs": {r"single_blocks_(\d|10)_": 0.001, "double_blocks": 0.002}, "train_t5xxl": True, "text_encoder_lr": [5e-05, 1e-05]}


_TYPES_CHILD = r"""
import importlib, json, os, sys, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
types = {}
for script in json.loads(sys.argv[2]):
    parser = importlib.import_module(script).setup_parser()
    types[script] = {action.dest: getattr(action.type, "__name__", str(action.type)) for action in parser._actions}
print("TYPES " + json.dumps(types), flush=True)
"""


@pytest.mark.skipif(not (kb.SD_SCRIPTS_DIR / "flux_train_network.py").is_file(), reason="sd-scripts not installed")
def test_whole_number_settings_follow_sd_scripts_argparse():
    """A setting refuses fractions exactly where sd-scripts' argparse option is an int (or it is one of the counts sd-scripts
    has no option for); float options and lr_warmup_steps (int or float) keep them."""
    scripts = sorted({script[:-3] for script in SCRIPTS.values()})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", PYTHONUTF8="1")
    env["PYTHONPATH"] = os.pathsep.join([str(kb.SD_SCRIPTS_DIR), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", _TYPES_CHILD, str(kb.SD_SCRIPTS_DIR), json.dumps(scripts)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        creationflags=0x08000000 if os.name == "nt" else 0,
    )
    line = next((line for line in result.stdout.splitlines() if line.startswith("TYPES ")), "")
    assert line, result.stderr[-4000:]
    types = json.loads(line[6:])
    sd_names = {"dataset_batch_size": "train_batch_size"}
    checked = 0
    for (model, kind), script in SCRIPTS.items():
        parsed = types[script[:-3]]
        for spec in kb.kohya_param_specs(model):
            if spec.kind not in ("number", "slider") or kind not in spec.modes:
                continue
            option = parsed.get(sd_names.get(spec.key, spec.key))
            if option not in (None, "None"):
                checked += 1
                assert kb.kohya_whole_number(model, spec.key) == (option == "int"), (script, spec.key, option)
            else:
                assert kb.kohya_whole_number(model, spec.key) == (spec.key in kb._WHOLE_NUMBER_KEYS), (script, spec.key)
    assert checked > 100


# --------------------------------------------------------------------------------------------- Automagic on sd-scripts
@pytest.mark.parametrize("optimizer", ["Automagic3", "Automagic4", "AutomagicEXPERIMENT"])
def test_automagic_reaches_sd_scripts_as_its_musubi_class(tmp_path, optimizer):
    values = _values(tmp_path, "flux1_dev", "lora", optimizer_type=optimizer, max_grad_norm=0, gradient_accumulation_steps=1)
    assert kb.kohya_validation_problems("flux1_dev", values) == []
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["optimizer_type"] == f"musubi_tuner.optimizers.{optimizer}"
    # the preset's Adafactor-only arguments go, weight decay stays; fused only when sd-scripts' loop allows it
    assert data["optimizer_args"] == ["weight_decay=0.01", "fused=True"]
    values = _values(tmp_path, "flux1_dev", "lora", optimizer_type=optimizer, max_grad_norm=1.0)
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["optimizer_args"][-1] == "fused=False"
    values = _values(tmp_path, "flux1_dev", "lora", optimizer_type=optimizer, optimizer_args="fused=False weight_decay=0")
    _path, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True))
    assert data["optimizer_args"] == ["fused=False", "weight_decay=0"]


def test_automagic_refuses_an_lr_warmup(tmp_path):
    values = _values(tmp_path, "sdxl", "lora", optimizer_type="Automagic3", lr_scheduler="constant_with_warmup", lr_warmup_steps=50)
    assert any("manages its own learning rate" in problem for problem in kb.kohya_validation_problems("sdxl", values))


_OPTIMIZER_CHILD = r"""
import sys, json, argparse
sys.path.insert(0, sys.argv[1])
import torch
from library import train_util
parser = argparse.ArgumentParser()
train_util.add_optimizer_arguments(parser)
args = parser.parse_args([])
for key, value in json.loads(sys.argv[2]).items():
    setattr(args, key, value)
params = [torch.nn.Parameter(torch.ones(4, 8))]
name, text, optimizer = train_util.get_optimizer(args, params)
print("OPTIMIZER", type(optimizer).__module__, type(optimizer).__name__, optimizer.fused, flush=True)
"""


@pytest.mark.skipif(not (kb.SD_SCRIPTS_DIR / "flux_train_network.py").is_file(), reason="sd-scripts not installed")
@pytest.mark.parametrize("optimizer", ["Automagic3", "Automagic4", "AutomagicEXPERIMENT"])
def test_sd_scripts_constructs_the_musubi_automagic(optimizer):
    """sd-scripts' own get_optimizer imports the class from musubi-tuner (installed in the same venv)."""
    args = {"optimizer_type": f"musubi_tuner.optimizers.{optimizer}", "optimizer_args": ["weight_decay=0.01", "fused=False"], "learning_rate": 1e-4,
            "max_grad_norm": 0.0, "lr_scheduler": "constant", "fused_backward_pass": False}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1")
    env["PYTHONPATH"] = os.pathsep.join([str(kb.SD_SCRIPTS_DIR), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", _OPTIMIZER_CHILD, str(kb.SD_SCRIPTS_DIR), json.dumps(args)], env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=600)
    assert f"OPTIMIZER musubi_tuner.optimizers" in result.stdout and f" {optimizer} False" in result.stdout, result.stdout[-1500:] + result.stderr[-3000:]


def test_flux_lora_h2d_only_block_swap_reaches_the_runtime_toml(tmp_path):
    """kohya-ss/sd-scripts #2467 (ported): H2D-only block swap streams the frozen FLUX blocks to the GPU without copying
    them back; it is written only for LoRA runs that swap blocks."""
    values = _values(tmp_path, "flux1_dev", "lora", blocks_to_swap=13, block_swap_h2d_only=True, gradient_checkpointing=True)
    _, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True, runner=None))
    assert data["block_swap_h2d_only"] is True and data["blocks_to_swap"] == 13
    values = _values(tmp_path, "flux1_dev", "lora", blocks_to_swap=0, block_swap_h2d_only=True)
    _, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True, runner=None))
    assert "block_swap_h2d_only" not in data  # nothing to stream without block swap
    values = _values(tmp_path, "flux1_dev", "full", blocks_to_swap=10, block_swap_h2d_only=True)
    _, data = _runtime_toml(kb.train_kohya_model("flux1_dev", values, print_only=True, runner=None))
    assert "block_swap_h2d_only" not in data  # fine-tuning trains the swapped weights

