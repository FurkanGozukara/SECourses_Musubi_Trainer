from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
import toml

from musubi_tuner_gui import (
    flux2_lora_gui,
    flux_klein_lora_gui,
    flux_lora_gui,
    modern_image_lora_gui,
    qwen_image_lora_gui,
    wan_lora_gui,
    zimage_lora_gui,
)
from musubi_tuner_gui.common_gui import SaveConfigFile, SaveConfigFileToRun
from musubi_tuner_gui.full_finetune_gui import (
    FULL_FINE_TUNING_MODE,
    FULL_FINE_TUNING_NETWORK_KEYS,
    LORA_TRAINING_MODE,
    normalize_image_training_parameters,
    training_mode_runtime_exclusions,
)
from musubi_tuner_gui.modern_image_lora_gui import (
    ModernImageWorkflow,
    _architecture_defaults,
    _build_workflow_script,
    get_architecture,
    prepare_modern_image_workflow,
)
from musubi_tuner_gui.class_tab_config_manager import TabConfigManager


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute_command(self, **kwargs):
        self.calls.append(kwargs)


def _touch(path: Path, contents: str = "placeholder") -> str:
    path.write_text(contents, encoding="utf-8")
    return str(path)


def _ordered_parameters(keys: list[str], values: dict[str, object]) -> list[tuple[str, object]]:
    return [(key, values.get(key)) for key in keys]


def _base_full_values(tmp_path: Path, keys: list[str], *, model_version: str) -> dict[str, object]:
    dataset = _touch(tmp_path / "dataset.toml", "[[datasets]]\nimage_directory = 'images'\n")
    values = {key: None for key in keys}
    values.update(
        {
            "training_mode": FULL_FINE_TUNING_MODE,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "num_processes": 1,
            "num_machines": 1,
            "multi_gpu": False,
            "gpu_ids": "0",
            "dynamo_backend": "no",
            "dataset_config_mode": "Use TOML File",
            "dataset_config": dataset,
            "model_version": model_version,
            "dit": _touch(tmp_path / "dit.safetensors"),
            "vae": _touch(tmp_path / "vae.safetensors"),
            # Gradio sends an unselected dropdown as an empty string, not None.
            "vae_dtype": "",
            "text_encoder": _touch(tmp_path / "text_encoder.safetensors"),
            "fp8_base": False,
            "fp8_scaled": False,
            "blocks_to_swap": 0,
            "sdpa": True,
            "use_legacy_sdpa": True,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
            "full_bf16": True,
            "full_fp16": False,
            "fused_backward_pass": True,
            "block_swap_optimizer_patch_params": False,
            "optimizer_type": "AdaFactor",
            "optimizer_args": [],
            "learning_rate": 1e-5,
            "max_grad_norm": 0,
            "lr_scheduler": "constant",
            "network_module": "networks.lora_flux_2",
            "network_dim": 32,
            "network_alpha": 32,
            "network_dropout": 0,
            "network_args": [],
            "output_dir": str(tmp_path / "output"),
            "output_name": "full-smoke",
            "save_precision": "bf16",
            "save_every_n_steps": 1,
            "caching_latent_skip_existing": False,
            "caching_teo_skip_existing": False,
        }
    )
    return values


@pytest.mark.parametrize("save_config", [SaveConfigFile, SaveConfigFileToRun])
@pytest.mark.parametrize("blank_value", ["", "   "])
def test_config_serializers_omit_blank_optional_vae_dtype(tmp_path: Path, save_config, blank_value: str):
    config_path = tmp_path / "config.toml"

    save_config(
        [("mixed_precision", "bf16"), ("vae_dtype", blank_value)],
        str(config_path),
    )

    assert toml.load(config_path) == {"mixed_precision": "bf16"}


@pytest.mark.parametrize(
    "loader",
    [flux_lora_gui.open_flux_configuration, flux2_lora_gui.open_flux2_configuration],
)
def test_flux_config_loaders_normalize_legacy_blank_vae_dtype(tmp_path: Path, monkeypatch, loader):
    config_path = tmp_path / "legacy.toml"
    config_path.write_text('vae_dtype = ""\n', encoding="utf-8")
    monkeypatch.setattr(gr, "Info", lambda *_args, **_kwargs: None)

    loaded = loader(False, str(config_path), [("vae_dtype", "bfloat16")])

    assert loaded[2] == "float32"


def test_shared_full_finetune_rules_remove_quantized_base_and_align_precision():
    parameters = [
        ("training_mode", "DreamBooth Fine-Tuning"),
        ("mixed_precision", "bf16"),
        ("full_bf16", True),
        ("full_fp16", False),
        ("fp8_base", True),
        ("fp8_scaled", True),
        ("block_swap_h2d_only", True),
        ("dit_dtype", "float16"),
        ("save_precision", "fp32"),
        ("blocks_to_swap", 8),
        ("num_processes", 1),
        ("gradient_accumulation_steps", 1),
        ("optimizer_type", "AdaFactor"),
        ("fused_backward_pass", True),
        ("block_swap_optimizer_patch_params", False),
    ]

    values, normalized, full_finetune = normalize_image_training_parameters(parameters)

    assert full_finetune is True
    assert values["training_mode"] == FULL_FINE_TUNING_MODE
    assert values["fp8_base"] is False
    assert values["fp8_scaled"] is False
    assert values["block_swap_h2d_only"] is False
    assert values["dit_dtype"] == "bfloat16"
    assert values["save_precision"] == "bf16"
    assert dict(normalized) == values


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"full_fp16": True}, "Full FP16"),
        ({"num_processes": 2, "blocks_to_swap": 1}, "multi-process"),
        ({"optimizer_type": "AdamW", "fused_backward_pass": True}, "Adafactor"),
        ({"gradient_accumulation_steps": 2, "fused_backward_pass": True}, "accumulation"),
        ({"blocks_to_swap": 0, "block_swap_optimizer_patch_params": True}, "blocks_to_swap"),
        (
            {"blocks_to_swap": 1, "optimizer_type": "Automagic2", "block_swap_optimizer_patch_params": True},
            "Automagic3",
        ),
    ],
)
def test_shared_full_finetune_rules_reject_unsupported_combinations(overrides, message):
    values = {
        "training_mode": FULL_FINE_TUNING_MODE,
        "mixed_precision": "bf16",
        "full_bf16": True,
        "full_fp16": False,
        "blocks_to_swap": 0,
        "num_processes": 1,
        "gradient_accumulation_steps": 1,
        "optimizer_type": "AdaFactor",
        "fused_backward_pass": False,
        "block_swap_optimizer_patch_params": False,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        normalize_image_training_parameters(list(values.items()))


@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic3"])
def test_full_finetune_block_swap_patch_accepts_non_fused_automagic(optimizer_type):
    values = {
        "training_mode": FULL_FINE_TUNING_MODE,
        "mixed_precision": "bf16",
        "full_bf16": True,
        "full_fp16": False,
        "blocks_to_swap": 1,
        "num_processes": 1,
        "gradient_accumulation_steps": 1,
        "optimizer_type": optimizer_type,
        "fused_backward_pass": False,
        "block_swap_optimizer_patch_params": True,
    }

    normalized_values, normalized, full_finetune = normalize_image_training_parameters(list(values.items()))

    assert full_finetune is True
    assert normalized_values["block_swap_optimizer_patch_params"] is True
    assert dict(normalized)["block_swap_optimizer_patch_params"] is True


def test_training_mode_exclusions_keep_runtime_configs_mode_correct():
    full = training_mode_runtime_exclusions(FULL_FINE_TUNING_MODE)
    lora = training_mode_runtime_exclusions(LORA_TRAINING_MODE)

    assert {"network_module", "network_dim", "network_alpha", "full_fp16"} <= full
    assert {"fused_backward_pass", "block_swap_optimizer_patch_params", "mem_eff_save"} <= lora
    assert "network_module" not in lora


def test_qwen_dreambooth_preview_normalizes_full_model_runtime_config(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]\nimage_directory = 'images'\n", encoding="utf-8")
    dit = _touch(tmp_path / "dit.safetensors")
    vae = _touch(tmp_path / "vae.safetensors")
    text_encoder = _touch(tmp_path / "text_encoder.safetensors")
    captured: dict[str, object] = {}

    def capture_preview(command, config_path):
        captured["command"] = command
        captured["config"] = toml.load(config_path)

    monkeypatch.setattr(qwen_image_lora_gui, "print_command_and_toml", capture_preview)
    parameters = [
        ("training_mode", "DreamBooth Fine-Tuning"),
        ("dataset_config_mode", "Use TOML File"),
        ("dataset_config", str(dataset)),
        ("model_version", "original"),
        ("dit", dit),
        ("vae", vae),
        ("text_encoder", text_encoder),
        ("output_dir", str(tmp_path / "output")),
        ("output_name", "qwen-full-preview"),
        ("mixed_precision", "bf16"),
        ("full_bf16", True),
        ("full_fp16", False),
        ("save_precision", "fp32"),
        ("dit_dtype", "float16"),
        ("fp8_base", True),
        ("fp8_scaled", True),
        ("block_swap_h2d_only", True),
        ("block_swap_ring_size", 2),
        ("blocks_to_swap", 6),
        ("gradient_checkpointing", True),
        ("gradient_accumulation_steps", 1),
        ("num_processes", 1),
        ("num_machines", 1),
        ("num_cpu_threads_per_process", 1),
        ("gpu_ids", "0"),
        ("dynamo_backend", "no"),
        ("optimizer_type", "Automagic3"),
        ("optimizer_args", ["scale_parameter=False", "relative_step=False"]),
        ("fused_backward_pass", False),
        ("block_swap_optimizer_patch_params", False),
        ("network_module", "networks.lora_qwen_image"),
        ("network_dim", 32),
        ("network_alpha", 32),
        ("network_dropout", 0),
        ("network_args", []),
        ("sdpa", True),
        ("additional_parameters", ""),
        ("mem_eff_save", True),
    ]

    qwen_image_lora_gui.train_qwen_image_model(True, True, parameters)

    command = captured["command"]
    config = captured["config"]
    assert any(str(part).endswith("qwen_image_train.py") for part in command)
    assert not any(str(part).endswith("qwen_image_train_network.py") for part in command)
    assert config.get("fp8_base", False) is False
    assert config.get("fp8_scaled", False) is False
    assert config.get("dit_dtype", "bfloat16") == "bfloat16"
    assert config["save_precision"] == "bf16"
    assert config["optimizer_type"] == "Automagic3"
    assert config["mem_eff_save"] is True
    assert not FULL_FINE_TUNING_NETWORK_KEYS.intersection(config)
    assert "block_swap_h2d_only" not in config


def test_shipped_qwen_default_enables_resident_compile():
    defaults_path = Path(__file__).resolve().parents[1] / "qwen_image_defaults.toml"

    assert toml.load(defaults_path)["compile_resident_blocks_only"] is True


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, True),
        ({"compile_resident_blocks_only": True}, True),
        ({"compile_resident_blocks_only": False}, False),
    ],
)
def test_qwen_resident_compile_default_preserves_config_override(config, expected):
    with gr.Blocks():
        model = qwen_image_lora_gui.QwenImageModel(headless=True, config=config)

    assert model.compile_resident_blocks_only.value is expected


def test_qwen_resident_compile_callback_order_and_search_mapping_are_wired():
    parameter_names = list(inspect.signature(qwen_image_lora_gui.qwen_image_gui_actions).parameters)
    compile_start = parameter_names.index("compile")
    assert parameter_names[compile_start : compile_start + 7] == [
        "compile",
        "compile_resident_blocks_only",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_cache_size_limit",
    ]

    tab_source = inspect.getsource(qwen_image_lora_gui.qwen_image_lora_tab)
    component_names = [
        "compile",
        "compile_resident_blocks_only",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_cache_size_limit",
    ]
    positions = [tab_source.index(f"qwen_model.{name},") for name in component_names]
    assert positions == sorted(positions)
    assert (
        '"compile_resident_blocks_only": ("Qwen Image Model Settings", "Compile Resident Blocks Only")'
        in tab_source
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_qwen_resident_compile_persistent_config_round_trip(tmp_path: Path, enabled):
    config_path = tmp_path / "qwen.toml"
    parameters = [
        ("training_mode", "LoRA Training"),
        ("network_module", qwen_image_lora_gui.QWEN_IMAGE_NETWORK_MODULE),
        ("fused_backward_pass", False),
        ("compile_resident_blocks_only", enabled),
    ]

    qwen_image_lora_gui.save_qwen_image_configuration(False, str(config_path), parameters)

    assert toml.load(config_path)["compile_resident_blocks_only"] is enabled
    loaded = qwen_image_lora_gui.open_qwen_image_configuration(
        False,
        str(config_path),
        [("compile_resident_blocks_only", not enabled)],
    )
    assert loaded[2] is enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_qwen_resident_compile_reaches_training_preview_config(tmp_path: Path, monkeypatch, enabled):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]\nimage_directory = 'images'\n", encoding="utf-8")
    captured = {}

    def capture_preview(command, config_path):
        captured["command"] = command
        captured["config"] = toml.load(config_path)

    monkeypatch.setattr(qwen_image_lora_gui, "print_command_and_toml", capture_preview)
    parameters = [
        ("training_mode", "LoRA Training"),
        ("dataset_config_mode", "Use TOML File"),
        ("dataset_config", str(dataset)),
        ("model_version", "original"),
        ("dit", _touch(tmp_path / "dit.safetensors")),
        ("vae", _touch(tmp_path / "vae.safetensors")),
        ("text_encoder", _touch(tmp_path / "text_encoder.safetensors")),
        ("output_dir", str(tmp_path / "output")),
        ("output_name", f"qwen-resident-{enabled}"),
        ("mixed_precision", "bf16"),
        ("gradient_checkpointing", True),
        ("blocks_to_swap", 6),
        ("block_swap_h2d_only", False),
        ("block_swap_ring_size", 2),
        ("num_processes", 1),
        ("num_machines", 1),
        ("num_cpu_threads_per_process", 1),
        ("gpu_ids", "0"),
        ("dynamo_backend", "no"),
        ("network_module", qwen_image_lora_gui.QWEN_IMAGE_NETWORK_MODULE),
        ("network_dim", 16),
        ("network_alpha", 16),
        ("network_dropout", 0),
        ("network_args", []),
        ("compile", True),
        ("compile_backend", "inductor"),
        ("compile_mode", "default"),
        ("compile_dynamic", "auto"),
        ("compile_fullgraph", False),
        ("compile_cache_size_limit", 0),
        ("compile_resident_blocks_only", enabled),
        ("sdpa", True),
        ("additional_parameters", ""),
    ]

    qwen_image_lora_gui.train_qwen_image_model(True, True, parameters)

    assert any(str(part).endswith("qwen_image_train_network.py") for part in captured["command"])
    if enabled:
        assert captured["config"]["compile_resident_blocks_only"] is True
    else:
        assert "compile_resident_blocks_only" not in captured["config"]


def test_zimage_dreambooth_preview_normalizes_full_model_runtime_config(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]\nimage_directory = 'images'\n", encoding="utf-8")
    captured: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        zimage_lora_gui,
        "print_command_and_toml",
        lambda command, config_path: captured.append((command, config_path)),
    )
    parameters = [
        ("training_mode", "DreamBooth Fine-Tuning"),
        ("dataset_config_mode", "Use TOML File"),
        ("dataset_config", str(dataset)),
        ("dit", _touch(tmp_path / "dit.safetensors")),
        ("vae", _touch(tmp_path / "vae.safetensors")),
        ("text_encoder", _touch(tmp_path / "text_encoder.safetensors")),
        ("output_dir", str(tmp_path / "output")),
        ("output_name", "zimage-full-preview"),
        ("mixed_precision", "bf16"),
        ("full_bf16", True),
        ("full_fp16", False),
        ("save_precision", "fp32"),
        ("fp8_base", True),
        ("fp8_scaled", True),
        ("block_swap_h2d_only", True),
        ("block_swap_ring_size", 2),
        ("blocks_to_swap", 6),
        ("gradient_checkpointing", True),
        ("gradient_accumulation_steps", 1),
        ("num_processes", 1),
        ("num_machines", 1),
        ("num_cpu_threads_per_process", 1),
        ("gpu_ids", "0"),
        ("dynamo_backend", "no"),
        ("optimizer_type", "Automagic3"),
        ("optimizer_args", []),
        ("fused_backward_pass", False),
        ("block_swap_optimizer_patch_params", False),
        ("network_module", "networks.lora_zimage"),
        ("network_dim", 32),
        ("network_alpha", 32),
        ("network_dropout", 0),
        ("network_args", []),
        ("base_weights", ["old-adapter.safetensors"]),
        ("base_weights_multiplier", [1.0]),
        ("sdpa", True),
        ("caching_latent_skip_existing", False),
        ("caching_teo_skip_existing", False),
        ("additional_parameters", ""),
        ("mem_eff_save", True),
    ]

    zimage_lora_gui.train_zimage_model(True, True, parameters)

    command, config_path = captured[-1]
    config = toml.load(config_path)
    assert any(str(part).endswith("zimage_train.py") for part in command)
    assert not any(str(part).endswith("zimage_train_network.py") for part in command)
    assert config.get("fp8_base", False) is False
    assert config.get("fp8_scaled", False) is False
    assert config["save_precision"] == "bf16"
    assert config["optimizer_type"] == "Automagic3"
    assert config["mem_eff_save"] is True
    assert not FULL_FINE_TUNING_NETWORK_KEYS.intersection(config)
    assert "block_swap_h2d_only" not in config


def test_flux_accelerate_lookup_prefers_the_active_python_environment(tmp_path: Path, monkeypatch):
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    python = scripts / "python.exe"
    accelerate = scripts / "accelerate.exe"
    python.touch()
    accelerate.touch()
    monkeypatch.setattr(flux2_lora_gui.shutil, "which", lambda _name: r"C:\Python310\Scripts\accelerate.exe")

    command = flux2_lora_gui._find_accelerate_launch(str(python))

    assert Path(command[0]) == accelerate
    assert command[1] == "launch"


def test_custom_config_is_isolated_between_gui_tabs(tmp_path: Path):
    config_path = tmp_path / "custom.toml"
    config_path.write_text('training_mode = "Full Fine-Tuning"\n[nested]\nvalue = 1\n', encoding="utf-8")
    manager = TabConfigManager(str(config_path))

    wan_config = manager.get_config_for_tab("wan")
    flux_config = manager.get_config_for_tab("flux")
    wan_config.config["max_timestep"] = 0
    wan_config.config["nested"]["value"] = 2

    assert wan_config is not flux_config
    assert "max_timestep" not in flux_config.config
    assert flux_config.config["nested"]["value"] == 1
    assert "max_timestep" not in manager.base_config.config


@pytest.mark.parametrize(
    "spec_key, expected_script",
    [("ideogram4", "ideogram4_train.py"), ("krea2", "krea2_train.py")],
)
def test_modern_full_finetune_workflow_selects_full_trainer_and_strips_lora(
    tmp_path: Path,
    spec_key: str,
    expected_script: str,
):
    spec = get_architecture(spec_key)
    values = _architecture_defaults(spec)
    values.update(
        {
            "training_mode": FULL_FINE_TUNING_MODE,
            "mixed_precision": "bf16",
            "full_bf16": True,
            "full_fp16": False,
            "fused_backward_pass": True,
            "block_swap_optimizer_patch_params": False,
            "optimizer_type": "AdaFactor",
            "gradient_accumulation_steps": 1,
            "fp8_base": True,
            "fp8_scaled": True,
            "block_swap_h2d_only": True,
            "use_legacy_sdpa": True,
            "dataset_config": _touch(tmp_path / "dataset.toml"),
            "dit": _touch(tmp_path / "dit.safetensors"),
            "vae": _touch(tmp_path / "vae.safetensors"),
            "text_encoder": _touch(tmp_path / "text_encoder.safetensors"),
            "output_dir": str(tmp_path / "output"),
            "output_name": "modern-full",
            "save_precision": "bf16",
            "dit_variant": "raw",
            "turbo_dit": "",
            "turbo_dit_cache": False,
            "blocks_to_swap": 32 if spec.is_ideogram else spec.max_blocks_to_swap,
        }
    )
    config_path = tmp_path / "runtime.toml"

    workflow = prepare_modern_image_workflow(
        spec_key,
        list(values.items()),
        str(config_path),
        python_cmd="python",
    )
    runtime = toml.load(config_path)

    assert Path(workflow.train_command[-3]).name == expected_script
    assert workflow.train_command[-2:] == ["--config_file", str(config_path)]
    assert runtime["full_bf16"] is True
    assert runtime["fused_backward_pass"] is True
    assert runtime["use_legacy_sdpa"] is True
    assert runtime["save_precision"] == "bf16"
    assert "network_module" not in runtime
    assert "network_dim" not in runtime
    assert "fp8_base" not in runtime
    assert "block_swap_h2d_only" not in runtime


@pytest.mark.parametrize("spec_key", ["ideogram4", "krea2"])
def test_modern_legacy_sdpa_persists_in_runtime_config(tmp_path: Path, spec_key: str):
    spec = get_architecture(spec_key)
    values = _architecture_defaults(spec)
    values.update(
        {
            "dataset_config": _touch(tmp_path / "dataset.toml"),
            "dit": _touch(tmp_path / "dit.safetensors"),
            "vae": _touch(tmp_path / "vae.safetensors"),
            "text_encoder": _touch(tmp_path / "text_encoder.safetensors"),
            "output_dir": str(tmp_path / "output"),
            "output_name": "legacy-sdpa",
            "use_legacy_sdpa": True,
        }
    )
    config_path = tmp_path / "runtime.toml"

    workflow = prepare_modern_image_workflow(spec_key, list(values.items()), str(config_path), python_cmd="python")
    runtime = toml.load(config_path)

    assert dict(workflow.parameters)["use_legacy_sdpa"] is True
    assert runtime["use_legacy_sdpa"] is True
    assert _architecture_defaults(spec)["use_legacy_sdpa"] is False


@pytest.mark.parametrize(
    "operating_system",
    ["Windows", "Linux"],
)
def test_workflow_script_does_not_mutate_attention_environment(tmp_path: Path, monkeypatch, operating_system):
    monkeypatch.setattr(modern_image_lora_gui.platform, "system", lambda: operating_system)
    workflow = ModernImageWorkflow(
        parameters=[("use_legacy_sdpa", True)],
        config_path=str(tmp_path / "runtime.toml"),
        latent_cache_command=None,
        text_cache_command=None,
        train_command=["python", "train.py"],
    )

    script_path, content = _build_workflow_script(get_architecture("krea2"), workflow)
    Path(script_path).unlink()

    assert "MUSUBI_DISABLE_EXTERNAL_FLASH_SDPA" not in content


@pytest.mark.parametrize(
    "module, trainer, keys, model_version, model_family",
    [
        (flux2_lora_gui, flux2_lora_gui.train_flux2_model, flux2_lora_gui.FLUX2_PARAM_KEYS, "dev", None),
        (
            flux_klein_lora_gui,
            flux_klein_lora_gui.train_flux_klein_model,
            flux2_lora_gui.FLUX2_PARAM_KEYS,
            "klein-base-4b",
            None,
        ),
        (flux_lora_gui, flux_lora_gui.train_flux_model, flux_lora_gui.FLUX_PARAM_KEYS, "dev", "FLUX.2"),
        (
            flux_lora_gui,
            flux_lora_gui.train_flux_model,
            flux_lora_gui.FLUX_PARAM_KEYS,
            "klein-base-4b",
            "FLUX Klein",
        ),
    ],
)
def test_flux_full_finetune_runtime_uses_full_script_and_valid_config(
    tmp_path: Path,
    monkeypatch,
    module,
    trainer,
    keys: list[str],
    model_version: str,
    model_family: str | None,
):
    values = _base_full_values(tmp_path, keys, model_version=model_version)
    if model_family:
        values["model_family"] = model_family
    fake_executor = _FakeExecutor()
    monkeypatch.setattr(module, "executor", fake_executor)
    monkeypatch.setattr(module, "save_executed_script", lambda **_kwargs: None)
    monkeypatch.setattr(module, "setup_environment", lambda **_kwargs: {})

    trainer(
        headless=True,
        print_only=False,
        parameters=_ordered_parameters(keys, values),
    )

    assert len(fake_executor.calls) == 1
    command = fake_executor.calls[0]["run_cmd"]
    assert any(Path(str(part)).name == "flux_2_train.py" for part in command)
    runtime_files = list((tmp_path / "output").glob("full-smoke_*.toml"))
    assert len(runtime_files) == 1
    runtime = toml.load(runtime_files[0])
    assert runtime["full_bf16"] is True
    assert runtime["fused_backward_pass"] is True
    assert runtime["use_legacy_sdpa"] is True
    assert runtime["save_precision"] == "bf16"
    assert runtime["vae_dtype"] == "float32"
    assert "training_mode" not in runtime
    assert "network_module" not in runtime
    assert "network_dim" not in runtime


def test_wan_compile_callback_parameter_order_includes_residency_control():
    assert wan_lora_gui.WAN_COMPILE_PARAMETER_NAMES == (
        "compile",
        "compile_resident_blocks_only",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_fullgraph",
        "compile_cache_size_limit",
    )


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("t2v-A14B", True),
        ("i2v-A14B", True),
        ("t2v-14B", False),
        ("i2v-14B", False),
    ],
)
def test_wan_compile_residency_default_is_wan22_only(task: str, expected: bool):
    assert wan_lora_gui.wan_compile_resident_blocks_default(task) is expected


@pytest.mark.parametrize(
    ("task", "explicit_value", "expected"),
    [
        ("t2v-A14B", None, True),
        ("t2v-14B", None, False),
        ("t2v-A14B", False, False),
        ("t2v-14B", True, True),
    ],
)
def test_wan_model_settings_control_uses_task_aware_compile_default(
    task: str,
    explicit_value: bool | None,
    expected: bool,
):
    values = {"task": task}
    if explicit_value is not None:
        values["compile_resident_blocks_only"] = explicit_value
    config = SimpleNamespace(config=values, get=values.get)

    with gr.Blocks():
        settings = wan_lora_gui.WanModelSettings(headless=True, config=config)

    assert settings.compile_resident_blocks_only.value is expected
    assert settings.compile_resident_blocks_only_overridden.value is (
        explicit_value is not None
    )


@pytest.mark.parametrize(
    ("task", "explicit_value"),
    [
        ("t2v-A14B", False),
        ("t2v-14B", True),
    ],
)
def test_wan_compile_residency_default_preserves_explicit_override(
    task: str,
    explicit_value: bool,
):
    parameters = [
        ("task", task),
        ("compile_resident_blocks_only", explicit_value),
    ]

    normalized = wan_lora_gui.apply_wan_compile_residency_default(parameters)

    assert dict(normalized)["compile_resident_blocks_only"] is explicit_value
    assert sum(key == "compile_resident_blocks_only" for key, _ in normalized) == 1


@pytest.mark.parametrize(
    ("task", "configured_value", "expected"),
    [
        ("t2v-A14B", None, True),
        ("t2v-14B", None, False),
        ("t2v-A14B", False, False),
        ("t2v-14B", True, True),
    ],
)
def test_wan_load_config_uses_task_compile_default_only_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    configured_value: bool | None,
    expected: bool,
):
    monkeypatch.setattr(wan_lora_gui.gr, "Info", lambda *_args, **_kwargs: None)
    config_path = tmp_path / "wan.toml"
    config = {"task": task}
    if configured_value is not None:
        config["compile_resident_blocks_only"] = configured_value
    config_path.write_text(toml.dumps(config), encoding="utf-8")

    result = wan_lora_gui.open_wan_configuration(
        False,
        str(config_path),
        [
            ("task", "t2v-14B"),
            ("compile_resident_blocks_only", False),
        ],
    )

    assert result[2] == task
    assert result[3] is expected
    assert wan_lora_gui.wan_config_has_compile_residency_override(config_path) is (
        configured_value is not None
    )


@pytest.mark.parametrize(
    ("task", "explicit_value", "expected"),
    [
        ("t2v-A14B", None, True),
        ("t2v-14B", None, False),
        ("t2v-A14B", False, False),
        ("t2v-14B", True, True),
    ],
)
def test_wan_training_preview_serializes_effective_compile_residency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    explicit_value: bool | None,
    expected: bool,
):
    captured: list[str] = []
    monkeypatch.setattr(
        wan_lora_gui,
        "print_command_and_toml",
        lambda _command, config_path: captured.append(config_path) if config_path else None,
    )

    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[[datasets]]\nimage_directory = 'images'\n", encoding="utf-8")
    parameters = [
        ("training_mode", "LoRA Training"),
        ("task", task),
        ("dataset_config_mode", "Use TOML File"),
        ("dataset_config", str(dataset)),
        ("dit", str(tmp_path / "dit.safetensors")),
        ("vae", str(tmp_path / "vae.safetensors")),
        ("t5", str(tmp_path / "t5.safetensors")),
        ("clip", str(tmp_path / "clip.safetensors")),
        ("output_name", "wan-compile-default"),
        ("network_module", "networks.lora_wan"),
        ("compile", True),
        ("compile_backend", "inductor"),
        ("compile_mode", "default"),
        ("compile_dynamic", "auto"),
        ("caching_latent_skip_existing", False),
        ("caching_teo_skip_existing", False),
        ("dynamo_backend", "no"),
        ("additional_parameters", ""),
    ]
    if explicit_value is not None:
        parameters.append(("compile_resident_blocks_only", explicit_value))

    wan_lora_gui.train_wan_model(True, True, parameters)

    assert captured
    runtime_config = toml.load(captured[-1])
    if expected:
        assert runtime_config["compile_resident_blocks_only"] is True
    else:
        assert "compile_resident_blocks_only" not in runtime_config


def test_wan_saved_gui_config_keeps_explicit_compile_residency_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(wan_lora_gui.gr, "Info", lambda *_args, **_kwargs: None)
    config_path = tmp_path / "wan-explicit.toml"

    wan_lora_gui.save_wan_configuration(
        False,
        str(config_path),
        [
            ("training_mode", "LoRA Training"),
            ("task", "t2v-A14B"),
            ("compile_resident_blocks_only", False),
        ],
    )

    saved = toml.load(config_path)
    assert saved["compile_resident_blocks_only"] is False


@pytest.mark.parametrize(
    ("task", "current_value", "overridden", "expected"),
    [
        ("t2v-A14B", False, False, True),
        ("t2v-14B", True, False, False),
        ("t2v-A14B", False, True, False),
        ("t2v-14B", True, True, True),
    ],
)
def test_wan_task_switch_updates_only_automatic_compile_default(
    task: str,
    current_value: bool,
    overridden: bool,
    expected: bool,
):
    result = wan_lora_gui.WanModelSettings._update_compile_residency_default(
        task,
        current_value,
        overridden,
    )

    assert result is expected
