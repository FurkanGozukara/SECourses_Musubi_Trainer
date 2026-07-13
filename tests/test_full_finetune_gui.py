from __future__ import annotations

from pathlib import Path

import pytest
import toml

from musubi_tuner_gui import flux2_lora_gui, flux_klein_lora_gui, flux_lora_gui, modern_image_lora_gui
from musubi_tuner_gui.full_finetune_gui import (
    FULL_FINE_TUNING_MODE,
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
    assert "training_mode" not in runtime
    assert "network_module" not in runtime
    assert "network_dim" not in runtime
