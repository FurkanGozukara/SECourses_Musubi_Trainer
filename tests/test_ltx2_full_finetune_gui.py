"""LTX 2.3 full fine-tuning mode of the LTX tab (ltx2_train.py + bundle merge)."""

import os
import sys

import gradio as gr
import pytest
import toml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui import ltx2_lora_gui as ltx
from musubi_tuner_gui.full_finetune_gui import FULL_FINE_TUNING_MODE, LORA_TRAINING_MODE


def _params(tmp_path, **overrides):
    values = dict(ltx.LTX2_DEFAULTS)
    values.update(
        {
            "ltx2_checkpoint": str(tmp_path / "ltx-2.3-22b-dev.safetensors"),
            "gemma_safetensors": str(tmp_path / "gemma_3_12B_it_fp8_e4m3fn.safetensors"),
            "dataset_config": str(tmp_path / "dataset.toml"),
            "output_dir": str(tmp_path / "out"),
            "output_name": "ltx_ft",
        }
    )
    values.update(overrides)
    return [(key, values[key]) for key in ltx.LTX2_PARAM_KEYS]


def _full(tmp_path, **overrides):
    base = {"training_mode": FULL_FINE_TUNING_MODE, "blocks_to_swap": 24, **ltx.LTX2_FULL_FINE_TUNING_DEFAULTS}
    base.update(overrides)
    return _params(tmp_path, **base)


def test_full_finetune_workflow_uses_ltx2_train_and_merges_bundle(tmp_path):
    config_path = str(tmp_path / "run.toml")
    workflow = ltx.prepare_ltx2_workflow(_full(tmp_path), config_path, validate_paths=False, python_cmd="python")
    assert workflow.train_command[-3].endswith("ltx2_train.py")
    data = toml.load(config_path)
    assert data["training_mode"] == FULL_FINE_TUNING_MODE
    assert data["full_bf16"] is True and data["fused_backward_pass"] is True
    assert data["ltx2_block_swap_trainable_ring"] is True
    for key in ("network_module", "network_dim", "network_alpha", "lora_target_preset", "train_connectors", "block_swap_h2d_only"):
        assert key not in data, key
    assert len(workflow.post_train_commands) == 1
    merge = workflow.post_train_commands[0]
    assert merge[1].endswith("ltx2_merge_fft_to_model.py")
    assert merge[merge.index("--fft_checkpoint") + 1].endswith("ltx_ft.safetensors")
    assert merge[merge.index("--output") + 1].endswith("ltx_ft_ltx2_bundle.safetensors")
    # the training command stays last in .commands (tools rely on it); the merge runs after it
    assert workflow.commands[-1] is workflow.train_command
    assert workflow.all_commands[-1] is merge


def test_lora_workflow_keeps_lora_script_and_drops_full_only_keys(tmp_path):
    config_path = str(tmp_path / "run.toml")
    workflow = ltx.prepare_ltx2_workflow(_params(tmp_path), config_path, validate_paths=False, python_cmd="python")
    assert workflow.train_command[-3].endswith("ltx2_train_network.py")
    assert workflow.post_train_commands == []
    data = toml.load(config_path)
    assert data["network_module"] == "networks.lora_ltx2"
    assert data["training_mode"] == LORA_TRAINING_MODE
    for key in ("full_bf16", "fused_backward_pass", "ltx2_block_swap_trainable_ring", "mem_eff_save", "save_merged_checkpoint"):
        assert key not in data, key


def test_merged_checkpoint_saving_replaces_the_merge_step(tmp_path):
    workflow = ltx.prepare_ltx2_workflow(
        _full(tmp_path, save_merged_checkpoint=True), str(tmp_path / "run.toml"), validate_paths=False, python_cmd="python"
    )
    assert workflow.post_train_commands == []


def test_full_finetune_rejects_quantized_base_and_fixes_swap_flags(tmp_path):
    with pytest.raises(ValueError, match="dense BF16"):
        ltx._normalize_ltx2_parameters(_full(tmp_path, int8_convrot_dynamic=True), validate_paths=False)
    param_dict, _ = ltx._normalize_ltx2_parameters(
        _full(tmp_path, blocks_to_swap=0, block_swap_h2d_only=True, ltx2_block_swap_trainable_ring=True), validate_paths=False
    )
    assert param_dict["block_swap_h2d_only"] is False
    assert param_dict["ltx2_block_swap_trainable_ring"] is False


def test_full_finetune_fused_backward_requires_adafactor(tmp_path):
    with pytest.raises(ValueError, match="Adafactor"):
        ltx._normalize_ltx2_parameters(_full(tmp_path, optimizer_type="AdamW8bit"), validate_paths=False)


def test_training_mode_round_trip(tmp_path):
    config_path = str(tmp_path / "run.toml")
    ltx.prepare_ltx2_workflow(_full(tmp_path), config_path, validate_paths=False, python_cmd="python")
    lora_screen = _params(tmp_path)  # GUI currently shows LoRA

    def loaded_mode(path):
        result = ltx.open_ltx2_configuration(False, path, lora_screen)
        return result[2 + ltx.LTX2_PARAM_KEYS.index("training_mode")]

    assert loaded_mode(config_path) == FULL_FINE_TUNING_MODE
    legacy = toml.load(config_path)
    legacy.pop("training_mode")
    legacy_path = str(tmp_path / "legacy.toml")
    with open(legacy_path, "w", encoding="utf-8") as handle:
        toml.dump(legacy, handle)
    assert loaded_mode(legacy_path) == FULL_FINE_TUNING_MODE  # inferred from the missing network_module
    lora_path = str(tmp_path / "lora.toml")
    with open(lora_path, "w", encoding="utf-8") as handle:
        toml.dump({"network_module": "networks.lora_ltx2", "network_dim": 32}, handle)
    full_screen = _full(tmp_path)
    result = ltx.open_ltx2_configuration(False, lora_path, full_screen)
    assert result[2 + ltx.LTX2_PARAM_KEYS.index("training_mode")] == LORA_TRAINING_MODE


def test_workflow_script_labels_merge_step(tmp_path):
    workflow = ltx.prepare_ltx2_workflow(_full(tmp_path), str(tmp_path / "run.toml"), validate_paths=False, python_cmd="python")
    _path, content = ltx._build_workflow_script(workflow)
    assert "training" in content and "bundle merge" in content
    assert content.index("ltx2_train.py") < content.index("ltx2_merge_fft_to_model.py")


def test_ltx_tab_builds_with_training_mode():
    with gr.Blocks():
        settings = ltx.ltx2_lora_tab(headless=True, config={})
    assert len(settings) == len(ltx.LTX2_PARAM_KEYS)


def test_cpu_text_cache_with_fp8_single_file_gemma_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="FP8 Gemma"):
        ltx._normalize_ltx2_parameters(_params(tmp_path, caching_teo_device="cpu"), validate_paths=False)
    # cuda (default) is fine
    ltx._normalize_ltx2_parameters(_params(tmp_path, caching_teo_device="cuda"), validate_paths=False)