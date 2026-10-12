"""LoRA-to-LoRA merging adds tensors key by key: DoRA files are refused (their magnitudes would be summed)."""

import torch
from safetensors.torch import save_file

from musubi_tuner_gui.lora_merge_worker import _merge_loras_only


def _write(path, dora: bool):
    state = {"lora_unet_a.lora_down.weight": torch.ones(2, 4), "lora_unet_a.lora_up.weight": torch.ones(4, 2), "lora_unet_a.alpha": torch.tensor(2.0)}
    if dora:
        state["lora_unet_a.dora_scale"] = torch.ones(1, 4)
    save_file(state, str(path))
    return str(path)


def test_dora_files_are_refused(tmp_path):
    status, message = _merge_loras_only(
        [_write(tmp_path / "a.safetensors", False), _write(tmp_path / "b.safetensors", True)],
        [0.5, 0.5],
        str(tmp_path / "merged.safetensors"),
        overwrite_existing=True,
        target_dtype_choice=None,
    )
    assert status == "error" and "DoRA" in message
    assert not (tmp_path / "merged.safetensors").exists()


def test_plain_loras_still_merge(tmp_path):
    status, _ = _merge_loras_only(
        [_write(tmp_path / "a.safetensors", False), _write(tmp_path / "b.safetensors", False)],
        [0.5, 0.5],
        str(tmp_path / "merged.safetensors"),
        overwrite_existing=True,
        target_dtype_choice=None,
    )
    assert status == "success"
