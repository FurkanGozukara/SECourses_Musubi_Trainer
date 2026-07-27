from unittest.mock import patch

import pytest

from musubi_tuner.torch_compile_toolchain import CompileToolchainStatus
from musubi_tuner_gui.common_gui import setup_environment


def test_gui_continues_with_eager_training_when_compile_toolchain_is_unavailable(
    monkeypatch,
):
    monkeypatch.delenv("MUSUBI_TORCH_COMPILE_FALLBACK", raising=False)
    unavailable = CompileToolchainStatus(False, "synthetic compiler failure")

    with patch(
        "musubi_tuner_gui.common_gui.ensure_compile_environment",
        return_value=unavailable,
    ):
        env = setup_environment(compile_requested=True)

    assert env["MUSUBI_TORCH_COMPILE_READY"] == "0"
    assert env["MUSUBI_TORCH_COMPILE_ACTIVE"] == "0"
    assert env["MUSUBI_TORCH_COMPILE_DETAIL"] == unavailable.detail


def test_gui_strict_compile_mode_can_still_fail_fast(monkeypatch):
    monkeypatch.setenv("MUSUBI_TORCH_COMPILE_FALLBACK", "0")
    unavailable = CompileToolchainStatus(False, "synthetic compiler failure")

    with (
        patch(
            "musubi_tuner_gui.common_gui.ensure_compile_environment",
            return_value=unavailable,
        ),
        pytest.raises(RuntimeError, match="toolchain unavailable"),
    ):
        setup_environment(compile_requested=True)
