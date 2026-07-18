import ast
import io
import sys
from pathlib import Path

import musubi_tuner_gui.model_quantizer_gui as model_quantizer_gui
import pytest

from musubi_tuner_gui.common_gui import utf8_subprocess_options
from musubi_tuner_gui.model_quantizer_gui import (
    MODEL_PRESET_NONE,
    MODEL_PRESET_PRIMARY_CHOICES,
    PRESET_INT8_CONVROT_HQ,
    QUANT_FORMAT_INT8,
    WORKFLOW_QUANTIZE,
    ModelQuantizer,
    _combined_preset_settings,
    _model_preset_filters,
    _model_preset_label,
    _model_preset_value,
)


LTX_2_3_LABEL = "LTX 2.3 (INT8 ConvRot Learned HQ)"


def _flag_value(command, flag):
    return command[command.index(flag) + 1]


def test_ltx_2_3_hq_is_a_distinct_primary_model_preset():
    assert _model_preset_label("ltx2_3") == LTX_2_3_LABEL
    assert _model_preset_value(LTX_2_3_LABEL) == "ltx2_3"
    assert LTX_2_3_LABEL in MODEL_PRESET_PRIMARY_CHOICES

    # Preserve the legacy combined preset and its stored label.
    assert _model_preset_value("LTX (2 / 2.3)") == "ltxv2"
    assert _model_preset_value(MODEL_PRESET_NONE) == MODEL_PRESET_NONE


def test_ltx_2_3_hq_resolves_the_learned_fixed_convrot_recipe():
    effective_preset, settings = _combined_preset_settings(
        "ltx2_3",
        None,
        use_model_default_preset=True,
    )

    assert effective_preset == PRESET_INT8_CONVROT_HQ
    assert settings["quant_format"] == QUANT_FORMAT_INT8
    assert settings["comfy_quant"] is True
    assert settings["scaling_mode"] == "row"
    assert settings["block_size"] == 128
    assert settings["convrot"] is True
    assert settings["convrot_group_size"] == 256
    assert settings["dynamic_convrot"] is False
    assert settings["simple"] is False
    assert settings["calib_samples"] == 2048
    assert settings["optimizer"] == "prodigy"
    assert settings["num_iter"] == 2000
    assert settings["lr"] == 1.0
    assert settings["lr_schedule"] == "adaptive"
    assert settings["lr_factor"] == 0.965
    assert settings["lr_cooldown"] == 1
    assert settings["top_p"] == 0.2
    assert settings["min_k"] == 128
    assert settings["max_k"] == 1280
    assert settings["scale_optimization"] == "fixed"
    assert settings["scale_refinement_rounds"] == 1
    assert settings["full_matrix"] is False
    assert settings["full_precision_matrix_mult"] is False
    assert settings["low_memory"] is True
    assert settings["save_quant_metadata"] is True
    assert _model_preset_filters("ltx2_3") == {"ltx2_3"}


def test_ltx_2_3_hq_command_uses_native_convrot_and_ignores_layer_config():
    _, params = _combined_preset_settings(
        "ltx2_3",
        None,
        use_model_default_preset=True,
    )
    params.update(
        {
            "workflow": WORKFLOW_QUANTIZE,
            "model_preset": "ltx2_3",
            "model_filters": {"ltx2_3": True},
            "layer_config_path": "stale-layer-config.json",
            "layer_config_fullmatch": True,
        }
    )

    command = ModelQuantizer(headless=True, config=None)._build_command(
        "ltx-2.3-22b-distilled-1.1.safetensors",
        "ltx-2.3-22b-distilled-1.1-int8-convrot-hq.safetensors",
        params,
    )

    assert "--comfy_quant" in command
    assert "--int8" in command
    assert "--convrot" in command
    assert _flag_value(command, "--convrot-group-size") == "256"
    assert _flag_value(command, "--scaling_mode") == "row"
    assert _flag_value(command, "--block_size") == "128"
    assert _flag_value(command, "--calib_samples") == "2048"
    assert _flag_value(command, "--optimizer") == "prodigy"
    assert _flag_value(command, "--num_iter") == "2000"
    assert _flag_value(command, "--scale-refinement") == "1"
    assert _flag_value(command, "--scale-optimization") == "fixed"
    assert "--ltx2_3" in command
    assert "--simple" not in command
    assert "--dynamic-convrot" not in command
    assert "--layer-config" not in command
    assert "--fullmatch" not in command


def test_manual_no_model_preset_keeps_custom_layer_config_without_model_filter():
    command = ModelQuantizer(headless=True, config=None)._build_command(
        "model.safetensors",
        "model-int8.safetensors",
        {
            "workflow": WORKFLOW_QUANTIZE,
            "model_preset": MODEL_PRESET_NONE,
            "model_filters": {},
            "quant_format": QUANT_FORMAT_INT8,
            "layer_config_path": "custom-layer-config.json",
            "layer_config_fullmatch": True,
        },
    )

    assert "--int8" in command
    assert _flag_value(command, "--layer-config") == "custom-layer-config.json"
    assert "--fullmatch" in command
    assert "--ltx2_3" not in command


def test_utf8_subprocess_options_override_host_locale_without_mutating_input():
    original_env = {
        "PYTHONIOENCODING": "cp1252",
        "PYTHONUTF8": "0",
        "KEEP_ME": "yes",
    }

    options = utf8_subprocess_options(original_env)

    assert original_env["PYTHONIOENCODING"] == "cp1252"
    assert options["text"] is True
    assert options["encoding"] == "utf-8"
    assert options["errors"] == "replace"
    assert options["env"]["PYTHONIOENCODING"] == "utf-8:backslashreplace"
    assert options["env"]["PYTHONUTF8"] == "1"
    assert options["env"]["KEEP_ME"] == "yes"


def test_progress_output_survives_non_utf8_console(monkeypatch):
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", stream)
        ModelQuantizer._write_stdout(f"progress {chr(0x258F)}")

    assert buffer.getvalue() == b"progress \\u258f"


def test_all_gui_text_subprocesses_declare_a_non_throwing_decoder():
    gui_root = Path(model_quantizer_gui.__file__).parent
    missing_policy = []

    for source_path in gui_root.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "subprocess":
                continue

            keyword_values = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
            if node.func.attr == "Popen":
                stdout = keyword_values.get("stdout")
                captures_stdout = (
                    isinstance(stdout, ast.Attribute)
                    and isinstance(stdout.value, ast.Name)
                    and stdout.value.id == "subprocess"
                    and stdout.attr == "PIPE"
                )
                if not captures_stdout:
                    continue
                uses_shared_policy = any(
                    keyword.arg is None
                    and isinstance(keyword.value, ast.Call)
                    and isinstance(keyword.value.func, ast.Name)
                    and keyword.value.func.id == "utf8_subprocess_options"
                    for keyword in node.keywords
                )
                if not uses_shared_policy:
                    missing_policy.append(f"{source_path.name}:{node.lineno}")
            elif node.func.attr == "run":
                text_mode = isinstance(keyword_values.get("text"), ast.Constant) and keyword_values["text"].value is True
                captures_output = (
                    isinstance(keyword_values.get("capture_output"), ast.Constant)
                    and keyword_values["capture_output"].value is True
                )
                if text_mode and captures_output and "errors" not in keyword_values:
                    missing_policy.append(f"{source_path.name}:{node.lineno}")

    assert missing_policy == []


@pytest.mark.parametrize(
    "config",
    [None, object()],
    ids=["manual-no-config", "config-driven"],
)
def test_run_command_decodes_utf8_and_invalid_bytes_from_real_subprocess(monkeypatch, config):
    monkeypatch.setattr(model_quantizer_gui, "save_executed_script", lambda **_kwargs: None)
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(b'Optimizing (AdaRound-prodigy-adaptive): 1%|\\xe2\\x96\\x8f| 16/2000\\nNative byte: \\xff\\n')",
    ]

    output, return_code = ModelQuantizer(headless=True, config=config)._run_command(
        command,
        "single_process",
        "UTF-8 subprocess regression",
    )

    assert return_code == 0
    assert output.splitlines() == [
        f"Optimizing (AdaRound-prodigy-adaptive): 1%|{chr(0x258F)}| 16/2000",
        f"Native byte: {chr(0xFFFD)}",
    ]
