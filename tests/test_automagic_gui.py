import gradio as gr
import pytest

from musubi_tuner_gui.class_optimizer_and_scheduler import OptimizerAndScheduler
from musubi_tuner_gui.class_gui_config import GUIConfig
from musubi_tuner_gui.optimizer_catalog import (
    AUTOMAGIC_OPTIMIZER_CHOICES,
    add_automagic_optimizer_choices,
    optimizer_guidance,
)
from musubi_tuner_gui.qwen_image_lora_gui import QwenImageOptimizerSettings


def dropdown_values(component):
    return [value for _, value in component.choices]


def test_add_automagic_choices_preserves_defaults_and_avoids_duplicates():
    choices = add_automagic_optimizer_choices(["AdamW", "automagic"])

    assert choices[:2] == ["AdamW", "automagic"]
    assert sum(choice.casefold() == "automagic" for choice in choices) == 1
    assert {choice.casefold() for choice in choices} >= {choice.casefold() for choice in AUTOMAGIC_OPTIMIZER_CHOICES}


@pytest.mark.parametrize("optimizer_type", AUTOMAGIC_OPTIMIZER_CHOICES)
def test_automagic_guidance_explains_adaptive_lr_and_scheduler(optimizer_type):
    guidance = optimizer_guidance(optimizer_type)

    expected_version = "v1" if optimizer_type == "Automagic" else optimizer_type.replace("Automagic", "v")
    assert expected_version in guidance
    assert "starting rate" in guidance
    assert "scheduler" in guidance.casefold()
    assert "Fused Backward Pass" in guidance
    assert "Adafactor-only preset arguments" in guidance
    assert "ignored automatically" in guidance
    assert "block swapping" in guidance.casefold()


def test_automagic2_guidance_explains_hard_compatibility_limits():
    guidance = optimizer_guidance("Automagic2")

    for expected in ("single-process", "Gradient Accumulation Steps = 1", "Max Gradient Norm = 0", "fp16"):
        assert expected in guidance


def test_automagic3_guidance_explains_automatic_safe_mode():
    guidance = optimizer_guidance("Automagic3")

    assert "automatically" in guidance
    assert "fused=False" in guidance
    assert "fused=True" in guidance


@pytest.mark.parametrize(
    ("component_factory", "selected"),
    [
        (lambda: OptimizerAndScheduler(config={"optimizer_type": "Automagic3"}), "Automagic3"),
        (lambda: QwenImageOptimizerSettings(False, {"optimizer_type": "Automagic2"}), "Automagic2"),
    ],
)
def test_all_optimizer_component_variants_expose_choices_and_initial_guidance(component_factory, selected):
    with gr.Blocks():
        component = component_factory()

    values = dropdown_values(component.optimizer_type)
    assert set(AUTOMAGIC_OPTIMIZER_CHOICES) <= set(values)
    expected_version = "v1" if selected == "Automagic" else selected.replace("Automagic", "v")
    assert expected_version in component.optimizer_guidance.value


def test_custom_optimizer_guidance_is_explicit():
    guidance = optimizer_guidance("example.CustomOptimizer")

    assert "Custom optimizer" in guidance
    assert "example.CustomOptimizer" in guidance


@pytest.mark.parametrize("setting_name", ["lr_warmup_steps", "lr_decay_steps"])
@pytest.mark.parametrize("value", [10, 0.1])
def test_lr_step_inputs_accept_absolute_counts_and_ratios(setting_name, value):
    with gr.Blocks():
        settings = OptimizerAndScheduler(config={setting_name: value})

    component = getattr(settings, setting_name)

    assert component.maximum is None
    assert component.preprocess(component.value) == value


def test_gui_config_loads_utf8_bom_without_prefixing_first_key(tmp_path):
    config_path = tmp_path / "preset.toml"
    config_path.write_bytes(b"\xef\xbb\xbfadditional_parameters = \"test=true\"\n")

    config = GUIConfig(str(config_path))

    assert config.get("additional_parameters") == "test=true"
