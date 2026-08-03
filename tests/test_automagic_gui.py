import gradio as gr
import pytest

from musubi_tuner_gui.class_optimizer_and_scheduler import OptimizerAndScheduler
from musubi_tuner_gui.class_gui_config import GUIConfig
from musubi_tuner_gui import (
    flux_lora_gui,
    ltx2_lora_gui,
    modern_image_lora_gui,
    qwen_image_lora_gui,
    wan_lora_gui,
    zimage_lora_gui,
)
from musubi_tuner_gui.optimizer_catalog import (
    AUTOMAGIC_OPTIMIZER_CHOICES,
    add_automagic_optimizer_choices,
    optimizer_guidance,
    validate_automagic_configuration,
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
    assert "choose Automagic3 instead" in guidance


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


def test_automagic2_preflight_accepts_compatible_internal_fused_mode():
    warnings = validate_automagic_configuration(
        {
            "optimizer_type": "Automagic2",
            "optimizer_args": [],
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 0,
            "mixed_precision": "bf16",
            "num_processes": 1,
            "num_machines": 1,
            "multi_gpu": False,
            "block_swap_optimizer_patch_params": False,
        }
    )

    assert warnings == ()


def test_automagic2_preflight_lists_every_fix_and_recommends_safe_alternative():
    with pytest.raises(ValueError) as exc_info:
        validate_automagic_configuration(
            {
                "optimizer_type": "Automagic2",
                "optimizer_args": "fused=False",
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1,
                "mixed_precision": "fp16",
                "num_processes": 2,
                "multi_gpu": True,
                "block_swap_optimizer_patch_params": True,
            }
        )

    message = str(exc_info.value)
    for expected in (
        "remove the `fused` optimizer argument",
        "Gradient Accumulation Steps is 4",
        "Max Gradient Norm is 1",
        "Mixed Precision is fp16",
        "multi-GPU",
        "Patch Optimizer for Block Swap",
        "choose Automagic3",
    ):
        assert expected in message


def test_separate_fused_backward_checkbox_explains_automagic2_internal_fusion():
    with pytest.raises(ValueError, match="already performs its own fused updates"):
        validate_automagic_configuration(
            {
                "optimizer_type": "Automagic2",
                "training_mode": "Full Fine-Tuning",
                "fused_backward_pass": True,
            }
        )


def test_automagic3_preflight_announces_automatic_non_fused_mode():
    displayed = []

    warnings = validate_automagic_configuration(
        {
            "optimizer_type": "Automagic3",
            "gradient_accumulation_steps": 2,
            "max_grad_norm": 1,
            "mixed_precision": "bf16",
        },
        warning_callback=displayed.append,
    )

    assert warnings == tuple(displayed)
    assert "safe non-fused mode" in warnings[0]
    assert "Gradient Accumulation Steps is 2" in warnings[0]
    assert "Max Gradient Norm is 1" in warnings[0]


@pytest.mark.parametrize("optimizer_type", ["Automagic", "Automagic3"])
def test_explicit_fused_automagic_rejects_incompatible_values(optimizer_type):
    with pytest.raises(ValueError, match="fused=True cannot start"):
        validate_automagic_configuration(
            {
                "optimizer_type": optimizer_type,
                "optimizer_args": "fused=True",
                "gradient_accumulation_steps": 2,
            }
        )


@pytest.mark.parametrize(
    ("module", "invoke"),
    [
        pytest.param(
            qwen_image_lora_gui,
            lambda parameters: qwen_image_lora_gui.train_qwen_image_model(True, True, parameters),
            id="qwen-image",
        ),
        pytest.param(
            wan_lora_gui,
            lambda parameters: wan_lora_gui.train_wan_model(True, True, parameters),
            id="wan",
        ),
        pytest.param(
            flux_lora_gui,
            lambda parameters: flux_lora_gui.train_flux_model(True, True, parameters),
            id="flux",
        ),
        pytest.param(
            zimage_lora_gui,
            lambda parameters: zimage_lora_gui.train_zimage_model(True, True, parameters),
            id="z-image",
        ),
        pytest.param(
            modern_image_lora_gui,
            lambda parameters: modern_image_lora_gui.train_modern_image_model("ideogram4", True, True, parameters),
            id="ideogram-4",
        ),
        pytest.param(
            modern_image_lora_gui,
            lambda parameters: modern_image_lora_gui.train_modern_image_model("krea2", True, True, parameters),
            id="krea-2",
        ),
        pytest.param(
            ltx2_lora_gui,
            lambda parameters: ltx2_lora_gui.train_ltx2_model(True, True, parameters),
            id="ltx-2.3",
        ),
    ],
)
def test_every_training_tab_runs_automagic_preflight_before_loading_files(monkeypatch, module, invoke):
    seen = []

    def stop_at_preflight(config, **_kwargs):
        seen.append(config["optimizer_type"])
        raise RuntimeError("preflight reached")

    monkeypatch.setattr(module, "validate_automagic_configuration", stop_at_preflight)
    with pytest.raises(RuntimeError, match="preflight reached"):
        invoke([("training_mode", "LoRA Training"), ("optimizer_type", "Automagic2")])

    assert seen == ["Automagic2"]


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
