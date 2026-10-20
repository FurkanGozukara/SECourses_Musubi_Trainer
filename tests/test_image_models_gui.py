"""Tests for the spec-driven new-model tabs (Qwen-Image 2.1, ERNIE-Image, Mage-Flow, HiDream-O1, Kandinsky 6)."""

from __future__ import annotations

import os

import gradio as gr
import pytest
import toml

from musubi_tuner_gui import image_models_gui as m
from musubi_tuner_gui import preset_library
from musubi_tuner_gui.full_finetune_gui import FULL_FINE_TUNING_MODE, LORA_TRAINING_MODE


def _touch(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _parameters(spec_key, tmp_path, **overrides):
    spec = m.get_spec(spec_key)
    # a switch to Full Fine-Tuning loads the fine-tuning defaults (their optimizer is AdaFactor, as fused backward needs)
    defaults = m.architecture_defaults(spec, overrides.get("training_mode") == FULL_FINE_TUNING_MODE)
    values = {}
    for item in spec.model_paths:
        target = tmp_path / "models" / item.filename
        if item.is_dir:
            target.mkdir(parents=True, exist_ok=True)
            values[item.key] = str(target)
        else:
            values[item.key] = _touch(target)
    dataset = tmp_path / "dataset.toml"
    dataset.write_text('[general]\nresolution = [512, 512]\n[[datasets]]\nimage_directory = "x"\n', encoding="utf-8")
    values.update(
        {
            "dataset_config_mode": "Use TOML File",
            "dataset_config": str(dataset),
            "output_dir": str(tmp_path / "out"),
            "output_name": "unit",
            "sample_prompts": "",
        }
    )
    values.update(overrides)
    keys = m.param_keys(spec)
    return [(key, values.get(key, defaults.get(key))) for key in keys]


@pytest.mark.parametrize("spec_key", sorted(m.ARCHITECTURES))
def test_tab_builds_and_registers_every_key(spec_key):
    with gr.Blocks():
        settings = m.image_model_tab(spec_key, headless=True, config={})
    assert len(settings) == len(m.param_keys(m.get_spec(spec_key)))


@pytest.mark.parametrize("spec_key", sorted(m.ARCHITECTURES))
def test_lora_runtime_toml_is_clean(spec_key, tmp_path):
    spec = m.get_spec(spec_key)
    config_path = tmp_path / "run.toml"
    workflow = m.prepare_workflow(spec_key, _parameters(spec_key, tmp_path), str(config_path), python_cmd="python")
    data = toml.load(config_path)
    assert data["network_module"] == spec.network_module
    assert data["training_mode"] == LORA_TRAINING_MODE
    # Guardrail 1: logging disabled -> no logging keys at all
    assert "logging_dir" not in data and "log_with" not in data
    # GUI-only and caching keys never reach the backend
    for key in ("parent_folder_path", "sample_width", "caching_teo_batch_size", "cache_latents", "dit_dtype", "gpu_ids"):
        assert key not in data, key
    for key in spec.gui_only_keys:
        assert key not in data, key
    assert workflow.train_command[-2:] == ["--config_file", str(config_path)]
    assert any(spec.train_script in part for part in workflow.train_command)


@pytest.mark.parametrize("spec_key", sorted(m.ARCHITECTURES))
def test_debug_logging_resolves_logging_dir_under_output(spec_key, tmp_path):
    config_path = tmp_path / "run.toml"
    params = _parameters(spec_key, tmp_path, debug_mode="Enable Logging (TensorBoard)")
    workflow = m.prepare_workflow(spec_key, params, str(config_path), python_cmd="python")
    data = toml.load(config_path)
    assert data["log_with"] == "tensorboard"
    expected = os.path.normcase(os.path.abspath(str(tmp_path / "out" / "logs")))
    assert os.path.normcase(os.path.abspath(data["logging_dir"])) == expected
    assert "--log_with" not in " ".join(workflow.train_command)


@pytest.mark.parametrize("spec_key", [key for key, spec in m.ARCHITECTURES.items() if spec.supports_full_finetune])
def test_full_finetune_round_trip(spec_key, tmp_path):
    spec = m.get_spec(spec_key)
    config_path = tmp_path / "run.toml"
    params = _parameters(spec_key, tmp_path, training_mode=FULL_FINE_TUNING_MODE, full_bf16=True, fused_backward_pass=True)
    workflow = m.prepare_workflow(spec_key, params, str(config_path), python_cmd="python")
    data = toml.load(config_path)
    assert data["training_mode"] == FULL_FINE_TUNING_MODE
    assert "network_module" not in data and "network_dim" not in data
    assert "fp8_base" not in data and "fp8_scaled" not in data
    assert any(spec.full_train_script in part for part in workflow.train_command)

    shown = _parameters(spec_key, tmp_path, training_mode=LORA_TRAINING_MODE)
    loaded = m.open_configuration(spec_key, False, str(config_path), shown)
    mode_index = m.param_keys(spec).index("training_mode") + 2
    assert loaded[mode_index] == FULL_FINE_TUNING_MODE

    legacy = dict(data)
    legacy.pop("training_mode")
    legacy_path = tmp_path / "legacy.toml"
    legacy_path.write_text(toml.dumps(legacy), encoding="utf-8")
    assert m.open_configuration(spec_key, False, str(legacy_path), shown)[mode_index] == FULL_FINE_TUNING_MODE

    lora_path = tmp_path / "lora.toml"
    lora_path.write_text(toml.dumps({"network_module": spec.network_module}), encoding="utf-8")
    shown_full = _parameters(spec_key, tmp_path, training_mode=FULL_FINE_TUNING_MODE)
    assert m.open_configuration(spec_key, False, str(lora_path), shown_full)[mode_index] == LORA_TRAINING_MODE


@pytest.mark.parametrize("spec_key", [key for key, spec in m.ARCHITECTURES.items() if not spec.supports_full_finetune])
def test_lora_only_models_reject_full_finetune(spec_key, tmp_path):
    with pytest.raises(ValueError, match="LoRA training only"):
        m.prepare_workflow(spec_key, _parameters(spec_key, tmp_path, training_mode=FULL_FINE_TUNING_MODE), str(tmp_path / "r.toml"))


def test_cache_commands_carry_model_specific_flags(tmp_path):
    def commands(spec_key, **overrides):
        spec = m.get_spec(spec_key)
        param_dict, _ = m.normalize_parameters(spec, _parameters(spec_key, tmp_path, **overrides))
        return m.build_cache_commands(spec, param_dict, python_cmd="python")

    latent, text = commands("qwen_image_21", vae_tiling=True, caching_teo_fp8_vl=True)
    assert "--vae_tiling" in latent and "--fp8_vl" in text and "--text_encoder_dtype" in text

    latent, text = commands("mage_flow", is_edit=True)
    assert "--is_edit" in latent and "--is_edit" in text

    latent, text = commands("hidream_o1", model_type="dev")
    assert "--vae" not in latent
    assert text[text.index("--model_type") + 1] == "dev"

    latent, text = commands("kandinsky6")
    assert "--audio_vae" in latent and "--text_encoder_qwen" in text and "--text_encoder_clip" in text
    assert "--vae_tiling_memory_gb" not in latent
    latent, _text = commands("kandinsky6", caching_latent_vae_memory_gb=3)
    assert latent[latent.index("--vae_tiling_memory_gb") + 1] == "3"

    latent, text = commands("ernie_image")
    assert "--vae" in latent and "--text_encoder" in text


def test_model_specific_validation(tmp_path):
    with pytest.raises(ValueError, match="fullgraph"):
        m.normalize_parameters(m.MAGE_FLOW, _parameters("mage_flow", tmp_path, compile=True, compile_fullgraph=True))
    with pytest.raises(ValueError, match="4\\*n\\+1"):
        m.normalize_parameters(m.KANDINSKY_6, _parameters("kandinsky6", tmp_path, num_frames=120))
    with pytest.raises(ValueError, match="at most 30"):
        m.normalize_parameters(m.KANDINSKY_6, _parameters("kandinsky6", tmp_path, model_variant="lite", blocks_to_swap=40))
    with pytest.raises(ValueError, match="mutually exclusive"):
        m.normalize_parameters(m.KANDINSKY_6, _parameters("kandinsky6", tmp_path, convrot_int8=True, fp8_base=True))
    with pytest.raises(ValueError, match="i2i"):
        m.normalize_parameters(m.HIDREAM_O1, _parameters("hidream_o1", tmp_path, task="i2i", use_control_images=False))
    with pytest.raises(ValueError, match="between 0 and"):
        m.normalize_parameters(m.MAGE_FLOW, _parameters("mage_flow", tmp_path, blocks_to_swap=11))


def test_sample_prompt_defaults_follow_spec_flags(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a cat\n# comment\na dog --w 512\n", encoding="utf-8")
    spec = m.KANDINSKY_6
    params = _parameters("kandinsky6", tmp_path, sample_prompts=str(prompts), sample_num_frames=120, sample_cfg_scale=5.0)
    param_dict, params = m.enhance_sample_prompts(spec, dict(params), params)
    lines = [line.strip() for line in open(param_dict["sample_prompts"], encoding="utf-8") if line.strip() and not line.startswith("#")]
    assert "--f 117" in lines[0] and "--g 5.0" in lines[0] and "--l" not in lines[0]
    assert lines[1].count("--w") == 1


def test_shipped_presets_are_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_PRESETS_DIR", str(tmp_path))
    preset = tmp_path / "Qwen_Image_2.1_Training_Configs" / "LoRA_Training" / "8_GB_GPUs_6.4_GB_rank_32.toml"
    preset.parent.mkdir(parents=True)
    preset.write_text("# description: test preset\nnetwork_dim = 32\n", encoding="utf-8")
    assert preset_library.is_shipped_preset(str(preset))
    if os.name == "nt":  # Windows paths are case-insensitive: a lower-cased path is the same shipped file
        assert preset_library.is_shipped_preset(str(preset).lower())
    entries = preset_library.list_presets(["Qwen_Image_2.1_Training_Configs"])
    assert entries and entries[0].tier_gb == 8 and entries[0].used_gb == 6.4 and entries[0].mode == "LoRA"
    assert entries[0].description == "test preset"
    params = _parameters("qwen_image_21", tmp_path)
    path, status = m.save_configuration("qwen_image_21", False, str(preset), params)
    assert path == str(preset)
    assert "read-only" in status["value"]
    assert preset.read_text(encoding="utf-8").startswith("# description: test preset")


def test_tabs_without_defaults_file_ignore_stray_config_toml(tmp_path, monkeypatch):
    """GUIConfig() is empty: a legacy ./config.toml in the working directory must not leak into tabs."""
    from musubi_tuner_gui.class_gui_config import GUIConfig
    from musubi_tuner_gui.class_tab_config_manager import TabConfigManager

    (tmp_path / "config.toml").write_text('dit = "C:/somewhere/else.safetensors"\ndiscrete_flow_shift = 7\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert GUIConfig().config == {}
    manager = TabConfigManager(config_file_path=str(tmp_path / "missing_defaults.toml"))
    for tab in ("qwen_image_21", "ernie_image", "mage_flow", "flux3_klein", "moss_image", "hidream_o1", "kandinsky6", "image_preprocessing"):
        assert manager.get_config_for_tab(tab).config == {}
    prepared = m._prepare_config(m.QWEN_IMAGE_21, manager.get_config_for_tab("qwen_image_21"))
    assert prepared.get("discrete_flow_shift") == m.architecture_defaults(m.QWEN_IMAGE_21)["discrete_flow_shift"]
    assert prepared.get("dit") != "C:/somewhere/else.safetensors"

def test_preset_library_families_tiers_and_recommendation(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_PRESETS_DIR", str(tmp_path))

    def write(rel):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# description: x\n", encoding="utf-8")

    write("Wan22_Training_Configs/Wan22_Text_To_Video_Configs/32_GB_GPUs_28.2_GB_Quality_1.toml")
    write("Wan22_Training_Configs/Wan22_Text_To_Video_Configs/32_GB_GPUs_24_GB_Quality_1_FP8_Scaled_Low_RAM_Fast.toml")
    write("Wan22_Training_Configs/Wan22_Image_To_Video_Configs/32_GB_GPUs_28.2_GB_Quality_1.toml")
    write("Wan_21_Training_Configs/LoRA_Training/24_GB_GPUs_21.9_GB_Quality_1.toml")
    write("Wan_21_Training_Configs/LoRA_Training/8_GB_GPUs_6.3_GB_Quality_3.toml")
    entries = preset_library.list_presets(["Wan22_Training_Configs", "Wan_21_Training_Configs"])
    families = sorted({entry.family for entry in entries})
    assert families == ["Wan 2.1", "Wan 2.2 · Image To Video", "Wan 2.2 · Text To Video"]
    t2v = preset_library.filter_presets(entries, family="Wan 2.2 · Text To Video", mode="LoRA", tier="32")
    best = preset_library.best_preset(t2v)
    assert best is not None and "Low_RAM" not in best.path  # Quality 1 regular beats the Low-RAM variant
    wan21 = preset_library.filter_presets(entries, family="Wan 2.1")
    assert preset_library.best_tier_for(wan21, 31.8) == "24"
    assert preset_library.best_tier_for(wan21, 12.0) == "8"
    assert preset_library.best_tier_for(wan21, 6.0) == "8"  # nothing fits: smallest tier is offered
    assert preset_library.best_tier_for(wan21, None) == preset_library.ALL_TIERS


def test_preset_library_legacy_names_and_example_datasets(tmp_path, monkeypatch):
    """Qwen_Training_Configs: TierN_<MB>_MB names (Tier1 = best quality) and example datasets beside them."""
    monkeypatch.setenv("MUSUBI_PRESETS_DIR", str(tmp_path))
    for rel in (
        "Qwen_Training_Configs/LoRA_Training/100 epoch/Tier1_15150_MB_VRAM_More_RAM_and_Slow_On_Windows.toml",
        "Qwen_Training_Configs/LoRA_Training/100 epoch/Tier2_14750_MB.toml",
        "Qwen_Training_Configs/LoRA_Training/100 epoch/Tier5_6550_MB_VRAM.toml",
        "Qwen_Training_Configs/LoRA_Training/100 epoch/Tier6_5075_MB_VRAM.toml",
        "Qwen_Training_Configs/Qwen_Image_Training_Example_Dataset/dataset_config_20260712_064132.toml",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("network_dim = 32\n", encoding="utf-8")
    entries = preset_library.list_presets(["Qwen_Training_Configs"])
    assert len(entries) == 4 and not any("dataset_config" in entry.path for entry in entries)
    assert {entry.quality for entry in entries} == {1, 2, 5, 6}
    assert preset_library.best_preset(entries) is None  # no GPU tiers in these names
    assert "Tier1_15150" in preset_library.default_preset(entries, 15.99).path
    assert "Tier5_6550" in preset_library.default_preset(entries, 7.996).path
    assert "Tier6_5075" in preset_library.default_preset(entries, 6.0).path
    assert preset_library.default_preset(entries, 5.0) is None
    assert preset_library.gpu_label("NVIDIA GeForce RTX 5090") == "RTX 5090"
    # a lone unmeasured preset (the Ideogram 4 BF16 fine-tuning one) is still pre-selected
    lone_path = tmp_path / "Ideogram_4_Training_Configs/Fine_Tuning_Training/BF16_Checkpoint_Required_Ideogram_4_Fine_Tuning.toml"
    lone_path.parent.mkdir(parents=True, exist_ok=True)
    lone_path.write_text('training_mode = "Full Fine-Tuning"\n', encoding="utf-8")
    lone = preset_library.list_presets(["Ideogram_4_Training_Configs"])
    assert len(lone) == 1 and lone[0].used_gb is None and lone[0].tier_gb is None
    assert preset_library.default_preset(lone, 24.0) is lone[0]
    assert preset_library.default_preset(lone, None) is lone[0]

def test_training_monitor_tick(monkeypatch):
    import time as _time

    from musubi_tuner_gui import class_command_executor as ce

    with gr.Blocks():
        executor = ce.CommandExecutor(headless=True)

    class FakeProcess:
        pid = 2**30
        returncode = None

        def poll(self):
            return self.returncode

    process = FakeProcess()
    executor.process = process
    executor.started_at = _time.time() - 65
    monkeypatch.setattr(ce, "_gpu_lines", lambda: [(0, "RTX Test", 12.5, 32.0, 97)])
    monkeypatch.setattr(executor, "_phase", lambda: "Training")
    markdown, timer = executor._monitor_tick()
    assert "Training" in markdown.value and "12.5 / 32.0 GB" in markdown.value and "0:01:05" in markdown.value
    assert timer.active is True
    process.returncode = 0
    markdown, timer = executor._monitor_tick()
    assert "Finished" in markdown.value and timer.active is False
    process.returncode = 3
    markdown, _ = executor._monitor_tick()
    assert "exit code 3" in markdown.value

def test_reference_images_enabled_without_control_folder_is_rejected(tmp_path):
    """Mage-Flow-Edit / Qwen-Image 2.1 editing: a missing control sub-folder must fail before caching starts."""
    target = tmp_path / "dataset" / "1_ohwx"
    _touch(target / "image_0.png")
    (target / "image_0.txt").write_text("ohwx", encoding="utf-8")
    params = dict(_parameters("mage_flow", tmp_path, use_control_images=True, control_directory_name="control_images"))
    params.update({"parent_folder_path": str(tmp_path / "dataset"), "dataset_config_mode": "Generate from Folder Structure"})
    with pytest.raises(ValueError, match="no 'control_images' sub-folder"):
        m.create_dataset_config(m.MAGE_FLOW, params)
    _touch(target / "control_images" / "image_0.png")
    path, _messages = m.create_dataset_config(m.MAGE_FLOW, params)
    assert toml.load(path)["datasets"][0]["control_directory"].endswith("control_images")


def test_flux3_references_keep_their_framing(tmp_path):
    """FLUX.3 editing: references keep their own aspect ratio at the training pixel count (control_resolution) instead
    of being cropped to the target's bucket; batches of differently shaped references cannot be stacked."""
    target = tmp_path / "dataset" / "1_ohwx"
    _touch(target / "image_0.png")
    (target / "image_0.txt").write_text("ohwx", encoding="utf-8")
    _touch(target / "control_images" / "image_0.png")
    params = dict(_parameters("flux3_klein", tmp_path, use_control_images=True, control_directory_name="control_images"))
    params.update({"parent_folder_path": str(tmp_path / "dataset"), "dataset_config_mode": "Generate from Folder Structure",
                   "dataset_resolution_width": 1024, "dataset_resolution_height": 768, "dataset_batch_size": 1})
    path, _messages = m.create_dataset_config(m.FLUX_3_KLEIN, params)
    assert toml.load(path)["datasets"][0]["control_resolution"] == [1024, 768]
    params["dataset_batch_size"] = 2
    with pytest.raises(ValueError, match="batch size to 1"):
        m.create_dataset_config(m.FLUX_3_KLEIN, params)
    params.update(use_control_images=False)
    path, _messages = m.create_dataset_config(m.FLUX_3_KLEIN, params)  # text-to-image keeps any batch size
    assert "control_resolution" not in toml.load(path)["datasets"][0]
    mage = dict(_parameters("mage_flow", tmp_path, use_control_images=True, control_directory_name="control_images"))
    mage.update({"parent_folder_path": str(tmp_path / "dataset"), "dataset_config_mode": "Generate from Folder Structure"})
    path, _messages = m.create_dataset_config(m.MAGE_FLOW, mage)
    assert "control_resolution" not in toml.load(path)["datasets"][0]  # other editing models keep the target's bucket


def test_paths_after_download_keep_the_preset_file(tmp_path, monkeypatch):
    """A Mage-Flow-Edit preset applied before downloading must get the Edit DiT, not the T2I default."""
    models = tmp_path / "Training_Models_Mage_Flow"
    for name in ("mage_flow_base_bf16.safetensors", "mage_flow_edit_base_bf16.safetensors", "mage_flow_vae_bf16.safetensors"):
        _touch(models / name)
    monkeypatch.setenv("MUSUBI_TRAINING_MODELS_DIR", str(models))
    defaults = {"dit": str(models / "mage_flow_base_bf16.safetensors"), "vae": str(models / "mage_flow_vae_bf16.safetensors"), "text_encoder": ""}
    monkeypatch.setattr(m, "_default_model_paths", lambda spec: defaults)
    keys = [item.key for item in m.MAGE_FLOW.model_paths]
    current = {
        "dit": "G:/elsewhere/Training_Models_Mage_Flow/mage_flow_edit_base_bf16.safetensors",  # shipped preset path
        "vae": "",
        "text_encoder": "",
    }
    resolved = m.paths_after_download(m.MAGE_FLOW, [current.get(key, "") for key in keys])
    assert os.path.basename(resolved["dit"]) == "mage_flow_edit_base_bf16.safetensors"
    assert resolved["vae"] == defaults["vae"]
    assert resolved["text_encoder"] == ""


def test_preset_picker_follows_the_tab_training_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_PRESETS_DIR", str(tmp_path))
    for rel in (
        "HiDream_O1_Training_Configs/LoRA_Training/24_GB_GPUs_19.2_GB_Quality_1.toml",
        "HiDream_O1_Training_Configs/Fine_Tuning_Training/24_GB_GPUs_21.5_GB_Quality_1.toml",
        "ERNIE_Image_Training_Configs/LoRA_Training/24_GB_GPUs_20.8_GB_Quality_1.toml",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# description: x\n", encoding="utf-8")
    monkeypatch.setattr(preset_library, "detect_gpus", lambda: (("NVIDIA GeForce RTX 4090", 23.99),))
    with gr.Blocks():
        hidream = preset_library.PresetLibrary(("HiDream_O1_Training_Configs",), model_name="HiDream-O1")
        ernie = preset_library.PresetLibrary(("ERNIE_Image_Training_Configs",), model_name="ERNIE-Image")
    assert hidream.mode_filter.value == "LoRA" and hidream.preset.value.endswith("19.2_GB_Quality_1.toml")
    mode, tier, preset, _details = hidream.follow_mode(FULL_FINE_TUNING_MODE, hidream.family.value, "24")
    assert mode.value == "Fine-Tuning" and tier.value == "24"
    assert preset.value.endswith(os.path.join("Fine_Tuning_Training", "24_GB_GPUs_21.5_GB_Quality_1.toml"))
    mode, *_rest = hidream.follow_mode(LORA_TRAINING_MODE, hidream.family.value, "24")
    assert mode.value == "LoRA"
    # no fine-tuning presets: the picker is left alone
    assert all(update == gr.update() for update in ernie.follow_mode("DreamBooth Fine-Tuning", ernie.family.value, "24"))


def test_hidream_freeze_text_embeddings_is_full_finetune_only(tmp_path):
    full = _parameters("hidream_o1", tmp_path, training_mode=FULL_FINE_TUNING_MODE, full_bf16=True, fused_backward_pass=True, freeze_text_embeddings=True)
    m.prepare_workflow("hidream_o1", full, str(tmp_path / "full.toml"), python_cmd="python")
    assert toml.load(tmp_path / "full.toml")["freeze_text_embeddings"] is True
    lora = _parameters("hidream_o1", tmp_path, freeze_text_embeddings=True)
    m.prepare_workflow("hidream_o1", lora, str(tmp_path / "lora.toml"), python_cmd="python")
    assert "freeze_text_embeddings" not in toml.load(tmp_path / "lora.toml")


def test_training_mode_switch_applies_the_model_recipe():
    spec = m.HIDREAM_O1
    optimizer, args, lr, grad_norm, full_bf16, fused, mem_eff, h2d = m.training_mode_switch_updates(spec, FULL_FINE_TUNING_MODE, "AdamW8bit", 4e-5)
    assert optimizer["value"] == "AdaFactor" and "relative_step=False" in args["value"]
    assert lr["value"] == spec.full_defaults["learning_rate"] and grad_norm["value"] == 0.0
    assert full_bf16["value"] is True and fused["value"] is True and mem_eff["value"] is True and h2d["value"] is False
    # an LR the user already lowered for fine-tuning is kept
    assert m.training_mode_switch_updates(spec, FULL_FINE_TUNING_MODE, "AdaFactor", 2e-6)[2] == gr.update()
    # back to LoRA: a fine-tuning LR is raised to the LoRA recipe, fused backward is switched off
    updates = m.training_mode_switch_updates(spec, LORA_TRAINING_MODE, "AdaFactor", 5e-6)
    assert updates[2]["value"] == spec.lora_defaults["learning_rate"] and updates[5]["value"] is False


def test_kandinsky6_fine_tuning_runs_kandinsky6_train_with_the_ring_only_under_block_swap(tmp_path):
    """Kandinsky 6 Fine-Tuning runs kandinsky6_train.py (BF16 DiT, fused AdaFactor); the trainable block-swap ring is
    written to the run's settings only with block swap, and with pinned memory (its CPU copies are pinned). A saved
    preset keeps the choice; ConvRot INT8 stays a LoRA option."""
    spec = m.get_spec("kandinsky6")
    full = dict(training_mode=FULL_FINE_TUNING_MODE, block_swap_trainable_ring=True, convrot_int8=True)
    config_path = tmp_path / "run.toml"
    workflow = m.prepare_workflow("kandinsky6", _parameters("kandinsky6", tmp_path, blocks_to_swap=0, **full), str(config_path), python_cmd="python")
    data = toml.load(config_path)
    assert any(part.endswith("kandinsky6_train.py") for part in workflow.train_command)
    assert data["optimizer_type"] == "AdaFactor" and data["fused_backward_pass"] and data["full_bf16"]
    assert data["block_swap_trainable_ring"] is False and "convrot_int8" not in data
    m.prepare_workflow("kandinsky6", _parameters("kandinsky6", tmp_path, blocks_to_swap=8, use_pinned_memory_for_block_swap=False, **full), str(config_path), python_cmd="python")
    data = toml.load(config_path)
    assert data["block_swap_trainable_ring"] is True and data["use_pinned_memory_for_block_swap"] is True
    _, saved = m.normalize_parameters(spec, _parameters("kandinsky6", tmp_path, blocks_to_swap=0, **full), validate_paths=False)
    assert dict(saved)["block_swap_trainable_ring"] is True


def test_kandinsky6_r2av_needs_a_reference_dataset_and_trained_slots_for_lora(tmp_path):
    """Kandinsky 6 r2av: caching encodes the references (letterbox grey, max references), the dataset is a JSONL
    (Use TOML File), and LoRA refuses a DiT without trained reference slots (a released checkpoint) before caching."""
    import torch
    from safetensors.torch import save_file

    full = dict(training_mode=FULL_FINE_TUNING_MODE, task="r2av", ref_pad_value=128, caching_latent_max_references=4)
    config_path = tmp_path / "run.toml"
    workflow = m.prepare_workflow("kandinsky6", _parameters("kandinsky6", tmp_path, **full), str(config_path), python_cmd="python")
    data = toml.load(config_path)
    assert data["task"] == "r2av" and data["ref_pad_value"] == 128 and data["ref_drop_prob"] == 0.1
    latent = workflow.latent_cache_command
    assert latent[latent.index("--ref_pad_value") + 1] == "128" and latent[latent.index("--max_references") + 1] == "4"
    with pytest.raises(ValueError, match="JSONL"):
        m.prepare_workflow("kandinsky6", _parameters("kandinsky6", tmp_path, dataset_config_mode="Generate from Folder Structure", **full), str(config_path), python_cmd="python")

    released = tmp_path / "released.safetensors"
    save_file({"visual_token_type_embeddings.weight": torch.zeros(2, 4)}, str(released))
    lora = dict(training_mode=LORA_TRAINING_MODE, task="r2av")
    with pytest.raises(ValueError, match="reference slots"):
        m.prepare_workflow("kandinsky6", _parameters("kandinsky6", tmp_path, dit=str(released), **lora), str(config_path), python_cmd="python")
    finetuned = tmp_path / "r2av_finetune.safetensors"
    save_file({"visual_ref_slot_embeddings.weight": torch.zeros(8, 4)}, str(finetuned))
    m.prepare_workflow("kandinsky6", _parameters("kandinsky6", tmp_path, dit=str(finetuned), **lora), str(config_path), python_cmd="python")
