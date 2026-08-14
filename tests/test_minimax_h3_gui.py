"""Tests for the MiniMax H3 trainer tab plumbing (no GPU / no real models needed).

Covers the AI_TRAINER_GUARDRAILS rules that apply to a LoRA-only trainer:
- Guardrail 1: no empty/rootish logging_dir in the runtime TOML; enabled logging
  resolves under output_dir/logs.
- Save/load round-trip through the canonical MINIMAX_H3_PARAM_KEYS list.
- Command construction: teacher matching flips the latent cache task to fl2va and
  adds --teacher_conditions to TE caching; the guidance loss auto-wires the uncond
  cache; disabled guidance never leaks scale keys into the run TOML.
"""

import os
import sys

import pytest
import toml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from musubi_tuner_gui.dataset_config_generator import (
    generate_minimax_h3_dataset_config_from_folders,
    round_frames_to_minimax_h3,
)
from musubi_tuner_gui.minimax_h3_lora_gui import (
    MINIMAX_H3_DEFAULTS,
    MINIMAX_H3_PARAM_KEYS,
    build_minimax_h3_cache_commands,
    open_minimax_h3_configuration,
    prepare_minimax_h3_workflow,
    save_minimax_h3_configuration,
)


def _base_parameters(tmp_path, **overrides):
    values = dict(MINIMAX_H3_DEFAULTS)
    dit = tmp_path / "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    video_vae = tmp_path / "minimax_h3_video_vae_fp16.safetensors"
    audio_vae = tmp_path / "minimax_h3_audio_vae_fp32.safetensors"
    text_encoder = tmp_path / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    dataset = tmp_path / "dataset.toml"
    for f in (dit, video_vae, audio_vae, text_encoder):
        f.write_bytes(b"stub")
    dataset.write_text("[general]\nresolution = [512, 512]\n[[datasets]]\nvideo_directory = 'x'\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir(exist_ok=True)
    values.update(
        {
            "dit": str(dit),
            "video_vae": str(video_vae),
            "audio_vae": str(audio_vae),
            "text_encoder": str(text_encoder),
            "dataset_config_mode": "Use TOML File",
            "dataset_config": str(dataset),
            "output_dir": str(output_dir),
            "output_name": "h3-test",
        }
    )
    values.update(overrides)
    return [(key, values[key]) for key in MINIMAX_H3_PARAM_KEYS]


def _run_toml(tmp_path, parameters):
    config_path = str(tmp_path / "out" / "h3-test-run.toml")
    workflow = prepare_minimax_h3_workflow(parameters, config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        return workflow, toml.load(handle)


def test_frame_rounding_follows_17n_plus_5():
    assert round_frames_to_minimax_h3(124) == 124
    assert round_frames_to_minimax_h3(125) == 124
    assert round_frames_to_minimax_h3(140) == 124
    assert round_frames_to_minimax_h3(141) == 141
    assert round_frames_to_minimax_h3(345) == 345
    assert round_frames_to_minimax_h3(39) == 39
    assert round_frames_to_minimax_h3(38) == 22
    assert round_frames_to_minimax_h3(1) == 5
    assert round_frames_to_minimax_h3(None) == 124


def test_runtime_toml_has_no_logging_keys_when_logging_disabled(tmp_path):
    workflow, data = _run_toml(tmp_path, _base_parameters(tmp_path))
    assert "logging_dir" not in data
    assert "log_with" not in data


def test_runtime_toml_resolves_tensorboard_logging_under_output_dir(tmp_path):
    parameters = _base_parameters(tmp_path, log_with="tensorboard", logging_dir="")
    workflow, data = _run_toml(tmp_path, parameters)
    assert data["log_with"] == "tensorboard"
    logging_dir = data["logging_dir"].replace("\\", "/")
    assert logging_dir.endswith("out/logs")


def test_runtime_toml_core_contents(tmp_path):
    workflow, data = _run_toml(tmp_path, _base_parameters(tmp_path))
    assert data["task"] == "t2va"
    assert data["network_module"] == "networks.lora_minimax_h3"
    assert data["network_dim"] == 128
    assert data["network_alpha"] == 128
    assert data["blocks_to_swap"] == 48
    assert data["block_swap_h2d_only"] is True
    assert data["text_encoder_blocks_to_swap"] == 50
    # guidance loss enabled by default -> scale + auto uncond cache present
    assert data["h3_guidance_loss_scale"] == 4.0
    assert data["h3_guidance_loss_uncond_cache"].endswith("h3-test_h3_uncond_space.safetensors")
    # audio guidance scale left at "same as video" -> omitted
    assert "h3_guidance_loss_scale_audio" not in data
    # GUI-only keys must never leak
    for forbidden in (
        "cache_latents",
        "cache_text_encoder_outputs",
        "caching_latent_device",
        "caching_teo_device",
        "dataset_resolution_width",
        "allow_experimental_duration",
        "parent_folder_path",
        "sample_steps",
        "sample_seed",
        "width",
        "height",
        "sample_num_frames",
    ):
        assert forbidden not in data, forbidden
    # store_true flags that are False are omitted
    for absent_flag in ("convrot_int8", "prune_adaln", "video_only", "h3_teacher_matching", "nvfp4_scaled_mm", "disable_mmap"):
        assert absent_flag not in data, absent_flag
    # timestep sampling family is left to the backend defaults (uniform enforced by H3)
    assert "timestep_sampling" not in data
    assert "fp8_base" not in data


def test_cache_commands_default_flow(tmp_path):
    param_dict = dict(_base_parameters(tmp_path))
    latent, text = build_minimax_h3_cache_commands(param_dict, python_cmd="python")
    latent_line = " ".join(latent)
    text_line = " ".join(text)
    assert "minimax_h3_cache_latents.py" in latent_line
    assert "--task t2va" in latent_line
    assert "--video_vae" in latent_line and "--audio_vae" in latent_line
    assert "--cache_seed 42" in latent_line
    assert "--skip_existing" in latent_line
    assert "--allow_experimental_duration" not in latent_line
    assert "minimax_h3_cache_text_encoder_outputs.py" in text_line
    assert "--text_cache_dtype bf16" in text_line
    assert "--text_encoder_blocks_to_swap 50" in text_line
    # guidance loss default-enabled -> uncond output is written during TE caching
    assert "--uncond_output" in text_line
    assert "--teacher_conditions" not in text_line


def test_teacher_matching_switches_latent_cache_task_and_te_conditions(tmp_path):
    parameters = _base_parameters(
        tmp_path,
        h3_teacher_matching=True,
        h3_guidance_loss_scale=0.0,
    )
    param_dict = dict(parameters)
    latent, text = build_minimax_h3_cache_commands(param_dict, python_cmd="python")
    latent_line = " ".join(latent)
    text_line = " ".join(text)
    assert "--task fl2va" in latent_line  # teacher needs the first/last condition latents
    assert "--task t2va" in text_line
    assert "--teacher_conditions first,last" in text_line
    assert "--uncond_output" not in text_line  # guidance disabled

    workflow, data = _run_toml(tmp_path, parameters)
    assert data["h3_teacher_matching"] is True
    assert data["task"] == "t2va"
    for forbidden in ("h3_guidance_loss_scale", "h3_guidance_loss_scale_audio", "h3_guidance_loss_uncond_cache"):
        assert forbidden not in data, forbidden


def test_teacher_matching_and_guidance_are_mutually_exclusive(tmp_path):
    parameters = _base_parameters(tmp_path, h3_teacher_matching=True, h3_guidance_loss_scale=4.0)
    with pytest.raises(Exception, match="mutually exclusive"):
        prepare_minimax_h3_workflow(parameters, str(tmp_path / "out" / "run.toml"))


def test_experimental_duration_flag_reaches_latent_cache_only(tmp_path):
    parameters = _base_parameters(tmp_path, allow_experimental_duration=True, dataset_num_frames=39)
    param_dict = dict(parameters)
    latent, _ = build_minimax_h3_cache_commands(param_dict, python_cmd="python")
    assert "--allow_experimental_duration" in " ".join(latent)
    workflow, data = _run_toml(tmp_path, parameters)
    assert "allow_experimental_duration" not in data  # trainer does not accept it


def test_short_frames_without_experimental_flag_is_rejected(tmp_path):
    parameters = _base_parameters(tmp_path, dataset_num_frames=39, allow_experimental_duration=False)
    with pytest.raises(Exception, match="Allow Experimental Duration"):
        prepare_minimax_h3_workflow(parameters, str(tmp_path / "out" / "run.toml"))


def test_fp8_flags_are_rejected(tmp_path):
    parameters = _base_parameters(tmp_path)
    parameters.append(("fp8_base", True))
    with pytest.raises(Exception, match="FP8"):
        prepare_minimax_h3_workflow(parameters, str(tmp_path / "out" / "run.toml"))


def test_save_load_round_trip(tmp_path):
    parameters = _base_parameters(
        tmp_path,
        network_dim=64,
        network_alpha=32,
        blocks_to_swap=20,
        h3_guidance_loss_scale=3.0,
        task="fl2va",
    )
    saved_path = str(tmp_path / "preset.toml")
    result_path, _status = save_minimax_h3_configuration(False, saved_path, parameters)
    assert result_path == saved_path
    assert os.path.isfile(saved_path)

    # Load into a GUI showing pure defaults; every saved value must come back.
    defaults = [(key, MINIMAX_H3_DEFAULTS[key]) for key in MINIMAX_H3_PARAM_KEYS]
    loaded = open_minimax_h3_configuration(False, saved_path, defaults)
    loaded_values = dict(zip(MINIMAX_H3_PARAM_KEYS, loaded[2:]))
    assert loaded_values["network_dim"] == 64
    assert loaded_values["network_alpha"] == 32
    assert loaded_values["blocks_to_swap"] == 20
    assert loaded_values["h3_guidance_loss_scale"] == 3.0
    assert loaded_values["task"] == "fl2va"
    assert loaded_values["network_module"] == "networks.lora_minimax_h3"


def test_dataset_generator_h3_rules(tmp_path):
    parent = tmp_path / "data"
    sub = parent / "3_mystyle"
    sub.mkdir(parents=True)
    (sub / "clip.mp4").write_bytes(b"stub")
    images_only = parent / "1_pictures"
    images_only.mkdir()
    (images_only / "img.png").write_bytes(b"stub")

    config, messages = generate_minimax_h3_dataset_config_from_folders(
        parent_folder=str(parent),
        resolution=(512, 512),
        num_frames=130,  # rounds down to 124
        allow_experimental_duration=False,
    )
    assert config["general"]["batch_size"] == 1
    assert config["general"]["resolution"] == [512, 512]
    assert len(config["datasets"]) == 1
    entry = config["datasets"][0]
    assert entry["target_frames"] == [124]
    assert entry["num_repeats"] == 3
    assert "source_fps" not in entry and "target_fps" not in entry
    joined = "\n".join(messages)
    assert "adjusted from 130 to 124" in joined
    assert "Only images found" in joined


def test_dataset_generator_rejects_non_32px_resolution(tmp_path):
    parent = tmp_path / "data"
    sub = parent / "1_x"
    sub.mkdir(parents=True)
    (sub / "clip.mp4").write_bytes(b"stub")
    with pytest.raises(ValueError, match="multiples of 32"):
        generate_minimax_h3_dataset_config_from_folders(
            parent_folder=str(parent),
            resolution=(500, 512),
        )
