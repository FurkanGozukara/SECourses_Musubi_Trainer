"""Tests for the LTX 2.3 video training tab (musubi_tuner_gui.ltx2_lora_gui)."""

import os
import sys

import pytest
import toml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.dataset_config_generator import round_frames_to_ltx2
from musubi_tuner_gui.ltx2_lora_gui import (
    LTX2_DEFAULTS,
    LTX2_PARAM_KEYS,
    LTX2_RUN_EXCLUSIONS,
    _normalize_ltx2_parameters,
    build_ltx2_cache_commands,
    prepare_ltx2_workflow,
)


def _params(**overrides):
    values = dict(LTX2_DEFAULTS)
    values.update(
        {
            "ltx2_checkpoint": "C:/models/ltx-2.3-22b-dev.safetensors",
            "gemma_safetensors": "C:/models/gemma_3_12B_it_fp8_e4m3fn.safetensors",
            "dataset_config": "C:/data/dataset.toml",
            "output_dir": "C:/outputs",
            "output_name": "test-lora",
            "int8_convrot_dynamic": True,
        }
    )
    values.update(overrides)
    return [(key, values[key]) for key in LTX2_PARAM_KEYS]


class TestParamListConsistency:
    def test_every_param_has_default(self):
        missing = [key for key in LTX2_PARAM_KEYS if key not in LTX2_DEFAULTS]
        assert missing == []

    def test_no_duplicate_keys(self):
        assert len(LTX2_PARAM_KEYS) == len(set(LTX2_PARAM_KEYS))

    def test_run_exclusions_are_known_keys(self):
        unknown = LTX2_RUN_EXCLUSIONS - set(LTX2_PARAM_KEYS)
        assert unknown == set()


class TestFrameRounding:
    @pytest.mark.parametrize(
        "raw,expected",
        [(1, 1), (8, 1), (9, 9), (16, 9), (17, 17), (33, 33), (50, 49), (121, 121), (0, 1), (-5, 1)],
    )
    def test_round_frames_to_ltx2(self, raw, expected):
        assert round_frames_to_ltx2(raw) == expected


class TestValidation:
    def test_int8_convrot_and_fp8_conflict(self):
        with pytest.raises(ValueError, match="one base quantization"):
            _normalize_ltx2_parameters(
                _params(int8_convrot_dynamic=True, fp8_base=True, fp8_scaled=True),
                validate_paths=False,
            )

    def test_int8_convrot_base_and_dynamic_conflict(self):
        with pytest.raises(ValueError, match="INT8 ConvRot"):
            _normalize_ltx2_parameters(
                _params(int8_convrot_dynamic=True, int8_convrot_base=True),
                validate_paths=False,
            )

    def test_fp8_scaled_requires_fp8_base(self):
        with pytest.raises(ValueError, match="fp8_scaled requires fp8_base"):
            _normalize_ltx2_parameters(
                _params(int8_convrot_dynamic=False, fp8_scaled=True),
                validate_paths=False,
            )

    def test_gemma_root_and_safetensors_conflict(self):
        with pytest.raises(ValueError, match="not both"):
            _normalize_ltx2_parameters(
                _params(gemma_root="C:/models/gemma-root"),
                validate_paths=False,
            )

    def test_gemma_quant_requires_hf_dir(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            _normalize_ltx2_parameters(
                _params(gemma_load_in_8bit=True),
                validate_paths=False,
            )

    def test_h2d_requires_block_swap(self):
        with pytest.raises(ValueError, match="block_swap_h2d_only"):
            _normalize_ltx2_parameters(
                _params(block_swap_h2d_only=True, blocks_to_swap=0),
                validate_paths=False,
            )

    def test_single_attention_backend_required(self):
        with pytest.raises(ValueError, match="only one attention"):
            _normalize_ltx2_parameters(
                _params(sdpa=True, flash_attn=True),
                validate_paths=False,
            )

    def test_mode_alias_normalization(self):
        param_dict, _ = _normalize_ltx2_parameters(_params(ltx2_mode="va"), validate_paths=False)
        assert param_dict["ltx2_mode"] == "av"

    def test_network_module_forced(self):
        param_dict, _ = _normalize_ltx2_parameters(_params(network_module="something_else"), validate_paths=False)
        assert param_dict["network_module"] == "networks.lora_ltx2"

    def test_defaults_are_valid(self):
        param_dict, _ = _normalize_ltx2_parameters(_params(), validate_paths=False)
        assert param_dict["ltx_version"] == "2.3"


class TestCacheCommands:
    def test_latent_cache_command_video(self):
        param_dict, _ = _normalize_ltx2_parameters(_params(), validate_paths=False)
        latent, text = build_ltx2_cache_commands(param_dict, python_cmd="python")
        assert "--ltx2_checkpoint" in latent
        assert "--ltx2_mode" in latent
        assert latent[latent.index("--ltx2_mode") + 1] == "video"
        assert "--skip_existing" in latent
        assert "--atomic_cache_writes" in latent
        assert "--gemma_safetensors" in text
        assert "--gemma_root" not in text

    def test_latent_cache_av_mode_includes_audio_source(self):
        param_dict, _ = _normalize_ltx2_parameters(_params(ltx2_mode="av"), validate_paths=False)
        latent, _ = build_ltx2_cache_commands(param_dict, python_cmd="python")
        assert "--ltx2_audio_source" in latent

    def test_gpu_id_routing(self):
        param_dict, _ = _normalize_ltx2_parameters(_params(gpu_ids="1"), validate_paths=False)
        latent, text = build_ltx2_cache_commands(param_dict, python_cmd="python")
        assert latent[latent.index("--device") + 1] == "cuda:1"
        assert text[text.index("--device") + 1] == "cuda:1"

    def test_caches_can_be_disabled(self):
        param_dict, _ = _normalize_ltx2_parameters(
            _params(cache_latents=False, cache_text_encoder_outputs=False, sample_every_n_epochs=0),
            validate_paths=False,
        )
        latent, text = build_ltx2_cache_commands(param_dict, python_cmd="python")
        assert latent is None and text is None


class TestWorkflow:
    def test_run_toml_contents(self, tmp_path):
        config_path = str(tmp_path / "run.toml")
        workflow = prepare_ltx2_workflow(_params(), config_path, validate_paths=False, python_cmd="python")
        data = toml.load(config_path)

        # Backend-required keys present with correct types
        assert data["ltx2_checkpoint"].endswith("ltx-2.3-22b-dev.safetensors")
        assert data["network_module"] == "networks.lora_ltx2"
        assert isinstance(data["network_dim"], int)
        assert data["ltx_version"] == "2.3"
        assert data["timestep_sampling"] == "shifted_logit_normal"
        assert data["int8_convrot_dynamic"] is True

        # GUI-only keys never reach the run TOML
        for key in ("parent_folder_path", "dataset_num_frames", "cache_latents", "caching_teo_batch_size",
                    "sample_steps", "gpu_ids", "additional_parameters", "dataset_config_mode"):
            assert key not in data, key

        # Optional empty strings dropped so argparse defaults apply
        assert "fp8_keep_blocks" not in data
        assert "quantize_device" not in data
        assert "w8a8_backend" not in data
        assert "gemma_root" not in data

        # False store_true flags are dropped
        assert "fp8_base" not in data
        assert "nf4_base" not in data

        # Train command shape
        assert workflow.train_command[-2:] == ["--config_file", config_path]
        assert any("ltx2_train_network.py" in str(part) for part in workflow.train_command)

    def test_quality_report_injection(self, tmp_path):
        config_path = str(tmp_path / "run.toml")
        prepare_ltx2_workflow(
            _params(int8_convrot_write_quality_report=True, int8_convrot_dynamic=True),
            config_path,
            validate_paths=False,
            python_cmd="python",
        )
        data = toml.load(config_path)
        assert data["int8_convrot_quality_report"].endswith("_int8_convrot_quality.json")

    def test_workflow_commands_order(self, tmp_path):
        config_path = str(tmp_path / "run.toml")
        workflow = prepare_ltx2_workflow(_params(), config_path, validate_paths=False, python_cmd="python")
        commands = workflow.commands
        assert len(commands) == 3  # latent cache, text cache, train
        assert "ltx2_cache_latents.py" in " ".join(map(str, commands[0]))
        assert "ltx2_cache_text_encoder_outputs.py" in " ".join(map(str, commands[1]))
        assert "ltx2_train_network.py" in " ".join(map(str, commands[2]))


class TestDemoPreset:
    PRESET = "G:/SECourses_Musubi_Trainer_v29/Demo_Training_Configs_FLUX-2_Z-Image_FLUX-Klein_WAN-21_Krea2_Ideogram4/LTX_2.3_LoRA_Demo.toml"

    @pytest.mark.skipif(not os.path.isfile(PRESET), reason="demo preset not present")
    def test_demo_preset_loads_and_validates(self):
        preset = toml.load(self.PRESET)
        params = [(key, preset.get(key, LTX2_DEFAULTS[key])) for key in LTX2_PARAM_KEYS]
        param_dict, _ = _normalize_ltx2_parameters(params, validate_paths=False)
        assert param_dict["network_dim"] == 128
        assert param_dict["network_alpha"] == 128
        assert param_dict["int8_convrot_dynamic"] is True
        assert param_dict["ltx_version"] == "2.3"
        assert param_dict["timestep_sampling"] == "shifted_logit_normal"
