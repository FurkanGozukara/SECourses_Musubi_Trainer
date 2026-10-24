"""LTX 2.5 in the LTX tab (musubi_tuner_gui.ltx2_lora_gui) and the unified UI: split-pack files, version-gated flags
and settings, and LTX 2.3 runs left exactly as they were."""

import os
import sys

import pytest
import toml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui import ltx2_lora_gui as ltx
from musubi_tuner_gui.ltx2_lora_gui import (
    LTX2_DEFAULTS,
    LTX2_PARAM_KEYS,
    LTX25_ONLY_KEYS,
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


def _ltx25(**overrides):
    values = {
        "ltx_version": "2.5",
        "ltx2_checkpoint": "C:/models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "gemma_safetensors": "",
        "ltx2_text_encoder_checkpoint": "C:/models/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "ltx2_video_vae": "C:/models/ltx-2.5-video-vae-bf16.safetensors",
        "ltx2_audio_vae": "C:/models/ltx-2.5-audio-vae-bf16.safetensors",
    }
    values.update(overrides)
    return _params(**values)


def _safetensors(path, *keys):
    import torch
    from safetensors.torch import save_file

    save_file({key: torch.zeros(1) for key in keys}, str(path))
    return str(path)


class TestKeys:
    def test_new_keys_have_defaults_and_drop_when_empty(self):
        for key in LTX25_ONLY_KEYS:
            assert key in LTX2_PARAM_KEYS and LTX2_DEFAULTS[key] == ""
            assert key in ltx.LTX2_OPTIONAL_STRING_KEYS


class TestCommands:
    def test_ltx25_cache_commands_use_the_split_pack(self):
        param_dict, _ = _normalize_ltx2_parameters(_ltx25(ltx2_mode="av"), validate_paths=False)
        latent, text = build_ltx2_cache_commands(param_dict, python_cmd="python")
        assert latent[latent.index("--ltx_version") + 1] == "2.5"
        assert latent[latent.index("--vae") + 1].endswith("ltx-2.5-video-vae-bf16.safetensors")
        assert latent[latent.index("--ltx2_audio_vae") + 1].endswith("ltx-2.5-audio-vae-bf16.safetensors")
        assert text[text.index("--ltx_version") + 1] == "2.5"
        assert text[text.index("--ltx2_text_encoder_checkpoint") + 1].endswith("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
        assert "--gemma_safetensors" not in text and "--gemma_root" not in text

    def test_video_mode_leaves_the_audio_vae_out_of_latent_caching(self):
        param_dict, _ = _normalize_ltx2_parameters(_ltx25(), validate_paths=False)
        latent, _ = build_ltx2_cache_commands(param_dict, python_cmd="python")
        assert "--ltx2_audio_vae" not in latent

    def test_ltx23_commands_ignore_filled_ltx25_fields(self):
        filled = {key: f"C:/models/{key}.safetensors" for key in LTX25_ONLY_KEYS}
        plain, _ = _normalize_ltx2_parameters(_params(), validate_paths=False)
        with_fields, _ = _normalize_ltx2_parameters(_params(**filled), validate_paths=False)
        assert build_ltx2_cache_commands(plain, python_cmd="python") == build_ltx2_cache_commands(with_fields, python_cmd="python")
        latent, text = build_ltx2_cache_commands(with_fields, python_cmd="python")
        for flag in ("--ltx_version", "--vae", "--ltx2_audio_vae", "--ltx2_text_encoder_checkpoint"):
            assert flag not in latent and flag not in text

    def test_ltx23_run_toml_has_no_ltx25_keys(self, tmp_path):
        filled = {key: f"C:/models/{key}.safetensors" for key in LTX25_ONLY_KEYS}
        config_path = str(tmp_path / "run.toml")
        prepare_ltx2_workflow(_params(**filled), config_path, validate_paths=False, python_cmd="python")
        data = toml.load(config_path)
        assert data["ltx_version"] == "2.3"
        assert not set(LTX25_ONLY_KEYS) & set(data)

    def test_ltx25_run_toml_carries_the_split_pack_and_full_sampling(self, tmp_path):
        config_path = str(tmp_path / "run.toml")
        prepare_ltx2_workflow(_ltx25(ltx2_mode="av"), config_path, validate_paths=False, python_cmd="python")
        data = toml.load(config_path)
        assert data["ltx_version"] == "2.5"
        assert data["ltx2_video_vae"].endswith("ltx-2.5-video-vae-bf16.safetensors")
        assert data["ltx2_audio_vae"].endswith("ltx-2.5-audio-vae-bf16.safetensors")
        assert data["ltx2_text_encoder_checkpoint"].endswith("gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
        # the trained dev transformer samples with its full schedule, not the distilled two-stage default
        assert data["sample_sampling_preset"] == "ltx25_full"
        assert "gemma_safetensors" not in data

    def test_an_explicit_sampling_preset_is_kept(self, tmp_path):
        param_dict, _ = _normalize_ltx2_parameters(_ltx25(sample_sampling_preset="ltx25"), validate_paths=False)
        assert param_dict["sample_sampling_preset"] == "ltx25"
        param_dict, _ = _normalize_ltx2_parameters(_params(), validate_paths=False)
        assert param_dict["sample_sampling_preset"] == "defaults"

    def test_ltx25_fine_tuning_has_no_bundle_merge(self, tmp_path):
        full = {"training_mode": "Full Fine-Tuning", "int8_convrot_dynamic": False, "optimizer_type": "AdaFactor",
                "optimizer_args": "scale_parameter=False relative_step=False warmup_init=False", "max_grad_norm": 0.0}
        config_path = str(tmp_path / "run.toml")
        workflow = prepare_ltx2_workflow(_ltx25(**full), config_path, validate_paths=False, python_cmd="python")
        assert workflow.post_train_commands == []
        workflow = prepare_ltx2_workflow(_params(**full), config_path, validate_paths=False, python_cmd="python")
        assert len(workflow.post_train_commands) == 1


@pytest.fixture(scope="module")
def parsers():
    """script -> run TOML path -> the trainer's arguments, as its read_config_from_file builds them."""
    import argparse

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from trainer_arg_types import capture_parser

    parsers = {script: capture_parser(f"musubi_tuner.{script}") for script in ("ltx2_train_network", "ltx2_train")}
    return lambda script, path: parsers[script].parse_args([], namespace=argparse.Namespace(**toml.load(path)))


class TestTrainerReadsTheRunToml:
    """The run TOML read as the trainers read it, parse_args(namespace=Namespace(**toml)) with their own parser: that
    fills only argparse dests, and the GUI keys ltx2_mode / no_convert_to_comfy / no_save_original_lora are option
    names whose dests are ltx_mode / convert_to_comfy / save_original_lora, so the run TOML writes the dests too
    (Guardrail 20: every Audio and Audio + Video run had trained video only)."""

    @pytest.mark.parametrize("version", ["2.3", "2.5"])
    @pytest.mark.parametrize("mode", ["video", "av", "audio"])
    def test_the_modality_reaches_the_trainer(self, parsers, tmp_path, version, mode):
        params = _ltx25(ltx2_mode=mode) if version == "2.5" else _params(ltx2_mode=mode)
        config_path = str(tmp_path / "run.toml")
        prepare_ltx2_workflow(params, config_path, validate_paths=False, python_cmd="python")
        args = parsers("ltx2_train_network", config_path)
        assert args.ltx_mode == mode
        assert args.convert_to_comfy is True and args.save_original_lora is True

    def test_fine_tuning_reads_the_modality_too(self, parsers, tmp_path):
        full = {"training_mode": "Full Fine-Tuning", "int8_convrot_dynamic": False, "optimizer_type": "AdaFactor",
                "optimizer_args": "scale_parameter=False relative_step=False warmup_init=False", "max_grad_norm": 0.0}
        config_path = str(tmp_path / "run.toml")
        prepare_ltx2_workflow(_params(ltx2_mode="av", **full), config_path, validate_paths=False, python_cmd="python")
        assert parsers("ltx2_train", config_path).ltx_mode == "av"

    def test_ltx25_samples_use_the_prompts_cached_with_the_captions(self, parsers, tmp_path):
        """LTX 2.5 text caching encodes the sample prompts too, so training never loads the 24.5 GB Gemma 4 (with the
        BF16 transformer moved to the CPU for the samples the two peaked at 61 GB of RAM)."""
        config_path = str(tmp_path / "run.toml")
        prompts = str(tmp_path / "prompts.txt")
        workflow = prepare_ltx2_workflow(_ltx25(sample_prompts=prompts, output_dir=str(tmp_path / "out")), config_path,
                                         validate_paths=False, python_cmd="python")
        cache = os.path.abspath(str(tmp_path / "out" / "test-lora_sample_prompts_te_cache.pt"))
        text = workflow.text_cache_command
        assert "--precache_sample_prompts" in text
        assert text[text.index("--sample_prompts") + 1] == prompts
        assert text[text.index("--sample_prompts_cache") + 1] == cache
        assert text[text.index("--sample_sampling_preset") + 1] == "ltx25_full"  # its default negative prompt
        args = parsers("ltx2_train_network", config_path)
        assert args.use_precached_sample_prompts is True and args.sample_prompts_cache == cache

    @pytest.mark.parametrize("case", ["ltx23", "train_connectors", "no_prompts"])
    def test_runs_that_encode_their_sample_prompts_themselves(self, parsers, tmp_path, case):
        overrides = {"sample_prompts": str(tmp_path / "prompts.txt"), "output_dir": str(tmp_path / "out")}
        if case == "train_connectors":
            overrides["train_connectors"] = True  # trained connectors change the embeddings during training
        elif case == "no_prompts":
            overrides["sample_prompts"] = ""
        params = _params(**overrides) if case == "ltx23" else _ltx25(**overrides)
        config_path = str(tmp_path / "run.toml")
        workflow = prepare_ltx2_workflow(params, config_path, validate_paths=False, python_cmd="python")
        assert "--precache_sample_prompts" not in workflow.text_cache_command
        assert parsers("ltx2_train_network", config_path).use_precached_sample_prompts is False

    def test_comfy_export_switches_reach_the_trainer(self, parsers, tmp_path):
        config_path = str(tmp_path / "run.toml")
        prepare_ltx2_workflow(_params(no_convert_to_comfy=True, no_save_original_lora=True), config_path, validate_paths=False,
                              python_cmd="python")
        args = parsers("ltx2_train_network", config_path)
        assert args.convert_to_comfy is False and args.save_original_lora is False


class TestValidation:
    def test_gemma3_file_is_refused_for_ltx25(self):
        with pytest.raises(ValueError, match="Gemma 4"):
            _normalize_ltx2_parameters(_ltx25(gemma_safetensors="C:/models/gemma_3_12B_it_fp8_e4m3fn.safetensors"), validate_paths=False)

    def test_text_encoder_file_and_directory_conflict(self):
        with pytest.raises(ValueError, match="not both"):
            _normalize_ltx2_parameters(_ltx25(gemma_root="C:/models/gemma4"), validate_paths=False)

    def test_split_pack_needs_its_files(self, tmp_path):
        transformer = _safetensors(tmp_path / "transformer.safetensors", "model.diffusion_model.patchify_proj.weight")
        text_encoder = _safetensors(tmp_path / "te.safetensors", "model.embed_tokens.weight")
        video_vae = _safetensors(tmp_path / "video_vae.safetensors", "encoder.conv_in.weight")
        audio_vae = _safetensors(tmp_path / "audio_vae.safetensors", "audio_vae.encoder.conv_in.weight")
        dataset = tmp_path / "dataset.toml"
        dataset.write_text("[general]\n", encoding="utf-8")
        base = dict(ltx2_checkpoint=transformer, ltx2_text_encoder_checkpoint=text_encoder, ltx2_video_vae=video_vae,
                    ltx2_audio_vae=audio_vae, dataset_config=str(dataset), output_dir=str(tmp_path / "out"))
        _normalize_ltx2_parameters(_ltx25(**base, ltx2_mode="av"))  # complete: no error
        with pytest.raises(ValueError, match="Video VAE"):
            _normalize_ltx2_parameters(_ltx25(**{**base, "ltx2_video_vae": ""}))
        with pytest.raises(ValueError, match="Audio VAE"):
            _normalize_ltx2_parameters(_ltx25(**{**base, "ltx2_audio_vae": ""}, ltx2_mode="av"))
        _normalize_ltx2_parameters(_ltx25(**{**base, "ltx2_audio_vae": ""}))  # video only: no audio VAE needed
        with pytest.raises(ValueError, match="text encoder"):
            _normalize_ltx2_parameters(_ltx25(**{**base, "ltx2_text_encoder_checkpoint": ""}))

    def test_unified_checkpoint_carries_its_vaes(self, tmp_path):
        unified = _safetensors(tmp_path / "ltx-2.5-22b-pt-bf16.safetensors", "model.diffusion_model.patchify_proj.weight",
                               "vae.encoder.conv_in.weight", "audio_vae.encoder.conv_in.weight")
        text_encoder = _safetensors(tmp_path / "te.safetensors", "model.embed_tokens.weight")
        dataset = tmp_path / "dataset.toml"
        dataset.write_text("[general]\n", encoding="utf-8")
        _normalize_ltx2_parameters(_ltx25(ltx2_checkpoint=unified, ltx2_text_encoder_checkpoint=text_encoder, ltx2_video_vae="",
                                          ltx2_audio_vae="", ltx2_mode="av", dataset_config=str(dataset), output_dir=str(tmp_path / "out")))

    def test_version_is_kept(self):
        param_dict, _ = _normalize_ltx2_parameters(_ltx25(), validate_paths=False)
        assert param_dict["ltx_version"] == "2.5"
        param_dict, _ = _normalize_ltx2_parameters(_params(ltx_version="9.9"), validate_paths=False)
        assert param_dict["ltx_version"] == "2.3"


class TestUnified:
    def test_ltx25_card(self):
        from musubi_tuner_gui.unified.registry import get_entry

        entry = get_entry("ltx25")
        assert entry.backend == "ltx2" and entry.fixed["ltx_version"] == "2.5"
        assert entry.model_folder == ltx.LTX25_MODEL_FOLDER
        assert ("LTX 2.5 models", "ltx25") in entry.download_bundles
        for key in ("ltx23", "ltx23_audio"):
            assert set(LTX25_ONLY_KEYS) <= get_entry(key).locked_keys
            assert "2.5" not in [value for _label, value in get_entry(key).choice_overrides["ltx_version"]]
        audio = get_entry("ltx25_audio")
        assert audio.fixed["ltx_version"] == "2.5" and audio.fixed["ltx2_mode"] == "audio"
        assert audio.model_folder == ltx.LTX25_MODEL_FOLDER and ("LTX 2.5 models", "ltx25") in audio.download_bundles
        for key in ("ltx25", "ltx25_audio"):
            assert get_entry(key).default_overrides["ltx_version_check_mode"] == "error"

    def test_version_link_switches_files_and_sampling(self, tmp_path, monkeypatch):
        from musubi_tuner_gui.unified import links as L

        v25 = tmp_path / "Training_Models_LTX_2_5"
        v25.mkdir()
        for name in ltx.ltx25_default_files().values():
            (v25 / name).write_bytes(b"x")
        monkeypatch.setattr(ltx, "ltx25_model_dir", lambda: str(v25))
        values = {"ltx_version": "2.5", "ltx2_checkpoint": "D:/other/Training_Models_LTX_2/ltx-2.3-22b-dev.safetensors",
                  "ltx2_text_encoder_checkpoint": "", "ltx2_video_vae": "", "ltx2_audio_vae": "",
                  "sample_sampling_preset": "defaults", "sample_vae_tile_overlap": 64, "sample_vae_temporal_tile_overlap": 8}
        changes = L.on_load("ltx2", {"ltx_version": "2.5"}, {}, values, preset_loaded=True)
        assert changes["ltx2_checkpoint"] == str(v25 / ltx.LTX25_TRANSFORMER_FILENAME)
        assert changes["ltx2_text_encoder_checkpoint"] == str(v25 / ltx.LTX25_TEXT_ENCODER_FILENAME)
        assert changes["ltx2_video_vae"] == str(v25 / ltx.LTX25_VIDEO_VAE_FILENAME)
        assert changes["ltx2_audio_vae"] == str(v25 / ltx.LTX25_AUDIO_VAE_FILENAME)
        assert changes["sample_sampling_preset"] == "ltx25_full"
        assert changes["sample_vae_tile_overlap"] == 160 and changes["sample_vae_temporal_tile_overlap"] == 40
        # a file the user picked stays
        picked = {**values, "ltx2_checkpoint": "E:/mine/my_ltx25_finetune.safetensors", "ltx2_video_vae": "E:/mine/vae.safetensors"}
        changes = L.on_load("ltx2", {"ltx_version": "2.5"}, {}, picked, preset_loaded=True)
        assert "ltx2_checkpoint" not in changes and "ltx2_video_vae" not in changes

    def test_ltx23_card_does_not_run_the_version_link_on_load(self):
        from musubi_tuner_gui.unified import links as L
        from musubi_tuner_gui.unified.registry import get_entry

        entry = get_entry("ltx23")
        values = {"ltx_version": "2.3", "ltx2_checkpoint": "C:/m/ltx-2.3-22b-dev.safetensors", "sample_sampling_preset": "defaults",
                  "ltx2_text_encoder_checkpoint": "", "ltx2_video_vae": "", "ltx2_audio_vae": "",
                  "sample_vae_tile_overlap": 64, "sample_vae_temporal_tile_overlap": 8}
        assert L.on_load("ltx2", entry.fixed, entry.default_overrides, values, preset_loaded=False) == {}
