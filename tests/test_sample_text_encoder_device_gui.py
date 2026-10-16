"""A preset that caches text encoder outputs on the CPU (its text encoder does not fit the GPU) also encodes the sample
prompts on the CPU: the runtime TOML gets sample_text_encoder_device = "cpu"."""

from __future__ import annotations

import os
import sys

import toml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.common_gui import SaveConfigFileToRun  # noqa: E402


def test_cpu_text_encoder_caching_moves_the_sample_prompts_text_encoder_to_the_cpu(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a photo --d 1\n", encoding="utf-8")

    def runtime(device: str, sample: str, extra=()) -> dict:
        path = tmp_path / f"run_{device}_{bool(sample)}_{len(extra)}.toml"
        parameters = [("caching_teo_device", device), ("sample_prompts", sample), ("learning_rate", 1e-4), *extra]
        SaveConfigFileToRun(parameters=parameters, file_path=str(path), exclusion=["caching_teo_device"])
        return toml.load(path)

    assert runtime("cpu", str(prompts))["sample_text_encoder_device"] == "cpu"
    assert runtime(" CPU ", str(prompts))["sample_text_encoder_device"] == "cpu"
    assert "sample_text_encoder_device" not in runtime("cuda", str(prompts))
    assert "sample_text_encoder_device" not in runtime("cpu", "")  # no samples: nothing to encode
    assert runtime("cpu", str(prompts), [("sample_text_encoder_device", "cuda")])["sample_text_encoder_device"] == "cpu"
    assert "caching_teo_device" not in runtime("cpu", str(prompts))
