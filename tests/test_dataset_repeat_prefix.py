"""Tests for dataset folder repeat-count parsing and the missing-prefix notification.

Dataset folders follow the Kohya convention "<repeats>_<name>" (e.g. "5_ohwx").
When the prefix is missing the GUI must default to 1 repeat and tell the user,
instead of silently counting the folder once.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.dataset_config_generator import (
    extract_repeat_count,
    generate_dataset_config_from_folders,
    generate_ltx2_dataset_config_from_folders,
    generate_wan_dataset_config_from_folders,
    parse_repeat_count,
)

# Minimal valid 1x1 PNG (no PIL dependency needed to write test images)
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8cfc000000301010018dd8db00000000049"
    "454e44ae426082"
)


def _write_png(path):
    with open(path, "wb") as handle:
        handle.write(_PNG_1X1)


class TestParseRepeatCount:
    @pytest.mark.parametrize(
        "folder,expected_count,expected_name",
        [
            ("3_ohwx", 3, "ohwx"),
            ("10_my_dataset", 10, "my_dataset"),
            ("1_a", 1, "a"),
            ("007_agent", 7, "agent"),
            ("2_5_things", 2, "5_things"),
        ],
    )
    def test_valid_prefix_no_notice(self, folder, expected_count, expected_name):
        count, name, notice = parse_repeat_count(folder)
        assert (count, name) == (expected_count, expected_name)
        assert notice is None

    @pytest.mark.parametrize(
        "folder",
        [
            "dataset",
            "my images",
            "ohwx woman",
            "_leading_underscore",
            "x5_name",
            "5x_name",
            "-3_name",
            "3.5_name",
        ],
    )
    def test_missing_prefix_defaults_to_one_with_notice(self, folder):
        count, name, notice = parse_repeat_count(folder)
        assert count == 1
        assert name == folder
        assert notice is not None
        assert notice.startswith("[INFO]")
        assert "counted as 1 repeat" in notice

    def test_zero_repeats_clamped_to_one_with_warning(self):
        count, name, notice = parse_repeat_count("0_ohwx")
        assert count == 1
        assert name == "ohwx"
        assert notice is not None
        assert notice.startswith("[WARNING]")

    @pytest.mark.parametrize("folder", ["5", "5_", "0", "123_"])
    def test_number_only_names_default_to_one_with_notice(self, folder):
        count, name, notice = parse_repeat_count(folder)
        assert count == 1
        assert name == folder
        assert notice is not None

    def test_empty_and_none_are_safe(self):
        assert parse_repeat_count("")[0] == 1
        assert parse_repeat_count(None)[0] == 1

    def test_extract_repeat_count_backward_compatible(self):
        assert extract_repeat_count("3_ohwx") == (3, "ohwx")
        assert extract_repeat_count("10_my_dataset") == (10, "my_dataset")
        assert extract_repeat_count("dataset") == (1, "dataset")


def _build_parent(tmp_path):
    """Parent folder with one well-formed and one prefix-less dataset folder."""
    parent = tmp_path / "datasets"
    good = parent / "5_goodconcept"
    forgotten = parent / "forgottenconcept"
    good.mkdir(parents=True)
    forgotten.mkdir()
    _write_png(str(good / "img1.png"))
    _write_png(str(forgotten / "img1.png"))
    return parent


def _datasets_by_folder(config, directory_key="image_directory"):
    return {
        os.path.basename(str(entry[directory_key]).replace("\\", "/").rstrip("/")): entry
        for entry in config["datasets"]
    }


class TestGeneratorsNotifyMissingPrefix:
    def test_image_generator(self, tmp_path):
        config, messages = generate_dataset_config_from_folders(
            parent_folder=str(_build_parent(tmp_path)),
            resolution=(1024, 1024),
        )
        by_dir = _datasets_by_folder(config)
        assert by_dir["5_goodconcept"]["num_repeats"] == 5
        assert by_dir["forgottenconcept"]["num_repeats"] == 1

        joined = "\n".join(messages)
        assert "'forgottenconcept'" in joined
        assert "counted as 1 repeat" in joined
        assert "'5_goodconcept': you forgot" not in joined

    def test_wan_generator(self, tmp_path):
        config, messages = generate_wan_dataset_config_from_folders(
            parent_folder=str(_build_parent(tmp_path)),
            resolution=(960, 544),
        )
        by_dir = _datasets_by_folder(config)
        assert by_dir["5_goodconcept"]["num_repeats"] == 5
        assert by_dir["forgottenconcept"]["num_repeats"] == 1
        assert any("counted as 1 repeat" in m for m in messages)

    def test_ltx2_generator(self, tmp_path):
        config, messages = generate_ltx2_dataset_config_from_folders(
            parent_folder=str(_build_parent(tmp_path)),
            resolution=(768, 512),
        )
        by_dir = _datasets_by_folder(config)
        assert by_dir["5_goodconcept"]["num_repeats"] == 5
        assert by_dir["forgottenconcept"]["num_repeats"] == 1
        assert any("counted as 1 repeat" in m for m in messages)

    def test_zero_prefix_folder_clamped_and_warned(self, tmp_path):
        parent = tmp_path / "datasets"
        zero = parent / "0_zeroconcept"
        zero.mkdir(parents=True)
        _write_png(str(zero / "img1.png"))

        config, messages = generate_dataset_config_from_folders(
            parent_folder=str(parent),
            resolution=(1024, 1024),
        )
        by_dir = _datasets_by_folder(config)
        assert by_dir["0_zeroconcept"]["num_repeats"] == 1
        assert any(m.startswith("[WARNING]") and "0_zeroconcept" in m for m in messages)

    def test_folder_name_captions_strip_prefix_only_when_present(self, tmp_path):
        parent = _build_parent(tmp_path)
        generate_dataset_config_from_folders(
            parent_folder=str(parent),
            resolution=(1024, 1024),
            create_missing_captions=True,
            caption_strategy="folder_name",
        )
        good_caption = (parent / "5_goodconcept" / "img1.txt").read_text(encoding="utf-8")
        forgotten_caption = (parent / "forgottenconcept" / "img1.txt").read_text(encoding="utf-8")
        assert good_caption == "goodconcept"
        assert forgotten_caption == "forgottenconcept"
