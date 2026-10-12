"""Inner folders of training folders (ignored, never a reason to skip the folder), the repeats & steps summary
and 🧹 Clean old caches (dataset_cleanup.py): only cache folders are deleted, after re-checking each target.

CPU only; tiny real images from tests/media_fixtures.py.
"""

from __future__ import annotations

import os
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (APP_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from media_fixtures import caption, write_image, write_video  # noqa: E402

from musubi_tuner_gui.dataset_config_generator import (  # noqa: E402
    generate_dataset_config_from_folders,
    generate_kandinsky6_dataset_config_from_folders,
    generate_ltx2_dataset_config_from_folders,
    generate_minimax_h3_dataset_config_from_folders,
    generate_wan_dataset_config_from_folders,
)
from musubi_tuner_gui.unified.dataset_cleanup import classify_folder, remove_cache_folders, validate_target  # noqa: E402
from musubi_tuner_gui.unified.dataset_inspector import (  # noqa: E402
    clean_plan,
    clean_question_html,
    inspect_dataset,
    report_html,
)


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))


def _cache_folder(folder, name: str, files: int = 3) -> str:
    path = os.path.join(folder, name)
    os.makedirs(path)
    for index in range(files):
        with open(os.path.join(path, f"img_{index:03d}_1024x1024_krea2.safetensors"), "wb") as handle:
            handle.write(b"\0" * 64)
    return path


def _dataset(tmp_path):
    """The user's case: 1_ohwx with its images, the current cache and old VRAM-lab caches beside them."""
    root = tmp_path / "dataset"
    folder = root / "1_ohwx"
    folder.mkdir(parents=True)
    for index in range(3):
        image = write_image(str(folder / f"photo_{index}.png"), size=(64, 64))
        caption(image, "ohwx man")
    _cache_folder(str(folder), "cache_dir", 2)
    for size in ("512x512", "1024x1024", "1328x1328"):
        _cache_folder(str(folder), f"cache_dir_lab_{size}")
    return root, folder


@pytest.mark.parametrize("generator", ["images", "wan", "ltx2", "kandinsky6"])
def test_generators_train_a_folder_that_holds_old_cache_folders(tmp_path, generator):
    root, folder = _dataset(tmp_path)
    if generator == "images":
        config, messages = generate_dataset_config_from_folders(str(root), (64, 64), cache_directory_name="cache_dir")
    elif generator == "wan":
        config, messages = generate_wan_dataset_config_from_folders(str(root), (64, 64), cache_directory_name="cache_dir")
    elif generator == "ltx2":
        config, messages = generate_ltx2_dataset_config_from_folders(str(root), (64, 64), cache_directory_name="cache_dir")
    else:
        config, messages = generate_kandinsky6_dataset_config_from_folders(str(root), (64, 64), cache_directory_name="cache_dir")
    assert len(config["datasets"]) == 1, messages
    assert not any("Contains subdirectories" in message for message in messages)
    assert any("cache_dir_lab_1024x1024" in message and "ignored" in message for message in messages), messages


def test_minimax_h3_trains_a_video_folder_that_holds_another_folder(tmp_path):
    root = tmp_path / "videos"
    folder = root / "1_clips"
    folder.mkdir(parents=True)
    caption(write_video(str(folder / "clip.mp4"), frames=9, size=(64, 64)), "a clip")
    (folder / "backup").mkdir()
    config, messages = generate_minimax_h3_dataset_config_from_folders(str(root), (64, 64), num_frames=124, cache_directory_name="cache_dir")
    assert len(config["datasets"]) == 1, messages


def test_a_folder_that_only_holds_folders_names_them(tmp_path):
    root = tmp_path / "dataset"
    (root / "project" / "1_ohwx").mkdir(parents=True)
    write_image(str(root / "project" / "1_ohwx" / "a.png"))
    with pytest.raises(ValueError) as error:
        generate_dataset_config_from_folders(str(root), (64, 64))
    assert "only holds the folder(s) 1_ohwx" in str(error.value)


def test_check_dataset_lists_inner_folders_and_the_repeat_math(tmp_path):
    root, folder = _dataset(tmp_path)
    extra = folder / "extra_photos"
    extra.mkdir()
    write_image(str(extra / "x.png"))
    (folder / "notes").mkdir()
    report = inspect_dataset(str(root), profile="image", control_name="", resolution=(64, 64), epochs=200, max_steps=80000)
    summary = report.folders[0]
    assert not summary.skipped and summary.images == 3
    kinds = {item.name: item.kind for item in summary.inner}
    assert kinds == {"cache_dir_lab_1024x1024": "cache", "cache_dir_lab_1328x1328": "cache", "cache_dir_lab_512x512": "cache",
                     "extra_photos": "media", "notes": "empty"}
    assert summary.active_caches == [str(folder / "cache_dir")]
    texts = " ".join(problem[1] for problem in report.problems)
    assert "hold media that is NOT trained" in texts and "3 old cache folder(s)" in texts and "Clean old caches" in texts
    page = report_html(report, "Krea 2")
    assert "Repeats &amp; steps" in page and "≈ 3</b> step(s) per epoch" in page and "≈ 600</b> steps in total" in page
    assert "+5 inner folders (ignored)" in page


def test_repeat_prefixes_are_explained(tmp_path):
    root = tmp_path / "dataset"
    for name, count in (("5_style", 2), ("plain", 1), ("0_zero", 1)):
        (root / name).mkdir(parents=True)
        for index in range(count):
            caption(write_image(str(root / name / f"{index}.png")), "x")
    report = inspect_dataset(str(root), profile="image", control_name="", batch_size=2, gradient_accumulation=2, max_steps=100)
    sources = {folder.name: (folder.repeats, folder.repeat_source) for folder in report.folders}
    assert sources == {"5_style": (5, "name"), "plain": (1, "none"), "0_zero": (1, "zero")}
    page = report_html(report)
    assert "<b>12</b> sample(s) per epoch" in page  # 2 × 5 + 1 + 1
    assert "÷ batch size 2 ÷ gradient accumulation 2 = <b>≈ 3</b>" in page
    assert "a <code>0_</code> prefix trains nothing" in page and "no repeat prefix" in page


def test_clean_deletes_only_old_cache_folders(tmp_path):
    root, folder = _dataset(tmp_path)
    keep = folder / "cache_dir_backup_with_photo"
    keep.mkdir()
    write_image(str(keep / "p.png"))  # a cache-named folder that holds an image is never a cache
    report = inspect_dataset(str(root), profile="image", control_name="", probe_seconds=0)
    plan = clean_plan(report)
    assert sorted(os.path.basename(path) for path, _root in plan["old"]) == ["cache_dir_lab_1024x1024", "cache_dir_lab_1328x1328", "cache_dir_lab_512x512"]
    assert [os.path.basename(path) for path, _root in plan["current"]] == ["cache_dir"]
    question, result = clean_question_html(report, plan)
    assert 'data-clean-count="4"' in question and not result
    removed, problems, freed = remove_cache_folders(plan["old"])
    assert len(removed) == 3 and not problems and freed == 3 * 3 * 64
    assert sorted(os.listdir(folder)) == ["cache_dir", "cache_dir_backup_with_photo", "photo_0.png", "photo_0.txt", "photo_1.png",
                                         "photo_1.txt", "photo_2.png", "photo_2.txt"]
    # the current cache only when asked for, and nothing to clean afterwards
    remove_cache_folders(plan["current"])
    assert not (folder / "cache_dir").exists()
    question, result = clean_question_html(*(lambda r: (r, clean_plan(r)))(inspect_dataset(str(root), profile="image", control_name="", probe_seconds=0)))
    assert not question and "nothing to clean" in result


def test_clean_re_checks_every_target_before_deleting(tmp_path):
    root, folder = _dataset(tmp_path)
    report = inspect_dataset(str(root), profile="image", control_name="", probe_seconds=0)
    plan = clean_plan(report)
    target = next(path for path, _root in plan["old"] if path.endswith("1024x1024"))
    with open(os.path.join(target, "my_notes.docx"), "wb") as handle:  # changed after the question was asked
        handle.write(b"x")
    removed, problems, _freed = remove_cache_folders(plan["old"])
    assert len(removed) == 2 and os.path.isdir(target)
    assert any("no longer holds only cache files" in problem for problem in problems)


def test_clean_never_leaves_the_training_folder(tmp_path):
    root, folder = _dataset(tmp_path)
    outside = _cache_folder(str(tmp_path), "cache_elsewhere")
    assert validate_target(outside, str(folder)) == "outside the dataset folder"
    assert validate_target(str(folder), str(folder)) == "the dataset folder itself"
    nested = _cache_folder(str(folder / "cache_dir"), "cache_inner")
    assert validate_target(nested, str(folder)) == "not directly inside its training folder"
    assert validate_target("", str(folder)) == "empty path"
    assert validate_target("relative/cache", str(folder)) == "not an absolute path"
    anchor = os.path.abspath(os.sep)
    assert validate_target(anchor, anchor) == "a drive or share root"
    removed, problems, _freed = remove_cache_folders([(outside, str(folder))])
    assert not removed and os.path.isdir(outside) and problems


def test_a_loras_folder_full_of_safetensors_is_not_a_cache(tmp_path):
    folder = tmp_path / "1_ohwx"
    folder.mkdir()
    loras = _cache_folder(str(folder), "trained_loras")
    assert classify_folder(loras).kind == "other"
    assert classify_folder(_cache_folder(str(folder), "my_latent_cache")).kind == "cache"


def test_links_are_never_followed_or_deleted(tmp_path):
    folder = tmp_path / "1_ohwx"
    folder.mkdir()
    target = _cache_folder(str(tmp_path), "real_cache")
    link = folder / "cache_link"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symbolic link needs a privilege here")
    assert classify_folder(str(link)).kind == "linked"
    assert validate_target(str(link), str(folder)) == "it is a link or junction"
    holder = _cache_folder(str(folder), "cache_with_link")
    os.symlink(target, os.path.join(holder, "inner_link"), target_is_directory=True)
    assert classify_folder(holder).kind == "linked"
    removed, problems, _freed = remove_cache_folders([(holder, str(folder)), (str(link), str(folder))])
    assert not removed and len(problems) == 2 and os.path.isdir(target) and len(os.listdir(target)) == 3
