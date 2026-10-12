"""Dataset folders -> dataset TOMLs for every kind of data each model supports (unified/dataset_tools.py).

Uses tiny real media written with PyAV (tests/media_fixtures.py). CPU only; nothing is trained.
"""

from __future__ import annotations

import os
import sys
import time

import pytest
import toml

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (APP_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from media_fixtures import audio_seconds, caption, write_image, write_video, write_wav  # noqa: E402
from musubi_tuner.dataset.config_utils import ConfigSanitizer  # noqa: E402

from musubi_tuner_gui.unified.backends import get_backend  # noqa: E402
from musubi_tuner_gui.unified.dataset_tools import (  # noqa: E402
    EXTRACTED_AUDIO_DIRNAME,
    FOLDER_MODE,
    audio_from_videos,
    extract_audio_track,
    has_audio_track,
    prepare_dataset,
)


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))


def _values(backend_name: str, parent, out, **changes) -> dict:
    backend = get_backend(backend_name)
    values = backend.defaults("lora")
    values.update(dataset_config_mode=FOLDER_MODE, parent_folder_path=str(parent), output_dir=str(out), create_missing_captions=True, caption_strategy="folder_name")
    values.update(changes)
    return backend.normalized(values)


def _load(values: dict) -> dict:
    path = values["dataset_config"]
    assert os.path.isfile(path) and values["generated_toml_path"] == path
    config = toml.load(path)
    ConfigSanitizer().sanitize_user_config(config)  # musubi's own dataset schema accepts it
    return config


def _entries(config: dict, key: str) -> list[dict]:
    return [entry for entry in config["datasets"] if key in entry]


def _same(a: str, b) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(str(b)))


# --------------------------------------------------------------------------------------------- audio extraction
def test_audio_track_extraction(tmp_path):
    talking = write_video(str(tmp_path / "talk.mp4"), frames=24, audio_seconds=1.0)
    silent = write_video(str(tmp_path / "silent.mp4"), frames=12)
    assert has_audio_track(talking) and not has_audio_track(silent)
    target = str(tmp_path / "talk.wav")
    assert extract_audio_track(talking, target)
    assert audio_seconds(target) == pytest.approx(1.0, abs=0.05)
    assert not extract_audio_track(silent, str(tmp_path / "silent.wav"))
    assert not os.path.exists(str(tmp_path / "silent.wav")) and not os.path.exists(target + ".partial")


def test_a_failed_extraction_leaves_no_clip_behind(tmp_path, monkeypatch):
    """Audio that breaks mid-stream (damaged data): nothing musubi would read as a clip stays behind."""
    import av
    from musubi_tuner.dataset.image_video_dataset import glob_audio

    folder = tmp_path / "1_voice"
    folder.mkdir()
    caption(write_video(str(folder / "bad.mp4"), frames=24, audio_seconds=1.0), "bad")
    caption(write_video(str(folder / "good.mp4"), frames=24, audio_seconds=0.5), "good")
    real, partial_seen = av.AudioResampler, []

    class BreaksOnBadClip:
        def __init__(self, *args, **kwargs):
            self.inner, self.calls = real(*args, **kwargs), 0

        def resample(self, frame):
            self.calls += 1
            partial = folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME / "bad.mp4.wav.partial"
            if partial.exists() and self.calls > 3:
                partial_seen.append(True)
                raise RuntimeError("Invalid data found when processing input")
            return self.inner.resample(frame)

    monkeypatch.setattr(av, "AudioResampler", BreaksOnBadClip)
    messages = []
    target_dir, count = audio_from_videos(str(folder), "cache_dir", ".txt", None, messages)
    assert partial_seen and count == 1 and any("could not read the audio of 1 video" in line for line in messages)
    assert sorted(os.listdir(target_dir)) == ["good.mp4.txt", "good.mp4.wav", "sources.json"]
    assert [os.path.basename(path) for path in glob_audio(target_dir)] == ["good.mp4.wav"]


def test_ltx_audio_uses_clips_and_the_audio_of_videos(tmp_path):
    folder = tmp_path / "data" / "2_my voice"
    folder.mkdir(parents=True)
    caption(write_wav(str(folder / "clip.wav")), "a calm voice")
    caption(write_video(str(folder / "talk.mp4"), frames=24, audio_seconds=1.0), "a person talking")
    write_video(str(folder / "no_sound.mp4"), frames=12)
    values = _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio")
    notes = prepare_dataset("ltx2", values)
    config = _load(values)
    audio = _entries(config, "audio_directory")
    assert len(audio) == 2 and not _entries(config, "video_directory") and not _entries(config, "image_directory")
    extracted = folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME
    assert {os.path.normcase(os.path.normpath(entry["audio_directory"])) for entry in audio} == {os.path.normcase(str(folder)), os.path.normcase(str(extracted))}
    assert all(entry["num_repeats"] == 2 for entry in audio)
    assert len({entry["cache_directory"] for entry in audio}) == 2  # never share a cache folder
    assert (extracted / "talk.mp4.wav").is_file() and (extracted / "talk.mp4.txt").read_text(encoding="utf-8") == "a person talking"
    assert not (extracted / "no_sound.mp4.wav").exists()
    assert any("no audio track" in note and "no_sound.mp4" in note for note in notes)
    # a second Start reuses the extraction; an edited caption is copied again
    stamp = os.path.getmtime(extracted / "talk.mp4.wav")
    time.sleep(0.05)
    caption(str(folder / "talk.mp4"), "an edited caption")
    prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio"))
    assert os.path.getmtime(extracted / "talk.mp4.wav") == stamp
    assert (extracted / "talk.mp4.txt").read_text(encoding="utf-8") == "an edited caption"


def test_ltx_audio_from_videos_only_and_missing_captions(tmp_path):
    folder = tmp_path / "data" / "1_speaker"
    folder.mkdir(parents=True)
    write_video(str(folder / "a.mp4"), frames=24, audio_seconds=0.5)
    values = _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio")
    prepare_dataset("ltx2", values)
    config = _load(values)
    assert len(_entries(config, "audio_directory")) == 1
    assert (folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME / "a.mp4.txt").read_text(encoding="utf-8") == "speaker"


def _extracted(folder) -> set:
    return set(os.listdir(folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME)) - {"sources.json"}


def test_extracted_audio_follows_the_videos(tmp_path):
    """Removed, replaced (even with an older file date), renamed-twin and uncaptioned videos never train stale audio."""
    from musubi_tuner.dataset.image_video_dataset import glob_audio

    folder = tmp_path / "data" / "1_voice"
    folder.mkdir(parents=True)
    caption(write_video(str(folder / "a.mp4"), frames=24, audio_seconds=0.5), "a")
    caption(write_video(str(folder / "b.mp4"), frames=24, audio_seconds=0.5), "b")
    write_video(str(folder / "clip.mov"), frames=24, audio_seconds=0.5)  # same stem as clip.mp4, different video
    caption(write_video(str(folder / "clip.mp4"), frames=24, audio_seconds=0.7), "clip mp4")
    caption(str(folder / "clip.mov"), "clip mov")
    (folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME).mkdir(parents=True)
    (folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME / "a.wav.partial.wav").write_bytes(b"RIFF")  # left by an older version
    run = lambda **changes: prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio", **changes))  # noqa: E731
    run()
    assert _extracted(folder) == {"a.mp4.wav", "a.mp4.txt", "b.mp4.wav", "b.mp4.txt", "clip.mp4.wav", "clip.mp4.txt", "clip.mov.wav", "clip.mov.txt"}
    extracted = folder / "cache_dir" / EXTRACTED_AUDIO_DIRNAME
    assert audio_seconds(str(extracted / "clip.mp4.wav")) == pytest.approx(0.7, abs=0.05)
    # a.mp4 removed, b.mp4 replaced by a longer clip whose file date is older than the extraction
    os.remove(folder / "a.mp4")
    old = os.path.getmtime(extracted / "b.mp4.wav") - 3600
    write_video(str(folder / "b.mp4"), frames=24, audio_seconds=1.5)
    os.utime(folder / "b.mp4", (old, old))
    run()
    assert "a.mp4.wav" not in _extracted(folder) and "a.mp4.txt" not in _extracted(folder)
    assert audio_seconds(str(extracted / "b.mp4.wav")) == pytest.approx(1.5, abs=0.05)
    # videos without a caption are skipped while missing captions are not created
    os.remove(folder / "clip.txt")  # the caption both clip.mp4 and clip.mov read
    notes = run(create_missing_captions=False)
    assert not {"clip.mov.wav", "clip.mp4.wav"} & _extracted(folder)
    assert any("no .txt caption and are skipped" in note and "clip.mov" in note for note in notes)
    assert {os.path.basename(path) for path in glob_audio(str(extracted))} == {"b.mp4.wav"}  # what musubi trains on


def test_uncaptioned_audio_clips_stop_at_start_with_a_clear_message(tmp_path):
    folder = tmp_path / "data" / "1_voice"
    folder.mkdir(parents=True)
    caption(write_wav(str(folder / "good.wav")), "good")
    write_wav(str(folder / "no caption.wav"))
    with pytest.raises(ValueError, match="no caption.wav.*Create Missing Captions"):
        prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio", create_missing_captions=False))
    prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio"))  # creates the caption
    assert (folder / "no caption.txt").read_text(encoding="utf-8") == "voice"


def test_no_audio_error_says_what_went_wrong(tmp_path):
    folder = tmp_path / "data" / "1_voice"
    folder.mkdir(parents=True)
    caption(write_video(str(folder / "talk.mp4"), frames=24, audio_seconds=0.5), "talk")
    (folder / "cache_dir").write_text("a file where the cache folder belongs", encoding="utf-8")
    with pytest.raises(ValueError, match="(?s)videos with.*sound.*cannot create the folder for the extracted audio"):
        prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio"))


def test_stop_during_audio_extraction_ends_the_run(tmp_path, monkeypatch):
    from musubi_tuner_gui.common_gui import TrainingCancelled
    from musubi_tuner_gui.unified import dataset_tools
    from musubi_tuner_gui.unified.runner import RUNNER

    folder = tmp_path / "data" / "1_voice"
    folder.mkdir(parents=True)
    for name in ("a", "b"):
        caption(write_video(str(folder / f"{name}.mp4"), frames=12, audio_seconds=0.3), name)
    calls = []

    def stop_after_first(video, target):
        calls.append(video)
        monkeypatch.setattr(RUNNER, "_cancel_requested", True)
        monkeypatch.setattr(RUNNER, "_job_thread_ident", __import__("threading").get_ident())
        return True

    monkeypatch.setattr(dataset_tools, "extract_audio_track", stop_after_first)
    with pytest.raises(TrainingCancelled):
        prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio"))
    assert len(calls) == 1


def test_ltx_audio_without_any_audio_explains_what_to_add(tmp_path):
    folder = tmp_path / "data" / "1_x"
    folder.mkdir(parents=True)
    write_video(str(folder / "silent.mp4"), frames=12)
    write_image(str(folder / "a.png"))
    with pytest.raises(ValueError, match="videos with sound"):
        prepare_dataset("ltx2", _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="audio"))


# --------------------------------------------------------------------------------------------- images / videos / mixed
def _mixed_dataset(tmp_path):
    root = tmp_path / "data"
    mixed = root / "1_mixed"
    images = root / "3_images"
    videos = root / "2_videos"
    for folder in (mixed, images, videos):
        folder.mkdir(parents=True)
    caption(write_video(str(mixed / "clip.mp4"), frames=17, audio_seconds=0.7), "clip")
    caption(write_image(str(mixed / "still.png")), "still")
    caption(write_image(str(images / "a.png")), "a")
    caption(write_video(str(videos / "b.mp4"), frames=17), "b")
    return root, mixed, images, videos


@pytest.mark.parametrize(
    "backend_name, changes",
    [
        ("wan", {"target_frames": 17}),
        ("ltx2", {"ltx2_mode": "video", "dataset_num_frames": 17}),
        ("kandinsky6", {"num_frames": 17, "dataset_resolution_width": 64, "dataset_resolution_height": 64}),
    ],
)
def test_video_models_train_images_videos_and_mixed_folders(tmp_path, backend_name, changes):
    root, mixed, images, videos = _mixed_dataset(tmp_path)
    values = _values(backend_name, root, tmp_path / "out", **changes)
    notes = prepare_dataset(backend_name, values)
    config = _load(values)
    image_dirs = [entry["image_directory"] for entry in _entries(config, "image_directory")]
    video_dirs = [entry["video_directory"] for entry in _entries(config, "video_directory")]
    assert any(_same(path, mixed) for path in video_dirs) and any(_same(path, videos) for path in video_dirs)
    assert any(_same(path, images) for path in image_dirs)
    mixed_images = [entry for entry in _entries(config, "image_directory") if _same(entry["image_directory"], mixed)]
    assert len(mixed_images) == 1, "the images of a mixed folder must train too"
    mixed_video = next(entry for entry in _entries(config, "video_directory") if _same(entry["video_directory"], mixed))
    assert mixed_images[0]["num_repeats"] == mixed_video["num_repeats"] == 1
    assert mixed_images[0].get("cache_directory") != mixed_video.get("cache_directory")
    assert any("images train too" in note or "as a separate image dataset" in note for note in notes)
    assert not any("images are not used" in note for note in notes)


@pytest.mark.parametrize(
    "backend_name, changes",
    [
        ("wan", {"target_frames": 17}),
        ("ltx2", {"ltx2_mode": "video", "dataset_num_frames": 17}),
        ("kandinsky6", {"num_frames": 17, "dataset_resolution_width": 64, "dataset_resolution_height": 64}),
        ("qwen_image", {}),
    ],
)
def test_the_next_start_after_caching_still_finds_every_folder(tmp_path, backend_name, changes):
    """Caching creates <folder>/cache_dir and, for the images of a mixed folder, <folder>/cache_dir_images.
    The generators skip folders holding other folders, but never because of their own cache folders."""
    root, mixed, images, videos = _mixed_dataset(tmp_path)
    first = _values(backend_name, root, tmp_path / "out", **changes)
    prepare_dataset(backend_name, first)
    for config_entry in _load(first)["datasets"]:
        os.makedirs(config_entry["cache_directory"], exist_ok=True)  # what musubi's caching does
    assert (mixed / "cache_dir_images").is_dir() or backend_name == "qwen_image"
    (images / "cache_dir_images").mkdir(exist_ok=True)
    second = _values(backend_name, root, tmp_path / "out", **changes)
    notes = prepare_dataset(backend_name, second)
    assert not any("Contains subdirectories" in note for note in notes), notes
    assert len(_load(second)["datasets"]) == len(_load(first)["datasets"])


def test_check_dataset_ignores_inner_folders_instead_of_skipping_the_folder(tmp_path):
    """The trainers read only the files directly inside a training folder, so an inner folder (an old
    cache_dir_lab_1024x1024, a backup) is ignored; it once made Start skip the whole folder."""
    from musubi_tuner_gui.unified.dataset_tools import scan_dataset

    root, mixed, images, _videos = _mixed_dataset(tmp_path)
    (mixed / "cache_dir_images").mkdir()
    (mixed / "cache_dir").mkdir()
    report = scan_dataset(str(root), profile="video")
    assert "Dataset looks good" in report and "Inner folders are ignored" not in report  # the current cache is no inner folder
    (images / "originals").mkdir()
    report = scan_dataset(str(root), profile="video")
    assert "Inner folders are ignored (only the files directly inside a training folder are trained): `3_images/originals`" in report
    assert "skipped" not in report and "Dataset looks good" in report
    assert "`3_images/originals`" not in scan_dataset(str(root), profile="video", report_inner=False)
    (images / "my_cache").mkdir()  # a cache folder of its own name is not listed once the check knows it
    assert "`3_images/my_cache`" in scan_dataset(str(root), profile="video")
    assert "`3_images/my_cache`" not in scan_dataset(str(root), profile="video", cache_directory="my_cache")


def test_check_dataset_explains_a_parent_folder_that_holds_the_files_itself(tmp_path):
    from musubi_tuner_gui.unified.dataset_tools import scan_dataset

    folder = tmp_path / "ohwx"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"")
    report = scan_dataset(str(folder), profile="image")
    assert "holds 1 media file(s) directly but no training folders" in report and "1_ohwx" in report


def test_ltx_audio_video_mode_keeps_videos_only_in_mixed_folders(tmp_path):
    root, mixed, _images, _videos = _mixed_dataset(tmp_path)
    values = _values("ltx2", root, tmp_path / "out", ltx2_mode="av", dataset_num_frames=17)
    notes = prepare_dataset("ltx2", values)
    config = _load(values)
    assert not [entry for entry in _entries(config, "image_directory") if _same(entry["image_directory"], mixed)]
    assert any("images are not used" in note for note in notes)


@pytest.mark.parametrize("backend_name", ["wan", "kandinsky6"])
def test_video_models_accept_image_only_datasets(tmp_path, backend_name):
    folder = tmp_path / "data" / "5_person"
    folder.mkdir(parents=True)
    for index in range(2):
        write_image(str(folder / f"{index}.png"))
    changes = {"dataset_resolution_width": 64, "dataset_resolution_height": 64} if backend_name == "kandinsky6" else {}
    values = _values(backend_name, tmp_path / "data", tmp_path / "out", **changes)
    prepare_dataset(backend_name, values)
    config = _load(values)
    assert [entry["num_repeats"] for entry in _entries(config, "image_directory")] == [5]
    assert not _entries(config, "video_directory")
    assert (folder / "0.txt").read_text(encoding="utf-8") == "person"  # missing captions from the folder name


def test_ltx_video_mode_accepts_image_only_datasets(tmp_path):
    folder = tmp_path / "data" / "1_person"
    folder.mkdir(parents=True)
    write_image(str(folder / "a.png"))
    values = _values("ltx2", tmp_path / "data", tmp_path / "out", ltx2_mode="video")
    prepare_dataset("ltx2", values)
    assert len(_entries(_load(values), "image_directory")) == 1


def test_minimax_h3_explains_that_it_needs_videos(tmp_path):
    from musubi_tuner_gui.dataset_config_generator import generate_minimax_h3_dataset_config_from_folders

    folder = tmp_path / "data" / "1_person"
    folder.mkdir(parents=True)
    write_image(str(folder / "a.png"))
    with pytest.raises(ValueError, match="video targets"):
        generate_minimax_h3_dataset_config_from_folders(parent_folder=str(tmp_path / "data"), resolution=(64, 64))


def test_check_dataset_tells_what_the_model_will_do(tmp_path):
    from musubi_tuner_gui.unified.dataset_tools import dataset_profile, scan_dataset

    root, _mixed, _images, _videos = _mixed_dataset(tmp_path)
    assert dataset_profile("krea2", {}) == "image" and dataset_profile("ltx2", {"ltx2_mode": "av"}) == "av"
    assert "videos only; folders with only images are skipped: `3_images`" in scan_dataset(f'"{root}"', profile="video_only")
    assert "mix images and videos: both are trained" in scan_dataset(str(root), profile="video")
    assert "the images next to them are not used" in scan_dataset(str(root), profile="av")
    assert "This model trains on images: the 2 video(s)" in scan_dataset(str(root), profile="image")
    audio_report = scan_dataset(str(root), profile="audio")
    assert "1 video(s) have sound" in audio_report and "1 video(s) have no audio track" in audio_report
    empty = tmp_path / "empty" / "1_x"
    empty.mkdir(parents=True)
    write_image(str(empty / "a.png"))
    assert "Nothing this model can train on was found: add audio clips or videos with sound" in scan_dataset(str(empty.parent), profile="audio")


def test_messy_dataset_paths_still_work(tmp_path):
    folder = tmp_path / "my data" / "1_ohwx"
    folder.mkdir(parents=True)
    caption(write_image(str(folder / "a.png")), "ohwx")
    messy = f'  "{str(tmp_path / "my data").replace(os.sep, "/")}/"  \n'
    values = _values("qwen_image", messy, f"'{tmp_path / 'out'}'")
    prepare_dataset("qwen_image", values)
    assert [entry["num_repeats"] for entry in _entries(_load(values), "image_directory")] == [1]


# --------------------------------------------------------------------------------------------- dataset check (inspector)
def _edit_dataset(root) -> str:
    folder = root / "2_ohwx edits"
    controls = folder / "control_images"
    controls.mkdir(parents=True)
    for name in ("t1", "t2", "t3"):
        caption(write_image(str(folder / f"{name}.png"), size=(768, 512)), f"make the sky of {name} purple")
    write_image(str(controls / "t1.png"))
    write_image(str(controls / "t2_0.png"))
    write_image(str(controls / "t2_1.png"))
    write_image(str(controls / "t1_extra.png"))  # musubi stops on this name: only digits after "t1_"
    write_image(str(controls / "zzz.png"))  # matches no target
    write_image(str(folder / "nocap.jpg"), size=(300, 200))
    nested = root / "1_nested"
    (nested / "originals").mkdir(parents=True)
    write_image(str(nested / "a.png"))
    return str(root)


def test_edit_dataset_check_pairs_every_target_with_its_controls(tmp_path):
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset, report_html, sample_previews

    report = inspect_dataset(_edit_dataset(tmp_path), profile="image", control_name="control_images", control_required=True, resolution=(1024, 1024))
    folders = {folder.name: folder for folder in report.folders}
    # an inner folder is ignored by the trainer, never a reason to skip its training folder
    assert not folders["1_nested"].skipped and [(item.name, item.kind) for item in folders["1_nested"].inner] == [("originals", "empty")]
    edits = folders["2_ohwx edits"]
    # nocap.jpg has no caption: musubi does not train it, so it is no edit target either
    assert (edits.images, edits.paired, edits.unpaired, edits.unused_controls, edits.missing_captions) == (4, 2, 1, 1, 1)
    problems = {text.split(" ", 2)[1]: (level, examples) for level, text, examples in report.problems}
    joined = "\n".join(text for _level, text, _examples in report.problems)
    assert "stop musubi's caching" in joined and "have no control image" in joined and "match no trained target" in joined
    assert any(level == "error" and "t1_extra.png" in " ".join(examples) for level, _text, examples in report.problems)
    assert problems  # every problem names its files
    by_name = {os.path.basename(item.path): item for item in report.items}
    assert [os.path.basename(path) for path in by_name["t2.png"].controls] == ["t2_0.png", "t2_1.png"]
    html_text = report_html(report, "Mage-Flow Edit")
    assert "Edit pairs" in html_text and "2 / 3" in html_text and "+1 inner folder (ignored)" in html_text and "NOT trained" in html_text
    previews = sample_previews(report, count=8)
    assert len(previews) == 3 and all(os.path.isfile(path) for path, _caption in previews)
    from PIL import Image

    pair = next(path for path, text in previews if "t2.png" in text)
    with Image.open(pair) as image:
        assert image.width > 2 * image.height  # two controls, an arrow and the target side by side
    assert sample_previews(report, count=8) == previews  # cached by file version, same order for one seed


def test_dataset_check_reads_the_folders_of_a_dataset_toml(tmp_path):
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset

    folder = tmp_path / "images"
    (tmp_path / "controls").mkdir()
    folder.mkdir()
    caption(write_image(str(folder / "a.png")), "edit a")
    caption(write_image(str(folder / "b.png")), "edit b")
    write_image(str(tmp_path / "controls" / "a.png"))
    musubi = tmp_path / "dataset.toml"
    musubi.write_text(
        toml.dumps({"general": {"resolution": [512, 512], "batch_size": 2, "caption_extension": ".txt"},
                    "datasets": [{"image_directory": str(folder), "control_directory": str(tmp_path / "controls"), "num_repeats": 3}]}),
        encoding="utf-8",
    )
    report = inspect_dataset("", profile="image", control_name="control_images", dataset_toml=str(musubi))
    assert not report.error and report.batch_size == 2 and report.folders[0].repeats == 3
    assert (report.folders[0].paired, report.folders[0].unpaired) == (1, 1)
    kohya = tmp_path / "kohya.toml"
    kohya.write_text(toml.dumps({"datasets": [{"resolution": 1024, "subsets": [{"image_dir": str(folder), "num_repeats": 5}]}]}), encoding="utf-8")
    report = inspect_dataset("", profile="image", control_name="", dataset_toml=str(kohya))
    assert [(folder.images, folder.repeats) for folder in report.folders] == [(2, 5)]
    assert "not found" in inspect_dataset("", dataset_toml=str(tmp_path / "missing.toml")).error


def test_dataset_check_measures_videos_and_audio_and_previews_them(tmp_path):
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset, report_html, sample_previews

    clips = tmp_path / "1_clips"
    clips.mkdir()
    caption(write_video(str(clips / "c1.mp4"), frames=25, size=(96, 64), audio_seconds=1.0), "a person walking")
    write_image(str(clips / "still.png"))
    report = inspect_dataset(str(tmp_path), profile="video")
    video = next(item for item in report.items if item.kind == "video")
    assert video.size == (96, 64) and video.fps == pytest.approx(24) and video.has_sound and video.duration == pytest.approx(1.0, abs=0.15)
    assert "Total length" in report_html(report) and "Frame rate" in report_html(report)
    assert len(sample_previews(report)) == 1  # still.png has no caption: musubi skips it
    voice = tmp_path / "audio" / "1_voice"
    voice.mkdir(parents=True)
    caption(write_wav(str(voice / "v1.wav"), seconds=2.0), "hello there")
    report = inspect_dataset(str(tmp_path / "audio"), profile="audio")
    assert report.items[0].duration == pytest.approx(2.0, abs=0.05)
    assert len(sample_previews(report)) == 1


def test_dataset_check_html_escapes_names_and_captions(tmp_path):
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset, report_html

    folder = tmp_path / "1_Tom & Jerry's best"  # (Windows allows no < > or " in names)
    folder.mkdir()
    write_image(str(folder / "a.png"))
    text = report_html(inspect_dataset(str(tmp_path)))
    assert "Tom &amp; Jerry&#x27;s best" in text and "Tom & Jerry" not in text


def test_control_image_rules_per_model():
    """As each model's dataset generator uses control folders (a folder it does not use is ignored like any inner folder)."""
    from musubi_tuner_gui.unified.dataset_inspector import control_settings

    assert control_settings("qwen_image", {"model_version": "edit-2511", "dataset_control_directory": "edit_images"}) == ("edit_images", True)
    assert control_settings("qwen_image", {"model_version": "edit-2511", "dataset_control_directory": ""}) == ("control_images", True)  # as Start
    assert control_settings("qwen_image", {"model_version": "original", "dataset_control_directory": "refs"}) == ("refs", False)
    assert control_settings("mage_flow", {"use_control_images": True}) == ("control_images", True)
    assert control_settings("qwen_image_21", {"use_control_images": False, "control_directory_name": "control_images"}) == ("", False)
    assert control_settings("hidream_o1", {"use_control_images": True, "task": "i2i", "control_directory_name": "src"}) == ("src", True)
    assert control_settings("flux", {}) == ("control_images", False)  # FLUX.2 uses the folder when it exists
    assert control_settings("krea2", {}) == ("", False) and control_settings("zimage", {}) == ("", False)


def test_a_control_folder_the_model_does_not_use_is_ignored(tmp_path):
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset

    folder = tmp_path / "1_ohwx"
    (folder / "control_images").mkdir(parents=True)
    caption(write_image(str(folder / "a.png")), "a photo")
    write_image(str(folder / "control_images" / "a.png"))
    report = inspect_dataset(str(tmp_path), profile="image", control_name="")  # Krea 2: no control images
    assert not report.folders[0].skipped and [os.path.basename(item.path) for item in report.used] == ["a.png"]
    assert [(item.name, item.kind) for item in report.folders[0].inner] == [("control_images", "media")]
    assert any(level == "warn" and "NOT trained" in text for level, text, _examples in report.problems)


def test_kohya_folder_rules(tmp_path):
    """kohya trains <repeats>_<name> folders only and does not mind folders inside them."""
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset

    for name in ("10_ohwx man", "ohwx", "0_empty"):
        (tmp_path / name).mkdir()
        caption(write_image(str(tmp_path / name / "a.png")), "ohwx man")
    (tmp_path / "10_ohwx man" / "originals").mkdir()
    report = inspect_dataset(str(tmp_path), profile="image", control_name="", kohya=True)
    folders = {folder.name: folder for folder in report.folders}
    assert not folders["10_ohwx man"].skipped and folders["10_ohwx man"].repeats == 10 and folders["10_ohwx man"].images == 1
    assert folders["ohwx"].skipped.startswith("ignored by kohya") and folders["0_empty"].skipped.startswith("ignored by kohya")


def test_control_pairing_follows_musubi_exactly():
    """musubi: longest target first, only the text after the last "_" must be digits, numeric order."""
    from musubi_tuner_gui.unified.dataset_inspector import match_controls

    matches, unused, rejected = match_controls(
        ["photo.png", "pic.png", "x.png"],
        ["photo_a_0.png", "photo_10.png", "photo_2.png", "pic.1.png", "x.png", "lonely.png"],
    )
    assert matches["photo.png"] == ["photo_2.png", "photo_a_0.png", "photo_10.png"] or matches["photo.png"] == ["photo_a_0.png", "photo_2.png", "photo_10.png"]
    assert [name for name in matches["photo.png"] if name != "photo_a_0.png"] == ["photo_2.png", "photo_10.png"]  # numeric, not text order
    assert rejected == ["pic.1.png"] and unused == ["lonely.png"] and matches["x.png"] == ["x.png"]
    # the longest target owns a shared prefix: "a_b_0" goes to a_b.png, not to a.png
    matches, _unused, _rejected = match_controls(["a.png", "a_b.png"], ["a_b_0.png", "a_0.png"])
    assert matches == {"a_b.png": ["a_b_0.png"], "a.png": ["a_0.png"]}


def test_control_pairing_is_fast_on_big_datasets():
    import time as clock

    from musubi_tuner_gui.unified.dataset_inspector import match_controls

    targets = [f"img{index:05d}.png" for index in range(20000)]
    controls = [f"img{index:05d}_0.png" for index in range(20000)]
    started = clock.perf_counter()
    matches, unused, rejected = match_controls(targets, controls)
    assert len(matches) == 20000 and not unused and not rejected
    assert clock.perf_counter() - started < 2.0


def test_what_musubi_refuses_is_an_error(tmp_path):
    """Unpaired edit targets, non-UTF-8 captions, cut-off images and uncaptioned videos stop caching."""
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset

    folder = tmp_path / "1_edits"
    (folder / "control_images").mkdir(parents=True)
    caption(write_image(str(folder / "a.png")), "edit a")
    caption(write_image(str(folder / "b.png")), "edit b")
    write_image(str(folder / "control_images" / "a.png"))
    (folder / "c.txt").write_bytes("caf\xe9".encode("cp1252"))
    write_image(str(folder / "c.png"))
    jpeg = folder / "d.jpg"
    write_image(str(jpeg), size=(256, 256))
    jpeg.write_bytes(jpeg.read_bytes()[:600])  # cut off
    caption(str(jpeg), "cut")
    report = inspect_dataset(str(tmp_path), profile="image", control_name="control_images", control_required=False)
    errors = "\n".join(text for level, text, _examples in report.problems if level == "error")
    assert "have no control image" in errors and "not UTF-8" in errors and "cannot be read" in errors
    videos = tmp_path / "videos" / "1_clips"
    videos.mkdir(parents=True)
    write_video(str(videos / "v.mp4"), frames=9)
    report = inspect_dataset(str(tmp_path / "videos"), profile="video")
    assert any(level == "error" and "video / audio file(s) have no" in text for level, text, _examples in report.problems)
    report = inspect_dataset(str(tmp_path / "videos"), profile="video", create_missing_captions=True)
    assert not any(level == "error" for level, _text, _examples in report.problems)


def test_exif_rotation_is_reported_as_stored(tmp_path):
    from PIL import Image

    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset

    folder = tmp_path / "1_photos"
    folder.mkdir()
    image = Image.new("RGB", (400, 200), (90, 120, 200))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 when shown
    image.save(folder / "p.jpg", exif=exif)
    caption(str(folder / "p.jpg"), "a photo")
    report = inspect_dataset(str(tmp_path), profile="image")
    assert report.items[0].size == (400, 200) and report.items[0].rotated
    assert any("EXIF rotation" in text for _level, text, _examples in report.problems)


def test_toml_datasets_count_only_their_own_media(tmp_path):
    """A mixed folder in the app's own TOML: the video dataset reads the videos, the image dataset the images."""
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset, steps_per_epoch

    folder = tmp_path / "1_mixed"
    folder.mkdir()
    caption(write_video(str(folder / "v.mp4"), frames=9), "a clip")
    caption(write_image(str(folder / "i.png")), "a still")
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        toml.dumps({"general": {"resolution": [512, 512], "caption_extension": ".txt"},
                    "datasets": [{"video_directory": str(folder), "num_repeats": 2}, {"image_directory": str(folder), "num_repeats": 3}]}),
        encoding="utf-8",
    )
    report = inspect_dataset("", profile="video", dataset_toml=str(dataset), resolution=(1328, 1328))
    assert report.resolution == (512, 512)  # the TOML's own resolution, not the hidden UI value
    assert [(item.kind, item.used) for item in report.items if item.used] == [("image", True), ("video", True)] or sorted(
        item.kind for item in report.items if item.used
    ) == ["image", "video"]
    assert steps_per_epoch(report) == 2 + 3


def test_kohya_folders_follow_sd_scripts(tmp_path):
    """sd-scripts: the number before the first "_" (also "10" and "5_"); captions fall back to the class name;
    regularization images double the steps of an epoch."""
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset, steps_per_epoch

    for name in ("10", "5_", "2_ohwx man"):
        (tmp_path / "train" / name).mkdir(parents=True)
        write_image(str(tmp_path / "train" / name / "a.png"))
    (tmp_path / "reg" / "1_man").mkdir(parents=True)
    write_image(str(tmp_path / "reg" / "1_man" / "r.png"))
    report = inspect_dataset(str(tmp_path / "train"), profile="image", control_name="", kohya=True, regularization_folder=str(tmp_path / "reg"))
    assert {folder.name: folder.repeats for folder in report.folders if not folder.skipped} == {"10": 10, "5_": 5, "2_ohwx man": 2, "reg: 1_man": 1}
    assert not any(level in ("error", "warn") for level, _text, _examples in report.problems)
    assert steps_per_epoch(report) == (10 + 5 + 2) * 2


def test_unreadable_toml_values_never_crash_the_check(tmp_path):
    from musubi_tuner_gui.unified.dataset_inspector import inspect_dataset

    folder = tmp_path / "imgs"
    folder.mkdir()
    caption(write_image(str(folder / "a.png")), "a")
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(toml.dumps({"datasets": [{"image_directory": str(folder), "num_repeats": "3x"}]}), encoding="utf-8")
    report = inspect_dataset("", dataset_toml=str(dataset))
    assert any("num_repeats" in text for _level, text, _examples in report.problems) and report.folders[0].repeats == 1
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert inspect_dataset("", dataset_toml=str(tmp_path / "list.json")).error
