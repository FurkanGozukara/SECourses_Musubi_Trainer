"""Folders holding both videos and images: a video dataset reads only the videos of its folder."""

from musubi_tuner_gui.dataset_config_generator import generate_ltx2_dataset_config_from_folders, generate_wan_dataset_config_from_folders


def _mixed_dataset(tmp_path):
    folder = tmp_path / "dataset" / "2_ohwx"
    folder.mkdir(parents=True)
    for name in ("clip.mp4", "still.png"):
        (folder / name).write_bytes(b"")
        (folder / name).with_suffix(".txt").write_text("ohwx", encoding="utf-8")
    return tmp_path / "dataset"


def test_wan_keeps_the_images_of_a_video_folder_in_their_own_dataset(tmp_path):
    config, messages = generate_wan_dataset_config_from_folders(str(_mixed_dataset(tmp_path)), (512, 512), create_missing_captions=False)
    entries = config["datasets"]
    assert [("video_directory" in entry, "image_directory" in entry) for entry in entries] == [(True, False), (False, True)]
    assert entries[0]["num_repeats"] == entries[1]["num_repeats"] == 2
    assert entries[0]["cache_directory"] != entries[1]["cache_directory"]
    assert any("separate image dataset" in message for message in messages)


def test_ltx_says_that_images_of_a_video_folder_are_not_used(tmp_path):
    config, messages = generate_ltx2_dataset_config_from_folders(str(_mixed_dataset(tmp_path)), (512, 512), create_missing_captions=False)
    assert len(config["datasets"]) == 1 and "video_directory" in config["datasets"][0]
    assert any("images are not used" in message for message in messages)


def test_folders_the_trainer_creates_do_not_skip_the_dataset_on_the_next_run(tmp_path):
    """After the first caching, <cache> and <cache>_images sit inside the dataset folder."""
    parent = _mixed_dataset(tmp_path)
    for name in ("cache_dir", "cache_dir_images"):
        (parent / "2_ohwx" / name).mkdir()
    config, messages = generate_wan_dataset_config_from_folders(str(parent), (512, 512), create_missing_captions=False)
    assert len(config["datasets"]) == 2, messages


def test_a_generated_toml_is_reused_only_for_its_folder_and_resolution(tmp_path):
    """Start reused the last generated TOML whenever it existed, so a new folder or resolution trained the old one."""
    import toml

    from musubi_tuner_gui.dataset_config_generator import dataset_toml_matches

    first, second = tmp_path / "first", tmp_path / "second"
    (first / "1_a").mkdir(parents=True)
    (second / "1_b").mkdir(parents=True)
    generated = tmp_path / "dataset.toml"
    generated.write_text(toml.dumps({"general": {"resolution": [1024, 1024]}, "datasets": [{"image_directory": str(first / "1_a")}]}), encoding="utf-8")
    assert dataset_toml_matches(str(generated), str(first), (1024, 1024))
    assert not dataset_toml_matches(str(generated), str(second), (1024, 1024))  # another folder
    assert not dataset_toml_matches(str(generated), str(first), (768, 768))  # another resolution
    assert not dataset_toml_matches(str(tmp_path / "missing.toml"), str(first), (1024, 1024))
    # datasets written by hand (e.g. audio) inside the folder are still reused
    generated.write_text(toml.dumps({"datasets": [{"audio_directory": str(first / "1_a")}]}), encoding="utf-8")
    assert dataset_toml_matches(str(generated), str(first), (1024, 1024))
