"""Best-checkpoint finder: test prompts, sample / checkpoint indexing, the ranking rule, the unified UI's run preparation
and the runner's after-step (no scoring models are loaded)."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from musubi_tuner_gui.best_checkpoint import prompts, ranking  # noqa: E402
from musubi_tuner_gui.best_checkpoint.samples import checkpoint_for, checkpoint_for_epoch, index_samples, point_kind  # noqa: E402
from musubi_tuner_gui.unified import checkpoint_finder as CF  # noqa: E402
from musubi_tuner_gui.unified.dashboard import render_dashboard  # noqa: E402
from musubi_tuner_gui.unified.runner import AfterStep, TrainingRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _own_ui_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path / "user_presets"))


def test_default_prompts_follow_the_subject_and_carry_their_seed(tmp_path):
    (tmp_path / "10_sks woman").mkdir()
    assert prompts.suggest_subject(str(tmp_path)) == "sks woman"
    assert prompts.suggest_subject("") == prompts.DEFAULT_SUBJECT
    (other := tmp_path / "other").mkdir()
    (other / "1_ohwx").mkdir()
    assert prompts.suggest_subject(str(other)) == "ohwx man"  # the folder names only the trigger word
    changed = prompts.apply_subject(prompts.DEFAULT_PROMPTS, "sks woman")
    assert all("ohwx man" not in p.lower() for p in changed) and any("sks woman" in p for p in changed)
    assert prompts.class_word("sks woman") == "woman"
    assert prompts.prompt_file_lines(["a", "b"], [1, 2]) == ["a --d 1", "a --d 2", "b --d 1", "b --d 2"]
    assert prompts.clean_prompts([["x <lora:y:1>"], [""], ["# note"], ["x"], ["y\x07"]]) == ["x", "y"]
    assert prompts.clean_prompts("one\n\ntwo") == ["one", "two"]
    assert prompts.prompt_text("a photo of ohwx man --w 1024 --h 1024 --d 42") == "a photo of ohwx man"
    assert "woman" in prompts.leakage_prompt("sks woman") and "sks" not in prompts.leakage_prompt("sks woman")


def test_samples_are_grouped_by_checkpoint_and_mapped_to_their_files(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    for name in (
        "run_e000000_00_20261012120000_1001_000.png",
        "run_e000020_01_20261012130000_1002_000.png",
        "run_e000020_00_20261012130000_1001_000.png",
        "run_e000040_00_20261012140000_1001.mp4",
        "other_e000020_00_20261012130000_1_000.png",
        "run_v2_e000020_00_20261012130000_1_000.png",  # another run whose name starts with this one's
        "run_e000020_00_20261012130000_1001_000.txt",
    ):
        (sample / name).touch()
    (tmp_path / "run-000020.safetensors").touch()
    (tmp_path / "run.safetensors").touch()
    found = index_samples(str(tmp_path), "run")
    assert sorted(found) == [0, 20, 40]
    assert [(s["prompt_index"], s["seed"]) for s in found[20]] == [(0, "1001"), (1, "1002")]
    assert found[40][0]["video"] and point_kind(found) == "epoch"
    assert checkpoint_for_epoch(str(tmp_path), "run", 20).endswith("run-000020.safetensors")
    assert checkpoint_for_epoch(str(tmp_path), "run", 40).endswith("run.safetensors")  # the last epoch
    assert checkpoint_for_epoch(str(tmp_path), "run", 10) is None
    assert checkpoint_for(str(tmp_path), "run", 0) is None  # the base model has no checkpoint


def test_step_samples_map_to_step_checkpoints(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    for name in ("lora_e000000_00_20261012120000_7.png", "lora_000500_00_20261012130000_7.png", "lora_001000_00_20261012140000_7.png"):
        (sample / name).touch()
    (tmp_path / "lora-step00000500.safetensors").touch()
    (tmp_path / "lora.safetensors").touch()
    found = index_samples(str(tmp_path), "lora")
    assert sorted(found) == [0, 500, 1000] and point_kind(found) == "step"
    assert checkpoint_for(str(tmp_path), "lora", 500, "step").endswith("lora-step00000500.safetensors")
    assert checkpoint_for(str(tmp_path), "lora", 1000, "step").endswith("lora.safetensors")


def _checkpoint(face, prompt, variety, saturation, nearest, found=1.0):
    return {"summary": {"face_mean": face, "dino_nearest_train": nearest, "prompt": prompt, "saturation": saturation,
                        "colourfulness": 40 + saturation * 10, "clipped": 0.01, "hf_energy": 100 + saturation * 50,
                        "face_found": found, "variety": variety}}


def test_the_rule_picks_the_plateau_and_explains_the_overcooked_end():
    report = {"training_faces": 10, "checkpoints": {
        0: _checkpoint(0.10, 0.20, 0.50, 0.40, 0.60),
        20: _checkpoint(0.35, 0.20, 0.49, 0.41, 0.70),
        40: _checkpoint(0.55, 0.199, 0.48, 0.42, 0.75),
        60: _checkpoint(0.62, 0.198, 0.47, 0.43, 0.80),
        80: _checkpoint(0.64, 0.195, 0.45, 0.45, 0.84),
        100: _checkpoint(0.65, 0.17, 0.37, 0.60, 0.95, found=0.6),
    }}
    result = ranking.rank(report)
    assert result["best_epoch"] == 60 and result["plateau_start"] == 60
    flags = next(row for row in result["rows"] if row["epoch"] == 100)["flags"]
    assert any("prompt following" in f for f in flags) and any("variety" in f for f in flags)
    assert any("copies training photos" in f for f in flags)
    assert any("from epoch 100" in reason for reason in result["reasons"])
    assert ranking.rank({"training_faces": 0, "checkpoints": {0: report["checkpoints"][0]}}).get("error")


def test_a_short_run_is_called_undertrained_and_tiny_drifts_are_not_flagged():
    # the scores of a real 52-step Krea 2 run (2026-10-12): likeness had not moved, colours shifted a few percent
    def point(face, saturation, colour, hf, clipped):
        return {"summary": {"face_mean": face, "prompt": 0.18, "variety": 0.96, "saturation": saturation, "colourfulness": colour,
                            "hf_energy": hf, "clipped": clipped, "face_found": 1.0, "dino_nearest_train": 0.22}}

    report = {"training_faces": 12, "checkpoints": {
        0: point(0.085, 0.311, 38.8, 316, 0.390), 1: point(0.081, 0.326, 39.1, 332, 0.374), 2: point(0.087, 0.329, 39.9, 317, 0.371),
        3: point(0.073, 0.332, 40.4, 325, 0.375), 4: point(0.111, 0.333, 40.3, 337, 0.371),
    }}
    result = ranking.rank(report)
    assert result["undertrained"] and result["best_epoch"] == 4
    assert "undertrained" in result["reasons"][0] and all(not row["flags"] for row in result["rows"])


def test_without_faces_the_rule_uses_image_similarity():
    report = {"training_faces": 0, "checkpoints": {
        0: {"summary": {"dino_nearest_train": 0.30, "prompt": 0.2, "variety": 0.5}},
        20: {"summary": {"dino_nearest_train": 0.60, "prompt": 0.2, "variety": 0.5}},
        40: {"summary": {"dino_nearest_train": 0.62, "prompt": 0.2, "variety": 0.5}},
    }}
    result = ranking.rank(report)
    assert result["likeness_signal"] == "dino_nearest_train" and result["best_epoch"] in (20, 40)


def test_paired_changes_ignore_the_prompt_mix_and_flag_real_damage():
    from musubi_tuner_gui.best_checkpoint import scoring

    def samples(sharpness, prompts):
        return [{"prompt_index": i, "seed": "1", "hf_energy": h, "prompt": p} for i, (h, p) in enumerate(zip(sharpness, prompts))]

    # sharpness differs 25x between prompts: the median over a checkpoint's samples moved -41% although no prompt changed
    base = samples([20, 60, 230, 500, 650], [0.15, 0.20, 0.22, 0.25, 0.28])
    same = samples([25, 66, 200, 520, 700], [0.152, 0.198, 0.221, 0.251, 0.279])
    change, agree = scoring.paired_change(same, base, "hf_energy", ratio=True)
    assert abs(change) < 0.15 and agree >= 0.6
    worse = samples([25, 66, 200, 520, 700], [0.13, 0.18, 0.20, 0.23, 0.26])
    assert scoring.paired_change(worse, base, "prompt") == pytest.approx((-0.02, 1.0))
    assert scoring.paired_change(worse, [], "prompt") is None

    def point(face, vs_base):
        return {"summary": {"face_mean": face, "prompt": 0.21, "variety": 0.9, "hf_energy": 120.0, "face_found": 1.0,
                            "dino_nearest_train": 0.2, "vs_base": vs_base}}

    calm = {"prompt": (0.001, 0.5), "hf_energy": (0.05, 0.6), "saturation": (0.02, 0.6)}
    report = {"training_faces": 13, "checkpoints": {
        0: {"summary": {"face_mean": 0.09, "prompt": 0.21, "variety": 0.9, "hf_energy": 230.0, "face_found": 1.0}},
        20: point(0.45, calm), 40: point(0.55, calm), 60: point(0.58, calm),
        80: point(0.60, {"prompt": (-0.02, 0.9), "hf_energy": (0.6, 0.8), "saturation": (0.3, 0.9)}),
    }}
    result = ranking.rank(report)
    rows = {row["epoch"]: row for row in result["rows"]}
    assert not rows[20]["flags"] and not rows[60]["flags"]  # the unpaired medians would call every checkpoint 48% less sharp
    assert any("prompt following" in f for f in rows[80]["flags"]) and any("saturation drift (+35%" in f for f in rows[80]["flags"])
    assert result["best_epoch"] == 60 and rows[80]["prompt"] == pytest.approx(0.19)


def test_long_prompts_are_scored_in_sentence_groups_that_fit_the_text_encoder(tmp_path):
    from musubi_tuner_gui.best_checkpoint import scoring

    class Words:  # one token per word
        def __call__(self, text):
            return {"input_ids": text.split()}

    text = "One two three four. Five six seven. Eight nine ten eleven twelve! Thirteen"
    assert scoring.prompt_chunks(text, Words(), limit=7) == ["One two three four. Five six seven.", "Eight nine ten eleven twelve! Thirteen"]
    assert scoring.prompt_chunks(text, Words(), limit=3, most=2) == ["One two three four.", "Five six seven."]
    assert scoring.prompt_chunks("a photo", Words()) == ["a photo"]
    cache = str(tmp_path / "scores_cache.json")
    scoring.save_cache(cache, {"a.png|1|2": {"prompt": 0.2}})
    assert scoring.load_cache(cache) == {"a.png|1|2": {"prompt": 0.2}}
    with open(cache, "w", encoding="utf-8") as handle:
        json.dump({"a.png|1|2": {"prompt": 0.2}}, handle)  # the first format (no version): scored again
    assert scoring.load_cache(cache) == {}


def test_a_checkpoint_still_being_sampled_is_left_out(tmp_path, monkeypatch):
    from musubi_tuner_gui.best_checkpoint import scoring, worker

    sample = tmp_path / "sample"
    sample.mkdir()
    for point in (0, 20, 40):
        for index in range(3):
            (sample / f"run_e{point:06d}_{index:02d}_20261012120000_1001_000.png").touch()
    (sample / "run_e000060_00_20261012130000_1001_000.png").touch()  # Analyze while epoch 60 is being sampled
    for point in (20, 40, 60):
        (tmp_path / f"run-{point:06d}.safetensors").touch()
    (tmp_path / "train.png").touch()
    likeness = {0: 0.1, 20: 0.5, 40: 0.55, 60: 0.9}

    def fake_scores(scorer, items, *args):
        point = int(os.path.basename(items[0]["path"]).split("_")[1][1:])
        return [{**item, "face_mean": likeness[point], "prompt": 0.2, "dino_nearest_train": 0.5} for item in items]

    monkeypatch.setattr(scoring, "Scorer", lambda device: None)
    monkeypatch.setattr(scoring, "reference_set", lambda scorer, paths: {"count": 1, "faces": [0]})
    monkeypatch.setattr(scoring, "score_samples", fake_scores)
    settings = {"prompts": ["a", "b", "c"], "references": [str(tmp_path / "train.png")]}
    report = worker.analyze(str(tmp_path), "run", settings, device="cpu", log=lambda *a: None)
    assert sorted(report["checkpoints"]) == [0, 20, 40] and report["result"]["best_epoch"] == 40
    assert "Epoch 60 was left out: 1 of its 3 samples were found" in report["result"]["reasons"][-1]


# ------------------------------------------------------------------ unified UI
def test_settings_survive_ui_state_and_a_preset_header(tmp_path):
    settings = CF.normalize({"enabled": True, "subject": "  sks   woman ", "prompts": "a \"quoted\" prompt\nünïcode 🙂 prompt\n", "seeds": "9"}, "image")
    assert settings["subject"] == "sks woman" and settings["seeds"] == CF.MAX_SEEDS and len(settings["prompts"]) == 2
    CF.remember("image", settings)
    assert CF.remembered("image") == settings
    assert CF.remembered("video") == CF.defaults("video")  # per category
    from musubi_tuner_gui.unified import presets as P
    from musubi_tuner_gui.unified.registry import get_entry

    path = tmp_path / "preset.toml"
    path.write_text('learning_rate = 1e-4\n[extra]\nx = 1\n', encoding="utf-8")
    P.write_metadata_header(str(path), get_entry("krea2"), "lora", extra={CF.HEADER_KEY: CF.header_value(settings)})
    import toml

    assert toml.load(str(path))["learning_rate"] == 1e-4  # still a valid TOML file
    assert CF.from_metadata(P.read_metadata(str(path)), "image") == settings
    assert CF.from_metadata({}, "image") is None


def test_start_samples_the_test_prompts_at_every_save(tmp_path):
    references = [str(tmp_path / "data" / "a.png")]
    own = tmp_path / "own_prompts.txt"
    own.write_text("# mine\nmy own prompt --w 512\n", encoding="utf-8")
    values = {"output_dir": str(tmp_path / "out"), "output_name": "lora", "save_every_n_epochs": 10, "save_every_n_steps": 0,
              "sample_prompts": str(own), "sample_every_n_epochs": 0, "sample_every_n_steps": 100, "sample_at_first": False, "gpu_ids": "1"}
    settings = CF.normalize({"enabled": True, "subject": "sks woman", "prompts": ["photo of sks woman", "sks woman hiking"], "seeds": 2, "leakage": True}, "image")
    run, after = CF.prepare_run(values, settings, references)
    assert values["sample_every_n_steps"] == 100  # the user's values are not changed
    assert run["sample_every_n_epochs"] == 10 and run["sample_every_n_steps"] == 0 and run["sample_at_first"] is True
    lines = open(run["sample_prompts"], encoding="utf-8").read().splitlines()
    assert lines[:2] == ["photo of sks woman --d 1001", "photo of sks woman --d 1002"]
    assert "woman" in lines[4] and "sks" not in lines[4] and lines[-1] == "my own prompt --w 512"
    finder = json.load(open(os.path.join(CF.run_folder(values["output_dir"], "lora"), "settings.json"), encoding="utf-8"))
    assert finder["lines"] == 6 and finder["leakage_indices"] == [4, 5] and len(finder["prompts"]) == 6 and finder["references"] == references
    assert after.env["CUDA_VISIBLE_DEVICES"] == "1" and "musubi_tuner_gui.best_checkpoint.worker" in after.cmd
    with pytest.raises(CF.FinderError):
        CF.prepare_run({**values, "save_every_n_epochs": 0}, settings, references)
    with pytest.raises(CF.FinderError):
        CF.prepare_run(values, {**settings, "prompts": []}, references)


def test_results_render_from_a_report(tmp_path):
    out = tmp_path / "out"
    (out / "sample").mkdir(parents=True)
    from PIL import Image

    checkpoints = {}
    for point in (0, 10, 20):
        path = out / "sample" / f"lora_e{point:06d}_00_20261012120000_1001_000.png"
        Image.new("RGB", (64, 48), (point * 5, 80, 120)).save(path)
        checkpoints[point] = {"summary": {"face_mean": 0.1 + point / 40, "prompt": 0.2, "variety": 0.5},
                              "samples": [{"path": str(path), "prompt_index": 0, "seed": "1001"}]}
    report = {"output_dir": str(out), "output_name": "lora", "kind": "epoch", "training_images": 3, "training_faces": 3,
              "checkpoints": checkpoints, "settings": {"prompts": ["photo of sks woman"]}}
    report["result"] = ranking.rank(report)
    report["result"].update(checkpoint=str(out / "lora.safetensors"), label="epoch 20")
    folder = CF.run_folder(str(out), "lora")
    os.makedirs(folder)
    json.dump(report, open(os.path.join(folder, "report.json"), "w", encoding="utf-8"))
    loaded = CF.load_report(str(out), "lora")
    page = CF.results_html(loaded, str(out), "lora")
    assert "Best checkpoint: epoch 20" in page and "bcf-best" in page and "base model" in page
    items = CF.comparison_gallery(loaded)
    assert len(items) == 2 and all(os.path.isfile(path) for path, _caption in items)  # base and best (= last)
    summary = CF.result_summary(str(out), "lora")
    assert summary["headline"] == "Best checkpoint: epoch 20"
    assert "No finder report" in CF.results_html(None, str(out), "other")


# ------------------------------------------------------------------ runner after-step
def _python(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", code]


def _wait(runner: TrainingRunner, seconds: float = 30.0) -> None:
    deadline = time.time() + seconds
    while runner.is_active() and time.time() < deadline:
        time.sleep(0.05)


def test_the_after_step_runs_once_training_succeeded():
    runner = TrainingRunner()
    after = AfterStep("Finding the best checkpoint", _python("print('scoring', flush=True)"),
                      read_result=lambda: {"kind": "best_checkpoint", "headline": "Best checkpoint: epoch 2", "checkpoint": "x.safetensors", "reasons": ["why"]})
    started, _ = runner.start_job(lambda: runner.execute_command(_python("print('training', flush=True)")), title="t", meta={}, after=after)
    assert started
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "finished" and "Best checkpoint: epoch 2" in snapshot["message"]
    assert [stage["id"] for stage in snapshot["stages"]][-1] == "score" and snapshot["stages"][-1]["state"] == "done"
    assert snapshot["result"]["headline"] == "Best checkpoint: epoch 2"
    assert "ud-best" in render_dashboard(snapshot) and "scoring" in runner.log_tail(50)


def test_a_failed_after_step_never_fails_the_training_and_stop_ends_it():
    runner = TrainingRunner()
    after = AfterStep("Finding the best checkpoint", _python("print('BEST_CHECKPOINT_ERROR: no samples', flush=True); raise SystemExit(1)"))
    runner.start_job(lambda: runner.execute_command(_python("print('training')")), title="t", meta={}, after=after)
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "finished" and snapshot["stages"][-1]["state"] == "failed" and "no samples" in snapshot["message"]

    slow = AfterStep("Finding the best checkpoint", _python("import time\nwhile True: time.sleep(0.1)"))
    runner.start_job(lambda: runner.execute_command(_python("print('training')")), title="t", meta={}, after=slow)
    deadline = time.time() + 20
    while runner.snapshot()["current_stage"] != "score" and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    assert runner.is_active()  # the run stays active while the after-step runs
    runner.stop()
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "finished" and snapshot["stages"][-1]["state"] == "stopped" and not runner.is_active()


def test_a_failed_training_skips_the_after_step_and_runs_without_one_hide_it():
    runner = TrainingRunner()
    after = AfterStep("Finding the best checkpoint", _python("print('never')"))
    runner.start_job(lambda: runner.execute_command(_python("raise SystemExit(3)")), title="t", meta={}, after=after)
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "failed" and "never" not in runner.log_tail(50)
    runner.start_job(lambda: runner.execute_command(_python("print('training')")), title="t", meta={})
    _wait(runner)
    assert "score" not in [stage["id"] for stage in runner.snapshot()["stages"]] and runner.snapshot()["result"] is None


def test_an_after_step_can_run_as_its_own_job():
    runner = TrainingRunner()
    after = AfterStep("Finding the best checkpoint", _python("print('scoring')"), read_result=lambda: {"kind": "best_checkpoint", "headline": "Best checkpoint: step 500"})
    started, _ = runner.start_after_step(after, title="analyze", meta={})
    assert started
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "finished" and snapshot["message"] == "Best checkpoint: step 500"
