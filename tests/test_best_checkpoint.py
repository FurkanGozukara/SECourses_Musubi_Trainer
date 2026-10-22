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


def test_default_prompts_take_the_subjects_pronouns_or_the_object_set():
    import re

    man = prompts.default_prompts("sks man")
    assert man == prompts.apply_subject(prompts.DEFAULT_PROMPTS, "sks man") and prompts.default_prompts() == prompts.DEFAULT_PROMPTS
    assert prompts.default_prompts("ohwx") == prompts.apply_subject(prompts.DEFAULT_PROMPTS, "ohwx")  # no class word: as written
    he = re.compile(r"\b(?:[Hh]e|[Hh]is|[Hh]im|[Hh]imself)\b")
    woman = prompts.default_prompts("sks woman")
    assert not any(he.search(p) for p in woman) and all("ohwx man" not in p for p in woman)
    assert "She is wearing a traditional black suit" in woman[0] and "sks woman, a woman in her early 40s" in woman[4]
    person = prompts.default_prompts("sks person")
    assert not any(he.search(p) for p in person)
    assert not any(re.search(r"\b[Tt]hey (?:is|has|\w+[bcdfgklmnprtvwxz]s)\b", p) for p in person)
    assert "They are wearing a traditional black suit" in person[0] and "which they hold under their arm" in person[4]
    assert "a person in their early 40s" in person[4] and "They wear a pair of durable" in person[4]
    dog = prompts.default_prompts("sks dog")
    assert dog[0].startswith("A photo of sks dog on a wooden kitchen counter") and dog[1].startswith("sks dog on a sandy beach")
    assert len(dog) == len(prompts.OBJECT_PROMPTS) and not any("eyeglasses" in p or he.search(p) for p in dog)
    assert prompts.leakage_prompt("sks dog") == "A photo of a dog on a park bench, soft afternoon light."
    assert prompts.leakage_prompt("sks owl") == "A photo of an owl on a park bench, soft afternoon light."
    assert prompts.leakage_prompt("sks girl").startswith("A candid photograph of a girl sitting at a small table")
    assert [prompts.subject_kind(s) for s in ("ohwx man", "ohwx", "sks Woman", "sks child", "sks toy")] == ["male", "male", "female", "person", "object"]
    assert CF.default_prompts("video", "sks dog") == [dog[i] for i in CF.VIDEO_PROMPTS]
    assert CF.defaults("image")["prompts"] == prompts.DEFAULT_PROMPTS
    # a subject change switches prompts that are still the defaults; edited ones only get the new subject
    text = CF.prompts_text({"prompts": CF.default_prompts("image")})
    assert CF.prompts_for_subject(text, "ohwx man", "sks woman", "image") == CF.prompts_text({"prompts": woman})
    assert CF.prompts_for_subject(CF.prompts_text({"prompts": woman}), "sks woman", "sks dog", "image") == CF.prompts_text({"prompts": dog})
    edited = text.replace("burgundy tie", "green tie")
    assert CF.prompts_for_subject(edited, "ohwx man", "sks woman", "image") == CF.replace_subject(edited, "ohwx man", "sks woman")


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


def _checkpoint(face, prompt, variety, saturation, nearest, found=1.0, copied=None):
    """``nearest``: the median sample's similarity to the nearest training image; ``copied``: the highest sample's."""
    return {"summary": {"face_mean": face, "dino_nearest_train": nearest, "dino_nearest_max": copied, "prompt": prompt, "saturation": saturation,
                        "colourfulness": 40 + saturation * 10, "clipped": 0.01, "hf_energy": 100 + saturation * 50,
                        "face_found": found, "variety": variety}}


def test_the_rule_picks_the_plateau_and_explains_the_overcooked_end():
    report = {"training_faces": 10, "checkpoints": {
        0: _checkpoint(0.10, 0.20, 0.50, 0.40, 0.60),
        20: _checkpoint(0.35, 0.20, 0.49, 0.41, 0.70),
        40: _checkpoint(0.55, 0.199, 0.48, 0.42, 0.75),
        60: _checkpoint(0.62, 0.198, 0.47, 0.43, 0.80),
        80: _checkpoint(0.64, 0.195, 0.45, 0.45, 0.84),
        100: _checkpoint(0.65, 0.17, 0.37, 0.60, 0.95, found=0.6, copied=0.97),
    }}
    result = ranking.rank(report)
    # likeness levels off from epoch 60; epoch 100 shows the over-training signs: the most likeness before them
    assert result["best_epoch"] == 80 and result["plateau_start"] == 60 and result["onset"] == 100
    flags = next(row for row in result["rows"] if row["epoch"] == 100)["flags"]
    assert any("prompt following" in f for f in flags) and any("variety" in f for f in flags)
    assert any("a sample copies a training image" in f for f in flags)
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


def test_face_signs_need_faces_in_the_training_photos():
    # FLUX.2 Klein 4B on five photos of a corgi (2026-10-17 object study): the face detector found dog faces and stray
    # people, read as "drawn closer than the prompts ask (faces 5.51x the base model's size)" and "faces found in 17%"
    def point(nearest):
        return {"summary": {"dino_nearest_train": nearest, "prompt": 0.2, "variety": 0.7, "face_found": 0.05,
                            "vs_base": {"face_area": (1.7, 1.0)}}}

    base = {"summary": {"dino_nearest_train": 0.05, "prompt": 0.2, "variety": 0.7, "face_found": 0.33, "face_area": 0.01}}
    checkpoints = {0: base, 40: point(0.42), 80: point(0.65), 120: point(0.70)}

    def flags(report):
        return [flag for row in ranking.rank(report)["rows"] for flag in row["flags"]]

    pet = flags({"training_faces": 0, "checkpoints": checkpoints})
    assert not any("drawn closer" in flag or "faces found" in flag for flag in pet)
    person = flags({"training_faces": 5, "training_face_area": 0.2, "checkpoints": checkpoints})
    assert any("drawn closer" in flag for flag in person) and any("faces found" in flag for flag in person)


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
        80: point(0.60, {"prompt": (-0.02, 0.9), "hf_energy": (0.6, 0.8), "saturation": (0.3, 0.9), "face_area": (0.75, 1.0)}),
    }}
    result = ranking.rank(report)
    rows = {row["epoch"]: row for row in result["rows"]}
    assert not rows[20]["flags"] and not rows[60]["flags"]  # the unpaired medians would call every checkpoint 48% less sharp
    assert any("prompt following" in f for f in rows[80]["flags"]) and any("saturation drift (+32% since epoch 20)" in f for f in rows[80]["flags"])
    assert any("drawn closer than the prompts ask, as in the training photos (faces 2.12x" in f for f in rows[80]["flags"])
    assert result["best_epoch"] == 60 and rows[80]["prompt"] == pytest.approx(0.19)


def test_the_pick_is_the_most_likeness_before_the_training_photos_framing_takes_over():
    # Krea 2 LoRA at its preset learning rate, 200 epochs (2026-10-12 research): paired face match gain and face size over
    # the base model per save. Looked at by eye: likeness full by epoch 60-80, the news anchor drawn closer from epoch 100-140
    gains = [0.407, 0.492, 0.544, 0.565, 0.571, 0.511, 0.518, 0.546, 0.628, 0.635]
    areas = [-0.005, 0.078, 0.130, 0.160, 0.224, 0.162, 0.278, 0.270, 0.275, 0.278]
    agree = [0.64, 0.55, 0.73, 0.82, 0.91, 0.91, 0.91, 1.0, 1.0, 0.82]
    checkpoints = {0: {"summary": {"face_mean": 0.089, "prompt": 0.1345, "variety": 0.909, "face_found": 1.0, "face_area": 0.0168}}}
    for index, (gain, area, share) in enumerate(zip(gains, areas, agree)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.089 + gain, "prompt": 0.1345, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "face_area": (area, share), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 12, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert (result["plateau_start"], result["onset"], result["best_epoch"]) == (80, 140, 80)
    assert "Over-training signs from epoch 140: drawn closer than the prompts ask" in result["reasons"][1]
    # a dataset framed wider than the prompts (full-body photos): faces drifting smaller are the training framing
    result = ranking.rank({"training_faces": 12, "training_face_area": 0.004, "checkpoints": checkpoints})
    assert result["onset"] is None and result["best_epoch"] == 180  # within PICK_TOLERANCE of epoch 200, earlier


def test_the_likeness_level_waits_until_the_faces_leave_the_base_models_identity():
    # Krea 2 DoRA at half its learning rate (2026-10-12 research): until epoch 80 the headshots kept the base model's own
    # face with the subject's glasses and hair, which SFace already rates 0.55-0.65 against the training faces; from
    # epoch 100 they are the subject's. The identity margin (training faces minus the base model's face of the same
    # prompt and seed) jumps there; the face match alone levels off at epoch 60.
    gains = [0.188, 0.369, 0.46, 0.485, 0.546, 0.477, 0.524, 0.487, 0.515, 0.518]
    margins = [-0.195, 0.073, 0.174, 0.202, 0.327, 0.353, 0.330, 0.350, 0.350, 0.364]
    areas = [0.039, 0.022, 0.094, 0.135, 0.116, 0.135, 0.191, 0.191, 0.172, 0.255]

    def report(with_identity: bool) -> dict:
        checkpoints = {0: {"summary": {"face_mean": 0.091, "prompt": 0.134, "variety": 0.9, "face_found": 1.0, "face_area": 0.0169}}}
        for index, (gain, margin, area) in enumerate(zip(gains, margins, areas)):
            vs_base = {"face_mean": (gain, 1.0), "face_area": (area, 0.82), "prompt": (0.0, 0.5)}
            if with_identity:
                vs_base["identity"] = (margin, 0.9)
            checkpoints[20 * (index + 1)] = {"summary": {"face_mean": 0.091 + gain, "prompt": 0.134, "variety": 0.9, "face_found": 1.0,
                                                         "dino_nearest_train": 0.2, "vs_base": vs_base}}
        return {"training_faces": 12, "training_face_area": 0.113, "checkpoints": checkpoints}

    result = ranking.rank(report(True))
    assert (result["plateau_start"], result["best_epoch"]) == (120, 120)  # by eye: e100-e160, best e120
    assert ranking.rank(report(False))["plateau_start"] == 60

    from musubi_tuner_gui.best_checkpoint import scoring
    import numpy as np

    subject, base_face = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    base = [{"prompt_index": 0, "seed": "1", "face_embedding": base_face.tolist()}, {"prompt_index": 1, "seed": "1", "face_embedding": None}]
    early = [{"prompt_index": 0, "seed": "1", "face_embedding": [0.6, 0.8]}, {"prompt_index": 1, "seed": "1", "face_embedding": [1.0, 0.0]}]
    assert scoring.identity_margin(early, base, subject) == pytest.approx((-0.2, 0.0))  # still nearer the base face
    late = [{"prompt_index": 0, "seed": "1", "face_embedding": [0.8, 0.6]}]
    assert scoring.identity_margin(late, base, subject) == pytest.approx((0.2, 1.0))
    assert scoring.identity_margin(late, base, None) is None


def test_a_warning_sign_that_does_not_last_does_not_hide_the_clean_checkpoints_after_it():
    # FLUX Klein 9B LoRA, 100 epochs (2026-10-13): faces x1.31 at the first save, x1.16 at the next, lasting signs from
    # epoch 60 on. By eye: the subject from epoch 20, the selfie loses its muscles from epoch 40-60: best epoch 40
    gains = [0.392, 0.456, 0.486, 0.500, 0.533]
    margins = [0.368, 0.400, 0.468, 0.428, 0.504]
    areas = [0.268, 0.152, 0.299, 0.250, 0.253]
    saturation = [-0.109, -0.159, -0.221, -0.292, -0.341]  # colours muted from epoch 60 on (-20%)
    checkpoints = {0: {"summary": {"face_mean": 0.287, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, area, muted) in enumerate(zip(gains, margins, areas, saturation)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.287 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "face_area": (area, 1.0), "prompt": (0.0, 0.5),
                        "saturation": (muted, 0.83)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert (result["plateau_start"], result["onset"], result["best_epoch"]) == (40, 60, 40)
    assert any("does not last: epoch 40 shows none" in reason for reason in result["reasons"])
    assert not any(reason.endswith(": .") or ": ." in reason for reason in result["reasons"])


def test_faces_that_never_leave_the_base_models_person_mean_an_undertrained_run():
    # Z-Image Turbo LoRA at its preset 1e-5 for 100 epochs (2026-10-13): face match 0.06 -> 0.33 (a gain of 0.27, more than
    # LIKENESS_MIN_GAIN) while every face stayed nearer the base model's man than the training photos; by eye: undertrained
    gains = [0.006, 0.077, 0.163, 0.248, 0.268]
    margins = [-0.565, -0.449, -0.295, -0.156, -0.168]
    checkpoints = {0: {"summary": {"face_mean": 0.059, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin) in enumerate(zip(gains, margins)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.059 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 0.1), "face_area": (0.02, 0.6), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert result["undertrained"] and result["best_epoch"] == 100
    assert "still the base model's own person at epoch 100" in result["reasons"][0]


def test_an_undertrained_run_with_damage_picks_the_last_checkpoint_before_it():
    # Mage-Flow DoRA (2026-10-13): the faces never left the base model's person, and from epoch 80 the textures turned
    # speckled (sharpness +52% / +67% since the first save): the latest checkpoint was the noisiest
    gains = [0.07, 0.15, 0.11, 0.13, 0.17]
    margins = [-0.16, 0.05, -0.07, 0.06, 0.05]
    sharpness = [0.46, 0.55, 0.60, 0.88, 0.98]
    checkpoints = {0: {"summary": {"face_mean": 0.116, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, sharp) in enumerate(zip(gains, margins, sharpness)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.116 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 0.5), "hf_energy": (sharp, 0.83), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert result["undertrained"] and result["onset"] == 80 and result["best_epoch"] == 60
    assert "the last checkpoint before the warning signs, epoch 60, is the pick" in result["reasons"][0]
    # the later checkpoint with more likeness is named for comparison (epoch 100: the most)
    assert any(r.startswith("Epoch 100 has more likeness (100% of the most, epoch 60") and r.endswith("compare the two in the grid.")
               for r in result["reasons"])


def test_an_undertrained_run_passes_over_a_sign_that_does_not_last():
    # HiDream-O1 LoRA (2026-10-13): still the base model's people at epoch 100; saturation dipped at epoch 80 only
    # (-24% since epoch 20, back at 100): the most trained clean checkpoint is the pick, not the one before the dip
    gains = [0.134, 0.225, 0.231, 0.239, 0.287]
    margins = [-0.284, -0.078, -0.078, -0.135, -0.046]
    saturation = [0.0, -0.05, -0.10, -0.28, -0.12]
    checkpoints = {0: {"summary": {"face_mean": 0.136, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, sat) in enumerate(zip(gains, margins, saturation)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.136 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "saturation": (sat, 1.0), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert result["undertrained"] and result["onset"] is None and result["best_epoch"] == 100
    assert "the latest checkpoint, epoch 100, is the pick" in result["reasons"][0]
    assert "not counted as over-training: saturation drift at epoch 80." in result["reasons"][1]
    row = next(r for r in result["rows"] if r["epoch"] == 80)
    assert row["flags"] and not row["signs"]  # shown, not counted
    # a sign that recurs at half of the later saves counts, and the clean last checkpoint is still the pick
    checkpoints[60]["summary"]["vs_base"]["saturation"] = (-0.28, 1.0)
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert result["onset"] == 60 and result["best_epoch"] == 100
    assert "do not last: epoch 100 shows none" in result["reasons"][1] and "lower learning rate" not in result["reasons"][1]


def test_an_undertrained_run_never_picks_a_checkpoint_that_is_still_the_base_model():
    # Kandinsky 6 LoRA at 3e-4 on three person clips (2026-10-14): the base model's clips are dim, blurry and nearly
    # still; epoch 40 looked the same (2% of the likeness gain), every later save is sharper and less saturated (toward
    # the training clips' own look) and flagged for it; by eye only epoch 200 learned anything (a partial likeness)
    gains = [-0.048, 0.102, 0.042, 0.044, 0.127]
    margins = [-0.253, -0.203, -0.139, -0.229, -0.024]
    sharpness = [0.0, 0.912, 1.418, 0.784, 1.075]
    checkpoints = {0: {"summary": {"face_mean": 0.08, "prompt": 0.1, "variety": 0.8, "face_found": 1.0, "face_area": 0.05}}}
    for index, (gain, margin, sharp) in enumerate(zip(gains, margins, sharpness)):
        checkpoints[40 * (index + 1)] = {"summary": {
            "face_mean": 0.08 + gain, "prompt": 0.1, "variety": 0.8, "face_found": 1.0, "dino_nearest_train": 0.3,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "hf_energy": (sharp, 1.0), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 8, "training_face_area": 0.2, "checkpoints": checkpoints})
    assert result["undertrained"] and result["onset"] == 80 and result["best_epoch"] == 200
    assert "the latest checkpoint, epoch 200, is the pick" in result["reasons"][0]
    assert any("(epoch 40) have hardly moved from the base model (at most 2% of the likeness gain, 0% of the move away from its own person)" in r
               for r in result["reasons"])


def test_an_undertrained_runs_pick_has_also_moved_away_from_the_base_models_person():
    # Mage-Flow LoRA with uniform timesteps at 3e-5 for 200 epochs (2026-10-15): a tiny likeness gain throughout, so the
    # first save had 41% of it while still drawing the base model's men (identity margin at its lowest); speckle
    # (sharpness drift) at every later save; by eye the most likeness is at epoch 200
    gains = [0.030, 0.081, 0.020, 0.080, 0.055, 0.061, 0.080, 0.130]
    margins = [-0.249, 0.045, -0.072, -0.017, -0.105, -0.074, -0.015, 0.041]
    sharpness = [0.0, 0.41, 0.83, 0.75, 0.96, 1.02, 1.19, 0.90]
    checkpoints = {0: {"summary": {"face_mean": 0.13, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, sharp) in enumerate(zip(gains, margins, sharpness)):
        checkpoints[25 * (index + 1)] = {"summary": {
            "face_mean": 0.13 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "hf_energy": (sharp, 1.0), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert result["undertrained"] and result["onset"] == 50 and result["best_epoch"] == 200
    assert any("(epoch 25) have hardly moved from the base model" in r and "0% of the move away from its own person" in r
               for r in result["reasons"])


def test_before_an_early_sign_the_pick_needs_faces_that_left_the_base_models_person():
    # Mage-Flow LoRA (2026-10-13): epoch 20 matched the training faces almost as well as epoch 60 (the subject's
    # glasses and hair) but was still the base model's man by eye; the subject from epoch 60, textures busier from 80
    gains = [0.277, 0.227, 0.232, 0.369, 0.324]
    margins = [-0.030, 0.062, 0.241, 0.191, 0.239]
    sharpness = [0.628, 0.663, 0.973, 1.129, 1.198]
    checkpoints = {0: {"summary": {"face_mean": 0.12, "prompt": 0.13, "variety": 0.93, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, sharp) in enumerate(zip(gains, margins, sharpness)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.12 + gain, "prompt": 0.13, "variety": 0.93, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "hf_energy": (sharp, 1.0), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.12, "checkpoints": checkpoints})
    assert not result["undertrained"] and result["onset"] == 80 and result["plateau_start"] == 80
    assert result["best_epoch"] == 60
    assert "epoch 60 has nearly the most likeness before them" in result["reasons"][1]


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


def test_cached_scores_belong_to_the_training_set_subject_and_prompt(tmp_path, monkeypatch):
    # klein9b_dog_shipped_e400 (2026-10-22), first scored against the person photos: analyzed again against the dog's own
    # photos, it kept every sample's likeness from the person photos
    from musubi_tuner_gui.best_checkpoint import scoring

    sample = tmp_path / "run_e000040_00_20261022120000_1001_000.png"
    sample.touch()
    calls = []
    monkeypatch.setattr(scoring, "_score_one", lambda scorer, item, prompts, references, subject, word: calls.append(subject) or {"prompt": 0.2})
    items, cache = [{"path": str(sample), "prompt_index": 0}], {}
    for fingerprint, subject, prompt in (("a", "ohwx dog", "p"), ("a", "ohwx dog", "p"), ("b", "ohwx dog", "p"), ("b", "ohwx man", "p"),
                                         ("b", "ohwx man", "q")):
        scoring.score_samples(None, items, [prompt], {"fingerprint": fingerprint}, subject, subject.split()[-1], cache)
    assert len(calls) == 4 and len(cache) == 4  # scored again for another training set, subject or prompt only
    monkeypatch.setattr(scoring, "_frames_of", lambda path, count: [])
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"1")
    first = scoring.reference_set(None, [str(photo)])["fingerprint"]
    assert scoring.reference_set(None, [str(photo)])["fingerprint"] == first
    photo.write_bytes(b"12")
    assert scoring.reference_set(None, [str(photo)])["fingerprint"] != first
    assert scoring.reference_set(None, [str(tmp_path / "other.png")])["fingerprint"] != first


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
    likeness = {0: 0.1, 20: 0.3, 40: 0.6, 60: 0.9}

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
    # a sign that comes and goes is shown muted; one that counts, and every flag of an older report, as a warning
    rows = loaded["result"]["rows"]
    rows[0].update(flags=["sharpness drift (+60% since epoch 10)"], signs=[])
    rows[1].update(flags=["weaker prompt following than the base model (-0.020 on the same prompts and seeds)"])
    rows[1].pop("signs", None)
    page = CF.results_html(loaded, str(out), "lora")
    assert 'bcf-flag bcf-flag-passing" title="Comes and goes' in page and page.count('<span class="bcf-flag">') == 1
    from musubi_tuner_gui.best_checkpoint import worker

    worker.write_html(str(tmp_path / "report.html"), {**report, "result": loaded["result"]})  # the worker's own report: int points
    text = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "comes and goes, not counted: sharpness drift" in text and "weaker prompt following" in text
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
    # 🔍 Analyze trains nothing: the monitor shows only the scoring, not empty training stages and loss tiles
    assert snapshot["after_only"] and [stage["id"] for stage in snapshot["stages"]] == ["score"]
    page = render_dashboard(snapshot)
    assert "Best checkpoint: step 500" in page and "Training finished" not in page and "loss chart" not in page and "Avg loss" not in page


def test_analyze_reuses_the_settings_a_finder_run_wrote(tmp_path):
    # 🔍 Analyze after a restart: the run's own settings.json names its prompts and training images, so the dataset
    # settings need not be filled in again
    out = str(tmp_path / "out")
    values = {"output_dir": out, "output_name": "mine"}
    assert CF.run_settings(out, "mine") is None
    folder = CF.run_folder(out, "mine")
    os.makedirs(folder)
    path = os.path.join(folder, "settings.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"prompts": ["photo of sks woman"], "lines": 1, "references": [str(tmp_path / "a.png")]}, handle)
    assert CF.run_settings(out, "mine") == path
    assert CF.analysis_settings(values, CF.defaults("image"), []) == path


def test_an_early_sign_that_does_not_last_keeps_the_full_likeness_after_it():
    # ERNIE-Image LoRA (2026-10-13): prompt following -0.010 at epoch 40 only; epoch 20 (before it) had half the
    # likeness, epochs 60-100 the subject without signs: the pick comes from the clean likeness level, not epoch 20
    gains = [0.369, 0.411, 0.433, 0.487, 0.447]
    margins = [0.120, 0.151, 0.320, 0.356, 0.346]
    prompt = [-0.006, -0.011, -0.003, -0.006, -0.006]
    checkpoints = {0: {"summary": {"face_mean": 0.17, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, change) in enumerate(zip(gains, margins, prompt)):
        checkpoints[20 * (index + 1)] = {"summary": {
            "face_mean": 0.17 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "prompt": (change, 0.8)}}}
    result = ranking.rank({"training_faces": 13, "training_face_area": 0.113, "checkpoints": checkpoints})
    assert not result["undertrained"] and result["plateau_start"] == 80 and result["best_epoch"] == 80  # judged 60-100, best 80
    assert result["onset"] is None
    assert any("not counted as over-training: weaker prompt following at epoch 40." in reason for reason in result["reasons"])


def test_the_finder_samples_at_the_step_saves_when_the_epochs_save_only_the_final_model(tmp_path):
    # the SDXL fine-tuning presets: 1 epoch of high-repeat folders, a save every epoch and every 451 steps
    values = {"output_dir": str(tmp_path / "out"), "output_name": "sdxl", "max_train_epochs": 1, "save_every_n_epochs": 1,
              "save_every_n_steps": 451, "sample_prompts": ""}
    settings = {**CF.defaults("image"), "enabled": True}
    run, after = CF.prepare_run(values, settings, [str(tmp_path / "a.png")])
    assert run["sample_every_n_steps"] == 451 and run["sample_every_n_epochs"] == 0 and after is not None
    assert CF.saved_checkpoints(values) is None  # the step count depends on the dataset
    # epochs that do save several checkpoints keep the epoch interval
    values.update(max_train_epochs=100, save_every_n_epochs=25)
    run, _after = CF.prepare_run(values, settings, [str(tmp_path / "a.png")])
    assert run["sample_every_n_epochs"] == 25 and run["sample_every_n_steps"] == 0


def test_a_lone_clean_checkpoint_among_lasting_signs_is_not_their_end():
    # LTX 2.3 GUI run (2026-10-14): framing x1.3-1.7 at epochs 50 and 100-175, clean at 75 and at 200, where the training
    # clip's garden replaced the prompts' scenes (no signal shows it); the subject from epoch 50: the pick is 75
    gains = [0.17, 0.34, 0.41, 0.49, 0.50, 0.50, 0.51, 0.43]
    margins = [0.05, 0.237, 0.356, 0.45, 0.50, 0.516, 0.548, 0.531]
    areas = [0.0, 0.477, 0.236, 0.368, 0.40, 0.721, 0.421, 0.22]
    checkpoints = {0: {"summary": {"face_mean": 0.126, "prompt": 0.09, "variety": 0.83, "face_found": 1.0, "face_area": 0.06}}}
    for index, (gain, margin, area) in enumerate(zip(gains, margins, areas)):
        checkpoints[25 * (index + 1)] = {"summary": {
            "face_mean": 0.126 + gain, "prompt": 0.09, "variety": 0.83, "face_found": 1.0, "dino_nearest_train": 0.5,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "face_area": (area, 1.0), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 8, "training_face_area": 0.2, "checkpoints": checkpoints})
    flagged = [row["epoch"] for row in result["rows"] if row["flags"]]
    assert flagged == [50, 100, 125, 150, 175] and result["best_epoch"] == 75
    assert any("the checkpoints after it carry them" in reason for reason in result["reasons"])


def test_signs_that_come_and_go_are_shown_but_not_counted():
    # Mage-Flow LoRA with uniform timesteps, 200 epochs (2026-10-14 server run): faces x1.32-1.37 at epochs 75-100,
    # x1.17 at 125-150, x1.47 at 175, x1.05 at 200, the scenes intact; by eye the anchor's suit, desk and map speckle
    # from epoch 125 (sharpness +54% / +52% / +59% since epoch 25), milder at 200 (+47%, just under the limit); the
    # subject from epoch 50: judged 50-100, best 100. Counting every flag picked epoch 200, the only clean save
    gains = [0.102, 0.209, 0.224, 0.205, 0.356, 0.186, 0.275, 0.282]
    margins = [-0.097, 0.083, 0.020, 0.103, 0.199, 0.095, 0.182, 0.109]
    areas = [(-0.070, 0.67), (0.158, 1.0), (0.276, 1.0), (0.314, 1.0), (0.162, 1.0), (0.153, 1.0), (0.388, 1.0), (0.048, 0.6)]
    sharpness = [0.579, 0.695, 0.963, 0.774, 1.010, 1.000, 1.041, 0.967]
    clipped = [0.002, 0.015, 0.011, 0.017, 0.083, 0.040, 0.061, 0.043]
    colour = [-0.019, 0.106, 0.095, 0.113, 0.137, 0.131, 0.167, 0.101]
    checkpoints = {0: {"summary": {"face_mean": 0.12, "prompt": 0.13, "variety": 0.9, "face_found": 1.0, "face_area": 0.03}}}
    for index, (gain, margin, area, sharp, clip, colourful) in enumerate(zip(gains, margins, areas, sharpness, clipped, colour)):
        checkpoints[25 * (index + 1)] = {"summary": {
            "face_mean": 0.12 + gain, "prompt": 0.13, "variety": 0.9, "face_found": 0.67 if index == 4 else 1.0, "dino_nearest_train": 0.2,
            "vs_base": {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "face_area": area, "hf_energy": (sharp, 1.0),
                        "clipped": (clip, 0.83), "colourfulness": (colourful, 0.83), "prompt": (0.0, 0.5)}}}
    result = ranking.rank({"training_faces": 12, "training_face_area": 0.113, "checkpoints": checkpoints})
    rows = {row["epoch"]: row for row in result["rows"]}
    assert [p for p, row in rows.items() if row["flags"]] == [75, 100, 125, 150, 175]
    assert [p for p, row in rows.items() if row["signs"]] == [125, 150, 175]  # the sharpness drift recurs, +47% at 200 included
    assert (result["plateau_start"], result["onset"], result["best_epoch"]) == (125, 125, 100)
    assert "They start as likeness levels off (epoch 125): epoch 100 has nearly the most likeness before them" in result["reasons"][1]
    passing = next(reason for reason in result["reasons"] if reason.startswith("Signs that come and go"))
    assert "the training photos' framing at epoch 75, epoch 100, epoch 175" in passing and "sharpness" not in passing


def test_the_training_scene_in_other_prompts_is_a_note_not_a_flag():
    # LTX 2.3 GUI run (2026-10-14): the training clip's garden behind the prompts' people from epoch 100 (backgrounds
    # +0.19 nearer the training frames than the base model's), the pick (75) stays where the rule put it
    gains = [0.17, 0.34, 0.41, 0.49, 0.50, 0.50, 0.51, 0.43]
    margins = [0.05, 0.237, 0.356, 0.45, 0.50, 0.516, 0.548, 0.531]
    areas = [0.0, 0.477, 0.236, 0.368, 0.40, 0.721, 0.421, 0.22]
    scene = [0.01, 0.02, 0.06, 0.19, 0.15, 0.18, 0.26, 0.39]

    def report(with_scene: bool) -> dict:
        checkpoints = {0: {"summary": {"face_mean": 0.126, "prompt": 0.09, "variety": 0.83, "face_found": 1.0, "face_area": 0.06}}}
        for index, (gain, margin, area, rise) in enumerate(zip(gains, margins, areas, scene)):
            vs_base = {"face_mean": (gain, 1.0), "identity": (margin, 1.0), "face_area": (area, 1.0), "prompt": (0.0, 0.5)}
            if with_scene:
                vs_base["scene_nearest_train"] = (rise, 1.0)
            checkpoints[25 * (index + 1)] = {"summary": {"face_mean": 0.126 + gain, "prompt": 0.09, "variety": 0.83, "face_found": 1.0,
                                                         "dino_nearest_train": 0.5, "vs_base": vs_base}}
        return {"training_faces": 8, "training_face_area": 0.2, "checkpoints": checkpoints}

    plain, noted = ranking.rank(report(False)), ranking.rank(report(True))
    assert plain["best_epoch"] == noted["best_epoch"] == 75 and plain["onset"] == noted["onset"]
    assert [row["flags"] for row in plain["rows"]] == [row["flags"] for row in noted["rows"]]
    note = next(reason for reason in noted["reasons"] if "surroundings" in reason)
    assert note.startswith("From epoch 100 the samples take on the training data's surroundings (backgrounds +0.19")
    assert not any("surroundings" in reason for reason in plain["reasons"])
    # training images without faces (a style or an object): nothing was masked, the whole image is what the run learns
    faceless = report(True)
    faceless["training_faces"] = 0
    assert not any("surroundings" in reason for reason in ranking.rank(faceless)["reasons"])
    # a single high save among low ones (a photo run's headshot backdrop) is not the training scene taking over
    scene[:] = [0.01, 0.13, 0.02, 0.03, 0.02, 0.04, 0.03, 0.05]
    assert not any("surroundings" in reason for reason in ranking.rank(report(True))["reasons"])


def test_the_fewest_signs_pick_counts_the_training_scene_as_a_sign():
    # Wan 2.2 T2I DoRA 2x (2026-10-17 server run): framing and prompt following flag every save from epoch 50, sharpness
    # drifts from 175; epoch 125 lacked the framing sign by chance while the training garden had replaced the news studio
    # from epoch 100 (backgrounds +0.15 nearer the training photos'): judged 25-75, best 50
    gains = [0.165, 0.285, 0.33, 0.295, 0.297, 0.252, 0.237, 0.303]
    margins = [(-0.159, 0.17), (0.131, 0.67), (0.266, 0.83), (0.234, 0.83), (0.138, 0.83), (0.208, 1.0), (0.173, 0.67), (0.213, 1.0)]
    areas = [(0.12, 0.5), (0.577, 0.83), (0.953, 1.0), (0.677, 1.0), (0.759, 0.67), (0.731, 1.0), (0.414, 0.67), (1.048, 1.0)]
    prompt = [(-0.004, 0.67), (-0.009, 0.67), (-0.013, 0.83), (-0.011, 0.83), (-0.011, 0.83), (-0.011, 0.83), (-0.012, 0.83), (-0.027, 0.83)]
    sharpness = [-0.329, -0.456, -0.48, -0.652, -0.645, -0.713, -0.899, -0.931]
    scene = [0.048, 0.085, 0.078, 0.149, 0.143, 0.149, 0.146, 0.183]

    def report(with_scene: bool) -> dict:
        checkpoints = {0: {"summary": {"face_mean": 0.089, "prompt": 0.133, "variety": 0.859, "face_found": 1.0, "face_area": 0.035}}}
        for index, (gain, margin, area, change, sharp, rise) in enumerate(zip(gains, margins, areas, prompt, sharpness, scene)):
            vs_base = {"face_mean": (gain, 1.0), "identity": margin, "face_area": area, "prompt": change, "hf_energy": (sharp, 1.0)}
            if with_scene:
                vs_base["scene_nearest_train"] = (rise, 1.0)
            checkpoints[25 * (index + 1)] = {"summary": {"face_mean": 0.089 + gain, "prompt": 0.133, "variety": 0.84, "face_found": 1.0,
                                                         "dino_nearest_train": 0.3, "vs_base": vs_base}}
        return {"training_faces": 12, "training_face_area": 0.113, "checkpoints": checkpoints}

    plain, noted = ranking.rank(report(False)), ranking.rank(report(True))
    assert (plain["plateau_start"], plain["onset"]) == (noted["plateau_start"], noted["onset"]) == (75, 50)
    assert [len(row["signs"]) for row in noted["rows"]] == [0, 1, 2, 2, 1, 2, 2, 3]
    assert plain["best_epoch"] == 125 and noted["best_epoch"] == 75
    assert "epoch 75 is the earliest with the fewest" in noted["reasons"][1] and "surroundings" not in noted["reasons"][1]
    assert any(reason.startswith("From epoch 100 the samples take on the training data's surroundings") for reason in noted["reasons"])


def test_one_prompts_scene_taken_over_is_noted_from_its_sample():
    # LTX 2.3 DoRA at 1e-4 on three clips (2026-10-15): the anchor prompt's studio turned into the training clip's garden
    # from epoch 100 (+0.38 for that sample) while the headshot and pilot kept their scenes (median +0.04)
    rises = {25: (-0.17, 0.00, 0.05), 50: (0.01, 0.00, 0.03), 75: (0.01, -0.01, 0.13), 100: (0.04, 0.01, 0.38), 125: (0.03, 0.02, 0.36),
             150: (0.00, 0.03, 0.26), 175: (0.06, 0.06, 0.31), 200: (0.07, 0.11, 0.30)}
    base_samples = [{"prompt_index": i, "seed": 1001, "scene_nearest_train": 0.3} for i in range(3)]
    checkpoints = {0: {"summary": {"face_mean": 0.08, "prompt": 0.1, "variety": 0.8, "face_found": 1.0}, "samples": base_samples}}
    for point, values in rises.items():
        checkpoints[point] = {"summary": {"face_mean": 0.5, "prompt": 0.1, "variety": 0.8, "face_found": 1.0, "dino_nearest_train": 0.4,
                                          "vs_base": {"face_mean": (0.4, 1.0), "scene_nearest_train": (sorted(values)[1], 1.0)}},
                              "samples": [{"prompt_index": i, "seed": 1001, "scene_nearest_train": 0.3 + v} for i, v in enumerate(values)]}
    result = ranking.rank({"training_faces": 8, "checkpoints": checkpoints})
    note = next(reason for reason in result["reasons"] if "surroundings" in reason)
    assert note.startswith("From epoch 100 the samples take on the training data's surroundings (one prompt's background +0.38")
    # one high sample that does not recur is no note
    for point in (125, 150, 175, 200):
        checkpoints[point]["samples"][2]["scene_nearest_train"] = 0.32
    assert not any("surroundings" in reason for reason in ranking.rank({"training_faces": 8, "checkpoints": checkpoints})["reasons"])


def test_the_background_mask_covers_the_person_below_the_face():
    from musubi_tuner_gui.best_checkpoint import scoring

    class Processor:  # dinov2-base: shortest side 256, centre crop 224, 16 x 16 patches
        size, crop_size = {"shortest_edge": 256}, {"height": 224, "width": 224}

    mask = scoring.person_patches((1024, 1024, 3), (412.0, 200.0, 200.0, 260.0), 16, Processor())
    assert mask[:, 4:12].all() and not mask[:, :4].any() and not mask[:, 12:].any()  # the face's columns, to the bottom
    assert not scoring.person_patches((480, 864, 3), None, 16, Processor()).any()  # no face: all background
    wide = scoring.person_patches((480, 864, 3), (400.0, 120.0, 60.0, 80.0), 16, Processor())  # 16:9 frame, centre crop
    assert wide.any() and not wide[:, 0].any() and not wide[:, -1].any()


def test_one_copied_sample_is_enough_and_old_reports_get_it_from_their_samples():
    # LTX 2.3 (2026-10-14): one of three samples copied a training clip's scene (DINOv2 0.88), the median stayed at 0.59;
    # the per-sample maximum flags it, also for reports written before the summary carried it
    base = {"face_mean": 0.1, "prompt": 0.13, "variety": 0.9, "face_found": 1.0}
    samples = [{"dino_nearest_train": value} for value in (0.59, 0.88, 0.25)]
    old = {"training_faces": 8, "checkpoints": {0: {"summary": base}, 20: {"summary": {**base, "face_mean": 0.5, "dino_nearest_train": 0.59},
                                                                         "samples": samples}}}
    flags = ranking.rank(old)["rows"][0]["flags"]
    assert flags == ["a sample copies a training image (similarity 0.88)"]
    clean = {"training_faces": 8, "checkpoints": {0: {"summary": base}, 20: {"summary": {**base, "face_mean": 0.5, "dino_nearest_max": 0.74}}}}
    assert not ranking.rank(clean)["rows"][0]["flags"]
