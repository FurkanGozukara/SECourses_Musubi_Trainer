"""End to end: messy UI input -> the runtime TOML each trainer actually reads.

For every model and training type, the unified backend builds the real runtime TOML (print-only:
nothing is trained, model files are empty placeholders, CPU only) from user input full of quoted,
padded and mixed-slash paths and numbers typed as text. Every value the trainer reads must arrive
in the type its argparse option declares (the TOML bypasses argparse conversion), and every path
must arrive clean.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import pytest
import toml

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (APP_DIR, TESTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from media_fixtures import caption, write_image, write_video, write_wav  # noqa: E402

from musubi_tuner_gui.unified import presets as P  # noqa: E402
from musubi_tuner_gui.unified.backends import get_backend, install_quiet_toasts  # noqa: E402
from musubi_tuner_gui.unified.registry import ENTRIES  # noqa: E402
from musubi_tuner_gui.unified.runner import install_runner  # noqa: E402
from musubi_tuner_gui.unified.schema import arg_type  # noqa: E402

OPTIONAL_MODEL_FILES = {"turbo_dit", "unconditional_dit", "training_adapter_path", "gemma_safetensors", "network_weights", "base_weights", "dit_high_noise", "resume"}
NUMERIC = ("int", "float", "custom:_int_or_float", "custom:int_or_float")
CASES = []
for _entry in ENTRIES:
    for _mode in _entry.modes:
        if not any(case[0].backend == _entry.backend and case[1] == _mode for case in CASES) or _entry.backend == "ltx2":
            CASES.append((_entry, _mode))


@pytest.fixture(scope="module", autouse=True)
def _app_like(tmp_path_factory):
    install_quiet_toasts()
    install_runner()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        # the classic print path appends to ./logs/print_command.txt: keep it out of the app folder
        patch.chdir(tmp_path_factory.mktemp("cwd"))
        yield


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets")
    image = root / "images" / "1_ohwx"
    video = root / "videos" / "1_clip"
    audio = root / "audio" / "1_voice"
    for folder in (image, video, audio):
        folder.mkdir(parents=True)
    caption(write_image(str(image / "a.png"), size=(512, 512)), "ohwx")
    caption(write_video(str(video / "b.mp4"), frames=22, audio_seconds=1.0), "clip")
    caption(write_wav(str(audio / "c.wav")), "voice")
    return {"image": root / "images", "video": root / "videos", "audio": root / "audio"}


def _messy_path(path: str, index: int) -> str:
    forms = [
        lambda p: f'"{p}"',
        lambda p: f"'{p}'",
        lambda p: "  " + p.replace("\\", "/") + "/  ",
        lambda p: p + "\\\n",
        lambda p: f' " {p} " ',
    ]
    return forms[index % len(forms)](path)


def _type_ok(expected: str, value) -> bool:
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "float":
        return isinstance(value, float)
    if expected in NUMERIC:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected in ("str", "custom:Path"):
        return isinstance(value, str)
    return True


def _runtime_toml(backend, values, out) -> dict:
    text = backend.print_command(values)
    written = [path for path in glob.glob(str(out / "*.toml")) if "dataset" not in os.path.basename(path).lower()]
    shown = re.findall(r"end of toml config file: (.+?\.toml)", text or "")
    runtime = sorted(written, key=os.path.getmtime)[-1] if written else (shown[-1].strip() if shown else None)
    assert runtime, f"no runtime TOML in the command preview:\n{(text or '')[-2000:]}"
    return toml.load(runtime)


@pytest.mark.parametrize("entry_key", ["qwen_image", "wan21", "ltx23", "flux2_dev"])
def test_base_weights_reach_the_trainer_as_lists(entry_key, datasets, tmp_path):
    """The unified type gate and the classic writer both handle base weights: paths with spaces survive."""
    from musubi_tuner_gui.unified.registry import ENTRY_BY_KEY

    entry = ENTRY_BY_KEY[entry_key]
    backend = get_backend(entry.backend)
    if "base_weights" not in backend.keys():
        pytest.skip("no base weights setting")
    values = backend.defaults("lora")
    values.update(entry.fixed)
    models = tmp_path / "models"
    models.mkdir()
    for spec in backend.specs:
        if spec.key in OPTIONAL_MODEL_FILES:
            values[spec.key] = ""
        elif spec.kind == "file" and spec.section == "model":
            (models / f"{spec.key}.safetensors").touch()
            values[spec.key] = str(models / f"{spec.key}.safetensors")
        elif spec.kind == "folder" and spec.section == "model":
            (models / spec.key).mkdir()
            values[spec.key] = str(models / spec.key)
    lora = tmp_path / "my loras" / "style lora.safetensors"
    lora.parent.mkdir()
    lora.touch()
    out = tmp_path / "out"
    out.mkdir()
    values.update(
        dataset_config_mode="Generate from Folder Structure",
        parent_folder_path=str(datasets["image"]),
        output_dir=str(out),
        sample_prompts="",
        base_weights=f'"{lora}"',
        base_weights_multiplier="0,5",
    )
    if "training_mode" in values:
        values["training_mode"] = backend.mode_value("lora")
    data = _runtime_toml(backend, values, out)
    assert data["base_weights"] == [str(lora)]
    assert data["base_weights_multiplier"] == [0.5]


@pytest.mark.parametrize("entry, mode", CASES, ids=[f"{entry.key}-{mode}" for entry, mode in CASES])
def test_runtime_toml_has_the_trainer_types_and_clean_paths(entry, mode, datasets, tmp_path):
    backend = get_backend(entry.backend)
    values = backend.defaults(mode)
    preset = P.recommended_preset(entry, mode, P.AUTO_TIER)
    if preset:
        values = backend.load(preset, values).values
    values.update(entry.fixed)
    if "training_mode" in values:
        values["training_mode"] = backend.mode_value(mode)
    models = tmp_path / "models"
    models.mkdir()
    for spec in backend.specs:
        if spec.key in entry.fixed:
            continue  # the model card decides it (LTX 2.5 clears the Gemma 3 paths); the form never changes it
        if spec.key in OPTIONAL_MODEL_FILES:
            values[spec.key] = ""
        elif spec.kind == "file" and spec.section == "model":
            target = models / f"{spec.key}.safetensors"
            target.touch()
            values[spec.key] = str(target)
        elif spec.kind == "folder" and spec.section == "model":
            (models / spec.key).mkdir()
            values[spec.key] = str(models / spec.key)
    kind = "audio" if entry.key == "ltx23_audio" else "video" if entry.backend == "minimax_h3" else "image"
    out = tmp_path / "out dir"
    out.mkdir()
    values.update(dataset_config_mode="Generate from Folder Structure", parent_folder_path=str(datasets[kind]), output_dir=str(out), sample_prompts="")
    if entry.backend == "minimax_h3":
        values.update(dataset_num_frames=22, allow_experimental_duration=True)

    # what a user might really type: quoted / padded / mixed-slash paths, numbers and switches as text
    expected_paths = {}
    for index, key in enumerate(sorted(k for k, v in values.items() if isinstance(v, str) and v and os.path.isabs(v) and os.path.exists(v))):
        expected_paths[key] = os.path.normpath(values[key])
        values[key] = _messy_path(values[key], index)
    for spec in backend.specs:
        value = values.get(spec.key)
        if spec.kind in ("number", "slider") and isinstance(value, (int, float)) and not isinstance(value, bool):
            values[spec.key] = f"  {value} "
        elif spec.kind == "checkbox" and isinstance(value, bool):
            values[spec.key] = f" {str(value).upper()} "

    assert backend.check(values, mode) == []
    text = backend.print_command(values)
    written = [path for path in glob.glob(str(out / "*.toml")) if "dataset" not in os.path.basename(path).lower()]
    shown = re.findall(r"end of toml config file: (.+?\.toml)", text or "")
    runtime = sorted(written, key=os.path.getmtime)[-1] if written else (shown[-1].strip() if shown else None)
    assert runtime, f"no runtime TOML in the command preview:\n{(text or '')[-2000:]}"
    data = toml.load(runtime)
    problems = []
    checked = 0
    for key, value in data.items():
        info = arg_type(entry.backend, key)
        if info is None:
            continue  # a GUI-only bookkeeping key (argparse ignores unknown keys)
        checked += 1
        nargs = info.get("nargs")
        if nargs not in (None, "?"):
            if not isinstance(value, list):
                problems.append(f"{key}={value!r}: the trainer expects a list ({info['type']})")
            elif info["type"] in NUMERIC and not all(_type_ok(info["type"], item) for item in value):
                problems.append(f"{key}={value!r}: list items must be {info['type']}")
        elif not _type_ok(info["type"], value):
            problems.append(f"{key}={value!r} ({type(value).__name__}): the trainer expects {info['type']}")
        if key in expected_paths and os.path.normcase(os.path.normpath(str(value))) != os.path.normcase(expected_paths[key]):
            problems.append(f"{key}={value!r}: path not cleaned (expected {expected_paths[key]})")
        if isinstance(value, str) and any(mark in value for mark in ('"', "\n")):
            problems.append(f"{key}={value!r}: quotes or line breaks reached the trainer")
    assert not problems, "\n".join(problems)
    assert checked >= 30 and len(expected_paths) >= 3, (checked, sorted(expected_paths))
