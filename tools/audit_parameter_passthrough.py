"""Per-setting pass-through audit for every model and training type of the unified UI.

Every setting the training script reads (an argparse option recorded in legacy_schema.json) is changed, one at a
time, from a working configuration (the model's defaults + its recommended shipped preset) to another valid value.
The backend then builds the real runtime TOML and launch command (print-only: nothing is trained, model files are
empty placeholders, CPU only). The setting must arrive unchanged: in the runtime TOML in the type the trainer
expects, or on the launch command line.

    venv/Scripts/python.exe tools/audit_parameter_passthrough.py [--models krea2,qwen_image] [--out report.json]

Statuses: ok (arrived unchanged), changed (arrived with another value), missing (in neither the TOML nor the
command), refused (the UI's own checks refuse that value: fine, it never reaches the trainer), error (the print path
crashed), skipped (no other valid value to try: paths, free text).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import tempfile
import time
import traceback

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(APP_DIR, "tests")
for _path in (APP_DIR, TESTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import toml  # noqa: E402

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from media_fixtures import caption, write_image, write_video, write_wav  # noqa: E402
from musubi_tuner_gui.unified import presets as P  # noqa: E402
from musubi_tuner_gui.unified.backends import get_backend, install_quiet_toasts  # noqa: E402
from musubi_tuner_gui.unified.registry import ENTRIES  # noqa: E402
from musubi_tuner_gui.unified.runner import install_runner  # noqa: E402
from musubi_tuner_gui.unified.schema import arg_type  # noqa: E402
from musubi_tuner_gui.unified.values import ValueProblem  # noqa: E402

OPTIONAL_MODEL_FILES = {"turbo_dit", "unconditional_dit", "training_adapter_path", "gemma_safetensors", "network_weights", "base_weights", "dit_high_noise", "resume"}
# set by the run itself (paths of this run, the dataset TOML it generates, the mode)
NOT_AUDITED = {"output_dir", "dataset_config", "generated_toml_path", "training_mode", "logging_dir", "sample_prompts", "parent_folder_path"}
SKIP = object()


def datasets(root: str) -> dict:
    image = os.path.join(root, "images", "1_ohwx")
    video = os.path.join(root, "videos", "1_clip")
    audio = os.path.join(root, "audio", "1_voice")
    for folder in (image, video, audio):
        os.makedirs(folder, exist_ok=True)
    caption(write_image(os.path.join(image, "a.png"), size=(512, 512)), "ohwx")
    caption(write_video(os.path.join(video, "b.mp4"), frames=22, audio_seconds=1.0), "clip")
    caption(write_wav(os.path.join(audio, "c.wav")), "voice")
    return {"image": os.path.dirname(image), "video": os.path.dirname(video), "audio": os.path.dirname(audio)}


def base_values(entry, mode, backend, data_roots, models_dir) -> dict:
    values = backend.defaults(mode)
    preset = P.recommended_preset(entry, mode, P.AUTO_TIER)
    if preset:
        values = backend.load(preset, values).values
    values.update(entry.fixed)
    if "training_mode" in values:
        values["training_mode"] = backend.mode_value(mode)
    for spec in backend.specs:
        if spec.key in OPTIONAL_MODEL_FILES:
            values[spec.key] = ""
        elif spec.kind == "file" and spec.section == "model":
            target = os.path.join(models_dir, f"{spec.key}.safetensors")
            open(target, "ab").close()
            values[spec.key] = target
        elif spec.kind == "folder" and spec.section == "model":
            os.makedirs(os.path.join(models_dir, spec.key), exist_ok=True)
            values[spec.key] = os.path.join(models_dir, spec.key)
    kind = "audio" if entry.key == "ltx23_audio" else "video" if entry.backend == "minimax_h3" else "image"
    values.update(dataset_config_mode="Generate from Folder Structure", parent_folder_path=data_roots[kind], sample_prompts="")
    if entry.backend == "minimax_h3":
        values.update(dataset_num_frames=22, allow_experimental_duration=True)
    return values


def _choice_values(spec) -> list:
    return [value for value in spec.choice_values() if value not in (None, "")]


def alternative(backend, spec, value):
    """Another valid value for ``spec`` (SKIP when there is no meaningful one)."""
    kind = spec.kind
    if spec.key in NOT_AUDITED or kind in ("file", "folder"):
        return SKIP
    if kind == "checkbox":
        current = str(value).strip().lower() in ("1", "true", "yes", "on") if not isinstance(value, bool) else value
        return not current
    choices = _choice_values(spec)
    if choices and kind in ("dropdown", "radio") and not spec.multiselect:
        others = [choice for choice in choices if str(choice) != str(value)]
        return others[0] if others else SKIP
    if kind in ("number", "slider"):
        try:
            number = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            number = None
        low, high = spec.minimum, spec.maximum
        whole = backend.whole_number(spec.key) or (spec.step is not None and float(spec.step).is_integer() and (number is None or float(number).is_integer()))
        if number is None:
            candidate = 1 if whole else 0.25
        elif whole:
            candidate = int(number) + 1
            if high is not None and candidate > high:
                candidate = int(number) - 1
        else:
            candidate = number * 1.5 if number else 0.125
            if high is not None and candidate > high:
                candidate = (number + (low if low is not None else 0)) / 2
        if low is not None and candidate < low:
            return SKIP
        if high is not None and candidate > high:
            return SKIP
        if number is not None and math.isclose(float(candidate), number):
            return SKIP
        return int(candidate) if whole else round(candidate, 6)
    info = arg_type(backend.name, spec.key) or {}
    if kind in ("text", "textarea") and info.get("nargs") not in (None, "?") and spec.key in ("optimizer_args", "lr_scheduler_args", "network_args"):
        return (str(value or "") + " audit_key=7").strip()
    return SKIP


def _same(expected, actual) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(expected) == len(actual) and all(_same(a, b) for a, b in zip(expected, actual))
    return str(expected) == str(actual)


def _runtime_toml(text: str, out: str):
    written = [path for path in glob.glob(os.path.join(out, "**", "*.toml"), recursive=True) if "dataset" not in os.path.basename(path).lower()]
    shown = re.findall(r"end of toml config file: (.+?\.toml)", text or "")
    runtime = sorted(written, key=os.path.getmtime)[-1] if written else (shown[-1].strip() if shown else None)
    return toml.load(runtime) if runtime and os.path.isfile(runtime) else None


def _on_command_line(key: str, expected, text: str) -> bool | None:
    """True / False when ``--key`` appears on a command line (with the expected value), None when it does not."""
    pattern = re.compile(rf"--{re.escape(key)}(?:[ =]+([^\s]+))?")
    matches = [match.group(1) for match in pattern.finditer(text or "")]
    if not matches:
        return None
    if isinstance(expected, bool):
        return expected  # a store_true flag on the line means True
    return any(match is not None and _same(expected, _number_or_text(match.strip("'\""))) for match in matches)


def _number_or_text(text: str):
    try:
        number = float(text)
        return int(number) if number.is_integer() and "." not in text and "e" not in text.lower() else number
    except ValueError:
        return text


def audit_case(entry, mode, data_roots, work: str) -> list[dict]:
    backend = get_backend(entry.backend)
    models_dir = os.path.join(work, "models")
    os.makedirs(models_dir, exist_ok=True)
    base = base_values(entry, mode, backend, data_roots, models_dir)
    base_problems = backend.check(dict(base, output_dir=os.path.join(work, "base")), mode)
    rows = []
    if base_problems:
        return [{"model": entry.key, "mode": mode, "key": "*", "status": "base_refused", "detail": "; ".join(base_problems)[:400]}]
    for spec in backend.specs:
        if mode not in spec.modes or not spec.visible or spec.key in entry.fixed:
            continue
        if arg_type(backend.name, spec.key) is None:
            continue  # GUI-only setting: never a trainer argument
        current = base.get(spec.key, spec.default)
        alt = alternative(backend, spec, current)
        row = {"model": entry.key, "mode": mode, "key": spec.key, "label": spec.label, "from": repr(current)[:80]}
        if alt is SKIP:
            rows.append({**row, "status": "skipped"})
            continue
        row["to"] = repr(alt)[:80]
        values = dict(base)
        values[spec.key] = alt
        with tempfile.TemporaryDirectory(dir=work) as out:
            values["output_dir"] = out
            try:
                expected = backend.trainer_value(spec.key, alt)
            except ValueProblem as exc:
                rows.append({**row, "status": "refused", "detail": str(exc)[:200]})
                continue
            problems = backend.check(values, mode)
            if problems:
                rows.append({**row, "status": "refused", "detail": "; ".join(problems)[:300]})
                continue
            started = time.time()
            try:
                text = backend.print_command(values)
            except ValueError as exc:  # the trainer's own validation refuses this combination before anything runs
                rows.append({**row, "status": "refused", "detail": f"trainer: {str(exc)[:300]}"})
                continue
            except Exception as exc:  # noqa: BLE001
                rows.append({**row, "status": "error", "detail": f"{type(exc).__name__}: {str(exc)[:300]}", "trace": traceback.format_exc()[-1500:]})
                continue
            data = _runtime_toml(text, out) or {}
            row["seconds"] = round(time.time() - started, 2)
            if spec.key in data:
                actual = data[spec.key]
                status = "ok" if _same(expected, actual) else "changed"
                rows.append({**row, "status": status, "expected": repr(expected)[:120], "actual": repr(actual)[:120]})
                continue
            on_line = _on_command_line(spec.key, expected, text)
            if on_line is True:
                rows.append({**row, "status": "ok", "where": "command line"})
            elif on_line is False:
                rows.append({**row, "status": "changed", "where": "command line", "expected": repr(expected)[:120], "detail": text[-600:]})
            elif expected in (None, False, "", []):
                rows.append({**row, "status": "ok", "where": "omitted (unset / off)"})
            else:
                rows.append({**row, "status": "missing", "expected": repr(expected)[:120]})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="comma-separated model keys (default: one entry per trainer and training type)")
    parser.add_argument("--out", default=os.path.join(APP_DIR, "logs", "parameter_audit.json"))
    args = parser.parse_args()
    install_quiet_toasts()
    install_runner()
    wanted = {key.strip() for key in args.models.split(",") if key.strip()}
    cases = []
    for entry in ENTRIES:
        for mode in entry.modes:
            if wanted and entry.key not in wanted:
                continue
            if not wanted and any(case[0].backend == entry.backend and case[1] == mode for case in cases) and entry.backend != "ltx2":
                continue
            cases.append((entry, mode))
    results = []
    with tempfile.TemporaryDirectory(prefix="param_audit_") as root:
        os.environ.setdefault("MUSUBI_USER_PRESETS_DIR", os.path.join(root, "user_presets"))
        cwd = os.getcwd()
        os.chdir(root)  # the classic print path appends to ./logs/print_command.txt
        try:
            data_roots = datasets(os.path.join(root, "datasets"))
            for entry, mode in cases:
                started = time.time()
                work = os.path.join(root, f"{entry.key}_{mode}")
                os.makedirs(work, exist_ok=True)
                try:
                    rows = audit_case(entry, mode, data_roots, work)
                except Exception as exc:  # noqa: BLE001
                    rows = [{"model": entry.key, "mode": mode, "key": "*", "status": "error", "detail": f"{type(exc).__name__}: {exc}"}]
                results.extend(rows)
                counts = {}
                for row in rows:
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                print(f"{entry.key:16s} {mode:5s} {time.time() - started:6.0f}s  {counts}", flush=True)
        finally:
            os.chdir(cwd)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)
    bad = [row for row in results if row["status"] in ("changed", "missing", "error", "base_refused")]
    for row in bad:
        print(f"{row['status']:8s} {row['model']:16s} {row['mode']:5s} {row['key']:34s} {row.get('to', '')} -> {row.get('actual', row.get('detail', ''))}"[:400])
    print(f"{len(results)} settings audited, {len(bad)} need a look; report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
