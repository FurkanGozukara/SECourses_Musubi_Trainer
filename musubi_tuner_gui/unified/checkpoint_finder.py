"""🏆 Best Checkpoint Finder in the unified UI: its settings, what Start changes for it, and its results.

Enabled, a run samples the finder's test prompts with the base model (Sample Before Training) and at every checkpoint
save; when training ends, the runner's after-step (``best_checkpoint.worker``) scores those samples (likeness, prompt
following, variety and over-training signs, all with local models) and names the checkpoint to use. The settings are
remembered per category in ui_state.json and saved with a user preset in its ``preset_best_checkpoint`` header line;
shipped presets never carry them, so loading one keeps yours. Image and video models only (audio: not needed)."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time

from ..best_checkpoint import prompts as BP
from ..best_checkpoint import ranking
from ..best_checkpoint.samples import index_samples, point_kind, point_label
from ..best_checkpoint.worker import FOLDER, run_folder
from . import presets as P

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the app folder (musubi_tuner_gui's parent)

CATEGORIES = ("image", "video")
HEADER_KEY = "preset_best_checkpoint"
STATE_KEY = "best_checkpoint"
MAX_SEEDS = 4
MAX_REFERENCES = 400  # training files the scores compare with (an even spread of a larger dataset)
VIDEO_PROMPTS = (0, 1, 5)  # default prompts of video models: each sample is a clip, so fewer of them
THUMB_SIZE = 448
LABEL = "Finding the best checkpoint"


class FinderError(ValueError):
    """Settings the finder cannot run with (Start shows it like its other problems)."""


# ------------------------------------------------------------------ settings
def available(category: str) -> bool:
    return category in CATEGORIES


def defaults(category: str) -> dict:
    video = category == "video"
    prompts = [BP.DEFAULT_PROMPTS[i] for i in VIDEO_PROMPTS] if video else list(BP.DEFAULT_PROMPTS)
    return {"enabled": False, "subject": BP.DEFAULT_SUBJECT, "prompts": prompts, "seeds": 1 if video else 2, "leakage": not video}


def normalize(raw, category: str) -> dict:
    """Settings of the right shape from ui_state.json, a preset header or the UI; anything unusable: the defaults."""
    out = defaults(category)
    if not isinstance(raw, dict):
        return out
    out["enabled"] = bool(raw.get("enabled", out["enabled"]))
    if isinstance(raw.get("subject"), str):
        out["subject"] = " ".join(raw["subject"].split())
    if isinstance(raw.get("prompts"), (list, str)):
        out["prompts"] = BP.clean_prompts(raw["prompts"])
    try:
        out["seeds"] = min(MAX_SEEDS, max(1, int(float(raw.get("seeds", out["seeds"])))))
    except (TypeError, ValueError):
        pass
    out["leakage"] = bool(raw.get("leakage", out["leakage"]))
    return out


def from_ui(category: str, enabled, subject, prompts_text, seeds, leakage) -> dict:
    return normalize({"enabled": enabled, "subject": subject or "", "prompts": prompts_text or "", "seeds": seeds, "leakage": leakage}, category)


def prompts_text(settings: dict) -> str:
    """One prompt per line, a blank line between them (long prompts stay readable; blank lines are ignored)."""
    return "\n\n".join(settings["prompts"])


def remembered(category: str) -> dict:
    state = P.load_ui_state().get(STATE_KEY)
    return normalize(state.get(category) if isinstance(state, dict) else None, category)


def remember(category: str, settings: dict) -> None:
    P.update_ui_state(**{STATE_KEY: {category: settings}})


def header_value(settings: dict) -> str:
    """The preset header's value: the settings as JSON (written as one TOML string)."""
    return json.dumps(settings, ensure_ascii=False, separators=(",", ":"))


def from_metadata(meta: dict, category: str) -> dict | None:
    """The settings a user preset was saved with (``read_metadata`` keeps the TOML string's escapes), else None."""
    raw = meta.get(HEADER_KEY)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return normalize(json.loads(json.loads(f'"{raw}"')), category)
    except ValueError:
        return None


def replace_subject(text: str, previous: str, subject: str) -> str:
    """The prompts with every whole-word ``previous`` subject (any letter case) written as ``subject``."""
    previous, subject = " ".join(previous.split()), " ".join(subject.split())
    if not previous or not subject or previous.casefold() == subject.casefold():
        return text
    return re.sub(rf"(?<![\w]){re.escape(previous)}(?![\w])", lambda _match: subject, text, flags=re.IGNORECASE)


def subject_from_folders(names: list[str], current: str = "") -> str:
    """The trigger and class of the first ``<repeats>_<trigger class>`` training folder name, else ''. A folder that
    names only the trigger (``1_ohwx``) keeps the class word of ``current`` (``ohwx man``)."""
    for name in names:
        match = re.match(r"^\d+_(.+)$", str(name or ""))
        if match and match.group(1).strip():
            found = " ".join(match.group(1).split())
            if " " not in found and BP.class_word(current):
                return f"{found} {BP.class_word(current)}"
            return found
    return ""


def summary_line(settings: dict) -> str:
    count = len(settings["prompts"]) + (1 if settings["leakage"] else 0)
    samples = count * settings["seeds"]
    return f"{count} prompt{'s' if count != 1 else ''} × {settings['seeds']} seed{'s' if settings['seeds'] != 1 else ''} = {samples} samples with the base model and at every saved checkpoint"


# ------------------------------------------------------------------ Start
def _own_prompt_lines(value) -> list[str]:
    """The user's own sample prompts (a prompt file, or kohya's prompt text): sampled after the finder's lines."""
    text = str(value or "").strip()
    if not text:
        return []
    if "\n" not in text:
        if not os.path.isfile(text):
            return []
        try:
            with open(text, encoding="utf-8") as handle:
                text = handle.read()
        except (OSError, UnicodeDecodeError):
            return []
    return [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]


def _whole(value) -> int:
    try:
        return max(0, int(float(value or 0)))
    except (TypeError, ValueError):
        return 0


def saved_checkpoints(values: dict) -> int | None:
    """How many checkpoints the run saves (the final one included), None when the length is not known here."""
    epochs, steps = _whole(values.get("save_every_n_epochs")), _whole(values.get("save_every_n_steps"))
    total_epochs, total_steps = _whole(values.get("max_train_epochs")), _whole(values.get("max_train_steps"))
    if epochs and total_epochs:
        return (total_epochs - 1) // epochs + 1
    if steps and total_steps and not total_epochs:
        return (total_steps - 1) // steps + 1
    return None


def sample_count(values: dict, settings: dict) -> int | None:
    """Samples the finder adds to the run: its lines at the start and at every saved checkpoint."""
    checkpoints = saved_checkpoints(values)
    lines = (len(settings["prompts"]) + (1 if settings["leakage"] else 0)) * settings["seeds"]
    return None if checkpoints is None else lines * (checkpoints + 1)


def prepare_run(values: dict, settings: dict, references: list[str], *, python: str | None = None,
                sample_size: tuple[int, int] | None = None) -> tuple[dict, "AfterStep"]:
    """The run's values with the finder's sampling, and the after-step that scores the samples. Writes
    ``<output_dir>/best_checkpoint/<output_name>/{sample_prompts.txt, settings.json}``. ``sample_size``: written into
    the finder's lines for trainers that do not add their own sample size to a prompt file (kohya sd-scripts)."""
    epochs, steps = _whole(values.get("save_every_n_epochs")), _whole(values.get("save_every_n_steps"))
    if not epochs and not steps:
        raise FinderError("🏆 Best Checkpoint Finder: set Save Every N Epochs (or Steps). The finder compares the checkpoints saved during training.")
    checkpoints = saved_checkpoints(values)
    if checkpoints is not None and checkpoints < 2:
        raise FinderError(
            "🏆 Best Checkpoint Finder: these settings save only the final model, so there is nothing to compare. "
            "Set a smaller Save Every N Epochs (or Steps) than the training length."
        )
    if not settings["prompts"]:
        raise FinderError("🏆 Best Checkpoint Finder: add at least one test prompt (↺ Default prompts restores the defaults).")
    output_dir = os.path.abspath(str(values.get("output_dir") or "").strip() or ".")
    output_name = str(values.get("output_name") or "").strip()
    if not output_name:
        raise FinderError("🏆 Best Checkpoint Finder: set the Output Name; the finder finds the run's checkpoints by it.")
    if not references:
        raise FinderError("🏆 Best Checkpoint Finder: no training images found to compare the samples with (check the dataset).")
    folder = run_folder(output_dir, output_name)
    os.makedirs(folder, exist_ok=True)
    seeds = BP.seeds(settings["seeds"])
    prompts = list(settings["prompts"])
    if settings["leakage"]:
        prompts.append(BP.leakage_prompt(settings["subject"]))
    lines = BP.prompt_file_lines(prompts, seeds, sample_size)
    prompt_file = os.path.join(folder, "sample_prompts.txt")
    with open(prompt_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines + _own_prompt_lines(values.get("sample_prompts"))) + "\n")
    per_line = [prompt for prompt in prompts for _seed in seeds]
    if len(references) > MAX_REFERENCES:
        step = len(references) / MAX_REFERENCES
        references = [references[int(i * step)] for i in range(MAX_REFERENCES)]
    finder_settings = {
        "created": time.time(),
        "prompts": per_line,
        "lines": len(lines),
        "leakage_indices": list(range(len(lines) - len(seeds), len(lines))) if settings["leakage"] else [],
        "subject": settings["subject"],
        "seeds": seeds,
        "references": references,
    }
    settings_path = os.path.join(folder, "settings.json")
    with open(settings_path, "w", encoding="utf-8") as handle:
        json.dump(finder_settings, handle, indent=1, ensure_ascii=False)
    run = dict(values)
    run["sample_prompts"] = prompt_file
    run["sample_at_first"] = True  # the base model's samples: the reference every checkpoint is compared with
    run["sample_every_n_epochs"] = epochs if epochs else 0
    run["sample_every_n_steps"] = 0 if epochs else steps
    return run, after_step(output_dir, output_name, settings_path, values.get("gpu_ids"), python=python)


def after_step(output_dir: str, output_name: str, settings_path: str, gpu_ids=None, *, python: str | None = None) -> "AfterStep":
    from .runner import AfterStep

    cmd = [python or sys.executable, "-m", "musubi_tuner_gui.best_checkpoint.worker", "--output-dir", output_dir,
           "--output-name", output_name, "--settings", settings_path, "--device", "auto"]
    env = dict(os.environ)
    first_gpu = str(gpu_ids or "").replace(" ", "").split(",")[0]
    if first_gpu.isdigit():  # the GPU the training used (accelerate's --gpu_ids sets the same variable)
        env["CUDA_VISIBLE_DEVICES"] = first_gpu
    env["PYTHONPATH"] = APP_DIR + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return AfterStep(LABEL, cmd, cwd=APP_DIR, env=env, read_result=lambda: result_summary(output_dir, output_name))


def run_settings(output_dir: str, output_name: str) -> str | None:
    """The settings.json of an earlier finder run or analysis (prompts, seeds and the training images it compared
    with), else None: 🔍 Analyze then needs no dataset settings."""
    path = os.path.join(run_folder(output_dir, output_name), "settings.json")
    return path if os.path.isfile(path) else None


def analysis_settings(values: dict, settings: dict, references: list[str]) -> str:
    """settings.json for 🔍 Analyze: the one the run wrote, else one for the run's own sample prompts."""
    output_dir = os.path.abspath(str(values.get("output_dir") or "").strip() or ".")
    path = run_settings(output_dir, str(values.get("output_name") or "").strip())
    if path:
        return path
    path = os.path.join(run_folder(output_dir, str(values.get("output_name") or "").strip()), "settings.json")
    prompts = [BP.prompt_text(line) for line in _own_prompt_lines(values.get("sample_prompts"))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"created": time.time(), "prompts": prompts, "lines": 0, "leakage_indices": [], "subject": settings["subject"],
            "references": references[:MAX_REFERENCES], "own_prompts": True}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1, ensure_ascii=False)
    return path


# ------------------------------------------------------------------ results
def load_report(output_dir: str, output_name: str) -> dict | None:
    try:
        with open(os.path.join(run_folder(output_dir, output_name), "report.json"), encoding="utf-8") as handle:
            report = json.load(handle)
        return report if isinstance(report, dict) and isinstance(report.get("result"), dict) else None
    except (OSError, ValueError):
        return None


def result_summary(output_dir: str, output_name: str) -> dict | None:
    """What the monitor shows when the finder is done (small: it is part of every dashboard update)."""
    report = load_report(output_dir, output_name)
    if report is None:
        return None
    result = report["result"]
    return {
        "kind": "best_checkpoint",
        "headline": f"Best checkpoint: {result.get('label', '')}" + (" · undertrained, train longer" if result.get("undertrained") else ""),
        "checkpoint": result.get("checkpoint") or "",
        "reasons": list(result.get("reasons") or [])[:4],
        "report": os.path.join(run_folder(output_dir, output_name), "report.html"),
    }


def _bar(value, low, high, best: bool = False, fmt: str = "{:.3f}") -> str:
    if value is None:
        return '<span class="bcf-na">—</span>'
    fraction = 0.0 if high is None or low is None or high <= low else (value - low) / (high - low)
    width = 6 + 94 * max(0.0, min(1.0, fraction))
    return (f'<div class="bcf-meter{" bcf-meter-best" if best else ""}"><span style="width:{width:.1f}%"></span></div>'
            f'<span class="bcf-num">{fmt.format(value)}</span>')


# face size: the samples' faces vs the base model's on the same prompts (1 = the same); an overtrained model draws its
# subject as close as in the training photos, whatever the prompt asks (ranking.FRAMING)
TREND_SERIES = (("likeness", "Likeness", "#10b981"), ("prompt", "Prompt following", "#8b5cf6"), ("framing", "Face size vs base model", "#ef4444"))
# the smallest range each score is drawn over: a change smaller than a meaningful one stays small on screen
MIN_SPAN = {"prompt": 2 * ranking.PROMPT_DROP, "variety": 0.10, "framing": 2 * (ranking.FRAMING - 1)}


def _span(values: list, minimum: float) -> tuple:
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    low, high = min(values), max(values)
    if high - low < minimum:
        middle = (low + high) / 2
        low, high = middle - minimum / 2, middle + minimum / 2
    return low, high


def _trend_svg(points: list[dict], best, spans: dict, onset=None) -> str:
    """Each score over the checkpoints (the base model first) with the pick and the first over-training signs marked,
    over ``spans`` (each score's own range, at least a meaningful change wide). The SVG stretches to the card's width (non-scaling strokes, round-capped
    dots); labels and the 🏆 are HTML at the same percentages, so text is never stretched."""
    if len(points) < 2:
        return ""
    width, height = 1000, 160
    x = [i * width / (len(points) - 1) for i in range(len(points))]
    lines = []
    best_index = next((i for i, point in enumerate(points) if point["epoch"] == best), None)
    if best_index is not None:
        lines.append(f'<line x1="{x[best_index]:.1f}" y1="0" x2="{x[best_index]:.1f}" y2="{height}" class="bcf-trend-best" vector-effect="non-scaling-stroke"/>')
    onset_index = next((i for i, point in enumerate(points) if onset is not None and point["epoch"] == onset), None)
    if onset_index is not None:
        lines.append(f'<line x1="{x[onset_index]:.1f}" y1="0" x2="{x[onset_index]:.1f}" y2="{height}" class="bcf-trend-onset" vector-effect="non-scaling-stroke"/>')
    for key, _name, color in TREND_SERIES:
        values = [point.get(key) for point in points]
        known = [value for value in values if value is not None]
        if len(known) < 2:
            continue
        low, high = spans[key]
        coords = [(x[i], 8 + (1 - (0.5 if high <= low else (value - low) / (high - low))) * (height - 16)) for i, value in enumerate(values) if value is not None]
        path = " ".join(f"{cx:.1f},{cy:.1f}" for cx, cy in coords)
        lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')
        lines.extend(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="{color}" stroke-width="8" stroke-linecap="round" '
                     'vector-effect="non-scaling-stroke"/>' for cx, cy in coords)
    position = [100 * i / (len(points) - 1) for i in range(len(points))]
    labels = "".join(f'<span style="left:{left:.2f}%">{html.escape(point["short"])}</span>' for left, point in zip(position, points))
    trophy = f'<span class="bcf-trend-trophy" style="left:{position[best_index]:.2f}%">🏆</span>' if best_index is not None else ""
    legend = "".join(f'<span><i style="background:{color}"></i>{name}</span>' for _key, name, color in TREND_SERIES)
    if onset_index is not None:
        legend += '<span><i class="bcf-onset-key"></i>over-training signs</span>'
    return (
        f'<div class="bcf-trend"><div class="bcf-trend-legend">{legend}<em>each score on its own scale</em></div>'
        f'<div class="bcf-trend-plot">{trophy}<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img" '
        f'aria-label="Scores over the checkpoints">{"".join(lines)}</svg></div><div class="bcf-trend-x">{labels}</div></div>'
    )


def results_html(report: dict | None, output_dir: str = "", output_name: str = "") -> str:
    if report is None:
        where = run_folder(output_dir, output_name) if output_dir and output_name else "the output folder"
        return (f'<div class="bcf-empty">No finder report in <code>{html.escape(where)}</code> yet. It is written when a run with the '
                "finder on ends, or by 🔍 Analyze for a run that saved samples.</div>")
    result = report["result"]
    kind = report.get("kind", "epoch")
    best = result.get("best_epoch")
    rows = result.get("rows") or []
    checkpoints = report.get("checkpoints") or {}
    base = (checkpoints.get("0") or checkpoints.get(0) or {}).get("summary") or {}

    def span(key: str) -> tuple:
        values = [row.get(key) for row in rows] + [base.get({"likeness": result.get("likeness_signal")}.get(key, key))]
        minimum = ranking.LIKENESS_MIN_GAIN.get(result.get("likeness_signal") or "", 0.1) if key == "likeness" else MIN_SPAN[key]
        return _span(values, minimum)

    spans = {key: span(key) for key in ("likeness", "prompt", "framing")}
    framing = any(row.get("framing") is not None for row in rows)
    body = ""
    if base:
        like = base.get(result.get("likeness_signal") or "")
        body += (
            '<tr class="bcf-base"><td>base model</td>'
            f'<td>{_bar(like, *spans["likeness"])}</td><td>{_bar(base.get("prompt"), *spans["prompt"])}</td>'
            f'<td>{_bar(1.0 if framing else None, *spans["framing"], fmt="×{:.2f}")}</td><td class="bcf-flags">reference</td></tr>'
        )
    for row in rows:
        is_best = row["epoch"] == best
        flags = "".join(f'<span class="bcf-flag">{html.escape(flag)}</span>' for flag in row["flags"])
        level = '<span class="bcf-level">likeness level</span>' if row.get("plateau") else ""
        body += (
            f'<tr class="{"bcf-best" if is_best else ""}"><td>{html.escape(row.get("label") or point_label(row["epoch"], kind))}{" 🏆" if is_best else ""}</td>'
            f'<td>{_bar(row.get("likeness"), *spans["likeness"], is_best)}</td><td>{_bar(row.get("prompt"), *spans["prompt"], is_best)}</td>'
            f'<td>{_bar(row.get("framing"), *spans["framing"], is_best, fmt="×{:.2f}")}</td><td class="bcf-flags">{level}{flags}</td></tr>'
        )
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in result.get("reasons") or [])
    trend_points = []
    if base:
        trend_points.append({"epoch": 0, "short": "base", "likeness": base.get(result.get("likeness_signal") or ""), "prompt": base.get("prompt"),
                             "framing": 1.0 if framing else None})
    for row in rows:
        label = row.get("label") or point_label(row["epoch"], kind)
        trend_points.append({"epoch": row["epoch"], "short": label.replace("epoch ", "e").replace("step ", "s"), **{key: row.get(key) for key in ("likeness", "prompt", "framing")}})
    trend = _trend_svg(trend_points, best, spans, result.get("onset"))
    checkpoint = result.get("checkpoint") or ""
    samples = sum(len(data.get("samples") or []) for data in checkpoints.values())
    created = time.strftime("%Y-%m-%d %H:%M", time.localtime(report.get("created") or time.time()))
    like_name = {"face_mean": "face match", "dino_nearest_train": "image similarity"}.get(result.get("likeness_signal") or "", "likeness")
    report_path = os.path.join(run_folder(report.get("output_dir") or output_dir, report.get("output_name") or output_name), "report.html")
    return (
        '<div class="bcf-result">'
        '<div class="bcf-hero"><div class="bcf-trophy">🏆</div><div class="bcf-hero-main">'
        f'<div class="bcf-hero-title">Best checkpoint: {html.escape(result.get("label") or "")}'
        + ('<span class="bcf-undertrained">⚠️ likeness not learned yet</span>' if result.get("undertrained") else "")
        + "</div>"
        + (f'<code class="bcf-path">{html.escape(checkpoint)}</code>' if checkpoint else '<div class="bcf-path">(not in the output folder)</div>')
        + f'<div class="bcf-hero-sub">{len(rows)} checkpoints and the base model compared · {samples} samples · '
        f'{report.get("training_images", 0)} training images / frames · likeness = {like_name} · {created}</div>'
        "</div></div>"
        f'<ul class="bcf-reasons">{reasons}</ul>'
        + trend
        + '<div class="bcf-table-wrap"><table class="bcf-table"><thead><tr><th>Checkpoint</th><th>Likeness</th>'
        "<th>Prompt following</th><th>Face size</th><th>Notes</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
        f'<div class="bcf-foot">Every sample of every checkpoint side by side: <code>{html.escape(report_path)}</code></div>'
        "</div>"
    )


def _thumb_dir(output_dir: str) -> str:
    import tempfile

    digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), "secourses_best_checkpoint", digest)


def _thumbnail(path: str, target: str) -> str | None:
    """A small JPEG of a sample (a video's middle frame) in the temp folder Gradio may serve."""
    if os.path.isfile(target) and os.path.getmtime(target) >= os.path.getmtime(path):
        return target
    try:
        from PIL import Image

        if path.lower().endswith((".mp4", ".webm", ".mov")):
            from ..best_checkpoint.scoring import video_frames

            frames = video_frames(path, 3)
            if not frames:
                return None
            image = Image.fromarray(frames[len(frames) // 2])
        else:
            image = Image.open(path)
        with image:
            image = image.convert("RGB")
            image.thumbnail((THUMB_SIZE, THUMB_SIZE))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            image.save(target + ".tmp.jpg", "JPEG", quality=88)
        os.replace(target + ".tmp.jpg", target)
        return target
    except Exception:  # noqa: BLE001 - a sample that cannot be previewed is left out of the comparison
        return None


def comparison_points(report: dict | None) -> list[int]:
    """The gallery's columns: the base model, the best and the last checkpoint (fewer when they coincide)."""
    if report is None:
        return []
    points = sorted(int(point) for point in (report.get("checkpoints") or {}))
    best = (report.get("result") or {}).get("best_epoch")
    return [point for point in dict.fromkeys([0, best, points[-1] if points else None]) if point in points]


def comparison_gallery(report: dict | None) -> list[tuple[str, str]]:
    """Base model | best | last checkpoint for each test prompt (first seed): what the recommendation rests on."""
    if report is None:
        return []
    kind = report.get("kind", "epoch")
    checkpoints = {int(point): data for point, data in (report.get("checkpoints") or {}).items()}
    best = (report.get("result") or {}).get("best_epoch")
    columns = comparison_points(report)
    prompts = (report.get("settings") or {}).get("prompts") or []
    leak = set((report.get("settings") or {}).get("leakage_indices") or [])
    first: dict[int, dict] = {}
    for point in columns:
        for sample in checkpoints[point].get("samples") or []:
            if sample["prompt_index"] in leak:
                continue
            group = first.setdefault(sample["prompt_index"], {})
            if point not in group or str(sample.get("seed", "")) < str(group[point].get("seed", "")):
                group[point] = sample
    folder = _thumb_dir(report.get("output_dir") or "")
    items = []
    for prompt_index in sorted(first):
        text = prompts[prompt_index] if prompt_index < len(prompts) else ""
        short = (text[:70] + "…") if len(text) > 70 else text
        for point in columns:
            sample = first[prompt_index].get(point)
            if not sample or not os.path.isfile(sample["path"]):
                continue
            stem = hashlib.sha1(sample["path"].encode("utf-8")).hexdigest()[:16]
            thumb = _thumbnail(sample["path"], os.path.join(folder, f"{stem}.jpg"))
            if thumb:
                tag = "🏆 " if point == best else ""
                items.append((thumb, f"{tag}{point_label(point, kind)} · prompt {prompt_index + 1}\n{short}"))
    return items


def run_status(output_dir: str, output_name: str) -> str:
    """One line about the run's samples so far (shown before a report exists)."""
    samples = index_samples(output_dir, output_name) if output_dir and output_name else {}
    if not samples:
        return ""
    kind = point_kind(samples)
    points = [point_label(point, kind) for point in samples]
    return f"{sum(len(v) for v in samples.values())} samples of {len(samples)} checkpoints so far ({', '.join(points[:6])}{'…' if len(points) > 6 else ''})."
