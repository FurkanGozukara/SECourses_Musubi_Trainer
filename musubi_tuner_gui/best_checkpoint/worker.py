"""Find the best checkpoint of a run from the samples it generated at every save (🏆 Best Checkpoint Finder).

usage: python -m musubi_tuner_gui.best_checkpoint.worker --output-dir <dir> --output-name <name>
       [--settings <output_dir>/best_checkpoint/<output_name>/settings.json] [--dataset <folder>] [--device auto|cuda|cpu]
settings.json (written by the GUI when a run starts with the finder on):
  {"prompts": [the prompt of each sample line], "lines": <the finder's sample lines>, "leakage_indices": [line, ...],
   "subject": "ohwx man", "references": [training images / videos]}
Writes report.json and report.html next to settings.json and prints "BEST_CHECKPOINT: <path> | <checkpoint>"
plus "BEST_CHECKPOINT_REASON: ..." lines, or "BEST_CHECKPOINT_ERROR: ..." (exit code 1).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import sys
import time

from . import ranking, scoring
from .prompts import class_word
from .samples import checkpoint_for, index_samples, point_kind, point_label

FOLDER = "best_checkpoint"


def run_folder(output_dir: str, output_name: str) -> str:
    """<output_dir>/best_checkpoint/<output_name>: the finder's files of one run (several runs can share a folder)."""
    return os.path.join(output_dir, FOLDER, output_name)
LEAK_DELTA = 0.15  # face similarity a prompt without the trigger may gain over the base model before it is noted
# signals compared with the base model's sample of the same prompt and seed (True: as a log ratio)
PAIRED = {"face_mean": False, "face_area": True, "prompt": False, "dino_nearest_train": False, "scene_nearest_train": False,
          "saturation": True, "colourfulness": True, "hf_energy": True, "gray_std": True, "clipped": False}
MAX_REFERENCES = 400


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def analyze(output_dir: str, output_name: str, settings: dict, device: str = "auto", dataset: str = "", log=print) -> dict:
    folder = run_folder(output_dir, output_name)
    samples = index_samples(output_dir, output_name)
    lines = int(settings.get("lines") or 0)
    if lines:  # the user's own sample prompts follow the finder's lines in the prompt file: not scored
        samples = {point: kept for point, items in samples.items() if (kept := [s for s in items if s["prompt_index"] < lines])}
    if not samples:
        return {"error": f"no samples of '{output_name}' in {os.path.join(output_dir, 'sample')}: train with the finder (or sampling) on first"}
    kind = point_kind(samples)
    partial = ""
    points = sorted(samples)
    if len(points) > 1 and len(samples[points[-1]]) < len(samples[points[-2]]):  # still sampling, or stopped while sampling
        last = points[-1]
        partial = f"{point_label(last, kind)} was left out: {len(samples[last])} of its {len(samples[points[-2]])} samples were found"
        log(f"Best checkpoint finder: {partial} (still being generated?)")
        samples.pop(last)
    if not [point for point in samples if point]:
        return {"error": "only the base model's samples were found: the finder compares the samples of saved checkpoints"}
    references = [path for path in settings.get("references") or [] if os.path.isfile(path)]
    if not references and dataset:
        references = scoring.dataset_files(dataset)
    if not references:
        return {"error": "no training images to compare the samples with (dataset folder empty or not found)"}
    if len(references) > MAX_REFERENCES:  # an even spread of a very large dataset
        step = len(references) / MAX_REFERENCES
        references = [references[int(i * step)] for i in range(MAX_REFERENCES)]
    prompts = settings.get("prompts") or []
    leak = set(settings.get("leakage_indices") or [])
    subject = str(settings.get("subject") or "")
    device = _device(device)
    scorer = scoring.Scorer(device)
    total = sum(len(items) for items in samples.values())
    log(f"Best checkpoint finder: {total} samples of {len(samples)} checkpoints (base model included), {len(references)} training files, {device}")
    log(f"Scoring models: {scoring.models_dir()} (downloaded on first use, about 4.7 GB)")
    references_set = scoring.reference_set(scorer, references)
    log(f"Training set: {references_set['count']} images / frames, {len(references_set['faces'])} with a face")
    cache_path = os.path.join(output_dir, FOLDER, "scores_cache.json")  # by sample file: shared by the folder's runs
    cache = scoring.load_cache(cache_path)
    checkpoints = {}
    try:
        from tqdm import tqdm

        progress = tqdm(samples.items(), total=len(samples), desc="Scoring checkpoints", unit="ckpt", file=sys.stdout, dynamic_ncols=False, ncols=100)
        write = tqdm.write
    except ImportError:
        progress, write = samples.items(), log
    base_main = None
    for point, items in progress:  # the base model (0) comes first
        scored = scoring.score_samples(scorer, items, prompts, references_set, subject, class_word(subject), cache)
        main = [s for s in scored if s["prompt_index"] not in leak and not s.get("error")]
        summary = scoring.summarize(main or [s for s in scored if not s.get("error")])
        if point == 0:
            base_main = main
        elif base_main:
            summary["vs_base"] = {key: change for key, ratio in PAIRED.items() if (change := scoring.paired_change(main, base_main, key, ratio))}
            identity = scoring.identity_margin(main, base_main, references_set.get("mean_face"))
            if identity:
                summary["vs_base"]["identity"] = identity
        leaking = [s.get("face_mean") for s in scored if s["prompt_index"] in leak and s.get("face_mean") is not None]
        summary["leakage_face"] = statistics.median(leaking) if leaking else None
        checkpoints[point] = {"summary": summary, "samples": [{k: v for k, v in s.items() if k not in ("dino_embedding", "face_embedding")} for s in scored]}
        write(
            f"  {point_label(point, kind)}: likeness {_num(summary.get('face_mean') if summary.get('face_mean') is not None else summary.get('dino_nearest_train'))}"
            f" · prompt {_num(summary.get('prompt'))} · variety {_num(summary.get('variety'))}"
        )
        scoring.save_cache(cache_path, cache)
    report = {
        "created": time.time(),
        "output_dir": os.path.abspath(output_dir),
        "output_name": output_name,
        "kind": kind,
        "training_images": references_set["count"],
        "training_faces": len(references_set["faces"]),
        "training_face_area": references_set.get("face_area"),
        "checkpoints": checkpoints,
        "settings": {key: value for key, value in settings.items() if key != "references"},
    }
    result = ranking.rank(report, label=lambda point: point_label(point, kind))
    if result.get("error"):
        return {"error": result["error"]}
    # leakage is a property of the run (no regularization images), not of one checkpoint: one note, not a flag on every row
    base_leak = checkpoints.get(0, {}).get("summary", {}).get("leakage_face")
    for row in result.get("rows", []):
        row["leakage"] = checkpoints[row["epoch"]]["summary"].get("leakage_face")
    leaking = [row for row in result.get("rows", []) if row["leakage"] is not None and base_leak is not None and row["leakage"] > base_leak + LEAK_DELTA]
    if leaking:
        result["reasons"].append(
            f"Without the trigger word, the class prompt looks like your subject from {leaking[0]['label']} on (face match "
            f"{leaking[0]['leakage']:.2f}, base model {base_leak:.2f}); usual without regularization images, so expect it when you generate other people."
        )
    best = result["best_epoch"]
    checkpoint = checkpoint_for(output_dir, output_name, best, kind)
    result["checkpoint"] = os.path.abspath(checkpoint) if checkpoint else None
    result["label"] = point_label(best, kind)
    if not checkpoint:
        result["reasons"].append(f"The checkpoint of {result['label']} is no longer in the output folder (removed by a Save Last N setting?).")
    if partial:
        result["reasons"].append(partial[0].upper() + partial[1:] + "; analyze again once they are all written.")
    report["result"] = result
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    write_html(os.path.join(folder, "report.html"), report)
    return report


def _num(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _times(value) -> str:
    return "—" if value is None else f"×{value:.2f}"


def _flags_cell(row: dict) -> str:
    """The counted signs, then the ones that come and go (reports written before 2026-10-14 count every flag)."""
    signs = row.get("signs", row["flags"])
    passing = [flag for flag in row["flags"] if flag not in signs]
    return html.escape("; ".join(signs)) + (f" <small>comes and goes, not counted: {html.escape('; '.join(passing))}</small>" if passing else "")


def write_html(path: str, report: dict) -> None:
    """A self-contained page: the recommendation and why, the ranked table and the prompt x checkpoint grid (images
    linked relative to the report, so the output folder can be moved as a whole)."""
    result = report["result"]
    kind = report.get("kind", "epoch")
    points = sorted(int(point) for point in report["checkpoints"])
    best = result.get("best_epoch")
    prompts = (report.get("settings") or {}).get("prompts") or []

    def short(label: str) -> str:
        return "base" if label == "base model" else label.replace("epoch ", "e").replace("step ", "s")

    rows = "".join(
        f"<tr class='{'best' if r['epoch'] == best else ''}'><td>{html.escape(r['label'])}{' 🏆' if r['epoch'] == best else ''}</td>"
        f"<td>{_num(r['likeness'])}</td><td>{_num(r['prompt'])}</td><td>{_num(r['variety'])}</td><td>{_times(r.get('framing'))}</td>"
        f"<td>{'✓' if r['plateau'] else ''}</td><td>{_flags_cell(r)}</td></tr>"
        for r in result.get("rows", [])
    )
    by_cell: dict[tuple, dict] = {}
    for point in points:
        for sample in report["checkpoints"][point]["samples"]:
            by_cell.setdefault((sample["prompt_index"], sample.get("seed", "")), {})[point] = sample["path"]
    folder = os.path.dirname(path)
    header = "".join(f"<th class='{'best' if p == best else ''}'>{html.escape(short(point_label(p, kind)))}</th>" for p in points)
    grid = ""
    for (index, seed), cells in sorted(by_cell.items()):
        tds = ""
        for point in points:
            sample = cells.get(point)
            if sample:
                rel = os.path.relpath(sample, folder).replace(os.sep, "/")
                if rel.lower().endswith((".mp4", ".webm", ".mov")):
                    media = f"<video src='{html.escape(rel)}' muted loop preload='metadata' onmouseover='this.play()' onmouseout='this.pause()'></video>"
                else:
                    media = f"<img loading='lazy' src='{html.escape(rel)}'>"
                tds += f"<td class='{'best' if point == best else ''}'><a href='{html.escape(rel)}'>{media}</a></td>"
            else:
                tds += "<td></td>"
        text = prompts[index] if index < len(prompts) else ""
        title = html.escape(text[:400])
        grid += f"<tr><th title='{title}'>#{index + 1}<br><small>seed {html.escape(str(seed))}</small></th>{tds}</tr>"
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in result.get("reasons", []))
    checkpoint = html.escape(result.get("checkpoint") or "(not in the output folder)")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Best checkpoint · {html.escape(report.get('output_name', ''))}</title>
<style>body{{font:15px system-ui,sans-serif;margin:24px;background:#0f1220;color:#e7e9f3}}h1{{font-size:24px;margin:0 0 6px}}
.card{{background:#1a1f35;border:1px solid #3b4370;border-radius:14px;padding:16px;margin:14px 0}}table{{border-collapse:collapse}}
td,th{{padding:5px 9px;border-bottom:1px solid #2a3050;text-align:left;vertical-align:top}}.best{{background:#1f3b2a}}
.grid{{overflow:auto}}.grid img,.grid video{{width:190px;border-radius:8px;display:block}}code{{color:#9fe0a7;word-break:break-all}}
small{{color:#9aa3c7}}</style></head><body>
<h1>🏆 Best checkpoint: {html.escape(result.get('label', ''))}</h1><div class="card"><div><code>{checkpoint}</code></div><ul>{reasons}</ul>
<small>Likeness signal: {html.escape(ranking.LIKENESS_NAMES.get(result.get('likeness_signal', ''), ''))} · {report.get('training_images')} training images / frames, {report.get('training_faces')} with a face</small></div>
<div class="card"><table><tr><th>Checkpoint</th><th>Likeness</th><th>Prompt following</th><th>Variety</th><th>Face size vs base</th><th>Likeness level</th><th>Warning signs</th></tr>{rows}</table></div>
<div class="card grid"><table><tr><th>Prompt</th>{header}</tr>{grid}</table></div></body></html>"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)


def main() -> int:
    parser = argparse.ArgumentParser(description="Find the best checkpoint of a training run from its samples.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--settings", default="")
    parser.add_argument("--dataset", default="", help="training folder (when settings.json lists no training files)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    settings_path = args.settings or os.path.join(run_folder(args.output_dir, args.output_name), "settings.json")
    settings = {}
    if os.path.isfile(settings_path):
        with open(settings_path, encoding="utf-8") as handle:
            settings = json.load(handle)
    try:
        report = analyze(args.output_dir, args.output_name, settings, args.device, args.dataset)
    except Exception as exc:  # noqa: BLE001 - the runner shows this line as the reason
        import traceback

        traceback.print_exc()
        print(f"BEST_CHECKPOINT_ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 1
    if report.get("error"):
        print(f"BEST_CHECKPOINT_ERROR: {report['error']}", flush=True)
        return 1
    result = report["result"]
    print(f"BEST_CHECKPOINT: {result.get('checkpoint') or '(not in the output folder)'} | {result['label']}", flush=True)
    for reason in result.get("reasons", []):
        print(f"BEST_CHECKPOINT_REASON: {reason}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
