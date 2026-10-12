"""The best-checkpoint finder's rule (LITERATURE: likeness plateaus early; overtraining shows as lost prompt following,
lost variety, copied training photos and colour / sharpness drift, not as lower likeness).

  likeness     median face similarity to the mean training face (DINOv2 similarity to the training images when the
               training images or samples have no faces)
  undertrained the best likeness gains less than LIKENESS_MIN_GAIN over the base model: the latest checkpoint is
               picked and the user is told to train longer
  plateau      first checkpoint within PLATEAU_FRACTION of the gap between base and best likeness
  paired       prompt following and colour / sharpness are compared with the base model's sample of the same prompt and
               seed (``summary["vs_base"]``: the median change and the share of pairs that moved that way); a change
               counts when at least AGREE of the pairs agree. The prompts of a run differ far more from each other than a
               checkpoint changes them, so medians over all samples drift with the prompt mix (no base model samples:
               the medians are compared, PROMPT_DROP and z-scores over the checkpoints' spread with DRIFT_MIN)
  flags        prompt following PAIRED_PROMPT_DROP below the base model, variety below VARIETY_RATIO x base, a sample
               nearer than COPY_LIMIT to a training photo, colour / sharpness drift beyond DRIFT_LOG (log ratio; clipped
               highlights / shadows: DRIFT_MIN), faces in fewer samples than FACE_FOUND_DROP x base
  pick         among plateau checkpoints without flags (else those with the fewest flags): the best harmonic mean of
               likeness (0 = base model, 1 = highest) and prompt following (1 = the best checkpoint's, 0 = PROMPT_SCALE
               below it), times min(1, variety ratio); ties -> the earlier one
Thresholds are research values (dont_upload/best_checkpoint_research) and may change with the calibration."""

from __future__ import annotations

import math
import statistics

PLATEAU_FRACTION = 0.10
PROMPT_DROP = 0.02  # unpaired: median SigLIP2 cosine below the base model's
PAIRED_PROMPT_DROP = 0.01  # paired: median SigLIP2 cosine lost on the same prompts and seeds
PROMPT_SCALE = 0.03  # prompt following this far below the best checkpoint's scores 0 in the pick
AGREE = 0.70
VARIETY_RATIO = 0.80
COPY_LIMIT = 0.92
Z_LIMIT = 2.0
FACE_FOUND_DROP = 0.75
LIKENESS_MIN_GAIN = {"face_mean": 0.15, "dino_nearest_train": 0.05}
# unpaired: the smallest change from the base model a drift flag needs (relative; "clipped": an absolute rise in the
# clipped share, also used paired), so a run whose few checkpoints barely differ is not flagged for a large z-score
DRIFT_MIN = {"saturation": 0.12, "colourfulness": 0.12, "hf_energy": 0.30, "clipped": 0.05}
# paired: |median log ratio| to the base model's sample of the same prompt and seed
DRIFT_LOG = {"saturation": math.log(1.20), "colourfulness": math.log(1.20), "hf_energy": math.log(1.50)}

LIKENESS_NAMES = {"face_mean": "face match", "dino_nearest_train": "image similarity"}
DRIFT_NAMES = {"saturation": "saturation", "colourfulness": "colourfulness", "hf_energy": "sharpness", "clipped": "clipped highlights / shadows"}


def _epoch_label(point: int) -> str:
    return "the base model" if point == 0 else f"epoch {point}"


def _real_change(key: str, value: float, base: float) -> bool:
    if key == "clipped":
        return value - base >= DRIFT_MIN[key]
    return abs(value - base) >= DRIFT_MIN[key] * max(abs(base), 1e-9)


def _paired(summary: dict, key: str) -> tuple[float, float] | None:
    change = (summary.get("vs_base") or {}).get(key)
    return (float(change[0]), float(change[1])) if change and change[0] is not None else None


def _unpaired_drift(checkpoints: dict, trained: list[int], base: dict) -> dict:
    """point -> {key: z} for the colour / sharpness medians of runs without paired changes."""
    drift = {}
    for key in ("saturation", "colourfulness", "hf_energy", "clipped"):
        values = [checkpoints[p].get(key) for p in trained if checkpoints[p].get(key) is not None]
        if len(values) < 3 or base.get(key) is None:
            continue
        spread = statistics.pstdev(values + [base[key]]) or 1e-9
        for p in trained:
            value = checkpoints[p].get(key)
            if value is not None and _real_change(key, value, base[key]):
                drift.setdefault(p, {})[key] = (value - base[key]) / spread
    return drift


def _drift_flags(summary: dict, base: dict, unpaired: dict) -> list[str]:
    flags = []
    if any(_paired(summary, key) for key in (*DRIFT_LOG, "clipped")):
        for key, limit in DRIFT_LOG.items():
            change = _paired(summary, key)
            if change and abs(change[0]) >= limit and change[1] >= AGREE:
                flags.append(f"{DRIFT_NAMES[key]} drift ({math.exp(change[0]) - 1:+.0%} vs the base model on the same prompts)")
        change = _paired(summary, "clipped")
        if change and change[0] >= DRIFT_MIN["clipped"] and change[1] >= AGREE:
            flags.append(f"{DRIFT_NAMES['clipped']} drift ({change[0]:+.0%} of the image clipped vs the base model)")
        return flags
    for key, z in unpaired.items():
        if abs(z) > Z_LIMIT:
            change = summary[key] - base[key]
            amount = f"{change:+.0%} of the image clipped" if key == "clipped" else f"{change / base[key]:+.0%}"
            flags.append(f"{DRIFT_NAMES.get(key, key)} drift ({amount} vs the base model)")
    return flags


def rank(report: dict, label=_epoch_label) -> dict:
    """``report["checkpoints"]``: point (epoch or step; 0 = the base model) -> {"summary": ...}. ``label`` names a point
    in the reasons and flags."""
    checkpoints = {int(point): data["summary"] for point, data in report["checkpoints"].items()}
    points = sorted(checkpoints)
    base_point = points[0] if points and points[0] == 0 else None
    base = checkpoints.get(base_point, {}) if base_point is not None else {}
    trained = [p for p in points if p != base_point]
    if not trained:
        return {"error": "no checkpoint samples (only the base model's)"}
    use_faces = report.get("training_faces", 0) > 0 and all(checkpoints[p].get("face_mean") is not None for p in trained)
    like_key = "face_mean" if use_faces else "dino_nearest_train"
    likeness = {p: checkpoints[p].get(like_key) or 0.0 for p in trained}
    prompt = {}  # paired: the base model's prompt following plus the median change on the same prompts and seeds
    for p in trained:
        change = _paired(checkpoints[p], "prompt")
        prompt[p] = base["prompt"] + change[0] if change and base.get("prompt") is not None else checkpoints[p].get("prompt") or 0.0
    base_like = base.get(like_key) if base.get(like_key) is not None else min(likeness.values())
    best_like = max(likeness.values())
    undertrained = best_like - base_like < LIKENESS_MIN_GAIN.get(like_key, 0.1)
    if undertrained:  # no plateau yet: the most trained checkpoint is the closest to one
        plateau_start = trained[-1]
    else:
        threshold = best_like - PLATEAU_FRACTION * (best_like - base_like)
        plateau_start = next((p for p in trained if likeness[p] >= threshold), trained[-1])
    unpaired = _unpaired_drift(checkpoints, trained, base)

    rows = []
    best_prompt = max(prompt.values())
    for p in trained:
        s = checkpoints[p]
        flags = []
        change = _paired(s, "prompt")
        if change:
            if change[0] < -PAIRED_PROMPT_DROP and change[1] >= AGREE:
                flags.append(f"weaker prompt following than the base model ({change[0]:+.3f} on the same prompts and seeds)")
        elif base.get("prompt") is not None and prompt[p] < base["prompt"] - PROMPT_DROP:
            flags.append(f"weaker prompt following than the base model ({prompt[p]:.3f} vs {base['prompt']:.3f})")
        if base.get("variety") and s.get("variety") is not None and s["variety"] < VARIETY_RATIO * base["variety"]:
            flags.append(f"less variety between prompts ({s['variety']:.3f}, base {base['variety']:.3f})")
        if s.get("dino_nearest_train") is not None and s["dino_nearest_train"] > COPY_LIMIT:
            flags.append(f"copies training photos (similarity {s['dino_nearest_train']:.3f})")
        flags += _drift_flags(s, base, unpaired.get(p, {}))
        if base.get("face_found") and s.get("face_found") is not None and s["face_found"] < FACE_FOUND_DROP * base["face_found"]:
            flags.append(f"faces found in {s['face_found']:.0%} of samples (base model {base['face_found']:.0%})")
        gap = best_like - base_like
        like_n = min(1.0, max(0.0, (likeness[p] - base_like) / gap)) if gap > 0 else 1.0
        prompt_n = min(1.0, max(0.0, 1.0 - (best_prompt - prompt[p]) / PROMPT_SCALE))
        harmonic = 0.0 if like_n + prompt_n == 0 else 2 * like_n * prompt_n / (like_n + prompt_n)
        variety_ratio = (s.get("variety") or 0) / base["variety"] if base.get("variety") else 1.0
        rows.append({"epoch": p, "label": label(p), "likeness": likeness[p], "prompt": prompt[p], "variety": s.get("variety"),
                     "plateau": p >= plateau_start, "flags": flags, "score": harmonic * min(1.0, variety_ratio)})
    level = [r for r in rows if r["plateau"]]
    fewest = min(len(r["flags"]) for r in level)
    candidates = [r for r in level if len(r["flags"]) == fewest]
    best = max(candidates, key=lambda r: (round(r["score"], 3), -r["epoch"]))
    name = LIKENESS_NAMES.get(like_key, like_key)
    if undertrained:
        reasons = [
            f"Likeness has not risen above the base model yet ({name}: base model {base_like:.3f}, best {best_like:.3f}): the run "
            f"looks undertrained. Train longer or with a higher learning rate; until then the latest checkpoint, {label(best['epoch'])}, is the pick."
        ]
    else:
        reasons = [
            f"Likeness levels off from {label(plateau_start)} ({name} {likeness[plateau_start]:.3f}; "
            f"base model {base_like:.3f}, highest {best_like:.3f})."
        ]
        if len(candidates) > 1:
            reasons.append(f"Of the {len(candidates)} checkpoints on that level, {label(best['epoch'])} balances likeness and prompt following best.")
    if best["flags"]:
        reasons.append("Every checkpoint on the likeness level shows a warning sign; this one the fewest: " + "; ".join(best["flags"]) + ".")
    later_flagged = [r for r in rows if r["epoch"] > best["epoch"] and r["flags"]]
    if later_flagged:
        first = later_flagged[0]
        reasons.append(f"Over-training signs from {first['label']}: " + "; ".join(first["flags"]) + ".")
    return {"best_epoch": best["epoch"], "likeness_signal": like_key, "plateau_start": plateau_start, "undertrained": undertrained,
            "reasons": reasons, "rows": rows}
