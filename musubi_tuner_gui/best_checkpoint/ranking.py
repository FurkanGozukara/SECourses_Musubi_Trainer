"""The best-checkpoint finder's rule (LITERATURE: likeness plateaus early; overtraining shows as lost prompt following,
lost variety, copied training photos and colour / sharpness drift, not as lower likeness).

  likeness     median face similarity to the mean training face (DINOv2 similarity to the training images when the
               training images or samples have no faces)
  undertrained the best likeness gains less than LIKENESS_MIN_GAIN over the base model: the latest checkpoint is
               picked and the user is told to train longer
  plateau      first checkpoint within PLATEAU_FRACTION of the gap between base and best likeness
  flags        prompt following below the base model, variety below VARIETY_RATIO x base, a sample nearer than
               COPY_LIMIT to a training photo, colour / sharpness drift (z > Z_LIMIT over the checkpoints' spread and a
               real change of at least DRIFT_MIN), fewer faces than FACE_FOUND_DROP x base
  pick         among plateau checkpoints without flags (else those with the fewest flags): the best harmonic mean of
               normalised likeness and prompt following, times min(1, variety ratio); ties -> the earlier one
Thresholds are research values (dont_upload/best_checkpoint_research) and may change with the calibration."""

from __future__ import annotations

import statistics

PLATEAU_FRACTION = 0.10
PROMPT_DROP = 0.02
VARIETY_RATIO = 0.80
COPY_LIMIT = 0.92
Z_LIMIT = 2.0
FACE_FOUND_DROP = 0.75
LIKENESS_MIN_GAIN = {"face_mean": 0.15, "dino_nearest_train": 0.05}
# smallest change from the base model a drift flag needs (relative; "clipped": an absolute rise in the clipped share),
# so a run whose few checkpoints barely differ is not flagged for a large z-score of a tiny spread
DRIFT_MIN = {"saturation": 0.12, "colourfulness": 0.12, "hf_energy": 0.30, "clipped": 0.05}


def _norm(values: dict, key: int) -> float:
    low, high = min(values.values()), max(values.values())
    return 0.0 if high <= low else (values[key] - low) / (high - low)


LIKENESS_NAMES = {"face_mean": "face match", "dino_nearest_train": "image similarity"}
DRIFT_NAMES = {"saturation": "saturation", "colourfulness": "colourfulness", "hf_energy": "sharpness", "clipped": "clipped highlights / shadows"}


def _epoch_label(point: int) -> str:
    return "the base model" if point == 0 else f"epoch {point}"


def _real_change(key: str, value: float, base: float) -> bool:
    if key == "clipped":
        return value - base >= DRIFT_MIN[key]
    return abs(value - base) >= DRIFT_MIN[key] * max(abs(base), 1e-9)


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
    prompt = {p: checkpoints[p].get("prompt") or 0.0 for p in trained}
    base_like = base.get(like_key) if base.get(like_key) is not None else min(likeness.values())
    best_like = max(likeness.values())
    undertrained = best_like - base_like < LIKENESS_MIN_GAIN.get(like_key, 0.1)
    if undertrained:  # no plateau yet: the most trained checkpoint is the closest to one
        plateau_start = trained[-1]
    else:
        threshold = best_like - PLATEAU_FRACTION * (best_like - base_like)
        plateau_start = next((p for p in trained if likeness[p] >= threshold), trained[-1])

    # colour / sharpness drift: z-scores against the trained checkpoints' own spread, anchored at the base model
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

    rows = []
    for p in trained:
        s = checkpoints[p]
        flags = []
        if base.get("prompt") is not None and prompt[p] < base["prompt"] - PROMPT_DROP:
            flags.append(f"weaker prompt following than the base model ({prompt[p]:.3f} vs {base['prompt']:.3f})")
        if base.get("variety") and s.get("variety") is not None and s["variety"] < VARIETY_RATIO * base["variety"]:
            flags.append(f"less variety between prompts ({s['variety']:.3f}, base {base['variety']:.3f})")
        if s.get("dino_nearest_train") is not None and s["dino_nearest_train"] > COPY_LIMIT:
            flags.append(f"copies training photos (similarity {s['dino_nearest_train']:.3f})")
        for key, z in drift.get(p, {}).items():
            if abs(z) > Z_LIMIT:
                change = s[key] - base[key]
                amount = f"{change:+.0%} of the image clipped" if key == "clipped" else f"{change / base[key]:+.0%}"
                flags.append(f"{DRIFT_NAMES.get(key, key)} drift ({amount} vs the base model)")
        if base.get("face_found") and s.get("face_found") is not None and s["face_found"] < FACE_FOUND_DROP * base["face_found"]:
            flags.append(f"faces found in {s['face_found']:.0%} of samples (base model {base['face_found']:.0%})")
        like_n, prompt_n = _norm(likeness, p), _norm(prompt, p)
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
