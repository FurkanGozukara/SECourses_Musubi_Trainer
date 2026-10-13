"""The best-checkpoint finder's rule (LITERATURE: likeness plateaus early; overtraining shows as lost prompt following,
lost variety, copied training photos, colour / sharpness drift and the training photos' framing, not as lower likeness).

  likeness     median face similarity to the mean training face (DINOv2 similarity to the training images when the
               training images or samples have no faces); its gain over the base model is paired (the base model's
               sample of the same prompt and seed) when the report has ``vs_base``, and smoothed over neighbouring saves
  undertrained the best likeness gains less than LIKENESS_MIN_GAIN over the base model: the latest checkpoint is
               picked and the user is told to train longer
  knee         the likeness level: the first checkpoint whose smoothed gain reaches KNEE_FRACTION of the highest (later
               checkpoints keep creeping up while the model copies more of its training photos) and, with faces, whose
               smoothed identity margin (``vs_base["identity"]``: similarity to the training faces minus similarity to
               the base model's face for the same prompt and seed) does too: a face that only took the subject's glasses
               and hair still scores high against the training faces
  paired       prompt following, colour / sharpness and framing are compared with the base model's sample of the same
               prompt and seed (``summary["vs_base"]``: the median change and the share of pairs that moved that way); a
               change counts when at least AGREE of the pairs agree. The prompts of a run differ far more from each other
               than a checkpoint changes them, so medians over all samples drift with the prompt mix (no base model
               samples: the medians are compared, PROMPT_DROP and z-scores over the checkpoints' spread with DRIFT_MIN)
  flags        prompt following PAIRED_PROMPT_DROP below the base model, variety below VARIETY_RATIO x base, a sample
               nearer than COPY_LIMIT to a training photo, colour / sharpness drift beyond DRIFT_LOG (log ratio; clipped
               highlights / shadows: DRIFT_MIN), faces in fewer samples than FACE_FOUND_DROP x base, faces FRAMING times
               the base model's size in the direction of the training photos' framing (the subject drawn as close, or as
               far, as in its training photos, whatever the prompt asks)
  pick         the onset of over-training is the first checkpoint with a flag. From the knee up to the onset (or,
               when the onset comes first, among the checkpoints before it that reach EARLY_LEVEL of the highest
               likeness): the earliest checkpoint within PICK_TOLERANCE of that window's most likeness (later ones
               add little likeness and copy more of the training photos). An early sign with nothing before it that
               reaches the level (FLUX Klein LoRA: framing x1.31 at the first save, x1.16 at the next): the first stretch
               of level checkpoints without flags. No such checkpoint: the earliest level checkpoint with the fewest
               flags
Thresholds are research values (dont_upload/best_checkpoint_research) and may change with the calibration."""

from __future__ import annotations

import math
import statistics

KNEE_FRACTION = 0.85
EARLY_LEVEL = 0.70  # smoothed likeness gain (share of the highest) a checkpoint before an early onset needs
PICK_TOLERANCE = 0.05  # share of the highest smoothed gain within which the earliest checkpoint is preferred
STILL_RISING = 0.03  # smoothed likeness gain over the last two saves that counts as still learning
PROMPT_DROP = 0.02  # unpaired: median SigLIP2 cosine below the base model's
PAIRED_PROMPT_DROP = 0.01  # paired: median SigLIP2 cosine lost on the same prompts and seeds
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
# paired: face box area / the base model's. Krea 2 (13 photos, 200 epochs): x1.3 where the news anchor and the
# headshots were visibly drawn closer (LoRA epoch 140, DoRA epoch 160); x1.2-1.25 also at checkpoints that still
# looked right (DoRA epoch 100)
FRAMING = 1.3

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


def _smooth(values: list[float]) -> list[float]:
    """Each value weighted 1/2 with 1/4 of each neighbour (2/3 and 1/3 at the ends): saves are noisy, a face match
    moves 0.03-0.06 between them, and a checkpoint's own value still decides between two."""
    if len(values) < 2:
        return list(values)
    inner = [0.25 * values[i - 1] + 0.5 * values[i] + 0.25 * values[i + 1] for i in range(1, len(values) - 1)]
    return [(2 * values[0] + values[1]) / 3, *inner, (2 * values[-1] + values[-2]) / 3]


def _framing_flag(summary: dict, base: dict, training_area: float | None) -> str | None:
    """Faces FRAMING times the base model's size (paired) towards the training photos' own framing."""
    change = _paired(summary, "face_area")
    if not change or change[1] < AGREE:
        return None
    toward = 1.0 if not training_area or not base.get("face_area") or training_area >= base["face_area"] else -1.0
    if toward * change[0] < math.log(FRAMING):
        return None
    word = "closer" if change[0] > 0 else "farther away"
    return f"drawn {word} than the prompts ask, as in the training photos (faces {math.exp(change[0]):.2f}x the base model's size)"


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
    gains = [(_paired(checkpoints[p], like_key) or (likeness[p] - base_like, 1.0))[0] for p in trained]
    smooth = dict(zip(trained, _smooth(gains)))
    undertrained = max(best_like - base_like, max(gains)) < LIKENESS_MIN_GAIN.get(like_key, 0.1)
    top = max(smooth.values())
    knee = trained[-1] if undertrained or top <= 0 else next(p for p in trained if smooth[p] >= KNEE_FRACTION * top)
    identity = [_paired(checkpoints[p], "identity") for p in trained]
    identity_knee = False
    if like_key == "face_mean" and not undertrained and all(identity):
        margins = dict(zip(trained, _smooth([value[0] for value in identity])))
        margin_top = max(margins.values())
        if margin_top > 0:
            knee = max(knee, next(p for p in trained if margins[p] >= KNEE_FRACTION * margin_top))
            identity_knee = True
    unpaired = _unpaired_drift(checkpoints, trained, base)

    rows = []
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
        framing = _framing_flag(s, base, report.get("training_face_area"))
        if framing:
            flags.append(framing)
        if base.get("face_found") and s.get("face_found") is not None and s["face_found"] < FACE_FOUND_DROP * base["face_found"]:
            flags.append(f"faces found in {s['face_found']:.0%} of samples (base model {base['face_found']:.0%})")
        area = _paired(s, "face_area")
        rows.append({"epoch": p, "label": label(p), "likeness": likeness[p], "prompt": prompt[p], "variety": s.get("variety"),
                     "framing": math.exp(area[0]) if area else None, "gain": smooth[p], "plateau": p >= knee, "flags": flags,
                     "score": smooth[p] / top if top > 0 else 0.0})
    onset = next((r for r in rows if r["flags"]), None)
    name = LIKENESS_NAMES.get(like_key, like_key)
    level_line = (f"Likeness levels off from {label(knee)}" + (": the faces match the training photos and have left the base model's own person" if identity_knee else "")
                  + f" ({name} {likeness[knee]:.3f}; base model {base_like:.3f}, highest {best_like:.3f}).")
    if undertrained:
        best = rows[-1]
        reasons = [
            f"Likeness has not risen above the base model yet ({name}: base model {base_like:.3f}, best {best_like:.3f}): the run "
            f"looks undertrained. Train longer or with a higher learning rate; until then the latest checkpoint, {label(best['epoch'])}, is the pick."
        ]
        if onset:
            reasons.append(f"Warning signs from {onset['label']}: " + "; ".join(onset["flags"]) + ".")
    else:
        early = onset is not None and onset["epoch"] <= knee
        isolated = None  # an early sign followed by clean checkpoints on the level
        if early:
            window = [r for r in rows if r["epoch"] < onset["epoch"] and r["gain"] >= EARLY_LEVEL * top]
            if not window:
                for r in rows:
                    if r["epoch"] < knee or (r["flags"] and not window):
                        continue
                    if r["flags"]:
                        break
                    window.append(r)
                if window:
                    isolated, early = onset, False
                    onset = next((r for r in rows if r["epoch"] > window[-1]["epoch"] and r["flags"]), None)
        else:
            window = [r for r in rows if r["epoch"] >= knee and (onset is None or r["epoch"] < onset["epoch"])]
        signs = f"Over-training signs from {onset['label']}: " + "; ".join(onset["flags"]) + "." if onset else ""
        if window:
            peak = max(r["gain"] for r in window)
            best = next(r for r in window if r["gain"] >= peak - PICK_TOLERANCE * top)
            later = "" if best is window[-1] else " (the later ones add little likeness and copy more of the training photos)"
            if early:
                reasons = [
                    signs,
                    f"They start before likeness levels off ({label(knee)}): {label(best['epoch'])} has nearly the most likeness before them"
                    f"{later}. A lower learning rate would reach the full likeness before them.",
                ]
            else:
                reasons = [level_line]
                if isolated:
                    reasons.append(f"A warning sign at {isolated['label']} (" + "; ".join(isolated["flags"]) + f") does not last: {window[0]['label']} shows none.")
                reasons += [signs] if signs else []
                if len(window) > 1:
                    where = "before them" if onset else "on that level, with no over-training signs"
                    reasons.append(f"{label(best['epoch'])} has nearly the most likeness of the {len(window)} checkpoints {where}{later}.")
                else:
                    reasons.append(f"{label(best['epoch'])} is the checkpoint with the full likeness before them." if onset
                                   else f"{label(best['epoch'])} is the only checkpoint on that level.")
        else:  # over-trained from the first saves on: the least of it
            level = [r for r in rows if r["epoch"] >= knee]
            fewest = min(len(r["flags"]) for r in level)
            best = next(r for r in level if len(r["flags"]) == fewest)
            reasons = [
                level_line,
                f"Every checkpoint on that level shows over-training signs; {label(best['epoch'])} is the earliest with the fewest: "
                + "; ".join(best["flags"]) + ". A lower learning rate (or saving more often) would reach the likeness before them.",
            ]
        if onset is None and best is rows[-1] and len(trained) >= 3 and smooth[trained[-1]] - smooth[trained[-3]] >= STILL_RISING:
            reasons.append(f"Likeness was still rising at {label(best['epoch'])} and nothing shows over-training yet: more epochs could add likeness.")
    for row in rows:  # the likeness level as shown: from the knee, or from an earlier pick
        row["plateau"] = row["epoch"] >= min(knee, best["epoch"])
    return {"best_epoch": best["epoch"], "likeness_signal": like_key, "plateau_start": knee, "undertrained": undertrained,
            "onset": onset["epoch"] if onset else None, "reasons": [r[0].upper() + r[1:] for r in reasons if r], "rows": rows}
