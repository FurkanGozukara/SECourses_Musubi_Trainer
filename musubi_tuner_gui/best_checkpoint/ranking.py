"""The best-checkpoint finder's rule (LITERATURE: likeness plateaus early; overtraining shows as lost prompt following,
lost variety, copied training photos, colour / sharpness drift and the training photos' framing, not as lower likeness).

  likeness     median face similarity to the mean training face (DINOv2 similarity to the training images when the
               training images or samples have no faces); its gain over the base model is paired (the base model's
               sample of the same prompt and seed) when the report has ``vs_base``, and smoothed over neighbouring saves
  undertrained the best likeness gains less than LIKENESS_MIN_GAIN over the base model, or (with faces) the faces never
               leave the base model's own person (smoothed identity margin below IDENTITY_SWITCH at every save; Z-Image
               Turbo LoRA at 1e-5 for 100 epochs: face match 0.06 -> 0.33, still the base model's man by eye): the
               latest checkpoint without warning signs among those with LEARNED_SHARE of the highest gain (and of the
               identity margin's rise) is picked (none: the one with the most likeness); the user is told to train longer
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
               nearer than COPY_LIMIT to a training photo, colour / sharpness drift beyond DRIFT_LOG since the first
               save (log ratio; clipped highlights / shadows: DRIFT_MIN; a shift already at the first save is the look of
               the training photos, reported once: Qwen-Image 2.1 samples 40% softer than the base model's from epoch 20
               on, unchanged after), faces in fewer samples than FACE_FOUND_DROP x base, faces FRAMING times
               the base model's size in the direction of the training photos' framing (the subject drawn as close, or as
               far, as in its training photos, whatever the prompt asks)
  pick         the onset of over-training is the first checkpoint with a flag. From the knee up to the onset: the
               earliest checkpoint within PICK_TOLERANCE of that window's most likeness (later ones add little likeness
               and copy more of the training photos). When the onset comes first and its signs do not last (FLUX Klein
               LoRA: framing x1.31 at the first save, x1.16 at the next; ERNIE-Image LoRA: prompt following -0.010 at
               epoch 40 only), the window is the first stretch of level checkpoints without flags; when they last, the
               checkpoints before the onset that reach EARLY_LEVEL of the highest likeness and, with faces, an identity
               margin of IDENTITY_SWITCH. Neither: the earliest level checkpoint with the fewest flags
  recurring    a kind of sign counts only when it recurs at RECUR_SHARE of the saves after its first one, a later save
               counting at NEAR of its limit (drift and framing hover around their limits); one that comes and goes is
               shown but not counted (Mage-Flow LoRA, 200 epochs: faces x1.32-1.37 at epochs 75-100, x1.17 at 125-150,
               x1.47 at 175, x1.05 at 200, the scenes intact; speckle from 125 on, sharpness +54% / +52% / +59% / +47%)
  scene        a note, not a flag: from the first save whose backgrounds are SCENE_NOTE nearer the training data's own
               (paired, ``vs_base["scene_nearest_train"]``) and stay so, the training scene shows in other prompts
Thresholds are research values (dont_upload/best_checkpoint_research) and may change with the calibration."""

from __future__ import annotations

import math
import statistics

KNEE_FRACTION = 0.85
EARLY_LEVEL = 0.70  # smoothed likeness gain (share of the highest) a checkpoint before an early onset needs
PICK_TOLERANCE = 0.05  # share of the highest smoothed gain within which the earliest checkpoint is preferred
STILL_RISING = 0.03  # smoothed likeness gain over the last two saves that counts as still learning
# an undertrained run's pick needs this share of the run's highest smoothed likeness gain and, with faces, of the run's
# own move away from the base model's person (smoothed identity margin, lowest to highest): a checkpoint below either
# has hardly left the base model (Kandinsky 6 LoRA at 3e-4: 2% of the gain at its only flag-free save; Mage-Flow LoRA
# uniform at 3e-5: 41% of a tiny gain at its first save with the margin at its lowest, the base model's men; Mage-Flow
# LoRA at 3e-5 with shift 6: 42% at its judged best, epoch 60, before the speckle). Stable from 0.10 to 0.35
LEARNED_SHARE = 0.25
# smoothed identity margin below which the faces are still the base model's own person: a run must reach it at some
# save, and a checkpoint before an early onset needs it
IDENTITY_SWITCH = 0.10
PROMPT_DROP = 0.02  # unpaired: median SigLIP2 cosine below the base model's
PAIRED_PROMPT_DROP = 0.01  # paired: median SigLIP2 cosine lost on the same prompts and seeds
AGREE = 0.70
VARIETY_RATIO = 0.80
# a sample this similar (DINOv2) to a training image copies it: 0 of 337 judged-acceptable checkpoints of the research
# runs had a sample above 0.80, the LTX 2.3 runs' over-trained ones up to 0.91 (the training clips' scenes). The median
# over the samples, used before, never reached the old 0.92 in any run
COPY_LIMIT = 0.80
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
# a kind of sign counts as over-training when it recurs at this share of the saves after its first one (over-training
# does not undo itself); a later save recurs a drift or framing sign at NEAR of its limit, as those signals hover
# around their limits (FLUX Klein 9B LoRA: faces x1.35 at epoch 60, then x1.28-1.29)
RECUR_SHARE = 0.5
NEAR = 0.9
# paired: the samples' surroundings (DINOv2 patch tokens outside the person) this much nearer the training data's own
# than the base model's samples of the same prompts and seeds: the training scene bleeds into other prompts. A note,
# not a flag: in the LTX 2.3 runs the training clips' garden came together with the likeness and the judged best
# checkpoints carried it (+0.20 to +0.46); photo runs stayed below it
SCENE_NOTE = 0.12

LIKENESS_NAMES = {"face_mean": "face match", "dino_nearest_train": "image similarity"}
DRIFT_NAMES = {"saturation": "saturation", "colourfulness": "colourfulness", "hf_energy": "sharpness", "clipped": "clipped highlights / shadows"}
SIGN_NAMES = {"prompt": "weaker prompt following", "variety": "less variety", "copy": "a copied training image", "framing": "the training photos' framing",
              "faces": "fewer faces found", **{key: f"{name} drift" for key, name in DRIFT_NAMES.items()}}


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


def _drift_flags(summary: dict, base: dict, unpaired: dict, first: dict | None = None, first_label: str = "",
                 near: float = 1.0) -> list[tuple[str, str]]:
    """Colour / sharpness drift as (kind, text). Paired: the change from the base model minus the change at the first
    save (``first``), so only drift that grows with training counts. ``near``: the share of the limits that counts."""
    flags = []
    if any(_paired(summary, key) for key in (*DRIFT_LOG, "clipped")):
        for key, limit in DRIFT_LOG.items():
            change, start = _paired(summary, key), _paired(first or {}, key)
            growth = change[0] - (start[0] if start else 0.0) if change else 0.0
            if change and abs(growth) >= near * limit and change[1] >= AGREE:
                flags.append((key, f"{DRIFT_NAMES[key]} drift ({math.exp(growth) - 1:+.0%} since {first_label or 'the first save'})"))
        change, start = _paired(summary, "clipped"), _paired(first or {}, "clipped")
        if change and change[0] - (start[0] if start else 0.0) >= near * DRIFT_MIN["clipped"] and change[1] >= AGREE:
            flags.append(("clipped", f"{DRIFT_NAMES['clipped']} drift ({change[0] - (start[0] if start else 0.0):+.0%} of the image clipped since "
                                     f"{first_label or 'the first save'})"))
        return flags
    for key, z in unpaired.items():
        if abs(z) > near * Z_LIMIT:
            change = summary[key] - base[key]
            amount = f"{change:+.0%} of the image clipped" if key == "clipped" else f"{change / base[key]:+.0%}"
            flags.append((key, f"{DRIFT_NAMES.get(key, key)} drift ({amount} vs the base model)"))
    return flags


def _with_nearest_max(data: dict) -> dict:
    """The checkpoint summary; reports written before ``dino_nearest_max`` get it from their samples."""
    summary = data["summary"]
    if "dino_nearest_max" not in summary and data.get("samples"):
        values = [s["dino_nearest_train"] for s in data["samples"] if s.get("dino_nearest_train") is not None]
        summary = {**summary, "dino_nearest_max": max(values) if values else None}
    return summary


def _smooth(values: list[float]) -> list[float]:
    """Each value weighted 1/2 with 1/4 of each neighbour (2/3 and 1/3 at the ends): saves are noisy, a face match
    moves 0.03-0.06 between them, and a checkpoint's own value still decides between two."""
    if len(values) < 2:
        return list(values)
    inner = [0.25 * values[i - 1] + 0.5 * values[i] + 0.25 * values[i + 1] for i in range(1, len(values) - 1)]
    return [(2 * values[0] + values[1]) / 3, *inner, (2 * values[-1] + values[-2]) / 3]


def _framing_flag(summary: dict, base: dict, training_area: float | None, near: float = 1.0) -> str | None:
    """Faces FRAMING times the base model's size (paired) towards the training photos' own framing (``near``: the share
    of the log limit that counts)."""
    change = _paired(summary, "face_area")
    if not change or change[1] < AGREE:
        return None
    toward = 1.0 if not training_area or not base.get("face_area") or training_area >= base["face_area"] else -1.0
    if toward * change[0] < near * math.log(FRAMING):
        return None
    word = "closer" if change[0] > 0 else "farther away"
    return f"drawn {word} than the prompts ask, as in the training photos (faces {math.exp(change[0]):.2f}x the base model's size)"


def rank(report: dict, label=_epoch_label) -> dict:
    """``report["checkpoints"]``: point (epoch or step; 0 = the base model) -> {"summary": ...}. ``label`` names a point
    in the reasons and flags."""
    checkpoints = {int(point): _with_nearest_max(data) for point, data in report["checkpoints"].items()}
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
    identity = [_paired(checkpoints[p], "identity") for p in trained]
    margins = dict(zip(trained, _smooth([value[0] for value in identity]))) if like_key == "face_mean" and all(identity) else {}
    still_base = bool(margins) and max(margins.values()) < IDENTITY_SWITCH
    undertrained = undertrained or still_base
    top = max(smooth.values())
    knee = trained[-1] if undertrained or top <= 0 else next(p for p in trained if smooth[p] >= KNEE_FRACTION * top)
    identity_knee = False
    if margins and not undertrained:
        margin_top = max(margins.values())
        knee = max(knee, next(p for p in trained if margins[p] >= KNEE_FRACTION * margin_top))
        identity_knee = True
    unpaired = _unpaired_drift(checkpoints, trained, base)

    def signs_at(p: int, near: float = 1.0) -> list[tuple[str, str]]:
        """(kind, text) of every warning sign at checkpoint ``p``; ``near`` < 1 for the recurrence check."""
        s = checkpoints[p]
        found = []
        change = _paired(s, "prompt")
        if change:
            if change[0] < -PAIRED_PROMPT_DROP and change[1] >= AGREE:
                found.append(("prompt", f"weaker prompt following than the base model ({change[0]:+.3f} on the same prompts and seeds)"))
        elif base.get("prompt") is not None and prompt[p] < base["prompt"] - PROMPT_DROP:
            found.append(("prompt", f"weaker prompt following than the base model ({prompt[p]:.3f} vs {base['prompt']:.3f})"))
        if base.get("variety") and s.get("variety") is not None and s["variety"] < VARIETY_RATIO * base["variety"]:
            found.append(("variety", f"less variety between prompts ({s['variety']:.3f}, base {base['variety']:.3f})"))
        if s.get("dino_nearest_max") is not None and s["dino_nearest_max"] > COPY_LIMIT:
            found.append(("copy", f"a sample copies a training image (similarity {s['dino_nearest_max']:.2f})"))
        found += _drift_flags(s, base, unpaired.get(p, {}), checkpoints[trained[0]], label(trained[0]), near)
        framing = _framing_flag(s, base, report.get("training_face_area"), near)
        if framing:
            found.append(("framing", framing))
        if base.get("face_found") and s.get("face_found") is not None and s["face_found"] < FACE_FOUND_DROP * base["face_found"]:
            found.append(("faces", f"faces found in {s['face_found']:.0%} of samples (base model {base['face_found']:.0%})"))
        return found

    signs = {p: signs_at(p) for p in trained}
    near = {p: {kind for kind, _ in signs_at(p, NEAR)} for p in trained}
    counted = set()  # the kinds of sign that recur at the later saves
    for kind in {kind for found in signs.values() for kind, _ in found}:
        first = next(i for i, p in enumerate(trained) if any(k == kind for k, _ in signs[p]))
        later = trained[first + 1:]
        if not later or sum(kind in near[p] for p in later) >= RECUR_SHARE * len(later):
            counted.add(kind)
    rows = []
    for p in trained:
        s = checkpoints[p]
        area = _paired(s, "face_area")
        rows.append({"epoch": p, "label": label(p), "likeness": likeness[p], "prompt": prompt[p], "variety": s.get("variety"),
                     "framing": math.exp(area[0]) if area else None, "gain": smooth[p], "plateau": p >= knee,
                     "flags": [text for _, text in signs[p]], "signs": [text for kind, text in signs[p] if kind in counted],
                     "score": smooth[p] / top if top > 0 else 0.0})
    passing = {kind: [r["label"] for r in rows if any(k == kind for k, _ in signs[r["epoch"]])]
               for kind in sorted({kind for found in signs.values() for kind, _ in found} - counted)}
    onset = next((r for r in rows if r["signs"]), None)
    name = LIKENESS_NAMES.get(like_key, like_key)
    level_line = (f"Likeness levels off from {label(knee)}" + (": the faces match the training photos and have left the base model's own person" if identity_knee else "")
                  + f" ({name} {likeness[knee]:.3f}; base model {base_like:.3f}, highest {best_like:.3f}).")
    if undertrained:
        # the most trained checkpoint is the closest to the likeness: the latest one without warning signs (Mage-Flow
        # DoRA: noisy textures from epoch 80 on, so epoch 60; HiDream-O1 LoRA: a saturation dip at epoch 80 only, so 100)
        # among those that learned something (LEARNED_SHARE of the highest gain; Kandinsky 6 LoRA at 3e-4: the only
        # clean save, epoch 40, was still the base model's blurry clips at 2%, epoch 200 the only one with a likeness)
        low_margin = min(margins.values()) if margins else 0.0
        span = max(margins.values()) - low_margin if margins else 0.0
        learned = [r for r in rows if (top <= 0 or r["gain"] >= LEARNED_SHARE * top)
                   and (span <= 0 or margins[r["epoch"]] - low_margin >= LEARNED_SHARE * span)]
        learned = learned or [max(reversed(rows), key=lambda r: r["gain"])]  # neither share: the most likeness
        clean = [r for r in learned if not r["signs"]]
        best = clean[-1] if clean else max(reversed(learned), key=lambda r: r["gain"])  # else the most likeness, the latest of equals
        unlearned = [r for r in rows if not r["signs"] and r not in learned]
        if best is rows[-1]:
            latest = "the latest checkpoint"
        elif best["signs"]:
            latest = "the checkpoint with the most likeness"
        elif onset is not None and best["epoch"] < onset["epoch"]:
            latest = "the last checkpoint before the warning signs"
        else:
            latest = "the latest checkpoint without warning signs"
        if still_base:
            reasons = [
                f"The faces are still the base model's own person at {label(rows[-1]['epoch'])}: they resemble its faces for the same "
                f"prompts more than your training photos ({name} {best_like:.3f}, base model {base_like:.3f}). The run looks "
                f"undertrained: train longer or with a higher learning rate; until then {latest}, {label(best['epoch'])}, is the pick."
            ]
        else:
            reasons = [
                f"Likeness has not risen above the base model yet ({name}: base model {base_like:.3f}, best {best_like:.3f}): the run "
                f"looks undertrained. Train longer or with a higher learning rate; until then {latest}, {label(best['epoch'])}, is the pick."
            ]
        if onset and best is rows[-1] and not best["signs"]:
            signed = [r for r in rows if r["signs"]]
            reasons.append(f"Warning signs at {', '.join(r['label'] for r in signed)} (" + "; ".join(onset["signs"])
                           + f") do not last: {label(best['epoch'])} shows none.")
        elif onset:
            reasons.append(f"Warning signs from {onset['label']}: " + "; ".join(onset["signs"]) + "."
                           + (" Training longer will not fix those: try a lower learning rate with more epochs." if best is not rows[-1] else ""))
        # the trade-off named when the likeness signals are this weak (Mage-Flow LoRA uniform at 3e-5: the pick, the last
        # save before the speckle, was still the base model's men by eye; the subject came with the speckle)
        stronger = [r for r in learned if r["epoch"] > best["epoch"] and r["signs"] and r["gain"] > best["gain"] + PICK_TOLERANCE * top]
        if not best["signs"] and stronger:
            alt = max(stronger, key=lambda r: r["gain"])
            reasons.append(f"{label(alt['epoch'])} has more likeness ({alt['gain'] / top:.0%} of the most, {label(best['epoch'])} "
                           f"{best['gain'] / top:.0%}) with the warning signs: compare the two in the grid.")
        if best["signs"] and unlearned:
            shares = f"{max(r['gain'] for r in unlearned) / top:.0%} of the likeness gain"
            if span > 0:
                shares += f", {max(margins[r['epoch']] - low_margin for r in unlearned) / span:.0%} of the move away from its own person"
            reasons.append(f"The checkpoints without them ({', '.join(r['label'] for r in unlearned)}) have hardly moved from the base "
                           f"model (at most {shares}): {label(best['epoch'])} is the pick despite them.")
    else:
        early = onset is not None and onset["epoch"] <= knee
        isolated = None  # an early sign followed by clean checkpoints on the level
        if early:
            # the clean checkpoints with the likeness (EARLY_LEVEL of the most; with faces, past the identity switch:
            # Mage-Flow LoRA's first save had the subject's glasses and hair but was the base model's man; not a share
            # of the highest margin, which keeps rising while a run over-trains, Krea 2 at 2x), in runs of consecutive
            # saves: the longest run, the earliest of equal ones. A sign that comes and goes is noise, but a lone clean
            # save among signs is not their end. FLUX Klein LoRA (framing x1.31 at the first save only) and ERNIE-Image
            # LoRA (prompt following -0.010 at epoch 40 only; epoch 20 had half the likeness): the clean run after the
            # sign. LTX 2.3 GUI run (framing at 50 and 100-175, clean at 75 and at 200, where the training clip's
            # garden replaced the prompts' scenes, which no signal shows): epoch 75
            runs, current = [], []
            for r in rows:
                if not r["signs"] and r["gain"] >= EARLY_LEVEL * top and (not margins or margins[r["epoch"]] >= IDENTITY_SWITCH):
                    current.append(r)
                elif current:
                    runs.append(current)
                    current = []
            if current:
                runs.append(current)
            window = max(runs, key=len) if runs else []
            if window and window[0]["epoch"] > onset["epoch"]:  # after the first sign: it did not last
                isolated, early = onset, False
                onset = next((r for r in rows if r["epoch"] > window[-1]["epoch"] and r["signs"]), None)
        else:
            window = [r for r in rows if r["epoch"] >= knee and (onset is None or r["epoch"] < onset["epoch"])]
        signs_line = f"Over-training signs from {onset['label']}: " + "; ".join(onset["signs"]) + "." if onset else ""
        if window:
            peak = max(r["gain"] for r in window)
            best = next(r for r in window if r["gain"] >= peak - PICK_TOLERANCE * top)
            later = "" if best is window[-1] else " (the later ones add little likeness and copy more of the training photos)"
            if early:
                reasons = [
                    signs_line,
                    f"They start {'as' if onset['epoch'] == knee else 'before'} likeness levels off ({label(knee)}): {label(best['epoch'])} has nearly the most likeness before them"
                    f"{later}. A lower learning rate would reach the full likeness before them.",
                ]
            else:
                reasons = [level_line]
                if isolated:
                    signed = [r for r in rows if r["signs"] and r["epoch"] < window[0]["epoch"]]
                    if len(signed) == 1:
                        reasons.append(f"A warning sign at {isolated['label']} (" + "; ".join(isolated["signs"]) + f") does not last: {window[0]['label']} shows none.")
                    else:
                        reasons.append(f"Warning signs at {', '.join(r['label'] for r in signed)} (from {isolated['label']}: "
                                       + "; ".join(isolated["signs"]) + f"); {window[0]['label']} shows none.")
                reasons += [signs_line] if signs_line else []
                if len(window) > 1:
                    where = "before them" if onset else ("on that level, with no over-training signs" if window[0]["epoch"] >= knee
                                                         else "without warning signs")
                    reasons.append(f"{label(best['epoch'])} has nearly the most likeness of the {len(window)} checkpoints {where}{later}.")
                elif best["epoch"] < knee:
                    reasons.append(f"{label(best['epoch'])} has {best['gain'] / top:.0%} of the most likeness and no warning signs; "
                                   + ("the checkpoints after it carry them." if onset else "the later ones add likeness with warning signs before them."))
                else:
                    reasons.append(f"{label(best['epoch'])} is the checkpoint with the full likeness before them." if onset
                                   else f"{label(best['epoch'])} is the only checkpoint on that level.")
        else:  # over-trained from the first saves on: the least of it
            level = [r for r in rows if r["epoch"] >= knee]
            fewest = min(len(r["signs"]) for r in level)
            best = next(r for r in level if len(r["signs"]) == fewest)
            reasons = [
                level_line,
                f"Every checkpoint on that level shows over-training signs; {label(best['epoch'])} is the earliest with the fewest: "
                + "; ".join(best["signs"]) + ". A lower learning rate (or saving more often) would reach the likeness before them.",
            ]
        if onset is None and best is rows[-1] and len(trained) >= 3 and smooth[trained[-1]] - smooth[trained[-3]] >= STILL_RISING:
            reasons.append(f"Likeness was still rising at {label(best['epoch'])} and nothing shows over-training yet: more epochs could add likeness.")
    scene = {p: _paired(checkpoints[p], "scene_nearest_train") for p in trained}
    scene_from = next((p for i, p in enumerate(trained) if scene[p] and scene[p][0] >= SCENE_NOTE and scene[p][1] >= AGREE
                       and sum(1 for q in trained[i + 1:] if scene[q] and scene[q][0] >= NEAR * SCENE_NOTE) >= RECUR_SHARE * len(trained[i + 1:])), None)
    if scene_from is not None:
        reasons.append(f"From {label(scene_from)} the samples take on the training data's surroundings (backgrounds {scene[scene_from][0]:+.2f} "
                       "nearer them than the base model's): expect that scene in other prompts; an earlier checkpoint keeps the prompts' own.")
    if passing:
        reasons.append("Signs that come and go at the later saves are not counted as over-training: "
                       + "; ".join(f"{SIGN_NAMES.get(kind, kind)} at {', '.join(labels)}" for kind, labels in passing.items()) + ".")
    looks = []
    for key, limit in DRIFT_LOG.items():
        start = _paired(checkpoints[trained[0]], key)
        if start and abs(start[0]) >= limit and start[1] >= AGREE:
            looks.append(f"{DRIFT_NAMES[key]} {math.exp(start[0]) - 1:+.0%}")
    if looks:
        reasons.append(f"From {label(trained[0])} on the samples differ from the base model's look ({', '.join(looks)} on the same prompts): "
                       "the training photos' look, not over-training.")
    for row in rows:  # the likeness level as shown: from the knee, or from an earlier pick
        row["plateau"] = row["epoch"] >= min(knee, best["epoch"])
    return {"best_epoch": best["epoch"], "likeness_signal": like_key, "plateau_start": knee, "undertrained": undertrained,
            "onset": onset["epoch"] if onset else None, "reasons": [r[0].upper() + r[1:] for r in reasons if r], "rows": rows}
