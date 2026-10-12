"""The samples a training run wrote for each checkpoint, and the checkpoint file each one belongs to.

musubi and sd-scripts write ``{output_dir}/sample/{output_name}_{point}_{index:02d}_{YYYYmmddHHMMSS}[_{seed}]...`` where
``point`` is ``e{epoch:06d}`` for samples at the end of an epoch and ``{step:06d}`` for samples every N steps (the
``e000000`` set comes from ``--sample_at_first``: the base model). Checkpoints are ``{output_name}-{epoch:06d}`` /
``{output_name}-step{step:08d}`` ``.safetensors`` plus ``{output_name}.safetensors`` at the end, which holds the last
epoch or step."""

from __future__ import annotations

import glob
import os
import re

SAMPLE_RE = re.compile(r"^_(e?)(\d{6,})_(\d{2,})_\d{14}(?:_(\d+))?")
SAMPLE_RE_ANYWHERE = re.compile(SAMPLE_RE.pattern[1:])
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXT = (".mp4", ".webm", ".mov")
EPOCH, STEP = "epoch", "step"


def index_samples(output_dir: str, output_name: str = "") -> dict[int, list[dict]]:
    """point -> [{path, prompt_index, seed, video, kind}] for the run's samples (only ``output_name``'s when given).
    The points are epochs, or steps for a run that sampled every N steps (``kind``); 0 is the base model."""
    found: dict[int, list[dict]] = {}
    kinds: dict[int, str] = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(output_dir), "sample", "*"))):
        name = os.path.basename(path)
        if output_name and not name.startswith(output_name + "_"):
            continue
        lower = name.lower()
        video = lower.endswith(VIDEO_EXT)
        if not (video or lower.endswith(IMAGE_EXT)):
            continue
        match = SAMPLE_RE.match(name[len(output_name):]) if output_name else SAMPLE_RE_ANYWHERE.search(name)
        if not match:
            continue
        point = int(match.group(2))
        kind = EPOCH if match.group(1) or point == 0 else STEP
        found.setdefault(point, []).append(
            {"path": path, "prompt_index": int(match.group(3)), "seed": match.group(4) or "", "video": video, "kind": kind}
        )
        kinds[point] = kind
    # a run samples by epoch or by step; a mixed folder (settings changed between runs) keeps the larger set
    epochs = [p for p, k in kinds.items() if k == EPOCH and p]
    steps = [p for p, k in kinds.items() if k == STEP]
    if epochs and steps:
        drop = STEP if len(epochs) >= len(steps) else EPOCH
        found = {p: items for p, items in found.items() if p == 0 or kinds[p] != drop}
    for items in found.values():
        items.sort(key=lambda item: (item["prompt_index"], item["seed"], item["path"]))
    return dict(sorted(found.items()))


def point_kind(samples: dict[int, list[dict]]) -> str:
    return STEP if any(item["kind"] == STEP for items in samples.values() for item in items) else EPOCH


def point_label(point: int, kind: str = EPOCH) -> str:
    if point == 0:
        return "base model"
    return f"epoch {point}" if kind == EPOCH else f"step {point:,}"


def checkpoint_for(output_dir: str, output_name: str, point: int, kind: str = EPOCH) -> str | None:
    """The saved weights that produced a point's samples: ``name-000040.safetensors`` / ``name-step00000500``, else the
    final ``name.safetensors`` when the point is later than every numbered checkpoint (the trainer saves the last epoch
    or step under the plain name)."""
    if point <= 0:
        return None
    numbered = os.path.join(output_dir, f"{output_name}-{point:06d}.safetensors" if kind == EPOCH else f"{output_name}-step{point:08d}.safetensors")
    if os.path.isfile(numbered):
        return numbered
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    pattern = r"-(\d{6})\.safetensors$" if kind == EPOCH else r"-step(\d{8})\.safetensors$"
    later = [
        int(match.group(1))
        for match in (re.search(pattern, path) for path in glob.glob(os.path.join(glob.escape(output_dir), f"{glob.escape(output_name)}-*.safetensors")))
        if match
    ]
    if os.path.isfile(final) and all(point > value for value in later):
        return final
    return None


def checkpoint_for_epoch(output_dir: str, output_name: str, epoch: int) -> str | None:
    return checkpoint_for(output_dir, output_name, epoch, EPOCH)
