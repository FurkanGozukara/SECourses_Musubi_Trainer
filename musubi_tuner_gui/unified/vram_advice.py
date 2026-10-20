"""What a settings change does to VRAM, compared with the preset (or defaults) it started from.

Shipped presets are measured: the file name carries the GPU tier and the peak VRAM a real run used
(``24_GB_GPUs_21.6_GB_...``). When the user changes a setting that moves VRAM (resolution, trained
frames, frame rate, batch size, control / edit images, block swap, checkpointing, quantized base
weights, LoRA rank, optimizer), ``advise`` lists each change with its direction and a rough size;
``advice_html`` adds how full the GPU already was with the preset, so a change that will not fit is
flagged before Start. The sizes are rough ratios of what grows, never a VRAM estimate.
"""

from __future__ import annotations

import html
import os

from .values import ValueProblem, parse_bool, parse_number, parse_numbers

MORE, LESS = "more", "less"
BIG, SMALL = "big", "small"

# on/off settings: (key, size, text when turning it off raises VRAM, text when turning it on lowers it)
SWITCHES = (
    ("gradient_checkpointing", BIG, "Gradient checkpointing off: every block keeps its activations (usually many GB more)", "Gradient checkpointing on: activations are recomputed (less VRAM, slower steps)"),
    ("gradient_checkpointing_cpu_offload", BIG, "Activation offloading off: checkpointed activations stay in VRAM", "Activation offloading on: checkpointed activations wait in CPU RAM"),
    ("cpu_offload_checkpointing", BIG, "CPU offload checkpointing off: activations stay in VRAM", "CPU offload checkpointing on: activations wait in CPU RAM"),
    ("blockwise_checkpointing", BIG, "Blockwise checkpointing off: more activations stay in VRAM", "Blockwise checkpointing on: less activation memory"),
    ("full_bf16", BIG, "Full BF16 off: weights and gradients in FP32 (much more VRAM in fine-tuning)", "Full BF16 on: half-size weights and gradients"),
    ("offload_inactive_dit", BIG, "Inactive expert offloading off: both Wan 2.2 experts stay in VRAM", "Inactive expert offloading on: the idle expert waits in CPU RAM"),
    ("img_in_txt_in_offloading", SMALL, "Input-layer offloading off: those layers stay in VRAM (a little more)", "Input-layer offloading on: a little less VRAM"),
    ("split_attn", SMALL, "Split attention off: attention runs on the whole batch at once", "Split attention on: less attention memory"),
    # LTX 2.3 Gemma text caching layer by layer (--low_vram_text_encoding): less than half the VRAM; the 10 and 12 GB
    # presets need it
    ("caching_teo_low_vram", BIG, "Low-VRAM text caching off: Gemma text caching needs about twice the VRAM", "Low-VRAM text caching on: Gemma text caching needs less than half the VRAM (slower caching)"),
)
# a quantized base model (any of these on) holds about half the weight memory of BF16; swapping one kind for
# another (FP8 for ConvRot INT8) changes little
QUANTIZED_BASE_KEYS = ("fp8_base", "fp8_base_unet", "nf4_base", "int8_convrot_base", "convrot_int8", "fp8_w8a8")
# trained frames per clip, by trainer: (keys always used, keys used only with "full" extraction, extraction key)
FRAME_RULES = {
    "wan": (("target_frames",), ("max_frames",), "frame_extraction"),
    "kandinsky6": (("num_frames",), ("max_frames",), "frame_extraction"),
    "ltx2": (("dataset_num_frames",), ("dataset_max_frames",), "dataset_frame_extraction"),
    "minimax_h3": (("dataset_num_frames",), ("dataset_max_frames",), "dataset_frame_extraction"),
}
CONTROL_SIZE_KEYS = (
    ("control_resolution_width", "control_resolution_height"),
    ("dataset_qwen_image_edit_control_resolution_width", "dataset_qwen_image_edit_control_resolution_height"),
)
# bytes of optimizer state per trained parameter (fp32 Adam moments: 8; 8-bit: 2; factored states: about 0)
OPTIMIZER_STATE_BYTES = (
    ("lion8bit", 1), ("8bit", 2), ("adafactor", 0.1), ("came", 0.2), ("prodigy", 16), ("dadapt", 12),
    ("lion", 4), ("adamw", 8), ("adam", 8), ("sgdnesterov", 4), ("sgd", 4),
)


def _number(value):
    try:
        return parse_number(value, whole=False)
    except (ValueProblem, TypeError):
        return None


def _flag(value) -> bool | None:
    try:
        return parse_bool(value)
    except (ValueProblem, TypeError):
        return None


def _largest(value):
    try:
        numbers = parse_numbers(value) if isinstance(value, (str, list, tuple)) else [parse_number(value)]
    except (ValueProblem, TypeError):
        return None
    numbers = [number for number in numbers if number is not None]
    return max(numbers) if numbers else None


def _ratio_text(ratio: float, countable: bool = True) -> str:
    """"≈1.7×" / "+4%" more, "≈2.0× fewer" (tokens) or "≈2.0× less" (memory)."""
    if ratio >= 1:
        return f"+{(ratio - 1) * 100:.0f}%" if ratio < 1.15 else f"≈{ratio:.1f}×"
    if ratio > 0.87:
        return f"−{(1 - ratio) * 100:.0f}%"
    return f"≈{1 / ratio:.1f}× {'fewer' if countable else 'less'}"


def optimizer_state_bytes(name: str) -> float | None:
    lowered = str(name or "").lower().replace("_", "").replace("-", "")
    return next((size for key, size in OPTIMIZER_STATE_BYTES if key in lowered), None)


def advise(backend: str, baseline: dict, current: dict, mode: str = "lora") -> list:
    """(direction "more" | "less", size "big" | "small", text) for every change that moves VRAM."""
    changes = []

    def add(direction, size, text):
        changes.append((direction, size, text))

    width0, height0 = _number(baseline.get("dataset_resolution_width")), _number(baseline.get("dataset_resolution_height"))
    width, height = _number(current.get("dataset_resolution_width")), _number(current.get("dataset_resolution_height"))
    if width0 and height0 and width and height:
        ratio = (width * height) / (width0 * height0)
        if abs(ratio - 1) >= 0.04:
            tail = "activation memory grows about as much" if ratio > 1 else "less activation memory"
            add(MORE if ratio > 1 else LESS, BIG, f"Resolution {width0:g}×{height0:g} → {width:g}×{height:g}: {_ratio_text(ratio)} image tokens per step; {tail}")
    controls_used = (
        (backend == "qwen_image" and str(current.get("model_version") or "").startswith("edit"))
        or backend == "flux"
        or bool(_flag(current.get("use_control_images")))
    )
    for wide_key, high_key in CONTROL_SIZE_KEYS:
        w0, h0, w1, h1 = (_number(source.get(key)) for source in (baseline, current) for key in (wide_key, high_key))
        if controls_used and w0 and h0 and w1 and h1 and abs((w1 * h1) / (w0 * h0) - 1) >= 0.04:
            ratio = (w1 * h1) / (w0 * h0)
            when = " (with control images in the dataset)" if backend == "flux" else ""
            add(MORE if ratio > 1 else LESS, BIG, f"Control image size {w0:g}×{h0:g} → {w1:g}×{h1:g}: {_ratio_text(ratio)} control tokens per step{when}")
    frame_keys, full_keys, extraction_key = FRAME_RULES.get(backend, ((), (), ""))
    full = str(current.get(extraction_key) or "").strip().lower() == "full"
    for key in frame_keys + (full_keys if full else ()):
        before, after = _largest(baseline.get(key)), _largest(current.get(key))
        if before and after and before != after:
            ratio = after / before
            add(MORE if ratio > 1 else LESS, BIG, f"Frames {before:g} → {after:g}: {_ratio_text(ratio)} video tokens per clip")
    before, after = _number(baseline.get("dataset_target_fps")), _number(current.get("dataset_target_fps"))
    if before and after and before != after:
        if after > before:
            add(MORE, SMALL, f"Frame rate {before:g} → {after:g} fps: clips shorter than the frame limit become more frames (more video tokens)")
        else:
            add(LESS, SMALL, f"Frame rate {before:g} → {after:g} fps: fewer frames per second of video")
    before, after = _number(baseline.get("dataset_batch_size")), _number(current.get("dataset_batch_size"))
    if before and after and before != after:
        add(MORE if after > before else LESS, BIG, f"Batch size {before:g} → {after:g}: {_ratio_text(after / before, countable=False)} activation memory")
    before, after = _number(baseline.get("blocks_to_swap")), _number(current.get("blocks_to_swap"))
    if before is not None and after is not None and before != after:
        moved = abs(after - before)
        if after < before:
            add(MORE, BIG, f"Block swap {before:g} → {after:g}: {moved:g} more blocks stay in VRAM (faster, more VRAM)")
        else:
            add(LESS, BIG, f"Block swap {before:g} → {after:g}: {moved:g} more blocks wait in CPU RAM (less VRAM, slower steps)")
    before, after = _number(baseline.get("text_encoder_blocks_to_swap")), _number(current.get("text_encoder_blocks_to_swap"))
    if before is not None and after is not None and before != after:
        add(MORE if after < before else LESS, SMALL, f"Text encoder block swap {before:g} → {after:g}: changes VRAM only while caching text and sampling")
    for key, size, more_text, less_text in SWITCHES:
        before, after = _flag(baseline.get(key)), _flag(current.get(key))
        if key in baseline and key in current and before is not None and after is not None and before != after:
            if key == "full_bf16" and mode != "full":
                continue
            add(MORE if not after else LESS, size, more_text if not after else less_text)
    present = [key for key in QUANTIZED_BASE_KEYS if key in baseline and key in current]
    if present:
        before = any(_flag(baseline.get(key)) for key in present)
        after = any(_flag(current.get(key)) for key in present)
        if before and not after:
            add(MORE, BIG, "Base model no longer quantized (FP8 / INT8 / NF4 off): it loads in BF16, about twice the weight memory")
        elif after and not before:
            add(LESS, BIG, "Base model quantized (FP8 / INT8 / NF4): about half the weight memory")
    before, after = _flag(baseline.get("use_control_images")), _flag(current.get("use_control_images"))
    if "use_control_images" in baseline and "use_control_images" in current and before is not None and after is not None and before != after:
        if backend == "moss_image":  # a control adapter: the maps widen each image token's input, no tokens are added
            add(MORE if after else LESS, SMALL, "Control adapter on: the two maps widen each image token's input and the text refiner's two blocks train too (no extra tokens)" if after else "Control adapter off")
        else:
            add(MORE if after else LESS, BIG, "Control images on: each control image adds about as many tokens as the target image (≈2× with one)" if after else "Control images off: no control-image tokens")
    for key in ("no_resize_control", "dataset_qwen_image_edit_no_resize_control"):
        before, after = _flag(baseline.get(key)), _flag(current.get(key))
        if controls_used and key in baseline and key in current and before is not None and after is not None and before != after:
            add(MORE if after else LESS, BIG, "Controls at their own size: large reference images add many tokens" if after else "Controls resized: their token count is capped")
    before, after = str(baseline.get("model_version") or ""), str(current.get("model_version") or "")
    if backend == "qwen_image" and before != after and (before.startswith("edit") or after.startswith("edit")):
        if after.startswith("edit") and not before.startswith("edit"):
            add(MORE, BIG, f"{after}: control images join every step (≈2× image tokens with one control image)")
        elif before.startswith("edit") and not after.startswith("edit"):
            add(LESS, BIG, f"{after}: no control-image tokens")
    before, after = str(baseline.get("task") or ""), str(current.get("task") or "")
    if backend == "hidream_o1" and before != after and "i2i" in (before, after):
        add(MORE if after == "i2i" else LESS, BIG, "Image-to-image: the source image joins every step" if after == "i2i" else "Text-to-image: no source-image tokens")
    before, after = _number(baseline.get("network_dim")), _number(current.get("network_dim"))
    if before and after and before != after and mode != "full":
        add(MORE if after > before else LESS, SMALL, f"LoRA rank {before:g} → {after:g}: {'more' if after > before else 'fewer'} trainable weights and optimizer state (a small share of VRAM)")
    name_before, name_after = str(baseline.get("optimizer_type") or ""), str(current.get("optimizer_type") or "")
    state_before, state_after = optimizer_state_bytes(name_before), optimizer_state_bytes(name_after)
    if name_before != name_after and state_before is not None and state_after is not None and state_before != state_after:
        size = BIG if mode == "full" else SMALL
        per = "trained parameter" if mode == "full" else "LoRA parameter"
        add(MORE if state_after > state_before else LESS, size, f"Optimizer {name_before} → {name_after}: about {state_after:g} instead of {state_before:g} bytes of state per {per}")
    return changes


def preset_vram(path: str | None) -> tuple:
    """(GPU tier GB, measured peak GB) from a shipped preset's file name, else (None, None)."""
    from ..preset_library import TIER_PATTERN

    match = TIER_PATTERN.search(os.path.basename(path or ""))
    if not match:
        return None, None
    return float(match.group("tier")), float(match.group("used"))


def advice_html(changes: list, preset_path: str | None, gpu: tuple | None) -> str:
    """The VRAM panel: nothing when no change moves VRAM."""
    if not changes:
        return ""
    tier, peak = preset_vram(preset_path)
    gpu_gb = gpu[1] if gpu else None
    raised = [change for change in changes if change[0] == MORE]
    big_raise = any(size == BIG for _direction, size, _text in raised)
    level = "warn" if big_raise else "info"
    headline = "These changes raise VRAM use." if raised else "These changes lower VRAM use."
    if peak and gpu_gb:
        free = gpu_gb - peak
        if free < 0:
            level = "bad" if raised else "warn"
            headline = f"This preset is made for {tier:g} GB GPUs (peak {peak:g} GB) and your GPU has {gpu_gb:g} GB." + (
                " These changes raise VRAM use further." if raised else " These changes lower VRAM use, but compare with a preset for your GPU."
            )
        elif big_raise and peak >= gpu_gb * 0.85:
            level = "bad"
            headline = (
                f"This preset already peaked at {peak:g} GB on your {gpu_gb:g} GB GPU ({free:.1f} GB left): with these changes it will likely run "
                "out of memory. Raise Block Swap, keep FP8 / checkpointing on, or pick a preset for a larger GPU."
            )
        elif raised:
            headline = f"These changes raise VRAM use. The preset peaked at {peak:g} GB; your GPU has {gpu_gb:g} GB ({free:.1f} GB to spare)."
    elif raised and peak:
        headline += f" The preset peaked at {peak:g} GB."
    source = f"the {tier:g} GB preset (measured peak {peak:g} GB)" if peak else ("the loaded preset" if preset_path else "the defaults")
    rows = "".join(
        f'<li class="vr-{direction}"><span>{"▲" if direction == MORE else "▼"}</span>{html.escape(text)}</li>'
        for direction, _size, text in sorted(changes, key=lambda change: (change[0] != MORE, change[1] != BIG))
    )
    return (
        f'<div class="vram-box vram-{level}"><div class="vram-head">⚡ VRAM · compared with {html.escape(source)}</div>'
        f'<div class="vram-headline">{html.escape(headline)}</div><ul>{rows}</ul></div>'
    )


WATCHED_KEYS = frozenset(
    {"dataset_resolution_width", "dataset_resolution_height", "dataset_target_fps", "dataset_batch_size", "blocks_to_swap",
     "text_encoder_blocks_to_swap", "model_version", "task", "network_dim", "optimizer_type", "use_control_images",
     "no_resize_control", "dataset_qwen_image_edit_no_resize_control", "frame_extraction", "dataset_frame_extraction"}
    | {key for keys, full_keys, _extraction in FRAME_RULES.values() for key in keys + full_keys}
    | {key for pair in CONTROL_SIZE_KEYS for key in pair}
    | {switch[0] for switch in SWITCHES}
    | set(QUANTIZED_BASE_KEYS)
)
