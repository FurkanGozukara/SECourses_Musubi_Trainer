"""Shared optimizer choices, guidance, and launch-time compatibility checks."""

from __future__ import annotations

import ast
import shlex
from collections.abc import Callable, Mapping

AUTOMAGIC_OPTIMIZER_CHOICES = ("Automagic", "Automagic2", "Automagic3")

_OPTIMIZER_GUIDANCE = {
    "adamw": (
        "**AdamW:** Standard full-precision PyTorch optimizer state. The selected learning-rate scheduler controls its learning rate."
    ),
    "adamw8bit": (
        "**AdamW 8-bit:** Bitsandbytes AdamW with lower optimizer-state memory. It requires a working bitsandbytes installation; "
        "the selected learning-rate scheduler remains active."
    ),
    "adafactor": (
        "**Adafactor:** Factored second-moment state reduces optimizer memory for large matrices. Its behavior depends on the supplied "
        "optimizer arguments; the existing model preset remains the safest starting point."
    ),
    "automagic": (
        "**Automagic v1 (experimental):** Uses an Adafactor-style second moment plus an 8-bit per-element adaptive learning-rate mask. "
        "The Learning Rate field is the starting rate and external LR schedulers are bypassed. It supports normal optimizer-step training, "
        "gradient accumulation, and trainer gradient clipping, but uses the most adaptive state of the three versions. Useful arguments: "
        "`min_lr`, `max_lr`, `lr_bump`, `beta2`, `clip_threshold`, and `weight_decay`. Adafactor-only preset arguments (`scale_parameter`, "
        "`relative_step`, and `warmup_init`) are ignored automatically. Keep the separate Fused Backward Pass option off; that option is "
        "only for Adafactor. LoRA block swapping and full fine-tune block swapping are supported. For a compatible full fine-tune with "
        "block swapping, Musubi automatically updates each parameter during backward, bit-packs polarity history, and offloads optimizer "
        "state to CPU so large models do not retain a full model of gradients in VRAM. Gradient accumulation, clipping, or Patch Optimizer "
        "for Block Swap selects the safer non-fused fallback and offloads retained gradients to CPU instead."
    ),
    "automagic2": (
        "**Automagic v2 (experimental, fused only):** Adapts one learning rate per parameter tensor and updates parameters as each gradient "
        "finishes during backward, reducing peak gradient memory. The Learning Rate field is the starting rate. It requires single-process "
        "training, Gradient Accumulation Steps = 1, "
        "Max Gradient Norm = 0, and mixed precision other than fp16. Unsupported combinations stop before training rather than silently "
        "producing an incorrect run. If you need accumulation, clipping, fp16, or multi-process training, choose Automagic3 instead; it "
        "automatically uses its safe non-fused mode. External LR schedulers are bypassed. Useful arguments: `min_lr`, `max_lr`, `lr_bump`, `beta2`, "
        "`clip_threshold`, `weight_decay`, and `agreement_threshold`. Adafactor-only preset arguments (`scale_parameter`, `relative_step`, "
        "and `warmup_init`) are ignored automatically. Keep the separate Fused Backward Pass option off; Automagic2 already owns its "
        "fused update path. Block swapping is supported, but keep Patch Optimizer for Block Swap off because gradients are consumed during "
        "backward."
    ),
    "automagic3": (
        "**Automagic v3 (experimental):** Uses a shared adaptive learning rate per parameter group, driven by compact packed sign history. "
        "The Learning Rate field is its starting rate and external LR schedulers are bypassed. With no `fused` argument, Musubi automatically "
        "uses fused backward only for a compatible single-process, one-backward, unclipped, non-fp16 run; otherwise it safely uses normal "
        "optimizer steps. Set `fused=False` to force the compatible mode or `fused=True` to request fused mode with strict validation. Useful "
        "arguments: `min_lr`, `max_lr`, `beta2`, `eps`, `clip_threshold`, `weight_decay`, and `polarity_history` (2-64). Adafactor-only preset arguments "
        "(`scale_parameter`, `relative_step`, and `warmup_init`) are ignored automatically. Keep the separate Fused Backward Pass option "
        "off; that option is only for Adafactor. Block swapping is supported. In full fine-tuning, selecting Patch Optimizer for Block Swap "
        "forces Automagic3 into its compatible non-fused mode; otherwise its normal automatic fused-mode decision is used."
    ),
}


def add_automagic_optimizer_choices(choices):
    result = list(choices)
    existing = {str(choice).casefold() for choice in result}
    for choice in AUTOMAGIC_OPTIMIZER_CHOICES:
        if choice.casefold() not in existing:
            result.append(choice)
    return result


def optimizer_guidance(optimizer_type) -> str:
    normalized = str(optimizer_type or "").strip().casefold()
    if normalized in _OPTIMIZER_GUIDANCE:
        return _OPTIMIZER_GUIDANCE[normalized]
    return (
        f"**Custom optimizer:** `{optimizer_type}` is passed to the Musubi backend. Confirm its constructor arguments, scheduler behavior, "
        "platform support, and checkpoint compatibility before a long run."
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optimizer_arguments(value: object) -> dict[str, object]:
    items = value if isinstance(value, (list, tuple)) else [value]
    result: dict[str, object] = {}
    for item in items:
        if item is None:
            continue
        try:
            tokens = shlex.split(str(item).replace(",", " "))
        except ValueError:
            tokens = str(item).replace(",", " ").split()
        for token in tokens:
            if "=" not in token:
                continue
            key, raw_value = token.split("=", 1)
            try:
                parsed = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                parsed = raw_value
            result[key.strip().casefold()] = parsed
    return result


def _is_full_finetune(config: Mapping[str, object]) -> bool:
    if _as_bool(config.get("full_finetune", False)):
        return True
    mode = str(config.get("training_mode") or "LoRA Training").strip().casefold()
    return mode in {"full fine-tuning", "full dit fine-tuning", "full dit finetuning", "dreambooth fine-tuning"}


def _fused_automagic_conflicts(config: Mapping[str, object]) -> list[str]:
    conflicts: list[str] = []
    accumulation = _as_int(config.get("gradient_accumulation_steps", 1), 1)
    if accumulation != 1:
        conflicts.append(f"Gradient Accumulation Steps is {accumulation}; set it to 1")

    max_grad_norm = _as_float(config.get("max_grad_norm", 0.0), 0.0)
    if max_grad_norm != 0.0:
        conflicts.append(f"Max Gradient Norm is {max_grad_norm:g}; set it to 0")

    mixed_precision = str(config.get("mixed_precision") or "no").strip().casefold()
    if mixed_precision == "fp16":
        conflicts.append("Mixed Precision is fp16; use bf16 or no mixed precision")

    num_processes = _as_int(config.get("num_processes", 1), 1)
    num_machines = _as_int(config.get("num_machines", 1), 1)
    if num_processes != 1 or num_machines != 1 or _as_bool(config.get("multi_gpu", False)):
        conflicts.append("distributed or multi-GPU launch is enabled; use one process on one machine")

    if _as_bool(config.get("block_swap_optimizer_patch_params", False)):
        conflicts.append("Patch Optimizer for Block Swap is enabled; turn it off")

    extra_backward_features = [
        label
        for key, label in (
            ("blank_preservation", "Blank Preservation"),
            ("dop", "DOP"),
            ("audio_dop", "Audio DOP"),
            ("motion_preservation_separate_backward", "separate Motion Preservation backward"),
        )
        if _as_bool(config.get(key, False))
    ]
    if extra_backward_features:
        conflicts.append(f"{', '.join(extra_backward_features)} requires additional backward passes; disable it")

    if _as_bool(config.get("ltx2_model_parallel", False)):
        conflicts.append("LTX-2 model parallelism is enabled; disable it")
    return conflicts


def validate_automagic_configuration(
    config: Mapping[str, object],
    *,
    warning_callback: Callable[[str], object] | None = None,
) -> tuple[str, ...]:
    """Reject unsafe fused combinations and announce automatic safe fallbacks."""
    optimizer_type = str(config.get("optimizer_type") or "").strip().casefold()
    if optimizer_type not in {"automagic", "automagic2", "automagic3"}:
        return ()

    if _is_full_finetune(config) and _as_bool(config.get("fused_backward_pass", False)):
        if optimizer_type == "automagic2":
            detail = "Automagic2 already performs its own fused updates"
        else:
            detail = "Automagic fused behavior is controlled by its optimizer arguments"
        raise ValueError(
            f"The separate Fused Backward Pass checkbox is Adafactor-only; {detail}. "
            "Turn that checkbox off."
        )

    optimizer_args = _optimizer_arguments(config.get("optimizer_args"))
    fused_configured = "fused" in optimizer_args
    fused_enabled = _as_bool(optimizer_args.get("fused", False))
    conflicts = _fused_automagic_conflicts(config)
    errors: list[str] = []
    warnings: list[str] = []

    if optimizer_type == "automagic2":
        if fused_configured:
            errors.append("remove the `fused` optimizer argument; Automagic2 is always internally fused")
        errors.extend(conflicts)
        if errors:
            detail = "\n".join(f"- {error}." for error in errors)
            raise ValueError(
                "Automagic2 cannot start with this configuration:\n"
                f"{detail}\n"
                "Apply the changes above, or choose Automagic3 to keep accumulation, clipping, fp16, or distributed training via its "
                "safe non-fused mode."
            )
    elif fused_configured and fused_enabled and conflicts:
        detail = "\n".join(f"- {conflict}." for conflict in conflicts)
        raise ValueError(
            f"{optimizer_type.title()} fused=True cannot start with this configuration:\n"
            f"{detail}\n"
            "Set `fused=False`, or remove `fused=True` and let Automagic3 select its safe mode automatically."
        )
    elif optimizer_type == "automagic3" and not fused_configured and conflicts:
        reason = "; ".join(conflicts)
        warnings.append(
            "Automagic3 will use its safe non-fused mode for this run because "
            f"{reason}. This preserves the selected training behavior. Set `fused=False` in Optimizer Arguments to make that intent explicit."
        )

    if warning_callback is not None:
        for message in warnings:
            warning_callback(message)
    return tuple(warnings)
