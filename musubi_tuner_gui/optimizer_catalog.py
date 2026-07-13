"""Shared optimizer choices and contextual GUI guidance."""

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
        "producing an incorrect run. External LR schedulers are bypassed. Useful arguments: `min_lr`, `max_lr`, `lr_bump`, `beta2`, "
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
        "arguments: `beta2`, `eps`, `clip_threshold`, `weight_decay`, and `polarity_history` (2-64). Adafactor-only preset arguments "
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
