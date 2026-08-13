"""Shared GUI rules for trainers that support LoRA and full-DiT fine-tuning."""

from __future__ import annotations


LORA_TRAINING_MODE = "LoRA Training"
FULL_FINE_TUNING_MODE = "Full Fine-Tuning"
TRAINING_MODE_CHOICES = [LORA_TRAINING_MODE, FULL_FINE_TUNING_MODE]

_FULL_MODE_ALIASES = {
    FULL_FINE_TUNING_MODE.casefold(),
    "dreambooth fine-tuning",
    "full dit fine-tuning",
    "full dit finetuning",
}

FULL_FINE_TUNING_NETWORK_KEYS = {
    "network_weights",
    "network_module",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "dim_from_weights",
    "scale_weight_norms",
    "base_weights",
    "base_weights_multiplier",
}

FULL_FINE_TUNING_ONLY_KEYS = {
    "fused_backward_pass",
    "block_swap_optimizer_patch_params",
}


def normalize_training_mode(value: object) -> str:
    mode = str(value or LORA_TRAINING_MODE).strip()
    if mode.casefold() in _FULL_MODE_ALIASES:
        return FULL_FINE_TUNING_MODE
    if mode.casefold() == LORA_TRAINING_MODE.casefold():
        return LORA_TRAINING_MODE
    raise ValueError(f"Unsupported training mode: {mode}")


def is_full_fine_tuning(value: object) -> bool:
    return normalize_training_mode(value) == FULL_FINE_TUNING_MODE


def infer_training_mode_from_loaded_config(
    data: dict,
    full_mode_label: str = FULL_FINE_TUNING_MODE,
    lora_mode_label: str = LORA_TRAINING_MODE,
) -> str:
    """Recover training_mode for a loaded run TOML, tolerating files saved
    before the flag was persisted.

    Not every trainer's Radio uses the shared canonical "Full Fine-Tuning" /
    "LoRA Training" strings verbatim (Qwen Image and Z-Image use "DreamBooth
    Fine-Tuning" as their own label) -- callers pass their own label spelling
    via full_mode_label/lora_mode_label so the returned value is always one
    of that trainer's actual Radio choices.

    When the TOML has an explicit (even legacy-alias-spelled) training_mode,
    normalize it and map back onto the caller's label. An unrecognized value
    falls back to LoRA rather than raising, since a hand-edited or corrupt
    file shouldn't crash config loading.

    When training_mode is missing outright -- an older run TOML saved before
    this flag was persisted, or a genuine full-fine-tune run that never
    writes network_module (see FULL_FINE_TUNING_NETWORK_KEYS) -- infer it
    from whether network_module is present: full-fine-tuning never writes
    it, LoRA training always does (network_module is a mandatory key
    whenever training_mode isn't Full Fine-Tuning).
    """
    if "training_mode" in data:
        try:
            full = is_full_fine_tuning(data.get("training_mode"))
        except ValueError:
            full = False
        return full_mode_label if full else lora_mode_label
    if not str(data.get("network_module") or "").strip():
        return full_mode_label
    return lora_mode_label


def replace_parameter(
    parameters: list[tuple[str, object]],
    key: str,
    value: object,
) -> list[tuple[str, object]]:
    return [(name, value if name == key else current) for name, current in parameters]


def normalize_image_training_parameters(
    parameters: list[tuple[str, object]],
) -> tuple[dict, list[tuple[str, object]], bool]:
    """Normalize shared runtime values and validate full-fine-tune-only combinations."""
    param_dict = dict(parameters)
    mode = normalize_training_mode(param_dict.get("training_mode"))
    param_dict["training_mode"] = mode
    parameters = replace_parameter(parameters, "training_mode", mode)
    full_finetune = mode == FULL_FINE_TUNING_MODE

    if not full_finetune:
        return param_dict, parameters, False

    if bool(param_dict.get("full_fp16", False)):
        raise ValueError("Full FP16 is not supported for full-DiT fine-tuning; use FP32 or Full BF16.")

    # Full-DiT training cannot use a frozen/quantized base or the LoRA-only H2D path.
    for key in ("fp8_base", "fp8_scaled", "convrot_int8", "block_swap_h2d_only"):
        if key in param_dict:
            param_dict[key] = False
            parameters = replace_parameter(parameters, key, False)

    full_bf16 = bool(param_dict.get("full_bf16", False))
    mixed_precision = str(param_dict.get("mixed_precision") or "no").strip().lower()
    if full_bf16 and mixed_precision != "bf16":
        raise ValueError("Full BF16 fine-tuning requires mixed_precision='bf16'.")

    trainable_dtype = "bfloat16" if full_bf16 else "float32"
    save_precision = "bf16" if full_bf16 else "fp32"
    if "dit_dtype" in param_dict:
        param_dict["dit_dtype"] = trainable_dtype
        parameters = replace_parameter(parameters, "dit_dtype", trainable_dtype)
    if "save_precision" in param_dict:
        param_dict["save_precision"] = save_precision
        parameters = replace_parameter(parameters, "save_precision", save_precision)

    try:
        blocks_to_swap = int(param_dict.get("blocks_to_swap", 0) or 0)
        num_processes = int(param_dict.get("num_processes", 1) or 1)
        gradient_accumulation_steps = int(param_dict.get("gradient_accumulation_steps", 1) or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Full fine-tuning process, accumulation, and block-swap values must be integers.") from exc

    if blocks_to_swap > 0 and num_processes > 1:
        raise ValueError("Block swap is not supported for multi-process full-DiT fine-tuning.")

    optimizer_type = str(param_dict.get("optimizer_type") or "").strip().casefold()
    fused_backward = bool(param_dict.get("fused_backward_pass", False))
    patch_optimizer = bool(param_dict.get("block_swap_optimizer_patch_params", False))
    if fused_backward:
        if optimizer_type.startswith("automagic"):
            detail = (
                "Automagic2 is already internally fused"
                if optimizer_type == "automagic2"
                else "Automagic fused behavior is controlled by its optimizer arguments"
            )
            raise ValueError(f"The separate Fused Backward Pass option is Adafactor-only; {detail}. Turn it off.")
        if optimizer_type != "adafactor":
            raise ValueError("Fused backward pass requires the Adafactor optimizer.")
        if num_processes != 1:
            raise ValueError("Fused backward pass is not supported for multi-process full-DiT fine-tuning.")
        if gradient_accumulation_steps != 1:
            raise ValueError("Fused backward pass requires gradient_accumulation_steps=1.")
        if mixed_precision == "fp16":
            raise ValueError("Fused backward pass is incompatible with mixed_precision='fp16'.")
    if patch_optimizer:
        if blocks_to_swap < 1:
            raise ValueError("Optimizer parameter patching requires blocks_to_swap to be greater than 0.")
        if optimizer_type not in {"adafactor", "adamw", "automagic", "automagic3"}:
            raise ValueError("Optimizer parameter patching supports Adafactor, AdamW, Automagic, and non-fused Automagic3.")
    if fused_backward and patch_optimizer:
        raise ValueError("Select either fused backward pass or optimizer parameter patching, not both.")

    return param_dict, parameters, True


def training_mode_runtime_exclusions(mode: object) -> set[str]:
    # training_mode itself is intentionally NOT excluded: it's harmless to the
    # backend (unknown TOML keys become unused argparse.Namespace attributes,
    # see training/parser_common.py:read_config_from_file) and persisting it
    # is what lets a saved/failed run be reloaded into the correct mode later.
    exclusions = set()
    if is_full_fine_tuning(mode):
        exclusions.update(FULL_FINE_TUNING_NETWORK_KEYS)
        exclusions.update({"full_fp16", "block_swap_h2d_only"})
    else:
        exclusions.update(FULL_FINE_TUNING_ONLY_KEYS)
        exclusions.update({"dit_variant", "mem_eff_save"})
    return exclusions
