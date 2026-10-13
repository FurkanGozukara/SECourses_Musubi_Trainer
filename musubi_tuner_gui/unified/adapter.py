"""The LoRA / DoRA switch of the unified UI.

``adapter_type`` is not a trainer setting: it is folded into Network Arguments (``use_dora=True``) whenever values
reach a trainer (save, print command, start) and read back from them when a preset loads, so presets stay plain
trainer configs that the classic tabs and the command line read as well. DoRA here is musubi's input-axis DoRA
(``<module>.dora_scale``), which ComfyUI and SwarmUI apply without conversion.
"""

from __future__ import annotations

import re

from .arguments import argument_items
from .param_spec import LORA, ParamSpec

ADAPTER_KEY = "adapter_type"
LORA_ADAPTER = "LoRA"
DORA_ADAPTER = "DoRA"
NETWORK_KEY = "network_args"
_TRUE = {"1", "true", "yes", "on"}
DORA_ITEM = "use_dora=True"
_LEADING_DORA = re.compile(r"\s*use_dora\s*=\s*['\"]?(?:true|1|yes|on)['\"]?(?=[\s,;]|$)[\s,;]*", re.IGNORECASE)


def adapter_spec() -> ParamSpec:
    return ParamSpec(
        key=ADAPTER_KEY,
        kind="dropdown",
        label="Adapter Type",
        default=LORA_ADAPTER,
        info="DoRA also learns a magnitude per input channel: more faithful at the same rank, about 10-15% slower per step, and at "
        "the same learning rate it can need more epochs (Krea 2: best checkpoint at epoch 120 instead of 80; FLUX Klein: no "
        "difference). Both load in ComfyUI / SwarmUI.",
        choices=(LORA_ADAPTER, DORA_ADAPTER),
        section="network",
        priority=-10,
        modes=frozenset({LORA}),
        extras={"family": "text", "synthetic": True},
    )


def is_synthetic(spec: ParamSpec) -> bool:
    return bool((spec.extras or {}).get("synthetic"))


def _split(item: str) -> tuple[str, str]:
    key, _, value = item.partition("=")
    return key.strip().casefold(), value.strip().strip("'\"").casefold()


def _as_input_type(items: list[str], original):
    """The items in the shape the value came in: a list stays a list, text becomes one line."""
    if isinstance(original, (list, tuple)):
        return items
    return " ".join(items)


def fold_adapter(values: dict) -> dict:
    """``values`` with the chosen adapter written into Network Arguments (a copy; nothing else changes)."""
    if ADAPTER_KEY not in values or NETWORK_KEY not in values:
        return values
    original = values.get(NETWORK_KEY)
    items = argument_items(original, literal=False)
    if str(values.get(ADAPTER_KEY) or LORA_ADAPTER).strip().casefold() == DORA_ADAPTER.casefold():
        kept = [item for item in items if _split(item)[0] != "use_dora"]
        if isinstance(original, str) and kept == items:  # keep the user's own text as typed after the switch's item
            return {**values, NETWORK_KEY: (DORA_ITEM + " " + original.strip()).strip()}
        kept.insert(0, DORA_ITEM)
    else:
        kept = [item for item in items if _split(item)[0] not in ("use_dora", "dora_axis")]
    if kept == items:
        return values
    return {**values, NETWORK_KEY: _as_input_type(kept, original)}


def unfold_adapter(values: dict) -> dict:
    """``values`` with ``adapter_type`` read from Network Arguments and the items the switch owns taken out of them
    (``dora_axis=output`` stays: it selects the PEFT layout, which the switch does not offer)."""
    if NETWORK_KEY not in values:
        return values
    original = values.get(NETWORK_KEY)
    items = argument_items(original, literal=False)
    dora = any(_split(item)[0] == "use_dora" and _split(item)[1] in _TRUE for item in items)
    kept = [
        item
        for item in items
        if _split(item)[0] != "use_dora" and not (_split(item)[0] == "dora_axis" and _split(item)[1] in ("", "input", "in"))
    ]
    result = {**values, ADAPTER_KEY: DORA_ADAPTER if dora else LORA_ADAPTER}
    if kept != items:
        lead = _LEADING_DORA.match(original) if isinstance(original, str) else None
        if lead and len(kept) == len(items) - 1:  # the item fold put in front: drop it, keep the rest as typed
            result[NETWORK_KEY] = original[lead.end():]
        else:
            result[NETWORK_KEY] = _as_input_type(kept, original)
    return result
