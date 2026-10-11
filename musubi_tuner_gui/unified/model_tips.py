"""Model tips shown in the base-model card (content lives in ``hints.py`` and ``kohya_backend.KOHYA_HINTS``).

Every tip of a list is shown: the lists are short on purpose, so none is ever cut off.
"""

from __future__ import annotations

from .param_spec import FULL, LORA
from .registry import ModelEntry


def _model_hints() -> dict:
    try:
        from .hints import MODEL_HINTS

        return MODEL_HINTS
    except Exception:
        return {}


def _kohya_hints(entry: ModelEntry) -> dict:
    try:
        from .kohya_backend import KOHYA_HINTS

        return KOHYA_HINTS.get(entry.kohya_model, {}) or {}
    except Exception:
        return {}


def _line_for(value, mode: str) -> str:
    """A fact line that is either shared by every training type (str) or given per type ({mode: str})."""
    if isinstance(value, dict):
        return str(value.get(mode) or "")
    return str(value or "")


def tips_for(entry: ModelEntry, mode: str) -> dict:
    """{"about": str, "tips": [str], "vram": str, "dataset": str} for the card."""
    if entry.is_kohya:
        data = _kohya_hints(entry)
        tips = list(data.get(mode) or data.get(LORA if mode == FULL else FULL) or [])
        return {"about": entry.tagline, "tips": tips, "vram": "", "dataset": ""}
    data = _model_hints().get(entry.key, {}) or {}
    tips = list(data.get("all") or []) + list(data.get(mode) or [])
    return {
        "about": data.get("about") or entry.tagline,
        "tips": tips,
        "vram": _line_for(data.get("vram"), mode),
        "dataset": _line_for(data.get("dataset"), mode),
    }
