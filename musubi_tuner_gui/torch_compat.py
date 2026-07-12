"""Compatibility helpers for the PyTorch stack used by the GUI and workers."""

import enum
import functools
import os
import warnings


_DEPENDENCY_INVALID_ESCAPE_WARNINGS = (
    (
        r"invalid escape sequence '\\\.'",
        r"ignore:invalid escape sequence '\.':SyntaxWarning",
    ),
    (
        r"invalid escape sequence '\\s'",
        r"ignore:invalid escape sequence '\s':SyntaxWarning",
    ),
)


def suppress_dependency_invalid_escape_warnings() -> None:
    """Hide known dependency docstring warnings in this process and workers."""
    for message, _ in _DEPENDENCY_INVALID_ESCAPE_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=SyntaxWarning)

    warning_options = os.environ.get("PYTHONWARNINGS", "")
    configured_options = [option.strip() for option in warning_options.split(",") if option.strip()]
    for _, option in _DEPENDENCY_INVALID_ESCAPE_WARNINGS:
        if option not in configured_options:
            configured_options.append(option)
    os.environ["PYTHONWARNINGS"] = ",".join(configured_options)


def apply_torchao_enum_pytree_compatibility() -> None:
    """Skip obsolete TorchAO Enum registration on newer PyTorch versions."""
    try:
        import torch.utils._pytree as pytree
        from torch._library.opaque_object import is_opaque_type
    except (AttributeError, ImportError):
        return

    original_register_constant = getattr(pytree, "register_constant", None)
    if original_register_constant is None:
        return
    if getattr(original_register_constant, "_musubi_torchao_enum_compat", False):
        return

    @functools.wraps(original_register_constant)
    def register_constant(cls, *args, **kwargs):
        if isinstance(cls, type) and issubclass(cls, enum.Enum) and is_opaque_type(cls):
            return None
        return original_register_constant(cls, *args, **kwargs)

    register_constant._musubi_torchao_enum_compat = True
    pytree.register_constant = register_constant


suppress_dependency_invalid_escape_warnings()
apply_torchao_enum_pytree_compatibility()
