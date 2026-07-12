"""Compatibility helpers for the PyTorch stack used by the GUI."""

import enum
import functools


def apply_torchao_enum_pytree_compatibility() -> None:
    """Skip obsolete TorchAO Enum registration on newer PyTorch versions."""
    try:
        import torch.utils._pytree as pytree
        from torch._library.opaque_object import is_opaque_type
    except (AttributeError, ImportError):
        return

    original_register_constant = pytree.register_constant
    if getattr(original_register_constant, "_musubi_torchao_enum_compat", False):
        return

    @functools.wraps(original_register_constant)
    def register_constant(cls):
        is_torchao_enum = (
            isinstance(cls, type)
            and issubclass(cls, enum.Enum)
            and cls.__module__.startswith("torchao.")
        )
        if is_torchao_enum and is_opaque_type(cls):
            return None
        return original_register_constant(cls)

    register_constant._musubi_torchao_enum_compat = True
    pytree.register_constant = register_constant


apply_torchao_enum_pytree_compatibility()
