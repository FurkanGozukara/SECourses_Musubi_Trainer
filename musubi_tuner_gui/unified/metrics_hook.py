"""Show the live learning rate on the trainer's ``steps`` progress bar.

Imported by ``sitecustomize`` in worker processes that the unified UI starts
(``SECOURSES_LIVE_METRICS=1``). A global optimizer step hook records the learning rate of the first
parameter group (times ``d`` for Prodigy / D-Adaptation style optimizers) and the ``steps`` bar's
``set_postfix`` adds it as ``lr=...``, which the dashboard parses. Display only: every failure is
swallowed so training can never be affected.
"""

from __future__ import annotations

_STATE = {"lr": None}
_INSTALLED = False


def _record_lr(optimizer, *_args, **_kwargs) -> None:
    try:
        group = optimizer.param_groups[0]
        lr = group.get("lr")
        if hasattr(lr, "item"):
            lr = lr.item()
        if lr is None:
            return
        lr = float(lr)
        d = group.get("d")
        if d is not None:
            lr *= float(d.item() if hasattr(d, "item") else d)
        _STATE["lr"] = lr
    except Exception:
        pass


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    try:
        from torch.optim.optimizer import register_optimizer_step_post_hook

        register_optimizer_step_post_hook(_record_lr)
    except Exception:
        return
    try:
        import tqdm.std

        original = tqdm.std.tqdm.set_postfix

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            try:
                lr = _STATE["lr"]
                if lr is not None and str(getattr(self, "desc", "")).startswith("steps") and "lr" not in kwargs:
                    kwargs["lr"] = f"{lr:.3g}"
            except Exception:
                pass
            return original(self, ordered_dict, refresh, **kwargs)

        tqdm.std.tqdm.set_postfix = set_postfix
    except Exception:
        pass
