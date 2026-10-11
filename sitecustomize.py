"""Apply project-wide Python compatibility hooks in launched worker processes."""

import os

import musubi_tuner_gui.torch_compat  # noqa: F401

if os.environ.get("SECOURSES_LIVE_METRICS") == "1":
    try:
        from musubi_tuner_gui.unified.metrics_hook import install as _install_live_metrics

        _install_live_metrics()
    except Exception:
        pass
