"""HTML for the live training dashboard (rendered from a ``TrainingRunner`` snapshot).

Pure functions: the timer callback renders a snapshot into a few KB of HTML. The loss chart is an
inline SVG built from the runner's decimated series (at most a few hundred points), so the payload
size does not grow with the length of the run.
"""

from __future__ import annotations

import html
import math
import time

STATUS_PILLS = {
    "idle": ("Ready", "idle"),
    "starting": ("Starting", "running"),
    "running": ("Running", "running"),
    "stopping": ("Stopping", "warn"),
    "finished": ("Finished", "ok"),
    "failed": ("Failed", "bad"),
    "stopped": ("Stopped", "muted"),
}
STALL_SECONDS = 15 * 60


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def format_duration(seconds) -> str:
    if seconds is None:
        return "—"
    try:
        seconds = int(max(0, float(seconds)))
    except (TypeError, ValueError):
        return "—"
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_lr(value) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if value == 0:
        return "0"
    return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def format_speed(seconds_per_it) -> str:
    if not seconds_per_it:
        return "—"
    if seconds_per_it >= 1:
        return f"{seconds_per_it:.2f} s/it"
    return f"{1 / seconds_per_it:.2f} it/s"


def _ago(timestamp, now) -> str:
    if not timestamp:
        return ""
    delta = max(0, now - timestamp)
    if delta < 60:
        return f"{int(delta)} s ago"
    if delta < 3600:
        return f"{int(delta // 60)} min ago"
    return f"{delta / 3600:.1f} h ago"


def _loss_chart(points: list, lr_points: list) -> str:
    points = [point for point in points if math.isfinite(point[1])]  # a NaN / inf would blank or flatten the chart
    lr_points = [point for point in lr_points if math.isfinite(point[1])]
    if len(points) < 2:
        return '<div class="ud-chart ud-chart-empty">The loss chart appears after the first training steps.</div>'
    width, height, pad = 640.0, 132.0, 6.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(x_min, x_max):
        x_max = x_min + 1
    if math.isclose(y_min, y_max):
        y_min, y_max = y_min - 0.5 * abs(y_min or 1) * 0.1, y_max + 0.5 * abs(y_max or 1) * 0.1

    def sx(x):
        return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def sy(y):
        return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    line = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    area = f"{sx(xs[0]):.1f},{height - pad:.1f} {line} {sx(xs[-1]):.1f},{height - pad:.1f}"
    lr_svg = ""
    lr_values = [p[1] for p in lr_points]
    if len(lr_points) >= 2 and max(lr_values) > 0 and not math.isclose(min(lr_values), max(lr_values)):
        lo, hi = min(lr_values), max(lr_values)
        lr_line = " ".join(
            f"{sx(x):.1f},{height - pad - (y - lo) / (hi - lo) * (height - 2 * pad):.1f}" for x, y in lr_points if x_min <= x <= x_max
        )
        lr_svg = f'<polyline class="ud-lr-line" points="{lr_line}" />'
    last_x, last_y = sx(xs[-1]), sy(ys[-1])
    return (
        '<div class="ud-chart">'
        '<div class="ud-chart-head"><span>Average loss</span>'
        f'<span class="ud-chart-range">min {y_min:.4f} · max {y_max:.4f} · steps {int(x_min):,}–{int(x_max):,}</span></div>'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="none" role="img" aria-label="Average loss over training steps">'
        '<defs><linearGradient id="udLossFill" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0%" stop-color="var(--ud-accent)" stop-opacity="0.35"/>'
        '<stop offset="100%" stop-color="var(--ud-accent)" stop-opacity="0.02"/></linearGradient></defs>'
        f'<polygon class="ud-loss-area" points="{area}" />'
        f'<polyline class="ud-loss-line" points="{line}" />'
        f"{lr_svg}"
        f'<circle class="ud-loss-dot" cx="{last_x:.1f}" cy="{last_y:.1f}" r="3.5" />'
        "</svg>"
        + ('<div class="ud-chart-legend"><span class="ud-key ud-key-loss"></span>avg loss <span class="ud-key ud-key-lr"></span>learning rate (scaled)</div>' if lr_svg else "")
        + "</div>"
    )


def _tile(label: str, value: str, sub: str = "", extra_class: str = "", hint: str = "", since: float | None = None) -> str:
    """``since``: the value is a running clock from that time (the browser ticks it every second)."""
    sub_html = f'<div class="ud-tile-sub">{_esc(sub)}</div>' if sub else ""
    title = f' title="{_esc(hint)}"' if hint else ""
    clock = f' data-since="{since:.3f}"' if since is not None else ""
    return f'<div class="ud-tile {extra_class}"{title}><div class="ud-tile-label">{_esc(label)}</div><div class="ud-tile-value"{clock}>{_esc(value)}</div>{sub_html}</div>'


def _ago_html(timestamp, now) -> str:
    """'3 min ago' that the browser keeps current."""
    if not timestamp:
        return "<em></em>"
    return f'<em data-ago="{float(timestamp):.3f}">{_esc(_ago(timestamp, now))}</em>'


def _stages(snapshot: dict) -> str:
    chips = []
    for stage in snapshot["stages"]:
        state = stage["state"]
        mark = {"done": "✓", "active": "●", "failed": "✕", "stopped": "■", "skipped": "–"}.get(state, "")
        detail = f'<span class="ud-stage-detail">{_esc(stage["detail"])}</span>' if stage.get("detail") and state == "active" else ""
        chips.append(
            f'<div class="ud-stage ud-stage-{state}"><span class="ud-stage-icon">{stage["icon"]}</span>'
            f'<span class="ud-stage-label">{_esc(stage["label"])}</span><span class="ud-stage-mark">{mark}</span>{detail}</div>'
        )
    return '<div class="ud-stages">' + '<div class="ud-stage-sep"></div>'.join(chips) + "</div>"


def training_gpus(snapshot: dict) -> list[dict]:
    """The GPUs this run uses: the ones its processes opened, else its GPU IDs setting, else GPU 0."""
    gpus = snapshot.get("gpus") or []
    in_use = [gpu for gpu in gpus if gpu.get("in_use")]
    if in_use:
        return in_use
    configured = [int(part) for part in str((snapshot.get("meta") or {}).get("gpu_ids") or "").split(",") if part.strip().isdigit()]
    return [gpu for gpu in gpus if gpu["index"] in configured] or gpus[:1]


def _vram(snapshot: dict) -> tuple[str, str, str]:
    gpus = training_gpus(snapshot)
    if not gpus:
        return "—", "sampled while running", ""
    hint = "Memory used on the whole GPU (every program on it, not only this training)\n" + "\n".join(
        f"GPU {gpu['index']} · {gpu['name']}: {gpu['used']:.1f} / {gpu['total']:.0f} GB "
        f"(peak {gpu.get('peak', gpu['used']):.1f} GB) · {gpu['util']}% busy"
        for gpu in gpus
    )
    used = sum(gpu["used"] for gpu in gpus)
    total = sum(gpu["total"] for gpu in gpus)
    if len(gpus) == 1:
        return f"{used:.1f} GB", f"of {total:.0f} · peak {gpus[0].get('peak', used):.1f}", hint
    return f"{used:.1f} GB", f"{len(gpus)} GPUs · of {total:.0f}", hint


def render_dashboard(snapshot: dict, now: float | None = None) -> str:
    now = now or time.time()
    status = snapshot["status"]
    pill_text, pill_class = STATUS_PILLS.get(status, (status.title(), "idle"))
    if status == "idle" and not snapshot.get("job_id"):
        return (
            '<div class="ud-dash ud-dash-idle">'
            '<div class="ud-dash-head"><div class="ud-dash-title">📈 Live Training Monitor</div>'
            f'<span class="ud-pill ud-pill-{pill_class}">{pill_text}</span></div>'
            '<div class="ud-idle-body"><div class="ud-idle-icon">🚀</div><div>'
            "<b>Ready to train.</b> Pick a base model, choose LoRA or Fine-Tuning, load a preset and press "
            "<b>▶ Start Training</b>. Caching and training stages, step progress, loss, learning rate, speed, ETA "
            "and GPU memory appear here live."
            "</div></div></div>"
        )

    total = snapshot["total_steps"]
    step = snapshot["step"]
    stage_progress = snapshot.get("stage_progress")
    training = snapshot["current_stage"] in ("train", "finish") and total > 0
    if training:
        fraction = min(1.0, step / total) if total else 0.0
        progress_label = f"Step {step:,} / {total:,}"
        remaining = snapshot.get("train_remaining")
        if remaining is None and snapshot.get("seconds_per_it") and total:
            remaining = (total - step) * snapshot["seconds_per_it"]
    elif stage_progress and stage_progress.get("total"):
        fraction = min(1.0, stage_progress["n"] / stage_progress["total"])
        desc = stage_progress.get("desc") or "Progress"
        progress_label = f"{desc} {stage_progress['n']:,} / {stage_progress['total']:,}"
        remaining = stage_progress.get("remaining")
    else:
        fraction = 1.0 if status == "finished" else 0.0
        progress_label = snapshot.get("message") or ""
        remaining = None
    indeterminate = status in ("starting", "running") and not training and not (stage_progress and stage_progress.get("total"))

    started = snapshot.get("started_at")
    ended = snapshot.get("ended_at")
    elapsed = (ended or now) - started if started else None
    vram_value, vram_sub, vram_hint = _vram(snapshot)
    live = status in ("running", "starting")

    loss = snapshot.get("avr_loss")
    best = snapshot.get("best_loss")
    epoch_value = f"{snapshot['epoch']} / {snapshot['total_epochs']}" if snapshot.get("total_epochs") else (str(snapshot["epoch"]) if snapshot.get("epoch") else "—")
    tiles = [
        _tile("Step", f"{step:,}" if total else "—", f"of {total:,}" if total else ""),
        _tile("Epoch", epoch_value),
        _tile("Avg loss", f"{loss:.4f}" if loss is not None else "—", f"best {best:.4f}" if best is not None else "", "ud-tile-accent"),
        _tile("Learning rate", format_lr(snapshot.get("lr")), "live from optimizer" if snapshot.get("lr") is not None else "shown once training steps run"),
        _tile("Speed", format_speed(snapshot.get("seconds_per_it"))),
        _tile("Elapsed", format_duration(elapsed), since=started if started and not ended else None),
        _tile("ETA", format_duration(remaining) if live else "—", "remaining" if live else ""),
        _tile("VRAM", vram_value, vram_sub, hint=vram_hint),
    ]

    alerts = []
    if snapshot.get("error"):
        alerts.append(f'<div class="ud-alert ud-alert-bad">❌ {_esc(snapshot["error"])}</div>')
    last_progress = snapshot.get("last_progress_at")
    if status == "running" and training and last_progress and now - last_progress > STALL_SECONDS:
        alerts.append(
            f'<div class="ud-alert ud-alert-warn">⏳ No step progress for {format_duration(now - last_progress)} — '
            "sampling or saving can take a while; check the log below if this persists.</div>"
        )
    if status == "finished":
        alerts.append('<div class="ud-alert ud-alert-ok">✅ Training finished. Your model files are in the output directory.</div>')

    activity = snapshot.get("activity") or ""
    checkpoints = snapshot.get("checkpoints") or []
    footer_bits = []
    if activity and status in ("running", "starting"):
        footer_bits.append(f"<span>⚡ {_esc(activity)} {_ago_html(snapshot.get('activity_at'), now)}</span>")
    if checkpoints:
        last = checkpoints[-1]
        footer_bits.append(f"<span>💾 Last checkpoint: <b>{_esc(last['name'])}</b> {_ago_html(last.get('time'), now)}</span>")
    if snapshot.get("samples"):
        footer_bits.append(f"<span>🖼️ Sample rounds: <b>{snapshot['samples']}</b></span>")
    if snapshot.get("log_file"):
        footer_bits.append(f"<span>📝 Full log: <code>{_esc(snapshot['log_file'])}</code></span>")

    meta = snapshot.get("meta") or {}
    subtitle = " · ".join(str(part) for part in (meta.get("model"), meta.get("mode"), meta.get("preset")) if part)
    start_text = time.strftime("%H:%M", time.localtime(started)) if started else ""
    bar_class = "ud-bar-fill ud-bar-indeterminate" if indeterminate else "ud-bar-fill"
    pct_text = "" if indeterminate else f"{fraction * 100:.1f}%"
    return (
        f'<div class="ud-dash ud-dash-{pill_class}">'
        '<div class="ud-dash-head">'
        f'<div><div class="ud-dash-title">📈 Live Training Monitor</div><div class="ud-dash-sub">{_esc(subtitle)}</div></div>'
        f'<div class="ud-dash-right"><span class="ud-started">{("started " + start_text) if start_text else ""}</span>'
        f'<span class="ud-pill ud-pill-{pill_class}">{_esc(pill_text)}</span></div>'
        "</div>"
        + _stages(snapshot)
        + '<div class="ud-progress">'
        f'<div class="ud-progress-text"><span>{_esc(progress_label)}</span><span>{pct_text}</span></div>'
        f'<div class="ud-bar"><div class="{bar_class}" style="width:{(100 if indeterminate else fraction * 100):.2f}%"></div></div>'
        "</div>"
        + "".join(alerts)
        + '<div class="ud-tiles">'
        + "".join(tiles)
        + "</div>"
        + _loss_chart(snapshot.get("loss_series") or [], snapshot.get("lr_series") or [])
        + (f'<div class="ud-foot">{"".join(footer_bits)}</div>' if footer_bits else "")
        + "</div>"
    )
