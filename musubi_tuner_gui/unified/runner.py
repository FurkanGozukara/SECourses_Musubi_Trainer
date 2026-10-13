"""Training process manager behind the unified UI's live dashboard.

``TrainingRunner`` replaces each trainer module's ``CommandExecutor`` (same ``execute_command`` /
``is_running`` / ``kill_command`` interface, but no Gradio components). It captures the output of
every caching and training process, mirrors it to the app console unchanged, and parses it into a
small bounded state: pipeline stages, step / total, epoch, loss, learning rate, speed, ETA, saved
checkpoints, GPU memory and the last log lines. Everything the UI reads is O(1) in the length of the
run, so a run of 100,000 steps or several days costs the browser the same as a run of 10 steps.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import psutil

from ..custom_logging import setup_logging

log = setup_logging()

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_LOG_DIR = os.path.join(APP_DIR, "logs", "training_runs")
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

LOG_LINES_KEPT = 600  # lines kept in memory (the panel shows the last 100)
LINE_CHARS_KEPT = 2000  # longer lines are shortened in memory (the run's log file keeps them whole)
SERIES_POINTS = 480  # loss / LR chart points kept (decimated, so any run length fits)
LOSS_WINDOW = 100  # steps averaged when a trainer shows only the per-step loss (LTX 2.3 fine-tuning)
GPU_SAMPLE_SECONDS = 3.0
PHASE_SAMPLE_SECONDS = 2.0

STAGES: tuple[tuple[str, str, str], ...] = (
    ("prepare", "Preparing", "⚙️"),
    ("latents", "Caching latents", "🧊"),
    ("text", "Caching text encoder", "🔤"),
    ("train", "Training", "🏋️"),
    ("finish", "Saving & finishing", "💾"),
    ("score", "Best checkpoint", "🏆"),  # only shown for a run with an after-step (AfterStep)
)
STAGE_INDEX = {stage_id: index for index, (stage_id, _label, _icon) in enumerate(STAGES)}

# Child-process script names -> stage (first match wins).
PROCESS_PHASES = (
    ("merge_fft_to_model", "finish"),
    ("cache_latents", "latents"),
    ("cache_pixel", "latents"),
    ("cache_text_encoder", "text"),
    ("train_network", "train"),
    ("_train.py", "train"),
    ("fine_tune.py", "train"),
)
TEXT_CUES = (
    (re.compile(r"cach(?:e|ing)[ _](?:the )?latents|latent caching|encoding latents", re.I), "latents"),
    (re.compile(r"cach(?:e|ing)[ _]text[ _]encoder|text[ -]encoder (?:output )?caching|caching text encoder outputs", re.I), "text"),
)

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")
DELIMITER_RE = re.compile(rb"\r\n|\r|\n")
TQDM_RE = re.compile(
    r"^(?P<desc>.*?)(?::\s*)?\s*(?P<pct>\d{1,3})%\|.*?\|\s*(?P<n>\d+(?:\.\d+)?[kMG]?)/(?P<total>\d+(?:\.\d+)?[kMG]?)"
    r"\s*\[(?P<elapsed>[\d:]+)(?:<(?P<remaining>[\d:?]+))?(?:,\s*(?P<rate>[\d.]+|\?)\s*(?P<unit>[a-zA-Z/]+))?(?P<postfix>[^\]]*)\]"
)
EPOCH_RE = re.compile(r"^\s*epoch\s+(\d+)\s*/\s*(\d+)\s*$", re.I)
CHECKPOINT_RE = re.compile(r"saving (?:checkpoint|model)\s*:?\s*(.*?)\s*$", re.I)
MODEL_FILE_RE = re.compile(r"\.(?:safetensors|ckpt|pt|pth|bin|gguf)$", re.I)
SAMPLE_RE = re.compile(r"generat\w* (?:sample|preview)s?|sample images?|sampling images", re.I)
SAMPLE_ROUND_RE = re.compile(r"generating sample images at step", re.I)  # musubi and sd-scripts: once per round
ERROR_RE = re.compile(r"^(?:[\w.]+(?:Error|Exception|Interrupt))(?::|$)")
OOM_RE = re.compile(r"out of memory|OutOfMemoryError", re.I)
# sd-scripts logs through Rich: "<time> INFO     message      train_util.py:5942", long messages wrap onto
# lines indented past the time and level columns.
RICH_SOURCE_RE = re.compile(r"\s{2,}[\w.-]+\.py:\d+\s*$")
RICH_CONTINUATION_RE = re.compile(r"^\s{8,}(?!(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\s)\S")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _shorten(line: str) -> str:
    """One kept log line: a trainer can print a single line of hundreds of KB (a dict of load results),
    which would otherwise be re-sent to every open browser tab for the rest of the run."""
    if len(line) <= LINE_CHARS_KEPT:
        return line
    return f"{line[:LINE_CHARS_KEPT]} … [{len(line) - LINE_CHARS_KEPT:,} more characters in the log file]"


def _to_number(token: str) -> float:
    multipliers = {"k": 1e3, "M": 1e6, "G": 1e9}
    if token and token[-1] in multipliers:
        return float(token[:-1]) * multipliers[token[-1]]
    return float(token)


def _number(text) -> float | None:
    """A progress-bar postfix value as a float (NaN / inf included), None when missing or not a number."""
    if text in (None, ""):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_interval(text: str | None) -> float | None:
    if not text or "?" in text:
        return None
    seconds = 0.0
    try:
        for part in text.split(":"):
            seconds = seconds * 60 + float(part)
    except ValueError:
        return None
    return seconds


def _parse_postfix(text: str) -> dict[str, str]:
    values = {}
    for item in text.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_tqdm(line: str) -> dict | None:
    """Parse one tqdm progress line; None if the line is not a progress bar."""
    if "%|" not in line:
        return None
    text = line.strip()
    match = TQDM_RE.search(text)
    if not match:
        return None
    try:
        n = _to_number(match.group("n"))
        total = _to_number(match.group("total"))
    except ValueError:
        return None
    rate = match.group("rate")
    unit = match.group("unit") or ""
    seconds_per_it = None
    if rate and rate != "?":
        value = float(rate)
        if unit.startswith("s/") and value > 0:
            seconds_per_it = value
        elif unit.endswith("/s") and value > 0:
            seconds_per_it = 1.0 / value
    desc = (match.group("desc") or "").strip().rstrip(":").strip()
    return {
        "desc": desc,
        "n": int(n),
        "total": int(total),
        "pct": int(match.group("pct")),
        "elapsed": _parse_interval(match.group("elapsed")),
        "remaining": _parse_interval(match.group("remaining")),
        "seconds_per_it": seconds_per_it,
        "postfix": _parse_postfix(match.group("postfix") or ""),
        "rest": text[match.end():].strip(),  # text printed right after the bar (a print() without a newline first)
    }


def _is_train_bar(progress: dict) -> bool:
    desc = progress["desc"].lower()
    return desc.startswith("steps") or desc == "training"


class DecimatedSeries:
    """At most ``limit`` (x, y) points: when full, every second point is dropped and the stride doubles."""

    def __init__(self, limit: int = SERIES_POINTS):
        self.limit = limit
        self.points: list[tuple[float, float]] = []
        self.stride = 1
        self._last_x: float | None = None

    def add(self, x: float, y: float) -> None:
        if self._last_x is not None:
            if x < self._last_x:  # a new run segment (e.g. resumed) restarts the axis
                self.points.clear()
                self.stride = 1
            elif x - self._last_x < self.stride:
                if self.points:
                    self.points[-1] = (self.points[-1][0], y) if x == self.points[-1][0] else self.points[-1]
                return
        self.points.append((x, y))
        self._last_x = x
        if len(self.points) > self.limit:
            self.points = self.points[::2]
            self.stride *= 2

    def clear(self) -> None:
        self.points.clear()
        self.stride = 1
        self._last_x = None


@dataclass
class AfterStep:
    """A process that runs once the training process exited successfully (the best checkpoint finder): the run stays
    active and Stop ends it, but its failure never fails the training. ``read_result`` returns the small summary the
    dashboard shows (``headline``, ``reasons``...) or None."""

    label: str
    cmd: list
    cwd: str | None = None
    env: dict | None = None
    read_result: Callable[[], dict | None] | None = None


@dataclass
class RunState:
    job_id: int = 0
    status: str = "idle"  # idle | starting | running | stopping | finished | failed | stopped
    title: str = ""
    subtitle: str = ""
    meta: dict = field(default_factory=dict)
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    error: str = ""
    message: str = "Ready"
    stage_states: dict = field(default_factory=dict)
    stage_detail: dict = field(default_factory=dict)
    current_stage: str = ""
    stage_progress: dict | None = None  # last non-training progress bar (caching etc.)
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    total_epochs: int = 0
    avr_loss: float | None = None
    loss: float | None = None
    best_loss: float | None = None
    lr: float | None = None
    seconds_per_it: float | None = None
    train_elapsed: float | None = None
    train_remaining: float | None = None
    last_progress_at: float | None = None
    activity: str = ""
    activity_at: float | None = None
    checkpoints: list = field(default_factory=list)
    samples: int = 0
    gpus: list = field(default_factory=list)
    peak_gb: dict = field(default_factory=dict)
    log_file: str = ""
    after: AfterStep | None = None
    after_only: bool = False  # a job that runs only the after-step (🔍 Analyze): no training stages
    result: dict | None = None  # the after-step's summary


class _JobLogHandler(logging.Handler):
    """Copies the GUI's own log messages emitted by the job thread into the run's log lines."""

    def __init__(self, runner: "TrainingRunner"):
        super().__init__(level=logging.INFO)
        self.runner = runner

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self.runner._job_thread_ident:
            return
        try:
            message = record.getMessage()
        except Exception:
            return
        for line in str(message).splitlines():
            self.runner._add_line(f"[{record.levelname.lower()}] {line}")


class TrainingRunner:
    """One training job at a time; thread-safe snapshots for any number of browser tabs."""

    def __init__(self):
        self.lock = threading.RLock()
        self.headless = True
        self.process: subprocess.Popen | None = None
        self._cancel_requested = False
        self.version = 0  # the run's state and the live progress line
        self.log_version = 0  # the committed log lines (sent to the browser only when they change)
        self.state = RunState()
        self.lines: collections.deque[str] = collections.deque(maxlen=LOG_LINES_KEPT)
        self.progress_line = ""
        self._progress_temporary = False  # the progress line is a bar other than the training steps
        self.loss_series = DecimatedSeries()
        self.lr_series = DecimatedSeries()
        self._step_losses: collections.deque[float] = collections.deque(maxlen=LOSS_WINDOW)
        self._loss_step = -1
        self._pending_checkpoint: str | None = None  # a "saving checkpoint:" message Rich wrapped onto more lines
        self._pending_checkpoint_lines = 0
        self._job_counter = 0
        self._job_thread: threading.Thread | None = None
        self._job_thread_ident: int | None = None
        self._job_done = threading.Event()
        self._job_done.set()
        self._after_pending = False  # a run with an after-step is active until that step ends
        self._readers: list[threading.Thread] = []
        self._monitor: threading.Thread | None = None
        self._log_handle = None
        self._last_progress_write = 0.0
        self._last_cache_cue = 0.0
        self._seen_train_progress = False
        self._error_lines: collections.deque[str] = collections.deque(maxlen=20)
        handler = _JobLogHandler(self)
        logging.getLogger().addHandler(handler)
        if log is not None and log is not logging.getLogger() and not log.propagate:  # a propagating logger reaches root
            log.addHandler(handler)

    # ------------------------------------------------------------------ state helpers
    def _touch(self) -> None:
        self.version += 1

    def _set_message(self, message: str) -> None:
        with self.lock:
            self.state.message = message
            self._touch()

    def _add_line(self, line: str) -> None:
        with self.lock:
            self.lines.append(_shorten(line))
            self.log_version += 1
        self._write_log(line + "\n")

    def _set_progress_line(self, text: str, temporary: bool = False) -> None:
        text, temporary = _shorten(text), temporary and bool(text)
        with self.lock:
            if text == self.progress_line and temporary == self._progress_temporary:
                return
            self.progress_line = text
            self._progress_temporary = temporary
            self._touch()

    def _write_log(self, text: str) -> None:
        handle = self._log_handle
        if handle is None:
            return
        try:
            handle.write(text)
            handle.flush()
        except (OSError, ValueError):
            pass

    def _open_log(self, slug: str) -> str:
        try:
            os.makedirs(RUN_LOG_DIR, exist_ok=True)
            path = os.path.join(RUN_LOG_DIR, f"{time.strftime('%Y%m%d-%H%M%S')}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', slug)[:60]}.log")
            self._log_handle = open(path, "a", encoding="utf-8", errors="replace")
            return path
        except OSError:
            self._log_handle = None
            return ""

    def _close_log(self) -> None:
        handle, self._log_handle = self._log_handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _advance(self, stage_id: str, detail: str = "") -> None:
        """Move the pipeline forward to ``stage_id`` (never backwards)."""
        with self.lock:
            states = self.state.stage_states
            current = self.state.current_stage
            if current and STAGE_INDEX.get(stage_id, -1) <= STAGE_INDEX.get(current, -1):
                if detail and stage_id == current:
                    self.state.stage_detail[stage_id] = detail
                    self._touch()
                return
            target = STAGE_INDEX[stage_id]
            for index, (other, _label, _icon) in enumerate(STAGES):
                if index < target:
                    if states.get(other) == "active":
                        states[other] = "done"
                    elif states.get(other) in (None, "pending"):
                        states[other] = "skipped"
            states[stage_id] = "active"
            self.state.current_stage = stage_id
            if detail:
                self.state.stage_detail[stage_id] = detail
            self.state.stage_progress = None
            self._touch()

    # ------------------------------------------------------------------ public API
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def is_active(self) -> bool:
        return not self._job_done.is_set() or self.is_running() or self._after_pending

    def start_job(
        self,
        target: Callable[[], Any],
        *,
        title: str,
        subtitle: str = "",
        meta: dict | None = None,
        wait: float = 2.5,
        after: AfterStep | None = None,
        after_only: bool = False,
    ) -> tuple[bool, str]:
        """Run ``target`` (a trainer's train function) in a worker thread; ``after`` runs once training succeeded.

        Returns (started, message). Fast failures (validation errors) are returned as messages so the
        caller can show them as an error toast; everything after that is reported through the state.
        """
        with self.lock:
            if self.is_active():
                return False, "A training run is already in progress. Stop it first or wait until it finishes."
            self._job_counter += 1
            self._reset(title=title, subtitle=subtitle, meta=meta or {})
            self.state.after = after
            self.state.after_only = after_only
            self._after_pending = after is not None
            self._job_done.clear()
            thread = threading.Thread(target=self._job_main, args=(target,), name=f"training-job-{self._job_counter}", daemon=True)
            self._job_thread = thread
        thread.start()
        deadline = time.time() + wait
        while time.time() < deadline:
            if self._job_done.is_set() or self.process is not None:
                break
            time.sleep(0.05)
        with self.lock:
            if self._job_done.is_set() and self.state.status == "failed":
                return False, self.state.error or "Training could not start."
        return True, "Training started."

    def start_after_step(self, after: AfterStep, *, title: str, meta: dict | None = None) -> tuple[bool, str]:
        """Run only an after-step, as a job of its own (the finder on a folder that already holds a run's samples)."""
        return self.start_job(lambda: self._run_after(after, ""), title=title, meta=meta, after=after, after_only=True)

    def stop(self) -> str:
        """Kill the running process tree (caching or training)."""
        with self.lock:
            active = self.is_active()
            self._cancel_requested = True
            if active:
                self.state.status = "stopping"
                self.state.message = "Stopping… waiting for the process tree to exit"
                self._touch()
        if not active:
            return "Nothing is running."
        process = self.process
        if process is not None and process.poll() is None:
            self._kill_tree(process.pid)
        self._add_line("[gui] Stop requested by user — terminating the training process tree.")
        return "Stop requested."

    def snapshot(self) -> dict:
        with self.lock:
            state = self.state
            return {
                "version": self.version,
                "job_id": state.job_id,
                "active": self.is_active(),
                "status": state.status,
                "title": state.title,
                "subtitle": state.subtitle,
                "meta": dict(state.meta),
                "started_at": state.started_at,
                "ended_at": state.ended_at,
                "exit_code": state.exit_code,
                "error": state.error,
                "message": state.message,
                "stages": [
                    {
                        "id": stage_id,
                        "label": label,
                        "icon": icon,
                        "state": state.stage_states.get(stage_id, "pending"),
                        "detail": state.stage_detail.get(stage_id, ""),
                    }
                    for stage_id, label, icon in STAGES
                    if (stage_id == "score" and state.after is not None) or (stage_id != "score" and not state.after_only)
                ],
                "after_only": state.after_only,
                "current_stage": state.current_stage,
                "stage_progress": dict(state.stage_progress) if state.stage_progress else None,
                "step": state.step,
                "total_steps": state.total_steps,
                "epoch": state.epoch,
                "total_epochs": state.total_epochs,
                "avr_loss": state.avr_loss,
                "loss": state.loss,
                "best_loss": state.best_loss,
                "lr": state.lr,
                "seconds_per_it": state.seconds_per_it,
                "train_elapsed": state.train_elapsed,
                "train_remaining": state.train_remaining,
                "last_progress_at": state.last_progress_at,
                "activity": state.activity,
                "activity_at": state.activity_at,
                "checkpoints": list(state.checkpoints[-6:]),
                "samples": state.samples,
                "gpus": [dict(gpu) for gpu in state.gpus],
                "loss_series": list(self.loss_series.points),
                "lr_series": list(self.lr_series.points),
                "log_file": state.log_file,
                "result": dict(state.result) if state.result else None,
                "log_version": self.log_version,
                "progress_line": self.progress_line,
            }

    def log_tail(self, count: int = 100) -> str:
        with self.lock:
            lines = list(self.lines)[-count:]
            if self.progress_line:
                lines.append(self.progress_line)
        return "\n".join(lines)

    def log_text(self, count: int = 100) -> str:
        """The last ``count`` committed log lines (without the live progress line)."""
        with self.lock:
            return "\n".join(list(self.lines)[-count:])

    def raise_if_cancelled(self) -> None:
        """For long preparation steps on the job thread (audio extraction...): Stop ends the run there."""
        if self._cancel_requested and threading.get_ident() == self._job_thread_ident:
            from ..common_gui import TrainingCancelled

            raise TrainingCancelled("Stopped by user")

    # ------------------------------------------------------------------ legacy executor interface
    def execute_command(self, run_cmd, **kwargs) -> None:
        """Start a process without waiting (the legacy CommandExecutor signature)."""
        if self.process is not None and self.process.poll() is None:
            log.info("A training process is already running; not starting another one.")
            return
        self.raise_if_cancelled()  # Stop pressed while the job was still preparing (dataset, audio, caching)
        if threading.get_ident() != self._job_thread_ident:
            self._cancel_requested = False  # a process started outside a job starts fresh
        command_text = run_cmd if isinstance(run_cmd, str) else " ".join(str(part) for part in run_cmd)
        self._add_line(f"[gui] $ {command_text}")
        process = self._spawn(run_cmd, kwargs)
        self._adopt(process, "Running")
        reader = threading.Thread(target=self._pump, args=(process, True), name="training-output", daemon=True)
        reader.start()
        self._readers.append(reader)
        self._ensure_monitor()

    def run_blocking(self, cmd, env, *, label: str) -> tuple[int, list[str]]:
        """Run a preparation step (e.g. caching) to completion, streaming its output into the run."""
        self.raise_if_cancelled()
        stage = "text" if "text" in label.lower() else ("latents" if any(word in label.lower() for word in ("latent", "pixel")) else "")
        if stage:
            self._advance(stage, label)
        self._add_line(f"[gui] {label}: {' '.join(str(part) for part in cmd) if not isinstance(cmd, str) else cmd}")
        process = self._spawn(cmd, {"env": env})
        self._adopt(process, None)
        self._ensure_monitor()
        tail: collections.deque[str] = collections.deque(maxlen=40)
        self._pump(process, False, tail)
        return process.returncode, list(tail)

    def kill_command(self):
        return self.stop()

    # ------------------------------------------------------------------ internals
    def _adopt(self, process: subprocess.Popen, message: str | None) -> None:
        """Make ``process`` the run's current process. A Stop that came in while it was being created
        kills it at once (Stop only sees the processes that exist when it is pressed)."""
        with self.lock:
            self.process = process
            self._error_lines.clear()  # a failure is explained by this process's own errors
            stopping = self._cancel_requested
            if not stopping:  # never turn "stopping" back into "running"
                if self.state.status in ("starting", "idle"):
                    self.state.status = "running"
                if message:
                    self.state.message = message
            self._touch()
        if stopping:
            self._kill_tree(process.pid)

    def _reset(self, *, title: str, subtitle: str, meta: dict) -> None:
        self._close_log()
        self.state = RunState(job_id=self._job_counter, status="starting", title=title, subtitle=subtitle, meta=meta)
        self.state.started_at = time.time()
        self.state.stage_states = {stage_id: "pending" for stage_id, _label, _icon in STAGES}
        self.state.stage_states["prepare"] = "active"
        self.state.current_stage = "prepare"
        self.state.message = "Preparing the training workflow…"
        self.lines.clear()
        self.log_version += 1
        self.progress_line = ""
        self._progress_temporary = False
        self.loss_series.clear()
        self.lr_series.clear()
        self._step_losses.clear()
        self._loss_step = -1
        self._pending_checkpoint = None
        self.process = None
        self._cancel_requested = False
        self._seen_train_progress = False
        self._last_cache_cue = 0.0
        self._error_lines.clear()
        self.state.log_file = self._open_log(title or "training")
        self._touch()

    def _job_main(self, target: Callable[[], Any]) -> None:
        from ..common_gui import TrainingCancelled

        self._job_thread_ident = threading.get_ident()
        error = ""
        cancelled = False
        try:
            target()
        except TrainingCancelled:
            cancelled = True
        except Exception as exc:  # gr.Error carries .message
            message = getattr(exc, "message", None) or str(exc) or type(exc).__name__
            if self._cancel_requested:
                cancelled = True
            else:
                error = str(message)
                log.error(f"Training workflow failed before the trainer started: {message}")
        finally:
            self._job_thread_ident = None
            with self.lock:
                launched = self.process is not None
                if cancelled or (self._cancel_requested and not self.is_running()):
                    self._finish("stopped", None, "Stopped by user")
                elif error:
                    self.state.error = error
                    self._finish("failed", None, error)
                elif not launched:
                    self._finish("finished", 0, "Nothing to run")
                self._job_done.set()
                self._touch()

    def _finish(self, status: str, exit_code: int | None, message: str) -> None:
        with self.lock:
            self._after_pending = False
            if self.state.status in ("finished", "failed", "stopped") and self.state.ended_at:
                return
            self.state.status = status
            self.state.exit_code = exit_code
            self.state.ended_at = time.time()
            self.state.message = message
            states = self.state.stage_states
            for stage_id, _label, _icon in STAGES:
                if states.get(stage_id) == "active":
                    states[stage_id] = "done" if status == "finished" else ("failed" if status == "failed" else "stopped")
                elif status == "finished" and states.get(stage_id) == "pending":
                    states[stage_id] = "skipped" if stage_id != "finish" else "done"
            self.state.activity = ""
            self._touch()
        self._add_line(f"[gui] {message}")
        if status in ("finished", "failed", "stopped"):
            self._close_log()

    def _spawn(self, run_cmd, kwargs: dict) -> subprocess.Popen:
        options = dict(kwargs)
        env = dict(options.pop("env", None) or os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8:backslashreplace")
        env.setdefault("COLUMNS", "160")
        env["SECOURSES_LIVE_METRICS"] = "1"
        for key in ("text", "universal_newlines", "encoding", "errors", "stdout", "stderr", "stdin", "bufsize"):
            options.pop(key, None)
        flags = options.pop("creationflags", 0) | CREATE_NO_WINDOW
        return subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=flags,
            **options,
        )

    def _mirror(self, chunk: bytes) -> None:
        stream = getattr(sys.stdout, "buffer", None)
        try:
            if stream is not None:
                stream.write(chunk)
                stream.flush()
            elif sys.stdout is not None:
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
        except (OSError, ValueError, AttributeError):
            pass

    def _pump(self, process: subprocess.Popen, final: bool, tail: collections.deque | None = None) -> None:
        stream = process.stdout
        buffer = b""
        try:
            while True:
                chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(4096)
                if not chunk:
                    break
                self._mirror(chunk)
                buffer += chunk
                hold_cr = buffer.endswith(b"\r")  # may be the first half of "\r\n"
                work = buffer[:-1] if hold_cr else buffer
                start = 0
                for match in DELIMITER_RE.finditer(work):
                    self._segment(work[start : match.start()], match.group() == b"\r", tail)
                    start = match.end()
                buffer = work[start:] + (b"\r" if hold_cr else b"")
                if len(buffer) > 1 << 16:
                    self._segment(buffer, False, tail)
                    buffer = b""
            if buffer.strip(b"\r\n"):
                self._segment(buffer.rstrip(b"\r\n"), False, tail)
            if self._pending_checkpoint is not None:
                self._record_checkpoint(self._pending_checkpoint)
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass
            code = process.wait()
            if final:
                self._process_exited(process, code)

    def _segment(self, raw: bytes, carriage_return: bool, tail: collections.deque | None) -> None:
        text = _strip_ansi(raw.decode("utf-8", "replace")).rstrip()
        if not text.strip():
            # A newline after a bar keeps that bar in the log. A bar drawn below the training bar
            # (sampling, model loading: tqdm moves down a line for each redraw) stays a live line only.
            if not carriage_return and self.progress_line and not self._progress_temporary:
                committed = self.progress_line
                self._set_progress_line("")
                self._add_line(committed)
            return
        progress = parse_tqdm(text)
        if progress is not None:
            self._on_progress(progress)
            temporary = self._seen_train_progress and not _is_train_bar(progress)
            if carriage_return or temporary:
                self._set_progress_line(text, temporary)
                now = time.time()
                if not temporary and now - self._last_progress_write > 15:
                    self._last_progress_write = now
                    self._write_log(text + "\n")
            else:
                self._set_progress_line("")
                self._add_line(text)
            if progress["rest"]:  # e.g. "steps: 50%|…| 20/40 [00:10<00:10]  generating sample images at step…"
                self._on_text(progress["rest"])
            return
        if tail is not None:
            tail.append(text)
        self._on_text(text)
        self._add_line(text)

    def _average_loss(self, postfix: dict, step: int) -> float | None:
        """The trainer's moving-average loss; trainers that show only the per-step loss (LTX 2.3
        fine-tuning) get the mean of their last steps, like the other trainers' avr_loss."""
        average = _number(postfix.get("avr_loss") or postfix.get("avg"))
        step_loss = _number(postfix.get("loss"))
        if step_loss is not None:
            self.state.loss = step_loss
        if average is not None or step_loss is None:
            return average
        if math.isfinite(step_loss):
            if step == self._loss_step and self._step_losses:
                self._step_losses[-1] = step_loss  # the same step redrawn
            else:
                self._step_losses.append(step_loss)
                self._loss_step = step
        return sum(self._step_losses) / len(self._step_losses) if self._step_losses else step_loss

    def _on_progress(self, progress: dict) -> None:
        now = time.time()
        if _is_train_bar(progress):
            self._seen_train_progress = True
            self._advance("train")
            postfix = progress["postfix"]
            with self.lock:
                state = self.state
                state.step = progress["n"]
                state.total_steps = progress["total"]
                state.train_elapsed = progress["elapsed"]
                state.train_remaining = progress["remaining"]
                if progress["seconds_per_it"]:
                    state.seconds_per_it = progress["seconds_per_it"]
                state.last_progress_at = now
                value = self._average_loss(postfix, state.step)
                if value is not None:
                    state.avr_loss = value
                    if math.isfinite(value):  # a NaN / inf loss is shown, but never becomes "best" or a chart point
                        if state.step > 0:
                            self.loss_series.add(state.step, value)
                        if state.best_loss is None or value < state.best_loss:
                            state.best_loss = value
                lr = _number(postfix.get("lr"))
                if lr is not None:
                    state.lr = lr
                    if state.step > 0 and math.isfinite(lr):
                        self.lr_series.add(state.step, lr)
                if state.total_steps and state.step >= state.total_steps:
                    state.activity = "Saving final model"
                    state.activity_at = now
                self._touch()
            if self.state.total_steps and self.state.step >= self.state.total_steps:
                self._advance("finish", "Saving the final model")
            return
        # any other bar: caching / preparation progress
        if not self._seen_train_progress and self.state.current_stage in ("prepare", ""):
            self._advance("latents" if "text" not in progress["desc"].lower() else "text")
        with self.lock:
            self.state.stage_progress = {
                "desc": progress["desc"],
                "n": progress["n"],
                "total": progress["total"],
                "pct": progress["pct"],
                "elapsed": progress["elapsed"],
                "remaining": progress["remaining"],
                "seconds_per_it": progress["seconds_per_it"],
            }
            self._touch()

    def _record_checkpoint(self, path: str) -> None:
        self._pending_checkpoint = None
        name = os.path.basename(path.strip().strip("'\"").rstrip("/\\"))
        if not name:
            return
        with self.lock:
            self.state.checkpoints.append({"name": name, "time": time.time(), "step": self.state.step, "epoch": self.state.epoch})
            if len(self.state.checkpoints) > 50:
                del self.state.checkpoints[:-50]
            self.state.activity = f"Saved {name}"
            self.state.activity_at = time.time()
            self._touch()

    def _on_text(self, text: str) -> None:
        if self._pending_checkpoint is not None:
            if self._pending_checkpoint_lines < 4 and RICH_CONTINUATION_RE.match(text):
                self._pending_checkpoint += RICH_SOURCE_RE.sub("", text).strip()
                self._pending_checkpoint_lines += 1
                if MODEL_FILE_RE.search(self._pending_checkpoint):
                    self._record_checkpoint(self._pending_checkpoint)
                return
            self._record_checkpoint(self._pending_checkpoint)  # it named a folder, or the wrap ended
        stripped = RICH_SOURCE_RE.sub("", text).strip()
        epoch = EPOCH_RE.match(stripped)
        if epoch:
            with self.lock:
                self.state.epoch = int(epoch.group(1))
                self.state.total_epochs = int(epoch.group(2))
                self._touch()
            return
        if not self._seen_train_progress:
            for pattern, stage in TEXT_CUES:
                if pattern.search(stripped):
                    self._last_cache_cue = time.time()
                    self._advance(stage)
                    break
        checkpoint = CHECKPOINT_RE.search(stripped)
        if checkpoint:
            path = checkpoint.group(1).strip().strip("'\"")
            if MODEL_FILE_RE.search(path):
                self._record_checkpoint(path)
            else:  # sd-scripts' Rich log wraps a long path onto the next lines
                self._pending_checkpoint, self._pending_checkpoint_lines = path, 0
            return
        if SAMPLE_RE.search(stripped):
            with self.lock:
                if SAMPLE_ROUND_RE.search(stripped):
                    self.state.samples += 1
                self.state.activity = "Generating sample previews"
                self.state.activity_at = time.time()
                self._touch()
        # Every traceback is kept: accelerate prints its own CalledProcessError traceback after the trainer's.
        if ERROR_RE.match(stripped) or OOM_RE.search(stripped):
            with self.lock:
                self._error_lines.append(stripped[:400])
                self._touch()

    def _process_exited(self, process: subprocess.Popen, code: int) -> None:
        with self.lock:
            if process is not self.process:
                return
        if self._cancel_requested:
            self._finish("stopped", code, "Stopped by user")
        elif code == 0:
            # the run did every step and saved its model, even when the bar's last redraw never reached us (an
            # sd-scripts FLUX.1 run ended on 2,599 / 2,600, so "Saving & finishing" showed as skipped)
            with self.lock:
                if self.state.total_steps and self.state.step < self.state.total_steps:
                    self.state.step = self.state.total_steps
            self._advance("finish")
            after = self.state.after
            if after is not None:
                self._run_after(after, "Training finished successfully ✓")
            else:
                self._finish("finished", 0, "Training finished successfully ✓")
        else:
            errors = list(self._error_lines)
            causes = [line for line in errors if "CalledProcessError" not in line] or errors  # accelerate's wrapper last
            reason = causes[-1] if causes else ""
            if any(OOM_RE.search(line) for line in errors):
                reason = "CUDA out of memory — pick a preset for a smaller GPU tier, increase block swap or lower the resolution."
            with self.lock:
                self.state.error = reason or f"The trainer exited with code {code}. See the Training Status log."
            self._finish("failed", code, f"Training failed (exit code {code})")

    def _run_after(self, after: AfterStep, prefix: str) -> None:
        """The after-step of a successful training (``prefix``: its message), or a job of its own (no prefix): streamed
        into the run like training and ended by Stop."""
        self._advance("score", after.label)
        self._set_message(f"{after.label}…")
        code, tail = None, collections.deque(maxlen=40)
        if not self._cancel_requested:
            self._add_line(f"[gui] {after.label}: {' '.join(str(part) for part in after.cmd)}")
            try:
                process = self._spawn(after.cmd, {"env": after.env, "cwd": after.cwd})
            except OSError as exc:
                tail.append(f"{type(exc).__name__}: {exc}")
            else:
                self._adopt(process, None)
                self._ensure_monitor()
                self._pump(process, False, tail)
                code = process.returncode
        result = None
        if code == 0 and not self._cancel_requested and after.read_result is not None:
            try:
                result = after.read_result()
            except Exception as exc:  # noqa: BLE001 - an unreadable result is the step's failure, never the run's
                tail.append(f"{type(exc).__name__}: {exc}")
        with self.lock:
            self.state.result = result
            if self._cancel_requested:
                outcome, note = "stopped", f"{after.label} stopped"
            elif result:
                outcome, note = "done", str(result.get("headline") or f"{after.label}: done")
            else:
                lines = [line.strip() for line in tail if line.strip()]
                reason = next((line.split(":", 1)[1].strip() for line in reversed(lines) if line.startswith("BEST_CHECKPOINT_ERROR:")), "")
                reason = reason or next((line for line in reversed(lines) if ERROR_RE.match(line)), "") or (lines[-1] if lines else "")
                outcome, note = "failed", f"{after.label} failed" + (f": {reason[:300]}" if reason else f" (exit code {code})")
                self.state.stage_detail["score"] = reason[:300]
            self.state.stage_states["score"] = outcome
        if prefix:  # the training itself succeeded, whatever the after-step did
            self._finish("finished", 0, f"{prefix} · {note}")
        else:
            status = {"done": "finished", "stopped": "stopped"}.get(outcome, "failed")
            self._finish(status, 0 if status == "finished" else code, note)

    # ------------------------------------------------------------------ monitor thread (phase + GPU)
    def _ensure_monitor(self) -> None:
        if self._monitor is not None and self._monitor.is_alive():
            return
        self._monitor = threading.Thread(target=self._monitor_loop, name="training-monitor", daemon=True)
        self._monitor.start()

    def _monitor_loop(self) -> None:
        last_gpu = 0.0
        while self.is_active():
            now = time.time()
            process = self.process
            if process is not None and process.poll() is None:
                phase = self._process_phase(process.pid)
                if phase == "train":
                    # The trainer process is up. Kohya caches inside that process, so only its step
                    # bar moves the pipeline to Training; musubi caches in separate processes before.
                    kohya = self.state.meta.get("engine") == "kohya"
                    if self._seen_train_progress or (not kohya and now - self._last_cache_cue > 20):
                        self._advance("train", "" if self._seen_train_progress else "Loading the model")
                elif phase:
                    self._advance(phase)
            if now - last_gpu >= GPU_SAMPLE_SECONDS:
                last_gpu = now
                gpus = query_gpus()
                if gpus:
                    busy = self._buses_in_use(process)
                    with self.lock:
                        for gpu in gpus:
                            peak = max(self.state.peak_gb.get(gpu["index"], 0.0), gpu["used"])
                            self.state.peak_gb[gpu["index"]] = peak
                            gpu["peak"] = peak
                            gpu["in_use"] = gpu.get("bus") in busy
                        self.state.gpus = gpus
                        self._touch()
            time.sleep(PHASE_SAMPLE_SECONDS)

    @staticmethod
    def _buses_in_use(process) -> set[str]:
        """PCI bus IDs of the GPUs that this run's processes have opened (empty before CUDA starts)."""
        if process is None or process.poll() is not None:
            return set()
        try:
            root = psutil.Process(process.pid)
            pids = {root.pid, *(child.pid for child in root.children(recursive=True))}
        except psutil.Error:
            return set()
        return {bus for pid, bus in query_gpu_processes().items() if pid in pids}

    @staticmethod
    def _process_phase(pid: int) -> str:
        try:
            children = psutil.Process(pid).children(recursive=True)
        except psutil.Error:
            return ""
        for child in reversed(children):
            try:
                command = " ".join(child.cmdline()).replace("\\", "/")
            except psutil.Error:
                continue
            for needle, phase in PROCESS_PHASES:
                if needle in command:
                    return phase
        return ""

    @staticmethod
    def _kill_tree(pid: int) -> None:
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            return
        try:
            children = parent.children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            parent.kill()
        except psutil.Error:
            pass


def _nvidia_smi(query: str) -> list[list[str]]:
    try:
        output = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=CREATE_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[part.strip() for part in row.split(",")] for row in output.strip().splitlines()]


def query_gpus() -> list[dict]:
    """Per-GPU memory / utilization through nvidia-smi (never creates a CUDA context here)."""
    gpus = []
    for parts in _nvidia_smi("--query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu"):
        if len(parts) != 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "bus": parts[1].upper(),
                    "name": parts[2].replace("NVIDIA ", "").replace("GeForce ", ""),
                    "used": round(float(parts[3]) / 1024, 2),
                    "total": round(float(parts[4]) / 1024, 2),
                    "util": int(float(parts[5])),
                }
            )
        except ValueError:
            continue
    return gpus


def query_gpu_processes() -> dict[int, str]:
    """PID -> PCI bus ID of every process that has a GPU open. nvidia-smi numbers GPUs in PCI order,
    which can differ from CUDA's device order, so the bus ID is the reliable link to the run."""
    processes = {}
    for parts in _nvidia_smi("--query-compute-apps=pid,gpu_bus_id"):
        if len(parts) == 2 and parts[0].isdigit():
            processes[int(parts[0])] = parts[1].upper()
    return processes


RUNNER = TrainingRunner()


def install_runner() -> TrainingRunner:
    """Route every trainer module's process launching through the shared runner."""
    from . import legacy_tabs
    from .. import common_gui

    legacy_tabs.install_executor(RUNNER)

    def run_subprocess_with_captured_errors(cmd, env, *, label: str, tail_lines: int = 40, executor=None):
        code, tail = RUNNER.run_blocking(cmd, env, label=label)
        if RUNNER._cancel_requested:
            raise common_gui.TrainingCancelled(f"{label} stopped by user")
        if code != 0:
            output_tail = "\n".join(tail[-tail_lines:])
            hint = common_gui.explain_subprocess_failure(output_tail)
            message = f"{label} failed with return code {code}."
            if hint:
                message += f"\n\nLikely cause: {hint}"
            if output_tail:
                message += f"\n\nLast output:\n{output_tail}"
            raise RuntimeError(message)
        return code

    common_gui.run_subprocess_with_captured_errors = run_subprocess_with_captured_errors
    for backend in legacy_tabs.LEGACY_BACKENDS.values():
        module = legacy_tabs.legacy_module(backend)
        if hasattr(module, "run_subprocess_with_captured_errors"):
            module.run_subprocess_with_captured_errors = run_subprocess_with_captured_errors
    return RUNNER
