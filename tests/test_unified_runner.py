"""The unified UI's training runner and live monitor (musubi_tuner_gui/unified/runner.py, monitor.py, dashboard.py).

Runs short fake trainer processes on the CPU (or feeds their output straight to the parser); nothing is trained.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import threading
import time

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from musubi_tuner_gui.common_gui import TrainingCancelled  # noqa: E402,F401  (imported once here, not mid-test)
from musubi_tuner_gui.unified import monitor  # noqa: E402
from musubi_tuner_gui.unified.dashboard import render_dashboard  # noqa: E402
from musubi_tuner_gui.unified.runner import TrainingRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _private_state(tmp_path, monkeypatch):
    """Run logs go to a temporary folder; the GPUs are never queried."""
    from musubi_tuner_gui.unified import runner

    monkeypatch.setattr(runner, "RUN_LOG_DIR", str(tmp_path / "training_runs"))
    monkeypatch.setattr(runner, "query_gpus", lambda: [])
    monkeypatch.setattr(runner, "query_gpu_processes", lambda: {})


def _python(code: str) -> list[str]:
    return [sys.executable, "-u", "-c", code]


def _wait(runner: TrainingRunner, seconds: float = 30.0) -> None:
    deadline = time.time() + seconds
    while runner.is_active() and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.2)  # the output thread's last bookkeeping


class _Output:
    """A finished process whose output is ``data`` (fed through the runner's real stream reader)."""

    def __init__(self, data: bytes):
        self.stdout = io.BytesIO(data)
        self.returncode = 0

    def wait(self):
        return 0


def _fresh_runner() -> TrainingRunner:
    runner = TrainingRunner()
    runner._job_counter = 1
    runner._reset(title="unit", subtitle="", meta={})
    return runner


def _feed(runner: TrainingRunner, text: str) -> None:
    runner._pump(_Output(text.encode("utf-8")), False)


# --------------------------------------------------------------------------------------------- Stop
def test_stop_while_the_job_is_still_preparing_never_starts_the_trainer(tmp_path):
    marker = tmp_path / "trainer_ran.txt"
    for blocking in (False, True):
        runner = TrainingRunner()
        code = f"open({str(marker)!r}, 'w').write('ran')"

        def target():
            time.sleep(0.6)  # dataset TOML, audio extraction, runtime TOML...
            if blocking:  # a caching step through run_subprocess_with_captured_errors
                runner.run_blocking(_python(code), None, label="Caching latents")
            else:
                runner.execute_command(_python(code))

        started, _message = runner.start_job(target, title="stop early", meta={}, wait=0.1)
        assert started
        time.sleep(0.15)
        assert runner.stop() == "Stop requested."
        assert runner.snapshot()["status"] == "stopping"
        _wait(runner)
        snapshot = runner.snapshot()
        assert snapshot["status"] == "stopped", snapshot["message"]
        assert runner.process is None and not marker.exists()


def test_a_stop_racing_the_process_start_kills_the_new_process():
    runner = _fresh_runner()
    runner._job_thread_ident = threading.get_ident()
    original = runner._spawn

    def spawn_then_stop(*args, **kwargs):
        process = original(*args, **kwargs)
        runner._cancel_requested = True  # Stop pressed while the process was being created
        return process

    runner._spawn = spawn_then_stop
    runner.execute_command(_python("import time\ntime.sleep(30)"))
    process = runner.process
    process.wait(timeout=20)
    assert runner.snapshot()["status"] != "running"


# --------------------------------------------------------------------------------------------- failures
@pytest.mark.parametrize(
    "error, expected",
    [
        ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB.", "CUDA out of memory —"),
        ("ValueError: blocks_to_swap must be below 40", "ValueError: blocks_to_swap must be below 40"),
    ],
)
def test_the_trainers_own_error_is_shown_not_accelerates_wrapper(error, expected):
    """Every trainer runs under ``accelerate launch``, which prints its own traceback after the trainer's."""
    output = (
        "Traceback (most recent call last):\n"
        '  File "train_network.py", line 10, in <module>\n'
        f"{error}\n"
        "Traceback (most recent call last):\n"
        '  File "accelerate/commands/launch.py", line 999, in simple_launcher\n'
        "    raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)\n"
        "subprocess.CalledProcessError: Command '['python', 'train_network.py']' returned non-zero exit status 1.\n"
    )
    runner = TrainingRunner()
    code = f"import sys\nsys.stdout.write({output!r})\nsys.exit(1)"
    runner.start_job(lambda: runner.execute_command(_python(code)), title="fails", meta={})
    _wait(runner)
    snapshot = runner.snapshot()
    assert snapshot["status"] == "failed" and snapshot["error"].startswith(expected), snapshot["error"]


# --------------------------------------------------------------------------------------------- output parsing
def test_kohya_rich_log_checkpoints_and_sample_rounds():
    runner = _fresh_runner()
    pad = " " * 20
    _feed(
        runner,
        "2026-10-11 21:12:53 INFO     generating sample images at step / サンプル画像生成 ステップ: 5"
        "                    train_util.py:6445\n"
        f"{pad}INFO     saving checkpoint: G:/out/my_sdxl-000001.safetensors          train_util.py:5942\n"
        f"{pad}INFO     saving checkpoint:                                         train_util.py:5942\n"
        f"{pad}         G:/Trainings/SDXL_Fine_Tuning/outputs/ohwx_man_sdxl_fullft_v2-step00001000.safetenso\n"
        f"{pad}         rs\n"
        "INFO:musubi_tuner.training.trainer_base:generating sample images at step / サンプル画像生成 ステップ: 10\n"
        "INFO:musubi_tuner.training.trainer_base:saving checkpoint: /workspace/out/my lora-000002.safetensors\n",
    )
    snapshot = runner.snapshot()
    assert [item["name"] for item in snapshot["checkpoints"]] == [
        "my_sdxl-000001.safetensors",
        "ohwx_man_sdxl_fullft_v2-step00001000.safetensors",
        "my lora-000002.safetensors",
    ]
    assert snapshot["samples"] == 2


def test_progress_bars_drawn_below_the_training_bar_stay_out_of_the_log():
    """Sampling opens its own bar while the 'steps' bar is still open: tqdm draws it one line down."""
    from tqdm import tqdm

    stream = io.StringIO()
    steps = tqdm(total=40, desc="steps", file=stream, mininterval=0, ascii=True)
    for step in range(1, 41):
        steps.update(1)
        steps.set_postfix(avr_loss=1.0 / step)
        if step in (20, 40):
            stream.write("\ngenerating sample images at step / サンプル画像生成 ステップ: %d\n" % step)
            for _ in tqdm(range(30), file=stream, mininterval=0, ascii=True):
                pass
    steps.close()
    runner = _fresh_runner()
    _feed(runner, stream.getvalue())
    bar_lines = [line for line in runner.lines if "%|" in line]
    assert len(bar_lines) <= 4, bar_lines  # one 'steps' line per sampling round plus the final bar
    snapshot = runner.snapshot()
    assert snapshot["step"] == 40 and snapshot["samples"] == 2


def test_text_printed_right_after_a_bar_is_still_read():
    runner = _fresh_runner()
    _feed(runner, "\rsteps:  50%|#####     | 20/40 [00:10<00:10, 2.00it/s, avr_loss=0.05]  generating sample images at step 20\n")
    snapshot = runner.snapshot()
    assert snapshot["samples"] == 1 and snapshot["step"] == 20


def test_nan_and_inf_losses_never_become_best_or_chart_points():
    runner = _fresh_runner()
    for step, loss in ((1, "nan"), (2, "0.30"), (3, "inf"), (4, "0.20"), (5, "nan")):
        _feed(runner, f"\rsteps:  1%|#| {step}/100 [00:0{step}<01:00, 1.00s/it, avr_loss={loss}, lr=1e-4]")
    snapshot = runner.snapshot()
    assert snapshot["best_loss"] == pytest.approx(0.20) and math.isnan(snapshot["avr_loss"])
    assert all(math.isfinite(y) for _x, y in snapshot["loss_series"])
    snapshot["loss_series"] = [(1, float("nan")), (2, 0.3), (3, float("inf")), (4, 0.2)]
    chart = render_dashboard(snapshot).split('<div class="ud-chart">', 1)[1].split("</svg>", 1)[0]
    assert "min 0.2000 · max 0.3000" in chart and "nan" not in chart.lower() and "inf" not in chart.lower()


def test_a_trainer_that_shows_only_the_step_loss_gets_a_moving_average():
    """LTX 2.3 fine-tuning reports the current step's loss; the tile and chart show an average like the others."""
    runner = _fresh_runner()
    for step in range(1, 21):
        loss = 1.0 if step % 2 else 0.0
        _feed(runner, f"\rsteps:  {step * 5}%|#| {step}/20 [00:{step:02d}<00:10, 1.00s/it, loss={loss}, lr=1e-05]")
    _feed(runner, "\rsteps: 100%|#| 20/20 [00:20<00:00, 1.00s/it, loss=0.0, lr=1e-05]")  # the same step redrawn
    snapshot = runner.snapshot()
    assert snapshot["avr_loss"] == pytest.approx(0.5) and snapshot["loss"] == 0.0
    runner2 = _fresh_runner()
    _feed(runner2, "\rsteps:  50%|#| 1/2 [00:01<00:01, 1.00s/it, avr_loss=0.42, loss=0.9]")
    assert runner2.snapshot()["avr_loss"] == pytest.approx(0.42)


def test_job_log_messages_appear_once():
    from musubi_tuner_gui.custom_logging import setup_logging

    log = setup_logging()
    runner = TrainingRunner()

    def target():
        log.info("[OK] Dataset TOML generated: unit-test.toml")

    runner.start_job(target, title="log once", meta={})
    _wait(runner)
    assert sum("unit-test.toml" in line for line in runner.lines) == 1


# --------------------------------------------------------------------------------------------- monitor payload
def test_monitor_sends_only_what_changed_and_long_lines_stay_small(monkeypatch):
    runner = _fresh_runner()
    runner.state.status = "running"
    monkeypatch.setattr(monitor, "RUNNER", runner)
    runner._add_line("load network weights: " + "x" * 300_000)
    assert len(runner.lines[-1]) < 2_100 and "more characters in the log file" in runner.lines[-1]

    first = monitor.monitor_poll([-1, -1, -1])
    assert {"dash", "log", "pl"} <= set(first) and len(json.dumps(first)) < 40_000
    seen = [first["v"], first["dv"], first["lv"]]
    idle = monitor.monitor_poll(seen)
    assert not {"dash", "log", "pl"} & set(idle) and len(json.dumps(idle)) < 200  # nothing changed, even while running

    runner._segment("steps:  10%|#| 10/100 [00:10<01:30, 1.00it/s, avr_loss=0.1]".encode(), True, None)  # a bar redraw
    redraw = monitor.monitor_poll(seen)
    assert "dash" in redraw and redraw["pl"].startswith("steps:") and "log" not in redraw
    seen = [redraw["v"], redraw["dv"], redraw["lv"]]
    runner._add_line("INFO: a new log line")
    lines = monitor.monitor_poll(seen)
    assert "log" in lines and "dash" not in lines and lines["log"].endswith("a new log line")


def test_a_finished_download_reaches_the_browser_with_its_status(monkeypatch):
    # the browser checks the model files again when a download finishes (the "— not found" chips)
    from musubi_tuner_gui.unified import downloads

    manager = downloads.DownloadManager()
    monkeypatch.setattr(monitor, "DOWNLOADS", manager)
    monkeypatch.setattr(monitor, "RUNNER", _fresh_runner())
    first = monitor.monitor_poll([-1, -1, -1])
    assert first["ds"] == "idle" and first["dl"] == ""
    seen = [first["v"], first["dv"], first["lv"]]
    assert "ds" not in monitor.monitor_poll(seen)  # unchanged: nothing about the download is sent
    with manager.lock:
        manager.status = "finished"
        manager.lines.append("=== FLUX.1 Dev models: finished ===")
        manager.version += 1
    done = monitor.monitor_poll(seen)
    assert done["ds"] == "finished" and done["dl"].endswith("finished ===")


def test_the_elapsed_clock_and_ago_texts_tick_in_the_browser():
    runner = _fresh_runner()
    runner.state.status = "running"
    runner._record_checkpoint("out/model-000001.safetensors")
    html = render_dashboard(runner.snapshot())
    assert f'data-since="{runner.state.started_at:.3f}"' in html and "data-ago=" in html
    runner._finish("finished", 0, "done")
    assert "data-since" not in render_dashboard(runner.snapshot())


def test_a_successful_run_ends_with_every_step_and_its_saving_stage_done():
    # sd-scripts' last bar redraw (2,600 / 2,600) can come with the exit unread: the run still did every step
    runner = TrainingRunner()
    code = (
        "import sys, time\n"
        "sys.stdout.write('steps:  99%|#########9| 39/40 [00:10<00:00, 3.90it/s, avr_loss=0.3]' + chr(13)); sys.stdout.flush()\n"
        "time.sleep(0.3)\n"
    )
    from musubi_tuner_gui.unified.runner import AfterStep

    for after in (None, AfterStep("Finding the best checkpoint", _python("print('scored')"))):
        runner = TrainingRunner()
        started, _message = runner.start_job(lambda: runner.execute_command(_python(code)), title="kohya", meta={}, after=after)
        assert started
        _wait(runner)
        snapshot = runner.snapshot()
        stages = {stage["id"]: stage["state"] for stage in snapshot["stages"]}
        assert snapshot["status"] == "finished" and snapshot["step"] == snapshot["total_steps"] == 40
        assert stages["train"] == "done" and stages["finish"] == "done"  # with the finder after it, it showed "skipped"
