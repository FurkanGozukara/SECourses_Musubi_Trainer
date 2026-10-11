"""Background model downloads (Download_Train_Models.py bundles) for the Model Files section."""

from __future__ import annotations

import collections
import os
import subprocess
import sys
import threading
import time

from .schema import ROOT_DIR

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
DOWNLOADER = os.path.join(ROOT_DIR, "Download_Train_Models.py")


class DownloadManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.lines: collections.deque[str] = collections.deque(maxlen=200)
        self.process: subprocess.Popen | None = None
        self.bundle = ""
        self.status = "idle"  # idle | running | finished | failed | stopped
        self.version = 0
        self.started_at: float | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, bundle: str, label: str) -> str:
        if not os.path.isfile(DOWNLOADER):
            return f"Downloader not found: {DOWNLOADER}"
        with self.lock:
            if self.is_running():
                return f"A download is already running ({self.bundle})."
            self.lines.clear()
            self.bundle = label or bundle
            self.status = "running"
            self.started_at = time.time()
            self.lines.append(f"=== Downloading {label or bundle} into {ROOT_DIR} ===")
            self.version += 1
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8:backslashreplace"
            self.process = subprocess.Popen(
                [sys.executable, "-u", DOWNLOADER, "--model", bundle],
                cwd=ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                creationflags=CREATE_NO_WINDOW,
            )
        threading.Thread(target=self._pump, args=(self.process,), daemon=True).start()
        return f"Downloading {label or bundle}…"

    def stop(self) -> str:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return "No download is running."
            self.status = "stopped"  # before the kill: the pump thread then reports the end as stopped, not failed
            self.version += 1
        try:
            import psutil

            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except Exception:
            process.kill()
        return "Download stopped. Run it again later to resume."

    def _pump(self, process: subprocess.Popen) -> None:
        pending = b""
        for chunk in iter(lambda: process.stdout.read1(8192) if hasattr(process.stdout, "read1") else process.stdout.read(1024), b""):
            pending += chunk
            parts = pending.replace(b"\r\n", b"\n").split(b"\n")
            pending = parts.pop()
            with self.lock:
                for part in parts:
                    text = part.decode("utf-8", "replace").split("\r")[-1].rstrip()
                    if text:
                        self.lines.append(text)
                if b"\r" in pending:  # progress bar redraw: show its latest state
                    latest = pending.split(b"\r")[-1].decode("utf-8", "replace").strip() or pending.split(b"\r")[-2].decode("utf-8", "replace").strip()
                    if latest:
                        if self.lines and self.lines[-1].startswith("⏬ "):
                            self.lines[-1] = "⏬ " + latest
                        else:
                            self.lines.append("⏬ " + latest)
                    pending = b""
                self.version += 1
        code = process.wait()
        with self.lock:
            if self.status == "running":
                self.status = "finished" if code == 0 else "failed"
            ending = {"finished": "finished", "stopped": "stopped — Download again to resume"}.get(self.status, f"failed (exit code {code})")
            self.lines.append(f"=== {self.bundle}: {ending} ===")
            self.version += 1

    def snapshot(self) -> tuple[int, str, str]:
        with self.lock:
            return self.version, self.status, "\n".join(list(self.lines)[-40:])


DOWNLOADS = DownloadManager()
