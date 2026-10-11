import subprocess
import psutil
import time
import os
import gradio as gr

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Child-process script names -> phase shown by the live training monitor (first match wins).
MONITOR_PHASES = (
    ("merge_fft_to_model", "Merging the full fine-tune into an inference bundle"),
    ("cache_latents", "Caching latents"),
    ("cache_pixel", "Caching pixel tokens"),
    ("cache_text_encoder", "Caching text-encoder outputs"),
    ("train_network", "Training"),
    ("_train.py", "Training"),
)


def _gpu_lines() -> list[tuple[int, str, float, float, int]]:
    """Live per-GPU memory/utilization via nvidia-smi (never creates a CUDA context in the GUI)."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=0x08000000 if os.name == "nt" else 0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    lines = []
    for row in output.strip().splitlines():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 5:
            try:
                lines.append((int(parts[0]), parts[1], float(parts[2]) / 1024, float(parts[3]) / 1024, int(float(parts[4]))))
            except ValueError:
                continue
    return lines


def _format_elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class CommandExecutor:
    """
    A class to execute and manage commands.
    """

    def __init__(self, headless: bool = False):
        """
        Initialize the CommandExecutor.
        """
        self.headless = headless
        self.process = None
        self._cancel_requested = False
        self.started_at = None
        self._peak_gb: dict[int, float] = {}

        with gr.Row():
            self.button_run = gr.Button(
                "Start training",
                variant="primary",
                elem_classes=["mbtn", "mbtn-emerald"],
            )
        
        # Training status indicator
        with gr.Row():
            self.training_status = gr.Textbox(
                label="Training Status",
                value="Ready",
                interactive=False
            )

        # Stop training controls - now under status
        with gr.Row(visible=self.process is not None or headless) as self.stop_row:
            self.button_stop_training = gr.Button(
                "Stop training",
                variant="stop",
                interactive=True,
                elem_id="stop_training_btn",
                elem_classes=["mbtn", "mbtn-red"],
            )

        # Live training monitor: phase, elapsed time and GPU memory while a run is in flight.
        self.monitor = gr.Markdown(visible=False, elem_classes=["training-monitor"])
        self.monitor_timer = gr.Timer(value=3.0, active=False)
        self.monitor_timer.tick(self._monitor_tick, outputs=[self.monitor, self.monitor_timer], show_progress="hidden")
        # A second listener on the tab's Start button; the tick deactivates itself when nothing runs.
        self.button_run.click(lambda: gr.Timer(active=True), outputs=[self.monitor_timer], queue=False, show_progress="hidden")

    def execute_command(self, run_cmd: str, **kwargs):
        """
        Execute a command if no other command is currently running.

        Parameters:
        - run_cmd (str): The command to execute.
        - **kwargs: Additional keyword arguments to pass to subprocess.Popen.
        """
        if self.process and self.process.poll() is None:
            log.info("The command is already running. Please wait for it to finish.")
        else:
            self._cancel_requested = False
            # for i, item in enumerate(run_cmd):
            #     log.info(f"{i}: {item}")

            # Reconstruct the safe command string for display
            command_to_run = " ".join(run_cmd)
            log.info(f"Executing command: {command_to_run}")

            # Execute the command securely
            self.process = subprocess.Popen(run_cmd, **kwargs)
            self.started_at = time.time()
            self._peak_gb = {}
            log.debug("Command executed.")

    def kill_command(self):
        """
        Kill the currently running command and its child processes.
        """
        if self.is_running():
            try:
                self._cancel_requested = True
                # Get the parent process and kill all its children (including any text encoder caching)
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                
                # Log which processes are being killed for debugging
                if children:
                    log.info(f"Terminating {len(children)} child process(es)...")
                    for child in children:
                        try:
                            log.debug(f"Killing child process: PID={child.pid}, Name={child.name()}")
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                            log.debug(f"Could not kill child process {child.pid}: {e}")
                
                # Kill the parent process
                parent.kill()
                log.info("The running process has been terminated.")
                gr.Warning("Training/caching is being cancelled... Please wait for the process to stop.")
                status_msg = "Training stopped by user"
                
                # Clean up any temporary script files that may have been created
                import glob
                import tempfile
                temp_dir = tempfile.gettempdir()
                for temp_script in glob.glob(os.path.join(temp_dir, "tmp*.bat")) + glob.glob(os.path.join(temp_dir, "tmp*.sh")):
                    try:
                        # Only delete if file is older than current session (safety check)
                        if os.path.exists(temp_script) and os.path.getmtime(temp_script) < time.time() - 1:
                            os.remove(temp_script)
                            log.debug(f"Cleaned up temporary script: {temp_script}")
                    except Exception as e:
                        log.debug(f"Could not clean up {temp_script}: {e}")
                        
            except psutil.NoSuchProcess:
                # Explicitly handle the case where the process does not exist
                log.info(
                    "The process does not exist. It might have terminated before the kill command was issued."
                )
                status_msg = "Process already terminated"
            except Exception as e:
                # General exception handling for any other errors
                log.info(f"Error when terminating process: {e}")
                status_msg = f"Error stopping: {e}"
        else:
            self.process = None
            log.info("There is no running process to kill.")
            status_msg = "No process to stop"

        return (
            gr.Button(visible=True),  # Show start button
            gr.Row(visible=False or self.headless),  # Hide stop row
            gr.Button(interactive=True),  # Keep stop button enabled
            gr.Textbox(value=status_msg)  # Update status
        )

    def wait_for_training_to_end(self):
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")
        
        # A user-requested termination is not a training failure.
        if self._cancel_requested:
            log.info("Training stopped by user.")
            status_msg = "Training stopped by user"
            self._cancel_requested = False
        elif self.process and self.process.returncode != 0:
            log.error(f"Training failed with exit code: {self.process.returncode}")
            gr.Warning(f"Training failed with exit code: {self.process.returncode}. Check console for details.")
            status_msg = f"Training failed (exit code: {self.process.returncode}). Check console for details."
        else:
            log.info("Training completed successfully.")
            gr.Info("Training completed successfully!")
            status_msg = "Training completed successfully"
        
        return (
            gr.Button(visible=True),  # Show start button
            gr.Row(visible=False or self.headless),  # Hide stop row
            gr.Button(interactive=True),  # Keep stop button enabled
            gr.Textbox(value=status_msg)  # Update status
        )

    def _phase(self) -> str:
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
        except (psutil.Error, AttributeError):
            return "Starting"
        for child in reversed(children):
            try:
                command = " ".join(child.cmdline()).replace("\\", "/")
            except psutil.Error:
                continue
            for needle, phase in MONITOR_PHASES:
                if needle in command:
                    return phase
        return "Starting"

    def _monitor_tick(self):
        if self.process is None or self.started_at is None:
            return gr.Markdown(visible=False), gr.Timer(active=False)
        running = self.process.poll() is None
        elapsed = _format_elapsed(time.time() - self.started_at)
        gpu_parts = []
        for index, name, used, total, utilization in _gpu_lines():
            self._peak_gb[index] = max(self._peak_gb.get(index, 0.0), used)
            gpu_parts.append(f"GPU {index} ({name}): **{used:.1f} / {total:.1f} GB** · {utilization}% · peak seen {self._peak_gb[index]:.1f} GB")
        if running:
            header = f"🟢 **{self._phase()}** · elapsed {elapsed}"
        elif self._cancel_requested:
            header = f"⏹️ **Stopped by user** after {elapsed}"
        elif self.process.returncode == 0:
            header = f"✅ **Finished** in {elapsed}"
        else:
            header = f"❌ **Failed** (exit code {self.process.returncode}) after {elapsed} - see the console for the error"
        lines = [header] + gpu_parts
        if running:
            lines.append("Step-by-step progress (loss, s/it, ETA) is printed in the console window.")
        return gr.Markdown(value="  \n".join(lines), visible=True), gr.Timer(active=running)

    def is_running(self):
        """
        Check if the command is currently running.

        Returns:
        - bool: True if the command is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None
    
    def wait_for_training_to_end_simple(self):
        """
        Simplified version for lora_gui compatibility - returns only 2 values
        """
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")
        
        # Check if process ended with error
        if self._cancel_requested:
            log.info("Training stopped by user.")
            self._cancel_requested = False
        elif self.process and self.process.returncode != 0:
            log.error(f"Training failed with exit code: {self.process.returncode}")
        else:
            log.info("Training completed successfully.")
        
        return (
            gr.Button(visible=True),  # Show start button
            gr.Button(visible=False),  # Hide stop button
        )
    
    def kill_command_simple(self):
        """
        Simplified version for lora_gui compatibility - returns appropriate values
        """
        if self.is_running():
            try:
                self._cancel_requested = True
                # Get the parent process and kill all its children
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info("The running process has been terminated.")
                status_msg = "Training stopped by user"
            except psutil.NoSuchProcess:
                log.info("The process does not exist. It might have terminated before the kill command was issued.")
                status_msg = "Process already terminated"
            except Exception as e:
                log.info(f"Error when terminating process: {e}")
                status_msg = f"Error stopping: {e}"
        else:
            self.process = None
            log.info("There is no running process to kill.")
            status_msg = "No process to stop"

        return (
            gr.Button(visible=True),  # Show start button
            gr.Row(visible=False or self.headless),  # Hide stop row
            gr.Button(interactive=True),  # Keep stop button enabled
            gr.Textbox(value=status_msg)  # Update status
        )
