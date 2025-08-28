import subprocess
import psutil
import time
import gradio as gr

from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


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
        
        with gr.Row():
            self.button_run = gr.Button("Start training", variant="primary")
        
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
                elem_id="stop_training_btn"
            )

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
            # for i, item in enumerate(run_cmd):
            #     log.info(f"{i}: {item}")

            # Reconstruct the safe command string for display
            command_to_run = " ".join(run_cmd)
            log.info(f"Executing command: {command_to_run}")

            # Execute the command securely
            self.process = subprocess.Popen(run_cmd, **kwargs)
            log.debug("Command executed.")

    def kill_command(self):
        """
        Kill the currently running command and its child processes.
        """
        if self.is_running():
            try:
                # Get the parent process and kill all its children
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info("The running process has been terminated.")
                gr.Warning("Training is being cancelled... Please wait for the process to stop.")
                status_msg = "Training stopped by user"
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
        
        # Check if process ended with error
        if self.process and self.process.returncode != 0:
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
        if self.process and self.process.returncode != 0:
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
