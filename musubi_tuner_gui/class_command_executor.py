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

        # Stop training controls with confirmation
        with gr.Row(visible=self.process is not None or headless) as self.stop_row:
            self.stop_confirm_checkbox = gr.Checkbox(
                label="Confirm stop training",
                value=False,
                info="Check this box to confirm you want to stop training"
            )
            self.button_stop_training = gr.Button(
                "Stop training", 
                variant="stop",
                interactive=False
            )
        
        # Training status indicator
        with gr.Row():
            self.training_status = gr.Textbox(
                label="Training Status",
                value="Ready",
                interactive=False
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

    def kill_command(self, confirm_checked):
        """
        Kill the currently running command and its child processes.
        Requires confirmation checkbox to be checked.
        """
        if not confirm_checked:
            log.warning("Please check the confirmation box to stop training.")
            return (
                gr.Button(visible=False), 
                gr.Row(visible=True),
                gr.Checkbox(value=False),
                gr.Button(interactive=False),
                gr.Textbox(value="⚠️ Please check confirmation box to stop")
            )
        
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
            gr.Checkbox(value=False),  # Reset checkbox
            gr.Button(interactive=False),  # Disable stop button
            gr.Textbox(value=status_msg)  # Update status
        )

    def wait_for_training_to_end(self):
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")
        
        # Check if process ended with error
        if self.process and self.process.returncode != 0:
            log.error(f"Training failed with exit code: {self.process.returncode}")
            status_msg = f"⚠️ Training failed (exit code: {self.process.returncode}). Check console for details."
        else:
            log.info("Training completed successfully.")
            status_msg = "✅ Training completed successfully"
        
        return (
            gr.Button(visible=True),  # Show start button
            gr.Row(visible=False or self.headless),  # Hide stop row
            gr.Checkbox(value=False),  # Reset checkbox
            gr.Button(interactive=False),  # Disable stop button
            gr.Textbox(value=status_msg)  # Update status
        )

    def is_running(self):
        """
        Check if the command is currently running.

        Returns:
        - bool: True if the command is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None
    
    def toggle_stop_button(self, checkbox_value):
        """
        Enable/disable stop button based on checkbox state.
        """
        return gr.Button(interactive=checkbox_value)
