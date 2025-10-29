import gradio as gr
from typing import Tuple
from .common_gui import (
    get_folder_path,
    get_any_file_path,
    list_files,
    list_dirs,
    create_refresh_button,
    document_symbol,
)


class AdvancedTraining:
    """
    This class configures and initializes the advanced training settings for a machine learning model,
    including options for headless operation, fine-tuning, training type selection, and default directory paths.

    Attributes:
        headless (bool): If True, run without the Gradio interface.
        finetuning (bool): If True, enables fine-tuning of the model.
        training_type (str): Specifies the type of training to perform.
        no_token_padding (gr.Checkbox): Checkbox to disable token padding.
        gradient_accumulation_steps (gr.Slider): Slider to set the number of gradient accumulation steps.
        weighted_captions (gr.Checkbox): Checkbox to enable weighted captions.
        debug_mode (gr.Dropdown): Dropdown to select debug mode for detailed logging and visualization.
    """

    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
    ) -> None:
        """
        Initializes the AdvancedTraining class with given settings.

        Parameters:
            headless (bool): Run in headless mode without GUI.
            finetuning (bool): Enable model fine-tuning.
            training_type (str): The type of training to be performed.
            config (dict): Configuration options for the training process.
        """
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config

        # Determine the current directories for VAE and output, falling back to defaults if not specified.
        self.current_vae_dir = self.config.get("advanced.vae_dir", "./models/vae")
        self.current_state_dir = self.config.get("advanced.state_dir", "./outputs")
        self.current_log_tracker_config_dir = self.config.get(
            "advanced.log_tracker_config_dir", "./logs"
        )

        # Handle migration of old debug mode values
        debug_mode_value = self.config.get("debug_mode", "None")
        debug_mode_value = self._migrate_debug_mode_value(debug_mode_value)

        with gr.Row():
            self.additional_parameters = gr.Textbox(
                label="Additional parameters",
                placeholder='(Optional) Use to provide additional parameters not handled by the GUI. Eg: --some_parameters "value"',
                value=self.config.get("additional_parameters", ""),
            )

        with gr.Row():
            self.debug_mode = gr.Dropdown(
                label="Debug Mode Selection",
                choices=[
                    "None",
                    "Show Timesteps (Image)",
                    "Show Timesteps (Console)",
                    "RCM Debug Save",
                    "Enable Logging (TensorBoard)",
                    "Enable Logging (WandB)",
                    "Enable Logging (All)",
                ],
                value=debug_mode_value,
                info="Select debug mode for training visualization and logging. Dataset debugging is handled separately in caching settings.",
                interactive=True,
                allow_custom_value=True,  # Allow old config values during migration
            )

    def get_debug_mode_description(self, debug_mode: str) -> str:
        """
        Get detailed description for the selected debug mode.

        Parameters:
            debug_mode (str): The selected debug mode

        Returns:
            str: Detailed description of the debug mode
        """
        descriptions = {
            "None": "No debug mode enabled - standard operation.",
            "Show Timesteps (Image)": "Visualizes timestep distribution used during training with matplotlib plots. Helps understand noise scheduling.",
            "Show Timesteps (Console)": "Displays timestep information in console format. Useful for debugging noise scheduling without GUI.",
            "RCM Debug Save": "Saves dynamically generated RCM (Regional Content Mask) masks during generation. Essential for debugging and adjusting RCM parameters.",
            "Enable Logging (TensorBoard)": "Enables TensorBoard logging for training metrics, losses, and visualizations. Requires logging directory to be set.",
            "Enable Logging (WandB)": "Enables Weights & Biases logging for experiment tracking and visualization. Requires WandB setup.",
            "Enable Logging (All)": "Enables both TensorBoard and WandB logging simultaneously. Provides comprehensive experiment tracking.",
        }
        return descriptions.get(debug_mode, "Unknown debug mode")

    def get_debug_parameters(self, debug_mode: str) -> str:
        """
        Convert selected debug mode to command-line parameters.

        Parameters:
            debug_mode (str): The selected debug mode

        Returns:
            str: Command-line parameters for the selected debug mode
        """
        debug_params = {
            "Show Timesteps (Image)": "--show_timesteps image",
            "Show Timesteps (Console)": "--show_timesteps console",
            "RCM Debug Save": "--rcm_debug_save",
            "Enable Logging (TensorBoard)": "--log_with tensorboard --logging_dir ./logs",
            "Enable Logging (WandB)": "--log_with wandb",
            "Enable Logging (All)": "--log_with all --logging_dir ./logs",
        }
        return debug_params.get(debug_mode, "")

    def _migrate_debug_mode_value(self, old_value: str) -> str:
        """
        Migrate old debug mode values to new valid ones.

        Parameters:
            old_value (str): The old debug mode value from config

        Returns:
            str: The migrated debug mode value
        """
        migration_map = {
            "Dataset Debug (Image)": "None",  # Dataset debugging moved to caching section
            "Dataset Debug (Console)": "None",  # Dataset debugging moved to caching section
            "Dataset Debug (Video)": "None",  # Dataset debugging moved to caching section
        }
        return migration_map.get(old_value, old_value)