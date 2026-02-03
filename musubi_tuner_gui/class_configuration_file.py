import gradio as gr
import os
from .common_gui import list_files, scriptdir
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()


class ConfigurationFile:
    """
    A class to handle configuration file operations in the GUI.
    """

    def __init__(
        self, headless: bool = False, config_dir: str = None, config: dict = {}
    ):
        """
        Initialize the ConfigurationFile class.

        Parameters:
        - headless (bool): Whether to run in headless mode.
        - config_dir (str): The directory for configuration files.
        """

        self.headless = headless

        self.config = config

        # Sets the directory for storing configuration files, defaults to a 'presets' folder within the script directory.
        self.current_config_dir = self.config.get(
            "config_dir", os.path.join(scriptdir, "presets")
        )

        # Initialize the GUI components for configuration.
        self.create_config_gui()

    def list_config_dir(self, path: str) -> list:
        """
        List directories in the data directory.

        Parameters:
        - path (str): The path to list directories from.

        Returns:
        - list: A list of directories.
        """
        self.current_config_dir = path if not path == "" else "."
        # This app saves/loads training presets as TOML. Keep .json for legacy compatibility.
        return list(list_files(self.current_config_dir, exts=[".toml", ".json"], all=True))

    def create_config_gui(self) -> None:
        """
        Create the GUI for configuration file operations.
        """
        # Starts a new group in the GUI for better layout organization.
        with gr.Group():
            # Creates a row within the group to align elements horizontally.
            with gr.Row():
                # Dropdown for selecting or entering the name of a configuration file.
                self.config_file_name = gr.Dropdown(
                    label="Load/Save Config file",
                    choices=[self.config.get("config_dir", "")] + self.list_config_dir(self.current_config_dir),
                    value=self.config.get("config_dir", ""),
                    interactive=True,
                    allow_custom_value=True,
                    info="Select a preset .toml file (or type a path) to load or save settings.",
                )

                # Refresh button removed - auto-refresh happens on dropdown change

                # Buttons for opening, saving, and loading configuration files, displayed conditionally based on headless mode.
                self.button_open_config = gr.Button(
                    "📂 Open",
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                    visible=(not self.headless),
                )
                self.button_save_config = gr.Button(
                    "💾 Save",
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                )
                self.button_load_config = gr.Button(
                    "📥 Load",
                    elem_id="open_folder_small",
                    elem_classes=["tool"],
                )
            
            # Status display for configuration operations
            self.config_status = gr.Textbox(
                label="Configuration Status",
                value="",
                interactive=False,
                visible=False,
                elem_id="config_status",
                info="Shows status messages after save/load actions.",
            )
            
            # Note: Change handler for auto-load is now set in the parent GUI component
