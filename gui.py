import os
import sys
import argparse
import subprocess
import contextlib
import gradio as gr

from musubi_tuner_gui.lora_gui import lora_tab
from musubi_tuner_gui.qwen_image_lora_gui import qwen_image_lora_tab
from musubi_tuner_gui.wan_lora_gui import wan_lora_tab
from musubi_tuner_gui.image_captioning_gui import image_captioning_tab
from musubi_tuner_gui.fp8_converter_gui import fp8_converter_tab
from musubi_tuner_gui.image_preprocessing_gui import image_preprocessing_tab
from musubi_tuner_gui.changelog_gui import version_history_tab
from musubi_tuner_gui.lora_extractor_gui import lora_extractor_tab
from musubi_tuner_gui.lora_merge_gui import lora_merge_tab
from musubi_tuner_gui.custom_logging import setup_logging
from musubi_tuner_gui.class_gui_config import GUIConfig
from musubi_tuner_gui.class_tab_config_manager import TabConfigManager
import toml

PYTHON = sys.executable
project_dir = os.path.dirname(os.path.abspath(__file__))

# Function to read file content, suppressing any FileNotFoundError
def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""

# Function to initialize the Gradio UI interface
def initialize_ui_interface(config_manager, headless, release_info, readme_content):
    # Load custom CSS if available
    css = read_file_content("./assets/style.css")

    # Create the main Gradio Blocks interface
    ui_interface = gr.Blocks(css=css, title="SECourses Musubi Trainer V24.0", theme=gr.themes.Soft())
    with ui_interface:
        # Add title with Patreon link
        gr.Markdown("# SECourses Musubi Trainer V25.0 : [https://www.patreon.com/posts/137551634](https://www.patreon.com/posts/137551634)")
        
        # Create tabs for different functionalities
        with gr.Tab("Qwen Image Training"):
            qwen_config = config_manager.get_config_for_tab("qwen_image")
            qwen_image_lora_tab(headless=headless, config=qwen_config)
        
        with gr.Tab("Wan Models Training"):
            wan_config = config_manager.get_config_for_tab("wan")
            wan_lora_tab(headless=headless, config=wan_config)
        
        with gr.Tab("Image Captioning"):
            captioning_config = config_manager.get_config_for_tab("image_captioning")
            image_captioning_tab(headless=headless, config=captioning_config)
        
        with gr.Tab("FP8 Model Converter"):
            fp8_config = config_manager.get_config_for_tab("fp8_converter")
            fp8_converter_tab(headless=headless, config=fp8_config)

        with gr.Tab("LoRA Extractor"):
            lora_extractor_tab(headless=headless, config=None)

        with gr.Tab("LoRA Merger"):
            lora_merge_tab(headless=headless, config=None)
        
        with gr.Tab("Image Preprocessing"):
            preprocessing_config = config_manager.get_config_for_tab("image_preprocessing")
            image_preprocessing_tab(headless=headless, config=preprocessing_config)
        
        with gr.Tab("Version History"):
            version_history_tab(headless=headless, config=None)
            
        with gr.Tab("Musubi Tuner (Deprecated)"):
            musubi_config = config_manager.get_config_for_tab("musubi_tuner")
            lora_tab(headless=headless, config=musubi_config)

    return ui_interface

# Function to configure and launch the UI
def UI(**kwargs):
    # Add custom JavaScript if specified
    log.info(f"headless: {kwargs.get('headless', False)}")

    # Load release and README information
    release_info = "v18.0"  # Hardcoded version since pyproject.toml is not needed
    
    readme_content = read_file_content("./README.md")
    
    # Initialize tab-aware configuration manager - default to qwen_image_defaults.toml
    config_manager = TabConfigManager(config_file_path=kwargs.get("config", "./qwen_image_defaults.toml"))
    if config_manager.user_loaded_config:
        log.info(f"Loaded user configuration from '{kwargs.get('config', './qwen_image_defaults.toml')}'...")
    else:
        log.info("No user config loaded - will use tab-specific defaults")

    # Initialize the Gradio UI interface
    ui_interface = initialize_ui_interface(config_manager, kwargs.get("headless", False), release_info, readme_content)

    # Construct launch parameters using dictionary comprehension
    launch_params = {
        "server_name": kwargs.get("listen"),
        "auth": (kwargs["username"], kwargs["password"]) if kwargs.get("username") and kwargs.get("password") else None,
        "server_port": kwargs.get("server_port", 0) if kwargs.get("server_port", 0) > 0 else None,
        "inbrowser": kwargs.get("inbrowser", True),
        "share": kwargs.get("share", False),
        "root_path": kwargs.get("root_path", None),
        "debug": kwargs.get("debug", False),
    }
  
    # This line filters out any key-value pairs from `launch_params` where the value is `None`, ensuring only valid parameters are passed to the `launch` function.
    launch_params = {k: v for k, v in launch_params.items() if v is not None}

    # Launch the Gradio interface with the specified parameters
    ui_interface.launch(**launch_params)

# Function to initialize argument parser for command-line arguments
def initialize_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./qwen_image_defaults.toml", help="Path to the toml config file for interface defaults")
    parser.add_argument("--debug", action="store_true", help="Debug on")
    parser.add_argument("--listen", type=str, default="127.0.0.1", help="IP to listen on for connections to Gradio")
    parser.add_argument("--username", type=str, default="", help="Username for authentication")
    parser.add_argument("--password", type=str, default="", help="Password for authentication")
    parser.add_argument("--server_port", type=int, default=0, help="Port to run the server listener on")
    parser.add_argument("--inbrowser", action="store_true", default=True, help="Open in browser")
    parser.add_argument("--share", action="store_true", help="Share the gradio UI")
    parser.add_argument("--headless", action="store_true", help="Is the server headless")
    parser.add_argument("--language", type=str, default=None, help="Set custom language")
    parser.add_argument("--use-ipex", action="store_true", help="Use IPEX environment")
    parser.add_argument("--use-rocm", action="store_true", help="Use ROCm environment")
    parser.add_argument("--do_not_use_shell", action="store_true", help="Enforce not to use shell=True when running external commands")
    parser.add_argument("--do_not_share", action="store_true", help="Do not share the gradio UI")
    parser.add_argument("--requirements", type=str, default=None, help="requirements file to use for validation")
    parser.add_argument("--root_path", type=str, default=None, help="`root_path` for Gradio to enable reverse proxy support. e.g. /kohya_ss")
    parser.add_argument("--noverify", action="store_true", help="Disable requirements verification")
    return parser

if __name__ == "__main__":
    # Initialize argument parser and parse arguments
    parser = initialize_arg_parser()
    args = parser.parse_args()

    # Set up logging based on the debug flag
    log = setup_logging(debug=args.debug)

    # Launch the UI with the provided arguments
    UI(**vars(args))