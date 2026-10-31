import os
import sys
import argparse
import subprocess
import contextlib

# Apply PyTorch compatibility before Transformers imports TorchAO.
import musubi_tuner_gui.torch_compat  # noqa: F401
import gradio as gr

from musubi_tuner_gui.qwen_image_lora_gui import qwen_image_lora_tab
from musubi_tuner_gui.wan_lora_gui import wan_lora_tab
from musubi_tuner_gui.flux_lora_gui import flux_lora_tab
from musubi_tuner_gui.zimage_lora_gui import zimage_lora_tab
from musubi_tuner_gui.ideogram4_lora_gui import ideogram4_lora_tab
from musubi_tuner_gui.krea2_lora_gui import krea2_lora_tab
from musubi_tuner_gui.ltx2_lora_gui import ltx2_lora_tab
from musubi_tuner_gui.minimax_h3_lora_gui import minimax_h3_lora_tab
from musubi_tuner_gui.image_models_gui import (
    ernie_image_tab,
    hidream_o1_tab,
    flux3_klein_tab,
    kandinsky6_tab,
    mage_flow_tab,
    moss_image_tab,
    fibo2_tab,
    qwen_image_21_tab,
)
from musubi_tuner_gui.image_captioning_gui import image_captioning_tab
from musubi_tuner_gui.model_quantizer_gui import model_quantizer_tab
from musubi_tuner_gui.image_preprocessing_gui import image_preprocessing_tab
from musubi_tuner_gui.changelog_gui import version_history_tab
from musubi_tuner_gui.lora_extractor_gui import lora_extractor_tab
from musubi_tuner_gui.lora_merge_gui import lora_merge_tab
from musubi_tuner_gui.lora_convert_gui import lora_convert_tab
from musubi_tuner_gui.custom_logging import setup_logging
from musubi_tuner_gui.class_gui_config import GUIConfig
from musubi_tuner_gui.class_tab_config_manager import TabConfigManager
import toml

PYTHON = sys.executable
project_dir = os.path.dirname(os.path.abspath(__file__))

def use_patient_localhost_check(total_seconds: float = 90.0):
    """Gradio's launch() aborts with "When localhost is not accessible..." if the first page request takes
    over 3 s. Building this app's page takes longer on a busy machine (e.g. while a training run uses
    every CPU core), so keep retrying with a longer timeout before giving up."""
    import time

    import httpx
    from gradio import networking

    def url_ok(url: str) -> bool:
        deadline = time.monotonic() + total_seconds
        while True:
            try:
                response = httpx.head(url, timeout=20, verify=False)
                if response.status_code in (200, 401, 301, 302, 303, 307, 308):
                    return True
            except (ConnectionError, httpx.ConnectError, httpx.TimeoutException):
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)

    networking.url_ok = url_ok


# Function to read file content, suppressing any FileNotFoundError
def read_file_content(file_path):
    with contextlib.suppress(FileNotFoundError):
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    return ""

# Function to initialize the Gradio UI interface
def initialize_ui_interface(config_manager, headless, release_info, readme_content, classic=False):
    if not classic:
        # Unified UI: Image / Video / Audio workspaces built on the trainer modules below.
        from musubi_tuner_gui.unified.app import build_app

        return build_app(headless=headless, config_manager=config_manager)

    # Classic UI (python gui.py --classic): one tab per trainer.
    # Create the main Gradio Blocks interface
    # NOTE: In Gradio 6, `css` and `theme` are passed to launch() instead of Blocks().
    ui_interface = gr.Blocks(title="SECourses Musubi Trainer Premium")
    with ui_interface:
        # Modern hero header with Patreon link
        gr.HTML(
            """
            <div class="app-hero">
                <div class="app-hero-left">
                    <div class="app-hero-badge">V35.0</div>
                    <div>
                        <h1>SECourses <span class="grad">Musubi Trainer</span></h1>
                        <p>Qwen Image &middot; Qwen Image 2.1 &middot; FLUX.2 &middot; FLUX.3 Klein &middot; MOSS-Image &middot; Z-Image &middot; Krea 2 &middot; Ideogram 4 &middot; ERNIE-Image &middot; Mage-Flow &middot; HiDream-O1 &middot; Wan &middot; LTX 2.3 &middot; MiniMax H3 &middot; Kandinsky 6 &mdash; LoRA &amp; fine-tune training with ready-made GPU presets</p>
                    </div>
                </div>
                <a class="app-hero-link" href="https://www.patreon.com/posts/137551634" target="_blank" rel="noopener">
                    &#11088; Patreon Guide &amp; Support
                </a>
            </div>
            """
        )

        with gr.Tabs(elem_id="main_tabs", elem_classes=["main-tabs"]):
            with gr.Tab("🖼️ Image Models", id="image_models", elem_classes=["group-tab"]):
                with gr.Tabs(elem_classes=["model-tabs"]):
                    with gr.Tab("Qwen Image"):
                        qwen_image_lora_tab(headless=headless, config=config_manager.get_config_for_tab("qwen_image"))
                    with gr.Tab("Qwen Image 2.1 ✨"):
                        qwen_image_21_tab(headless=headless, config=config_manager.get_config_for_tab("qwen_image_21"))
                    with gr.Tab("FLUX · FLUX.2 · Klein"):
                        flux_lora_tab(headless=headless, config=config_manager.get_config_for_tab("flux"))
                    with gr.Tab("Z-Image"):
                        zimage_lora_tab(headless=headless, config=config_manager.get_config_for_tab("zimage"))
                    with gr.Tab("Krea 2"):
                        krea2_lora_tab(headless=headless, config=config_manager.get_config_for_tab("krea2"))
                    with gr.Tab("Ideogram 4"):
                        ideogram4_lora_tab(headless=headless, config=config_manager.get_config_for_tab("ideogram4"))
                    with gr.Tab("ERNIE-Image ✨"):
                        ernie_image_tab(headless=headless, config=config_manager.get_config_for_tab("ernie_image"))
                    with gr.Tab("Mage-Flow ✨"):
                        mage_flow_tab(headless=headless, config=config_manager.get_config_for_tab("mage_flow"))
                    with gr.Tab("FLUX.3 Klein ✨"):
                        flux3_klein_tab(headless=headless, config=config_manager.get_config_for_tab("flux3_klein"))
                    with gr.Tab("MOSS-Image ✨"):
                        moss_image_tab(headless=headless, config=config_manager.get_config_for_tab("moss_image"))
                    with gr.Tab("FIBO 2.0 ✨"):
                        fibo2_tab(headless=headless, config=config_manager.get_config_for_tab("fibo2"))
                    with gr.Tab("HiDream-O1 ✨"):
                        hidream_o1_tab(headless=headless, config=config_manager.get_config_for_tab("hidream_o1"))

            with gr.Tab("🎬 Video & Audio Models", id="video_models", elem_classes=["group-tab"]):
                with gr.Tabs(elem_classes=["model-tabs"]):
                    with gr.Tab("Wan 2.1 · 2.2"):
                        wan_lora_tab(headless=headless, config=config_manager.get_config_for_tab("wan"))
                    with gr.Tab("LTX 2.3"):
                        ltx2_lora_tab(headless=headless, config=config_manager.get_config_for_tab("ltx2"))
                    with gr.Tab("MiniMax H3"):
                        minimax_h3_lora_tab(headless=headless, config=config_manager.get_config_for_tab("minimax_h3"))
                    with gr.Tab("Kandinsky 6 ✨"):
                        kandinsky6_tab(headless=headless, config=config_manager.get_config_for_tab("kandinsky6"))

            with gr.Tab("🧰 Tools", id="tools", elem_classes=["group-tab"]):
                with gr.Tabs(elem_classes=["model-tabs"]):
                    with gr.Tab("Image Captioning"):
                        image_captioning_tab(headless=headless, config=config_manager.get_config_for_tab("image_captioning"))
                    with gr.Tab("Image Preprocessing"):
                        image_preprocessing_tab(headless=headless, config=config_manager.get_config_for_tab("image_preprocessing"))
                    with gr.Tab("Model Quantizer") as model_quantizer_tab_item:
                        quantizer_queue_timer = model_quantizer_tab(headless=headless, config=config_manager.get_config_for_tab("model_quantizer"))
                    # one tick shows a conversion still running after a page reload; idle ticks stop the timer again
                    model_quantizer_tab_item.select(lambda: gr.Timer(active=True), outputs=[quantizer_queue_timer], queue=False, show_progress="hidden")
                    with gr.Tab("LoRA Extractor"):
                        lora_extractor_tab(headless=headless, config=None)
                    with gr.Tab("LoRA Merger"):
                        lora_merge_tab(headless=headless, config=None)
                    with gr.Tab("LoRA Converter"):
                        lora_convert_tab(headless=headless, config=None)

            with gr.Tab("📜 Version History", id="version_history", elem_classes=["group-tab"]):
                version_history_tab(headless=headless, config=None)

    return ui_interface

# Function to configure and launch the UI
def UI(**kwargs):
    # Add custom JavaScript if specified
    log.info(f"headless: {kwargs.get('headless', False)}")

    # Load release and README information
    release_info = "v18.0"  # Hardcoded version since pyproject.toml is not needed
    
    readme_content = read_file_content(os.path.join(project_dir, "README.md"))
    
    # Initialize tab-aware configuration manager - default to qwen_image_defaults.toml
    config_manager = TabConfigManager(config_file_path=kwargs.get("config", "./qwen_image_defaults.toml"))
    if config_manager.user_loaded_config:
        log.info(f"Loaded user configuration from '{kwargs.get('config', './qwen_image_defaults.toml')}'...")
    else:
        log.info("No user config loaded - will use tab-specific defaults")

    # Initialize the Gradio UI interface
    ui_interface = initialize_ui_interface(
        config_manager, kwargs.get("headless", False), release_info, readme_content, classic=kwargs.get("classic", False)
    )

    # Construct launch parameters using dictionary comprehension
    # Gradio 6: theme and css belong to launch(), not the Blocks constructor.
    launch_params = {
        "server_name": kwargs.get("listen"),
        "auth": (kwargs["username"], kwargs["password"]) if kwargs.get("username") and kwargs.get("password") else None,
        "server_port": kwargs.get("server_port", 0) if kwargs.get("server_port", 0) > 0 else None,
        "inbrowser": kwargs.get("inbrowser", True),
        "share": kwargs.get("share", False),
        "root_path": kwargs.get("root_path", None),
        "debug": kwargs.get("debug", False),
        "theme": gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            radius_size=gr.themes.sizes.radius_lg,
            # every theme text size 2px above Gradio's text_md scale (12px labels / info were too small to read)
            text_size=gr.themes.Size(name="text_md_plus_2", xxs="11px", xs="12px", sm="14px", md="16px", lg="18px", xl="24px", xxl="28px"),
        ),
        # next to this file, wherever the app is started from (a missing unified.css breaks the whole layout)
        "css": read_file_content(os.path.join(project_dir, "assets", "style.css")) + "\n" + read_file_content(os.path.join(project_dir, "assets", "unified.css")),
        # Gradio's run history serializes every event's inputs/outputs (200+ settings per trainer tab) into
        # localStorage on each click; it filled the 5 MB quota and slowed the whole UI. False also clears it.
        "run_history": False,
        # run_history=False only clears the current app id's key; drop the ones older launches left behind.
        "head": (
            "<script>try{Object.keys(localStorage).filter(function(k){return k.indexOf('gradio:run-history:')===0;})"
            ".forEach(function(k){localStorage.removeItem(k);});}catch(e){}</script>"
        ),
    }
  
    # This line filters out any key-value pairs from `launch_params` where the value is `None`, ensuring only valid parameters are passed to the `launch` function.
    launch_params = {k: v for k, v in launch_params.items() if v is not None}

    # Launch the Gradio interface with the specified parameters
    use_patient_localhost_check()
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
    parser.add_argument("--inbrowser", action=argparse.BooleanOptionalAction, default=True, help="Open in browser (--no-inbrowser: don't)")
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
    parser.add_argument("--classic", action="store_true", help="Use the classic one-tab-per-trainer interface")
    return parser

if __name__ == "__main__":
    # Initialize argument parser and parse arguments
    parser = initialize_arg_parser()
    args = parser.parse_args()

    # Set up logging based on the debug flag
    log = setup_logging(debug=args.debug)

    # Launch the UI with the provided arguments
    UI(**vars(args))
