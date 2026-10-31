"""The unified SECourses Musubi Trainer UI: Image / Video / Audio Models, Tools, Version History."""

from __future__ import annotations

import html
import time

import gradio as gr

from ..custom_logging import setup_logging
from . import presets as P
from .backends import install_quiet_toasts
from .registry import AUDIO, CATEGORY_TITLES, IMAGE, VIDEO
from .runner import install_runner
from .workspace import Workspace

log = setup_logging()

APP_VERSION = "V39 · FIBO 2.0 & Anima"


def _hero() -> str:
    gpu = P.detected_gpu()
    gpu_chip = f"🎮 {html.escape(gpu[0])} · {gpu[1]:g} GB" if gpu else "🎮 No NVIDIA GPU detected"
    return f"""
    <div class="app-hero ud-hero">
        <div class="app-hero-left">
            <div class="app-hero-badge">{APP_VERSION}</div>
            <div>
                <h1>SECourses <span class="grad">Musubi Trainer</span></h1>
                <p>One trainer for image, video and audio models — pick a base model, choose LoRA or Fine-Tuning,
                and start from a GPU preset measured for your card.</p>
            </div>
        </div>
        <div class="ud-hero-right">
            <span class="ud-chip">{gpu_chip}</span>
            <a class="app-hero-link" href="https://www.patreon.com/posts/137551634" target="_blank" rel="noopener">⭐ Patreon Guide &amp; Support</a>
        </div>
    </div>
    """


def _tools_tab(headless: bool, config_manager) -> None:
    from ..image_captioning_gui import image_captioning_tab
    from ..image_preprocessing_gui import image_preprocessing_tab
    from ..lora_convert_gui import lora_convert_tab
    from ..lora_extractor_gui import lora_extractor_tab
    from ..lora_merge_gui import lora_merge_tab
    from ..model_quantizer_gui import model_quantizer_tab

    with gr.Tabs(elem_classes=["model-tabs", "tools-tabs"]):
        with gr.Tab("🏷️ Image Captioning"):
            image_captioning_tab(headless=headless, config=config_manager.get_config_for_tab("image_captioning"))
        with gr.Tab("🖼️ Image Preprocessing"):
            image_preprocessing_tab(headless=headless, config=config_manager.get_config_for_tab("image_preprocessing"))
        with gr.Tab("🧮 Model Quantizer") as quantizer_tab:
            quantizer_timer = model_quantizer_tab(headless=headless, config=config_manager.get_config_for_tab("model_quantizer"))
        # the queue timer only runs during a conversion; one tick on opening shows a conversion still running after a
        # page reload (idle ticks stop it again)
        quantizer_tab.select(lambda: gr.Timer(active=True), outputs=[quantizer_timer], queue=False, show_progress="hidden")
        with gr.Tab("✂️ LoRA Extractor"):
            lora_extractor_tab(headless=headless, config=None)
        with gr.Tab("🧬 LoRA Merger"):
            lora_merge_tab(headless=headless, config=None)
        with gr.Tab("🔁 LoRA Converter"):
            lora_convert_tab(headless=headless, config=None)


def build_app(headless: bool = False, config_manager=None) -> gr.Blocks:
    install_quiet_toasts()
    install_runner()
    if config_manager is None:
        from ..class_tab_config_manager import TabConfigManager

        config_manager = TabConfigManager(config_file_path="./qwen_image_defaults.toml")

    started = time.time()
    demo = gr.Blocks(title="SECourses Musubi Trainer")
    workspaces: dict[str, Workspace] = {}
    with demo:
        gr.HTML(_hero())
        with gr.Tabs(elem_id="main_tabs", elem_classes=["main-tabs"], selected=P.load_ui_state().get("last_category") or IMAGE) as main_tabs:
            tabs = {}
            for category in (IMAGE, VIDEO, AUDIO):
                with gr.Tab(CATEGORY_TITLES[category], id=category, elem_classes=["group-tab"]) as tab:
                    workspace = Workspace(category, headless=headless)
                    if workspace.entries:
                        workspace.build()
                        workspaces[category] = workspace
                    else:  # every trainer of this category failed to load (a broken install): the other tabs still work
                        gr.HTML(_unavailable_card(category))
                tabs[category] = tab
            with gr.Tab("🧰 Tools", id="tools", elem_classes=["group-tab"]) as tools_tab:
                _tools_tab(headless, config_manager)
            with gr.Tab("📜 Version History", id="version_history", elem_classes=["group-tab"]) as history_tab:
                from ..changelog_gui import version_history_tab

                gr.HTML(WHATS_NEW_HTML)
                version_history_tab(headless=headless, config=None)

        # ------------------------------------------------------------ remember the open tab
        progress_sink = gr.HTML("", elem_classes=["ws-sink"])
        all_tabs = [*tabs.items(), ("tools", tools_tab), ("version_history", history_tab)]

        def remember_tab(tab_id: str):
            def remember(trigger):
                if trigger != "load":  # Gradio itself selects the first tab when a page opens
                    P.update_ui_state(last_category=tab_id)

            return remember

        for tab_id, tab in all_tabs:
            tab.select(
                remember_tab(tab_id),
                inputs=[progress_sink],
                js=USER_ACTION_JS,
                queue=False,
                show_progress="hidden",
                show_progress_on=[progress_sink],
            )

        def restore_tab():
            """A reload opens the tab used last (``selected`` above is fixed when the app starts)."""
            remembered = P.load_ui_state().get("last_category")
            return gr.Tabs(selected=remembered if remembered in dict(all_tabs) else IMAGE)

        demo.load(restore_tab, outputs=[main_tabs], queue=False, show_progress="hidden", show_progress_on=[progress_sink])

        # ------------------------------------------------------------ page load: re-apply the remembered selection
        def restorer(workspace: Workspace):
            def restore(request: gr.Request = None):
                return workspace.restore(request)

            return restore

        for workspace in workspaces.values():
            demo.load(
                restorer(workspace),
                outputs=workspace.apply_outputs,
                show_progress="hidden",
                show_progress_on=[workspace.progress_sink],
            )
        # No unload hook: Gradio can report an unload for a page that is still open (a reconnect after
        # sleep), and a forgotten session mirror would then skip updates that browser still needs. Each
        # workspace keeps at most MAX_SESSIONS mirrors and drops the least recently used.
    log.info(f"Unified UI built in {time.time() - started:.1f} s ({sum(len(ws.fields) for ws in workspaces.values())} shared settings)")
    return demo


# Selecting a tab with a click or a key press carries the browser's user activation; the select Gradio
# fires on its own when a page opens does not (it must not overwrite the remembered tab).
USER_ACTION_JS = "(v) => (navigator.userActivation && !navigator.userActivation.isActive) ? 'load' : 'user'"


def _unavailable_card(category: str) -> str:
    title = html.escape(CATEGORY_TITLES[category])
    return (
        f'<div class="ud-banner ud-banner-warn">⚠️ The trainers of {title} could not be loaded — the console log shows why. '
        "Running the installer / update again usually repairs a broken install. The other tabs work normally.</div>"
    )


WHATS_NEW_HTML = """
<div class="ud-whatsnew">
  <h3>✨ V39 — FIBO 2.0, Anima, three edit cards, Turbo LoRAs and faster INT8</h3>
  <ul>
    <li><b>🌸 FIBO 2.0 and 🎎 Anima</b> — DoRA / LoRA and full fine-tuning on Bria's FIBO 2.0 Dev (8.1B) and CircleStone Labs' Anima-Base (2B, anime and illustration), with measured presets from 6 to 32 GB.</li>
    <li><b>✏️ Edit cards</b> — FIBO 2.0 Edit, Krea 2 Edit and MOSS-Image Edit train image-editing LoRAs from target images and their references in <code>control_images</code>, fed the way ComfyUI's edit workflows feed them.</li>
    <li><b>⚡ Turbo LoRAs</b> — Krea 2 Turbo and Qwen-Image 2.1 Turbo train through ostris' training adapters, so the samples and the LoRA show the 8-step Turbo models.</li>
    <li><b>🎬 LTX 2.5 Pre-Trained</b> — a new Transformer Variant on the LTX 2.5 cards for specialized fine-tunes, with the same presets (9e-5).</li>
    <li><b>⚡ ConvRot INT8 compiles</b> (1.3-1.5x faster with Compile), the <b>Automagic4</b> optimizer and the <b>latest libraries</b> (transformers 5.21, diffusers 0.42.1, Gradio 6.31). The full version history is below.</li>
  </ul>
</div>
"""
