"""The existing musubi trainer modules, described as backends of the unified UI.

Every trainer module keeps its own command building, validation, runtime-TOML writing and
config loading (``*_gui_actions``). The unified UI calls those functions directly with the
values of its shared fields, in the order recorded in ``legacy_schema.json``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class LegacyBackend:
    name: str  # backend id (also the TabConfigManager tab name)
    display_name: str
    module: str  # module holding the actions / train functions
    actions: str  # "*_gui_actions" dispatcher
    train_function: str  # train_* function inside ``module``
    tab_module: str
    tab_function: str
    spec_key: str | None = None  # first argument of spec-driven dispatchers
    prefix: int = 5  # leading non-setting inputs of the dispatcher (wan has no ask_for_file flag)
    executor_style: str = "global"  # "global": module.executor, "dict": module.executors[spec_key]
    defaults_tab: str | None = None  # TabConfigManager tab name (defaults TOML)

    def build_tab(self):
        return build_legacy_tab(self)


LEGACY_BACKENDS: dict[str, LegacyBackend] = {
    backend.name: backend
    for backend in (
        LegacyBackend(
            "qwen_image",
            "Qwen Image",
            "musubi_tuner_gui.qwen_image_lora_gui",
            "qwen_image_gui_actions",
            "train_qwen_image_model",
            "musubi_tuner_gui.qwen_image_lora_gui",
            "qwen_image_lora_tab",
            defaults_tab="qwen_image",
        ),
        LegacyBackend(
            "qwen_image_21",
            "Qwen-Image 2.1",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "qwen_image_21_tab",
            spec_key="qwen_image_21",
            executor_style="dict",
            defaults_tab="qwen_image_21",
        ),
        LegacyBackend(
            "flux",
            "FLUX.2 / Klein",
            "musubi_tuner_gui.flux_lora_gui",
            "flux_gui_actions",
            "train_flux_model",
            "musubi_tuner_gui.flux_lora_gui",
            "flux_lora_tab",
            defaults_tab="flux",
        ),
        LegacyBackend(
            "zimage",
            "Z-Image",
            "musubi_tuner_gui.zimage_lora_gui",
            "zimage_gui_actions",
            "train_zimage_model",
            "musubi_tuner_gui.zimage_lora_gui",
            "zimage_lora_tab",
            defaults_tab="zimage",
        ),
        LegacyBackend(
            "krea2",
            "Krea 2",
            "musubi_tuner_gui.modern_image_lora_gui",
            "modern_image_gui_actions",
            "train_modern_image_model",
            "musubi_tuner_gui.krea2_lora_gui",
            "krea2_lora_tab",
            spec_key="krea2",
            executor_style="dict",
            defaults_tab="krea2",
        ),
        LegacyBackend(
            "ideogram4",
            "Ideogram 4",
            "musubi_tuner_gui.modern_image_lora_gui",
            "modern_image_gui_actions",
            "train_modern_image_model",
            "musubi_tuner_gui.ideogram4_lora_gui",
            "ideogram4_lora_tab",
            spec_key="ideogram4",
            executor_style="dict",
            defaults_tab="ideogram4",
        ),
        LegacyBackend(
            "ernie_image",
            "ERNIE-Image",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "ernie_image_tab",
            spec_key="ernie_image",
            executor_style="dict",
            defaults_tab="ernie_image",
        ),
        LegacyBackend(
            "mage_flow",
            "Mage-Flow",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "mage_flow_tab",
            spec_key="mage_flow",
            executor_style="dict",
            defaults_tab="mage_flow",
        ),
        LegacyBackend(
            "flux3_klein",
            "FLUX.3 Klein",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "flux3_klein_tab",
            spec_key="flux3_klein",
            executor_style="dict",
            defaults_tab="flux3_klein",
        ),
        LegacyBackend(
            "moss_image",
            "MOSS-Image",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "moss_image_tab",
            spec_key="moss_image",
            executor_style="dict",
            defaults_tab="moss_image",
        ),
        LegacyBackend(
            "fibo2",
            "FIBO 2.0",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "fibo2_tab",
            spec_key="fibo2",
            executor_style="dict",
            defaults_tab="fibo2",
        ),
        LegacyBackend(
            "anima",
            "Anima",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "anima_tab",
            spec_key="anima",
            executor_style="dict",
            defaults_tab="anima",
        ),
        LegacyBackend(
            "hidream_o1",
            "HiDream-O1",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "hidream_o1_tab",
            spec_key="hidream_o1",
            executor_style="dict",
            defaults_tab="hidream_o1",
        ),
        LegacyBackend(
            "wan",
            "Wan 2.1 / 2.2",
            "musubi_tuner_gui.wan_lora_gui",
            "wan_gui_actions",
            "train_wan_model",
            "musubi_tuner_gui.wan_lora_gui",
            "wan_lora_tab",
            prefix=4,
            defaults_tab="wan",
        ),
        LegacyBackend(
            "ltx2",
            "LTX 2.3",
            "musubi_tuner_gui.ltx2_lora_gui",
            "ltx2_gui_actions",
            "train_ltx2_model",
            "musubi_tuner_gui.ltx2_lora_gui",
            "ltx2_lora_tab",
            defaults_tab="ltx2",
        ),
        LegacyBackend(
            "minimax_h3",
            "MiniMax H3",
            "musubi_tuner_gui.minimax_h3_lora_gui",
            "minimax_h3_gui_actions",
            "train_minimax_h3_model",
            "musubi_tuner_gui.minimax_h3_lora_gui",
            "minimax_h3_lora_tab",
            defaults_tab="minimax_h3",
        ),
        LegacyBackend(
            "kandinsky6",
            "Kandinsky 6",
            "musubi_tuner_gui.image_models_gui",
            "gui_actions",
            "train_model",
            "musubi_tuner_gui.image_models_gui",
            "kandinsky6_tab",
            spec_key="kandinsky6",
            executor_style="dict",
            defaults_tab="kandinsky6",
        ),
    )
}


@lru_cache(maxsize=None)
def legacy_module(backend: LegacyBackend):
    return importlib.import_module(backend.module)


def actions_callable(backend: LegacyBackend):
    """``fn(action, ask_for_file, config_file_name, headless, print_only, *settings)`` for every backend."""
    module = legacy_module(backend)

    def call(action, ask_for_file, config_file_name, headless, print_only, *settings):
        dispatcher = getattr(module, backend.actions)
        if backend.prefix == 4:
            return dispatcher(action, config_file_name, headless, print_only, *settings)
        if backend.spec_key is not None:
            return dispatcher(backend.spec_key, action, ask_for_file, config_file_name, headless, print_only, *settings)
        return dispatcher(action, ask_for_file, config_file_name, headless, print_only, *settings)

    return call


def build_legacy_tab(backend: LegacyBackend):
    """Build the original tab (used only by the schema generator and the classic UI)."""
    from musubi_tuner_gui.class_tab_config_manager import TabConfigManager

    manager = TabConfigManager(config_file_path="./qwen_image_defaults.toml")
    tab_module = importlib.import_module(backend.tab_module)
    builder = getattr(tab_module, backend.tab_function)
    return builder(headless=True, config=manager.get_config_for_tab(backend.defaults_tab or backend.name))



def install_executor(runner) -> None:
    """Point every trainer module's executor at the unified runner (no Gradio components needed)."""
    for backend in LEGACY_BACKENDS.values():
        module = legacy_module(backend)
        if backend.executor_style == "dict":
            module.executors[backend.spec_key] = runner
        else:
            module.executor = runner
