import gradio as gr
import toml
from .class_gui_config import GUIConfig

class Network:
    def __init__(
        self,
        headless: bool,
        config: GUIConfig,
    ) -> None:
        self.config = config
        self.headless = headless

        # Initialize the UI components
        self.initialize_ui_components()

    def initialize_ui_components(self) -> None:
        with gr.Row():
            self.no_metadata = gr.Checkbox(
                label="Do Not Save Metadata",
                value=self.config.get("no_metadata", False),
                info="Disable saving metadata into the output model file.",
            )

            self.network_weights = gr.Textbox(
                label="Network Weights (LoRA Weights)",
                placeholder="Path to pretrained LoRA weights for network",
                value=self.config.get("network_weights", None),
                info="Optional. Load existing LoRA weights before training.",
            )

            default_module = self.config.get("network_module", "")
            if not default_module:
                default_module = "networks.lora"

            self.network_module = gr.Textbox(
                label="Network Module (LoRA Type)",
                placeholder="LoRA module type to train",
                value=default_module,
                info="LoRA module path used by training (e.g., networks.lora_flux_2 or networks.lora_zimage).",
            )

        with gr.Row():
            self.network_dim = gr.Number(
                label="Network Dimensions (LoRA Rank)",
                info="LoRA rank/dimensions (depends on the module type)",
                value=self.config.get("network_dim", 32),
                step=1,
                interactive=True,
            )

            self.network_alpha = gr.Number(
                label="Network Alpha (LoRA Alpha)",
                info="Alpha value for LoRA weight scaling (default: 1)",
                value=self.config.get("network_alpha", 1),
                step=1,
                interactive=True,
            )

            self.network_dropout = gr.Number(
                label="Network Dropout (LoRA Dropout)",
                info="LoRA dropout rate (0 or None for no dropout, 1 drops all neurons)",
                value=self.config.get("network_dropout", 0),
                step=0.01,
                minimum=0,
                maximum=1,
                interactive=True,
            )

        with gr.Row():
            self.network_args = gr.Textbox(
                label="Network Arguments (LoRA Args)",
                placeholder="Additional LoRA network arguments. Space separated. e.g. \"conv_dim=4 conv_alpha=1 algo=locon\"",
                value=" ".join(self.config.get("network_args", []) or []) if isinstance(self.config.get("network_args", []), list) else self.config.get("network_args", ""),
                interactive=True,
                info="Space-separated key=value args passed to the LoRA module.",
            )

        with gr.Row():
            self.training_comment = gr.Textbox(
                label="Training Comment",
                placeholder="Arbitrary comment string to store in metadata",
                value=self.config.get("training_comment", None),
                info="Optional comment stored in metadata.",
            )

        with gr.Row():
            self.dim_from_weights = gr.Checkbox(
                label="Determine Dimensions from LoRA Weights",
                value=self.config.get("dim_from_weights", False),
                info="Auto-detect LoRA rank/dim from network_weights.",
            )

            self.scale_weight_norms = gr.Number(
                label="Scale Weight Norms",
                info="Scaling factor for weights (1 is a good starting point)",
                value=self.config.get("scale_weight_norms", None),
                step=0.001,
                interactive=True,
                minimum=0,
            )

        with gr.Row():
            self.base_weights = gr.Textbox(
                label="Base Weights (LoRA Base)",
                placeholder="Paths to LoRA weights to merge into the model before training",
                value=self.config.get("base_weights", ""),
                info="Optional. Space-separated paths to LoRA weights to merge before training.",
            )

            self.base_weights_multiplier = gr.Textbox(
                label="Base Weights Multiplier",
                placeholder="Multipliers for network weights to merge into the model before training",
                value=self.config.get("base_weights_multiplier", ""),
                info="Optional. Space-separated multipliers aligned with base_weights order.",
            )
