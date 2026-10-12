"""Declarative parameter description shared by every trainer backend of the unified UI.

A backend (one of the existing ``*_lora_gui`` trainers, or the kohya sd-scripts backend)
describes each of its settings with a :class:`ParamSpec`. The unified workspace builds one
Gradio component per key and re-labels / shows / hides it when the selected base model changes,
so two backends that use the same key share the same on-screen field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# Widget kinds understood by the workspace builder.
KINDS = ("text", "textarea", "number", "slider", "checkbox", "dropdown", "radio", "file", "folder")

# Training types. Every ParamSpec lists the types it is shown for.
LORA = "lora"
FULL = "full"
ALL_MODES = frozenset({LORA, FULL})

# Section ids, top to bottom in the order the workspace renders them (most important first).
SECTIONS: tuple[tuple[str, str, str], ...] = (
    # id, title, one-line description shown under the title
    ("dataset", "📁 Dataset", "Training images or videos, resolution, captions and batching"),
    ("output", "💾 Output & Saving", "Where the trained model goes and how often checkpoints are saved"),
    ("training", "🎓 Training Length & Learning Rate", "Epochs, steps, learning rate, optimizer and scheduler"),
    ("model", "🧠 Model Files & Precision", "Base model, VAE and text encoder files and their precision"),
    ("network", "🎯 LoRA Network", "Rank, alpha, dropout and LoRA-specific options"),
    ("finetune", "🧬 Fine-Tuning", "Full fine-tuning (DreamBooth) specific options"),
    ("memory", "⚡ Memory & Speed", "Block swap, FP8, gradient checkpointing, attention and torch.compile"),
    ("samples", "🖼️ Sample Generation", "Preview images or videos rendered during training"),
    ("bcf", "🏆 Best Checkpoint Finder", "Picks the best checkpoint from samples of your test prompts when training ends, so you do not have to compare grids by eye"),
    ("caching", "🧊 Caching", "Latent and text-encoder caching before training starts"),
    ("advanced", "🔬 Advanced Training", "Timestep sampling, flow shift, noise and loss options"),
    ("model_specific", "🧩 Model-Specific Options", "Settings that only this base model has"),
    ("hardware", "🖥️ Hardware & Launch", "GPU selection, multi-GPU, accelerate and data-loader workers"),
    ("logging", "📊 Logging & Debug", "TensorBoard / WandB logging, debug modes and extra arguments"),
    ("metadata", "🏷️ Metadata & Hugging Face", "Model metadata and Hugging Face upload"),
)
SECTION_IDS = tuple(section[0] for section in SECTIONS)
# Sections opened by default when a model is selected.
OPEN_SECTIONS = frozenset({"dataset", "output", "training"})


@dataclass(frozen=True)
class ParamSpec:
    key: str
    kind: str
    label: str
    default: Any = None
    info: str = ""
    choices: tuple | None = None  # values, or (label, value) pairs
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    section: str = "model_specific"
    priority: int = 500  # lower renders first inside its section
    modes: frozenset = ALL_MODES
    visible: bool = True  # False: carried and saved but never shown (internal / legacy field)
    advanced: bool = False  # shown only when "Show advanced settings" is on
    allow_custom_value: bool = False
    multiselect: bool = False
    lines: int = 1
    placeholder: str = ""
    file_types: tuple = ()  # for kind == "file": e.g. (".safetensors",)
    extras: dict = field(default_factory=dict, compare=False, hash=False)

    def with_(self, **changes) -> "ParamSpec":
        return replace(self, **changes)

    @property
    def is_path(self) -> bool:
        return self.kind in ("file", "folder")

    def choice_values(self) -> list:
        values = []
        for choice in self.choices or ():
            if isinstance(choice, (list, tuple)) and len(choice) == 2:
                values.append(choice[1])
            else:
                values.append(choice)
        return values


def spec_index(specs: list[ParamSpec] | tuple[ParamSpec, ...]) -> dict[str, ParamSpec]:
    return {spec.key: spec for spec in specs}
