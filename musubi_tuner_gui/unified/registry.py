"""Base models offered by the unified UI, grouped by category (Image / Video / Audio).

A ``ModelEntry`` is what the user picks as "base model". Several entries can share one trainer
backend: FLUX.2 Dev / Klein 9B / Klein 4B all use the FLUX trainer and differ only in the values
the entry fixes (``fixed``) and the preset folders it lists. ``locked_keys`` are settings the entry
fixes and therefore hides (the model card already made that choice).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .param_spec import FULL, LORA

IMAGE = "image"
VIDEO = "video"
AUDIO = "audio"
CATEGORIES = (IMAGE, VIDEO, AUDIO)
CATEGORY_TITLES = {IMAGE: "🖼️ Image Models", VIDEO: "🎬 Video Models", AUDIO: "🔊 Audio Models"}

MODE_LABELS = {LORA: "🎯 LoRA", FULL: "🧬 Fine-Tuning"}
MODE_DESCRIPTIONS = {
    LORA: "Small adapter file, fast to train, low VRAM. Best for characters, styles and products.",
    FULL: "Trains every weight of the base model (DreamBooth). Highest quality, needs more VRAM and disk.",
}


@dataclass(frozen=True)
class ModelEntry:
    key: str
    category: str
    name: str
    icon: str
    tagline: str
    backend: str  # legacy backend name, or "kohya:<model>"
    modes: tuple = (LORA, FULL)
    preset_folders: tuple = ()
    preset_subfolders: tuple = ()  # only list presets inside these sub-folders of the preset folders
    preset_preference: tuple = ()  # path fragments the GPU auto-pick prefers, in order
    fixed: dict = field(default_factory=dict)  # values the entry forces
    locked_keys: frozenset = frozenset()  # fixed keys hidden from the form
    choice_overrides: dict = field(default_factory=dict)  # key -> allowed choices for this entry
    default_overrides: dict = field(default_factory=dict)  # entry defaults on top of the backend's
    badges: tuple = ()
    model_folder: str = ""  # Training_Models_* folder under the install root (download target)
    download_bundles: tuple = ()  # (label, Download_Train_Models.py --model key) pairs

    @property
    def is_kohya(self) -> bool:
        return self.backend.startswith("kohya:")

    @property
    def kohya_model(self) -> str:
        return self.backend.split(":", 1)[1] if self.is_kohya else ""

    @property
    def card_label(self) -> str:
        return f"{self.icon} {self.name}"


ENTRIES: tuple[ModelEntry, ...] = (
    # ------------------------------------------------------------------ image models
    ModelEntry(
        "krea2",
        IMAGE,
        "Krea 2",
        "🎨",
        "Photoreal and artistic text-to-image. Optional Turbo DiT for fast preview samples.",
        "krea2",
        preset_folders=("Krea_2_Training_Configs",),
        locked_keys=frozenset({"dit_dtype"}),  # LoRA always trains a bfloat16 DiT; fine-tuning derives it from Full BF16
        badges=("Recommended",),
        model_folder="Training_Models_Krea_2",
        download_bundles=(("Krea 2 models", "krea2"),),
    ),
    ModelEntry(
        "flux2_dev",
        IMAGE,
        "FLUX.2 Dev",
        "🌀",
        "Black Forest Labs FLUX.2 dev with the Mistral 3 text encoder. Top prompt adherence.",
        "flux",
        preset_folders=("FLUX_2_Training_Configs",),
        # FP8 for the text encoder is Klein-only: the classic tab forces it off for FLUX.2 dev
        fixed={"model_family": "FLUX.2", "model_version": "dev", "fp8_text_encoder": False, "caching_teo_fp8_text_encoder": False},
        locked_keys=frozenset({"model_family", "model_version", "fp8_text_encoder", "caching_teo_fp8_text_encoder"}),
        default_overrides={"sample_steps": 50, "sample_guidance_scale": 4.0},
        model_folder="Training_Models_FLUX_2_and_Klein",
        download_bundles=(("FLUX.2 Dev models", "flux2_dev"),),
    ),
    ModelEntry(
        "flux2_klein_9b",
        IMAGE,
        "FLUX.2 Klein 9B",
        "⚡",
        "Compact FLUX.2 Klein 9B (Qwen3 text encoder). Base or distilled 4-step variant.",
        "flux",
        preset_folders=("FLUX_Klein_9B_Training_Configs",),
        fixed={"model_family": "FLUX Klein"},
        locked_keys=frozenset({"model_family"}),
        choice_overrides={"model_version": (("klein-base-9b", "klein-base-9b"), ("klein-9b (distilled)", "klein-9b"))},
        default_overrides={"model_version": "klein-base-9b", "sample_steps": 50, "sample_guidance_scale": 4.0},
        model_folder="Training_Models_FLUX_2_and_Klein",
        download_bundles=(("FLUX.2 Klein 9B models", "flux_klein_9b"),),
    ),
    ModelEntry(
        "flux2_klein_4b",
        IMAGE,
        "FLUX.2 Klein 4B",
        "⚡",
        "Smallest FLUX.2 Klein (4B). Fast training on 8-12 GB GPUs.",
        "flux",
        preset_folders=("FLUX_Klein_4B_Training_Configs",),
        fixed={"model_family": "FLUX Klein"},
        locked_keys=frozenset({"model_family"}),
        choice_overrides={"model_version": (("klein-base-4b", "klein-base-4b"), ("klein-4b (distilled)", "klein-4b"))},
        default_overrides={"model_version": "klein-base-4b", "sample_steps": 50, "sample_guidance_scale": 4.0},
        model_folder="Training_Models_FLUX_2_and_Klein",
        download_bundles=(("FLUX.2 Klein 4B models", "flux_klein_4b"),),
    ),
    ModelEntry(
        "qwen_image",
        IMAGE,
        "Qwen Image",
        "🐉",
        "Qwen-Image (incl. 2512) and Qwen-Image-Edit 2509 / 2511 with control images.",
        "qwen_image",
        preset_folders=("Qwen_Training_Configs",),
        model_folder="Training_Models_Qwen",
        download_bundles=(("Qwen-Image", "qwen_image"), ("Qwen-Image 2512", "qwen_image_2512"), ("Qwen-Image-Edit Plus 2509", "qwen_image_edit_plus"), ("Qwen-Image-Edit 2511", "qwen_image_edit_2511")),
    ),
    ModelEntry(
        "qwen_image_21",
        IMAGE,
        "Qwen-Image 2.1",
        "✨",
        "Qwen-Image 2.1 with the Qwen3-VL 8B text encoder and RGBA VAE; editing via reference images.",
        "qwen_image_21",
        preset_folders=("Qwen_Image_2.1_Training_Configs",),
        badges=("New",),
        model_folder="Training_Models_Qwen_Image_2_1",
        download_bundles=(("Qwen-Image 2.1 models", "qwen_image_21"),),
    ),
    ModelEntry(
        "zimage_base",
        IMAGE,
        "Z-Image Base",
        "🧿",
        "Z-Image Base: quality-first 6B model, best for LoRAs that also work on Turbo.",
        "zimage",
        preset_folders=("Z_Image_Base_Training_Configs",),
        fixed={"zimage_variant": "base"},
        locked_keys=frozenset({"zimage_variant"}),
        default_overrides={"sample_cfg_scale": 4.0, "sample_steps": 25},
        model_folder="Training_Models_Z_Image",
        download_bundles=(("Z-Image Base models", "z_image_base"),),
    ),
    ModelEntry(
        "zimage_turbo",
        IMAGE,
        "Z-Image Turbo",
        "🚀",
        "Z-Image Turbo (distilled, 8-step) with the training adapter.",
        "zimage",
        preset_folders=("Z_Image_Turbo_Training_Configs",),
        fixed={"zimage_variant": "turbo"},
        locked_keys=frozenset({"zimage_variant"}),
        default_overrides={"sample_cfg_scale": 1.0, "sample_steps": 20},
        model_folder="Training_Models_Z_Image",
        download_bundles=(("Z-Image Turbo models", "z_image_turbo"),),
    ),
    ModelEntry(
        "ideogram4",
        IMAGE,
        "Ideogram 4",
        "🔤",
        "Ideogram 4 for typography, posters and design work.",
        "ideogram4",
        preset_folders=("Ideogram_4_Training_Configs",),
        model_folder="Training_Models_Ideogram_4",
        download_bundles=(("Ideogram 4 models", "ideogram4"),),
    ),
    ModelEntry(
        "hidream_o1",
        IMAGE,
        "HiDream-O1",
        "🌙",
        "HiDream-O1 pixel-space model (full or dev), text-to-image or image-to-image.",
        "hidream_o1",
        preset_folders=("HiDream_O1_Training_Configs",),
        badges=("New",),
        model_folder="Training_Models_HiDream_O1",
        download_bundles=(("HiDream-O1 Full", "hidream_o1"), ("HiDream-O1 Dev", "hidream_o1_dev")),
    ),
    ModelEntry(
        "ernie_image",
        IMAGE,
        "ERNIE-Image",
        "🧠",
        "Baidu ERNIE-Image text-to-image (LoRA).",
        "ernie_image",
        modes=(LORA,),
        preset_folders=("ERNIE_Image_Training_Configs",),
        badges=("New",),
        model_folder="Training_Models_ERNIE_Image",
        download_bundles=(("ERNIE-Image models", "ernie_image"),),
    ),
    ModelEntry(
        "mage_flow",
        IMAGE,
        "Mage-Flow",
        "🪄",
        "Microsoft Mage-Flow text-to-image (LoRA).",
        "mage_flow",
        modes=(LORA,),
        preset_folders=("Mage_Flow_Training_Configs",),
        fixed={"is_edit": False},
        locked_keys=frozenset({"is_edit"}),
        badges=("New",),
        model_folder="Training_Models_Mage_Flow",
        download_bundles=(("Mage-Flow models", "mage_flow"),),
    ),
    ModelEntry(
        "mage_flow_edit",
        IMAGE,
        "Mage-Flow Edit",
        "🖌️",
        "Mage-Flow image editing with reference (control) images (LoRA).",
        "mage_flow",
        modes=(LORA,),
        preset_folders=("Mage_Flow_Edit_Training_Configs",),
        fixed={"is_edit": True, "use_control_images": True},
        locked_keys=frozenset({"is_edit", "use_control_images"}),
        badges=("New",),
        model_folder="Training_Models_Mage_Flow",
        download_bundles=(("Mage-Flow Edit models", "mage_flow_edit"),),
    ),
    ModelEntry(
        "flux1_dev",
        IMAGE,
        "FLUX.1 Dev",
        "🔥",
        "FLUX.1 dev with the proven Kohya DreamBooth / LoRA recipes (kohya sd-scripts).",
        "kohya:flux1_dev",
        preset_folders=("FLUX_1_Dev_Training_Configs",),
        badges=("Kohya",),
        model_folder="Training_Models_FLUX_1",
        download_bundles=(("FLUX.1 Dev models", "flux1_dev"),),
    ),
    ModelEntry(
        "sdxl",
        IMAGE,
        "SDXL",
        "🏛️",
        "Stable Diffusion XL fine-tuning and LoRA (kohya sd-scripts), e.g. RealVisXL.",
        "kohya:sdxl",
        preset_folders=("SDXL_Training_Configs",),
        badges=("Kohya",),
        model_folder="Training_Models_SDXL",
        download_bundles=(("SDXL RealVis v5 model", "sdxl_realvis"),),
    ),
    # ------------------------------------------------------------------ video models
    ModelEntry(
        "wan22",
        VIDEO,
        "Wan 2.2",
        "🌊",
        "Wan 2.2 A14B dual-model (high + low noise): text-to-video, image-to-video and image LoRAs.",
        "wan",
        modes=(LORA,),
        preset_folders=("Wan22_Training_Configs",),
        preset_preference=("Text_To_Video",),
        choice_overrides={
            "task": (
                ("t2v-A14B · Text-to-Video (Wan 2.2 dual model)", "t2v-A14B"),
                ("i2v-A14B · Image-to-Video (Wan 2.2 dual model)", "i2v-A14B"),
            )
        },
        default_overrides={"task": "t2v-A14B"},
        model_folder="Training_Models_Wan",
        download_bundles=(("Wan 2.2 text-to-video models", "wan22_t2v"), ("Wan 2.2 image-to-video models", "wan22_i2v")),
    ),
    ModelEntry(
        "wan21",
        VIDEO,
        "Wan 2.1",
        "🌊",
        "Wan 2.1 14B / 1.3B: text-to-video, image-to-video, first-last-frame and Fun-Control.",
        "wan",
        modes=(LORA,),
        preset_folders=("Wan_21_Training_Configs",),
        choice_overrides={
            "task": (
                ("t2v-14B · Text-to-Video 14B", "t2v-14B"),
                ("t2v-1.3B · Text-to-Video 1.3B (faster)", "t2v-1.3B"),
                ("i2v-14B · Image-to-Video 14B", "i2v-14B"),
                ("t2i-14B · Text-to-Image 14B", "t2i-14B"),
                ("flf2v-14B · First-Last-Frame-to-Video 14B", "flf2v-14B"),
                ("t2v-1.3B-FC · Fun-Control 1.3B", "t2v-1.3B-FC"),
                ("t2v-14B-FC · Fun-Control 14B", "t2v-14B-FC"),
                ("i2v-14B-FC · Image-to-Video Fun-Control 14B", "i2v-14B-FC"),
            )
        },
        default_overrides={"task": "t2v-14B"},
        model_folder="Training_Models_Wan",
        download_bundles=(("Wan 2.1 text-to-video models", "wan21_t2v"),),
    ),
    ModelEntry(
        "ltx23",
        VIDEO,
        "LTX 2.3",
        "🎬",
        "Lightricks LTX 2.3 (22B) video or joint audio + video, LoRA and full fine-tuning.",
        "ltx2",
        preset_folders=("LTX_2.3_Training_Configs",),
        choice_overrides={"ltx2_mode": (("Video only", "video"), ("Audio + Video", "av"))},
        default_overrides={"ltx2_mode": "video"},
        badges=("🔊 Audio",),
        model_folder="Training_Models_LTX_2",
        download_bundles=(("LTX 2.3 models", "ltx2"), ("LTX 2.3 Gemma full text encoder", "ltx2_gemma_full")),
    ),
    ModelEntry(
        "minimax_h3",
        VIDEO,
        "MiniMax H3",
        "🎞️",
        "MiniMax H3 video + audio (T2VA, first/last-frame and reference modes), LoRA.",
        "minimax_h3",
        modes=(LORA,),
        preset_folders=("MiniMax_H3_Training_Configs",),
        badges=("🔊 Audio",),
        model_folder="Training_Models_MiniMax_H3",
        download_bundles=(("MiniMax H3 models", "minimax_h3"), ("MiniMax H3 INT8 text encoder", "minimax_h3_int8_te"), ("MiniMax H3 Ref2VA", "minimax_h3_ref2va")),
    ),
    ModelEntry(
        "kandinsky6",
        VIDEO,
        "Kandinsky 6",
        "🎥",
        "Kandinsky 6 video + audio (text or text+image to audio-video), LoRA.",
        "kandinsky6",
        modes=(LORA,),
        preset_folders=("Kandinsky_6_Training_Configs",),
        badges=("New", "🔊 Audio"),
        model_folder="Training_Models_Kandinsky_6",
        download_bundles=(("Kandinsky 6 Lite models", "kandinsky6_lite"),),
    ),
    # ------------------------------------------------------------------ audio models
    ModelEntry(
        "ltx23_audio",
        AUDIO,
        "LTX 2.3 Audio",
        "🎵",
        "Audio-only LoRA / fine-tune on the LTX 2.3 audio branch (speech, music, sound effects).",
        "ltx2",
        preset_folders=("LTX_2.3_Audio_Training_Configs",),
        fixed={"ltx2_mode": "audio"},
        locked_keys=frozenset({"ltx2_mode"}),
        default_overrides={"lora_target_preset": "audio", "sample_audio_only": True},
        model_folder="Training_Models_LTX_2",
        download_bundles=(("LTX 2.3 models", "ltx2"),),
    ),
)

ENTRY_BY_KEY: dict[str, ModelEntry] = {entry.key: entry for entry in ENTRIES}
DEFAULT_ENTRY = {IMAGE: "krea2", VIDEO: "wan22", AUDIO: "ltx23_audio"}


def entries_for(category: str) -> list[ModelEntry]:
    return [entry for entry in ENTRIES if entry.category == category]


def get_entry(key: str) -> ModelEntry:
    return ENTRY_BY_KEY[key]


def backends_for(category: str) -> list[str]:
    return list(dict.fromkeys(entry.backend for entry in entries_for(category)))
