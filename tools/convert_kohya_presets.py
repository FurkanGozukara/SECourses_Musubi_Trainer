"""Convert the kohya_ss GUI JSON presets of the Kohya FLUX / SDXL app into unified-trainer TOML presets.

Re-runnable: existing TOML / README / image files with the same names are overwritten, nothing is deleted.

    venv\\Scripts\\python.exe tools\\convert_kohya_presets.py
    venv\\Scripts\\python.exe tools\\convert_kohya_presets.py --source F:/Kohya_FLUX_DreamBooth_LoRA_v31 --target ../Training_Presets

Sources (the For_RunPod copies only differ in paths and are skipped):
  FLUX_LoRA_Training_LoRA_Tab_Configs                -> FLUX_1_Dev_Training_Configs/LoRA_Training
  FLUX_DreamBooth_Tab_Fine_Tuning_Training_Configs   -> FLUX_1_Dev_Training_Configs/Fine_Tuning_Training
  SDXL_DreamBooth_Fine_Tuning_Tab_Configs            -> SDXL_Training_Configs/Fine_Tuning_Training
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from musubi_tuner_gui.preset_library import QUALITY_PATTERN, TIER_PATTERN  # noqa: E402
from musubi_tuner_gui.unified import kohya_backend as kb  # noqa: E402

DEFAULT_SOURCE = Path("F:/Kohya_FLUX_DreamBooth_LoRA_v31")
DEFAULT_TARGET = APP_DIR.parent / "Training_Presets"

SOURCES = (
    ("FLUX_LoRA_Training_LoRA_Tab_Configs", "flux1_dev", "lora", "FLUX_1_Dev_Training_Configs/LoRA_Training"),
    ("FLUX_DreamBooth_Tab_Fine_Tuning_Training_Configs", "flux1_dev", "full", "FLUX_1_Dev_Training_Configs/Fine_Tuning_Training"),
    ("SDXL_DreamBooth_Fine_Tuning_Tab_Configs", "sdxl", "full", "SDXL_Training_Configs/Fine_Tuning_Training"),
)

# machine-specific values of the source JSONs that must not ship
NEUTRAL_KEYS = {
    "parent_folder_path": "",
    "reg_data_dir": "",
    "dataset_config": "",
    "dataset_config_mode": kb.FOLDER_DATASET_MODE,
    "sample_prompts": "",
    "resume": "",
    "network_weights": "",
    "logging_dir": "",
    "log_tracker_config": "",
    "resume_from_huggingface": "",
}


def _portable_model_paths(model_key: str, values: dict) -> dict:
    """Install-root-relative model paths (relocated to the real files when a preset is loaded)."""
    model = kb.KOHYA_MODELS[model_key]
    result = {}
    for key, default_name in model["model_files"].items():
        name = re.split(r"[\\/]", str(values.get(key) or "").strip())[-1] or default_name
        result[key] = f"{model['model_dir_name']}/{name}"
    if "default_vae" in model:
        vae = str(values.get("vae") or "").strip()
        if vae and not kb._looks_like_hf_repo_id(vae):
            vae = f"{model['model_dir_name']}/{re.split(r'[/]', vae.replace(chr(92), '/'))[-1]}"
        result["vae"] = vae
    return result


def _settings_summary(model_key: str, training_type: str, values: dict) -> list[str]:
    parts = [f"{values['dataset_resolution_width']}x{values['dataset_resolution_height']}"]
    if training_type == "lora":
        parts.append(f"rank {values['network_dim']}")
    if int(values["dataset_batch_size"]) > 1:
        parts.append(f"batch {values['dataset_batch_size']}")
    parts.append(f"LR {values['learning_rate']:g}")
    parts.append(f"block swap {values['blocks_to_swap']}")
    if model_key == "flux1_dev" and training_type == "lora" and values.get("fp8_scaled"):
        parts.append("FP8 scaled")
    if model_key == "flux1_dev" and training_type == "lora" and float(values.get("text_encoder_lr") or 0) > 0:
        parts.append("CLIP-L trained")
    if model_key == "flux1_dev" and values.get("apply_t5_attn_mask"):
        parts.append("T5 attention mask")
    if model_key == "flux1_dev" and values.get("cache_text_encoder_outputs_on_cpu"):
        parts.append("text encoders cached on CPU")
    if model_key == "sdxl" and training_type == "full":
        te1 = float(values.get("learning_rate_te1") or 0)
        parts.append(f"text encoder 1 trained at {te1:g}" if te1 > 0 else "text encoders frozen and cached")
    if not values.get("gradient_checkpointing"):
        parts.append("no gradient checkpointing")
    if int(values["num_processes"]) > 1:
        parts.append(f"{values['num_processes']} GPUs ({values['gpu_ids']}), {values['max_train_epochs']} epochs")
    return parts


def convert(source: Path, target: Path) -> list[dict]:
    rows = []
    for folder, model_key, training_type, target_folder in SOURCES:
        source_dir = source / folder
        if not source_dir.is_dir():
            print(f"skip (missing): {source_dir}")
            continue
        out_dir = target / target_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        model = kb.KOHYA_MODELS[model_key]
        for json_path in sorted(source_dir.glob("*.json")):
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
            values = kb.kohya_defaults(model_key, training_type)
            converted, notes = kb.convert_kohya_gui_json(data, model_key, training_type)
            values.update(converted)
            values.update(NEUTRAL_KEYS)
            values.update(_portable_model_paths(model_key, values))
            values["output_dir"] = f"../{model['output_dir_name']}"
            values["output_name"] = kb.kohya_defaults(model_key, training_type)["output_name"]
            name = kb.kohya_preset_file_name(json_path.name)
            tier_match = TIER_PATTERN.search(name)
            tier, used = float(tier_match.group("tier")), float(tier_match.group("used"))
            quality_match = QUALITY_PATTERN.search(name)
            quality = int(quality_match.group("quality")) if quality_match else None
            kind = "LoRA" if training_type == "lora" else "fine-tuning (DreamBooth)"
            summary = _settings_summary(model_key, training_type, values)
            description = (
                f"{model['short_name']} {kind} · {tier:g} GB GPUs · peak {used:g} GB (measured in the kohya_ss app)"
                + (f" · Quality {quality}" if quality else "")
                + " · "
                + " · ".join(summary)
            )
            comments = (
                f"Converted from the kohya_ss GUI preset {folder}/{json_path.name} by tools/convert_kohya_presets.py.",
                "The VRAM peak in the file name was measured with the kohya_ss app; it has not been re-measured in this trainer yet.",
            )
            # the kohya_ss GUI counted its epochs in steps of its own; where sd-scripts' epochs train more, say so
            length = [kb._int(data.get(key), default) for key, default in (("epoch", 0), ("num_processes", 1), ("train_batch_size", 1), ("gradient_accumulation_steps", 1))]
            steps_example = kb.kohya_steps_example(*length) if kb.kohya_steps_note(*length) in notes else None
            if steps_example:
                comments += (kb.kohya_steps_note(*length),)
            text = kb.build_kohya_preset_toml(model_key, training_type, values, description, comments)
            (out_dir / name).write_text(text, encoding="utf-8")
            rows.append(
                {
                    "folder": target_folder,
                    "file": name,
                    "tier": tier,
                    "used": used,
                    "quality": quality,
                    "notes": ", ".join(summary),
                    "source": json_path.name,
                    "conversion_notes": notes,
                    "steps_example": steps_example,
                }
            )
            print(f"{target_folder}/{name}  <-  {json_path.name}")
        for image in sorted(source_dir.glob("*.jpg")):
            shutil.copy2(image, out_dir / image.name)
            print(f"{target_folder}/{image.name}  (image)")
    return rows


README_INTRO = {
    "FLUX_1_Dev_Training_Configs/LoRA_Training": (
        "FLUX.1 Dev LoRA presets (kohya sd-scripts `flux_train_network.py`)",
        "Converted 1:1 from the Kohya FLUX app's LoRA tab presets. Quality 1 keeps the full bf16 FLUX model and trains a CLIP-L "
        "LoRA together with the transformer (T5-XXL outputs cached); FP8 Scaled presets load FLUX in scaled fp8. All train "
        "Adafactor at a fixed 5e-5 learning rate for 200 epochs at 1 repeat (save every 25 epochs); the 4x GPU presets use 1e-4 "
        "and train the same 200 epochs spread over 4 GPUs (the kohya_ss GUI ran their epoch 50 on every GPU).",
    ),
    "FLUX_1_Dev_Training_Configs/Fine_Tuning_Training": (
        "FLUX.1 Dev full fine-tuning presets (kohya sd-scripts `flux_train.py`, DreamBooth)",
        "Converted 1:1 from the Kohya FLUX app's DreamBooth tab presets: Adafactor with fused backward pass, learning rate 4e-6, "
        "200 epochs at 1 repeat, a full 23.8 GB checkpoint every 25 epochs. Text encoder outputs are cached.",
    ),
    "SDXL_Training_Configs/Fine_Tuning_Training": (
        "SDXL full fine-tuning presets (kohya sd-scripts `sdxl_train.py`, DreamBooth)",
        "Converted 1:1 from the Kohya app's SDXL DreamBooth presets. They were tuned with `<repeats>_ohwx man` training folders "
        "plus a regularization folder (e.g. `1_man` class images) and train 1 epoch, saving every 451 steps. Quality 1 also "
        "trains text encoder 1 (3e-6, U-Net 8e-6); the 6-8 GB presets freeze and cache both text encoders. Captions come from "
        "the folder name (caption extension `.caption`); set `.txt` to use caption files.",
    ),
}


def write_readmes(target: Path, rows: list[dict]) -> None:
    for folder, (title, intro) in README_INTRO.items():
        folder_rows = sorted((row for row in rows if row["folder"] == folder), key=lambda row: (-row["tier"], -row["used"], row["file"]))
        if not folder_rows:
            continue
        out_dir = target / folder
        lines = [
            f"# {title}",
            "",
            intro,
            "",
            "The VRAM in each file name was measured with the kohya_ss app (old environment). These presets have not been "
            "re-measured in this trainer yet, and several leave less than the usual 1.5 GB of headroom: on a GPU that also "
            "drives your display, pick the next lower tier or add a few blocks to swap. Model paths are relative to the "
            "install folder (`Training_Models_FLUX_1` / `Training_Models_SDXL`) and are relocated to your downloaded files "
            "when a preset is loaded.",
            "",
            "| File | GPU tier | Measured VRAM | Quality | Notes |",
            "|---|---:|---:|---:|---|",
        ]
        for row in folder_rows:
            quality = row["quality"] if row["quality"] is not None else "-"
            lines.append(f"| `{row['file']}` | {row['tier']:g} GB | {row['used']:g} GB | {quality} | {row['notes']} |")
        examples = [row for row in folder_rows if row.get("steps_example")]
        if examples:
            lines += [
                "",
                "Epochs here are sd-scripts epochs, each rounded up to whole steps per GPU. The kohya_ss app computed "
                "images × repeats ÷ batch ÷ gradient accumulation × epoch steps instead, so with an image count that is not a "
                "multiple of batch × GPUs × accumulation these presets train more steps than they did there. To match a "
                "kohya_ss run exactly, set Max Training Epochs 0 and Max Training Steps to its step count.",
                "",
            ]
            for row in examples:
                images, steps, kohya_steps = row["steps_example"]
                lines.append(f"- `{row['file']}`: {images} images → {steps} steps here, {kohya_steps} in the kohya_ss app")
        images = sorted(path.name for path in out_dir.glob("*.jpg"))
        if images:
            lines += ["", "Comparison images from the kohya_ss app:", ""]
            lines += [f"- [{name}](<{name}>)" for name in images]
        lines += ["", "Regenerate with `tools/convert_kohya_presets.py` (source: the kohya_ss JSON presets).", ""]
        (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"{folder}/README.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Kohya FLUX app folder holding the *_Configs folders")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Training_Presets folder of this install")
    parser.add_argument("--show-notes", action="store_true", help="print the conversion notes of every preset")
    args = parser.parse_args(argv)
    rows = convert(args.source, args.target)
    write_readmes(args.target, rows)
    if args.show_notes:
        for row in rows:
            print(f"\n{row['folder']}/{row['file']}:")
            for note in row["conversion_notes"]:
                print(f"  - {note}")
    print(f"\n{len(rows)} presets written to {args.target}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
