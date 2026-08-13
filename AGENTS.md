# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, or any other LLM-based tool) working in this repository.

## What this is

SECourses Musubi Trainer — a Gradio GUI (`musubi_tuner_gui/`) wrapping the vendored `musubi-tuner` training backend (`musubi-tuner/`) to train LoRAs and full fine-tunes for Qwen Image, Wan 2.1/2.2, FLUX / FLUX.2 / FLUX Klein, Z-Image, Ideogram 4, Krea 2, and LTX 2.3. `README.md` is the user-facing feature tour (screenshots, install instructions, marketing); this file and `AI_TRAINER_GUARDRAILS.md` are for people/agents changing the code.

## Read this before adding a new trainer tab or model

**`AI_TRAINER_GUARDRAILS.md` documents mistakes already made once in this codebase and fixed — read it before adding or modifying a trainer tab.** They're easy to reintroduce when copy-pasting an existing `*_lora_gui.py` as your starting point, because each bug only surfaces later, in a scenario a quick manual test won't hit:

1. **Logging paths** — an empty `logging_dir` reaching the runtime TOML silently becomes a filesystem-root path on the backend.
2. **Training-mode round-trip** — if the new trainer supports both LoRA and Full Fine-Tuning, `training_mode` must be persisted in the runtime TOML and correctly re-inferred when loading an older TOML that lacks it. Get this wrong and reloading a saved/failed full-finetune run silently retrains it as LoRA instead, with no error shown.

Update `AI_TRAINER_GUARDRAILS.md` with a new guardrail (same format: Rule / Why this exists / Required implementation pattern / Regression check) any time you fix a bug in this codebase that a future "add a new model" pass could plausibly reintroduce.

## Architecture map

- `gui.py` — mounts each trainer's `*_tab()` function into one Gradio `Blocks` app. This is the source of truth for which trainers are actually live.
- `musubi_tuner_gui/*_lora_gui.py` — one file per trainer (e.g. `qwen_image_lora_gui.py`, `wan_lora_gui.py`, `zimage_lora_gui.py`, `modern_image_lora_gui.py` for Ideogram 4 + Krea 2). Each defines the tab's UI, a `train_*_model()` function that builds the launch command and writes the runtime config TOML, and `open_*_configuration()` / `save_*_configuration()` for the Load/Save buttons.
- `musubi_tuner_gui/common_gui.py` — shared infra, including `SaveConfigFile()` (💾 Save button, hand-managed presets) and `SaveConfigFileToRun()` (the actual runtime TOML written to `output_dir`, the one passed to the backend via `--config_file`). These two paths have different exclusion rules — check both when changing what a trainer persists.
- `musubi_tuner_gui/full_finetune_gui.py` — shared LoRA-vs-Full-Fine-Tuning helpers used by FLUX, FLUX.2, FLUX Klein, Qwen Image, Z-Image, Ideogram 4, and Krea 2. Wan and LTX-2 don't use it: Wan's full-finetune mode is currently disabled in the UI (single-choice Radio, forced back to LoRA with a warning if anything else reaches it), and LTX-2 has no full-finetune mode at all.
- `musubi-tuner/` — the vendored training backend (actual training scripts, argparse setup, `read_config_from_file`). Unknown TOML keys are harmless there — they become unused `argparse.Namespace` attributes and are never read (see `musubi-tuner/src/musubi_tuner/training/parser_common.py::read_config_from_file`). This is why it's safe for the GUI to persist GUI-only bookkeeping fields (like `training_mode`) into the runtime TOML.
- `tests/test_full_finetune_gui.py` — the relevant suite for anything touching LoRA/full-finetune mode switching or config save/load round-tripping.

Not currently mounted in `gui.py` (present but dormant — still exercised directly by tests, so keep them working when changing shared helpers): `flux2_lora_gui.py` and `flux_klein_lora_gui.py`'s own tab-building functions are superseded by the unified `flux_lora_gui.py` tab (which handles both families via a `model_family` selector), but their `train_*`/`open_*`/`save_*` functions are still imported directly by `tests/test_full_finetune_gui.py`, and `flux_klein_lora_gui.py` still aliases `flux2_lora_gui.py`'s load/save functions. `lora_gui.py` (a legacy HunyuanVideo LoRA trainer) is fully dead, unmounted, and untested — don't build on it.

## Running tests

```
venv/Scripts/python.exe -m pytest tests/ -q
```

Keep the full suite green. When changing `full_finetune_gui.py` or any `*_lora_gui.py`'s save/load logic, also sanity-check that the actual Gradio tab still builds without error — most trainer functions are only exercised as plain Python calls in the test suite, not through real Gradio component construction, so wiring mistakes (bad `.then()`/`.change()` inputs or outputs, references to undefined components) can slip past pytest alone:

```python
import gradio as gr
from musubi_tuner_gui.<trainer>_lora_gui import <trainer>_lora_tab
with gr.Blocks():
    <trainer>_lora_tab(headless=True, config={})
```

For anything touching config Load/Open behavior specifically, prefer verifying live in the running app (`python gui.py`) over trusting the tab-build check alone — some bugs (like the numeric-bounds gotcha in Guardrail 2) only appear once real Gradio event round-trips run in a browser, not from constructing the component tree.
