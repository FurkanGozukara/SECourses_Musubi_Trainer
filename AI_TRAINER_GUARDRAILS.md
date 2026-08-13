## AI Trainer Guardrails

Use this checklist whenever adding a new trainer tab or training flow. Each guardrail below documents a mistake already made once in this codebase (and fixed) — they are easy to reintroduce when copy-pasting an existing `*_lora_gui.py` as the starting point for a new one, because the bug only shows up later, when a user reloads a saved/failed run.

---

## Guardrail 1: Logging paths

### Rule

- Never write `logging_dir = ""` into a runtime training TOML.
- Never write root-like logging paths such as `/`, `\`, `F:/`, or `C:/`.
- If logging is disabled, omit both `logging_dir` and `log_with` from the runtime TOML.
- If logging is enabled by UI fields, debug mode, or extra CLI args, default the base logging path to `output_dir/logs`.
- Let the backend create the final timestamped run folder under that base path.

### Why this exists

Empty-string `logging_dir` is treated by the backend as a real path. It gets a timestamp appended and turns into a root-level path:

- Linux: `/20260325100338`
- Windows: `F:\20260325050433`

That causes permission errors on Linux and misplaced TensorBoard folders on Windows.

### Required implementation pattern

- Route new runtime config generation through `musubi_tuner_gui/common_gui.py::SaveConfigFileToRun`.
- Keep `_normalize_logging_fields_for_run_config()` in the path for all new trainers.
- Keep backend protection in `musubi-tuner/src/musubi_tuner/hv_train.py` and `musubi-tuner/src/musubi_tuner/hv_train_network.py` so old/bad configs are still safe.

### Regression check

Before finishing a new trainer:

- Verify disabled logging produces no `logging_dir` and no `log_with` in the generated runtime TOML.
- Verify enabled TensorBoard logging resolves to `output_dir/logs/...`, not a filesystem root path.

---

## Guardrail 2: Training-mode round-trip (LoRA vs. Full Fine-Tuning)

Applies to any trainer that offers both a LoRA mode and a Full Fine-Tuning / DreamBooth mode (currently: FLUX, FLUX.2, FLUX Klein, Qwen Image, Z-Image, Ideogram 4, Krea 2). Skip this guardrail only if the new trainer is LoRA-only (like LTX-2) or has its full-finetune mode fully disabled in the UI (like Wan currently does).

### Rule

- The runtime TOML written to `output_dir` (via `SaveConfigFileToRun`) must include `training_mode`, in addition to correctly omitting `network_module` (and the rest of the LoRA-only keys) for full-finetune runs.
- The "Load"/"Open" config loader must not just do "if key present in file use it, else keep whatever the GUI currently shows" for `training_mode`. That fallback is exactly the bug: it silently trusts stale on-screen state instead of the file being loaded.

### Why this exists

A full-finetune run's runtime TOML correctly omits `network_module` (the backend's `--network_module` default is `None`, which means "full fine-tune the base model" — see `musubi-tuner/src/musubi_tuner/training/parser_common.py`). It used to *also* omit `training_mode`, since that's a GUI-only bookkeeping field with no backend meaning. That combination is the trap: if that run failed and the user reloaded the exact TOML it wrote (to inspect it or retry), the loader had no signal that it had been a full-finetune run. The mode selector silently fell back to "LoRA Training," and clicking Start Training launched a LoRA run instead of resuming the fine-tune — with no error, no warning, nothing on screen indicating anything was wrong.

### Required implementation pattern

Use the shared helpers in `musubi_tuner_gui/full_finetune_gui.py` — do not hand-roll this per trainer:

- `TRAINING_MODE_CHOICES`, `LORA_TRAINING_MODE`, `FULL_FINE_TUNING_MODE` — use these as your Radio's `choices=` when possible. If your trainer needs its own label (e.g. Qwen Image and Z-Image use `"DreamBooth Fine-Tuning"` instead of the canonical `"Full Fine-Tuning"`), that's fine — `normalize_training_mode()` / `is_full_fine_tuning()` accept both via an alias table, but you must tell the load-side helper about your label (see below).
- **Write side**: build your `SaveConfigFileToRun(..., exclusion=...)` list by calling `training_mode_runtime_exclusions(training_mode)` and folding its result in. Do **not** additionally hardcode `"training_mode"` into your own exclusion list — it's deliberately *not* in what that helper returns, specifically so it round-trips. (Persisting it is safe: the backend's `read_config_from_file` merges unknown TOML keys into an `argparse.Namespace` and never uses them — see `musubi-tuner/src/musubi_tuner/training/parser_common.py::read_config_from_file`.)
- **Load side**: in your `open_*_configuration()` function's per-key loop, special-case `training_mode` to resolve via `infer_training_mode_from_loaded_config(data, full_mode_label=<your Radio's actual full-finetune choice string>)` instead of the generic "in data ? file value : current UI value" branch. Pass `full_mode_label=` whenever your Radio doesn't use the canonical `"Full Fine-Tuning"` string. This one call handles three cases correctly: the modern case (`training_mode` present in the file), the legacy/broken case (`training_mode` absent, infers full-finetune from `network_module` also being absent), and the ordinary LoRA case (`training_mode` absent but `network_module` present).
- **Panel visibility**: if a `training_mode.change()` handler shows/hides accordions (e.g. "LoRA Settings" vs "Full Fine-Tuning Settings"), also chain the *same* sync function via `.then()` off both the Open and Load button `.click()` events, reading the now-updated `training_mode` component as input. Gradio does **not** re-fire `.change()` for a component whose value was set programmatically as part of another event's output tuple — see `flux_lora_gui.py`, `flux2_lora_gui.py`, and `modern_image_lora_gui.py` for the established `.then()` pattern.
- **Gotcha — keep the `.then()` sync function narrowly scoped to visibility.** If your `training_mode.change()` handler *also* recomputes some other component's value (e.g. `modern_image_lora_gui.py`'s `blocks_to_swap`, whose allowed maximum shrinks by 1 for Ideogram in full-finetune mode), do **not** reuse that same function for the Load/Open `.then()` chain. Doing so was tried and reverted: a value from the parent Load event and a bound from the `.then()` follow-up can validate against each other in the wrong order across the two server round-trips, producing a spurious `gradio.exceptions.Error: 'Value N is greater than maximum value M.'` that has nothing to do with the actual bug being fixed. Write a second, minimal function that returns only the accordion visibility updates (see `modern_image_lora_gui.py::sync_training_mode_visibility` vs. `toggle_training_mode`) and wire the full one only to the direct `.change()` handler.

### Regression check

Before finishing a new trainer with a LoRA/Full-Fine-Tuning toggle:

- Start (or use "Print Command" to preview) a Full Fine-Tuning run. Confirm the written runtime TOML contains `training_mode` and does **not** contain `network_module`.
- Take that exact TOML, reload it via the "Load" button, and confirm the mode selector shows Full Fine-Tuning again — not LoRA.
- Hand-edit a copy of that TOML to delete the `training_mode` line (simulating a file saved by a pre-fix version, or any other trainer that predates this pattern) and reload it. It must still resolve to Full Fine-Tuning, inferred from the missing `network_module`.
- Do the mirror check: reload an ordinary LoRA runtime TOML (has `network_module`, no `training_mode`) while the GUI currently shows Full Fine-Tuning, and confirm it correctly switches back to LoRA.
- If mode drives panel visibility, confirm the correct panel is visible after Load/Open, not just after manually clicking the mode radio — and confirm no numeric-bounds error appears in the process (see the gotcha above).
