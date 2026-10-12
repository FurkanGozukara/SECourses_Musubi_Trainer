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

Applies to any trainer that offers both a LoRA mode and a Full Fine-Tuning / DreamBooth mode (currently: FLUX, FLUX.2, FLUX Klein, Qwen Image, Qwen-Image 2.1, HiDream-O1, Z-Image, Ideogram 4, Krea 2, LTX 2.3). Skip this guardrail only if the new trainer is LoRA-only (like ERNIE-Image, Mage-Flow, Kandinsky 6, MiniMax H3) or has its full-finetune mode fully disabled in the UI (like Wan currently does).

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

---

## Guardrail 3: Attention-mode resolution ("flash_auto") for new architectures

### Rule

- Any new backend model whose attention routes through `musubi-tuner/src/musubi_tuner/modules/attention.py` must accept the attention mode string `"flash_auto"` in whatever `attn_mode` validation it performs.
- Never validate `attn_mode` against only the upstream set `{"torch", "sdpa", "flash", "flash3", "sageattn", "xformers"}`.

### Why this exists

This fork's `trainer_base.py` resolves the `--sdpa` flag through `resolve_sdpa_backend()`, which returns the fork-specific mode `"flash_auto"` on machines where PyTorch's native flash SDPA is unavailable but the external FlashAttention package passes a forward+backward probe (a common Windows configuration). Upstream models validate `attn_mode` against a closed set that does not contain it. The MiniMax H3 tab shipped with `--sdpa` as its default attention and the very first training launch crashed at transformer load with `ValueError: Unsupported MiniMax-H3 attention mode: flash_auto` — after latent and text-encoder caching had already run for minutes. Nothing in the GUI-level tests catches this because the failure happens inside the backend model constructor on a real launch.

### Required implementation pattern

- When merging a new architecture from upstream, grep its model file for `attn_mode not in` and extend the accepted set with `"flash_auto"` (see `minimax_h3/model.py`). The shared `modules.attention` dispatch already implements the runtime behavior; only the closed-set validation needs the extra member.
- Keep the regression test in `musubi-tuner/tests/test_attention_backend_selection.py` (`test_minimax_h3_model_accepts_resolved_flash_auto_mode`) and add an equivalent one for each new architecture.

### Regression check

Before finishing a new trainer: run a real "Start Training" (or at minimum construct the model with `attn_mode="flash_auto"`) on a machine where `resolve_sdpa_backend()` returns `"flash_auto"`, not just `pytest` — the crash only appears at real transformer load time.

Applies to every architecture added in V35 as well: Qwen-Image 2.1 (`qwen_image_21_model.segmented_attention`) and Mage-Flow (`mage_flow/attention.packed_attention`, `MageFlow.set_attention_backend`) shipped with their own attention dispatchers that raised on `flash_auto`; both now resolve it per call with `resolve_automatic_flash_mode` / `external_flash_supports_inputs`, and `test_attention_backend_selection.py` covers Qwen-Image 2.1. Kandinsky 6 ignores `attn_mode` (its bundled engine is SDPA-only), ERNIE-Image routes through `modules.attention`.

---

## Guardrail 4: Fork compile residency selectors on new trainers

### Rule

- Every `model_utils.compile_transformer(...)` call in a `*_train_network.py` must pass `offloaders=[...]` (one entry per compiled block container: the container's block-swap offloader, or `None`).

### Why this exists

The fork's compile path decides which blocks stay resident (and therefore compiled with their Linears) from the offloaders. Upstream trainers do not pass them; merged as-is, Qwen-Image 2.1, Kandinsky 6, ERNIE-Image and Mage-Flow compiled every block as if resident. `tests/test_torch_compile_toolchain.py::ResidentBlockCompileTests::test_every_training_compile_call_site_supplies_residency_selectors` fails for any new call site without it.

### Required implementation pattern

`offloaders=[model.offloader if self.blocks_to_swap > 0 else None]` for a single swapped container; `None` entries for containers that are never swapped (e.g. Kandinsky 6 `[None, None, transformer.offloader]`). Update fake `compile_transformer` stubs in merged tests to accept `offloaders=None`.

### Regression check

Run the backend suite after every upstream/fork merge; the residency-selector test lists the offending `file:line`.

---

## Guardrail 5: Debug-mode logging goes through `log_with`, never a bare CLI flag

### Rule

- When the Advanced → Debug Mode dropdown selects "Enable Logging (TensorBoard/WandB/All)", set the `log_with` parameter **before** `SaveConfigFileToRun` runs and do not append `--log_with` to the command line.

### Why this exists

`debug_mode` is a GUI-only key excluded from the runtime TOML before logging normalization, so the old pattern (append `--log_with tensorboard` to the CLI) reached the backend without a `logging_dir` and failed with "logging_dir is required when log_with is tensorboard" — after caching had already run. Found in the Ideogram 4 / Krea 2 module during the V35 review; `image_models_gui.py` and `modern_image_lora_gui.py` now map it into `log_with` so `_normalize_logging_fields_for_run_config` resolves `output_dir/logs`.

### Regression check

`tests/test_image_models_gui.py::test_debug_logging_resolves_logging_dir_under_output` (parametrized over every spec).

---

## Guardrail 6: Pin auxiliary assets to sources that still exist

### Rule

- Before shipping a model whose loader downloads tokenizer/processor/config assets from a Hugging Face repo, verify the repo is still public (`HfApi().model_info(repo)` with and without a token). If it was removed, point the loader at a verified official copy (compare git blob ids file by file) and pin the revision.

### Why this exists

`microsoft/Mage-Flow` and `microsoft/Lens` were deleted from the Hub after their releases. Mage-Flow's loader fetched its Qwen3-VL processor assets from `microsoft/Mage-Flow/text_encoder`; every file there was byte-identical to `Qwen/Qwen3-VL-4B-Instruct`, so the fork now loads them from Qwen's repo at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Lens (never officially released, assets only in the deleted Microsoft repo) was not integrated.

---

## Guardrail 7: Trainers that override `process_batch` must use the upstream loop

### Rule

- A `*_train_network.py` whose trainer overrides `process_batch()` (Kandinsky 6, MiniMax H3, FLUX.2 Self-Flow) must subclass `musubi_tuner.training.trainer_base.NetworkTrainer`, not the fork-extended `NetworkTrainer` re-exported by `hv_train_network`.

### Why this exists

The fork binds `trainer_ext.NetworkTrainer.train` to its own `training/training_loop.py`, which calls `call_dit()` directly and never calls `process_batch()`. Kandinsky 6 (AkaneTendo25 branch) feeds its audio latents/noise through `process_batch()`, so after the V35 merge every K6 run died on the first step with `KeyError: 'audio_latents'` — no unit test covered the real loop. MiniMax H3 already imported the upstream base and worked.

### Regression check

Run one real training step (the VRAM lab or a 2-step GUI run) for every newly merged trainer, not only the unit tests.

---

## Guardrail 8: `GUIConfig()` is an empty configuration

### Rule

- `GUIConfig()` (no path) must never read `./config.toml`. Tabs without a defaults file (`TAB_DEFAULT_FILES[...] = None`) get an empty config and fill their own architecture defaults.

### Why this exists

A legacy HunyuanVideo `config.toml` (foreign `C:/Users/.../mp_rank_00_model_states_fp8.safetensors` paths, flow shift 7, 960x544, "Use TOML File") sits in the app folder. `GUIConfig()` used to load it implicitly, so every V35 spec-driven tab started with those values and showed its downloaded models as missing. Test: `tests/test_image_models_gui.py::test_tabs_without_defaults_file_ignore_stray_config_toml`.

---

## Guardrail 9: Block-swap maximums leave two resident blocks

### Rule

- A tab's `max_blocks_to_swap` for training is `num_blocks - 2`, even when the model's own assertion allows `num_blocks - 1`.

### Why this exists

Qwen-Image 2.1 (32 blocks) accepted 31 swapped blocks, then crashed in the backward recompute of a full fine-tune ("mat2 is on cpu"). HiDream-O1 (36 decoder layers) had the same `num_blocks - 1` ceiling. Both GUI specs now stop at 30 / 34.

---

## Guardrail 10: After an upstream merge, diff function signatures against upstream

### Rule

- After resolving merge conflicts, compare every function signature in `src/musubi_tuner` with `upstream/main` (AST walk over `git show upstream/main:<file>`); any parameter upstream has and the fork lost is a merge mistake until proven otherwise.
- Also compare the *methods* of classes the fork keeps its own copy of (the datasources in `dataset/image_video_dataset.py` vs upstream `dataset/datasources.py`): a method upstream added to its classes but missing from the fork's falls back to the base class, which may raise `NotImplementedError` only at run time.

### Why this exists

The fork's `save_text_encoder_output_cache_hidream_o1` (defined twice: `dataset/cache_io.py` and `dataset/image_video_dataset.py`) had lost upstream's `pixel_values` parameter while `hidream_o1_cache_text_encoder_outputs.py` passes it, so HiDream-O1 text caching failed for every dataset. The V35 check over 239 files found exactly this one gap. A second one only showed in the GPU preset lab: the fork's `VideoDirectoryDatasource` / `VideoJsonlDatasource` lacked upstream's `get_item_extras`, so MiniMax-H3 latent caching from a video folder raised `NotImplementedError` (test: `musubi-tuner/tests/test_video_datasource_item_extras.py`). `image_video_dataset.py` now re-exports the `cache_io` writer; test: `musubi-tuner/tests/test_hidream_o1_cache_contract.py`.
---

## Guardrail 11: Vendored transformers model code must follow transformers 5

### Rule

- Any model class copied from transformers into `musubi-tuner` (today: `hidream_o1/qwen3_vl_transformers.py`) must work with the pinned transformers 5.x. In particular a rotary embedding whose `rope_type` is `"default"` must not index `ROPE_INIT_FUNCTIONS["default"]` (removed in v5) and must expose `compute_default_rope_parameters`, which `PreTrainedModel._init_weights` calls.

### Why this exists

Upstream bumped to transformers 5.17 without touching HiDream-O1's vendored Qwen3-VL code, so every HiDream-O1 run failed while building the model (`KeyError: 'default'`, then `'NoneType' object is not callable` from the weight initializer). Test: `musubi-tuner/tests/test_hidream_o1_rope_transformers5.py` compares the inverse frequencies with transformers' own class. The next failure was `create_causal_mask(input_embeds=..., cache_position=...)`: v5 renamed the argument to `inputs_embeds` and dropped `cache_position`; `_create_causal_mask_compat` passes whichever the installed version accepts, and `tests/test_hidream_o1_text_forward_transformers5.py` runs a tiny forward + backward on the CPU.

The same v5 change broke LTX-2 text caching: transformers 5 stores Gemma 3 rotary buffers per layer type (`full_attention_inv_freq`, `sliding_attention_original_inv_freq`), which the single-file Gemma loader treated as missing weights. `base_encoder._materialize_meta_buffer` now rebuilds them from the per-layer RoPE parameters (test: `musubi-tuner/tests/test_ltx2_gemma_rope_transformers5.py`). After any transformers bump, run a fresh text-encoder caching pass for every model (the VRAM lab's cache measurements do this).
---

## Guardrail 12: Reference/control images that are enabled must exist

### Rule

- When a spec-driven tab generates its dataset TOML with reference/control images enabled (`use_control_images`: Qwen-Image 2.1 editing, Mage-Flow-Edit, HiDream-O1 i2i), every dataset folder must contain the control sub-folder; `create_dataset_config` raises a ValueError naming the folders that lack it.

### Why this exists

`generate_dataset_config_from_folders` silently leaves `control_directory` out when the sub-folder is missing, so a Mage-Flow-Edit run with an image-only dataset printed a clean workflow and wrote a dataset TOML without references; the edit run could only fail later, during caching or training. Test: `tests/test_image_models_gui.py::test_reference_images_enabled_without_control_folder_is_rejected`. Older tabs (FLUX.2, Qwen Image Edit) treat control images as optional on purpose and are not affected.

---

## Guardrail 13: Latent caches are not resolution-specific by name

### Rule

- Image latent cache files are named by the *source* image size (`img_1024x1024_<arch>.safetensors`), and training loads every `*_<arch>.safetensors` in its cache folder. A new latent cache writer must go through `save_latent_cache_common` (it records `bucket_size` in the metadata, which `--skip_existing` compares in `cache_latents.latent_cache_matches_bucket`); a new video dataset must keep `VideoDataset.drop_stale_video_caches` in its training path.
- Measurement tooling must give each resolution / frame count its own cache folder (the VRAM lab uses `cache_dir_lab_<W>x<H>[_f<frames>]`).

### Why this exists

`--skip_existing` only tested that the file existed, so after a resolution change the old latents were reused silently: the V35 preset lab's 768 px and 512 px candidates trained on 1328 px / 1024 px latents, and Kandinsky 6 runs at 512x288 also trained on 864x480 clips left in the same folder. Users hit the same trap by applying a preset with another resolution to a dataset cached earlier. Tests: `musubi-tuner/tests/test_cache_latents_bucket_check.py`, `musubi-tuner/tests/test_video_stale_cache_filter.py`.

---

## Guardrail 14: Block-swap maxima follow the backend's offloader mode

### Rule

- A tab's `blocks_to_swap` limit must be the one the backend asserts for the selected mode, not one number per model. Ideogram 4 keeps 2 of its 34 blocks resident when gradients flow through swapped blocks (LoRA with standard swap, full fine-tuning) and 1 with H2D-only streaming, so 33 needs `block_swap_h2d_only`; `modern_image_lora_gui` validates this. FLUX.2 maps `blocks_to_swap` to a double/single split (`Flux2.split_blocks_to_swap`): the maxima are FLUX.2 dev 30, Klein 9B 18, Klein 4B 13.
- LTX 2.3 LoRA with the standard block swap must not swap exactly 1 block: that swap pair races the full-block unload hook and corrupts the weights (the tab rejects it, the lab job skips it with `skip_swaps`). Do not relax that check without a loss comparison over several epochs.
- When adding a model, read its `enable_block_swap` assertion for every offloader mode (standard, H2D-only, forward-only) and give the VRAM lab job a per-candidate `max_swap` (and `skip_swaps`) where the modes differ.

### Why this exists

The Ideogram 4 tab accepted 33 for every LoRA run, so 33 without H2D-only streaming failed at start-up with the backend's assertion; the V35 preset lab hit it on every activation-offload candidate and lost the 12 GB and lower Ideogram tiers until the job capped those candidates at 32. FLUX.2's proportional split stopped Klein 9B at 16 (6 double + 18 single blocks) although 6 + 22 can be swapped, leaving about 1.7 GB of BF16 weights on the GPU that low tiers could not shed. With one swapped block, LTX 2.3 LoRA trained normally for four steps and then sat at a loss of ~1.8 instead of ~0.3 (one run stopped with a CUDA device-side assert); 2, 4 and 10-26 swapped blocks reproduced the reference losses exactly. Tests: `tests/test_full_finetune_gui.py::test_ideogram_lora_swaps_its_33rd_block_only_with_h2d_streaming`, `tests/test_ltx2_gui.py::TestValidation::test_lora_rejects_exactly_one_swapped_block`, `musubi-tuner/tests/test_flux2_block_swap_split.py`.

---

## Guardrail 15: Runtime TOML values reach the backend unconverted

### Rule

- `read_config_from_file` builds the argparse namespace straight from the TOML, so no argparse `type=` runs on it. Write every value in the type the backend uses: `nargs="*"` options as lists (`base_weights`, `base_weights_multiplier`; `SaveConfigFileToRun` does this), float options keep their fractions (`network_alpha`: use `common_gui.number_keeping_fraction`, never an int-coercion list), and `lr_warmup_steps` / `lr_decay_steps` through `common_gui.steps_or_ratio` (the schedulers treat any float as a fraction of the total steps, so `100.0` must be written as `100`).
- Write TOML only with `toml.dump` / `toml.dumps` inside the `musubi_tuner_gui` package, whose `__init__.py` installs `toml_compat` (a correct string writer, and reading of files the old writer broke). Never format TOML text by hand. A script outside the package that writes TOML must call `musubi_tuner_gui.toml_compat.install()` first.

### Why this exists

A single base-weights path reached the backend as a string, which it read character by character. LTX 2.3 and MiniMax H3 put `network_alpha` and the warmup / decay steps in their int lists, so alpha 0.5 became 0 (which the LoRA module reads as alpha = rank) and a 10% warmup became none. toml 0.10.2 writes strings through `repr` and then rewrites `\x`, which un-escapes the backslashes of any string that contains a backslash followed by `x`. Any Windows path with a folder or file name starting with "x" (`C:\Users\xavier\...`) was saved as invalid TOML: presets no longer loaded, and the trainer could not read its runtime or dataset TOML. Tests: `tests/test_runtime_toml_types.py`, `tests/test_toml_compat.py` (checked with the strict stdlib `tomllib` as well), `tests/test_ltx2_gui.py::TestValidation::test_fractional_alpha_and_warmup_ratio_survive`, `tests/test_minimax_h3_gui.py::test_runtime_toml_keeps_fractional_alpha_and_warmup_ratio`.

---

## Guardrail 16: kohya sd-scripts (FLUX.1 / SDXL backend) keeps its transformers 5 shims

### Rule

- `sd-scripts/` (kohya premium fork) must build every `CLIPTextModel` through `library/transformers_compat.clip_text_model_from_config` (FLUX CLIP-L in `flux_utils.load_clip_l`, SDXL text encoder 1 in `sdxl_model_util`, SD1/2 in `model_util`), keep the `CLIPFeatureExtractor` import fallback in the two lpw pipelines, and keep `strategy_base._ensure_sentencepiece_proto_compat()`. Re-apply them after pulling a newer sd-scripts (local branch `transformers5-compat`).

### Why this exists

transformers 5 flattened `CLIPTextModel` (no `text_model` sub-module, keys without `text_model.`): `load_state_dict(strict=False, assign=True)` then loads nothing and FLUX CLIP-L stays on the meta device, SDXL's strict loader raises, `text_encoder1.text_model...` in `sdxl_train.py` fails and LoRA / SDXL checkpoint key names change. `google/t5-v1_1-xxl` only ships `spiece.model`; transformers 5 converts it with sentencepiece's pb2, which needs protobuf >= 3.20 while the trainer pins 3.19 (`descript-audiotools`), so every FLUX.1 T5 tokenizer load fails. The same protobuf limitation affects musubi-tuner's own `T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")` (FLUX.1 Kontext).

### Regression check

`tests/test_kohya_backend.py::test_runtime_tomls_pass_sd_scripts_setup_on_cpu` runs the real sd-scripts setup (tokenizers included) for FLUX.1 / SDXL LoRA and fine-tuning on the CPU. After an sd-scripts or transformers bump, also compare CLIP-L / T5 tokens and CLIP-L outputs with an environment running the previous version (the port check matched transformers 4.54.1 exactly).

---

## Guardrail 17: A setting that changes another setting needs a unified link

### Rule

- When a trainer tab sets one setting's value from another in a Gradio event (`.change` / `.input` / `.blur` writing to a different component, or rounding the field itself), add the same rule as a `Link` in `musubi_tuner_gui/unified/links.py`. The unified UI reuses the tab's load, save and train functions but never its events.
- A rule whose trigger the unified registry fixes per model (`ModelEntry.fixed`, e.g. Mage-Flow Edit's `is_edit`) or sets by default (`default_overrides`, e.g. Wan 2.2's `task`) must run at load time: mark the link `on_load`.
- A model-file switch between variants must never leave the other variant's checkpoint selected: clear the path when the target file is missing.

### Why this exists

The unified UI shipped without any of these couplings. Choosing HiDream-O1 "dev" kept the full checkpoint and the full model's noise schedule (the trainer accepts either file, so nothing failed); Mage-Flow Edit without a preset trained the text-to-image checkpoint; turning Wan's FP8 Base on left FP8 Scaled off (a plain FP8 cast), and turning it off after a Wan 2.2 preset hit the backend's `fp8_scaled requires fp8_base` assertion through a hidden setting; Wan 2.2 lost its resident-blocks-only compile default. Tests: `tests/test_unified_links.py` (every link's keys exist in its trainer, load-time and edit-time behaviour).

## Guardrail 18: DoRA files stay input-axis, and conversions keep each module's scale with it

### Rule

- DoRA (`network_args` `use_dora=True`; the unified UI's Adapter Type) is the input-axis form: `<module>.dora_scale` of shape `(1, in)` (`(1, in, 1, ...)` for convolutions), saved in fp32. Musubi's `DoRAInputModule` (`networks/lora.py`) and sd-scripts' `networks/dora.py` share the math in `musubi_tuner/networks/lora_shared.py`.
- A converter or key map may rename a DoRA module but never fuse modules that carry their own `dora_scale` (Z-Image `to_q` / `to_k` / `to_v`, Qwen-Image 2.1 `gate_layer` / `proj`): each magnitude belongs to its own module's column norms. ComfyUI decomposes a sliced target (fused qkv / gate_up loaded with an offset) on that slice alone, which matches separate modules exactly; a fused LoRA with one scale does not.
- A new base-weight format (quantization, packing, rotation) must be readable by `DoRABaseWeight` in the original basis; its one-time check against the layer's own output must pass (`tests/test_dora_input_axis.py`).
- One DoRA network trains magnitudes for one base model. Training that shares a network between several base models (Wan 2.2 high + low noise experts, swapped in with `load_state_dict(assign=True)`) must refuse DoRA, in the trainer and in the UI check; such presets stay LoRA. A replaced base weight re-reads its cached column norms (`DoRAInputModule._dora_base_norm_sq`).
- A DoRA (or LoRA) forward sets no attributes while `torch.compile` traces it. Dynamo refuses side effects inside an activation-checkpointed region and then runs that block eagerly for the rest of the run, without an error: Z-Image DoRA trained at 2.27 s/step instead of 1.46. Per-call values go through arguments (`_add_lora_output(x, org_forwarded, down_factor)`), caches are stored only when `torch.compiler.is_compiling()` is false, and one-time work runs before the compiled blocks (`run_dora_base_checks`, a forward pre-hook of the model; one layer per storage format checks the others).
- Base weights merged before the network is created (`--base_weights`, the Z-Image Turbo training adapter) are part of the base whose column norms DoRA learns, so a training-only adapter leaves its share in the saved magnitudes. Check its size before adding such an adapter: the Turbo adapter changes the column norms by 0.01% on average (fused q / k / v 0.001%, at most 2.7% in a few adaLN columns).

### Why this exists

ComfyUI picks the DoRA axis from the `dora_scale` shape, and its output-axis branch divides by the norm of the base weight, not of the merged weight: PEFT-style magnitudes (`lora_magnitude_vector`, the old musubi default) only load approximately and needed base weights to convert. The input-axis branch is exact, as in Forge and A1111. Z-Image's converter fused q, k and v into one LoRA for ComfyUI, which would have mixed three separately trained magnitudes into one. ConvRot INT8 stores weights in a rotated basis; reading them unrotated gives wrong column norms with no error.

### Regression check

`musubi-tuner/tests/test_dora_input_axis.py` (dense definition, ComfyUI's own `weight_decompose`, fp8 / ConvRot bases, block swap, compiled checkpointed blocks with `fullgraph=True`, save and load, the generation scripts' merge) and `tests/test_kohya_dora.py`; `dont_upload/vram_lab/comfy_dora_check.py` loads a trained file with ComfyUI's code and lists every key it would skip.
