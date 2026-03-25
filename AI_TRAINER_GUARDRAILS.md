## AI Trainer Guardrails

Use this checklist whenever adding a new trainer tab or training flow.

### Logging rule

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
