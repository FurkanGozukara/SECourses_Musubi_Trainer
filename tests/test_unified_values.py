"""What users type -> what the trainers receive (musubi_tuner_gui/unified/values.py, backends.py).

The trainers read the runtime TOML without argparse conversion, so every value must already have
the type the trainer's own argparse option declares. CPU only; nothing is trained.
"""

from __future__ import annotations

import os
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from musubi_tuner_gui.unified.backends import enforce_trainer_types, get_backend  # noqa: E402
from musubi_tuner_gui.unified.schema import PATH_KINDS, arg_type, load_schema  # noqa: E402
from musubi_tuner_gui.unified.values import (  # noqa: E402
    ValueProblem,
    clean_path,
    clean_paths,
    parse_bool,
    parse_number,
    parse_numbers,
    strip_quotes,
    to_trainer_value,
)

N = os.path.normpath
BACKENDS = sorted(load_schema()["arg_types"])


# --------------------------------------------------------------------------------------------- recorded trainer types
def test_trainer_types_are_recorded_for_every_backend():
    sys.path.insert(0, os.path.join(APP_DIR, "tools"))
    from trainer_arg_types import TRAINING_SCRIPTS

    assert set(BACKENDS) == set(TRAINING_SCRIPTS)
    for name in BACKENDS:
        types = load_schema()["arg_types"][name]
        assert types["learning_rate"]["type"] == "float" and types["output_dir"]["type"] in ("str", "custom:Path"), name
        assert types["max_train_epochs"]["type"] == "int", name


def test_recorded_trainer_types_match_the_live_training_scripts():
    """Run after pulling training-script changes into the fork: the stored types must equal the parsers."""
    sys.path.insert(0, os.path.join(APP_DIR, "tools"))
    from trainer_arg_types import collect

    live = collect()
    for name in BACKENDS:
        stored = load_schema()["arg_types"][name]
        stale = [key for key, info in stored.items() if live[name].get(key, {}).get("type") != info["type"] or live[name].get(key, {}).get("nargs") != info.get("nargs")]
        assert not stale, f"{name}: argparse changed for {stale}; run tools/generate_unified_schema.py"


# --------------------------------------------------------------------------------------------- paths
@pytest.mark.parametrize(
    "typed, expected",
    [
        ('"C:\\data\\set"', N("C:/data/set")),
        ("'C:/data/set'", N("C:/data/set")),
        ("  C:/data/set/  ", N("C:/data/set")),
        ("C:\\data\\set\\\n", N("C:/data/set")),
        ("C:\\data\\set\r\n", N("C:/data/set")),
        (' " C:\\my data\\set " ', N("C:/my data/set")),
        ("\u201cC:\\data\u201d", N("C:/data")),  # smart quotes pasted from chat / Word
        ('"C:\\data', N("C:/data")),  # one stray quote
        ("C:\\\\data\\\\set", N("C:/data/set")),  # doubled backslashes copied from JSON
        ("C:/data\\mixed/separators\\", N("C:/data/mixed/separators")),
        ("\\\\server\\share\\data\\", "\\\\server\\share\\data" if os.name == "nt" else N("//server/share/data")),
        ("file:///C:/my%20data/set", N("C:/my data/set")),
        ("..\\Training_Outputs", "../Training_Outputs"),  # relative paths stay portable (forward slashes)
        ("../Training_Outputs_FLUX_1/", "../Training_Outputs_FLUX_1"),
        (".\\outputs\\my lora", "outputs/my lora"),
        ("/home/Ubuntu/Downloads/data", "/home/Ubuntu/Downloads/data" if os.name == "nt" else N("/home/Ubuntu/Downloads/data")),
        ("'/workspace/models//dit.safetensors'", "/workspace/models/dit.safetensors"),
        ("C:\\Users\\Furkan\\Masaüstü\\veri İşçi", N("C:/Users/Furkan/Masaüstü/veri İşçi")),
        ("Qwen/Qwen-Image", "Qwen/Qwen-Image"),  # Hugging Face repo id stays as typed
        ("stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-xl-base-1.0"),
        ("https://huggingface.co/org/model", "https://huggingface.co/org/model"),
        ("model.safetensors", "model.safetensors"),
        ("", ""),
        ("   ", ""),
        ('""', ""),
        (None, ""),
    ],
)
def test_clean_path(typed, expected):
    assert clean_path(typed) == expected


def test_clean_path_expands_variables_and_home(monkeypatch):
    monkeypatch.setenv("UNIFIED_TEST_ROOT", N("D:/datasets"))
    assert clean_path("%UNIFIED_TEST_ROOT%\\faces" if os.name == "nt" else "$UNIFIED_TEST_ROOT/faces") == N("D:/datasets/faces")
    assert clean_path("~/faces") == N(os.path.join(os.path.expanduser("~"), "faces"))


def test_several_paths_split_like_the_classic_tabs():
    """One per line or ';', or several on one line (the fields' own "space-separated" syntax, quotes around
    paths with spaces). An unquoted path with spaces stays whole unless another absolute path starts."""
    assert clean_paths('"C:/my loras/a b.safetensors"') == [N("C:/my loras/a b.safetensors")]
    assert clean_paths("C:/my loras/a b.safetensors") == [N("C:/my loras/a b.safetensors")]
    assert clean_paths("C:/a.safetensors\nC:/b c.safetensors") == [N("C:/a.safetensors"), N("C:/b c.safetensors")]
    assert clean_paths("C:/a.safetensors; 'C:/b.safetensors'") == [N("C:/a.safetensors"), N("C:/b.safetensors")]
    assert clean_paths("C:/l/a.safetensors C:/l/b.safetensors") == [N("C:/l/a.safetensors"), N("C:/l/b.safetensors")]
    assert clean_paths('"C:/my loras/a.safetensors" C:/l/b.safetensors') == [N("C:/my loras/a.safetensors"), N("C:/l/b.safetensors")]
    assert clean_paths("a.safetensors b.safetensors") == ["a.safetensors", "b.safetensors"]
    assert clean_paths("/data/a.safetensors /data/b.safetensors") == ["/data/a.safetensors", "/data/b.safetensors"]
    assert clean_paths(["  C:/a/ ", ""]) == [N("C:/a")]
    assert clean_paths(None) == [] and clean_paths("") == []


def test_existing_path_with_spaces_stays_whole(tmp_path):
    weights = tmp_path / "my loras" / "base model.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"")
    assert clean_paths(str(weights)) == [str(weights)]


def test_text_of_several_lines_is_never_turned_into_one_path():
    prompts = "photo of ohwx man --w 1024 --d 42\n\nclose-up of ohwx man, 3/4 view --d 7"
    assert clean_path(prompts) == prompts
    assert clean_path(prompts.replace("\n", "\r\n") + "\r\n") == prompts


def test_multi_path_settings_show_and_read_back_the_same_paths():
    """base_weights is a list for the trainer and one text field in the UI: what the field shows must read back."""
    from musubi_tuner_gui.unified.workspace import FieldDef, Workspace

    backend = get_backend("krea2")
    spec = backend.spec_map()["base_weights"]
    shown = Workspace._ui_value(None, FieldDef("base_weights", "base_weights", "str", "file", "model", 0, 0), spec, [N("C:/my loras/a.safetensors"), N("C:/l/b.safetensors")])
    assert backend.coerce("base_weights", shown) == [N("C:/my loras/a.safetensors"), N("C:/l/b.safetensors")]


def test_strip_quotes_handles_empty_and_nested():
    assert strip_quotes("") == "" and strip_quotes("''") == "" and strip_quotes(" \"'x'\" ") == "x"


# --------------------------------------------------------------------------------------------- numbers
@pytest.mark.parametrize(
    "typed, integer, expected",
    [
        (" 1e-4 ", False, 1e-4),
        ("1E-04", False, 1e-4),
        (".0001", False, 1e-4),
        ("0,0001", False, 1e-4),  # decimal comma
        ("1,5", False, 1.5),
        ("1,004", False, 1.004),  # decimal setting: decimal comma, not a thousands separator
        ("1,000.5", False, 1000.5),
        ("1.000,5", False, 1000.5),
        ("+3", False, 3.0),
        ("\u22122.5", False, -2.5),  # unicode minus
        ("1_000", True, 1000),
        ("10,000", True, 10000),  # thousands separator
        ("1,000,000", True, 1000000),
        ("1 000", True, 1000),
        ("2.0", True, 2),
        (2.0, True, 2),
        (" 4 ", True, 4),
        ("'6e-5'", False, 6e-5),
        (6e-05, False, 6e-05),
        (0, False, 0),
        ("", False, None),
        ("  ", True, None),
        (None, True, None),
    ],
)
def test_parse_number(typed, integer, expected):
    result = parse_number(typed, label="X", integer=integer)
    assert result == expected
    if integer and expected is not None:
        assert isinstance(result, int)


@pytest.mark.parametrize("typed, integer", [("abc", False), ("1e-4x", False), ("nan", False), ("inf", False), ("1.5", True), (2.25, True), ("1,5", True)])
def test_parse_number_rejects_with_the_setting_name(typed, integer):
    with pytest.raises(ValueProblem, match="Learning Rate"):
        parse_number(typed, label="Learning Rate", integer=integer)


def test_parse_numbers():
    assert parse_numbers("0.5 1") == [0.5, 1.0] and parse_numbers("0.5, 1;2") == [0.5, 1.0, 2.0]
    assert parse_numbers("0,5") == [0.5] and parse_numbers("0,5 1,25") == [0.5, 1.25]  # decimal commas
    assert parse_numbers("0.5,1") == [0.5, 1.0] and parse_numbers("1, 2") == [1.0, 2.0] and parse_numbers("1,2,3") == [1.0, 2.0, 3.0]
    assert parse_numbers([1, "2"], integer=True) == [1, 2] and parse_numbers("") == [] and parse_numbers(None) == []


# --------------------------------------------------------------------------------------------- booleans
@pytest.mark.parametrize("typed", [True, "true", "TRUE", " yes ", "On", "1", 1, 2.0, "'true'"])
def test_parse_bool_true(typed):
    assert parse_bool(typed) is True


@pytest.mark.parametrize("typed", [False, "false", "No", "off", "0", 0, 0.0, "", None, "none"])
def test_parse_bool_false(typed):
    assert parse_bool(typed) is False


def test_parse_bool_rejects_nonsense():
    with pytest.raises(ValueProblem, match="Gradient Checkpointing"):
        parse_bool("maybe", label="Gradient Checkpointing")


# --------------------------------------------------------------------------------------------- one setting
def test_to_trainer_value_follows_the_argparse_type():
    assert to_trainer_value("2.0", label="x", kind="number", arg={"type": "int"}, is_path=False) == 2
    assert isinstance(to_trainer_value(0, label="x", kind="number", arg={"type": "float"}, is_path=False), float)
    assert to_trainer_value("1", label="x", kind="number", arg={"type": "custom:_int_or_float"}, is_path=False) == 1
    assert to_trainer_value("0.5", label="x", kind="number", arg={"type": "custom:_int_or_float"}, is_path=False) == 0.5
    assert to_trainer_value(1, label="x", kind="text", arg={"type": "float", "nargs": "*"}, is_path=False) == [1.0]
    assert to_trainer_value("0.5 1", label="x", kind="text", arg={"type": "float", "nargs": "*"}, is_path=False) == [0.5, 1.0]
    assert to_trainer_value('"C:/a b/x.safetensors"', label="x", kind="text", arg={"type": "str", "nargs": "*"}, is_path=True) == [N("C:/a b/x.safetensors")]
    assert to_trainer_value("true", label="x", kind="checkbox", arg={"type": "bool"}, is_path=False) is True
    assert to_trainer_value("  my lora  ", label="x", kind="text", arg={"type": "str"}, is_path=False) == "my lora"
    assert to_trainer_value("  keep spacing \n", label="x", kind="textarea", arg={"type": "str"}, is_path=False) == "  keep spacing \n"
    assert to_trainer_value("", label="x", kind="number", arg={"type": "int"}, is_path=False) is None
    assert to_trainer_value(3.0, label="x", kind="number", arg=None, is_path=False) == 3  # GUI-only: JSON-like


def _messy_forms(value, kind: str):
    if isinstance(value, bool):
        return [str(value).upper(), " " + str(value).lower() + " ", int(value)]
    if isinstance(value, (int, float)):
        return [f" {value} ", f"'{value}'", float(value), str(value).replace(".", ",") if isinstance(value, float) and "e" not in str(value) else str(value)]
    return []


@pytest.mark.parametrize("backend_name", [name for name in BACKENDS if not name.startswith("kohya:")])
def test_every_trainer_setting_gets_its_argparse_type_from_messy_input(backend_name):
    """Each number / switch the trainer reads, typed in messy forms, reaches it in the declared type.
    (SDXL / FLUX.1: kohya_backend converts to sd-scripts types itself; test_unified_runtime_toml checks them.)"""
    backend = get_backend(backend_name)
    checked = 0
    for spec in backend.specs:
        arg = arg_type(backend_name, spec.key)
        if arg is None or arg.get("nargs") not in (None, "?"):
            continue
        default = spec.default
        if arg["type"] == "bool":
            default = bool(default) if default is not None else False
        elif arg["type"] in ("int", "float", "custom:_int_or_float", "custom:int_or_float"):
            if not isinstance(default, (int, float)) or isinstance(default, bool):
                continue
        else:
            continue
        for messy in _messy_forms(default, spec.kind):
            value = backend.trainer_value(spec.key, messy)
            if arg["type"] == "bool":
                assert value is default, (spec.key, messy, value)
            elif arg["type"] == "int":
                assert isinstance(value, int) and not isinstance(value, bool) and value == int(default), (spec.key, messy, value)
            elif arg["type"] == "float":
                assert isinstance(value, float) and value == pytest.approx(float(default)), (spec.key, messy, value)
            else:
                assert value == pytest.approx(default), (spec.key, messy, value)
            checked += 1
    assert checked > 50, f"{backend_name}: only {checked} conversions checked"


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_every_path_setting_is_cleaned(backend_name):
    backend = get_backend(backend_name)
    for spec in backend.specs:
        if spec.key not in PATH_KINDS and spec.kind not in ("file", "folder"):
            continue
        cleaned = backend.trainer_value(spec.key, '  "C:/Training Data/folder/" \n')
        expected = N("C:/Training Data/folder")
        assert cleaned in (expected, [expected]), (spec.key, cleaned)


def test_check_lists_every_problem_by_name():
    backend = get_backend("krea2")
    values = backend.defaults("lora")
    values.update(learning_rate="abc", gradient_accumulation_steps=1.5, network_alpha="")  # alpha: the trainer default is not "not set"
    problems = backend.check(values, "lora")
    joined = "\n".join(problems)
    assert "Learning Rate: 'abc' is not a number." in joined
    assert "must be a whole number (got 1.5)" in joined
    assert "is empty. Enter a value" in joined
    assert backend.check(backend.defaults("lora"), "lora") == []


def test_cache_folder_names_are_cleaned_like_paths():
    """Explorer's "Copy as path" quotes would otherwise land inside the dataset TOML's cache_directory."""
    backend = get_backend("krea2")
    assert backend.coerce("dataset_cache_directory", '"C:\\data\\my cache"') == N("C:/data/my cache")
    assert backend.coerce("dataset_cache_directory", " 'cache_dir' ") == "cache_dir"


def test_float_sliders_keep_their_fraction():
    """network_alpha is a float for the trainer even where the classic tab shows a whole-step slider."""
    backend = get_backend("ltx2")
    assert backend.coerce("network_alpha", 0.5) == 0.5
    assert backend.coerce("network_alpha", "2,5") == 2.5


def test_a_paths_field_with_line_breaks_is_reported():
    backend = get_backend("krea2")
    values = backend.defaults("lora")
    values["dit"] = "C:/models/a.safetensors\nC:/models/b.safetensors"
    assert any("one path per field" in problem for problem in backend.check(values, "lora"))


def test_wan_presets_saved_with_zero_load_without_blank_numbers(tmp_path, monkeypatch):
    """The classic Wan loader reads a saved 0 back as None; Start must not then report the setting as empty."""
    from musubi_tuner_gui.unified.backends import install_quiet_toasts

    install_quiet_toasts()
    monkeypatch.chdir(tmp_path)
    backend = get_backend("wan")
    values = backend.defaults("lora")
    path = str(tmp_path / "wan.toml")
    backend.save(path, values)
    loaded = backend.load(path, backend.defaults("lora")).values
    for key in ("vae_chunk_size", "vae_spatial_tile_sample_min_size", "num_timestep_buckets", "blocks_to_swap"):
        assert loaded[key] == 0, key
    assert backend.check(loaded, "lora") == []


def test_a_file_named_like_a_failure_loads_cleanly(tmp_path, monkeypatch):
    from musubi_tuner_gui.unified.backends import install_quiet_toasts

    install_quiet_toasts()
    monkeypatch.chdir(tmp_path)
    backend = get_backend("krea2")
    path = str(tmp_path / "failsafe_settings.toml")
    backend.save(path, backend.defaults("lora"))
    result = backend.load(path, backend.defaults("lora"))
    assert not result.failed and result.warnings == []
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".saving.toml")]


def test_folder_mode_never_reuses_a_generated_dataset_toml(tmp_path):
    """A preset made with the classic tabs carries generated_toml_path; the classic train code would reuse it."""
    from musubi_tuner_gui.unified.backends import FOLDER_DATASET_MODE, _fresh_dataset

    old = tmp_path / "old_dataset.toml"
    old.write_text("[general]\n", encoding="utf-8")
    payload = {"dataset_config_mode": FOLDER_DATASET_MODE, "generated_toml_path": str(old), "dataset_config": str(old)}
    assert _fresh_dataset(dict(payload)) == {"dataset_config_mode": FOLDER_DATASET_MODE, "generated_toml_path": "", "dataset_config": ""}
    toml_mode = dict(payload, dataset_config_mode="Use TOML File")
    assert _fresh_dataset(dict(toml_mode)) == toml_mode


def test_saved_presets_do_not_keep_the_generated_dataset_toml(tmp_path, monkeypatch):
    import toml

    from musubi_tuner_gui.unified.backends import install_quiet_toasts

    install_quiet_toasts()
    monkeypatch.chdir(tmp_path)
    backend = get_backend("krea2")
    values = dict(backend.defaults("lora"), generated_toml_path=str(tmp_path / "dataset.toml"))
    path = str(tmp_path / "preset.toml")
    backend.save(path, values)
    assert not toml.load(path).get("generated_toml_path")


def test_overlapping_output_captures_keep_stdout_intact():
    """Two Print Command clicks at once: each gets its own output and sys.stdout is restored afterwards."""
    import threading

    from musubi_tuner_gui.unified.backends import capture_output

    original = sys.stdout
    entered, release = threading.Barrier(2), threading.Event()
    results = {}

    def work(name):
        with capture_output() as buffer:
            entered.wait(5)
            print(f"from {name}")
            if name == "first":
                release.wait(5)
        results[name] = buffer.getvalue()

    first = threading.Thread(target=work, args=("first",))
    second = threading.Thread(target=work, args=("second",))
    first.start(), second.start()
    second.join(5)
    release.set()
    first.join(5)
    assert results == {"first": "from first\n", "second": "from second\n"}
    assert sys.stdout is original


def test_kohya_refusals_are_listed_before_start():
    """What kohya_backend would refuse after Start (here: the adafactor scheduler without Adafactor) shows in preflight."""
    backend = get_backend("kohya:flux1_dev")
    values = dict(backend.defaults("lora"), lr_scheduler="adafactor", optimizer_type="AdamW")
    assert any("'adafactor' LR Scheduler" in problem for problem in backend.check(values, "lora"))


def test_toml_gate_fixes_classic_tab_quirks():
    """Qwen Image's classic action collapses list options to their first item; the gate restores the list."""
    fixed = dict(enforce_trainer_types("qwen_image", [("base_weights_multiplier", 1.0), ("learning_rate", "1e-4"), ("max_train_steps", "1,600"), ("unknown_gui_key", "x")]))
    assert fixed == {"base_weights_multiplier": [1.0], "learning_rate": 1e-4, "max_train_steps": 1600, "unknown_gui_key": "x"}
    assert isinstance(fixed["max_train_steps"], int)


@pytest.mark.parametrize("backend_name", [name for name in BACKENDS if not name.startswith("kohya:")])
def test_fine_tuning_defaults_follow_the_classic_mode_switch(backend_name):
    from musubi_tuner_gui.unified.registry import ENTRIES

    if not any(entry.backend == backend_name and "full" in entry.modes for entry in ENTRIES):
        pytest.skip("LoRA only")
    backend = get_backend(backend_name)
    full = backend.defaults("full")
    assert backend.check(full, "full") == [], backend_name
    if backend_name == "ltx2":
        from musubi_tuner_gui.ltx2_lora_gui import LTX2_BASE_QUANTIZATION_KEYS, LTX2_FULL_FINE_TUNING_DEFAULTS

        assert full["learning_rate"] == LTX2_FULL_FINE_TUNING_DEFAULTS["learning_rate"]
        assert not any(full.get(key) for key in LTX2_BASE_QUANTIZATION_KEYS)
    if backend_name == "ideogram4":
        assert full["blocks_to_swap"] <= 32


def test_max_timestep_zero_never_reaches_the_trainer():
    """0 would train without noise; the classic tabs read a saved 0 as "not set" (full range)."""
    backend = get_backend("qwen_image")
    assert backend.coerce("max_timestep", 0) is None and backend.coerce("max_timestep", 900) == 900
    values = dict(backend.defaults("lora"), max_timestep=None, max_train_epochs=None)
    assert not any("Max" in problem and "empty" in problem for problem in backend.check(values, "lora"))


def test_wan_timestep_boundary_takes_a_fraction():
    backend = get_backend("wan")
    assert backend.coerce("timestep_boundary", "0,875") == 0.875
    values = dict(backend.defaults("lora"), timestep_boundary=0.875)
    assert not any("timestep" in problem.lower() for problem in backend.check(values, "lora"))


def test_a_dataset_toml_built_at_start_goes_to_the_trainer_as_a_toml():
    from musubi_tuner_gui.unified.backends import FOLDER_DATASET_MODE, _hand_over_dataset

    payload = {"dataset_config_mode": FOLDER_DATASET_MODE, "dataset_config": "C:/x/ltx2_audio_dataset.toml"}
    _hand_over_dataset(payload, ["[OK] Dataset TOML generated: C:/x/ltx2_audio_dataset.toml"])
    assert payload["dataset_config_mode"] == "Use TOML File"
    untouched = {"dataset_config_mode": FOLDER_DATASET_MODE, "dataset_config": ""}
    _hand_over_dataset(untouched, [])  # trainers that build it themselves keep folder mode
    assert untouched["dataset_config_mode"] == FOLDER_DATASET_MODE


def test_relative_base_weights_with_spaces_survive_save_and_load(tmp_path, monkeypatch):
    """Several classic loaders join a path list with plain spaces; the file's own list wins."""
    from musubi_tuner_gui.unified.backends import install_quiet_toasts

    install_quiet_toasts()
    monkeypatch.chdir(tmp_path)
    backend = get_backend("krea2")
    values = dict(backend.defaults("lora"), base_weights='"../my loras/a.safetensors"\n../other/b.safetensors', base_weights_multiplier="0.5 1")
    path = str(tmp_path / "preset.toml")
    backend.save(path, values)
    loaded = backend.load(path, backend.defaults("lora")).values
    assert backend.coerce("base_weights", loaded["base_weights"]) == ["../my loras/a.safetensors", "../other/b.safetensors"]
