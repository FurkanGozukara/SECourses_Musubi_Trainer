"""toml 0.10.2 wrote a string holding a backslash + "x" (C:\\Users\\xavier\\...) as invalid TOML.

musubi_tuner_gui installs a correct string writer when it is imported, so presets, runtime TOMLs and dataset
TOMLs written by the GUI load in the trainer (checked here with the strict stdlib parser too), and files the
old writer corrupted load again.
"""

import tomllib

import toml

import musubi_tuner_gui  # noqa: F401  (installs the fixed writer)
from musubi_tuner_gui import toml_compat
from musubi_tuner_gui.common_gui import SaveConfigFileToRun

B = "\\"
STRINGS = [
    f"C:{B}a{B}x.safetensors",
    f"C:{B}Users{B}xavier{B}data",
    f"C:{B}temp{B}x1",
    f"D:{B}xl{B}u{B}x{B}Unet",
    f"{B}{B}server{B}xshare{B}model.safetensors",
    'quote " and tab \t newline \n bell \x07 delete \x7f',
    "Unicode: Masaüstü ğış 日本語 \u2028",
    f"plain {B}n is not a newline",
]


def test_every_string_round_trips_through_both_parsers():
    for value in STRINGS:
        text = toml.dumps({"p": value, "l": [value, "x"]})
        assert toml.loads(text) == {"p": value, "l": [value, "x"]}, text
        assert tomllib.loads(text) == {"p": value, "l": [value, "x"]}, text


def test_runtime_toml_with_an_x_folder_is_valid_toml(tmp_path):
    config = tmp_path / "runtime.toml"
    dit = f"C:{B}Users{B}xavier{B}models{B}dit.safetensors"
    SaveConfigFileToRun([("dit", dit), ("output_dir", f"C:{B}x{B}out"), ("base_weights", f"C:{B}x{B}a.safetensors")], str(config))
    data = tomllib.loads(config.read_text(encoding="utf-8"))
    assert data["dit"] == dit and data["output_dir"] == f"C:{B}x{B}out" and data["base_weights"] == [f"C:{B}x{B}a.safetensors"]


def test_files_the_old_writer_corrupted_load_again(tmp_path):
    # exactly what toml 0.10.2 wrote for these values (it un-escapes everything up to the last "\x"),
    # plus a hand-written Windows path
    corrupted = (
        'dit = "C:\\a\\x.safetensors"\n'
        'home = "C:\\Users\\xavier\\\\data"\n'
        'two = "D:\\xl\\u\\x\\\\Unet"\n'
        'hand = "C:\\Users\\me\\data"\n'
        'fine = "C:\\\\b\\\\c"\n'
        'tab = "a\\tb"\n'
    )
    path = tmp_path / "old.toml"
    path.write_text(corrupted, encoding="utf-8")
    expected = {
        "dit": f"C:{B}a{B}x.safetensors",
        "home": f"C:{B}Users{B}xavier{B}data",
        "two": f"D:{B}xl{B}u{B}x{B}Unet",
        "hand": f"C:{B}Users{B}me{B}data",
        "fine": f"C:{B}b{B}c",
        "tab": "a\tb",
    }
    assert toml.loads(corrupted) == expected
    assert toml.load(str(path)) == expected
    with open(path, encoding="utf-8") as handle:
        assert toml.load(handle) == expected


def test_repair_keeps_escaped_quotes_in_the_raw_part():
    # toml 0.10.2 output for: say "hi" at C:\x1\data -- its quotes were escaped even in the raw part
    corrupted = 'p = "say ' + B + '"hi' + B + '" at C:' + B + 'x1' + B + B + 'data"\n'
    assert toml.loads(corrupted) == {"p": f'say "hi" at C:{B}x1{B}data'}
    assert tomllib.loads(toml_compat.repair(corrupted)) == {"p": f'say "hi" at C:{B}x1{B}data'}


def test_repair_leaves_valid_files_alone_and_other_errors_are_raised():
    valid = toml.dumps({"a": f"C:{B}b{B}c", "u": "é\t"})
    assert toml_compat.repair(valid) == valid
    try:
        toml.loads('a = "C:\\x\nb = 1\n')
    except toml.TomlDecodeError:
        pass
    else:
        raise AssertionError("an unrelated syntax error must still raise")
