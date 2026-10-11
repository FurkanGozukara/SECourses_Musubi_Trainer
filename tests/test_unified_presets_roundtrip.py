"""Presets save and load perfectly: change every setting, save, load, and every field shows the same value.

Covers every model and training type of the unified UI (musubi trainers through their classic tabs'
own save / load actions, SDXL and FLUX.1 through the kohya backend), and every shipped preset
(load -> save as a user preset -> load again must not change anything). CPU only.
"""

from __future__ import annotations

import math
import os
import random
import sys

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
from musubi_tuner_gui.unified import presets as P  # noqa: E402
from musubi_tuner_gui.unified.backends import get_backend, install_quiet_toasts  # noqa: E402
from musubi_tuner_gui.unified.registry import CATEGORIES, ENTRIES  # noqa: E402
from musubi_tuner_gui.unified.schema import HIDDEN_KEYS, arg_type  # noqa: E402

# Settings the trainers validate together (attention backend, base quantization, precision, block swap
# with gradient checkpointing / multi-GPU, Gemma source): the classic save rightly refuses mixed values,
# so the round trip keeps their defaults. Everything else is changed.
COUPLED = {
    "sdpa", "flash_attn", "flash3", "sage_attn", "xformers", "split_attn",
    "gemma_root", "gemma_safetensors", "fp8_base", "fp8_scaled", "fp8_w8a8", "fp8_llm", "fp8_vl",
    "int8_convrot_dynamic", "int8_convrot_base", "nf4_base", "convrot_int8", "convrot_int8_bwd",
    "full_bf16", "full_fp16", "mixed_precision", "gradient_checkpointing", "multi_gpu", "num_processes",
    "blocks_to_swap", "block_swap_h2d_only", "block_swap_optimizer_patch_params", "ltx2_block_swap_trainable_ring",
    "fused_backward_pass", "turbo_dit", "turbo_dit_cache", "unconditional_dit", "dataset_config_mode",
    "gradient_checkpointing_cpu_offload", "activation_cpu_offload", "img_in_txt_in_offloading",
    "optimizer_type", "use_unconditional_dit_for_lora_sampling", "compile_fullgraph", "dino_loss_weight",
    "dino_loss_every_n_steps", "gemma_load_in_8bit", "gemma_load_in_4bit", "h3_teacher_matching",
    "h3_guidance_loss_scale", "h3_guidance_loss_scale_audio", "task", "gradient_accumulation_steps",
    "use_control_images", "h3_timestep_focus_min", "h3_timestep_focus_max", "h3_timestep_focus_prob",
    "min_timestep", "max_timestep", "caching_teo_device", "caching_latent_device",
    "debug_mode",  # "Enable Logging (WandB)" also sets Log With on save (Guardrail 1)
    "warn_on_caption_issues",  # Ideogram 4: only while captions are validated (the save turns it off)
}
# Made for one run's dataset folder and blanked on every save (a reused one trained the wrong images).
NOT_SAVED = {"generated_toml_path"}
LIST_TEXT = {"network_args": "conv_dim=8 conv_alpha=4", "optimizer_args": "weight_decay=0.02 betas=0.9,0.99", "lr_scheduler_args": "min_lr=1e-7"}
TEXT = "first line\nsecond line with 'quotes' and ünicode ğüşöçİ"
CASES = []
for _entry in ENTRIES:
    for _mode in _entry.modes:
        if not any(case[0].backend == _entry.backend and case[1] == _mode for case in CASES):
            CASES.append((_entry, _mode))


@pytest.fixture(scope="module", autouse=True)
def _private_state(tmp_path_factory):
    install_quiet_toasts()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        patch.chdir(tmp_path_factory.mktemp("cwd"))
        yield


@pytest.fixture(scope="module")
def workspaces():
    from musubi_tuner_gui.unified.workspace import Workspace

    return {category: Workspace(category, headless=True) for category in CATEGORIES}


def _shown(workspace, backend, values: dict) -> dict:
    """What every field of the workspace shows for these values (the UI's own conversion)."""
    shown = {}
    for spec in backend.specs:
        uid = workspace.backend_uid[backend.name].get(spec.key)
        item = workspace.fields.get(uid) if uid else None
        if item is not None:
            shown[spec.key] = workspace._ui_value(item, spec, values.get(spec.key, spec.default))
    return shown


def _same_for_trainer(backend, key, a, b) -> bool:
    """Equal for the trainer although shown differently: the classic multi-path syntax re-quotes paths
    with spaces (``"C:/my loras/x.safetensors"``) when it shows several base weights in one field."""
    if (arg_type(backend.name, key) or {}).get("nargs") in (None, "?"):
        return False
    convert = getattr(backend, "coerce", backend.trainer_value)
    return convert(key, a) == convert(key, b)


def _same_path(a, b) -> bool:
    """``G:/models/x.safetensors`` and ``G:\\models\\x.safetensors`` are one file on Windows."""
    if not (isinstance(a, str) and isinstance(b, str) and a and b) or not (":" in a[:3] or "\\" in a or "/" in a):
        return False
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def _mutate(backend, spec, value, tmp_path):
    arg = arg_type(backend.name, spec.key) or {}
    kind = spec.kind
    if spec.key in LIST_TEXT:
        return LIST_TEXT[spec.key]
    if kind == "checkbox":
        return not bool(value)
    if arg.get("type") in ("int", "float") and arg.get("nargs") not in (None, "?") and kind in ("text", "textarea"):
        return "0.5"
    if kind in ("number", "slider") or (arg.get("type") in ("int", "float") and kind in ("text", "dropdown")):
        lo, hi, step = spec.minimum, spec.maximum, spec.step or 1
        base = value if isinstance(value, (int, float)) and not isinstance(value, bool) else (lo if lo is not None else 0)
        integral = arg.get("type") == "int" or (kind == "slider" and float(step).is_integer()) or (
            float(base).is_integer() and float(step).is_integer() and arg.get("type") != "float"
        )
        candidates = [int(base) + int(step), int(base) - int(step)] if integral else [round(float(base) * 1.5 + 0.0125, 6), round(float(base) * 0.5 + 0.0025, 6)]
        if spec.key == "max_token_length":
            candidates = [150, 225]
        for candidate in candidates:
            if (lo is None or candidate >= lo) and (hi is None or candidate <= hi) and candidate != base:
                return candidate
        return value
    if kind in ("dropdown", "radio"):
        choices = spec.choice_values() if spec.choices else []
        if spec.multiselect:
            return choices[:2] if choices else value
        others = [choice for choice in choices if choice != value]
        return others[0] if others else value
    if kind == "file":
        return str(tmp_path / "files" / f"{spec.key} model.safetensors")
    if kind == "folder":
        return str(tmp_path / "folders" / f"{spec.key} folder")
    if kind == "textarea":
        return TEXT
    return f"{value}_rt" if value else f"rt {spec.key}"


@pytest.mark.parametrize("entry, mode", CASES, ids=[f"{entry.key}-{mode}" for entry, mode in CASES])
def test_every_setting_survives_save_and_load(entry, mode, workspaces, tmp_path):
    workspace = workspaces[entry.category]
    backend = get_backend(entry.backend)
    values = backend.defaults(mode)
    values.update(entry.fixed)
    changed = []
    for spec in backend.specs:
        if spec.key in HIDDEN_KEYS or spec.key in entry.fixed or spec.key in entry.locked_keys or spec.key in COUPLED:
            continue
        if mode not in spec.modes or (spec.extras or {}).get("interactive") is False:
            continue
        new = _mutate(backend, spec, values.get(spec.key, spec.default), tmp_path)
        if new != values.get(spec.key):
            values[spec.key] = new
            changed.append(spec.key)
    assert len(changed) > 40, changed
    path = str(tmp_path / "presets" / f"{entry.key} {mode}.toml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    backend.save(path, values)
    P.write_metadata_header(path, entry, mode)
    loaded = backend.load(path, backend.defaults(mode)).values
    before, after = _shown(workspace, backend, values), _shown(workspace, backend, loaded)
    lost = {key: (before[key], after.get(key)) for key in changed if key in before and before[key] != after.get(key) and not _same_for_trainer(backend, key, values.get(key), loaded.get(key))}
    assert not lost, "\n".join(f"{key}: saved {saved!r} -> loaded {got!r}" for key, (saved, got) in lost.items())
    identity = P.identify_preset(path)
    assert (identity.entry_key, identity.mode) == (entry.key, mode)


def _shipped_cases():
    cases = []
    for entry in ENTRIES:
        for mode in entry.modes:
            for option in P.preset_options(entry, mode, "all"):
                if option.shipped and not any(case[2] == option.path for case in cases):
                    cases.append((entry, mode, option.path))
    return cases


SHIPPED = _shipped_cases()


@pytest.mark.parametrize("entry, mode, path", SHIPPED, ids=[f"{entry.key}-{mode}-{os.path.basename(path)}" for entry, mode, path in SHIPPED])
def test_shipped_presets_load_and_resave_without_drift(entry, mode, path, workspaces, tmp_path):
    workspace = workspaces[entry.category]
    backend = get_backend(entry.backend)
    first = backend.load(path, backend.defaults(mode)).values
    first.update(entry.fixed)
    copy = str(tmp_path / "copy.toml")
    backend.save(copy, first)
    second = backend.load(copy, backend.defaults(mode)).values
    second.update(entry.fixed)
    before, after = _shown(workspace, backend, first), _shown(workspace, backend, second)
    drift = {
        key: (before[key], after.get(key))
        for key in before
        if key not in NOT_SAVED
        and before[key] != after.get(key)
        and not _same_for_trainer(backend, key, first.get(key), second.get(key))
        and not ({before[key], after.get(key)} <= {0, None})  # classic optional numbers: 0 and blank both mean "auto"
        and not _same_path(before[key], after.get(key))
    }
    assert not drift, "\n".join(f"{key}: {a!r} -> {b!r}" for key, (a, b) in drift.items())


def test_a_refused_save_never_looks_saved(tmp_path):
    """The classic save refuses invalid combinations; an older file must not stay looking freshly saved."""
    backend = get_backend("krea2")
    values = backend.defaults("full")
    path = str(tmp_path / "preset.toml")
    backend.save(path, values)
    stamp = os.path.getmtime(path)
    values.update(full_bf16=True, mixed_precision="fp16")  # refused by the classic save
    with pytest.raises(RuntimeError, match="(?i)bf16"):
        backend.save(path, values)
    assert os.path.getmtime(path) == stamp and not os.path.exists(str(tmp_path / "preset.saving.toml"))


# --------------------------------------------------------------------------------------------- messy input
FUZZ_SEEDS = range(1, 1 + int(os.environ.get("UNIFIED_FUZZ_SEEDS", "3")))  # more for a deeper run


def _whole(backend, spec, arg: dict) -> bool:
    """A whole-number setting (the backend's own rule; kohya: a whole step from a whole minimum)."""
    if backend.whole_number(spec.key) or arg.get("type") == "int":
        return True
    if backend.engine != "kohya" or spec.step is None:
        return False
    return float(spec.step).is_integer() and (spec.minimum is None or float(spec.minimum).is_integer())


def _typed_number(rng, number, whole: bool) -> str:
    """A number written the way people type it: padding, a plus sign, thousands separators, decimal commas."""
    if whole:
        number = int(number)
        styles = [str(number), f"  {number} ", f"+{number}" if number >= 0 else str(number)]
        if abs(number) >= 1000:
            styles.append(f"{number:,}")
        return rng.choice(styles)
    text = repr(float(number))
    styles = [text, f" {text} ", f"{float(number):.6e}"]
    if "e" not in text:
        styles.append(text.replace(".", ","))
    return rng.choice(styles)


def _messy(backend, spec, rng, root: str):
    """A random valid value for ``spec``, written the way people type or paste it."""
    from urllib.parse import quote

    from musubi_tuner_gui.unified.schema import PATH_KINDS, PATH_TEXT_KEYS

    arg = {} if backend.engine == "kohya" else (arg_type(backend.name, spec.key) or {})  # kohya fields hold one value
    many = arg.get("nargs") not in (None, "?") and spec.kind not in ("number", "slider")
    tag = rng.randrange(1000)
    if spec.key in LIST_TEXT:
        return rng.choice([LIST_TEXT[spec.key], f" {LIST_TEXT[spec.key]}, ", LIST_TEXT[spec.key].replace(" ", ", ")])
    if spec.kind == "checkbox":
        value = rng.random() < 0.5
        return rng.choice([value, "yes" if value else "no", "True" if value else "false", " 1 " if value else "0"])
    if spec.key in PATH_TEXT_KEYS:
        return rng.choice([f"'folder {tag}'", f' "folder_{tag}" ', f"folder_{tag}"])
    if spec.kind in ("file", "folder") or spec.key in PATH_KINDS:
        if many:
            return f'"{os.path.join(root, f"my loras {tag}", "a.safetensors")}" {os.path.join(root, f"b{tag}.safetensors")}'
        name = f"fuzz {tag}" + ("" if spec.kind == "folder" else ".safetensors")
        native = os.path.join(root, f"dir {tag}", name)
        style = rng.randrange(6)
        if style == 0:
            return f'"{native}"'
        if style == 1:
            return native.replace("\\", "/") + ("/" if spec.kind == "folder" else "")
        if style == 2:
            return "file:///" + quote(native.replace("\\", "/").lstrip("/"))
        if style == 3:
            return f"  {native}  "
        if style == 4:
            return f"\u201c{native}\u201d"
        return f"./rel {tag}/../rel/{name}"
    if spec.kind in ("dropdown", "radio") and spec.choices:
        choices = spec.choice_values()
        if spec.multiselect:
            return rng.sample(choices, k=min(len(choices), rng.randint(1, 2)))
        return rng.choice(choices)
    if spec.kind in ("number", "slider") or arg.get("type") in ("int", "float"):
        if many:
            return rng.choice(["0,5 1", "1; 0.25", "0.75, 1.5"])
        whole = _whole(backend, spec, arg)
        if spec.key.endswith("frames"):  # frame counts are 4n + 1 (Wan, Kandinsky 6); MiniMax H3 needs 5 s at 24 fps
            low = max(31, int(((spec.minimum or 1) - 1) // 4) + 1)
            high = max(low, int(((spec.maximum or 241) - 1) // 4))
            return _typed_number(rng, 4 * rng.randint(low, high) + 1, True)
        lo = spec.minimum if spec.minimum is not None else 0
        hi = spec.maximum if spec.maximum is not None else lo + (100000 if whole else 10)
        number = rng.randint(int(-(-lo // 1)), int(hi // 1)) if whole else float(f"{rng.uniform(lo, hi):.6g}")
        return _typed_number(rng, number, whole)
    if spec.kind == "textarea":
        return rng.choice([TEXT, f"line {tag}\n  indented 'q' \"dq\" ğüş\n"])
    return rng.choice([f"text {tag}", f"  padded {tag}  ", f"ünicode ğüşİ {tag}"])


def _same_value(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return a == b or _same_path(a, b)


@pytest.mark.parametrize("seed", FUZZ_SEEDS)
@pytest.mark.parametrize("entry, mode", CASES, ids=[f"{entry.key}-{mode}" for entry, mode in CASES])
def test_messy_input_survives_save_load_and_the_ui(entry, mode, seed, workspaces, tmp_path):
    """Random valid values typed the messy way: the trainer value survives the save and load, and the text the
    UI then shows converts back to that same trainer value."""
    rng = random.Random(f"{entry.key}-{mode}-{seed}")
    workspace = workspaces[entry.category]
    backend = get_backend(entry.backend)
    convert = getattr(backend, "coerce", backend.trainer_value)
    specs = backend.spec_map()
    values = backend.defaults(mode)
    values.update(entry.fixed)
    typed = {}
    for spec in backend.specs:
        if spec.key in HIDDEN_KEYS or spec.key in entry.fixed or spec.key in entry.locked_keys or spec.key in COUPLED:
            continue
        if mode not in spec.modes or (spec.extras or {}).get("interactive") is False:
            continue
        typed[spec.key] = _messy(backend, spec, rng, str(tmp_path / "fuzz"))
    values.update(typed)
    expected = {key: convert(key, value) for key, value in typed.items()}
    path = str(tmp_path / "messy.toml")
    backend.save(path, values)
    loaded = backend.load(path, backend.defaults(mode)).values
    problems = []
    for key, want in expected.items():
        got = convert(key, loaded.get(key))
        if not _same_value(want, got):
            problems.append(f"{key}: typed {typed[key]!r} -> {want!r}, after save + load {got!r}")
            continue
        item = workspace.fields.get(workspace.backend_uid[backend.name].get(key, ""))
        if item is not None:
            shown = workspace._ui_value(item, specs[key], loaded.get(key))
            back = convert(key, shown)
            if not _same_value(want, back):
                problems.append(f"{key}: typed {typed[key]!r} -> {want!r}; the UI shows {shown!r}, which gives {back!r}")
    assert not problems, "\n".join(problems)
