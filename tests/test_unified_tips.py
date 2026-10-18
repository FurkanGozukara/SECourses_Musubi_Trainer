"""Model tips of the unified UI (hints.py, model_tips.py, kohya_backend.KOHYA_HINTS).

Every tip list renders completely, every block-swap number a tip quotes is one the trainer accepts
(computed from the trainer / classic-tab code), a training type without shipped presets quotes no
measured numbers (and says what loads instead), and measured VRAM figures come from the presets.
CPU only; no model is loaded.
"""

from __future__ import annotations

import ast
import functools
import html
import os
import re
import sys
import types

import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401  (same import order as the app)

from musubi_tuner_gui import preset_library  # noqa: E402
from musubi_tuner_gui.unified import presets as P  # noqa: E402
from musubi_tuner_gui.unified.backends import get_backend  # noqa: E402
from musubi_tuner_gui.unified.hints import MODEL_HINTS  # noqa: E402
from musubi_tuner_gui.unified.model_tips import tips_for  # noqa: E402
from musubi_tuner_gui.unified.param_spec import FULL, LORA  # noqa: E402
from musubi_tuner_gui.unified.registry import CATEGORIES, ENTRIES  # noqa: E402

CASES = [pytest.param(entry, mode, id=f"{entry.key}-{mode}") for entry in ENTRIES for mode in entry.modes]
MUSUBI_ENTRIES = [entry for entry in ENTRIES if not entry.is_kohya]


@pytest.fixture(scope="module", autouse=True)
def _private_state(tmp_path_factory):
    """Keep the tests away from the real user_presets folder and its ui_state.json."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MUSUBI_USER_PRESETS_DIR", str(tmp_path_factory.mktemp("user_presets")))
        yield


@pytest.fixture(scope="module")
def workspaces():
    from musubi_tuner_gui.unified.workspace import Workspace

    return {category: Workspace(category, headless=True) for category in CATEGORIES}


def _kohya_hints() -> dict:
    from musubi_tuner_gui.unified.kohya_backend import KOHYA_HINTS

    return KOHYA_HINTS


def _expected_tips(entry, mode: str) -> list[str]:
    if entry.is_kohya:
        return list(_kohya_hints()[entry.kohya_model][mode])
    data = MODEL_HINTS[entry.key]
    return list(data["all"]) + list(data[mode])


def _texts(entry, mode: str) -> list[tuple[str, str]]:
    """(scope, text) of every line the card shows for ``mode``; scope is ``mode`` for lines of that training
    type only, "shared" for lines every training type of the entry shows."""
    if entry.is_kohya:
        return [(mode, tip) for tip in _kohya_hints()[entry.kohya_model][mode]]
    data = MODEL_HINTS[entry.key]
    texts = [("shared", tip) for tip in data["all"]] + [(mode, tip) for tip in data[mode]]
    for key in ("vram", "dataset"):
        value = data.get(key)
        if isinstance(value, dict):
            if value.get(mode):
                texts.append((mode, value[mode]))
        elif value:
            texts.append(("shared", value))
    return texts


def _defaults(entry, mode: str) -> dict:
    """What loads for ``entry`` / ``mode`` without a preset (Workspace.load_values)."""
    values = dict(get_backend(entry.backend).defaults(mode))
    values.update(entry.default_overrides)
    values.update(entry.fixed)
    return values


# --------------------------------------------------------------------------------------------- complete lists
def test_every_entry_and_training_type_has_tips():
    assert set(MODEL_HINTS) == {entry.key for entry in MUSUBI_ENTRIES}
    for entry in MUSUBI_ENTRIES:
        data = MODEL_HINTS[entry.key]
        assert data.get("about") and data.get("all") and data.get("dataset"), entry.key
        assert {key for key in data if key in (LORA, FULL)} == set(entry.modes), entry.key
        vram = data.get("vram")
        assert vram, entry.key
        if isinstance(vram, dict):
            assert set(vram) == set(entry.modes) and all(vram.values()), entry.key
        for mode in entry.modes:
            assert all(isinstance(tip, str) and tip.strip() for tip in _expected_tips(entry, mode)), (entry.key, mode)


@pytest.mark.parametrize("entry,mode", CASES)
def test_every_tip_list_renders_completely(workspaces, entry, mode):
    expected = _expected_tips(entry, mode)
    data = tips_for(entry, mode)
    assert data["tips"] == expected  # nothing dropped, nothing reordered
    card = workspaces[entry.category].model_card(entry, mode)
    missing = [tip for tip in expected if html.escape(tip) not in card]
    assert not missing, f"{entry.key} {mode}: not rendered: {missing}"
    for line in (data["vram"], data["dataset"]):
        if line:
            assert html.escape(line) in card


# --------------------------------------------------------------------------------------------- block swap limits
# a whole block count: not 14 of "14.2" or "14B", and not a size, length or count ("1024 px", "200 epochs")
NUMBER = r"\d+(?!\.\d)(?!\w)(?!\s*(?:GB|MB|px|epochs?|frames?|steps?|fps)\b)"
SWAP_RUN = re.compile(
    rf"(?:block swap|blocks to swap|\bswaps?|\bswapping)\s+(?:at most\s+|up to\s+)?(?P<run>{NUMBER}(?:(?:\s*,\s*|,?\s+(?:and|or|to)\s+|-){NUMBER})*)",
    re.I,
)
UP_TO_BLOCKS = re.compile(rf"\b(?:up to|at most)\s+(?P<run>{NUMBER})\s+(?:of\s+(?:the\s+)?\d+\s+)?(?:[\w-]+\s+)?blocks?\b", re.I)
MAXIMUM = re.compile(rf"\bmax(?:imum)?\s+(?:block\s+swap\s+|swap\s+)?(?P<run>{NUMBER})", re.I)


def _swap_numbers(text: str) -> list[int]:
    """Every block count a text quotes after "block swap", "swaps", "up to / at most N blocks" or "max"."""
    numbers = set()
    for pattern in (SWAP_RUN, UP_TO_BLOCKS, MAXIMUM):
        for match in pattern.finditer(text):
            numbers.update(int(number) for number in re.findall(r"\d+", match.group("run")))
    return sorted(numbers)


def test_swap_number_extraction():
    assert _swap_numbers("16, 12 and 10 GB tiers swap 18, 28 and 33 blocks (H2D-only); max swap 34.") == [18, 28, 33, 34]
    assert _swap_numbers("22.3 GB with block swap 3 and 14.2 GB with block swap 23") == [3, 23]
    assert _swap_numbers("up to 47 of the 48 blocks can be swapped; max block swap 30 of 32") == [30, 47]
    assert _swap_numbers("Lite swaps at most 30 blocks; 1.3B tasks swap at most 28 blocks, 14B tasks 38") == [28, 30]
    assert _swap_numbers("Max Gradient Norm 0, 768x512 at 33 frames, block swap 4, 14.3 GB") == [4]
    assert _swap_numbers("INT8 ConvRot base, block swap 0, 200 epochs; 10-12 GB GPUs use block swap 31-35") == [0, 31, 35]
    assert _claimed_maxima("(FP8 Scaled, block swap 29, the dev maximum); 16 is the Klein 9B block swap maximum") == [16, 29]
    assert _claimed_maxima("12, 10 and 8 GB swap 7, 9 and 10 blocks (the maximum); block swap 32 (the fine-tuning maximum)") == [10, 32]
    assert _claimed_maxima("Up to 18 blocks can be swapped (the Klein 9B maximum); 20.4 GB with block swap 4, 14.3 GB") == [18]


@functools.lru_cache(maxsize=None)
def _flux2_split_maximum(model_version: str) -> int:
    """Largest blocks_to_swap FLUX.2's own split code (split_blocks_to_swap in flux2_models.py) accepts for this
    model version. The split methods and the block counts are read from the trainer source and run on a stand-in
    object, which follows the code without importing the FLUX.2 model stack; the real module is the fallback."""
    folder = os.path.join(APP_DIR, "musubi-tuner", "src", "musubi_tuner", "flux_2")
    try:
        with open(os.path.join(folder, "flux2_models.py"), encoding="utf-8") as handle:
            models = ast.parse(handle.read())
        with open(os.path.join(folder, "flux2_utils.py"), encoding="utf-8") as handle:
            utils = ast.parse(handle.read())
        classes = {node.name: node for node in models.body if isinstance(node, ast.ClassDef)}

        def fields(name: str) -> dict:
            node, values = classes[name], {}
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in classes:
                    values.update(fields(base.id))
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value is not None:
                    try:
                        values[item.target.id] = ast.literal_eval(item.value)
                    except ValueError:
                        pass
            return values

        info = next(
            node.value
            for node in utils.body
            if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "FLUX2_MODEL_INFO" for target in node.targets)
        )
        params = next(
            fields(next(keyword.value for keyword in value.keywords if keyword.arg == "params").func.id)
            for key, value in zip(info.keys, info.values)
            if ast.literal_eval(key) == model_version
        )

        def defines(node, name: str) -> bool:
            return any(isinstance(item, ast.FunctionDef) and item.name == name for item in node.body)

        owner = next(node for node in classes.values() if defines(node, "split_blocks_to_swap"))
        wanted = ("_proportional_block_swap_split", "split_blocks_to_swap")
        methods = [item for item in owner.body if isinstance(item, ast.FunctionDef) and item.name in wanted]
        extra = {"type_params": []} if "type_params" in ast.ClassDef._fields else {}
        module = ast.Module(body=[ast.ClassDef(name="Split", bases=[], keywords=[], body=methods, decorator_list=[], **extra)], type_ignores=[])
        namespace: dict = {}
        exec(compile(ast.fix_missing_locations(module), "flux2_models.py", "exec"), namespace)
        split = namespace["Split"]()
        split.num_double_blocks, split.num_single_blocks = params["depth"], params["depth_single_blocks"]
        split_blocks = split.split_blocks_to_swap
        split_blocks(1)
    except Exception:  # the source changed shape: run the real module (slower import)
        from musubi_tuner.flux_2 import flux2_models
        from musubi_tuner.flux_2.flux2_utils import FLUX2_MODEL_INFO

        model = next(cls for cls in vars(flux2_models).values() if isinstance(cls, type) and "split_blocks_to_swap" in vars(cls))
        found = FLUX2_MODEL_INFO[model_version].params
        stand_in = types.SimpleNamespace(num_double_blocks=found.depth, num_single_blocks=found.depth_single_blocks)
        stand_in._proportional_block_swap_split = functools.partial(model._proportional_block_swap_split, stand_in)
        split_blocks = functools.partial(model.split_blocks_to_swap, stand_in)
    maximum = 0
    while maximum < 1000:
        try:
            split_blocks(maximum + 1)
        except ValueError:
            return maximum
        maximum += 1
    raise AssertionError(f"FLUX.2 {model_version}: the split code accepts any block count")


def _largest_accepted(spec, mode: str, maximum: int) -> int:
    """image_models_gui: the largest block count the tab's own validation accepts with the defaults (Kandinsky 6
    Lite, the default DiT, allows fewer than the Pro maximum of the spec)."""
    values = _defaults(next(entry for entry in ENTRIES if entry.backend == spec.key), mode)
    if not values.get("dit"):
        values["dit"] = spec.model_path("dit").filename  # the checkpoint the tab points at by default
    accepted = None
    for blocks in range(maximum + 1):
        try:
            spec.validate({**values, "blocks_to_swap": blocks}, mode == FULL)
        except ValueError:
            break
        accepted = blocks
    return maximum if accepted is None else accepted


@functools.lru_cache(maxsize=None)
def _swap_maximum(entry_key: str, mode: str) -> int:
    """The real block-swap maximum of ``entry_key`` / ``mode``, from the trainer or classic-tab code."""
    entry = next(item for item in ENTRIES if item.key == entry_key)
    name = entry.backend
    if name == "flux":
        versions = [value for _label, value in entry.choice_overrides.get("model_version", ())]
        versions = versions or [entry.fixed.get("model_version") or entry.default_overrides.get("model_version")]
        return max(_flux2_split_maximum(version) for version in versions)
    if name == "wan":
        from musubi_tuner_gui.wan_lora_gui import wan_max_blocks_to_swap

        tasks = [value for _label, value in entry.choice_overrides["task"]]
        return max(wan_max_blocks_to_swap(task) for task in tasks)  # the tab refuses more: two blocks stay on the GPU
    if name in ("krea2", "ideogram4"):
        from musubi_tuner_gui import modern_image_lora_gui as modern

        architecture = modern.get_architecture(name)
        # Ideogram 4 keeps one block resident only for LoRA with H2D-only streaming (the tab's validation)
        return architecture.max_blocks_to_swap - 1 if architecture.is_ideogram and mode == FULL else architecture.max_blocks_to_swap
    from musubi_tuner_gui import image_models_gui

    if name in image_models_gui.ARCHITECTURES:
        spec = image_models_gui.get_spec(name)
        maximum = spec.max_blocks_to_swap_full if mode == FULL else spec.max_blocks_to_swap
        return _largest_accepted(spec, mode, maximum) if spec.validate is not None else maximum
    if name == "ltx2":
        from musubi_tuner_gui import ltx2_lora_gui

        return ltx2_lora_gui.LTX2_MAX_BLOCKS_TO_SWAP
    if name == "minimax_h3":
        from musubi_tuner_gui import minimax_h3_lora_gui

        return minimax_h3_lora_gui.MINIMAX_H3_MAX_BLOCKS_TO_SWAP
    spec = get_backend(name).spec_map()["blocks_to_swap"]  # Z-Image, Qwen Image, kohya: the tab's own slider limit
    assert spec.maximum is not None, f"{name}: no block-swap maximum known"
    return int(spec.maximum)


def test_flux2_block_swap_maxima_come_from_the_split_code():
    maxima = {version: _flux2_split_maximum(version) for version in ("dev", "klein-base-9b", "klein-base-4b")}
    assert maxima["dev"] > maxima["klein-base-9b"] > maxima["klein-base-4b"] > 0


@pytest.mark.parametrize("entry,mode", CASES)
def test_quoted_block_swap_numbers_are_allowed(entry, mode):
    for scope, text in _texts(entry, mode):
        numbers = _swap_numbers(text)
        if not numbers:
            continue
        modes = entry.modes if scope == "shared" else (mode,)
        maximum = max(_swap_maximum(entry.key, item) for item in modes)
        too_high = [number for number in numbers if number > maximum]
        assert not too_high, f"{entry.key} {mode}: block swap {too_high} above the real maximum {maximum}: {text}"


# "10 blocks (the maximum)", "29, the dev maximum", "16 is the Klein 9B block swap maximum" (plain words only in between)
CLAIMED_MAXIMUM = re.compile(rf"(?P<run>{NUMBER})(?:\s+blocks?)?\s*(?:,\s*|\(\s*|\s+is\s+)(?:the\s+)?(?:[\w.-]+\s+)*?maximum\b", re.I)


def _claimed_maxima(text: str) -> list[int]:
    """Numbers a text states as the limit itself ("up to N blocks", "at most N blocks", "max swap N", "N (the maximum)")."""
    numbers = set()
    for pattern in (UP_TO_BLOCKS, MAXIMUM, CLAIMED_MAXIMUM):
        numbers.update(int(match.group("run")) for match in pattern.finditer(text))
    return sorted(numbers)


@pytest.mark.parametrize("entry,mode", CASES)
def test_quoted_block_swap_maxima_are_the_real_ones(entry, mode):
    """A stated maximum must not understate the limit either (FLUX.2 dev once said 29, Klein 9B 16)."""
    for scope, text in _texts(entry, mode):
        modes = entry.modes if scope == "shared" else (mode,)
        real = {_swap_maximum(entry.key, item) for item in modes}
        if entry.backend == "wan":  # every task has its own maximum (1.3B: 28, 14B / A14B: 38)
            from musubi_tuner_gui.wan_lora_gui import wan_max_blocks_to_swap

            real |= {wan_max_blocks_to_swap(value) for _label, value in entry.choice_overrides["task"]}
        wrong = [number for number in _claimed_maxima(text) if number not in real]
        assert not wrong, f"{entry.key} {mode}: stated maximum {wrong}, the code allows {sorted(real)}: {text}"


def test_tips_quote_block_swap_limits():
    """The extraction is not silently blind: the tips do quote limits (FLUX.2 dev's is its real maximum)."""
    quoted = [
        number
        for entry in ENTRIES
        for mode in entry.modes
        for _scope, text in _texts(entry, mode)
        for number in _swap_numbers(text)
    ]
    assert len(quoted) > 20
    dev = next(entry for entry in ENTRIES if entry.key == "flux2_dev")
    assert _swap_maximum("flux2_dev", LORA) in _swap_numbers(" ".join(text for _scope, text in _texts(dev, LORA)))


# --------------------------------------------------------------------------------------------- presets vs. tips
SHIPPED = {(entry.key, mode): bool(P.shipped_presets(entry, mode)) for entry in ENTRIES for mode in entry.modes}
HAVE_PRESETS = any(SHIPPED.values())
NO_PRESET = re.compile(r"\bno measured\b|\bno presets? yet\b", re.I)  # not "no preset fits 10 GB" (a tier without one)
UNMEASURED = re.compile(r"\b(?:no|not)\s+(?:VRAM-)?measured\b", re.I)
GB_ALLOWED_AFTER = re.compile(r"\+?\s*(?:GPUs?\b|cards?\b|:|(?:of\s+)?(?:free\s+)?system RAM|on disk)", re.I)


@pytest.mark.skipif(not HAVE_PRESETS, reason="no shipped presets next to the app")
@pytest.mark.parametrize("entry,mode", CASES)
def test_tips_without_presets_quote_no_measurements(entry, mode):
    texts = [text for _scope, text in _texts(entry, mode)]
    joined = " ".join(texts)
    if SHIPPED[(entry.key, mode)]:
        assert not NO_PRESET.search(joined), f"{entry.key} {mode} ships presets, but a tip says it has none"
        return
    assert NO_PRESET.search(joined), f"{entry.key} {mode}: no shipped preset, and no tip says so"
    for text in texts:
        assert not re.search(r"\bpeaks?\b|s/step|\btiers?\b|\bdemo\b", text, re.I), f"{entry.key} {mode}: {text}"
        for match in re.finditer(r"\bmeasured\b", text, re.I):
            assert UNMEASURED.search(text[max(0, match.start() - 12): match.end()]), f"{entry.key} {mode}: {text}"
        for match in re.finditer(r"\d+(?:\.\d+)?\s*GB", text):
            assert GB_ALLOWED_AFTER.match(text, match.end()), f"{entry.key} {mode}: a VRAM figure without a preset: {text}"


@pytest.mark.skipif(not HAVE_PRESETS, reason="no shipped presets next to the app")
@pytest.mark.parametrize("entry,mode", [case for case in CASES if not case.values[0].is_kohya])
def test_tips_without_presets_name_the_defaults_that_load(entry, mode):
    """A training type without presets loads the defaults: the tip saying so quotes their real values."""
    if SHIPPED[(entry.key, mode)]:
        return
    values = _defaults(entry, mode)
    stated = [tip for tip in _expected_tips(entry, mode) if NO_PRESET.search(tip) and "default" in tip.lower()]
    assert stated, f"{entry.key} {mode}: say what loads without a preset"
    checks = (
        (r"\bLR\s+(\d+(?:\.\d+)?(?:e-?\d+)?)", "learning_rate"),
        (r"\brank\s+(\d+)", "network_dim"),
        (r"\bblock swap\s+(\d+)", "blocks_to_swap"),
        (r"\b(\d+)\s+epochs\b", "max_train_epochs"),
        (r"\bcheckpoint every\s+(\d+)", "save_every_n_epochs"),
    )
    for tip in stated:
        for pattern, key in checks:
            for number in re.findall(pattern, tip):
                assert float(number) == pytest.approx(float(values[key])), f"{entry.key} {mode}: {key} loads as {values[key]}, the tip says {number}"
        for optimizer in re.findall(r"\b(AdamW8bit|AdaFactor|Adafactor|AdamW)\b", tip):
            assert str(values["optimizer_type"]).casefold() == optimizer.casefold(), f"{entry.key} {mode}: optimizer {values['optimizer_type']}"


PRESET_TIP = re.compile(r"\bpresets?\b", re.I)


@pytest.mark.skipif(not HAVE_PRESETS, reason="no shipped presets next to the app")
@pytest.mark.parametrize("entry,mode", [case for case in CASES if not case.values[0].is_kohya])
def test_preset_tips_quote_a_rate_the_presets_train_at(entry, mode):
    """A tip on the shipped presets of a training type quotes a learning rate those presets use (the 2026-10-18 rate
    changes: Z-Image Turbo, HiDream-O1, LTX 2.3 and Wan 2.1)."""
    presets = P.shipped_presets(entry, mode)
    if not presets:
        return
    rates = set()
    for item in presets:
        with open(item.path, encoding="utf-8-sig") as handle:
            match = re.search(r"(?m)^learning_rate\s*=\s*([0-9.eE+-]+)", handle.read())
        if match:
            rates.add(float(match.group(1)))
    for scope, text in _texts(entry, mode):
        if scope != mode or not PRESET_TIP.search(text) or NO_PRESET.search(text):
            continue
        for number in re.findall(r"\bLR\s+(\d+(?:\.\d+)?(?:e-?\d+)?)", text):
            assert any(float(number) == pytest.approx(rate) for rate in rates), (
                f"{entry.key} {mode}: the presets train at {sorted(rates)}, the tip says LR {number}: {text}"
            )


def _measured_values(entry) -> set[float]:
    """Every number in the entry's preset file names, their header comments and README files."""
    found: set[float] = set()
    for folder in preset_library.preset_folder_paths(entry.preset_folders):
        if not os.path.isdir(folder):
            continue
        for current, _dirs, files in os.walk(folder):
            for name in files:
                path = os.path.join(current, name)
                text = ""
                if name.lower().endswith(".toml"):
                    with open(path, encoding="utf-8-sig") as handle:
                        text = name + "\n" + "".join(line for line in handle.readlines()[:12] if line.startswith("#"))
                    text = text.replace("_", " ")
                elif name.lower() == "readme.md":
                    with open(path, encoding="utf-8") as handle:
                        text = handle.read()
                found.update(float(number) for number in re.findall(r"\d+(?:\.\d+)?", text))
    return found


@pytest.mark.skipif(not HAVE_PRESETS, reason="no shipped presets next to the app")
@pytest.mark.parametrize("entry", [pytest.param(entry, id=entry.key) for entry in MUSUBI_ENTRIES])
def test_measured_vram_figures_come_from_the_presets(entry):
    measured = _measured_values(entry)
    if not measured:
        pytest.skip(f"{entry.key}: no shipped presets")
    for mode in entry.modes:
        for _scope, text in _texts(entry, mode):
            for figure in re.findall(r"(\d+\.\d+)\s*GB", text):
                assert float(figure) in measured, f"{entry.key}: {figure} GB is in no preset name, header or README: {text}"


def test_no_tip_describes_demo_presets():
    """The unified UI ships GPU-tier presets; the old demo recipes (other block swap, FP8, resolution) are gone."""
    for entry, mode in ((item, mode) for item in ENTRIES for mode in item.modes):
        for _scope, text in _texts(entry, mode):
            assert not re.search(r"\bdemos?\b", text, re.I), f"{entry.key} {mode}: {text}"
