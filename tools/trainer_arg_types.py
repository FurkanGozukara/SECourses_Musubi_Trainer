"""Read the argparse definitions of the training scripts the GUI launches.

The trainers load their runtime TOML with ``parser.parse_args(namespace=Namespace(**toml))``. argparse
converts only command-line strings, so a TOML value reaches the training code in exactly the type it
was written with: ``1.0`` stays a float, ``"1e-4"`` stays a string. The unified UI therefore converts
every setting to the type the trainer's own parser declares; this module records those types.

Each script's ``main()`` is run with ``ArgumentParser.parse_args`` patched to stop as soon as the
parser is complete, so nothing is trained and no GPU is used. Run with ``CUDA_VISIBLE_DEVICES=-1``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SD_SCRIPTS_DIR = os.path.join(APP_DIR, "sd-scripts")

# backend name (unified/legacy_tabs.py, kohya:<model>) -> {mode: training script}
TRAINING_SCRIPTS: dict[str, dict[str, str]] = {
    "qwen_image": {"lora": "musubi_tuner.qwen_image_train_network", "full": "musubi_tuner.qwen_image_train"},
    "qwen_image_21": {"lora": "musubi_tuner.qwen_image_21_train_network", "full": "musubi_tuner.qwen_image_21_train"},
    "flux": {"lora": "musubi_tuner.flux_2_train_network", "full": "musubi_tuner.flux_2_train"},
    "zimage": {"lora": "musubi_tuner.zimage_train_network", "full": "musubi_tuner.zimage_train"},
    "krea2": {"lora": "musubi_tuner.krea2_train_network", "full": "musubi_tuner.krea2_train"},
    "ideogram4": {"lora": "musubi_tuner.ideogram4_train_network", "full": "musubi_tuner.ideogram4_train"},
    "ernie_image": {"lora": "musubi_tuner.ernie_image_train_network"},
    "mage_flow": {"lora": "musubi_tuner.mage_flow_train_network"},
    "flux3_klein": {"lora": "musubi_tuner.flux_3_train_network"},
    "moss_image": {"lora": "musubi_tuner.moss_image_train_network"},
    "hidream_o1": {"lora": "musubi_tuner.hidream_o1_train_network", "full": "musubi_tuner.hidream_o1_train"},
    "wan": {"lora": "musubi_tuner.wan_train_network"},
    "ltx2": {"lora": "musubi_tuner.ltx2_train_network", "full": "musubi_tuner.ltx2_train"},
    "minimax_h3": {"lora": "musubi_tuner.minimax_h3_train_network"},
    "kandinsky6": {"lora": "musubi_tuner.kandinsky6_train_network"},
    "kohya:flux1_dev": {"lora": "flux_train_network", "full": "flux_train"},
    "kohya:sdxl": {"lora": "sdxl_train_network", "full": "sdxl_train"},
}


class _Captured(Exception):
    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__("parser captured")
        self.parser = parser


def capture_parser(module_name: str) -> argparse.ArgumentParser:
    """The complete parser of a training script, taken from inside its own ``main()``."""
    if not module_name.startswith("musubi_tuner.") and SD_SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SD_SCRIPTS_DIR)
    import torch

    # wan/modules/t5.py evaluates torch.cuda.current_device() as a default argument at import time;
    # answer it without creating a CUDA context (this tool must never touch a GPU)
    real_current_device = torch.cuda.current_device
    torch.cuda.current_device = lambda: 0
    try:
        module = importlib.import_module(module_name)
    finally:
        torch.cuda.current_device = real_current_device
    original_parse_args = argparse.ArgumentParser.parse_args
    original_parse_known = argparse.ArgumentParser.parse_known_args

    def stop(self, *args, **kwargs):
        raise _Captured(self)

    argparse.ArgumentParser.parse_args = stop
    argparse.ArgumentParser.parse_known_args = stop
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if hasattr(module, "main"):
                module.main()
            else:
                module.setup_parser().parse_args()
    except _Captured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args
        argparse.ArgumentParser.parse_known_args = original_parse_known
    raise RuntimeError(f"{module_name}: main() finished without building an argument parser")


def _type_name(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return "bool"
    if isinstance(action, argparse._StoreConstAction):
        return type(action.const).__name__ if action.const is not None else "const"
    kind = action.type
    if kind in (int, float, str):
        return kind.__name__
    if kind is None:
        return "str"
    name = getattr(kind, "__name__", type(kind).__name__)
    if "bool" in name.lower():
        return "bool"
    return f"custom:{name}"


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def describe_parser(parser: argparse.ArgumentParser) -> dict[str, dict]:
    """dest -> {"type", "nargs", "choices", "default", "aliases"} for every option of the parser. ``aliases`` are the
    option names that differ from the dest (``--ltx2_mode`` for ``ltx_mode``): a runtime TOML key with such a name is
    ignored by ``parse_args(namespace=...)``, which fills dests only (Guardrail 20)."""
    described: dict[str, dict] = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or action.dest in (argparse.SUPPRESS, None):
            continue
        entry = {"type": _type_name(action), "default": _json_safe(action.default)}
        if action.nargs not in (None, 0):
            entry["nargs"] = action.nargs if isinstance(action.nargs, (int, str)) else repr(action.nargs)
        if action.choices is not None:
            entry["choices"] = _json_safe(list(action.choices))
        aliases = sorted({option.lstrip("-") for option in action.option_strings} - {action.dest})
        if aliases:
            entry["aliases"] = aliases
        described[action.dest] = entry
    return described


def collect(backends=None) -> dict[str, dict[str, dict]]:
    """backend -> dest -> description, merged over the backend's LoRA and full fine-tuning scripts."""
    result: dict[str, dict[str, dict]] = {}
    for backend, scripts in TRAINING_SCRIPTS.items():
        if backends is not None and backend not in backends:
            continue
        merged: dict[str, dict] = {}
        for mode, module_name in scripts.items():
            for dest, entry in describe_parser(capture_parser(module_name)).items():
                if dest in merged and merged[dest]["type"] != entry["type"]:
                    merged[dest]["type_" + mode] = entry["type"]
                record = merged.setdefault(dest, {**entry, "modes": []})
                record["modes"].append(mode)
                if entry.get("aliases"):
                    record["aliases"] = sorted(set(record.get("aliases", [])) | set(entry["aliases"]))
        result[backend] = merged
    return result


if __name__ == "__main__":
    import json

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    data = collect(sys.argv[1:] or None)
    for name, dests in data.items():
        kinds: dict[str, int] = {}
        for entry in dests.values():
            kinds[entry["type"]] = kinds.get(entry["type"], 0) + 1
        print(f"{name:16s} {len(dests):4d} options  {json.dumps(kinds)}")
