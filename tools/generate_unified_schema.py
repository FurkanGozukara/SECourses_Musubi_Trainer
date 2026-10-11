"""Regenerate ``musubi_tuner_gui/unified/legacy_schema.json`` from the trainer tabs.

The unified UI does not hand-copy the 13 musubi trainers' settings. This tool builds every
trainer tab headlessly (no GPU, no browser), finds the components its *train* action sends in
order, maps each position to its parameter key by calling the tab's ``*_gui_actions`` with
sentinel values (the real ``train_*`` function is replaced by a recorder), and stores every
component's widget type, label, info text, default value, choices and bounds. It also records the
argparse type of every setting the training scripts read (``tools/trainer_arg_types.py``): the
trainers load their runtime TOML without type conversion, so the UI converts each value to it.

Run it after changing any trainer tab (the unified UI test suite fails when the schema is stale):

    venv\\Scripts\\python.exe tools\\generate_unified_schema.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import musubi_tuner_gui.torch_compat  # noqa: E402,F401
import gradio as gr  # noqa: E402

from musubi_tuner_gui.unified.legacy_tabs import LEGACY_BACKENDS, actions_callable, legacy_module  # noqa: E402

OUTPUT = os.path.join(APP_DIR, "musubi_tuner_gui", "unified", "legacy_schema.json")
PROPS = (
    "label",
    "info",
    "value",
    "choices",
    "minimum",
    "maximum",
    "step",
    "lines",
    "max_lines",
    "precision",
    "placeholder",
    "interactive",
    "visible",
    "multiselect",
    "allow_custom_value",
    "type",
)


class _Recorded(Exception):
    pass


def _portable(value):
    """Replace install-specific absolute paths with {APP}/{ROOT} placeholders."""
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for marker, base in (("{APP}", APP_DIR), ("{ROOT}", ROOT_DIR)):
            prefix = base.replace("\\", "/")
            if normalized.lower().startswith(prefix.lower() + "/") or normalized.lower() == prefix.lower():
                return marker + normalized[len(prefix):]
        return value
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _component_info(component) -> dict:
    info = {"cls": type(component).__name__}
    for attr in PROPS:
        if hasattr(component, attr):
            info[attr] = _portable(_jsonable(getattr(component, attr)))
    chain = []
    parent = getattr(component, "parent", None)
    while parent is not None:
        label = getattr(parent, "label", None)
        if label and type(parent).__name__ in ("Accordion", "Tab", "Group", "Column", "Row"):
            chain.append(str(label))
        parent = getattr(parent, "parent", None)
    info["parents"] = list(reversed(chain))
    return info


def _browse_targets(demo, components) -> dict[int, str]:
    """Component id -> "file" | "folder" for fields with a 📁/📂 browse button next to them."""
    targets = {}
    for dep in demo.fns.values():
        if dep.inputs or len(dep.outputs) != 1:
            continue
        trigger_ids = [t[0] if isinstance(t[0], int) else getattr(t[0], "_id", None) for t in dep.targets]
        for trigger_id in trigger_ids:
            button = components.get(trigger_id)
            if button is None or type(button).__name__ != "Button":
                continue
            text = str(getattr(button, "value", "") or "")
            kind = "folder" if ("📂" in text or "folder" in text.lower()) else ("file" if "📁" in text or "file" in text.lower() else None)
            if kind:
                targets[dep.outputs[0]._id] = kind
    return targets


def build_backend(name: str) -> dict:
    backend = LEGACY_BACKENDS[name]
    module = legacy_module(backend)
    with gr.Blocks() as demo:
        backend.build_tab()
    components = {block._id: block for block in demo.blocks.values()}

    # The train event: its first input is the hidden "train_model" action textbox.
    train_dep = None
    for dep in demo.fns.values():
        if len(dep.inputs) > 50 and getattr(dep.inputs[0], "value", None) == "train_model":
            train_dep = dep
    if train_dep is None:
        raise RuntimeError(f"{name}: train event not found")
    settings = list(train_dep.inputs[backend.prefix :])

    recorded = {}

    def recorder(*args, **kwargs):
        params = kwargs.get("parameters")
        if params is None:
            params = next((arg for arg in args if isinstance(arg, list) and arg and isinstance(arg[0], tuple)), None)
        recorded["params"] = params
        raise _Recorded()

    import logging

    logging.disable(logging.CRITICAL)  # gui_actions logs the recorder's exception as a training failure
    original = getattr(module, backend.train_function)
    setattr(module, backend.train_function, recorder)
    try:
        sentinels = [f"__S{index}__" for index in range(len(settings))]
        try:
            actions_callable(backend)("train_model", False, "", True, True, *sentinels)
        except _Recorded:
            pass
        except Exception as exc:  # gui_actions wraps errors into gr.Error
            if "_Recorded" not in repr(exc) and not isinstance(exc.__cause__, _Recorded):
                raise
    finally:
        setattr(module, backend.train_function, original)
        logging.disable(logging.NOTSET)

    params = recorded.get("params") or []
    position_key = {}
    for key, value in params:
        if isinstance(value, str) and value.startswith("__S") and value.endswith("__"):
            position_key[int(value[3:-2])] = key
    if len(position_key) != len(settings):
        missing = [index for index in range(len(settings)) if index not in position_key]
        raise RuntimeError(f"{name}: could not map setting positions {missing[:10]}")

    browse = _browse_targets(demo, components)
    keys = [position_key[index] for index in range(len(settings))]
    entries = {}
    for index, component in enumerate(settings):
        info = _component_info(component)
        if component._id in browse:
            info["browse"] = browse[component._id]
        entries[keys[index]] = info
    return {"module": module.__name__, "keys": keys, "params": entries}


def trainer_arg_types(schema: dict) -> dict:
    """backend -> GUI key -> argparse description, for the keys the training scripts read."""
    sys.path.insert(0, os.path.join(APP_DIR, "tools"))
    from trainer_arg_types import collect

    from musubi_tuner_gui.unified.backends import get_backend

    result = {}
    for backend, dests in collect().items():
        if backend.startswith("kohya:"):
            keys = [spec.key for spec in get_backend(backend).specs]
        else:
            keys = schema["backends"][backend]["keys"]
        result[backend] = {key: {k: v for k, v in dests[key].items() if k != "modes"} for key in keys if key in dests}
        print(f"{backend:16s} {len(result[backend])} settings read by the trainer", flush=True)
    return result


def main() -> None:
    schema = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "gradio": gr.__version__,
        "backends": {},
    }
    for name in LEGACY_BACKENDS:
        schema["backends"][name] = build_backend(name)
        print(f"{name:14s} {len(schema['backends'][name]['keys'])} settings", flush=True)
    schema["arg_types"] = trainer_arg_types(schema)
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(schema, handle, indent=1, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
