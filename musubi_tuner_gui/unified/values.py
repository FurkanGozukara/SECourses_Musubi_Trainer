"""Turn what users type into exactly what the trainers expect.

The trainers read their runtime TOML with ``parser.parse_args(namespace=Namespace(**toml))``, which
does no type conversion: a value reaches the training code in the type it was written with. Every
setting is therefore converted here to the type the trainer's own argparse option declares
(``legacy_schema.json`` → ``arg_types``, recorded by ``tools/trainer_arg_types.py``), and paths are
cleaned of the quotes, spaces and slashes that copy-paste brings along.
"""

from __future__ import annotations

import math
import os
import re
from urllib.parse import unquote, urlparse

QUOTES = "\"'`“”‘’"
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")
THOUSANDS_RE = re.compile(r"^[+-]?[1-9]\d{0,2}(,\d{3})+$")  # 1,000 / 10,000 / 1,000,000 (never 0,001)
DECIMAL_COMMA_RE = re.compile(r"[+-]?\d+,\d+")  # 0,5 inside a list of numbers is one number
TRUE_WORDS = {"1", "true", "yes", "on", "y", "t"}
FALSE_WORDS = {"0", "false", "no", "off", "n", "f", "", "none"}


class ValueProblem(ValueError):
    """A setting the trainer cannot use; the message names the setting in plain words."""


# ------------------------------------------------------------------------------------------------ paths
def strip_quotes(text: str) -> str:
    """Remove surrounding whitespace and any number of matching or stray surrounding quotes."""
    previous = None
    while text != previous:
        previous = text
        text = text.strip()
        if not text:
            break
        if len(text) >= 2 and text[0] in QUOTES and text[-1] in QUOTES:
            text = text[1:-1]
        elif text[0] in QUOTES and text.count(text[0]) == 1:
            text = text[1:]
        elif text[-1] in QUOTES and text.count(text[-1]) == 1:
            text = text[:-1]
    return text


def looks_like_local_path(text: str) -> bool:
    return bool(
        DRIVE_RE.match(text)
        or text.startswith(("\\", "/", ".", "~", "%", "$"))
        or "\\" in text
        or os.path.exists(text)
    )


def clean_path(value) -> str:
    """A path as the trainer should receive it: no quotes or padding, native separators, ``%VAR%`` /
    ``~`` expanded, ``file:///`` URLs converted. Hugging Face repo ids (``org/model``) and URLs stay as typed.
    Text of several lines is never one path and stays as written (kohya's Sample Prompts also takes the
    prompts themselves)."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\n" in text:
        return text
    text = strip_quotes(text)
    if not text:
        return ""
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:", text):
            text = text[1:]
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            text = f"//{parsed.netloc}{text}"
        elif os.name != "nt" and text.startswith("//"):
            text = "/" + text.lstrip("/")  # file:////tmp/x ("file:///" + an absolute Linux path): Linux has no UNC
    elif re.match(r"^[a-z][a-z0-9+.-]*://", text, re.I):
        return text  # http(s):// and other URLs
    if HF_REPO_RE.match(text) and not os.path.exists(text):
        return text
    text = os.path.expandvars(os.path.expanduser(text))
    if not looks_like_local_path(text) and "/" not in text:
        return text  # a bare file name stays relative to the trainer's working folder
    return _portable(text)


def _portable(text: str) -> str:
    """Native separators for this machine's absolute paths; paths written for the other OS (a Linux path in
    a preset opened on Windows, or the reverse) and relative paths keep forward slashes, so a preset saved
    here still works on Linux / RunPod installs (``../Training_Outputs`` stays portable)."""
    import posixpath

    windows_absolute = bool(DRIVE_RE.match(text)) or text.startswith(("\\\\", "//"))
    if os.name == "nt":
        if windows_absolute:
            return os.path.normpath(text)
        if text.startswith("/"):
            return posixpath.normpath(text)  # a Linux path: leave it a Linux path
        return posixpath.normpath(text.replace("\\", "/"))
    # Linux: a backslash is a separator typed or pasted on Windows (..\Training_Outputs, C:\data\, a trailing \),
    # never part of a trainer file name. A Windows path stays one (C:/data) so model paths are still relocated.
    text = text.replace("\\", "/")
    if windows_absolute:
        return posixpath.normpath(text)  # C:/data/set, //server/share (posixpath keeps the double slash)
    return os.path.normpath(text)


NEW_PATH_RE = re.compile(r"\s+(?=[A-Za-z]:[\\/]|\\\\|/|~[\\/])")  # whitespace before the start of another absolute path
WEIGHT_FILE_RE = re.compile(r"\S+\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$", re.I)


def _split_path_line(line: str) -> list[str]:
    """One line of a multi-path field (the classic tabs' syntax): space-separated, with quotes around
    paths that contain spaces. Unquoted, a line splits only where another absolute path starts or where
    every word is a model file name, so ``C:/my loras/x.safetensors`` stays one path."""
    whole = clean_path(line)
    if not whole or os.path.exists(whole) or not any(char.isspace() for char in whole):
        return [line]
    if any(char in QUOTES for char in line):
        import shlex

        try:
            return shlex.split(line, posix=False)
        except ValueError:
            pass
    parts = []
    for part in NEW_PATH_RE.split(line.strip()):
        words = part.split()
        parts.extend(words if len(words) > 1 and all(WEIGHT_FILE_RE.fullmatch(word) for word in words) else [part])
    return parts


def clean_paths(value) -> list[str]:
    """Several paths: a list, or text with one path per line / ``;``, or several on one line (see above)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [part for line in re.split(r"[\r\n;]+", str(value)) for part in _split_path_line(line)]
    return [path for path in (clean_path(item) for item in items) if path]


# ------------------------------------------------------------------------------------------------ numbers
def _normalize_separators(text: str, integer: bool) -> str:
    """Decimal commas (Turkish / German keyboards) and thousands separators -> Python syntax.

    Whole-number settings read ``10,000`` as ten thousand; decimal settings read ``1,004`` or
    ``0,0001`` as decimal commas (who types a thousands separator into a learning rate?). With both
    separators the last one is the decimal mark: ``1,000.5`` and ``1.000,5`` are both 1000.5.
    """
    if "," not in text:
        return text
    if "." in text:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")
        return text.replace(",", "")
    if integer and THOUSANDS_RE.match(text):
        return text.replace(",", "")
    if text.count(",") == 1:
        return text.replace(",", ".")
    if THOUSANDS_RE.match(text):
        return text.replace(",", "")
    return text


def parse_number(value, *, label: str = "This setting", integer: bool = False, whole: bool = False):
    """int / float from what a user typed: ``" 1e-4 "``, ``"0,0001"``, ``"+3"``, ``2.0``. Blank -> None.
    ``whole``: a whole-number setting that is not checked here (``10,000`` still means ten thousand)."""
    if value is None or isinstance(value, bool):
        if isinstance(value, bool):
            return int(value)
        return None
    if isinstance(value, (int, float)):
        number = value
    else:
        text = strip_quotes(str(value)).replace("−", "-").replace("_", "").replace(" ", "").replace(" ", "")
        if text == "":
            return None
        text = _normalize_separators(text, integer or whole)
        try:
            number = float(text)
        except ValueError:
            raise ValueProblem(f"{label}: '{value}' is not a number.") from None
    if isinstance(number, float) and (math.isnan(number) or math.isinf(number)):
        raise ValueProblem(f"{label}: '{value}' is not a finite number.")
    if integer:
        if float(number) != int(float(number)):
            raise ValueProblem(f"{label} must be a whole number (got {value}).")
        return int(float(number))
    return number


def json_like_number(number):
    """The classic tabs receive numbers through JSON: whole numbers arrive as int."""
    if isinstance(number, float) and number.is_integer():
        return int(number)
    return number


def _number_tokens(text: str) -> list[str]:
    """Split a typed list of numbers. Spaces and ``;`` always separate; a comma separates too, except
    between digits of one number (``0,5`` is a decimal comma: 0.5) — so ``0.5,1`` and ``0.5, 1`` are two."""
    tokens = []
    for token in re.split(r"[\s;]+", text):
        if not token:
            continue
        if "," in token and not DECIMAL_COMMA_RE.fullmatch(token):
            tokens.extend(part for part in token.split(",") if part)
        else:
            tokens.append(token)
    return tokens


def parse_numbers(value, *, label: str = "This setting", integer: bool = False) -> list:
    """Several numbers: a list, or text separated by spaces, semicolons or commas."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else _number_tokens(strip_quotes(str(value)))
    numbers = [parse_number(item, label=label, integer=integer) for item in items]
    return [number for number in numbers if number is not None]


# ------------------------------------------------------------------------------------------------ booleans
def parse_bool(value, *, label: str = "This setting") -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = strip_quotes(str(value)).lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    raise ValueProblem(f"{label}: '{value}' is not on/off (use true or false).")


# ------------------------------------------------------------------------------------------------ argument lists
def split_args(value) -> list[str]:
    """``key=value`` argument lists (optimizer / network / scheduler args) as clean items, however they were typed:
    spaces, new lines, ``;`` or commas between items, ``key = value``, brackets and quotes (arguments.py). Commas
    inside a value stay (``betas=0.9,0.99`` is one tuple argument). The classic save removes every comma from
    text input, which breaks tuple arguments, but keeps list items as they are."""
    from .arguments import argument_items

    return argument_items(value, literal=False)


# ------------------------------------------------------------------------------------------------ one setting
def to_trainer_value(value, *, label: str, kind: str, arg: dict | None, is_path: bool, whole: bool = False):
    """Convert one UI value for the trainer.

    ``kind`` is the unified field kind (text, textarea, number, slider, checkbox, dropdown, radio, file,
    folder); ``arg`` the trainer's argparse description (type / nargs / default) or None for GUI-only
    settings, which keep the classic tabs' JSON-like types. ``whole``: the setting takes whole numbers
    (read ``10,000`` as ten thousand; fractions are reported by ``Backend.check_values``).
    """
    arg_type = (arg or {}).get("type")
    nargs = (arg or {}).get("nargs")
    many = nargs not in (None, "?")
    if is_path or kind in ("file", "folder") or arg_type == "custom:Path":
        if many:
            return clean_paths(value)
        return clean_path(value)
    if arg_type in ("int", "float", "custom:_int_or_float", "custom:int_or_float"):
        integer = arg_type == "int"
        if many:
            numbers = parse_numbers(value, label=label, integer=integer)
            return [float(n) for n in numbers] if arg_type == "float" else numbers
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        number = parse_number(value, label=label, integer=integer, whole=whole)
        if number is None:
            return None
        if arg_type == "float":
            return float(number)
        if arg_type in ("custom:_int_or_float", "custom:int_or_float"):
            # warmup / decay steps: musubi reads a float as a fraction of the total steps and an int as a step
            # count, so only a value below 1 stays a fraction (0.1 = 10 %) and 100.0 is 100 steps
            return number if 0 < number < 1 else int(round(number))
        return int(number) if integer else json_like_number(number)
    if arg_type == "bool" or kind == "checkbox":
        return parse_bool(value, label=label)
    if many and arg_type in (None, "str") and kind in ("text", "textarea", ""):
        return split_args(value)
    if kind in ("number", "slider"):
        return json_like_number(parse_number(value, label=label, whole=whole))
    if isinstance(value, str) and kind not in ("textarea",):
        return value.strip()
    return value
