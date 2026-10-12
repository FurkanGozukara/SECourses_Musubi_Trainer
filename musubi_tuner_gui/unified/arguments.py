"""``key=value`` argument lists (Optimizer / LR Scheduler / Network Arguments) as the trainers must receive them.

People type these lists in many ways; every one of them becomes the same clean list:

* separators: any run of spaces, tabs, new lines or ``;``, and a comma where the next ``key=`` starts
  (``a=1,b=2`` and ``a=1, b=2`` are two items);
* ``key = value`` / ``key =value`` / ``key= value`` are one item;
* brackets and quotes keep a value together: ``betas=(0.9, 0.999)``, ``betas=0.9,0.999``,
  ``mode='triangular2'``, ``exclude_patterns=['.*(norm|bias).*']``; one pair of quotes around a whole item goes;
* a list from a TOML file or a pasted Python list (``['a=1', 'b=2']``) is read item by item.

The trainers split each item at its first ``=`` and read optimizer / scheduler values with ``ast.literal_eval``
(``LITERAL_KEYS``): there ``true`` / ``false`` / ``none`` become ``True`` / ``False`` / ``None`` (as text the value
would be a non-empty string, which is always true) and a decimal comma (``weight_decay=0,01``) becomes a point.
Network modules read their values as text and get them unchanged.
"""

from __future__ import annotations

import ast
import re

LITERAL_KEYS = frozenset({"optimizer_args", "lr_scheduler_args", "base_optimizer_args"})
NETWORK_KEYS = frozenset({"network_args"})
ARGUMENT_KEYS = LITERAL_KEYS | NETWORK_KEYS

_KEY_RE = re.compile(r"[A-Za-z_][\w.\-]*\s*=(?!=)")
_NUMBER_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_DECIMAL_COMMA_RE = re.compile(r"^[+-]?\d+,\d+$")
_OPEN, _CLOSE = "([{", ")]}"
_QUOTES = "\"'"


def _scan(text: str) -> list[str]:
    """Tokens split at whitespace / ``;`` and at item commas, outside brackets and quotes."""
    tokens: list[str] = []
    current: list[str] = []
    depth, quote, equals = 0, "", 0
    for index, char in enumerate(text):
        if quote:
            current.append(char)
            if char == quote:
                quote = ""
        elif char in _QUOTES:
            quote = char
            current.append(char)
        elif char in _OPEN:
            depth += 1
            current.append(char)
        elif char in _CLOSE:
            depth = max(0, depth - 1)
            current.append(char)
        elif depth == 0 and (char.isspace() or char == ";"):
            if current:
                tokens.append("".join(current))
                current, equals = [], 0
        elif depth == 0 and char == ",":
            # a comma separates items only after one complete key=value and where the next key starts (or the text
            # ends); else it is part of a value: betas=0.9,0.999 and sd-scripts' network_reg_lrs=a_=1e-3,b=2e-3 stay one
            rest = text[index + 1:].lstrip()
            if (not rest or _KEY_RE.match(rest)) and equals <= 1:
                if current:
                    tokens.append("".join(current))
                    current, equals = [], 0
            else:
                current.append(char)
        else:
            if char == "=" and depth == 0:
                equals += 1
            current.append(char)
    if current:
        tokens.append("".join(current))
    return tokens


def _unwrap(item: str) -> str:
    """One pair of quotes around a whole item goes ("betas=(0.9, 0.999)"); quotes inside a value stay."""
    item = item.strip()
    if len(item) >= 2 and item[0] == item[-1] and item[0] in _QUOTES:
        inner = item[1:-1]
        if inner.count(item[0]) == 0:
            return inner.strip()
    return item


def _looks_numeric_end(text: str) -> bool:
    return bool(text) and (text[-1].isdigit() or text[-1] in ".)]")


def split_items(value) -> list[str]:
    """The ``key=value`` items of an argument list, however it was typed (see the module docstring)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for entry in value:
            items.extend(split_items(str(entry)))
        return items
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if text[0] == "[" and text[-1] == "]":
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and all(isinstance(entry, str) for entry in parsed):
            return split_items(list(parsed))
        if "=" not in text[1:2]:
            text = text[1:-1]
    items: list[str] = []
    pending_key = False  # the last item ended with "=": its value follows ("key = value")
    for raw in _scan(text):
        token = _unwrap(raw)
        if not token:
            continue
        if items and (pending_key or (token.startswith("=") and "=" not in items[-1])):  # "key = value", "key =value"
            items[-1] = items[-1] + token
            pending_key = items[-1].endswith("=")
            continue
        if _KEY_RE.match(token):
            items.append(token)
            pending_key = token.endswith("=")
            continue
        if items and "=" in items[-1]:
            previous = items[-1]
            if previous.endswith(","):  # "betas=0.9, 0.999"
                items[-1] = previous + token
                continue
            if _looks_numeric_end(previous) and (token[0].isdigit() or token[0] in "+-.(["):  # "betas=0.9 0.999"
                items[-1] = previous + "," + token
                continue
        items.append(token)  # no key: reported by argument_problems
        pending_key = False
    return [item.strip().rstrip(",").strip() for item in items if item.strip().strip(",")]


def _literal_value(value: str) -> str:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in ("true", "false"):
        return "True" if lowered == "true" else "False"
    if lowered in ("none", "null"):
        return "None"
    if _DECIMAL_COMMA_RE.match(stripped):
        try:
            ast.literal_eval(stripped)
        except (ValueError, SyntaxError):  # "0,01": a decimal comma, not a tuple (a leading 0 is no tuple element)
            return stripped.replace(",", ".")
    return stripped


def normalize_items(items: list[str], *, literal: bool) -> list[str]:
    """``key=value`` with no spaces around the key or the value; literal lists get Python literals for words."""
    result = []
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            result.append(item.strip())
            continue
        value = value.strip()
        if literal:
            value = _literal_value(value)
        result.append(f"{key.strip()}={value}")
    return result


def argument_items(value, *, literal: bool) -> list[str]:
    return normalize_items(split_items(value), literal=literal)


def argument_problems(label: str, value, *, literal: bool) -> list[str]:
    """Items the trainer cannot use (it would stop with an error only after loading the models)."""
    malformed, unreadable = [], []
    for item in argument_items(value, literal=literal):
        key, separator, raw = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][\w.\-]*", key) or raw == "":
            malformed.append(item)
            continue
        if literal:
            try:
                ast.literal_eval(raw)
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                if not re.fullmatch(r"[A-Za-z_][\w.\-]*", raw):  # a bare word is passed as text; anything else breaks
                    unreadable.append(item)
    problems = []
    if malformed:
        problems.append(
            f"{label}: every item must be key=value (separate items with spaces, commas or new lines). Not usable: {', '.join(malformed)}"
        )
    if unreadable:
        problems.append(
            f"{label}: these values cannot be read as Python values (put text in quotes, e.g. mode='triangular2'): {', '.join(unreadable)}"
        )
    return problems
