"""Correct string writing for the ``toml`` package (0.10.x), and reading of files it wrote wrongly.

``toml.dump`` writes a string through Python's repr and then rewrites ``\\x`` hex escapes. A string that
contains a backslash followed by ``x`` loses every backslash escape on the way: ``C:\\a\\x.safetensors``
or ``C:\\Users\\xavier\\data`` were written raw, which is invalid TOML (a preset that no longer loads, a
runtime or dataset TOML the trainer cannot read). ``install()`` replaces that writer with a plain TOML
basic-string encoder for every ``toml.dump`` / ``toml.dumps`` in the process, and lets ``toml.load`` /
``toml.loads`` read a file the old writer corrupted (see ``repair``). The training scripts run in their own
processes and only read files this process wrote.
"""

from __future__ import annotations

import re

import toml
import toml.decoder
import toml.encoder

_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r"}
BASIC_STRING_RE = re.compile(r'"((?:[^"\\\r\n]|\\.)*)"')
VALID_ESCAPE_RE = re.compile(r'\\(?:[btnfr"\\]|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})')
RAW_BACKSLASH_RE = re.compile(r'\\(?!")')

_original_loads = toml.decoder.loads


def dump_str(value) -> str:
    """A TOML basic string: quotes, backslashes and control characters escaped, everything else as is."""
    out = []
    for char in str(value):
        escaped = _ESCAPES.get(char)
        if escaped is None and (ord(char) < 0x20 or ord(char) == 0x7F):
            escaped = "\\u%04x" % ord(char)
        out.append(escaped or char)
    return '"' + "".join(out) + '"'


def repair(text: str) -> str:
    """``text`` with the strings the old writer left raw escaped again. Such a string always contains an
    invalid escape (the ``\\x`` that triggered the bug), so only strings with one change. The old writer
    un-escaped everything up to the last ``\\x`` and left the rest escaped, so the backslashes up to there
    become literal and the rest is kept; a hand-written string with an invalid escape and no such escaped
    rest has all its backslashes made literal. Correctly written strings are left exactly as they are."""

    def fix(match):
        content = match.group(1)
        if not _has_invalid_escape(content):
            return match.group(0)
        split = content.rfind("\\x") + 2
        if split < 2 or _has_invalid_escape(content[split:]):
            split = len(content)
        # the old writer escaped quotes as \" in the raw part as well; doubling that backslash would end the string
        return '"' + RAW_BACKSLASH_RE.sub(r"\\\\", content[:split]) + content[split:] + '"'

    return BASIC_STRING_RE.sub(fix, text)


def loads(s, _dict=dict, decoder=None):
    """``toml.loads`` that also reads a file the old writer corrupted; any other error is raised unchanged."""
    try:
        return _original_loads(s, _dict, decoder)
    except toml.TomlDecodeError:
        repaired = repair(s) if isinstance(s, str) else s
        if repaired == s:
            raise
        return _original_loads(repaired, _dict, decoder)


def install() -> None:
    """Every TomlEncoder created from now on writes strings with ``dump_str``; ``toml.load`` / ``toml.loads``
    read the files the old writer corrupted (``toml.decoder.load`` looks ``loads`` up at call time)."""
    toml.encoder._dump_str = dump_str
    toml.decoder.loads = loads
    toml.loads = loads


def _has_invalid_escape(content: str) -> bool:
    index = content.find("\\")
    while index >= 0:
        valid = VALID_ESCAPE_RE.match(content, index)
        if valid is None:
            return True
        index = content.find("\\", valid.end())
    return False
