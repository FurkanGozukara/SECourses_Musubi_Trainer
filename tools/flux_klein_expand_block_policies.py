#!/usr/bin/env python3
"""Expand compact block-route specifications into exact 112-layer policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _projection_names(block):
    if block.startswith("double_blocks."):
        return [
            f"{block}.{stream}_{kind}.{projection}"
            for stream in ("img", "txt")
            for kind, projections in (("attn", ("qkv", "proj")), ("mlp", ("0", "2")))
            for projection in projections
        ]
    if block.startswith("single_blocks."):
        return [f"{block}.linear1", f"{block}.linear2"]
    raise ValueError(f"Unsupported block name: {block}")


def main():
    args = _args()
    spec_path = Path(args.spec).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(spec_path, encoding="utf-8") as handle:
        spec = json.load(handle)
    candidates = list(spec.get("candidates", []))
    for sweep in spec.get("sweeps", []):
        for block in sweep["candidates"]:
            blocks = dict(sweep.get("base_blocks", {}))
            blocks[block] = sweep.get("mode", "int8")
            candidates.append(
                {
                    "name": f"{sweep['name_prefix']}_{block.replace('.', '_')}",
                    "description": sweep.get("description", "One-at-a-time block route sweep"),
                    "default_mode": sweep.get("default_mode", "int4"),
                    "blocks": blocks,
                }
            )
    manifest = []
    for candidate in candidates:
        layers = {}
        for block, mode in candidate.get("blocks", {}).items():
            if mode not in {"bf16", "int8", "int4"}:
                raise ValueError(f"Invalid mode {mode!r} for {block}")
            layers.update({name: mode for name in _projection_names(block)})
        for name, mode in candidate.get("layers", {}).items():
            if mode not in {"bf16", "int8", "int4"}:
                raise ValueError(f"Invalid mode {mode!r} for {name}")
            layers[name] = mode
        policy = {
            "schema_version": 1,
            "name": candidate["name"],
            "description": candidate.get("description", "Block-expanded sensitivity candidate"),
            "default_mode": candidate.get("default_mode", "int4"),
            "layers": dict(sorted(layers.items())),
            "metadata": {
                "source_spec": str(spec_path),
                "block_routes": candidate.get("blocks", {}),
            },
        }
        path = output_dir / f"{candidate['name']}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(policy, handle, indent=2)
            handle.write("\n")
        manifest.append(str(path))
        print(f"{path.name}: {len(layers)} explicit projection routes", flush=True)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "policies": manifest}, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
