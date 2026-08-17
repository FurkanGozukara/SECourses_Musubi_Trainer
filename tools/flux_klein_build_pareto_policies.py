#!/usr/bin/env python3
"""Build cumulative mixed-precision candidates from forward sensitivity data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        help="Target fractions of eligible weight elements routed to INT4.",
    )
    return parser.parse_args()


def main():
    args = _args()
    source = Path(args.sensitivity).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(source, encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report.get("scopes", {}).get("layers", [])
    if len(rows) != 112:
        raise ValueError(f"Expected 112 layer measurements, found {len(rows)}")
    if not all(0.0 < target <= 1.0 for target in args.targets):
        raise ValueError("Targets must be in (0, 1]")

    total_elements = sum(int(row["weight_elements"]) for row in rows)
    # Minimize measured damage per byte saved.  Negative-damage candidates are
    # selected first, followed by the least expensive error-energy tradeoffs.
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["error_energy_increase_pct"]) / max(int(row["weight_elements"]), 1),
            float(row["error_energy_increase_pct"]),
            row["name"],
        ),
    )
    manifest = []
    for target in sorted(set(args.targets)):
        selected = []
        selected_elements = 0
        for row in ranked:
            selected.append(row)
            selected_elements += int(row["weight_elements"])
            if selected_elements >= target * total_elements:
                break
        actual = selected_elements / total_elements
        label = f"{round(target * 100):02d}"
        path = output_dir / f"forward_pareto_{label}.json"
        policy = {
            "schema_version": 1,
            "name": f"forward_pareto_{label}",
            "description": (
                "Cumulative INT4 substitutions selected by end-to-end error-energy "
                "damage per weight byte saved from an all-INT8 ConvRot baseline."
            ),
            "default_mode": "int8",
            "layers": {row["name"]: "int4" for row in selected},
            "calibration": {
                "method": report.get("method"),
                "source": str(source),
                "target_int4_weight_fraction": target,
                "actual_int4_weight_fraction": actual,
                "int4_layer_count": len(selected),
                "int4_weight_elements": selected_elements,
                "eligible_weight_elements": total_elements,
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(policy, handle, indent=2)
            handle.write("\n")
        manifest.append({"path": str(path), **policy["calibration"]})
        print(
            f"{path.name}: {len(selected):3} layers, "
            f"{actual * 100:6.2f}% of eligible weights INT4",
            flush=True,
        )
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "policies": manifest}, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
