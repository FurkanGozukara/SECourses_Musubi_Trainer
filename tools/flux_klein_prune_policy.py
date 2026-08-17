#!/usr/bin/env python3
"""Build compact policies from interaction-aware one-layer W8 ablations.

The input ablation set must contain policies made by changing exactly one W8
route in ``--base-policy`` to W4.  Candidates drop routes in ascending measured
damage-per-byte order and retain W8 only until each requested weight budget is
met.  Every emitted policy still needs an end-to-end validation because layer
errors are not additive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--sensitivity", required=True, help="Report containing weight_elements for all 112 projections")
    parser.add_argument("--base-rrmse", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--retained-mode",
        choices=("int8", "bf16"),
        default="int8",
        help="Precision used for retained safeguards (ranking is still derived from W8 ablations).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=float,
        default=[0.32, 0.28, 0.24, 0.20, 0.16, 0.12, 0.08],
        help="Maximum retained W8 fraction of all eligible projection weights.",
    )
    return parser.parse_args()


def _read(path):
    with open(Path(path).expanduser().resolve(), encoding="utf-8") as handle:
        return json.load(handle)


def _routes(policy, names):
    default = policy.get("default_mode", "int4")
    return {name: policy.get("layers", {}).get(name, default) for name in names}


def main():
    args = _args()
    if not all(0.0 <= target <= 1.0 for target in args.targets):
        raise ValueError("Targets must be in [0, 1]")

    base = _read(args.base_policy)
    ablation = _read(args.ablation)
    sensitivity = _read(args.sensitivity)
    rows = sensitivity.get("scopes", {}).get("layers", [])
    weights = {row["name"]: int(row["weight_elements"]) for row in rows}
    if len(weights) != 112:
        raise ValueError(f"Expected 112 projection weights, found {len(weights)}")
    total_weight = sum(weights.values())
    base_routes = _routes(base, weights)
    base_w8 = {name for name, mode in base_routes.items() if mode == "int8"}
    if set(base_routes.values()) - {"int8", "int4"}:
        raise ValueError("Base policy may only contain int8/int4 routes")

    measured = []
    for result in ablation.get("policies", []):
        candidate = _read(result["path"])
        candidate_routes = _routes(candidate, weights)
        changed = [
            name
            for name in weights
            if candidate_routes[name] != base_routes[name]
        ]
        if len(changed) != 1:
            raise ValueError(f"{result['path']} changes {len(changed)} routes, expected one")
        name = changed[0]
        if base_routes[name] != "int8" or candidate_routes[name] != "int4":
            raise ValueError(f"{result['path']} is not a W8-to-W4 ablation")
        delta = float(result["relative_rmse_pct"]) - args.base_rrmse
        measured.append(
            {
                "name": name,
                "weight_elements": weights[name],
                "ablation_rrmse_pct": float(result["relative_rmse_pct"]),
                "delta_rrmse_pct": delta,
                "damage_per_billion_weights": delta * 1e9 / weights[name],
            }
        )
    if {row["name"] for row in measured} != base_w8:
        missing = sorted(base_w8 - {row["name"] for row in measured})
        raise ValueError(f"Ablation set does not cover the base W8 routes: {missing[:3]}")

    ranked = sorted(
        measured,
        key=lambda row: (
            row["damage_per_billion_weights"],
            row["delta_rrmse_pct"],
            row["name"],
        ),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target in sorted(set(args.targets), reverse=True):
        retained = set(base_w8)
        retained_weight = sum(weights[name] for name in retained)
        dropped = []
        for row in ranked:
            if retained_weight <= target * total_weight:
                break
            retained.remove(row["name"])
            retained_weight -= weights[row["name"]]
            dropped.append(row)
        actual = retained_weight / total_weight
        label = f"{round(target * 100):02d}"
        mode_tag = "w8" if args.retained_mode == "int8" else "bf16"
        path = output_dir / f"pruned_{mode_tag}_{label}.json"
        policy = {
            "schema_version": 1,
            "name": f"w4a8_pruned_{mode_tag}_{label}",
            "description": (
                "Packed W4A8 baseline with safeguards selected by complete-policy "
                "W8 ablation damage per byte."
            ),
            "default_mode": "int4",
            "layers": {name: args.retained_mode for name in sorted(retained)},
            "calibration": {
                "base_policy": str(Path(args.base_policy).expanduser().resolve()),
                "ablation_report": str(Path(args.ablation).expanduser().resolve()),
                "base_relative_rmse_pct": args.base_rrmse,
                "target_w8_weight_fraction": target,
                "actual_w8_weight_fraction": actual,
                "retained_layer_count": len(retained),
                "w4_layer_count": len(weights) - len(retained),
                "retained_mode": args.retained_mode,
                "dropped_from_base": [row["name"] for row in dropped],
            },
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(policy, handle, indent=2)
            handle.write("\n")
        manifest.append({"path": str(path), **policy["calibration"]})
        print(
            f"{path.name}: {mode_tag.upper()}={len(retained):2}, "
            f"W4={len(weights) - len(retained):3}, retained weights={actual * 100:5.2f}%",
            flush=True,
        )

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "ranking": ranked,
                "policies": manifest,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
