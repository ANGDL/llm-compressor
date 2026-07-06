#!/usr/bin/env python3
"""
Fix the quantization_config.ignore list in a previously quantized model's config.json.

Removes spurious container module entries (e.g. ExpertMLPWithGate) that were
incorrectly added by compressed_tensors' get_vllm_module_type heuristic.

The fix logic: if a module name in `ignore` has quantized children (identified by
the presence of `<name>.<child>.weight_scale` in the safetensors index), it is a
container and should be removed from the ignore list.

Usage:
    python fix_ignore_list.py /path/to/quantized-model [--dry-run]

This reads the model's config.json and model.safetensors.index.json, filters out
the spurious ignore entries, and writes back config.json (unless --dry-run).
"""

import argparse
import json
import os
import sys


def load_weight_names(model_dir: str) -> set[str]:
    """Load all weight names from the safetensors index or single file."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return set(index["weight_map"].keys())

    # Fallback: try to read keys from a single safetensors file
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_path):
        try:
            from safetensors import safe_open

            with safe_open(single_path, framework="pt") as f:
                return set(f.keys())
        except ImportError:
            print(
                "ERROR: safetensors package not installed, "
                "and no model.safetensors.index.json found.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(
        f"ERROR: No model.safetensors.index.json or model.safetensors in {model_dir}",
        file=sys.stderr,
    )
    sys.exit(1)


def has_quantized_children(module_name: str, weight_names: set[str]) -> bool:
    """Check if a module has quantized children by looking for weight_scale entries.

    A module is a container with quantized children if there exists any key like:
        <module_name>.<child_name>.weight_scale
    where <child_name> is a direct child (one level below module_name).

    For modules whose runtime path doesn't match the checkpoint path (e.g. MTP layer
    uses 'mtp.decoder.mlp.experts.N' at runtime but 'model.layers.78.mlp.experts.N'
    in checkpoint), falls back to checking the suffix pattern across all weight keys.
    """
    prefix = module_name + "."
    for name in weight_names:
        if not name.startswith(prefix):
            continue
        # Check if this is a weight_scale of a direct child
        suffix = name[len(prefix):]
        # suffix should be like "gate_proj.weight_scale" (one dot)
        parts = suffix.split(".")
        if len(parts) == 2 and parts[1] == "weight_scale":
            return True

    # Fallback: for expert-like paths (*.experts.N), check if ANY checkpoint path
    # with the same tail pattern has weight_scale children.
    # This handles MTP naming mismatch (mtp.decoder.* vs model.layers.78.*)
    if ".experts." in module_name:
        # Extract the tail: "mlp.experts.N"
        experts_idx = module_name.find("mlp.experts.")
        if experts_idx >= 0:
            tail = module_name[experts_idx:]  # e.g. "mlp.experts.0"
            tail_prefix = tail + "."
            for name in weight_names:
                idx = name.find(tail_prefix)
                if idx < 0:
                    continue
                after_tail = name[idx + len(tail_prefix):]
                parts = after_tail.split(".")
                if len(parts) == 2 and parts[1] == "weight_scale":
                    return True

    return False


def fix_ignore_list(
    model_dir: str, dry_run: bool = False, config_path: str | None = None
):
    """Fix the ignore list in config.json."""
    if config_path is None:
        config_path = os.path.join(model_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    quant_config = config.get("quantization_config")
    if quant_config is None:
        print("No quantization_config found in config.json, nothing to fix.")
        return

    ignore = quant_config.get("ignore")
    if not ignore:
        print("ignore list is empty, nothing to fix.")
        return

    print(f"Loading weight names from {model_dir}...")
    weight_names = load_weight_names(model_dir)
    print(f"  Found {len(weight_names)} weight entries")
    print(f"  Current ignore list: {len(ignore)} entries")

    # Filter: remove entries whose direct children have weight_scale
    to_remove = []
    for entry in ignore:
        if has_quantized_children(entry, weight_names):
            to_remove.append(entry)

    if not to_remove:
        print("\nNo spurious container entries found. Ignore list is clean.")
        return

    # Categorize removed entries for reporting
    print(f"\n  Found {len(to_remove)} spurious container entries to remove:")

    # Sample some entries for display
    sample_size = min(5, len(to_remove))
    for entry in to_remove[:sample_size]:
        print(f"    - {entry}")
    if len(to_remove) > sample_size:
        print(f"    ... and {len(to_remove) - sample_size} more")

    # Build filtered list
    remove_set = set(to_remove)
    filtered_ignore = [entry for entry in ignore if entry not in remove_set]

    print(f"\n  After fix: {len(filtered_ignore)} entries (removed {len(to_remove)})")

    if filtered_ignore:
        print("\n  Remaining ignore entries:")
        for entry in filtered_ignore[:20]:
            print(f"    - {entry}")
        if len(filtered_ignore) > 20:
            print(f"    ... and {len(filtered_ignore) - 20} more")

    if dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    # Write back
    quant_config["ignore"] = filtered_ignore
    config["quantization_config"] = quant_config

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n✅ Fixed config written to {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix spurious expert container entries in quantization_config.ignore"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to the quantized model directory (containing config.json and safetensors)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be changed, don't write",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json (default: <model_dir>/config.json)",
    )
    args = parser.parse_args()

    fix_ignore_list(args.model_dir, dry_run=args.dry_run, config_path=args.config)


if __name__ == "__main__":
    main()
