"""
Convert a HuggingFace-format quantized DeepSeek V4 checkpoint to raw checkpoint
format for sglang/vLLM compatibility.

Transformations:
  - Strip 'model.' prefix from all tensor keys
  - Rename '.weight_scale' → '.scale'

Usage:
    python tools/convert_to_raw_checkpoint_format.py /path/to/quantized-model
"""

import argparse
import json
import os
from glob import glob

from safetensors.torch import load_file, save_file


def convert_to_raw_checkpoint_format(save_dir: str):
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        shard_files = sorted(glob(os.path.join(save_dir, "*.safetensors")))
    else:
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(
            os.path.join(save_dir, v) for v in index["weight_map"].values()
        ))

    print(f"Converting {len(shard_files)} shards to raw checkpoint format")

    for shard_path in shard_files:
        tensors = load_file(shard_path)
        renamed = {}
        for key, tensor in tensors.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            new_key = new_key.replace(".weight_scale", ".scale")
            new_key = new_key.replace("lm_head.", "head.")
            renamed[new_key] = tensor
        save_file(renamed, shard_path)
        print(f"  {os.path.basename(shard_path)}: {len(tensors)} tensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        new_weight_map = {}
        for key, shard in index["weight_map"].items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            new_key = new_key.replace(".weight_scale", ".scale")
            new_key = new_key.replace("lm_head.", "head.")
            new_weight_map[new_key] = shard
        index["weight_map"] = new_weight_map
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    print(f"Done: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Path to quantized model directory")
    args = parser.parse_args()
    convert_to_raw_checkpoint_format(args.model_dir)
