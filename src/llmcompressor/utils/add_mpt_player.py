import os
import re
import json
import shutil
from safetensors import safe_open
from safetensors.torch import save_file


def _resolve_mtp_layer_name(mtp_layer_id=None, mtp_layer_name=None):
    if mtp_layer_name is not None:
        if not isinstance(mtp_layer_name, str) or not mtp_layer_name.strip():
            raise ValueError("mtp_layer_name must be a non-empty string")
        return mtp_layer_name.strip().rstrip(".")

    if mtp_layer_id is None:
        raise ValueError("Either mtp_layer_id or mtp_layer_name must be provided")

    return f"model.layers.{mtp_layer_id}"


def _build_ignore_patterns(layer_name):
    escaped_layer_name = re.escape(layer_name)
    literal_ignore = layer_name
    regex_ignore = f"re:.*{escaped_layer_name}\\..*"
    return literal_ignore, regex_ignore


def add_mpt_player(
    bf16_model_path,
    quant_model_path,
    mtp_layer_id=None,
    mtp_layer_name=None,
):
    layer_name = _resolve_mtp_layer_name(mtp_layer_id, mtp_layer_name)

    # Read source BF16 index to locate layer tensors.
    bf16_index_file = os.path.join(bf16_model_path, "model.safetensors.index.json")
    with open(bf16_index_file, "r") as f:
        bf16_index_map = json.load(f)

    bf16_weight_map = bf16_index_map.get("weight_map", {})

    # Read target quantized index and merge into it, never overwrite it with source index.
    quant_index_file = os.path.join(quant_model_path, "model.safetensors.index.json")
    with open(quant_index_file, "r") as f:
        quant_index_map = json.load(f)

    quant_weight_map = quant_index_map.get("weight_map", {})
    metadata = quant_index_map.get("metadata", {})

    mtp_weight = {}
    mtp_size = 0
    mtp_output_file = "mtp.safetensors"
    mtp_tensors_by_file = {}
    matched_tensor_count = 0

    for tensor_name, file_name in bf16_weight_map.items():
        if not tensor_name.startswith(f"{layer_name}."):
            continue
        quant_weight_map[tensor_name] = mtp_output_file
        mtp_tensors_by_file.setdefault(file_name, []).append(tensor_name)
        matched_tensor_count += 1

    print(
        f"Found {matched_tensor_count} tensors for {layer_name}. "
        f"Starting extraction..."
    )

    processed_tensor_count = 0

    for file_name, tensor_names in mtp_tensors_by_file.items():
        source_file = os.path.join(bf16_model_path, file_name)
        print(f"Reading {len(tensor_names)} tensors from {source_file}")
        with safe_open(source_file, framework="pt") as f:
            for tensor_name in tensor_names:
                processed_tensor_count += 1
                print(
                    f"[{processed_tensor_count}/{matched_tensor_count}] "
                    f"Processing tensor: {tensor_name}"
                )
                tensor = f.get_tensor(tensor_name)
                mtp_weight[tensor_name] = tensor
                mtp_size += tensor.numel() * tensor.element_size()

    if not mtp_weight:
        raise ValueError(
            f"No tensors found for layer {layer_name} in {bf16_index_file}"
        )

    save_file(mtp_weight, os.path.join(quant_model_path, mtp_output_file))

    # Backup the original quantized index before writing updates.
    backup_index_file = os.path.join(quant_model_path, "model.safetensors.index.json.bak")
    shutil.copy(quant_index_file, backup_index_file)
    print(f"Backed up the original index file to {backup_index_file}")

    metadata["total_size"] = int(metadata.get("total_size", 0)) + mtp_size
    quant_index_map["weight_map"] = quant_weight_map
    quant_index_map["metadata"] = metadata

    with open(quant_index_file, "w") as f:
        json.dump(quant_index_map, f, indent=2, sort_keys=True)
        f.write("\n")

    print(
        "Added MPT player tensors to the quantized model. "
        f"Total size of added tensors: {mtp_size / (1024 * 1024):.2f} MB"
    )


def update_config_ignores(quant_model_path, mtp_layer_id=None, mtp_layer_name=None):
    layer_name = _resolve_mtp_layer_name(mtp_layer_id, mtp_layer_name)
    config_file = os.path.join(quant_model_path, "config.json")
    if not os.path.exists(config_file):
        raise ValueError(f"Config file not found at {config_file}")

    with open(config_file, "r") as f:
        config_data = json.load(f)

    quant_config = config_data.get("quantization_config", {})
    if not isinstance(quant_config, dict) or not quant_config:
        raise ValueError("quantization_config not found in config.json")

    ignores = quant_config.get("ignore", [])
    if isinstance(ignores, str):
        ignores = [ignores]
    elif not isinstance(ignores, list):
        raise ValueError("quantization_config.ignore must be a list or string")

    literal_ignore, regex_ignore = _build_ignore_patterns(layer_name)
    updated = False

    if literal_ignore not in ignores:
        ignores.append(literal_ignore)
        updated = True

    if regex_ignore not in ignores:
        ignores.append(regex_ignore)
        updated = True

    quant_config["ignore"] = ignores
    config_data["quantization_config"] = quant_config

    with open(config_file, "w") as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)
        f_out.write("\n")

    if updated:
        print(
            f"Updated config.json to ignore {layer_name} during quantization."
        )
    else:
        print(
            f"{layer_name} is already in the ignore list of config.json."
        )

    return config_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add MPT player to quantized model")
    parser.add_argument(
        "--bf16_model_path",
        type=str,
        required=True,
        help="Path to the original bf16 model",
    )
    # Keep the misspelled flag as a deprecated alias for compatibility.
    parser.add_argument(
        "--quant_model_path",
        dest="quant_model_path",
        type=str,
        required=True,
        help="Path to the quantized model",
    )

    selector_group = parser.add_mutually_exclusive_group(required=True)
    selector_group.add_argument(
        "--mtp_layer_id",
        type=int,
        help="The layer id of the MPT player to be added",
    )
    selector_group.add_argument(
        "--mtp_layer_name",
        type=str,
        help="The full layer name of the MPT player to be added, e.g. mtp.layers.0",
    )

    args = parser.parse_args()
    add_mpt_player(
        args.bf16_model_path,
        args.quant_model_path,
        mtp_layer_id=args.mtp_layer_id,
        mtp_layer_name=args.mtp_layer_name,
    )
    update_config_ignores(
        args.quant_model_path,
        mtp_layer_id=args.mtp_layer_id,
        mtp_layer_name=args.mtp_layer_name,
    )