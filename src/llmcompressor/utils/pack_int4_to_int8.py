import os
import json
import re
import shutil
from multiprocessing import Pool
from safetensors import safe_open
from safetensors.torch import save_file
import argparse
from tqdm import tqdm
import torch


def _pack_int4_to_int8(tensor):
    """Pack two int4 values into one int8 value."""
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, but got {tensor.ndim}D")

    rows, cols = tensor.shape
    if cols % 2 != 0:
        raise ValueError(
            f"Expected even number of columns for int4 packing, but got shape {tensor.shape}"
        )

    # Pack along the last dimension so values from different rows are never mixed.
    low = tensor[:, 0::2] & 0x0F
    high = tensor[:, 1::2] & 0x0F
    qweight_int8_pack = (high << 4) | low
    return qweight_int8_pack.reshape(rows, cols // 2)


def _process_tensor_file(args):
    """Process a single safetensors file in a subprocess."""
    model_path, save_path, tensor_name, bias32, quant_config_parser, rename = args

    packed_tensor_index_map = {}
    packed_tensors = {}
    total_size = 0

    with safe_open(os.path.join(model_path, tensor_name), framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if not key.endswith('weight') or tensor.dtype != torch.int8 or tensor.ndim != 2:
                if key.endswith('bias') and bias32:
                    packed_tensors[key] = tensor.to(torch.float32)
                else:
                    packed_tensors[key] = tensor
                packed_tensor_index_map[key] = tensor_name
                continue

            if quant_config_parser.is_int4_layer(key):
                print(f"Packing {key} from {tensor_name} with shape {tensor.shape} and dtype {tensor.dtype}")
                packed_tensor = _pack_int4_to_int8(tensor)
                if rename:
                    new_key = f"{key[:-len('weight')]}weight_packed"
                else:
                    new_key = key
                packed_tensors[new_key] = packed_tensor
                packed_tensor_index_map[new_key] = tensor_name
            else:
                packed_tensors[key] = tensor
                packed_tensor_index_map[key] = tensor_name

    for tensor in packed_tensors.values():
        total_size += tensor.numel() * tensor.element_size()

    save_file(packed_tensors, os.path.join(save_path, tensor_name))
    return packed_tensor_index_map, total_size

class QuantConfigParser:
    def __init__(self, model_path) -> None:
        config_path = os.path.join(model_path, 'config.json')
        if not os.path.exists(config_path):
            raise ValueError("config.json not found in model path")
        with open(config_path, 'r') as f:
            self.config = json.load(f).get('quantization_config', {})
            if not self.config:
                raise ValueError("quantization_config not found in config.json")
        
    @staticmethod
    def _as_module_name(layer_name):
        if layer_name.endswith('.weight'):
            return layer_name[: -len('.weight')]
        return layer_name

    @staticmethod
    def _int4_condition(weights_scheme):
        if not isinstance(weights_scheme, dict):
            return False

        num_bits_value = weights_scheme.get('num_bits')
        if num_bits_value is None:
            return False

        try:
            num_bits = int(num_bits_value)
        except (TypeError, ValueError):
            return False

        weight_type = str(weights_scheme.get('type', 'int')).lower()
        return num_bits == 4 and weight_type == 'int'

    def is_int4_layer(self, layer_name):
        module_name = self._as_module_name(layer_name)
        config_groups = self.config.get('config_groups', {})
        for _, group in config_groups.items():
            if not isinstance(group, dict):
                continue

            weights_scheme = group.get('weights', {})
            if not self._int4_condition(weights_scheme):
                continue

            targets = group.get('targets', [])
            if isinstance(targets, str):
                targets = [targets]
            if not isinstance(targets, list):
                continue

            for target in targets:
                if not isinstance(target, str):
                    continue

                # "Linear" is a broad target in recipes and should match linear weights.
                if target == 'Linear':
                    return True

                if target.startswith("re:"):
                    pattern = target[3:]
                    if re.match(pattern, module_name) or re.match(pattern, layer_name):
                        return True
                elif target == module_name or target == layer_name:
                    return True
        return False


class Int8Packer:
    def __init__(self, model_path, save_path, bias32=False, rename=False):
        self.bias32 = bias32
        self.rename = rename
        self.model_path = model_path
        self.save_path = save_path

        if not os.path.exists(self.model_path) or not os.path.isdir(self.model_path):
            raise ValueError("Invalid model path")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.quant_config_parser = QuantConfigParser(model_path)
        self.model_tensors = sorted(
            [f for f in os.listdir(self.model_path) if f.endswith('.safetensors')]
        )

        if not self.model_tensors:
            raise ValueError("No .safetensors files found in model path")

    def save(self):
        print("Start Packing Int4 Weights")
        final_weight_map = {}
        final_total_size = 0

        # Prepare args for multiprocessing
        process_args = [(self.model_path, self.save_path, name, self.bias32, self.quant_config_parser, self.rename) for name in self.model_tensors]

        with Pool(processes=min(len(self.model_tensors), 36)) as pool:
            for local_map, local_size in tqdm(
                pool.imap_unordered(_process_tensor_file, process_args),
                total=len(self.model_tensors),
                desc="Pack Int8 Weights"
            ):
                final_weight_map.update(local_map)
                final_total_size += local_size

        print("Pack Int4 Weights Done")

        # Save model.safetensors.index.json
        print("Updating model.safetensors.index.json")
        org_weight_map_file = os.path.join(self.model_path, 'model.safetensors.index.json')
        if os.path.exists(org_weight_map_file):
            with open(org_weight_map_file, 'r') as f:
                org_weight_map = json.load(f)
            org_weight_map['weight_map'] = final_weight_map

            metadata = org_weight_map.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            metadata['total_size'] = final_total_size
            org_weight_map['metadata'] = metadata

            with open(os.path.join(self.save_path, 'model.safetensors.index.json'), 'w') as f:
                json.dump(org_weight_map, f, sort_keys=True, indent=2)

        print("Copying Other Files")
        # Copy other files (excluding .safetensors and already copied index.json)
        for file in os.listdir(self.model_path):
            if file.endswith('.safetensors') or file == 'model.safetensors.index.json':
                continue
            shutil.copy(os.path.join(self.model_path, file), os.path.join(self.save_path, file))

        print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--model_path', type=str, required=True)
    parser.add_argument('-o', '--save_path', type=str, required=True)
    parser.add_argument('-bias32', action='store_true', default=False)
    parser.add_argument('-rename_packed', action='store_true', default=False, help="Whether to rename packed int8 weight keys to *weight_packed*")
    args = parser.parse_args()

    packer = Int8Packer(args.model_path, args.save_path, args.bias32, args.rename_packed)
    packer.save()
