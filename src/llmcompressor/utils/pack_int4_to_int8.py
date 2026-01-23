import os
import json
import shutil
from multiprocessing import Pool
from safetensors.torch import safe_open, save_file
import argparse
from tqdm import tqdm
import torch


def _pack_int4_to_int8(tensor):
    """Pack two int4 values into one int8 value."""
    org_shape = tensor.shape
    fatten = tensor.flatten()
    low = fatten[0::2] & 0x0F
    high = fatten[1::2] & 0x0F
    qweight_int8_pack = (high << 4) | low
    qweight_int8_pack = qweight_int8_pack.reshape(org_shape[0], -1)
    return qweight_int8_pack


def _process_tensor_file(args):
    """Process a single safetensors file in a subprocess."""
    model_path, save_path, tensor_name, bias32 = args

    packed_tensor_index_map = {}
    packed_tensors = {}

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

            packed_tensor = _pack_int4_to_int8(tensor)
            new_key = key.replace('weight', 'weight_packed')
            packed_tensors[new_key] = packed_tensor
            packed_tensor_index_map[new_key] = tensor_name

    save_file(packed_tensors, os.path.join(save_path, tensor_name))
    return packed_tensor_index_map


class Int8Packer:
    def __init__(self, model_path, save_path, bias32=False):
        self.bias32 = bias32
        self.model_path = model_path
        self.save_path = save_path

        if not os.path.exists(self.model_path) or not os.path.isdir(self.model_path):
            raise ValueError("Invalid model path")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.model_tensors = [f for f in os.listdir(self.model_path) if f.endswith('.safetensors')]

    def save(self):
        print("Start Packing Int4 Weights")
        final_weight_map = {}

        # Prepare args for multiprocessing
        process_args = [(self.model_path, self.save_path, name, self.bias32) for name in self.model_tensors]

        with Pool(processes=min(len(self.model_tensors), 36)) as pool:
            for local_map in tqdm(
                pool.imap_unordered(_process_tensor_file, process_args),
                total=len(self.model_tensors),
                desc="Pack Int8 Weights"
            ):
                final_weight_map.update(local_map)

        print("Pack Int4 Weights Done")

        # Save model.safetensors.index.json
        print("Updating model.safetensors.index.json")
        org_weight_map_file = os.path.join(self.model_path, 'model.safetensors.index.json')
        if os.path.exists(org_weight_map_file):
            with open(org_weight_map_file, 'r') as f:
                org_weight_map = json.load(f)
            org_weight_map['weight_map'] = final_weight_map
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
    args = parser.parse_args()

    packer = Int8Packer(args.model_path, args.save_path, args.bias32)
    packer.save()
            