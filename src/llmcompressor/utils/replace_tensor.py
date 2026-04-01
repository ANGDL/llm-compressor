import argparse
import copy
import json
import math
import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
OUTPUT_SHARD_STEM = "matched_tensors"

DTYPE_BYTE_SIZES = {
	"BOOL": 1,
	"U8": 1,
	"I8": 1,
	"F8_E4M3": 1,
	"F8_E5M2": 1,
	"F8_E8M0": 1,
	"U16": 2,
	"I16": 2,
	"F16": 2,
	"BF16": 2,
	"U32": 4,
	"I32": 4,
	"F32": 4,
	"U64": 8,
	"I64": 8,
	"F64": 8,
}

TENSOR_SUFFIXES = [
	"weight_global_scale",
	"input_global_scale",
	"weight_zero_point",
	"weight_packed",
	"weight_scale",
	"weight_g_idx",
	"input_scale",
	"input_zero_point",
	"weight",
	"bias",
	"k_scale",
	"v_scale",
	"q_scale",
]


def parse_args():
	parser = argparse.ArgumentParser(description=main.__doc__)
	parser.add_argument(
		"--source_model_path",
		type=str,
		required=True,
		help="Path to the source model directory or a single safetensors file",
	)
	parser.add_argument(
		"--target_model_path",
		type=str,
		required=True,
		help="Path to the target quantized model directory",
	)
	parser.add_argument(
		"--tensor_name_pattern",
		type=str,
		required=True,
		help="Regular expression used to match source tensor names",
	)
	parser.add_argument(
		"--suffix",
		type=str,
		required=True,
		help="Suffix used for the generated safetensors/index/config files",
	)
	return parser.parse_args()


def replace_tensors(
	source_model_path: str | os.PathLike,
	target_model_path: str | os.PathLike,
	tensor_name_pattern: str,
	suffix: str,
) -> dict[str, str | int]:
	source_path = Path(source_model_path).expanduser().resolve()
	target_path = Path(target_model_path).expanduser().resolve()

	if not target_path.is_dir():
		raise ValueError(f"Target model path must be a directory: {target_path}")

	normalized_suffix = _normalize_suffix(suffix)
	matcher = re.compile(tensor_name_pattern)

	source_root, source_weight_map = _load_source_weight_map(source_path)
	target_index_path = target_path / SAFETENSORS_INDEX_NAME
	if not target_index_path.is_file():
		raise ValueError(f"Target index file not found: {target_index_path}")

	target_index = _load_json(target_index_path)
	target_weight_map = target_index.get("weight_map", {})
	if not isinstance(target_weight_map, dict):
		raise ValueError(
			f"weight_map not found or invalid in target index: {target_index_path}"
		)

	matched_weight_map = {
		tensor_name: file_name
		for tensor_name, file_name in source_weight_map.items()
		if matcher.search(tensor_name)
	}
	if not matched_weight_map:
		raise ValueError(
			f"No tensors matched pattern {tensor_name_pattern!r} in {source_path}"
		)

	output_shard_name = f"{OUTPUT_SHARD_STEM}.{normalized_suffix}.safetensors"
	output_shard_path = target_path / output_shard_name
	output_index_path = target_path / f"{SAFETENSORS_INDEX_NAME}.{normalized_suffix}"
	output_config_path = target_path / f"{CONFIG_NAME}.{normalized_suffix}"

	_assert_new_file_paths(
		[output_shard_path, output_index_path, output_config_path], normalized_suffix
	)

	matched_tensors, added_size = _extract_tensors(source_root, matched_weight_map)
	replaced_names = set(matched_weight_map).intersection(target_weight_map)
	replaced_size = _compute_tensor_sizes(target_path, target_weight_map, replaced_names)

	save_file(matched_tensors, str(output_shard_path))

	updated_index = copy.deepcopy(target_index)
	updated_weight_map = dict(target_weight_map)
	updated_weight_map.update(
		{tensor_name: output_shard_name for tensor_name in matched_weight_map}
	)
	updated_index["weight_map"] = updated_weight_map

	metadata = updated_index.get("metadata", {})
	if not isinstance(metadata, dict):
		metadata = {}
	original_total_size = _coerce_int(metadata.get("total_size", 0), field_name="total_size")
	metadata["total_size"] = original_total_size + added_size - replaced_size
	updated_index["metadata"] = metadata
	_write_json(output_index_path, updated_index)

	target_config_path = target_path / CONFIG_NAME
	if not target_config_path.is_file():
		raise ValueError(f"Target config file not found: {target_config_path}")

	target_config = _load_json(target_config_path)
	source_config = _load_optional_json(source_root / CONFIG_NAME)
	updated_config = _build_updated_config(
		target_config=target_config,
		source_config=source_config,
		matched_tensor_names=sorted(matched_weight_map),
	)
	_write_json(output_config_path, updated_config)

	print(
		"Saved matched tensors to "
		f"{output_shard_path.name}; matched={len(matched_tensors)}, added_size={added_size}, "
		f"replaced_size={replaced_size}"
	)
	print(f"Saved updated index to {output_index_path.name}")
	print(f"Saved updated config to {output_config_path.name}")

	return {
		"output_shard": output_shard_path.name,
		"output_index": output_index_path.name,
		"output_config": output_config_path.name,
		"matched_tensors": len(matched_tensors),
		"added_size": added_size,
		"replaced_size": replaced_size,
	}


def _normalize_suffix(suffix: str) -> str:
	normalized = suffix.strip().lstrip(".")
	if not normalized:
		raise ValueError("suffix must not be empty")

	path_separators = {os.sep}
	if os.altsep is not None:
		path_separators.add(os.altsep)

	if any(separator in normalized for separator in path_separators):
		raise ValueError(f"suffix must not contain path separators: {suffix}")

	return normalized


def _assert_new_file_paths(file_paths: list[Path], suffix: str):
	for file_path in file_paths:
		if file_path.exists():
			raise FileExistsError(
				f"Output file already exists for suffix {suffix}: {file_path}"
			)


def _load_source_weight_map(source_path: Path) -> tuple[Path, dict[str, str]]:
	if source_path.is_dir():
		index_path = source_path / SAFETENSORS_INDEX_NAME
		if index_path.is_file():
			index_data = _load_json(index_path)
			weight_map = index_data.get("weight_map", {})
			if not isinstance(weight_map, dict):
				raise ValueError(
					f"weight_map not found or invalid in source index: {index_path}"
				)
			return source_path, weight_map

		shard_paths = sorted(source_path.glob("*.safetensors"))
		if len(shard_paths) != 1:
			raise ValueError(
				"Source directory must contain model.safetensors.index.json or exactly "
				f"one .safetensors file: {source_path}"
			)
		return source_path, _build_single_shard_weight_map(shard_paths[0])

	if source_path.is_file() and source_path.name.endswith(".safetensors"):
		return source_path.parent, _build_single_shard_weight_map(source_path)

	raise ValueError(
		"Source model path must be a directory or a single .safetensors file: "
		f"{source_path}"
	)


def _build_single_shard_weight_map(shard_path: Path) -> dict[str, str]:
	with safe_open(str(shard_path), framework="pt") as handle:
		return {tensor_name: shard_path.name for tensor_name in handle.keys()}


def _extract_tensors(
	source_root: Path, matched_weight_map: dict[str, str]
) -> tuple[dict[str, object], int]:
	tensors_by_file: dict[str, list[str]] = {}
	for tensor_name, file_name in matched_weight_map.items():
		tensors_by_file.setdefault(file_name, []).append(tensor_name)

	extracted_tensors: dict[str, object] = {}
	total_size = 0
	for file_name, tensor_names in sorted(tensors_by_file.items()):
		shard_path = source_root / file_name
		if not shard_path.is_file():
			raise FileNotFoundError(f"Source shard not found: {shard_path}")

		with safe_open(str(shard_path), framework="pt") as handle:
			for tensor_name in sorted(tensor_names):
				tensor = handle.get_tensor(tensor_name)
				extracted_tensors[tensor_name] = tensor
				total_size += tensor.numel() * tensor.element_size()

	return extracted_tensors, total_size


def _compute_tensor_sizes(
	model_root: Path, weight_map: dict[str, str], tensor_names: set[str]
) -> int:
	if not tensor_names:
		return 0

	header_cache: dict[str, dict[str, dict[str, object]]] = {}
	total_size = 0
	for tensor_name in tensor_names:
		file_name = weight_map[tensor_name]
		if file_name not in header_cache:
			header_cache[file_name] = _load_safetensors_header(model_root / file_name)

		tensor_info = header_cache[file_name].get(tensor_name)
		if tensor_info is None:
			raise ValueError(f"Tensor metadata not found for {tensor_name} in {file_name}")

		total_size += _tensor_info_nbytes(tensor_info, tensor_name=tensor_name)

	return total_size


def _load_safetensors_header(shard_path: Path) -> dict[str, dict[str, object]]:
	if not shard_path.is_file():
		raise FileNotFoundError(f"Shard file not found: {shard_path}")

	with shard_path.open("rb") as file:
		header_size = int.from_bytes(file.read(8), byteorder="little", signed=False)
		header = json.loads(file.read(header_size).decode("utf-8"))

	tensor_info: dict[str, dict[str, object]] = {}
	for tensor_name, info in header.items():
		if tensor_name == "__metadata__" or not isinstance(info, dict):
			continue
		tensor_info[tensor_name] = info
	return tensor_info


def _tensor_info_nbytes(tensor_info: dict[str, object], tensor_name: str) -> int:
	dtype = tensor_info.get("dtype")
	shape = tensor_info.get("shape")
	if not isinstance(dtype, str) or not isinstance(shape, list):
		raise ValueError(f"Invalid tensor metadata for {tensor_name}: {tensor_info}")

	if dtype not in DTYPE_BYTE_SIZES:
		raise ValueError(f"Unsupported dtype {dtype!r} for tensor {tensor_name}")

	return math.prod(shape) * DTYPE_BYTE_SIZES[dtype]


def _build_updated_config(
	target_config: dict[str, object],
	source_config: dict[str, object] | None,
	matched_tensor_names: list[str],
) -> dict[str, object]:
	updated_config = copy.deepcopy(target_config)
	target_qconfig = updated_config.get("quantization_config")
	source_qconfig = None
	if source_config is not None:
		source_qconfig = source_config.get("quantization_config")

	matched_targets = _resolve_matched_targets(matched_tensor_names)

	if isinstance(source_qconfig, dict) and source_qconfig:
		updated_config["quantization_config"] = _merge_quantized_source_config(
			target_qconfig=target_qconfig,
			source_qconfig=source_qconfig,
			matched_tensor_names=matched_tensor_names,
			matched_targets=matched_targets,
		)
	else:
		updated_config["quantization_config"] = _merge_full_precision_source_config(
			target_qconfig=target_qconfig,
			matched_targets=matched_targets,
		)

	return updated_config


def _resolve_matched_targets(matched_tensor_names: list[str]) -> list[str]:
	matched_targets = []
	seen = set()
	for tensor_name in matched_tensor_names:
		target_name = _tensor_to_target_name(tensor_name)
		if target_name not in seen:
			seen.add(target_name)
			matched_targets.append(target_name)
	return matched_targets


def _tensor_to_target_name(tensor_name: str) -> str:
	for suffix in TENSOR_SUFFIXES:
		full_suffix = f".{suffix}"
		if tensor_name.endswith(full_suffix):
			return tensor_name[: -len(full_suffix)]
	return tensor_name


def _merge_full_precision_source_config(
	target_qconfig: object,
	matched_targets: list[str],
) -> dict[str, object]:
	if target_qconfig is None:
		merged_qconfig: dict[str, object] = {}
	elif isinstance(target_qconfig, dict):
		merged_qconfig = copy.deepcopy(target_qconfig)
	else:
		raise ValueError("target config.json quantization_config must be a JSON object")

	ignore = _normalize_list(merged_qconfig.get("ignore"))
	for target_name in matched_targets:
		if target_name not in ignore:
			ignore.append(target_name)
	merged_qconfig["ignore"] = ignore
	return merged_qconfig


def _merge_quantized_source_config(
	target_qconfig: object,
	source_qconfig: dict[str, object],
	matched_tensor_names: list[str],
	matched_targets: list[str],
) -> dict[str, object]:
	if target_qconfig is None:
		merged_qconfig = copy.deepcopy(source_qconfig)
		merged_qconfig["config_groups"] = {}
		merged_qconfig["ignore"] = _normalize_list(merged_qconfig.get("ignore"))
	elif isinstance(target_qconfig, dict):
		merged_qconfig = copy.deepcopy(target_qconfig)
		merged_qconfig["ignore"] = _normalize_list(merged_qconfig.get("ignore"))
	else:
		raise ValueError("target config.json quantization_config must be a JSON object")

	config_groups = merged_qconfig.get("config_groups")
	if config_groups is None:
		config_groups = {}
	elif not isinstance(config_groups, dict):
		raise ValueError("quantization_config.config_groups must be a JSON object")
	else:
		config_groups = copy.deepcopy(config_groups)

	source_groups = source_qconfig.get("config_groups", {})
	if not isinstance(source_groups, dict):
		raise ValueError("source quantization_config.config_groups must be a JSON object")

	selected_groups = _select_source_groups(
		source_groups=source_groups,
		matched_tensor_names=matched_tensor_names,
		matched_targets=matched_targets,
	)

	for group_name, group_data in selected_groups:
		unique_name = _dedupe_group_name(config_groups, preferred_name=group_name)
		config_groups[unique_name] = group_data

	merged_qconfig["config_groups"] = config_groups
	ignore = merged_qconfig.get("ignore", [])
	merged_qconfig["ignore"] = [
		ignore_name for ignore_name in ignore if ignore_name not in matched_targets
	]
	return merged_qconfig


def _select_source_groups(
	source_groups: dict[str, object],
	matched_tensor_names: list[str],
	matched_targets: list[str],
) -> list[tuple[str, dict[str, object]]]:
	selected_groups: list[tuple[str, dict[str, object]]] = []
	for group_name, group_value in source_groups.items():
		if not isinstance(group_value, dict):
			continue

		raw_targets = _normalize_list(group_value.get("targets"))
		exact_targets: list[str] = []
		seen_targets = set()

		for matched_target in matched_targets:
			if any(_target_matches(pattern, matched_target) for pattern in raw_targets):
				if matched_target not in seen_targets:
					seen_targets.add(matched_target)
					exact_targets.append(matched_target)

		for tensor_name in matched_tensor_names:
			if any(_target_matches(pattern, tensor_name) for pattern in raw_targets):
				exact_target = _tensor_to_target_name(tensor_name)
				if exact_target not in seen_targets:
					seen_targets.add(exact_target)
					exact_targets.append(exact_target)

		if not exact_targets and "Linear" in raw_targets and matched_targets:
			exact_targets = list(matched_targets)

		if not exact_targets:
			continue

		new_group = copy.deepcopy(group_value)
		new_group["targets"] = sorted(exact_targets)
		selected_groups.append((group_name, new_group))

	return selected_groups


def _target_matches(pattern: str, value: str) -> bool:
	if pattern == "Linear":
		return True
	if pattern.startswith("re:"):
		return re.match(pattern[3:], value) is not None
	return pattern == value


def _dedupe_group_name(config_groups: dict[str, object], preferred_name: str) -> str:
	if preferred_name not in config_groups:
		return preferred_name

	index = 1
	while f"{preferred_name}_{index}" in config_groups:
		index += 1
	return f"{preferred_name}_{index}"


def _normalize_list(value: object) -> list[str]:
	if value is None:
		return []
	if isinstance(value, str):
		return [value]
	if isinstance(value, list) and all(isinstance(item, str) for item in value):
		return list(value)
	raise ValueError("Expected a string or a list of strings")


def _coerce_int(value: object, field_name: str) -> int:
	if isinstance(value, int):
		return value
	if isinstance(value, str):
		return int(value)
	raise ValueError(f"{field_name} must be an integer-compatible value, got {value!r}")


def _load_json(path: Path) -> dict[str, object]:
	with path.open("r", encoding="utf-8") as file:
		data = json.load(file)
	if not isinstance(data, dict):
		raise ValueError(f"Expected JSON object in {path}")
	return data


def _load_optional_json(path: Path) -> dict[str, object] | None:
	if not path.is_file():
		return None
	return _load_json(path)


def _write_json(path: Path, data: dict[str, object]):
	with path.open("w", encoding="utf-8") as file:
		json.dump(data, file, indent=2, sort_keys=True)
		file.write("\n")


def main():
	args = parse_args()
	replace_tensors(
		source_model_path=args.source_model_path,
		target_model_path=args.target_model_path,
		tensor_name_pattern=args.tensor_name_pattern,
		suffix=args.suffix,
	)


if __name__ == "__main__":
	main()
