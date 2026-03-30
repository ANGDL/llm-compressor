import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


TensorMeta = Tuple[str, List[int]]

DTYPE_ALIASES = {
	"float16": "F16",
	"half": "F16",
	"bfloat16": "BF16",
	"float32": "F32",
	"float": "F32",
	"float64": "F64",
	"double": "F64",
	"int8": "I8",
	"uint8": "U8",
	"int16": "I16",
	"uint16": "U16",
	"int32": "I32",
	"uint32": "U32",
	"int64": "I64",
	"uint64": "U64",
	"bool": "BOOL",
}


def normalize_dtype(dtype: str) -> str:
	normalized = dtype.strip()
	if not normalized:
		return normalized

	lowered = normalized.lower()
	if lowered in DTYPE_ALIASES:
		return DTYPE_ALIASES[lowered]

	return normalized.upper()


def should_skip_tensor_by_layer_id(tensor_name: str, excluded_layer_ids: set[int]) -> bool:
	if not excluded_layer_ids:
		return False

	# Match patterns like: model.layers.10.*, transformer.h.12.*, blocks.3.*
	match = re.search(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)(?:\.|$)", tensor_name)
	if match is None:
		return False

	return int(match.group(1)) in excluded_layer_ids


def resolve_index_path(input_path: str) -> Path:
	path = Path(input_path).expanduser().resolve()

	if path.is_file():
		if path.name != "model.safetensors.index.json":
			raise ValueError(f"Index file name must be model.safetensors.index.json: {path}")
		return path

	if path.is_dir():
		index_path = path / "model.safetensors.index.json"
		if not index_path.is_file():
			raise FileNotFoundError(f"Index file not found: {index_path}")
		return index_path

	raise FileNotFoundError(f"Input path does not exist: {path}")


def load_weight_map(index_path: Path) -> Dict[str, str]:
	with index_path.open("r", encoding="utf-8") as f:
		index_data = json.load(f)

	weight_map = index_data.get("weight_map")
	if not isinstance(weight_map, dict):
		raise ValueError(f"Invalid index file (missing dict weight_map): {index_path}")

	return weight_map


def load_safetensors_header(shard_path: Path) -> Dict[str, TensorMeta]:
	with shard_path.open("rb") as f:
		header_size = int.from_bytes(f.read(8), byteorder="little", signed=False)
		header = json.loads(f.read(header_size).decode("utf-8"))

	tensor_meta: Dict[str, TensorMeta] = {}
	for tensor_name, info in header.items():
		if tensor_name == "__metadata__":
			continue
		if not isinstance(info, dict):
			continue

		dtype = info.get("dtype")
		shape = info.get("shape")
		if isinstance(dtype, str) and isinstance(shape, list):
			tensor_meta[tensor_name] = (dtype, shape)

	return tensor_meta


def get_tensor_meta(
	tensor_name: str,
	shard_name: str,
	root_dir: Path,
	shard_cache: Dict[str, Dict[str, TensorMeta]],
) -> TensorMeta:
	if shard_name not in shard_cache:
		shard_path = root_dir / shard_name
		if not shard_path.is_file():
			raise FileNotFoundError(f"Shard file not found: {shard_path}")
		shard_cache[shard_name] = load_safetensors_header(shard_path)

	shard_meta = shard_cache[shard_name]
	if tensor_name not in shard_meta:
		raise KeyError(f"Tensor {tensor_name} not found in shard {shard_name}")

	return shard_meta[tensor_name]


def compare_tensors(path_a: str, path_b: str, dtypes: List[str], excluded_layer_ids: List[int]) -> int:
	index_a = resolve_index_path(path_a)
	index_b = resolve_index_path(path_b)

	root_a = index_a.parent
	root_b = index_b.parent

	map_a = load_weight_map(index_a)
	map_b = load_weight_map(index_b)

	tensors_a = set(map_a.keys())
	tensors_b = set(map_b.keys())
	common_tensors = sorted(tensors_a & tensors_b)
	dtype_filter = {normalize_dtype(dtype) for dtype in dtypes if dtype.strip()}
	excluded_layer_set = set(excluded_layer_ids)

	print(f"A index: {index_a}")
	print(f"B index: {index_b}")
	print(f"A tensors: {len(tensors_a)}")
	print(f"B tensors: {len(tensors_b)}")
	print(f"Common tensors: {len(common_tensors)}")
	if dtype_filter:
		print(f"Dtype filter: {sorted(dtype_filter)}")
	if excluded_layer_set:
		print(f"Excluded layer ids: {sorted(excluded_layer_set)}")

	if not common_tensors:
		print("No common tensor names found.")
		return 0

	shard_cache_a: Dict[str, Dict[str, TensorMeta]] = {}
	shard_cache_b: Dict[str, Dict[str, TensorMeta]] = {}
	mismatch_count = 0
	compared_count = 0
	skipped_by_dtype = 0
	skipped_by_layer_id = 0

	for tensor_name in common_tensors:
		if should_skip_tensor_by_layer_id(tensor_name, excluded_layer_set):
			skipped_by_layer_id += 1
			continue

		shard_a = map_a[tensor_name]
		shard_b = map_b[tensor_name]

		try:
			dtype_a, shape_a = get_tensor_meta(
				tensor_name=tensor_name,
				shard_name=shard_a,
				root_dir=root_a,
				shard_cache=shard_cache_a,
			)
			dtype_b, shape_b = get_tensor_meta(
				tensor_name=tensor_name,
				shard_name=shard_b,
				root_dir=root_b,
				shard_cache=shard_cache_b,
			)
		except (FileNotFoundError, KeyError, ValueError) as e:
			mismatch_count += 1
			print(f"\n[ERROR] tensor={tensor_name}")
			print(f"  reason: {e}")
			print(f"  A shard: {shard_a}")
			print(f"  B shard: {shard_b}")
			continue

		normalized_dtype_a = normalize_dtype(dtype_a)
		normalized_dtype_b = normalize_dtype(dtype_b)
		if dtype_filter and normalized_dtype_a not in dtype_filter and normalized_dtype_b not in dtype_filter:
			skipped_by_dtype += 1
			continue

		compared_count += 1

		if dtype_a != dtype_b or shape_a != shape_b:
			mismatch_count += 1
			print(f"\n[MISMATCH] tensor={tensor_name}")
			print(f"  A: dtype={dtype_a}, shape={shape_a}, shard={shard_a}")
			print(f"  B: dtype={dtype_b}, shape={shape_b}, shard={shard_b}")

	print("\n==============================")
	print(f"Compared common tensors: {compared_count}")
	if dtype_filter:
		print(f"Skipped by dtype filter: {skipped_by_dtype}")
	if excluded_layer_set:
		print(f"Skipped by layer_id filter: {skipped_by_layer_id}")
	print(f"Mismatched tensors: {mismatch_count}")
	print("==============================")

	return mismatch_count


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Compare shape and dtype of tensors with same names between two "
			"model.safetensors.index.json inputs"
		)
	)
	parser.add_argument("path_a", help="Path to model directory or model.safetensors.index.json")
	parser.add_argument("path_b", help="Path to model directory or model.safetensors.index.json")
	parser.add_argument(
		"--dtype",
		nargs="+",
		default=[],
		help=(
			"Only compare tensors whose dtype (from either model) matches provided values. "
			"Supports safetensors dtypes like F16/BF16/I8 and aliases like float16/bfloat16/int8"
		),
	)
	parser.add_argument(
		"--exclude-layer-id",
		nargs="+",
		type=int,
		default=[],
		help="Skip comparing tensors whose layer id matches any provided value",
	)
	args = parser.parse_args()

	mismatch_count = compare_tensors(args.path_a, args.path_b, args.dtype, args.exclude_layer_id)
	raise SystemExit(1 if mismatch_count > 0 else 0)


if __name__ == "__main__":
	main()
