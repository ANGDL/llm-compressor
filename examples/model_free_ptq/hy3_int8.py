import argparse
from pathlib import Path

from llmcompressor import model_free_ptq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the Hy3 model with W8A8")
    parser.add_argument(
        "--model-id",
        default="Tencent-Hunyuan/Hy3",
        help="Hugging Face model id or local model path",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Directory to save quantized weights. Defaults to <model-name>-W8A8-INT8",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of worker threads to process safetensors files",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to accelerate quantization with",
    )
    parser.add_argument(
        "--strict-symmetric",
        action="store_true",
        help="Use the strict symmetric INT8 range [-127, 127] for weights",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = args.model_id
    save_dir = args.save_dir or f"{Path(model_id.rstrip('/')).name}-W8A8-INT8"

    # Apply W8A8 to the model
    # Once quantized, the model is saved
    # using compressed-tensors to the save_dir.
    model_free_ptq(
        model_stub=model_id,
        save_directory=save_dir,
        scheme="W8A8",
        ignore=[
            "lm_head",
            "re:.*eh_proj$",
            "re:.*mlp.gate$",
            "re:.*mlp.router.gate$",
            "re:.*mlp.shared_expert_gate.*",
            "re:.*norm.*",
            "re:.*embed_tokens.*",
        ],
        max_workers=args.max_workers,
        device=args.device,
        strict_symmetric=args.strict_symmetric,
    )


if __name__ == "__main__":
    main()
