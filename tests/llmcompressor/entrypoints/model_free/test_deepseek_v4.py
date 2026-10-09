import torch

from examples.model_free_ptq.dsv4_w4a8 import (
    DeepSeekV4FlashConverter,
)


def test_dequantizes_packed_fp4_weight_and_removes_source_scale():
    converter = DeepSeekV4FlashConverter(
        [r"re:.*\.ffn\.experts\.\d+\.w[123]$"],
        source_kind="fp4",
        fp4_block_size=2,
        dtype=torch.float32,
    )
    tensors = {
        "layers.0.ffn.experts.0.w1.weight": torch.tensor(
            [[0x12, 0x34], [0x98, 0x76]], dtype=torch.uint8
        ).view(torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.full(
            (2, 2), 127, dtype=torch.uint8
        ),
    }

    assert converter.process_non_quantized(tensors, "cpu") is tensors
    assert "layers.0.ffn.experts.0.w1.scale" in tensors
    converter.validate(tensors)
    converted = converter.process(tensors)

    assert torch.equal(
        converted["layers.0.ffn.experts.0.w1.weight"],
        torch.tensor(
            [[1.0, 0.5, 2.0, 1.5], [0.0, -0.5, 4.0, 6.0]],
            dtype=torch.float32,
        ),
    )
    assert "layers.0.ffn.experts.0.w1.scale" not in converted


def test_dequantizes_fp8_block_and_bf16_target():
    converter = DeepSeekV4FlashConverter(
        [r"re:.*\.attn\.wq_a$"],
        source_kind="fp8",
        bf16_targets=[r"re:.*\.attn\.wo_a$"],
        fp8_block_size=(2, 2),
    )
    scale = torch.tensor([[127, 128]], dtype=torch.uint8)
    tensors = {
        "layers.0.attn.wq_a.weight": torch.ones((2, 4), dtype=torch.float32),
        "layers.0.attn.wq_a.scale": scale.clone(),
        "layers.0.attn.wo_a.weight": torch.ones((2, 4), dtype=torch.float32),
        "layers.0.attn.wo_a.scale": scale.clone(),
    }

    converted = converter.process(tensors)

    expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]]).expand(2, -1)
    assert torch.equal(converted["layers.0.attn.wq_a.weight"].float(), expected)
    assert converted["layers.0.attn.wo_a.weight"].dtype == torch.bfloat16
    assert not any(name.endswith(".scale") for name in converted)


def test_reports_scale_dependency_for_targeted_weights_only():
    converter = DeepSeekV4FlashConverter(
        [r"re:.*\.attn\.wq_a$"], source_kind="fp8"
    )

    assert converter.get_dependencies("layers.0.attn.wq_a.weight") == {
        "layers.0.attn.wq_a.scale"
    }
    assert converter.get_dependencies("layers.0.attn.wq_b.weight") == set()
    assert converter.get_dependencies("layers.0.attn.wq_a.bias") == set()

    mtp_converter = DeepSeekV4FlashConverter(
        [r"re:.*mtp\.\d+\.(e_proj|h_proj|main_proj)$"], source_kind="fp8"
    )
    assert mtp_converter.get_dependencies("mtp.0.e_proj.weight") == {
        "mtp.0.e_proj.scale"
    }
