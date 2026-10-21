"""Configuration for the original DeepSeek-V4.1 checkpoint layout."""

from copy import deepcopy

from transformers import PretrainedConfig

from llmcompressor.modeling.deepseekv4.config import ModelConfig as V4Config


class DeepseekV41TextConfig(V4Config):
    model_type = "deepseek_v41_text_native"

    def __init__(
        self,
        kv_source_layer_ids=None,
        index_source_layer_ids=None,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        engram_layer_ids=None,
        engram_num_embeddings=None,
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=99092,
        dspark_n_routed_experts=128,
        dspark_num_experts_per_tok=3,
        norm_topk_prob=True,
        gate_temp=1.0,
        **kwargs,
    ):
        kwargs = V4Config._normalize_config_dict(kwargs)
        # Older source configs carried this flag for activation QDQ
        # simulation. The native model no longer performs source
        # quantization in forward, so ignore it when loading old configs.
        kwargs.pop("simulate_source_quantization", None)
        defaults = dict(
            hidden_size=5120,
            moe_intermediate_size=2304,
            num_hidden_layers=40,
            num_hash_layers=0,
            num_nextn_predict_layers=3,
            num_attention_heads=64,
            n_routed_experts=384,
            num_experts_per_tok=6,
            q_lora_rank=1280,
            o_groups=8,
            index_n_heads=32,
            index_topk=512,
            rms_norm_eps=1e-20,
            routed_scaling_factor=1.5,
            dspark_block_size=5,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[37, 38, 39],
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
        )
        defaults.update(kwargs)
        defaults.pop("model_type", None)
        super().__init__(**defaults)
        self.kv_source_layer_ids = list(
            [2, 8, 14, 20] if kv_source_layer_ids is None else kv_source_layer_ids
        )
        self.index_source_layer_ids = list(
            [2, 8, 14, 20, 24, 28, 32, 36]
            if index_source_layer_ids is None
            else index_source_layer_ids
        )
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.engram_layer_ids = list(
            [1, 14] if engram_layer_ids is None else engram_layer_ids
        )
        self.engram_num_embeddings = list(
            [384006168, 384016682]
            if engram_num_embeddings is None
            else engram_num_embeddings
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.gate_temp = gate_temp
        self.engram_checkpoint = None

    def validate_layout(self):
        if len(self.compress_ratios) != self.n_layers + self.num_nextn_predict_layers:
            raise ValueError("compress_ratios must cover backbone and DSpark layers")
        if self.num_hash_layers:
            raise ValueError("DeepSeek-V4.1 does not use hash-routed MoE layers")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("Each Engram layer needs its own embedding table size")
        valid = set(range(self.n_layers))
        for name in (
            "kv_source_layer_ids",
            "index_source_layer_ids",
            "engram_layer_ids",
            "dspark_target_layer_ids",
        ):
            values = getattr(self, name)
            if len(set(values)) != len(values) or not set(values) <= valid:
                raise ValueError(f"Invalid {name}: {values}")
            if values != sorted(values):
                raise ValueError(f"{name} must be in execution order")
        if not set(self.kv_source_layer_ids) <= set(self.index_source_layer_ids):
            raise ValueError("Every KV source must also produce fresh sparse indices")
        source = None
        for index, ratio in enumerate(self.compress_ratios[: self.n_layers]):
            if ratio not in (0, 1, 2):
                raise ValueError("V4.1 supports compression ratios 0, 1 and 2")
            if index in self.kv_source_layer_ids:
                if not ratio:
                    raise ValueError(
                        "A KV source requires a positive compression ratio"
                    )
                source = index
            if ratio and (source is None or self.compress_ratios[source] != ratio):
                raise ValueError(
                    "CSA2 consumers need a preceding KV source of the same ratio"
                )
            if index in self.index_source_layer_ids and not ratio:
                raise ValueError(
                    "An index source requires a positive compression ratio"
                )
        if self.candidate_source_layer_id >= 0:
            if self.candidate_source_layer_id not in self.index_source_layer_ids:
                raise ValueError("Candidate source must be an index source")
            if min(self.candidate_topk_blocks, self.candidate_block_size) <= 0:
                raise ValueError("Candidate pool dimensions must be positive")
        if self.num_nextn_predict_layers:
            if self.dspark_block_size <= 0 or not self.dspark_target_layer_ids:
                raise ValueError("DSpark needs a draft block and target layers")
            if any(self.compress_ratios[self.n_layers :]):
                raise ValueError("DSpark layers use sliding-window attention only")
        for count, topk in (
            (self.n_routed_experts, self.num_experts_per_tok),
            (self.dspark_n_routed_experts, self.dspark_num_experts_per_tok),
        ):
            if not 0 < topk <= count:
                raise ValueError("MoE top-k must be within its expert count")

    def to_dict(self):
        result = super().to_dict()
        result.pop("engram_checkpoint", None)
        return result


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision_native"

    def __init__(
        self,
        num_hidden_layers=32,
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=2816,
        patch_size=14,
        rope_theta=10000,
        downsample_ratio=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio


class ModelConfig(PretrainedConfig):
    model_type = "deepseek_v41_native"
    sub_configs = {
        "text_config": DeepseekV41TextConfig,
        "vision_config": DeepseekV41VisionConfig,
    }

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        quantization_config=None,
        calibrate_dspark=True,
        image_token_id=129264,
        **kwargs,
    ):
        # Recent Transformers exports duplicate derived text fields at the root.
        # Normalize them before PretrainedConfig validates RoPE/layer schemas;
        # the native model uses the authoritative nested text configuration.
        kwargs = V4Config._normalize_config_dict(kwargs)
        kwargs.pop("model_type", None)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("pad_token_id", 2)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        saved_source = kwargs.pop("source_quantization_config", {})
        source_quantization = deepcopy(
            saved_source if quantization_config is None else quantization_config
        )
        compressed = source_quantization.get("quantization_status") == "compressed"
        if compressed:
            kwargs["quantization_config"] = source_quantization
        super().__init__(**kwargs)
        self.text_config = (
            text_config
            if isinstance(text_config, DeepseekV41TextConfig)
            else DeepseekV41TextConfig(**(text_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, DeepseekV41VisionConfig)
            else DeepseekV41VisionConfig(**(vision_config or {}))
        )
        self.source_quantization_config = (
            deepcopy(saved_source) if compressed else source_quantization
        )
        self.calibrate_dspark = calibrate_dspark
        self.image_token_id = image_token_id
        self.text_config.validate_layout()

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = dict(config_dict)
        config_dict["model_type"] = cls.model_type
        return super().from_dict(config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        return cls.from_dict(config_dict, **kwargs)


class DeepseekV41Config(ModelConfig):
    """AutoConfig bridge for original checkpoints, while upstream has no V4.1."""

    model_type = "deepseek_v41"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
