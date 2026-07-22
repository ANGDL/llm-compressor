from typing import Literal, Optional

from transformers.configuration_utils import PretrainedConfig


# NOTE: We intentionally use a distinct model_type from upstream HF transformers
# (which ships its own `deepseek_v4` implementation starting in v5.12). This
# implementation targets the original DeepSeek-V4 reference checkpoint layout
# (with MTP / Indexer / hc_split kernels), and must coexist with HF's. Both
# implementations therefore live under different model_type strings so that
# AutoConfig / AutoModelForCausalLM dispatch is unambiguous.
class ModelConfig(PretrainedConfig):
    model_type = "deepseek_v4_native"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Tolerate raw upstream checkpoints whose config.json carries
        # `model_type: "deepseek_v4"`. When they are loaded explicitly through
        # this class (the LC native impl) we silently treat them as our own
        # model_type. This does not affect HF's dispatch path: that path uses
        # AutoConfig, which keeps returning HF's DeepseekV4Config for
        # "deepseek_v4".
        if (
            isinstance(config_dict, dict)
            and config_dict.get("model_type") == "deepseek_v4"
        ):
            config_dict = dict(config_dict)
            config_dict["model_type"] = cls.model_type
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        max_batch_size: int = 8,
        max_seq_len: Optional[int] = None,
        dtype: Literal["bf16", "fp8"] = "bf16",
        scale_fmt: Optional[str] = None,
        expert_dtype: Optional[Literal["fp4", "fp8"]] = None,
        scale_dtype: Literal["fp32", "fp8"] = "fp32",
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        moe_intermediate_size: int = 3072,
        num_hidden_layers: int = 61,
        num_hash_layers: int = 3,
        num_nextn_predict_layers: int = 1,
        num_attention_heads: int = 128,
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus",
        routed_scaling_factor: float = 2.5,
        swiglu_limit: float = 10.0,
        q_lora_rank: int = 1536,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        o_groups: int = 16,
        o_lora_rank: int = 1024,
        sliding_window: int = 128,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 1024,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        torch_dtype: str = "bfloat16",
        compress_rope_theta: float = 160000.0,
        compress_ratios: Optional[list[int]] = None,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ):
        if max_seq_len is None:
            max_seq_len = max_position_embeddings

        rope_scaling = rope_scaling or {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "type": "yarn",
        }
        compress_ratios = compress_ratios or [0] * (
            num_hidden_layers + num_nextn_predict_layers
        )

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.scale_fmt = scale_fmt
        self.expert_dtype = expert_dtype
        self.scale_dtype = scale_dtype
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hash_layers = num_hash_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_limit = swiglu_limit
        self.q_lora_rank = q_lora_rank
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype
        self.compress_rope_theta = compress_rope_theta
        self.compress_ratios = compress_ratios
        # Raw DeepSeek FP4/FP8 metadata must not trigger Transformers' generic
        # pre-quantized loader. Checkpoints produced by llm-compressor, however,
        # need their compressed config preserved so weight_scale/zero_point
        # tensors are initialized before loading.
        quantization_config = dict(quantization_config or {})
        is_compressed = quantization_config.get("quantization_status") == "compressed"
        self.source_quantization_config = (
            {} if is_compressed else quantization_config
        )
        if is_compressed:
            kwargs["quantization_config"] = quantization_config

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def dim(self) -> int:
        return self.hidden_size

    @property
    def moe_inter_dim(self) -> int:
        return self.moe_intermediate_size

    @property
    def n_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def n_hash_layers(self) -> int:
        return self.num_hash_layers

    @property
    def n_mtp_layers(self) -> int:
        return self.num_nextn_predict_layers

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def n_activated_experts(self) -> int:
        return self.num_experts_per_tok

    @property
    def score_func(self) -> str:
        return self.scoring_func

    @property
    def route_scale(self) -> float:
        return self.routed_scaling_factor

    @property
    def rope_head_dim(self) -> int:
        return self.qk_rope_head_dim

    @property
    def norm_eps(self) -> float:
        return self.rms_norm_eps

    @property
    def window_size(self) -> int:
        return self.sliding_window

    @property
    def original_seq_len(self) -> int:
        return int(self.rope_scaling.get("original_max_position_embeddings", 0))

    @property
    def rope_factor(self) -> float:
        return float(self.rope_scaling.get("factor", 1.0))

    @property
    def beta_fast(self) -> int:
        return int(self.rope_scaling.get("beta_fast", 32))

    @property
    def beta_slow(self) -> int:
        return int(self.rope_scaling.get("beta_slow", 1))
