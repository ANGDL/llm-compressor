"""MoE calibration preserving V4.1's modality routing and pre-w2 weights."""

from llmcompressor.modeling.deepseekv41.model import DeepseekV41MoE
from llmcompressor.modeling.moe_context import MoECalibrationModule


@MoECalibrationModule.register("DeepseekV41MoE")
class CalibrationDeepseekV41MoE(MoECalibrationModule):
    is_permanent = True
    forward = DeepseekV41MoE.forward

    def __init__(self, original, config, calibrate_all_experts=True):
        super().__init__()
        self.config = original.config
        self.dim = original.dim
        self.n_routed_experts = original.n_routed_experts
        self.gate = original.gate
        self.experts = original.experts
        self.shared_experts = original.shared_experts
        self.calibrate_all_experts = calibrate_all_experts
