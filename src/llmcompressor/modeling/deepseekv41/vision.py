"""Vision branch preserving the original V4.1 parameter names."""

import torch
from torch import nn
from torch.nn import functional as F

from llmcompressor.modeling.deepseekv4.model import RMSNorm


def _rotary(x, positions, theta):
    dim = x.shape[-1] // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = (positions[..., None] * inv_freq).flatten(-2).unsqueeze(1)
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat(
        (
            first * angles.cos() - second * angles.sin(),
            second * angles.cos() + first * angles.sin(),
        ),
        dim=-1,
    ).to(x.dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches):
        return self.proj(patches.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.theta = config.rope_theta
        self.wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, positions):
        q, k, v = (
            value.unflatten(-1, (self.heads, -1))
            for value in self.wqkv(x).chunk(3, dim=-1)
        )
        q, k = _rotary(q, positions, self.theta), _rotary(k, positions, self.theta)
        result = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(result.transpose(0, 1).flatten(1))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = (
            RMSNorm(config.hidden_size),
            RMSNorm(config.hidden_size),
        )
        self.attn, self.mlp = VisionAttention(config), VisionMLP(config)

    def forward(self, x, positions):
        x = x + self.attn(self.norm1(x), positions)
        return x + self.mlp(self.norm2(x))


class Vision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [DeepseekV41VisionBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, patches, height, width):
        x = self.patch_embed(patches)
        rows, columns = torch.meshgrid(
            torch.arange(height, device=x.device),
            torch.arange(width, device=x.device),
            indexing="ij",
        )
        positions = torch.stack((rows, columns), dim=-1).flatten(0, 1).float()
        for block in self.blocks:
            x = block(x, positions)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.ratio = vision_config.downsample_ratio
        self.w1 = nn.Linear(vision_config.hidden_size * self.ratio**2, text_config.dim)
        self.w2 = nn.Linear(text_config.dim, text_config.dim)

    def forward(self, x, height, width):
        ratio = self.ratio
        x = x.view(height, width, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -width % ratio, 0, -height % ratio))
        x = F.unfold(x.unsqueeze(0), ratio, stride=ratio)
        return self.w2(F.gelu(self.w1(x.squeeze(0).transpose(0, 1))))
