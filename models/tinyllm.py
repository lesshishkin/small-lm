import torch
from torch import nn
from models.attention import Attention
from models.sublayers import RMSNorm, FeedForward
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerBlock(nn.Module):
    # TODO
    def __init__(self, layer_id: int, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            multiple_of=config.multiple_of,     # todo разобраться с этим
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TinyLLM(nn.Module):
    """TinyLLM -- tiny but large. Like LLaMA3"""

    def __init__(self, config):
        super(TinyLLM, self).__init__()
        # TODO

        self.freqs_cis = self.precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta,
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, device, theta: float = 10000.0):
        """For RoPE. Заранее считаем комплексные множители"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis