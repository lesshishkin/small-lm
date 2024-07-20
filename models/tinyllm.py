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
        mask,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class TinyLLM(nn.Module):
    """TinyLLM -- tiny but large. Like LLaMA3"""

    def __init__(self, config, vocab_size):
        super(TinyLLM, self).__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, config.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, self.vocab_size, bias=False)

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

    def forward(self, tokens: torch.Tensor, start_pos: int, mask):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()

        return output