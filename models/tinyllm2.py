import torch
from torch import nn
from models.attention import Attention2
from models.sublayers import LayerNorm, FeedForward2
import torch.nn.functional as F
from typing import Optional, Tuple


class TransformerBlock2(nn.Module):
    def __init__(self, layer_id: int, config):
        super(TransformerBlock2, self).__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.attention = Attention2(config)
        self.feed_forward = FeedForward2(config)
        self.layer_id = layer_id
        self.attention_norm = LayerNorm(config)
        self.ffn_norm = LayerNorm(config)
        self.ff_dropout = nn.Dropout(config.dropout_rate)
        self.attention_dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask,
    ):
        h = x + self.attention_dropout(self.attention(self.attention_norm(x), freqs_cis, mask))
        out = h + self.ff_dropout(self.feed_forward(self.ffn_norm(h)))

        return out


class TinyLLM2(nn.Module):
    """TinyLLM2 -- my second try"""
    def __init__(self, config, vocab_size, device):
        super(TinyLLM2, self).__init__()

        self.config = config
        self.vocab_size = vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, config.dim, padding_idx=0, device=device)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock2(layer_id, config))

        self.norm = LayerNorm(config)
        self.output = nn.Linear(config.dim, self.vocab_size, bias=False)

        self.freqs_cis = self.precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            device,
            config.rope_theta,
        )

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, device, theta: float = 10000.0):
        """For RoPE. Заранее считаем комплексные множители"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
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
            h = layer(h, freqs_cis, mask)

        h = self.norm(h)

        return self.output(h)
