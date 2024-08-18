import torch
from torch import nn
from typing import Optional, Tuple
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    """RMSNorm like LLaMA3"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """FF block like LLaMA3. 3 layers"""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FeedForward2(nn.Module):
    """A class for implementing Feed Forward layer.

    For each sample this block makes two linear transformations defined as follows:
            FF(x) = max(0, x * W_1 + b_1) * W_2 + b_2,

            where:
                - x is an input tensor of shape (sequence length, d_model)
                - W_1 and b_1 are trainable parameters of first Linear layer
                        with output features num = d_ff and input features num = d_model
                - W_2 and b_2 are trainable parameters of first Linear layer
                        with output features num = d_model and input features num = d_ff
    """

    def __init__(self, config):
        """Layers initialization."""
        super(FeedForward2, self).__init__()
        self.weights_1 = nn.Linear(config.dim, config.d_ff, bias=False)
        self.weights_2 = nn.Linear(config.d_ff, config.dim, bias=False)
        self.gelu = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.xavier_uniform_(self.weights_1.weight)
        # nn.init.zeros_(self.weights_1.bias)
        nn.init.xavier_uniform_(self.weights_2.weight)
        # nn.init.zeros_(self.weights_2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Feed-Forward layer.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model).

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        return self.weights_2(self.gelu(self.weights_1(inputs)))


class LayerNorm(nn.Module):
    """A class for implementing Layer normalization."""

    def __init__(self, config):
        """Parameters initialization."""
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(config.dim))
        self.beta = nn.Parameter(torch.zeros(config.dim))
        self.eps = config.eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Layer normalization block.

        Args:
            inputs: tensor of shape (batch_size, sequence length, d_model)

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        mean = inputs.mean(-1, keepdim=True)
        var = inputs.var(-1, keepdim=True, unbiased=False)
        normalized_inputs = (inputs - mean) / torch.sqrt(var + self.eps)

        return self.gamma * normalized_inputs + self.beta
