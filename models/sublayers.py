import torch
from torch import nn

from models.modules import ScaledDotProductAttention, ScaledDotProductAttentionSimple


class SingleHeadAttention(nn.Module):
    """A class for implementing Single-Head Attention block (for explanation simplicity).

    For each sample this block makes projections for queries, keys and values and a final projection
        of SDPA result as follows:
            Q_projection = Q * W_Q,
            K_projection = K * W_K,
            V_projection = V * W_V,

            SHA(Q, K, V) = SDPA(Q_projection, K_projection, V_projection) * W_O,

            where:
                - Q is of shape (M, d_model), K and V are each of shape (N, d_model)
                - W_Q, W_K, W_V are trainable parameters of shape (d_model, d_k)
                - W_O are trainable parameters of shape (d_k, d_model) that make projection back to d_model dimension
    """

    def __init__(self, config):
        """Layers initialization."""
        super(SingleHeadAttention, self).__init__()
        d_k = config.d_model // config.heads_num
        self.scaled_dot_product_attention = ScaledDotProductAttentionSimple(d_k)

        self.weights_q = nn.Linear(config.d_model, d_k, bias=config.attention_bias)
        self.weights_k = nn.Linear(config.d_model, d_k, bias=config.attention_bias)
        self.weights_v = nn.Linear(config.d_model, d_k, bias=config.attention_bias)
        self.weights_o = nn.Linear(d_k, config.d_model, bias=config.attention_bias)

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.xavier_uniform_(self.weights_q.weight)
        nn.init.xavier_uniform_(self.weights_k.weight)
        nn.init.xavier_uniform_(self.weights_v.weight)
        nn.init.xavier_uniform_(self.weights_o.weight)

        if self.weights_q.bias is not None:  # True means that all weights have bias too
            nn.init.normal_(self.weights_q.bias, std=1e-6)
            nn.init.normal_(self.weights_k.bias, std=1e-6)
            nn.init.normal_(self.weights_v.bias, std=1e-6)
            nn.init.normal_(self.weights_o.bias, std=1e-6)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the Single-Head Attention block.

         Args:
            queries: Query tensor of shape (batch size, M, d_model).
            keys: Key tensor of shape (batch size, N, d_model).
            values: Value tensor of shape (batch size, N, d_model).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, M, d_model)
        """
        queries_projection = self.weights_q(queries)
        keys_projection = self.weights_k(keys)
        values_projection = self.weights_v(values)

        attention = self.scaled_dot_product_attention(queries_projection, keys_projection, values_projection, mask)
        out = self.weights_o(attention)

        return out


class MultiHeadAttentionSimple(nn.Module):
    """A class for implementing Multi-Head Attention block with Single-Head Attention blocks (for explanation simplicity).

    This block simply sums up all Single-Head Attention blocks results.
    """

    def __init__(self, config):
        """Layers initialization."""
        super(MultiHeadAttentionSimple, self).__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.heads_num)])

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the Multi-Head Attention block.

        Args:
            queries: Query tensor of shape (batch size, M, d_model).
            keys: Key tensor of shape (batch size, N, d_model).
            values: Value tensor of shape (batch size, N, d_model).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, M, d_model)
        """
        return torch.stack([head(queries, keys, values, mask) for head in self.heads]).sum(0)


class MultiHeadAttention(nn.Module):
    """A class for implementing Multi-Head Attention block.

    For each sample this block makes projections for queries, keys and values and a final projection
        of concatenated SDPA heads as follows:
            Q_projection = Q * W_Q,
            K_projection = K * W_K,
            V_projection = V * W_V,

            MHA(Q, K, V) = SDPA(Q_projection, K_projection, V_projection) * W_O,

            where:
                - Q is of shape (M, d_model), K and V are of shape (N, d_model)
                - W_Q, W_K, W_V and W_O are trainable parameters of shape (d_model, d_model)

    Note that W_Q, W_K and W_V are stacked parameters for all heads (each of shape (d_model, d_k))
            assuming d_k = d_model // heads_num
    """

    def __init__(self, config):
        """Layers initialization."""
        super(MultiHeadAttention, self).__init__()
        self.config = config

        d_k = config.d_model // config.heads_num
        self.scaled_dot_product_attention = ScaledDotProductAttention(d_k, config.heads_num)

        self.weights_q = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.weights_k = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.weights_v = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)
        self.weights_o = nn.Linear(config.d_model, config.d_model, bias=config.attention_bias)

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.xavier_uniform_(self.weights_q.weight)
        nn.init.xavier_uniform_(self.weights_k.weight)
        nn.init.xavier_uniform_(self.weights_v.weight)
        nn.init.xavier_uniform_(self.weights_o.weight)

        if self.weights_q.bias is not None:  # Means all weights have bias too
            nn.init.normal_(self.weights_q.bias, std=1e-6)
            nn.init.normal_(self.weights_k.bias, std=1e-6)
            nn.init.normal_(self.weights_v.bias, std=1e-6)
            nn.init.normal_(self.weights_o.bias, std=1e-6)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """Forward pass for the Multi-Head Attention block.

        Args:
            queries: Query tensor of shape (batch size, M, d_model).
            keys: Key tensor of shape (batch size, N, d_model).
            values: Value tensor of shape (batch size, N, d_model).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, M, d_model)
        """
        queries_projection = self.weights_q(queries)
        keys_projection = self.weights_k(keys)
        values_projection = self.weights_v(values)

        attention, attention_weights = self.scaled_dot_product_attention(queries_projection, keys_projection,
                                                                         values_projection, mask)

        batch_size = queries.size(0)
        z = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.config.d_model)
        out = self.weights_o(z)

        return out, attention_weights


class FeedForward(nn.Module):
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
        super(FeedForward, self).__init__()
        self.weights_1 = nn.Linear(config.d_model, config.d_ff)
        self.weights_2 = nn.Linear(config.d_ff, config.d_model)
        self.relu = getattr(nn, config.activation)()

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.xavier_uniform_(self.weights_1.weight)
        nn.init.zeros_(self.weights_1.bias)
        nn.init.xavier_uniform_(self.weights_2.weight)
        nn.init.zeros_(self.weights_2.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Feed-Forward layer.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model).

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        return self.weights_2(self.relu(self.weights_1(inputs)))


class LayerNorm(nn.Module):
    """A class for implementing Layer normalization."""

    def __init__(self, config):
        """Parameters initialization."""
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))
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
