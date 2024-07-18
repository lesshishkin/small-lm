from math import sqrt

import torch
from torch import nn


class ScaledDotProductAttentionSimple(nn.Module):
    """A class for implementing Scaled Dot-Product Attention block for one head (for explanation simplicity).

   SDPA performs the following steps for each sample:
        1. Takes projected Q (queries) of shape (M, d_k), K (keys) and V (values) each of shape (N, d_k) as inputs
        2. Calculates scaled attention scores as matrix multiplication of queries and transposed keys:
                attention_scores = Q * K^T / sqrt(d_k),
        3. Applies Softmax to the scaled attention scores row-wise (i.e. by the last dimension):
                weights = Softmax(attention_scores),

                where:
                    - attention_scores is a matrix of shape (M, N)
        4. Gets the whole block output by applying computed weights to the values projection as follows:
                SDPA(Q, K, V) = weights * V

    This block performs SDPA for one head resulting in a matrix of shape (M, d_k) wrt the one sample.
    """

    def __init__(self, d_k: int = 64):
        """Layers initialization."""
        super(ScaledDotProductAttentionSimple, self).__init__()
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for the Scaled Dot-Product Attention block.

        Performs self-attention mechanism on queries, keys and values tensors.

        Args:
            queries: Query tensor of shape (batch size, M, d_k).
            keys: Key tensor of shape (batch size, N, d_k).
            values: Value tensor of shape (batch size, N, d_k).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, M, d_k) representing the values weighted with attention.
        """
        attention_scores = queries @ keys.transpose(-2, -1) / sqrt(self.d_k)
        if mask is not None:
            attention_scores += (mask * torch.tensor(-1e9))

        weights = self.softmax(attention_scores)
        output = weights @ values

        return output


class ScaledDotProductAttention(nn.Module):
    """A class for implementing Scaled Dot-Product Attention block for all heads at once.

    SDPA performs the following steps for each sample and each head:
        1. Takes projected Q (queries) of shape (M, d_model), K (keys) and V (values) each of shape (N, d_model) as inputs
        2. Calculates scaled attention scores as matrix multiplication of queries and transposed keys:
                attention_scores = Q * K^T / sqrt(d_k),
        3. Applies Softmax to the scaled attention scores row-wise (i.e. by the last dimension):
                weights = Softmax(attention_scores),

                where:
                    - attention_scores is a matrix of shape (M, N)
        4. Gets the whole block output by applying computed weights to the values projection as follows:
                SDPA(Q, K, V) = weights * V

    This block performs SDPA for all heads at once resulting in a tensor of shape (heads num, M, d_k)
            wrt the one sample.
    """

    def __init__(self, d_k: int = 64, heads_num: int = 8):
        """Layers initialization."""
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.heads_num = heads_num
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                mask: torch.Tensor = None) -> (torch.Tensor,torch.Tensor):
        """Forward pass for the Scaled Dot-Product Attention block.

        Performs self-attention mechanism on queries, keys and values tensors for all heads at once.

        Args:
            queries: Query tensor of shape (batch size, M, d_model).
            keys: Key tensor of shape (batch size, N, d_model).
            values: Value tensor of shape (batch size, N, d_model).
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, heads num, M, d_k) representing the values weighted with attention.
        """
        batch_size = queries.size(0)
        keys_split = keys.view(batch_size, -1, self.heads_num, self.d_k).transpose(1, 2)
        queries_split = queries.view(batch_size, -1, self.heads_num, self.d_k).transpose(1, 2)
        values_split = values.view(batch_size, -1, self.heads_num, self.d_k).transpose(1, 2)

        attention_scores = (queries_split @ keys_split.transpose(-2, -1)) / sqrt(self.d_k)
        if mask is not None:
            attention_scores += (mask * torch.tensor(-1e9))

        weights = self.softmax(attention_scores)
        output = weights @ values_split

        return output, weights


class PositionalEncoding(nn.Module):
    """A class for implementing Positional Encoding block.

    For each sample this block will add positional encoding (PE) matrix to the inputs. PE is defined as follows:
            PE(pos, 2 * i) = sin(pos / (10000 ^ (2 * i / d_model)))
            PE(pos, 2 * i + 1) = cos(pos / (10000 ^ (2 * i / d_model))),

            where:
                - pos is input sequence position number (from 0 to max sequence length - 1)
                - i is a counter which is used to represent even and odd embedding positions for each sequence element
                        (from 0 to d_model - 1 // 2)

    This block adds PE to each sample and applies Dropout to the resulting sum.
    """

    def __init__(self, max_sequence_length: int, d_model: int, dropout_rate: float):
        """Positional Encoding initialization.

        Args:
            max_sequence_length: maximum sequence length to expect
            d_model: embeddings dimension
            dropout_rate: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        positional_encoding = self.get_positional_encoding(max_sequence_length, d_model)
        self.register_buffer('positional_encoding', positional_encoding)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def get_positional_encoding(max_sequence_length: int, d_model: int) -> torch.Tensor:
        """Constructs PE matrix."""
        positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        scaling_factor = torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2) / d_model)

        positional_encoding = torch.zeros(max_sequence_length, d_model)
        positional_encoding[:, 0::2] = torch.sin(positions / scaling_factor)
        positional_encoding[:, 1::2] = torch.cos(positions / scaling_factor)
        return positional_encoding.unsqueeze(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Positional Encoding block.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model)

        Returns:
            Tensor of shape (batch size, sequence length, d_model) representing "positional encoded" inputs
        """
        return self.dropout(inputs + self.positional_encoding[:, :inputs.size(1)])
