import torch
from torch import nn


class Embedding(nn.Module):
    """A class for the embedding layer.

    These embeddings are often used to represent textual data inputs.
    """

    def __init__(self, vocabulary_size: int, d_model: int):
        """Embedding layer initialization.

        Args:
            vocabulary_size: data vocabulary size (i.e. the number of embeddings to store)
            d_model: embedding dimension
        """
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.Embedding(vocabulary_size, d_model, padding_idx=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Embedding layer.

        Args:
            inputs: tensor of shape (batch size, sequence length) representing raw inputs data

        Returns:
            Tensor of shape (batch_size, sequence length, d_model) representing the inputs embeddings
        """
        embeddings = self.embeddings(inputs)
        return embeddings
