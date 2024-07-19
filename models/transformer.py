import torch
from torch import nn

from models.embeddings import Embedding
from models.layers import EncoderLayer, DecoderLayer, EncoderPreNormLayer, DecoderPreNormLayer
from models.modules import PositionalEncoding


class Encoder(nn.Module):
    """A class for implementing Transformer Encoder."""

    def __init__(self, config):
        """Initializes the Encoder consisting of multiple EncoderLayer layers."""
        super(Encoder, self).__init__()
        encoder_layer = EncoderPreNormLayer if config.pre_normalization else EncoderLayer
        self.layers = torch.nn.ModuleList([encoder_layer(config) for _ in range(config.layers_num)])
        self.config = config

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """Forward pass for the Encoder.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model)
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model).
        """
        outputs = inputs
        all_attention_weights = []
        for layer in self.layers:
            outputs, attention_weights = layer(outputs, mask)
            all_attention_weights.append(attention_weights)
        return outputs, all_attention_weights


class Decoder(nn.Module):
    """A class for implementing Transformer Encoder."""

    def __init__(self, config):
        """Initializes the Encoder consisting of multiple EncoderLayer layers."""
        super(Decoder, self).__init__()
        decoder_layer = DecoderPreNormLayer if config.pre_normalization else DecoderLayer
        self.layers = torch.nn.ModuleList([decoder_layer(config) for _ in range(config.layers_num)])
        self.config = config

    def forward(self, inputs: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward pass for the Encoder.

        Args:
            inputs: tensor of shape (batch size, target sequence length, d_model)
            encoder_outputs: tensor of shape (batch size, source sequence length, d_model)
            src_mask: source sequence mask with ones at positions that should be masked out
            tgt_mask: target sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model).
        """
        outputs = inputs
        all_self_attention_weights = []
        all_encoder_attention_weights = []
        for layer in self.layers:
            outputs, self_attention_weights, encoder_attention_weights = layer(outputs, encoder_outputs, src_mask,
                                                                               tgt_mask)
            all_self_attention_weights.append(self_attention_weights)
            all_encoder_attention_weights.append(encoder_attention_weights)
        return outputs, all_self_attention_weights, all_encoder_attention_weights


class TransformerOutput(nn.Module):
    """Transformer output layer."""

    def __init__(self, d_model, vocabulary_size):
        super(TransformerOutput, self).__init__()
        self.output_feed_forward = nn.Linear(d_model, vocabulary_size - 1)

        self._init_weights()

    def _init_weights(self):
        """Weights initialization."""
        nn.init.xavier_uniform_(self.output_feed_forward.weight)
        nn.init.zeros_(self.output_feed_forward.bias)

    def forward(self, inputs):
        """Forward pass for Transformer output layer."""
        return self.output_feed_forward(inputs)


class Transformer(nn.Module):
    """Transformer model.

    A class for implementing Transformer model from 'Attention Is All You Need' (https://arxiv.org/pdf/1706.03762).
    """

    def __init__(self, config, encoder_vocabulary_size, decoder_vocabulary_size):
        """Layers initialization."""
        super(Transformer, self).__init__()
        self.embeddings_encoder = Embedding(encoder_vocabulary_size, config.model.d_model)
        self.embeddings_decoder = Embedding(decoder_vocabulary_size, config.model.d_model)

        self.positional_encoding = PositionalEncoding(
            config.model.max_sequence_length, config.model.d_model, config.model.dropout_rate
        )
        self.encoder = Encoder(config.model)
        self.decoder = Decoder(config.model)
        self.output = TransformerOutput(config.model.d_model, decoder_vocabulary_size)

    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """Transformer forward pass.

        Args:
            encoder_inputs: tensor of shape (batch size, sequence length)
            decoder_inputs: tensor of shape (batch size, sequence length)
            src_mask: source sequence mask with ones at positions that should be masked out
            tgt_mask: target sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        encoder_inputs_embeddings = self.positional_encoding(self.embeddings_encoder(encoder_inputs))
        encoder_output, encoder_attention_weights = self.encoder(encoder_inputs_embeddings, src_mask)

        decoder_inputs_embeddings = self.positional_encoding(self.embeddings_decoder(decoder_inputs))
        decoder_output, decoder_self_attention_weights, decoder_encoder_attention_weights = self.decoder(
            decoder_inputs_embeddings, encoder_output, src_mask, tgt_mask)

        output = self.output(decoder_output)
        return output, encoder_attention_weights, decoder_self_attention_weights, decoder_encoder_attention_weights


class TinyLLM(nn.Module):
    """TinyLLM -- tiny but large"""

    def __init__(self):
        super(TinyLLM, self).__init__()
        # TODO