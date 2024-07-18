import torch
from torch import nn

from models.sublayers import MultiHeadAttention, MultiHeadAttentionSimple, LayerNorm, FeedForward


class EncoderLayer(nn.Module):
    """A class for implementing a single Encoder layer."""

    def __init__(self, config):
        """Layers initialization."""
        # TODO: change MultiHeadAttention to MultiHeadAttentionSimple if simple stack is implemented!
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.multi_head_attention_dropout = nn.Dropout(config.dropout_rate)
        self.multi_head_attention_normalization = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)
        self.feed_forward_normalization = LayerNorm(config)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """Forward pass for a single layer of the Encoder.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model)
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        self_attention, attention_weights = self.multi_head_attention(inputs, inputs, inputs, mask)
        x = self.multi_head_attention_dropout(self_attention) + inputs
        x = self.multi_head_attention_normalization(x)

        ff_output = self.feed_forward(x)
        output = self.feed_forward_normalization(self.feed_forward_dropout(ff_output) + x)
        return output, attention_weights


class EncoderPreNormLayer(nn.Module):
    """A class for implementing a single Encoder layer with Pre-LN."""

    def __init__(self, config):
        """Layers initialization."""
        # TODO: change MultiHeadAttention to MultiHeadAttentionSimple if simple stack is implemented!
        super(EncoderPreNormLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.multi_head_attention_dropout = nn.Dropout(config.dropout_rate)
        self.multi_head_attention_normalization = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)
        self.feed_forward_normalization = LayerNorm(config)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """Forward pass for a single layer of the Encoder.

        Args:
            inputs: tensor of shape (batch size, sequence length, d_model)
            mask: sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        x_normalized = self.multi_head_attention_normalization(inputs)
        self_attention, attention_weights = self.multi_head_attention(x_normalized, x_normalized, x_normalized, mask)
        x = self.multi_head_attention_dropout(self_attention) + inputs

        x_normalized = self.feed_forward_normalization(x)
        ff_output = self.feed_forward(x_normalized)
        output = self.feed_forward_dropout(ff_output) + x
        return output, attention_weights


class DecoderLayer(nn.Module):
    """A class for implementing a single Decoder layer."""

    def __init__(self, config):
        """Layers initialization."""
        # TODO: change MultiHeadAttention to MultiHeadAttentionSimple if simple stack is implemented!
        super(DecoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.multi_head_attention_dropout = nn.Dropout(config.dropout_rate)
        self.multi_head_attention_normalization = LayerNorm(config)

        self.multi_head_attention_enc = MultiHeadAttention(config)
        self.multi_head_attention_enc_dropout = nn.Dropout(config.dropout_rate)
        self.multi_head_attention_enc_normalization = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)
        self.feed_forward_normalization = LayerNorm(config)

    def forward(self, inputs: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward pass for a single layer of the Encoder.

        Args:
            inputs: tensor of shape (batch size, target sequence length, d_model)
            encoder_outputs: tensor of shape (batch size, source sequence length, d_model)
            src_mask: source sequence mask with ones at positions that should be masked out
            tgt_mask: target sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        self_attention, self_attention_weights = self.multi_head_attention(inputs, inputs, inputs, tgt_mask)
        x = self.multi_head_attention_dropout(self_attention) + inputs
        x = self.multi_head_attention_normalization(x)

        encoder_attention, encoder_attention_weights = self.multi_head_attention_enc(x, encoder_outputs,
                                                                                     encoder_outputs, src_mask)
        x = self.multi_head_attention_enc_dropout(encoder_attention) + x
        x = self.multi_head_attention_enc_normalization(x)

        ff_output = self.feed_forward(x)
        output = self.feed_forward_normalization(self.feed_forward_dropout(ff_output) + x)
        return output, self_attention_weights, encoder_attention_weights


class DecoderPreNormLayer(nn.Module):
    """A class for implementing a single Decoder layer."""

    def __init__(self, config):
        """Layers initialization."""
        # TODO: change MultiHeadAttention to MultiHeadAttentionSimple if simple stack is implemented!
        super(DecoderPreNormLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.multi_head_attention_dropout = nn.Dropout(config.dropout_rate)
        self.multi_head_attention_normalization = LayerNorm(config)

        self.multi_head_attention_enc = MultiHeadAttention(config)
        self.multi_head_attention_enc_dropout = nn.Dropout(config.dropout_rate)
        self.multi_head_attention_enc_normalization = LayerNorm(config)

        self.feed_forward = FeedForward(config)
        self.feed_forward_dropout = nn.Dropout(config.dropout_rate)
        self.feed_forward_normalization = LayerNorm(config)

    def forward(self, inputs: torch.Tensor, encoder_outputs: torch.Tensor, src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Forward pass for a single layer of the Encoder.

        Args:
            inputs: tensor of shape (batch size, target sequence length, d_model)
            encoder_outputs: tensor of shape (batch size, source sequence length, d_model)
            src_mask: source sequence mask with ones at positions that should be masked out
            tgt_mask: target sequence mask with ones at positions that should be masked out

        Returns:
            Tensor of shape (batch size, sequence length, d_model)
        """
        x_normalized = self.multi_head_attention_normalization(inputs)
        self_attention, self_attention_weights = self.multi_head_attention(x_normalized, x_normalized, x_normalized, tgt_mask)
        x = self.multi_head_attention_dropout(self_attention) + inputs

        x_normalized = self.multi_head_attention_enc_normalization(x)
        encoder_attention, encoder_attention_weights = self.multi_head_attention_enc(x_normalized, encoder_outputs, encoder_outputs, src_mask)
        x = self.multi_head_attention_enc_dropout(encoder_attention) + x

        x_normalized = self.feed_forward_normalization(x)
        ff_output = self.feed_forward(x_normalized)
        output = self.feed_forward_dropout(ff_output) + x
        return output, self_attention_weights, encoder_attention_weights
