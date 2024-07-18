import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from configs.experiment_config import experiment_cfg


class TransformerInterpretation:
    """A class for Transformer model interpretation."""

    def __init__(self, trainer, source_lang_preprocessor, target_lang_preprocessor):
        self.trainer = trainer
        self.model = trainer.model
        self.source_lang_preprocessor = source_lang_preprocessor
        self.target_lang_preprocessor = target_lang_preprocessor

    def visualize_attention(self, source_text: str, layer: int = -1, head: int = 0, is_decoder: bool = False,
                            is_cross_attention: bool = False):
        """Visualizes attention weights for a given input text.

        Args:
            source_text: source input text
            layer: layer index to visualize (default: -1 for the last layer)
            head: head index to visualize (default: 0)
            is_decoder: whether to visualize decoder attention weights
            is_cross_attention: whether to visualize cross-attention weights in the decoder
        """
        predicted_text, encoder_attn_weights, decoder_self_attn_weights, decoder_encoder_attn_weights = \
            self.get_attention_weights(source_text)

        if is_decoder:
            if is_cross_attention:
                attention_weights = decoder_encoder_attn_weights[layer].squeeze().detach().numpy()[head]
                tokens = self.source_lang_preprocessor.tokenize(source_text)[1:-1]
                tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
                decoder_tokens = self.target_lang_preprocessor.tokenize(predicted_text[0])[:-1]
                decoder_tokens = [t.replace('</w>', '') if t != '</w>' else t for t in decoder_tokens]
                xticklabels = tokens
                yticklabels = decoder_tokens
                title = f"Decoder Cross-Attention Weights (Layer {layer + 1}, Head {head + 1})"
            else:
                attention_weights = decoder_self_attn_weights[layer].squeeze().detach().numpy()[head]
                tokens = self.target_lang_preprocessor.tokenize(predicted_text[0])[:-1]
                tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
                xticklabels = tokens
                yticklabels = tokens
                title = f"Decoder Self-Attention Weights (Layer {layer + 1}, Head {head + 1})"
        else:
            attention_weights = encoder_attn_weights[layer].squeeze().detach().numpy()[head]
            tokens = self.source_lang_preprocessor.tokenize(source_text)[1:-1]
            tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
            xticklabels = tokens
            yticklabels = tokens
            title = f"Encoder Self-Attention Weights (Layer {layer + 1}, Head {head + 1})"

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attention_weights, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
        ax.set_title(title, fontsize=15)
        plt.show()

    def visualize_attention_layer(self, source_text: str, layer: int = -1, is_decoder: bool = False,
                                  is_cross_attention: bool = False):
        """Visualizes attention weights for a given input text.

        Args:
            source_text: source input text
            layer: layer index to visualize (default: -1 for the last layer)
            is_decoder: whether to visualize decoder attention weights
            is_cross_attention: whether to visualize cross-attention weights in the decoder
        """
        predicted_text, encoder_attn_weights, decoder_self_attn_weights, decoder_encoder_attn_weights = \
            self.get_attention_weights(source_text)

        if is_decoder:
            if is_cross_attention:
                attention_weights = decoder_encoder_attn_weights[layer].squeeze().detach().numpy().mean(axis=0)
                tokens = self.source_lang_preprocessor.tokenize(source_text)[1:-1]
                tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
                decoder_tokens = self.target_lang_preprocessor.tokenize(predicted_text[0])[:-1]
                decoder_tokens = [t.replace('</w>', '') if t != '</w>' else t for t in decoder_tokens]
                xticklabels = tokens
                yticklabels = decoder_tokens
                title = f"Decoder Cross-Attention Weights (Layer {layer + 1})"
            else:
                attention_weights = decoder_self_attn_weights[layer].squeeze().detach().numpy().mean(axis=0)
                tokens = self.target_lang_preprocessor.tokenize(predicted_text[0])[:-1]
                tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
                xticklabels = tokens
                yticklabels = tokens
                title = f"Decoder Self-Attention Weights (Layer {layer + 1})"
        else:
            attention_weights = encoder_attn_weights[layer].squeeze().detach().numpy().mean(axis=0)
            tokens = self.source_lang_preprocessor.tokenize(source_text)[1:-1]
            tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
            xticklabels = tokens
            yticklabels = tokens
            title = f"Encoder Self-Attention Weights (Layer {layer + 1})"

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attention_weights, xticklabels=xticklabels, yticklabels=yticklabels, ax=ax)
        ax.set_title(title, fontsize=15)
        plt.show()

    def get_attention_weights(self, source_text: str):
        """Gets attention weights for a given input text.

        Args:
            source_text: source input text

        Returns:
            Tuple: (encoder_attention_weights, decoder_self_attention_weights, decoder_encoder_attention_weights)
        """
        encoder_inputs = torch.tensor(self.source_lang_preprocessor.encode(source_text))[1:-1].unsqueeze(0)

        inference_config = experiment_cfg.inference
        prediction, encoder_attn_weights, decoder_self_attn_weights, decoder_encoder_attn_weights = \
            self.trainer.inference(encoder_inputs, inference_config, return_attention=True)
        decoded_prediction = self.target_lang_preprocessor.decode(prediction, batch=True)

        return decoded_prediction, encoder_attn_weights, decoder_self_attn_weights, decoder_encoder_attn_weights

    def attention_rollout(self, attention_weights: list[torch.Tensor], start_layer: int = 0, is_decoder: bool = False):
        """Computes attention rollout for given attention weights.

        Args:
            attention_weights: List of attention weights from each layer
            start_layer: Layer index to start the rollout from
            is_decoder: Whether to handle decoder masking

        Returns:
            np.array: aggregated attention matrix
        """
        result = attention_weights[start_layer].squeeze().detach().numpy().mean(axis=0)

        for i in range(start_layer + 1, len(attention_weights)):
            attention = attention_weights[i].squeeze().detach().numpy()
            attention = attention.mean(axis=0)  # Average over heads
            if is_decoder:
                # Normalize based on the receptive field of attention for decoders
                attention = self.normalize_attention_for_receptive_field(attention)
            attention += np.eye(attention.shape[-1])  # Add identity for residual connection
            attention /= attention.sum(axis=-1, keepdims=True)  # Normalize
            result = np.matmul(attention, result)  # Matrix multiplication

        return result

    @staticmethod
    def normalize_attention_for_receptive_field(attention: np.ndarray):
        """Normalizes attention based on the receptive field for decoder.

        Args:
            attention: attention weights to be normalized

        Returns:
            np.array: normalized attention weights
        """
        seq_len_target, seq_len_source = attention.shape

        for i in range(seq_len_target):
            attention[i, :i + 1] /= (i + 1)
        return attention

    def visualize_attention_rollout(self, source_text: str, start_layer: int = 0, is_decoder: bool = False):
        """Visualizes attention rollout for a given input text.

        Args:
            source_text: source input text
            start_layer: layer index to start the rollout from
            is_decoder: whether to visualize decoder attention rollout
        """
        predicted_text, encoder_attn_weights, decoder_self_attn_weights, decoder_encoder_attn_weights = \
            self.get_attention_weights(source_text)

        if is_decoder:
            attention_weights = decoder_self_attn_weights
        else:
            attention_weights = encoder_attn_weights

        rollout_matrix = self.attention_rollout(attention_weights, start_layer, is_decoder)
        if is_decoder:
            tokens = self.target_lang_preprocessor.tokenize(predicted_text[0])[:-1]
            tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
            tokens_x = tokens
            tokens = self.source_lang_preprocessor.tokenize(source_text)[1:-1]
            tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
            tokens_y = tokens
        elif is_decoder:
            tokens = self.target_lang_preprocessor.tokenize(predicted_text[0])[:-1]
            tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
            tokens_x, tokens_y = tokens, tokens
        else:
            tokens = self.source_lang_preprocessor.tokenize(source_text)[1:-1]
            tokens = [t.replace('</w>', '') if t != '</w>' else t for t in tokens]
            tokens_x, tokens_y = tokens, tokens

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(rollout_matrix, xticklabels=tokens_x, yticklabels=tokens_y, ax=ax)
        plt.show()
