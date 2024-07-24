import os
import random
import sys
import numpy as np
import evaluate

import torch
from models.tinyllm import TinyLLM
from utils.common_functions import set_seed
from utils.data_utils import get_sequence_mask, collate_function
from utils.enums import SetType, InferenceType
from torch.nn.functional import softmax

import youtokentome as yttm


class Inferencer:
    """A class for model inferencing."""
    # TODO Доделать

    def __init__(self, config, init_logger=True):
        self.config = config
        set_seed(self.config.seed)
        self.tokenizer = yttm.BPE(model=self.config.data.tokenizer_path, n_threads=-1)
        self._prepare_model()
        print('Model ready')

    def _prepare_model(self):
        """Preparing model, optimizer and loss function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TinyLLM(self.config.model,
                             vocab_size=self.tokenizer.vocab_size(),
                             device=self.device).to(self.device)

        self.load(self.config.inference.model_path)

    def load(self, filepath: str):
        """Loads trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def inference_step(self, encoded_input: torch.Tensor, decoded_sequence: torch.Tensor, source_mask: torch.Tensor):
        """Gets model decoder output given encoder output and sequence made by decoder at current step.

        Args:
            encoded_input: source sequences passed through the model encoder (batch size, source sequence length, d_model)
            decoded_sequence: sequences with all tokens generated up to the current inference step (batch size, generated sequence length)
            source_mask: a sequence mask with ones at positions that should be masked out (for encoder outputs)

        Returns:
            Model output generated wrt the already generated sequence (decoded_sequence)
        """
        # TODO переделать это тоже
        target_mask = get_sequence_mask(decoded_sequence, mask_future_positions=True, device=self.device)

        with torch.no_grad():
            decoder_inputs_embeddings = self.model.positional_encoding(self.model.embeddings_decoder(decoded_sequence))
            decoder_output, decoder_self_attention_weights, decoder_encoder_attention_weights = self.model.decoder(
                decoder_inputs_embeddings, encoded_input, source_mask, target_mask)
            output = self.model.output(decoder_output)

        return output, decoder_self_attention_weights, decoder_encoder_attention_weights

    @torch.no_grad()
    def inference(self, sequence: torch.Tensor, inference_config, return_attention=False):
        """Makes inference with auto-regressive decoding for the given sequence."""
        # TODO переделать инференс
        self.model.eval()
        batch_size = sequence.size(0)
        sos_token_id = self.config.data.special_tokens.index("<BOS>")
        eos_token_id = self.config.data.special_tokens.index("<EOS>")
        inference_step = 0
        decoded_sequence = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device) * sos_token_id
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        input_mask = get_sequence_mask(sequence, device=self.device)
        encoder_inputs_embeddings = self.model.positional_encoding(self.model.embeddings_encoder(sequence))
        encoded_input, encoder_attention_weights = self.model.encoder(encoder_inputs_embeddings, input_mask)

        while not finished_sequences.all() and inference_step < inference_config.stop_predict:
            output, decoder_self_attention_weights, decoder_encoder_attention_weights = self.inference_step(
                encoded_input, decoded_sequence, input_mask)
            if inference_config.type == InferenceType.greedy.value:
                current_token = torch.argmax(output, dim=-1)[:, inference_step].view(-1, 1) + 1
            elif inference_config.type == InferenceType.temperature.value:
                output = output / (inference_config.temperature_value + inference_config.eps)
                probabilities = softmax(output, dim=-1)
                current_token = probabilities[:, inference_step, :].multinomial(num_samples=1) + 1
            else:
                raise Exception('Unknown inference type!')

            decoded_sequence = torch.hstack([decoded_sequence, current_token])
            finished_sequences |= current_token.squeeze() == eos_token_id
            inference_step += 1

        eos_subsequence_mask = torch.cummax(decoded_sequence == eos_token_id, dim=1).values
        decoded_sequence = decoded_sequence.masked_fill(eos_subsequence_mask, eos_token_id)
        if return_attention:
            return decoded_sequence.cpu().tolist(), encoder_attention_weights, decoder_self_attention_weights, decoder_encoder_attention_weights
        else:
            return decoded_sequence.cpu().tolist()
