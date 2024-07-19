import random
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence


def get_sequence_mask(sequence: torch.Tensor, pad_idx: int = 0, mask_future_positions: bool = False,
                      device: str = 'cpu') -> torch.Tensor:
    """Creates a mask for sequences.

    Args:
        sequence: Tensor containing the sequences.
        pad_idx: Index for padding.
        mask_future_positions: Whether to mask future positions.
        device: Device to perform the operation on.

    Returns:
        Tensor containing the mask.
    """
    batch_size, batch_max_seq_len = sequence.size()
    padding_mask = (sequence == pad_idx).unsqueeze(1).unsqueeze(2)
    if mask_future_positions:
        attention_shape = (batch_max_seq_len, batch_max_seq_len)
        future_positions_mask = torch.triu(torch.ones(attention_shape, device=device), diagonal=1).bool()
        return torch.max(padding_mask, future_positions_mask)

    return padding_mask


def collate_function(batch):
    """Collates a batch of data.

    Args:
        batch: List of data dictionaries.

    Returns:
        Tuple containing collated batch data and masks.
    """
    decoder_inputs, decoder_outputs, sample_indices = [], [], []

    for data_dict in batch:
        decoder_inputs.append(torch.tensor(data_dict['tokens'][:-1]))
        decoder_outputs.append(torch.tensor(data_dict['tokens'][1:]))
        sample_indices.append(torch.tensor(data_dict['id'], dtype=torch.int))

    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True)
    decoder_outputs = pad_sequence(decoder_outputs, batch_first=True)
    sample_indices = torch.vstack(sample_indices)

    decoder_mask = get_sequence_mask(decoder_inputs, mask_future_positions=True)

    return sample_indices, decoder_inputs, decoder_outputs, decoder_mask
