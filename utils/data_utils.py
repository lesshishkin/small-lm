import random
from typing import Union

import torch
from PIL import ImageFilter, ImageOps, Image
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from utils.enums import ImageAugmentationType


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
    encoder_inputs, decoder_inputs, decoder_outputs, sample_indices = [], [], [], []

    for data_dict in batch:
        encoder_inputs.append(torch.tensor(data_dict['source_lang_tokens'][1:-1]))
        decoder_inputs.append(torch.tensor(data_dict['target_lang_tokens'][:-1]))
        decoder_outputs.append(torch.tensor(data_dict['target_lang_tokens'][1:]))
        sample_indices.append(torch.tensor(data_dict['sample_pair_id'], dtype=torch.int))

    encoder_inputs = pad_sequence(encoder_inputs, batch_first=True)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True)
    decoder_outputs = pad_sequence(decoder_outputs, batch_first=True)
    sample_indices = torch.vstack(sample_indices)

    encoder_mask = get_sequence_mask(encoder_inputs)
    decoder_mask = get_sequence_mask(decoder_inputs, mask_future_positions=True)

    return sample_indices, encoder_inputs, decoder_inputs, decoder_outputs, encoder_mask, decoder_mask


class Grayscale:
    """A class for gray scale image transformation."""

    def __init__(self, p: float = 0.2):
        self.p = p
        self.transform = transforms.Grayscale(num_output_channels=3)

    def __call__(self, image: Union[Image.Image, torch.Tensor]):
        if random.random() < self.p:
            return self.transform(image)
        return image


class GaussianBlur:
    """A class for gaussian blur image transformation."""

    def __init__(self, p: float = 0.2, radius_min: float = 0.1, radius_max: float = 2.):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: Image.Image):
        if random.random() < self.p:
            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        return image


class Solarization:
    """A class for solarization image transformation."""

    def __init__(self, p: float = 0.2, threshold: float = 128):
        self.p = p
        self.threshold = threshold

    def __call__(self, image: Image.Image):
        if random.random() < self.p:
            return ImageOps.solarize(image, self.threshold)
        return image


def get_image_augmentations(model_config, augmentation_type: ImageAugmentationType,
                            remove_random_resized_crop: bool = False, color_jitter: float = 0.3, prob: float = 1.0):
    """Gets image augmentation transforms."""
    if augmentation_type == ImageAugmentationType.base:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(model_config.image_size),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet parameters
        ])
    elif augmentation_type == ImageAugmentationType.deit_3:
        image_size = (model_config.image_size, model_config.image_size)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if remove_random_resized_crop:
            primary_tfl = [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip()
            ]
        else:
            primary_tfl = [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]

        secondary_tfl = [transforms.RandomChoice([Grayscale(p=prob), Solarization(p=prob), GaussianBlur(p=prob)])]
        if color_jitter is not None and not color_jitter == 0:
            secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))

        final_tfl = [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
        train_transforms = transforms.Compose(primary_tfl + secondary_tfl + final_tfl)
    else:
        raise Exception('Unknown augmentation type')

    return train_transforms
