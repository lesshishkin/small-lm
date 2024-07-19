import os

from torch.utils.data import Dataset

from dataset.preprocessing import Preprocessing
from utils.common_functions import write_file, read_file
from utils.enums import SetType


class RussianStoriesDataset(Dataset):
    """A class for Russian Tiny Stories Dataset."""

    def __init__(self, config, set_type: SetType):
        self.config = config
        self.set_type = set_type

        if set_type.name == SetType.train:
            self.dataset = read_file(config.tokenized_train_data_path)

        elif set_type.name == SetType.validation:
            self.dataset = read_file(config.tokenized_valid_data_path)

        else:
            raise Exception('Unknown set type')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample_data = {
            'id': idx,
            'tokens': self.dataset[idx]
        }

        return sample_data
