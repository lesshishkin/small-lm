import os
from torch.utils.data import Dataset
from utils.common_functions import write_file, read_file
from utils.enums import SetType


class TinyStoriesDataset(Dataset):
    """A class for Russian Tiny Stories Dataset."""

    def __init__(self, config, set_type: SetType):
        self.config = config
        self.set_type = set_type

        if self.set_type == SetType.train:
            self.dataset = read_file(os.path.join(self.config.path_to_data,
                                                  self.config.tokenized_train_data_path))

        elif self.set_type == SetType.validation:
            self.dataset = read_file(os.path.join(self.config.path_to_data,
                                                  self.config.tokenized_valid_data_path))

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
