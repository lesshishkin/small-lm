import random
from itertools import chain

from torch.utils.data.sampler import Sampler


class RandomSortingSampler(Sampler):
    """A class implementing batch random sampler for the sorted dataset."""

    def __init__(self, dataset, batch_size: int, shuffle: bool = False, drop_last: bool = False,
                 dataset_length: int = None):
        super(RandomSortingSampler, self).__init__(dataset)
        self.dataset_length = len(dataset) if dataset_length is None else dataset_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self._reset_sampler()

    def _reset_sampler(self):
        """Resets sampler iterator."""
        sample_ids = range(self.dataset_length)
        batches = [sample_ids[i:i + self.batch_size] for i in range(0, self.dataset_length, self.batch_size)]
        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            random.shuffle(batches)
        self.sampler = iter(chain.from_iterable(batches))

    def __iter__(self):
        """Iterates over batches."""
        batch = []

        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

        self._reset_sampler()

    def __len__(self):
        if self.drop_last:
            return self.dataset_length // self.batch_size
        else:
            return (self.dataset_length + self.batch_size - 1) // self.batch_size
