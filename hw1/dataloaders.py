import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.first = True

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):  # -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        if self.first:
            if self.i >= len(self.data_source) / 2: #even check
                raise StopIteration
            self.first = False
            return self.i
        else:
            if self.i+1 > len(self.data_source) / 2: #odd check
                raise StopIteration
            self.first = True
            self.i += 1
            return len(self.data_source) - self.i

    def __len__(self):
        return len(self.data_source)


class CutSampler(Sampler):

    def __init__(self, data_source: Sized, begin, end):
        super().__init__(data_source)
        self.data_source = data_source
        self.begin = begin
        self.end = end
        self.indices = list(self)

    def __iter__(self):
        self.i = self.begin - 1
        return self

    def __next__(self):  # -> Iterator[int]:
        self.i += 1
        if self.i > self.end:
            raise StopIteration
        return self.i

    def __len__(self):
        return len(self.data_source)

class ByIndicesSampler(Sampler):

    def __init__(self, data_source: Sized, indices):
        super().__init__(data_source)
        self.data_source = data_source
        self.indices = indices

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):  # -> Iterator[int]:
        self.i += 1
        if self.i >= len(self.indices):
            raise StopIteration
        return self.indices[self.i]

    def __len__(self):
        return len(self.indices)

def create_train_validation_loaders(
        dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    train_indices, valid_indices = torch.utils.data.random_split(range(len(dataset)), [round((1-validation_ratio) * len(dataset)), round(validation_ratio * len(dataset))])
    dl_train = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           sampler=ByIndicesSampler(dataset, train_indices),
                                           )
    dl_valid = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           sampler=ByIndicesSampler(dataset, valid_indices),
                                           )
    return dl_train, dl_valid
