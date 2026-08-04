from __future__ import annotations

import math
import random
from collections.abc import Iterator

from torch.utils.data import Sampler


class BoundedCyclicSampler(Sampler[int]):
    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int,
        seed: int,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive.")
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def set_samples_per_epoch(self, value: int) -> None:
        if value <= 0:
            raise ValueError("samples_per_epoch must be positive.")
        self.samples_per_epoch = value

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        start = self.epoch * self.samples_per_epoch
        stop = start + self.samples_per_epoch
        first_cycle = start // self.dataset_size
        last_cycle = (stop - 1) // self.dataset_size

        permutations: dict[int, list[int]] = {}
        for cycle in range(first_cycle, last_cycle + 1):
            permutation = list(range(self.dataset_size))
            random.Random(self.seed + cycle).shuffle(permutation)
            permutations[cycle] = permutation

        for absolute_position in range(start, stop):
            cycle = absolute_position // self.dataset_size
            offset = absolute_position % self.dataset_size
            yield permutations[cycle][offset]

    def epochs_per_full_coverage(self) -> int:
        return math.ceil(self.dataset_size / self.samples_per_epoch)
