import torch
from torch.utils.data import Dataset


class BinaryPatternDataset(Dataset):
    def __init__(self, num_patterns, num_repeats, pattern_length,
                 pattern_sparsity, seed=None):
        self.num_patterns = num_patterns
        self.num_repeats = num_repeats
        self.pattern_length = pattern_length
        self.pattern_sparsity = pattern_sparsity

        self.seed = seed
        self.generator = torch.Generator()
        if self.seed is not None:
            self.generator.manual_seed(self.seed)

        self.patterns, self.pattern_ids = self.generate_dataset()

    def generate_dataset(self):
        patterns = torch.zeros((self.num_patterns, self.pattern_length))
        pattern_ids = torch.arange(self.num_patterns)

        for i in range(self.num_patterns):
            pattern = torch.zeros((self.pattern_length,))
            pattern[torch.randperm(self.pattern_length, generator=self.generator)[
                    :int(self.pattern_sparsity * self.pattern_length)]] = 1
            patterns[i] = pattern

        patterns = patterns.repeat((self.num_repeats, 1))
        pattern_ids = pattern_ids.repeat((self.num_repeats,))

        return patterns, pattern_ids

    def __len__(self):
        return self.num_patterns * self.num_repeats

    def __getitem__(self, idx):
        return self.patterns[idx], self.pattern_ids[idx]
