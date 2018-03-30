import torch
from torch.utils.data import sampler

class RandomSampler(sampler.Sampler):
    """Samples elements randomly, without replacement.
    Arguments:
        num_samples: number of desired datapoints
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(torch.randperm(self.num_samples).tolist())

    def __len__(self):
        return self.num_samples




