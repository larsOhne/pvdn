from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor
import numpy as np


class RandomGamma(object):
    """Random gamma correction."""
    def __init__(self, gamma=(0.8, 1.2)):  # gamma=(0.5, 2)):
    # 1.2)):
        self.gamma = gamma

    def __call__(self, sample):
        sample = np.array(sample, dtype=object) # important to allow ops on arbitrary sized floats
        gamma = np.random.rand() * (self.gamma[1] - self.gamma[0]) + \
                self.gamma[0]

        sample = sample ** (1 / gamma)

        return sample.astype(np.float32)