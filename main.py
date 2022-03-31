import torch
import torch.nn as nn


class JDDNetwork(nn.model):
    def __init__(self, width, depth, noise_level, pre_trained=False, padding=True):
        super(JDDNetwork, self).__init__()

        self.width = width
        self.depth = depth
        self.noise_level = noise_level

        
