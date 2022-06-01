import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, output):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output)
        )

    def forward(self, input):
        state_layer = self.layers(input)
        return state_layer
