import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class REINFORCE(nn.Module):
    def __init__(self, num_actions):
        super(REINFORCE, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_actions),

        )
        self.apply(REINFORCE.weights_init)

    def act(self, x):
        x = F.softmax(self.fc(x), dim=0)
        return x

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 1 / np.sqrt(5))
