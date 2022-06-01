import torch
from torch import nn
import torch.nn.functional as F


class LinearDDQNModel(nn.Module):
    def __init__(self, num_actions):
        super(LinearDDQNModel, self).__init__()
        # self.seed = torch.manual_seed(0)
        self.num_actions = num_actions
        # self.shared = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.Tanh(),
        # )
        self.value = nn.Sequential(
            nn.Linear(3, 64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(64, self.num_actions)
        )

    def forward(self, x):
        # x = self.layers(x)
        # x = x.view(x.shape[0], -1)
        # x = self.shared(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean()