import torch
import torch.nn as nn
import torch.nn.functional as F
from ddpg.updating_model import UpdatingModel


class CriticModel(nn.Module, UpdatingModel):
    def __init__(self, alpha, tau, input, output):
        super(CriticModel, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU()
        )
        self.action_layer = nn.Sequential(
            nn.Linear(output, 64),
            nn.ReLU()
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output)
        )
        self.alpha = alpha
        self.tau = tau
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, action):
        state_layer = self.state_layer(state)
        action_layer = self.action_layer(action)
        shared = torch.cat([state_layer, action_layer], dim=1)
        out = self.shared_layers(shared)
        return out