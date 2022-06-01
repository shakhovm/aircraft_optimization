import torch.nn as nn


class DoubleDQNModel(nn.Module):
    def __init__(self, output):
        super(DoubleDQNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, output)
        )

    def forward(self, x):
        state_layer = self.layers(x)
        return state_layer
