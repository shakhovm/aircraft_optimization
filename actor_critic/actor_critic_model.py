import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh()

        )
        self.fc_pi = nn.Sequential(
            nn.Linear(32, num_actions)
        )
        self.fc_v = nn.Sequential(
            nn.Linear(32, 1)
        )

    def pi(self, x, softmax_dim = 0):
        x = self.fc1(x)
        x = self.fc_pi(x)
        # prob = F.softmax(x, dim=softmax_dim)
        return x

    def v(self, x):
        x = self.fc1(x)
        v = self.fc_v(x)
        return v