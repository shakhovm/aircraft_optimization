import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(3, 256)
        self.fc_pi = nn.Linear(256,num_actions)
        self.fc_v = nn.Linear(256,1)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v