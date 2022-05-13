from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, max_items):
        self.buffer = deque(maxlen=max_items)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states = []
        actions = []
        rewards = []
        states_next = []
        dones = []
        for sample_one in samples:
            states.append(sample_one[0])
            actions.append(sample_one[1])
            rewards.append([sample_one[2]])
            states_next.append(sample_one[3])
            dones.append([sample_one[4]])
        return [states, actions, rewards, states_next, dones]

    def __len__(self):
        return len(self.buffer)