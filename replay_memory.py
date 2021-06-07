import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'a', 'da'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, state, action, mask, next_state, reward, a, da):
        """Saves a transition."""
        self.memory.append(Transition(state, action, mask, next_state, reward, a, da))

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
