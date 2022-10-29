import collections
import random

# An experience of an agent. This includes a state, action taken in state, the next state and a received reward
Experience = collections.namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """ Class for ReplayMemory: a collection of experiences to be used for training the agent.

    Attributes:
        memory: A queue used to store experiences
    """

    def __init__(self, capacity):
        """ Inits class with a capacity (max number of experiences to store at a given time) """
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, experience):
        """Save an experience"""
        self.memory.append(experience)

    def get_sample(self, batch_size):
        """ Returns a random experience from the replay memory """
        return random.sample(self.memory, batch_size)

    def is_sample_available(self, batch_size):
        """ A sample is only available for a given batch_size if ReplayMemory has enough experiences"""
        return len(self.memory) >= batch_size

    def __len__(self):
        """Returns length of replay memory"""
        return len(self.memory)
