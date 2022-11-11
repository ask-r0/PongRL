import collections
import numpy as np

# An experience of an agent. Includes: a state, action taken in state, resulting state, received reward, done status
Experience = collections.namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    """ Class for ReplayMemory: a collection of experiences to be used for training the agent.
    Attributes:
        memory: A queue used to store experiences
    """

    def __init__(self, capacity):
        """ Inits class with a capacity (max number of experiences to store at a given time) """
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, experience):
        """Save an experience to memory """
        self.memory.append(experience)

    def get_sample(self, batch_size):
        """ Returns a random batch of experiences from the replay memory.
        One batch is returned as 5 arrays. One array for each experience parameter (states, actions, next_states,
        rewards and dones). Experience number x from batch corresponds to index x in the different arrays.

        """
        indices = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        states, actions, next_states, rewards, dones = zip(*[self.memory[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

    def is_sample_available(self, batch_size):
        """ A sample is only available for a given batch_size if ReplayMemory has enough experiences"""
        return len(self.memory) >= batch_size

    def __len__(self):
        """Returns length of replay memory"""
        return len(self.memory)
