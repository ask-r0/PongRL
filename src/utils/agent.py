import numpy as np

import torch
from src.utils.replay_memory import Experience
import random


class Agent:
    """ Agent acting in an environment.

    In this application, the agent does the following: Selects & performs actions based on epsilon and saving
    the resulting experiences from actions in replay memory.

    In the background the agent keeps track of 4 consecutive observations at every time instance, these observations
    represents a state. A state can not only consist of 1 observation as it would not capture motion, therefor
    4 consecutive frames are stacked to represent a state.

    Attributes:
        env_manager: Wrapper for the environment the agent is interacting with.
        replay_memory: Buffer to save experiences of the agent.
        device: Device used by torch (typically cpu or cuda)
        reward_sum: Sum of rewards in current episode. Resets to zero in-between episodes
        state_stack: Current state represented as the last 4 consecutive observations stacked
    """
    def __init__(self, env_manager, replay_memory, device):
        self.env_manager = env_manager
        self.replay_memory = replay_memory
        self.device = device
        self.reward_sum = 0

        self.state_stack = None
        self.reset()

    def reset(self):
        """ Resets environment, state stack and reward sum. """
        obs = self.env_manager.reset()
        self.state_stack = np.stack((obs, obs, obs, obs))
        self.reward_sum = 0

    def select_action(self, epsilon, target_net):
        """ Selects action to perform in environment. NB! Does not perform action

        Attributes:
            epsilon: Epsilon determines the percentage for choosing exploration. e.g. epsilon=0.9 90% chance of
              exploration and 10% chance of exploitation.
            target_net: Target neural network used in DQN when calculating Q-value for state-action pairs

        Returns:
            Action to be performed in environment
        """

        r = random.uniform(0, 1)
        if r < epsilon:  # Exploration
            action = self.env_manager.get_random_action()
        else:  # Exploitation
            with torch.no_grad():
                t = torch.tensor(self.state_stack, dtype=torch.float, device=self.device).unsqueeze(0)
                qs = target_net(t)
                action = torch.argmax(qs).item()
        return action

    def perform_action(self, epsilon, target_net):
        """ Performs action in environment with help of select_action method

        Attributes:
            epsilon: Epsilon used to determine action (see select_action docs)
            target_net: Target neural network used to determine action (see select_action docs)

        Returns:
            None                          if episode was not finished by this action
            Sum of rewards for episode    if episode was finished by this action
        """
        action = self.select_action(epsilon, target_net)  # Determine next action
        new_state, reward, done, info = self.env_manager.step(action)  # Perform action
        self.reward_sum += reward

        old_stack = self.state_stack
        self.state_stack = np.stack((new_state, self.state_stack[0], self.state_stack[1], self.state_stack[2]))

        #  Save to replay memory
        exp = Experience(old_stack, action, self.state_stack, reward, done)
        self.replay_memory.push(exp)

        #  Handle done
        reward_ret = None
        if done:
            reward_ret = self.reward_sum
            self.reset()

        return reward_ret
