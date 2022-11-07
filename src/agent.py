import numpy as np

import torch
from src.replay_memory import Experience
import random


class Agent:
    def __init__(self, env_manager, replay_memory, device):
        self.env_manager = env_manager
        self.replay_memory = replay_memory
        self.device = device
        self.reward_sum = 0

        self.state_stack = None
        self.reset()

    def reset(self):
        obs = self.env_manager.reset()
        self.state_stack = np.stack((obs, obs, obs, obs))
        self.reward_sum = 0

    def select_action(self, epsilon, target_net):  # MUST VERIFY
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
