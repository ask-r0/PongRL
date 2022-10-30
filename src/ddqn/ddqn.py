import numpy as np

from src.pong_env_manager import PongEnvManager
from src.ddqn.network_2 import DQN
from src.replay_memory import ReplayMemory, Experience

import random
import torch
import torch.nn.functional as F
import torch.optim as optim


# MODEL CONSTANTS
GAMMA = 0.99  # The discount rate. Close to 1: future rewards matter more. Close to 0: immediate rewards matter more
EPSILON_START = 0
EPSILON_END = 0
LEARNING_RATE = 0.1

BATCH_SIZE = 50
MEMORY_SIZE = 500
NUM_EPISODES = 1000
MAX_STEPS = 600

TARGET_UPDATE = 10

FRAMES_PER_STATE = 4

epsilon = EPSILON_START


def select_action(env_manager, target_network):
    r = random.uniform(0, 1)
    if r < epsilon:  # Exploration
        return env_manager.get_random_action()
    else:  # Exploitation
        tensor = torch.from_numpy(env_manager.get_processed_state())
        tensor = tensor.to(torch.float32)
        tensor = tensor[None, :]
        with torch.no_grad():
            return target_network(tensor).argmax().item()


def train():
    env_manager = PongEnvManager(FRAMES_PER_STATE, enable_render=False)

    processed_height = env_manager.img_processor.processed_height
    processed_width = env_manager.img_processor.processed_width
    print(env_manager.get_processed_state().shape)

    replay_memory = ReplayMemory(MEMORY_SIZE)
    policy_net = DQN(processed_height, processed_width, 6)
    target_net = DQN(processed_height, processed_width, 6)

    optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)

    for episode in range(NUM_EPISODES):
        env_manager.reset()
        steps = 0

        while (env_manager.is_done == False) and (steps <= MAX_STEPS):
            steps += 1

            #  Choosing action and performing it and saving experience to replay memory
            before_state = env_manager.get_processed_state()
            action = select_action(env_manager, target_net)
            reward = env_manager.step(action)
            after_state = env_manager.get_processed_state()
            replay_memory.push(Experience(before_state, action, after_state, reward))

            if replay_memory.is_sample_available(BATCH_SIZE):
                experiences = replay_memory.get_sample(BATCH_SIZE)

                #  Handling the states, actions, next_states and rewards from batch...

                states, actions, next_states, rewards = zip(*experiences)

                #  Staking the states into following format: (BATCH_SIZE, FRAMES_PER_STATE, height, width)
                states = np.stack(states)
                next_states = np.stack(next_states)

                #  Converting them to tensors
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                #  get nn evaluation for current and next state
                states_eval = policy_net(states)  # 256x6, tensor
                next_states_eval = target_net(next_states)  # 256x6, tensor

                #  Calculating the values used to calculate loss.
                #  target_q_values is the max q-value for next-state multiplied by gamma added to reward
                #  current_q_values is the q-value for the action taken it the state
                target_q_values, _ = torch.max(next_states_eval, dim=1)
                target_q_values = target_q_values * GAMMA + rewards
                current_q_values = states_eval.gather(dim=1, index=actions.unsqueeze(-1)).squeeze()

                #  Calculate loss & update neural network
                loss = F.mse_loss(current_q_values, target_q_values)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                print(episode)
        if episode % TARGET_UPDATE == 0:
            #  Update the target network to match the policy network
            target_net.load_state_dict(policy_net.state_dict())


train()
