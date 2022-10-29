import numpy as np

from src.pong_env_manager import PongEnvManager
from src.ddqn.network_2 import DQN
from src.replay_memory import ReplayMemory, Experience

import random
import torch

# MODEL CONSTANTS
GAMMA = 0.99  # The discount rate. Close to 1: future rewards matter more. Close to 0: immediate rewards matter more
EPSILON_START = 0
EPSILON_END = 0
LEARNING_RATE = 0.1

BATCH_SIZE = 256
MEMORY_SIZE = 500
NUM_EPISODES = 1000

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

    for episode in range(NUM_EPISODES):
        env_manager.reset()

        while not env_manager.is_done:
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
                actions = torch.tensor(actions, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)

                #  get q values for current and next state
                states_q_values = policy_net(states)  # 256x6, tensor
                next_states_q_values = target_net(next_states)  # 256x6, tensor

                """ TODO:
                vi har nå states_q_values og next_states_q_values. Disse må oversettes til lister med 256 verdier
                * for states_q_values ønsker vi q-value til action-taken for hver state (husk actions lista)
                * for next_states_q_values ønsker vi max q-value for alle actions fra hver next_state,
                dette vil si da å istedenfor ha 6 q-values per next_state ønsker vi kun den største
                
                deretter kan vi kalkulere loss for hele batch, og oppdatere weights
                HUSK! loss = MSE((next_q_values * gamma) + rewards, current_q_values)
                hvor next_q_values er den nye 256 lista fra next_states_q_values
                og current_q_values er den nye lista fra states_q_values
                
                """



train()
