from src.pong_env_manager import PongEnvManager
from src.ddqn.network import DQN

import random

# MODEL CONSTANTS
GAMMA = 0.99  # The discount rate. Close to 1: future rewards matter more. Close to 0: immediate rewards matter more
EPSILON_START = 0
EPSILON_END = 1
LEARNING_RATE = 0.1

BATCH_SIZE = 256
MEMORY_SIZE = 500
NUM_EPISODES = 1000

TARGET_UPDATE = 10

FRAMES_PER_STATE = 4

epsilon = EPSILON_START


def select_action(env_manager, target_network):
    r = random.uniform(0, 1)
    if r < epsilon: # Exploration
        return env_manager.get_random_action()
    else:  # Exploitation
        pass # TODO: implement


def train():
    env_manager = PongEnvManager(FRAMES_PER_STATE, enable_render=False)

    processed_height = env_manager.img_processor.processed_height
    processed_width = env_manager.img_processor.processed_width

    policy_net = DQN(processed_height, processed_width)
    target_net = DQN(processed_height, processed_width)

    for episode in range(NUM_EPISODES):
        env_manager.reset()

        while not env_manager.is_done:
            pass
