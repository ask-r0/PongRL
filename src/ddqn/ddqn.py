import numpy as np

from src.pong_env_manager import PongEnvManager
from src.ddqn.network_2 import DQN
from src.replay_memory import ReplayMemory, Experience
from src.progress_plotter import plot_episode_vs_reward

import random
import torch
import torch.nn.functional as F
import torch.optim as optim

# MODEL CONSTANTS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE BEING USED: {DEVICE}")

GAMMA = 0.97  # The discount rate. Close to 1: future rewards matter more. Close to 0: immediate rewards matter more
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.00025

BATCH_SIZE = 64
MEMORY_SIZE = 50000
MIN_MEMORY_SIZE = 40000  # Training should not start before having MIN_MEMORY_SIZE
NUM_EPISODES = 100
MAX_STEPS = 1000

TARGET_UPDATE = 1

FRAMES_PER_STATE = 4

episode_list = []
reward_list = []


def get_new_epsilon(x):
    return max(EPSILON_END, EPSILON_DECAY * x)


def select_action(env_manager, target_network, epsilon):
    r = random.uniform(0, 1)
    if r < epsilon:  # Exploration
        return env_manager.get_random_action()
    else:  # Exploitation
        tensor = torch.from_numpy(env_manager.get_processed_state()).to(DEVICE)
        tensor = tensor.to(torch.float32)
        tensor = tensor[None, :]
        with torch.no_grad():
            return target_network(tensor).argmax().item()


def train(epsilon=EPSILON_START, has_training_started=False):
    env_manager = PongEnvManager(FRAMES_PER_STATE, enable_render=False)

    processed_height = env_manager.img_processor.out_height_width
    processed_width = env_manager.img_processor.out_height_width

    replay_memory = ReplayMemory(MEMORY_SIZE)
    policy_net = DQN(processed_height, processed_width, 6).to(DEVICE)
    target_net = DQN(processed_height, processed_width, 6).to(DEVICE)

    optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)

    for episode in range(NUM_EPISODES):
        env_manager.reset()
        steps = 0
        reward_sum = 0

        while (env_manager.is_done == False) and (steps <= MAX_STEPS):
            steps += 1

            #  Choosing action and performing it and saving experience to replay memory
            before_state = env_manager.get_processed_state()
            action = select_action(env_manager, target_net, epsilon)
            reward = env_manager.step(action)
            reward_sum += reward
            after_state = env_manager.get_processed_state()
            replay_memory.push(Experience(before_state, action, after_state, reward))

            if replay_memory.is_sample_available(BATCH_SIZE) and len(replay_memory) >= MIN_MEMORY_SIZE:  # Train NN
                has_training_started = True

                experiences = replay_memory.get_sample(BATCH_SIZE)

                #  Handling the states, actions, next_states and rewards from batch...

                states, actions, next_states, rewards = zip(*experiences)

                #  Staking the states into following format: (BATCH_SIZE, FRAMES_PER_STATE, height, width)
                states = np.stack(states)
                next_states = np.stack(next_states)

                #  Converting them to tensors
                states = torch.tensor(states, dtype=torch.float32, device=DEVICE)
                actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
                next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)

                #  get nn evaluation for current and next state
                states_eval = policy_net(states)  # output: 256x6, tensor
                next_states_eval = target_net(next_states)  # output: 256x6, tensor

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

        print(f"{episode + 1} ✅ ε={epsilon}")

        if has_training_started:
            if episode % TARGET_UPDATE == 0:
                #  Update the target network to match the policy network
                print("‼️TARGET NET UPDATED‼️")
                target_net.load_state_dict(policy_net.state_dict())

            #  Update epsilon
            epsilon = get_new_epsilon(epsilon)

        #  Update logging (used for plotting)
        episode_list.append(episode)
        reward_list.append(reward_sum)

    # Plotting at end of training
    plot_episode_vs_reward(episode_list, reward_list)


train()
