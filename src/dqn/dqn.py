import torch
import torch.nn as nn
import torch.optim as optim
from src.pong_env_manager import PongEnvManager
from src.dqn.net import DQN
from src.replay_memory import ReplayMemory
from src.agent import Agent
import json

# STORAGE
SAVE_TO_FILE = True
MODEL_PATH = "storage/nn.pth"
PARAM_PATH = "storage/param.json"

# MODEL CONSTANTS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE BEING USED: {DEVICE}")

GAMMA = 0.99  # The discount rate. Close to 1: future rewards matter more. Close to 0: immediate rewards matter more
EPSILON_START = 1.0  # Initial epsilon
EPSILON_END = 0.02  # Epsilon is always greater or equal to this value
FRAMES_EPSILON_END = 140000  # Number of frames before hitting EPSILON_END
LEARNING_RATE = 1e-4

BATCH_SIZE = 32  # Number of experiences as input to nn for each optimization
MEMORY_SIZE = 10000  # Maximum amount of experiences to hold
MIN_MEMORY_SIZE = 10000  # Training should not start before having MIN_MEMORY_SIZE experiences

NUM_EPISODES = 100  # Number of episodes for the training loop
MAX_FRAMES = 1_000_000  # Maximum amount of frames (total steps by agent)
LOGGING_RATE = 1000  # Amount of frames between loggings

TARGET_UPDATE_INTERVAL = 1000  # How often target net is updated to equal the policy net


def get_new_epsilon(cur_epsilon):
    """Returns new epsilon based on current epsilon"""
    return max(EPSILON_END, cur_epsilon - 1.0/FRAMES_EPSILON_END)


def get_loss(batch, policy_net, target_net, device):
    states, actions, rewards, dones, next_states = batch
    states_t = torch.tensor(states).to(device)
    next_states_t = torch.tensor(next_states).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    done_t = torch.ByteTensor(dones).to(device)

    state_action_values = policy_net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    next_state_values = target_net(next_states_t).max(1)[0]
    next_state_values[done_t] = 0.0
    next_state_values = next_state_values.detach()
    expected_state_action_values = next_state_values * GAMMA + rewards_t
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def training_loop(device):
    # Plotting
    rewards = []
    episodes = []

    #  Env setup
    env_manager = PongEnvManager(4, False)
    policy_net = DQN((4, 84, 84), 6).to(device)
    target_net = DQN((4, 84, 84), 6).to(device)

    replay_memory = ReplayMemory(MEMORY_SIZE)
    agent = Agent(env_manager, replay_memory, device)
    epsilon = EPSILON_START

    # Optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # Training loop
    for i in range(MAX_FRAMES):
        reward = agent.perform_action(epsilon, target_net)
        if reward is not None:  # Episode is done, and reward_sum for episode is returned
            rewards.append(reward)
            episodes.append(i)

        if len(replay_memory) >= MIN_MEMORY_SIZE:  # Perform training
            batch = replay_memory.get_sample_v2(BATCH_SIZE)
            loss = get_loss(batch, policy_net, target_net, device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i % TARGET_UPDATE_INTERVAL == 0:  # Sync target_net with policy_net
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = get_new_epsilon(epsilon)

        if i % LOGGING_RATE == 0:
            print(f"{i} ✅ ε={epsilon}")

            if SAVE_TO_FILE:
                #  save nn
                torch.save(policy_net.state_dict(), MODEL_PATH)

                #  save model params
                params = {
                    "start_frame": i,
                    "epsilon": epsilon,
                    "episodes": episodes,
                    "rewards": rewards
                }
                with open(PARAM_PATH, "w") as f:
                    json.dump(params, f)


training_loop(DEVICE)
