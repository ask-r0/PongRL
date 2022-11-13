import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.gym_env_manager import GymEnvManager
from src.utils.replay_memory import ReplayMemory
from src.utils.agent import Agent
import src.networks.nature_cnn as nature_cnn
import src.networks.ann as ann
import json
import time


def get_new_epsilon(epsilon, epsilon_min, epsilon_decay):
    """Returns new epsilon based on current epsilon"""
    return max(epsilon_min, epsilon - 1.0 / epsilon_decay)


def get_loss(batch, policy_net, target_net, gamma, device, loss_function):
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
    expected_state_action_values = next_state_values * gamma + rewards_t

    if loss_function == "huber":
        return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
    else:
        return nn.MSELoss()(state_action_values, expected_state_action_values)


def training_loop(env_name, device, memory_size, epsilon_initial, learning_rate, max_frames, memory_size_min, batch_size,
                  gamma, target_net_update, epsilon_min, epsilon_decay, logging_rate, save_to_file, model_path,
                  params_path, optimizer, loss_function, network):
    # Logging
    rewards = []
    frames = []

    #  Env setup
    env_manager = GymEnvManager(4, env_name, False, True, 33, 15)
    num_actions = env_manager.get_num_actions()

    if network == "ann":
        policy_net = ann.DQN(num_actions).to(device)
        target_net = ann.DQN(num_actions).to(device)
    else:
        policy_net = nature_cnn.DQN(num_actions).to(device)
        target_net = nature_cnn.DQN(num_actions).to(device)

    replay_memory = ReplayMemory(memory_size)
    agent = Agent(env_manager, replay_memory, device)
    epsilon = epsilon_initial

    # Optimizer
    if optimizer == "rmsprop":
        opt = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
    else:
        opt = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Training loop
    time_start = time.time()
    for i in range(max_frames):
        reward = agent.perform_action(epsilon, target_net)
        if reward is not None:  # Episode is done, and reward_sum for episode is returned
            rewards.append(reward)
            frames.append(i)

        if len(replay_memory) >= memory_size_min:  # Perform training
            batch = replay_memory.get_sample(batch_size)
            loss = get_loss(batch, policy_net, target_net, gamma, device, loss_function)
            loss.backward()
            opt.step()
            opt.zero_grad()

            if i % target_net_update == 0:  # Sync target_net with policy_net
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = get_new_epsilon(epsilon, epsilon_min, epsilon_decay)

        if i % logging_rate == 0:
            if save_to_file:
                #  save nn
                torch.save(policy_net.state_dict(), model_path)

                #  save model params
                params = {
                    "total_frames": i,
                    "epsilon": epsilon,
                    "frames": frames,
                    "rewards": rewards
                }
                with open(params_path, "w") as f:
                    json.dump(params, f)

            fps = logging_rate / (time.time() - time_start)
            time_start = time.time()
            print(f"{i} ✅ ε={epsilon}, {fps:.2f} fps")


def train_from_settings(settings_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(settings_path, "r") as f:
        s = json.load(f)

    training_loop(s["env_name"], device, s["memory_size"], s["epsilon_initial"], s["learning_rate"], s["max_frames"],
                  s["memory_size_min"], s["batch_size"], s["gamma"], s["target_net_update"], s["epsilon_min"],
                  s["epsilon_decay"], s["logging_rate"], s["save_to_file"], s["model_path"], s["params_path"],
                  s["optimizer"], s["loss_function"], s["network"])


