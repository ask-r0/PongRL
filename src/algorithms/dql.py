import torch
import torch.nn as nn
import torch.optim as optim
from utils.gym_env_manager import GymEnvManager
from utils.replay_memory import ReplayMemory
from utils.agent import Agent
from networks.nature_cnn import NatureCNN
from networks.dueling_cnn import DuelingCNN
import json
import time


def get_new_epsilon(epsilon, epsilon_min, epsilon_decay):
    """Returns new epsilon based on current epsilon"""
    return max(epsilon_min, epsilon - 1.0 / epsilon_decay)


def get_loss_ddqn(batch, policy_net, target_net, gamma, device, loss_function):
    """ Calculates loss with method used in Double DQN"""
    states, actions, rewards, dones, next_states = batch  # Gets attributes from batch
    # Transforming attributes to tensors
    states_t = torch.tensor(states).to(device)
    next_states_t = torch.tensor(next_states).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    done_t = torch.ByteTensor(dones).to(device)

    q = policy_net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)  # Q(s,a)
    next_state_actions = policy_net(next_states_t).max(1)[1]  # The action to be taken from next state

    #  Q(next_state, best action from next state according to policy net)
    next_state_q = target_net(next_states_t).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)

    next_state_q[done_t] = 0.0  # Only count the rewards for experiences where next_state was the last step
    next_state_q = next_state_q.detach()
    target_q = next_state_q * gamma + rewards_t

    if loss_function == "huber":
        return nn.SmoothL1Loss()(q, target_q)
    else:
        return nn.MSELoss()(q, target_q)


def get_loss_dqn(batch, policy_net, target_net, gamma, device, loss_function):
    """ Calculates loss with method used in DQN"""
    states, actions, rewards, dones, next_states = batch  # Gets attributes from batch
    #  Transforming attributrs to tensors
    states_t = torch.tensor(states).to(device)
    next_states_t = torch.tensor(next_states).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)
    done_t = torch.ByteTensor(dones).to(device)

    q = policy_net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)  # Q(s,a)
    next_states_max_q = target_net(next_states_t).max(1)[0]  # max Q-value for next state
    next_states_max_q[done_t] = 0.0  # Only count the rewards for experiences where next_state was the last step
    next_states_max_q = next_states_max_q.detach()
    target_q = next_states_max_q * gamma + rewards_t

    if loss_function == "huber":
        return nn.SmoothL1Loss()(q, target_q)
    else:
        return nn.MSELoss()(q, target_q)


def training_loop(env_name, device, memory_size, epsilon_initial, learning_rate, max_frames, memory_size_min, batch_size,
                  gamma, target_net_update, epsilon_min, epsilon_decay, logging_rate, save_to_file, model_path,
                  params_path, optimizer, loss_function, target_q_equation, network, preprocessing):
    """ Trains a neural network to play a game from the Gym library. See settings folder for parameters info. """
    # Logging
    frames = []  # Frames where episodes ended
    rewards = []  # Corresponding rewards for the episode
    speeds = []  # Corresponding speed for the episode

    #  Env setup
    env_manager = GymEnvManager(4, env_name, False, True, 33, 15, preprocessing)
    num_actions = env_manager.get_num_actions()

    if network == "dueling":
        policy_net = DuelingCNN(preprocessing, num_actions).to(device)
        target_net = DuelingCNN(preprocessing, num_actions).to(device)
        print("[INIT] DuelingCNN initialized.")
    else:
        policy_net = NatureCNN(preprocessing, num_actions).to(device)
        target_net = NatureCNN(preprocessing, num_actions).to(device)
        print("[INIT] NatureCNN initialized.")

    replay_memory = ReplayMemory(memory_size)
    agent = Agent(env_manager, replay_memory, device)
    epsilon = epsilon_initial

    # Optimizer
    if optimizer == "rmsprop":
        opt = optim.RMSprop(policy_net.parameters(), lr=learning_rate)
        print("[INIT] Optimizer RMSprop initialized.")
    else:
        opt = optim.Adam(policy_net.parameters(), lr=learning_rate)
        print("[INIT] Optimizer Adam initialized.")

    # Training loop
    time_start = time.time()
    for i in range(max_frames):
        reward = agent.perform_action(epsilon, target_net)
        if reward is not None:  # Episode is done, and reward_sum for episode is returned
            #  Logging
            frames.append(i)
            rewards.append(reward)
            speeds.append((frames[len(frames) - 1] - frames[len(frames) - 2]) / (time.time() - time_start))
            time_start = time.time()

        if len(replay_memory) >= memory_size_min:  # Perform training
            batch = replay_memory.get_sample(batch_size)

            # Calculate loss
            if target_q_equation == "ddqn":
                loss = get_loss_ddqn(batch, policy_net, target_net, gamma, device, loss_function)
            else:
                loss = get_loss_dqn(batch, policy_net, target_net, gamma, device, loss_function)

            # Update weights
            loss.backward()
            opt.step()
            opt.zero_grad()

            if i % target_net_update == 0:  # Sync target_net with policy_net
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = get_new_epsilon(epsilon, epsilon_min, epsilon_decay)  #  Update epsilon

        if i % logging_rate == 0:
            if save_to_file:
                #  save nn
                torch.save(policy_net.state_dict(), model_path)

                #  save model params
                params = {
                    "total_frames": i,
                    "epsilon": epsilon,
                    "frames": frames,
                    "rewards": rewards,
                    "speeds": speeds
                }
                with open(params_path, "w") as f:
                    json.dump(params, f)

            frames_idx = len(frames) - 1
            if frames_idx >= 0:
                print(f"{i} | ??={epsilon:.5f}, last episode ended @ frame={frames[frames_idx]}:"
                      f" fps={speeds[frames_idx]:.2f}, reward={rewards[frames_idx]}")


def train_from_settings(settings_path):
    """ Trains neural network from a settings file. See settings folder for more info."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] DEVICE={device}")
    with open(settings_path, "r") as f:
        s = json.load(f)

    training_loop(s["env_name"], device, s["memory_size"], s["epsilon_initial"], s["learning_rate"], s["max_frames"],
                  s["memory_size_min"], s["batch_size"], s["gamma"], s["target_net_update"], s["epsilon_min"],
                  s["epsilon_decay"], s["logging_rate"], s["save_to_file"], s["model_path"], s["params_path"],
                  s["optimizer"], s["loss_function"], s["target_q_equation"], s["network"], s["preprocessing"])
