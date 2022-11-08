from src.dqn.net import DQN
from src.pong_env_manager import PongEnvManager
from src.replay_memory import ReplayMemory
from src.agent import Agent
import torch


def load_nn_and_play(nn_path, device):
    #  Loading nn
    nn = DQN((4, 84, 84), 6)
    if device == "cuda":
        nn.load_state_dict(torch.load(nn_path))
        nn.to(torch.device("cuda"))
        nn.eval()
    else:
        nn.load_state_dict(torch.load(nn_path, map_location=torch.device("cpu")))
        nn.eval()

    env_manager = PongEnvManager(4, True)
    replay_memory = ReplayMemory(1)  # Size=1 because it is not really needed...
    agent = Agent(env_manager, replay_memory, device)

    max_frames = 10000
    for i in range(max_frames):
        agent.perform_action(0, nn)
