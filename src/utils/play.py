from utils.gym_env_manager import GymEnvManager
from utils.replay_memory import ReplayMemory
from utils.agent import Agent
import networks.nature_cnn as nature_cnn
import networks.dueling_cnn as dueling_cnn
import torch


def load_nn_and_play_pong(nn_path, nn_type, num_frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #  Loading nn
    if nn_type == "dueling":
        nn = dueling_cnn.DuelingCNN(84, 6).to(device)
    else:
        nn = nature_cnn.NatureCNN(84, 6).to(device)

    if device == "cuda":
        nn.load_state_dict(torch.load(nn_path))
        nn.to(torch.device("cuda"))
        nn.eval()
    else:
        nn.load_state_dict(torch.load(nn_path, map_location=torch.device("cpu")))
        nn.eval()

    env_manager = GymEnvManager(4, "PongNoFrameskip-v4", True, True, 33, 15, 84)
    replay_memory = ReplayMemory(1)  # Size=1 because it is not really needed...
    agent = Agent(env_manager, replay_memory, device)

    for i in range(num_frames):
        agent.perform_action(0, nn)
