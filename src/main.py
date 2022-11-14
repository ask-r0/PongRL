from src.algorithms.dqn import train_from_settings
from src.utils.progress_plotter import plot_frame_vs_reward
from src.utils.play import load_nn_and_play_pong

def main():
    # load_nn_and_play_pong("../trained_networks/pong_dqn_cnn/nn.pth", "cnn", "cpu")
    train_from_settings("../settings/pong_v3.json")
    pass


if __name__ == "__main__":
    main()
