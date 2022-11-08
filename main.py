from src.play import load_nn_and_play
from src.progress_plotter import plot_frame_vs_reward
def main():
    load_nn_and_play("src/dqn/storage/nn.pth", "cpu")

if __name__ == "__main__":
    main()
