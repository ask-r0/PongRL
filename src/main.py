from src.algorithms.dqn import train_from_settings


def main():
    # load_nn_and_play("src/dqn/storage/nn.pth", "cpu")
    train_from_settings("../settings/pong_dqn.json")
    pass


if __name__ == "__main__":
    main()
