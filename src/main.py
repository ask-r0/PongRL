from src.algorithms.dql import train_from_settings
from utils.play import load_nn_and_play_pong
import sys


def main():
    args = sys.argv
    if len(args) < 2:
        print("Use commands train or play. See README.md for more info.")
        exit()

    if args[1] == 'play':
        net_path = args[2]
        net_type = args[3]
        num_frames = int(args[4])
        load_nn_and_play_pong(net_path, net_type, num_frames)
    elif args[1] == 'train':
        settings_path = args[2]
        train_from_settings(settings_path)
    else:
        print("Use commands train or play. See README.md for more info.")


if __name__ == "__main__":
    main()
