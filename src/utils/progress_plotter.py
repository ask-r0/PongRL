import matplotlib.pyplot as plt
import numpy as np


def plot_frame_vs_reward(frames, rewards, plot_raw, plot_avg, avg_window=40):
    """ Plots episode (x-axis) v reward (y-axis)
    Attributes:
        frames: list of frame numbers. episodes[i] corresponds to rewards from rewards[i]
        rewards: list of rewards.
        avg_window: Window for plotting running average
    """
    # Plot data
    if plot_raw:
        plt.plot(frames, rewards)

    # Plot running average
    if plot_avg:
        avg_data = []
        for i in range(avg_window-1):
            avg_data.append(np.nan)

        for i in range(len(rewards) - avg_window + 1):
            avg_data.append(np.mean(rewards[i:i+avg_window]))

        plt.plot(frames, avg_data)

    # Final styling
    plt.title("Reward vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("Reward")
    plt.show()
