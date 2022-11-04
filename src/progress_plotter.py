import matplotlib.pyplot as plt
import numpy as np


def plot_episode_vs_reward(episodes, rewards, avg_window=40):
    """ Plots episode (x-axis) v reward (y-axis)
    Attributes:
        episodes: list of episode numbers. episodes[i] corresponds to rewards from rewards[i]
        rewards: list of rewards.
        avg_window: Window for plotting running average
    """
    # Plot data
    plt.plot(episodes, rewards)

    # Plot running average
    avg_data = []
    for i in range(avg_window-1):
        avg_data.append(np.nan)

    for i in range(len(rewards) - avg_window + 1):
        avg_data.append(np.mean(rewards[i:i+avg_window]))

    plt.plot(episodes, avg_data)

    # Final styling
    plt.title("Reward vs Episode")
    plt.xlabel("Episode number")
    plt.ylabel("Reward of episode")
    plt.show()
