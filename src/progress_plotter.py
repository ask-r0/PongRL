import matplotlib.pyplot as plt


def plot_episode_vs_reward(episodes, rewards):
    """ Plots episode (x-axis) v reward (y-axis)
    Attributes:
        episodes: list of episode numbers. episodes[i] corresponds to rewards from rewards[i]
        rewards: list of rewards.
    """
    plt.plot(episodes, rewards)
    plt.title("Reward vs Episode")
    plt.xlabel("Episode number")
    plt.ylabel("Reward of episode")
    plt.show()
