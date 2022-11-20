import torch.nn as nn


class DuelingCNN(nn.Module):
    """
    Only for input: 4x84x84, which is typical for atari games
    """
    def __init__(self, in_wh, n_actions):
        super(DuelingCNN, self).__init__()
        #  Convolutional layer, same for both calculating A and V
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),  # in: 4x84x84, out: 32x20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # in: 32x20x20, out: 64x9x9
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # in: 64x9x9, out: 64x7x7
            nn.ReLU()
        )

        fc_input_size = (((((in_wh - 8) // 4 + 1) - 4) // 2 + 1) - 3) + 1

        #  Used only for calculating V
        self.fc_value = nn.Sequential(
            nn.Linear(64 * fc_input_size * fc_input_size, 512),  # 64 * 7 * 7 = 3136
            nn.ReLU(),
            nn.Linear(512, 1)  # Value for state, there is only 1 state for one forward-pass
        )

        #  Used only for calculating A
        self.fc_advantage = nn.Sequential(
            nn.Linear(64 * fc_input_size * fc_input_size, 512),  # 64 * 7 * 7 = 3136
            nn.ReLU(),
            nn.Linear(512, n_actions)  # Advantage for each action, n_actions total.
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)  # Both V and A calculated using conv

        # V and A go separate ways in the network
        value = self.fc_value(conv_out)  # Output is V(s)
        advantage = self.fc_advantage(conv_out)  # Output is A(s,a)
        q = value + advantage - advantage.mean()  # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return q
