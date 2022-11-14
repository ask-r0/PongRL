import torch.nn as nn


class NatureCNN(nn.Module):
    """
    Only for input: 4x84x84, which is typical for atari games
    """
    def __init__(self, n_actions):
        super(NatureCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),  # in: 4x84x84, out: 32x20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # in: 32x20x20, out: 64x9x9
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # in: 64x9x9, out: 64x7x7
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, 512),  # 64 * 7 * 7 = 3136
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
