import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(28224, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )

    def forward(self, x):
        return self.net(x)
