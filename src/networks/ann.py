import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, n_actions):
        super(ANN, self).__init__()
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(28224, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return self.net(x)
