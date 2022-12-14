import torch.nn as nn


class NatureCNN(nn.Module):
    """
    DQN according to Google DeepMind

    Attributes:
        in_wh: Height and width of input images
        n_actions: Number of output

    Note that in/out comments is for a 84x84 input image
    """
    def __init__(self, in_wh, n_actions):
        super(NatureCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),  # in: 4x84x84, out: 32x20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # in: 32x20x20, out: 64x9x9
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # in: 64x9x9, out: 64x7x7
            nn.ReLU()
        )

        fc_input_size = (((((in_wh-8)//4+1)-4)//2 + 1)-3) + 1
        self.fc = nn.Sequential(
            nn.Linear(64*fc_input_size*fc_input_size, 512),  # 64 * 7 * 7 = 3136
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
