import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Make a better model with CNN
class DQN(nn.Module):
    def __init__(self, image_height, image_width):
        super().__init__()

        self.fc1 = nn.Linear(in_features=image_width * image_height * 4, out_features=24)  # * 4 because 4 in stack
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=6)  # Final layer, 6 actions therefor 6 output nodes

    def forward(self, x):
        """
        Attributes:
            x: an image
        """
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
