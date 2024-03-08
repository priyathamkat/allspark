import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForwardNet(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(d, d, 1)
        self.conv2 = nn.Conv1d(d, d, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x
