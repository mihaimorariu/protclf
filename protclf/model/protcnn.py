import torch
from protclf.model import ResidualBlock


class ProtCNN(torch.nn.Module):

    def __init__(self, num_classes: int) -> None:
        """
        Implements the ProtoCNN model:
        https://www.biorxiv.org/content/10.1101/626507v3.full

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.conv = torch.nn.Conv1d(22,
                                    128,
                                    kernel_size=1,
                                    padding=0,
                                    bias=False)
        self.res_blocks = torch.nn.Sequential(
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3))
        self.max_pool = torch.nn.MaxPool1d(3, stride=2, padding=1)
        self.fcn = torch.nn.Linear(7680, num_classes)

    def forward(self, x):
        h = self.conv(x.float())
        h = self.res_blocks(h)
        h = self.max_pool(h)
        y = self.fcn(h.flatten(start_dim=1))
        return y
