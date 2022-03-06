import torch
from torch.nn.functional import relu


class ResidualBlock(torch.nn.Module):
    """
    Implements a residual block (https://arxiv.org/pdf/1512.03385.pdf).

    Args:
        in_channels (int): Number of channels (feature maps) of the incoming
            embedding.
        out_channels (int): Number of channels after the first convolution.
        dilation (int): Dilation rate of the first convolution. Default value
            is 1.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int = 1) -> None:
        super().__init__()

        self.skip = torch.nn.Sequential()
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     bias=False,
                                     dilation=dilation,
                                     padding=dilation)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels,
                                     out_channels=out_channels,
                                     kernel_size=3,
                                     bias=False,
                                     padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(relu(self.bn2(x1)))
        return x2 + self.skip(x)
