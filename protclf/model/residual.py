import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Module, Sequential


class ResidualBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ) -> None:
        """
        Implements a residual block (https://arxiv.org/pdf/1512.03385.pdf).

        Args:
            in_channels: Number of channels (feature maps) of the
                incoming embedding
            out_channels: Number of channels after the first convolution
            dilation: Dilation rate of the first convolution (default is 1)
        """
        super().__init__()

        self.skip = Sequential()
        self.bn1 = BatchNorm1d(in_channels)
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            bias=False,
            dilation=dilation,
            padding=dilation,
        )
        self.bn2 = BatchNorm1d(out_channels)
        self.conv2 = Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            bias=False,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass.

        Args:
            x: Input tensor with shape
                [batch_size, in_channels, max_data_len]

        Returns:
            Tensor: Tensor containing the result of the residual block
                (shape is [batch_size, out_channels, max_data_len])
        """
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))
        return x2 + self.skip(x)
