import torch
import unittest

from protclf.model.residual import ResidualBlock


class TestResidualBlock(unittest.TestCase):

    def test_forward(self):
        batch_size = 512
        in_channels = 128
        out_channels = 128
        num_samples = 120

        model = ResidualBlock(in_channels, out_channels)
        x = torch.rand(batch_size, in_channels, num_samples)
        y = model(x)

        self.assertEqual(y.shape, (batch_size, out_channels, num_samples))
        self.assertTrue(isinstance(y, torch.Tensor))
