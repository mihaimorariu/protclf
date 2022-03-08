import torch
import unittest

from protclf.model import ProtCNN


class TestProtCNN(unittest.TestCase):

    def test_forward(self):
        batch_size = 512
        num_unique_aminos = 22
        num_samples = 120
        num_unique_labels = 6141

        model = ProtCNN(num_unique_labels)
        x = torch.rand(batch_size, num_unique_aminos, num_samples)
        y = model(x)

        self.assertEqual(y.shape, (batch_size, num_unique_labels))
        self.assertTrue(isinstance(y, torch.Tensor))
