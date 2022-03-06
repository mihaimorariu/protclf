import torch
import unittest

from protclf.model import ProtCNN


class TestProtCNN(unittest.TestCase):

    def test_forward(self):
        batch_size = 512
        num_word2id = 22
        num_samples = 120
        num_fam2lab = 6141

        model = ProtCNN(num_fam2lab)
        x = torch.rand(batch_size, num_word2id, num_samples)
        y = model(x)

        self.assertEqual(y.shape, (batch_size, num_fam2lab))
        self.assertTrue(isinstance(y, torch.Tensor))
