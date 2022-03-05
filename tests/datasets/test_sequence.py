import os
import unittest

from protclf.datasets.sequence import SequenceDataset


class TestSequence(unittest.TestCase):

    def test_sequence(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(base_dir, "data")

        dataset = SequenceDataset(data_path, "train")
        self.assertEqual(len(dataset), 1)

        sequence, label = dataset[0]
        self.assertEqual(len(sequence), 20)
        self.assertEqual(label, 1)
