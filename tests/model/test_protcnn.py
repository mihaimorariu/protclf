import os
import torch
import unittest

from protclf.model import ProtCNN
from protclf.dataset import SequenceDataset


class TestProtCNN(unittest.TestCase):

    def test_forward(self):
        batch_size = 512
        num_unique_aminos = 22
        num_samples = 120
        num_unique_labels = 6141

        model = ProtCNN(num_unique_aminos, num_unique_labels)
        x = torch.rand(batch_size, num_unique_aminos, num_samples)
        y = model(x)

        self.assertEqual(y.shape, (batch_size, num_unique_labels))
        self.assertTrue(isinstance(y, torch.Tensor))

    def test_training_step(self):
        base_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(base_dir, "..", "dataset", "data")

        batch_size = 1
        max_epochs = 10

        dataset = SequenceDataset(data_dir, "train")
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size)
        model = ProtCNN(num_unique_aminos=dataset.get_num_unique_aminos(),
                        num_unique_labels=dataset.get_num_unique_labels())

        optimizers = model.configure_optimizers()
        optimizer = optimizers["optimizer"]
        lr_scheduler = optimizers["lr_scheduler"]

        initial_loss = 0
        has_converged = False

        for epoch in range(max_epochs):
            for batch_idx, batch in enumerate(data_loader):
                loss = model.training_step(batch, batch_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            if epoch == 0:
                initial_loss = float(loss)

            if float(loss) <= initial_loss * 0.1:
                has_converged = True
                break

        return self.assertTrue(has_converged)
