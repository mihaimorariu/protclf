import torch
import pytorch_lightning as pl
import torchmetrics
from protclf.model.residual import ResidualBlock
from typing import Callable, Dict, List
from argparse import ArgumentParser


class Lambda(torch.nn.Module):

    def __init__(self, func: Callable):
        """
        Implements a lambda layer.

        Args:
            func (function): Lamba function to be called.
        """
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the result of calling func(x).
        """
        return self.func(x)


class ProtCNN(pl.LightningModule):

    def __init__(self,
                 num_classes: int,
                 learning_rate: float = 1e-2,
                 weight_decay: float = 1e-2,
                 momentum: float = 0.9,
                 gamma: float = 0.9,
                 milestones: List[int] = None) -> None:
        """
        Implements the ProtoCNN model:
        https://www.biorxiv.org/content/10.1101/626507v3.full

        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(22, 128, kernel_size=1, padding=0, bias=False),
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(7680, num_classes))

        # Save the learning hyperparameters in self.hparams.
        self.save_hyperparameters()

        self.train_acc = torchmetrics.Accuracy()
        self.eval_acc = torchmetrics.Accuracy()
        self.metric = torch.nn.CrossEntropyLoss()

    @staticmethod
    def add_model_specific_args(parent: ArgumentParser) -> ArgumentParser:
        """
        Adds model-specific arguments to an existing argument parser.

        Args:
            parent (argparse.ArgumentParser): Parent parser.

        Returns:
            argparse.ArgumentParser: Parser with the new arguments.
        """
        parser = parent.add_argument_group("ProtCNN")
        parser.add_argument("--learning_rate", "-l", type=float, default=1e-2)
        parser.add_argument("--weight_decay", "-w", type=float, default=1e-2)
        parser.add_argument("--momentum", "-m", type=float, default=0.9)
        parser.add_argument("--gamma", "-g", type=float, default=0.9)
        parser.add_argument("--milestones",
                            "-t",
                            nargs="+",
                            default=[5, 8, 10, 12, 14, 16, 18, 20])
        return parent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape
                [batch_size, num_unique_labels, max_seq_len].

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x.float())

    def training_step(self, batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """
        Performs one forward pass on the given batch and returns the loss.

        Args:
            batch (Dict[str, torch.Tensor]): Input data batch.
            batch_idx (int): (Optional) Index of the current batch.

        Returns:
            torch.Tensor: Tensor containing the resulting loss.
        """

        x, y = batch["sequence"], batch["label"]
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)

        loss = self.metric(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        self.train_acc(pred, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        return loss

    def __shared_eval_step(self, batch: Dict[str, torch.Tensor],
                           mode: str) -> torch.Tensor:
        # Performs one forward pass and returns the computed accuracy. This
        # method is called by both the validation and the testing step.
        assert mode in ["valid", "test"]

        x, y = batch["sequence"], batch["label"]
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)

        acc = self.eval_acc(pred, y)
        self.log(mode + "_acc", self.eval_acc, on_step=False, on_epoch=True)

        return acc

    def validation_step(self, batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """
        Performs one forward pass for validation and returns the accuracy.

        Args:
            batch (Dict[str, torch.Tensor]): Input data batch.
            batch_idx (int): (Optional) Index of the current batch.

        Returns:
            torch.Tensor: Tensor containing the resulting accuracy.
        """
        return self.__shared_eval_step(batch, "valid")

    def test_step(self, batch: Dict[str, torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        """
        Performs one forward pass for evaluation and returns the accuracy.

        Args:
            batch (Dict[str, torch.Tensor]): Input data batch.
            batch_idx (int): (Optional) Index of the current batch.

        Returns:
            torch.Tensor: Tensor containing the resulting accuracy.
        """

        return self.__shared_eval_step(batch, "test")

    def configure_optimizers(self):
        """
        Creates the optimizer and the LR scheduler.

        Returns:
            dict: Dictionary containing the optimizer and LR scheduler objects.
        """
        optimizer = torch.optim.SGD(self.parameters(),
                                    lr=self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.gamma)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
