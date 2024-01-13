from enum import StrEnum
from typing import Any, Callable, Dict

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Conv1d, Linear, MaxPool1d, Module, Sequential
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

from protclf.model.residual import ResidualBlock


class EvaluationMode(StrEnum):
    VALID = "valid"
    TEST = "test"


class Lambda(Module):
    def __init__(self, func: Callable):
        """
        Implements a lambda layer.

        Args:
            func: Lamba function to be called
        """
        super().__init__()
        self.func = func

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass.

        Args:
            x: Input tensor

        Returns:
            Tensor: Tensor containing the result of calling func(x)
        """
        return self.func(x)


class ProtCNN(LightningModule):
    def __init__(
        self,
        num_unique_aminos: int,
        num_unique_labels: int,
    ) -> None:
        """
        Implements the ProtoCNN model:
        https://www.biorxiv.org/content/10.1101/626507v3.full

        Args:
            num_unique_aminos: Number of unique amino acid codes
            num_unique_labels: Number of unique family labels
        """
        super().__init__()
        self.model = Sequential(
            Conv1d(
                num_unique_aminos,
                128,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            ResidualBlock(
                128,
                128,
                dilation=2,
            ),
            ResidualBlock(
                128,
                128,
                dilation=3,
            ),
            MaxPool1d(
                3,
                stride=2,
                padding=1,
            ),
            Lambda(lambda x: x.flatten(start_dim=1)),
            Linear(
                7680,
                num_unique_labels,
            ),
        )

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_unique_labels,
        )
        self.eval_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_unique_labels,
        )
        self.metric = torch.nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """
        Performs a forward pass.

        Args:
            x: Input tensor of shape
                [batch_size, num_unique_labels, max_seq_len]

        Returns:
            Tensor: Output tensor
        """
        return self.model(x.float())

    def training_step(  # type: ignore[override]
        self,
        batch: Dict[str, Tensor],
        __batch_idx__: int,
    ) -> Tensor:
        """
        Performs one forward pass on the given batch and returns the loss.

        Args:
            batch: Input data batch
            batch_idx: (Optional) Index of the current batch

        Returns:
            Tensor: Tensor containing the resulting loss
        """

        x, y = batch["sequence"], batch["label"]
        y_hat = self.forward(x)
        pred = torch.argmax(y_hat, dim=1)

        loss = self.metric(y_hat, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
        )

        self.train_acc(pred, y)
        self.log(
            "train_acc",
            self.train_acc,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def __shared_eval_step(
        self,
        batch: Dict[str, Tensor],
        mode: EvaluationMode,
    ) -> Tensor:
        # Performs one forward pass and returns the computed accuracy. This
        # method is called by both the validation and the testing step.
        x, y = batch["sequence"], batch["label"]
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)

        acc = self.eval_acc(pred, y)
        self.log(
            mode + "_acc",
            self.eval_acc,
            on_step=False,
            on_epoch=True,
        )

        return acc

    def validation_step(  # type: ignore[override]
        self,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """
        Performs one forward pass for validation and returns the accuracy.

        Args:
            batch: Input data batch

        Returns:
            Tensor: Tensor containing the resulting accuracy
        """
        return self.__shared_eval_step(batch, EvaluationMode.VALID)

    def test_step(  # type: ignore[override]
        self,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """
        Performs one forward pass for evaluation and returns the accuracy.

        Args:
            batch: Input data batch

        Returns:
            Tensor: Tensor containing the resulting accuracy
        """

        return self.__shared_eval_step(batch, EvaluationMode.TEST)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Creates the optimizer and the LR scheduler.

        Returns:
            Tuple: Tuple containing the optimizer and LR scheduler objects
        """
        optimizer = SGD(
            self.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=1e-2,
        )
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=[5, 8, 10, 12, 14, 16, 18, 20],
            gamma=0.9,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
