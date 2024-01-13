import argparse

import torch
from pytorch_lightning import Trainer

from protclf.dataset import SequenceDataset
from protclf.model import ProtCNN


def main(args):
    test_dataset = SequenceDataset(args.data_dir, "test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    model = ProtCNN.load_from_checkpoint(args.checkpoint_file)
    model.eval()

    trainer = Trainer()
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_file",
        "-c",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=512,
    )
    args = parser.parse_args()

    main(args)
