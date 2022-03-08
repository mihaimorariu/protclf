import argparse
import torch
import pytorch_lightning as pl
from protclf.datasets import SequenceDataset
from protclf.model import ProtCNN


def main(args):
    pl.seed_everything(args.seed)

    train_dataset = SequenceDataset(args.data_dir, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8)

    dev_dataset = SequenceDataset(args.data_dir, "dev")
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8)

    model = ProtCNN(num_classes=train_dataset.get_num_unique_labels())
    model.train()

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs)
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-i", required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=512)
    parser.add_argument("--seed", "-s", type=int, default=0)

    parser = ProtCNN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
