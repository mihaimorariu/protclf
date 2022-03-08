import argparse
import torch
import pytorch_lightning as pl
from protclf.dataset import SequenceDataset
from protclf.model import ProtCNN


def main(args):
    pl.seed_everything(args.seed)

    train_dataset = SequenceDataset(args.data_dir, "train")
    train_dataset.plot_label_distribution()
    train_dataset.plot_seq_len_distribution()
    train_dataset.plot_amino_distribution()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8)

    dev_dataset = SequenceDataset(args.data_dir, "dev")
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=8)

    model = ProtCNN(
        num_classes=train_dataset.get_num_unique_labels(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        gamma=args.gamma,
        milestones=args.milestones
    )
    model.train()

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, dev_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=512)
    parser.add_argument("--seed", "-s", type=int, default=0)

    parser = ProtCNN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
