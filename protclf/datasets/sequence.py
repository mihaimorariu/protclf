import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, data_path, split, **kwargs):
        assert split in ["train", "dev", "test"]

        col_names = ["sequence", "family_accession"]
        all_data = self.__load_data(data_path, split, col_names)

        self.max_seq_len = kwargs.get("max_seq_len", 120)
        self.sequences = all_data["sequence"]
        self.families = all_data["family_accession"]

        self.word2id = self.__create_word2id_dict(self.sequences)
        self.fam2lab = self.__create_fam2lab_dict(self.families)

    def __load_data(self, data_path, split, col_names=None):
        all_data = []

        for fn in os.listdir(os.path.join(data_path, split)):
            with open(os.path.join(data_path, split, fn)) as file:
                csv = pd.read_csv(file, index_col=None, usecols=col_names)
                all_data.append(csv)

        all_data = pd.concat(all_data)
        return all_data

    def __create_fam2lab_dict(self, families):
        unique_families = self.families.unique()
        labels = {f: i for i, f in enumerate(unique_families, start=1)}
        labels["<unk>"] = 0

        return labels

    def __create_word2id_dict(self, sequences):
        vocabulary = set()
        rare_aminos = {"X", "U", "B", "O", "Z"}

        for s in sequences:
            vocabulary.update(s)

        unique_aminos = sorted(vocabulary - rare_aminos)
        mapping = {w: i for i, w in enumerate(unique_aminos, start=2)}
        mapping["<pad>"] = 0
        mapping["<unk>"] = 1

        return mapping

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.__preprocess(self.sequences.iloc[index])
        label = self.fam2lab.get(self.families.iloc[index],
                                 self.fam2lab['<unk>'])
        return sequence, label

    def __preprocess(self, text):
        sequence = []

        for word in text[:self.max_seq_len]:
            sequence.append(self.word2id.get(word, self.word2id['<unk>']))

        if len(sequence) < self.max_seq_len:
            sequence += [self.word2id['<pad>']
                         ] * (self.max_seq_len - len(sequence))

        # Convert list into tensor
        sequence = torch.from_numpy(np.array(sequence))

        # One-hot encode
        sequence_one_hot = one_hot(sequence, num_classes=len(self.word2id))

        # Permute channel (one-hot) dim first
        sequence_one_hot = sequence_one_hot.permute(1, 0)

        return sequence_one_hot
