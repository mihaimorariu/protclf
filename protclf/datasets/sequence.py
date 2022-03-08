import pandas as pd
import os
import torch

from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from typing import List, Dict
from pandas.core.frame import DataFrame, Series


class SequenceDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 split: str,
                 max_seq_len: int = 120) -> None:
        """
        Creates an instance of the sequence dataset.

        Args:
            data_dir (str): Path to the dataset directory.
            split (str): Subset of the dataset that needs to be loaded. It must
                take one of the following values: 'train', 'dev' or 'test'.
            max_seq_len (int): Maximum sequence length. Default value is 120.
        """
        if split not in ["train", "dev", "test"]:
            raise ValueError("'split' needs to take one of the following " +
                             "values: 'train', 'dev' or 'test'")

        col_names = ["sequence", "family_accession"]
        all_data = self.__load_from_csv(data_dir, split, col_names)

        self.max_seq_len = max_seq_len
        self.sequences = all_data["sequence"]
        self.labels = all_data["family_accession"]

        self.amino2id = self.__create_amino2id_dict(self.sequences)
        self.label2id = self.__create_label2id_dict(self.labels)

    def __load_from_csv(self,
                        data_dir: str,
                        split: str,
                        col_names: List[str] = None) -> DataFrame:
        # Loads data (filtered by given column names) from the CSV files into a
        # pandas DataFrame.
        all_data = []

        for fn in os.listdir(os.path.join(data_dir, split)):
            with open(os.path.join(data_dir, split, fn)) as file:
                csv = pd.read_csv(file, index_col=None, usecols=col_names)
                all_data.append(csv)

        all_data = pd.concat(all_data)
        return all_data

    def get_num_unique_labels(self) -> int:
        return len(self.label2id)

    def get_num_unique_aminos(self) -> int:
        return len(self.amino2id)

    def __create_label2id_dict(self, labels: Series) -> Dict[str, int]:
        # Maps a (family) label to an integer and returns the corresponding
        # dictionary.
        unique_labels = labels.unique()
        labels = {f: i for i, f in enumerate(unique_labels, start=1)}
        labels["<unk>"] = 0

        return labels

    def __create_amino2id_dict(self, sequences):
        # Maps a sequence (as str) to an ID (as int) and returns the
        # corresponding dictionary.
        vocabulary = set()
        rare_aminos = {"X", "U", "B", "O", "Z"}

        for s in sequences:
            vocabulary.update(s)

        unique_aminos = sorted(vocabulary - rare_aminos)
        mapping = {w: i for i, w in enumerate(unique_aminos, start=2)}
        mapping["<pad>"] = 0
        mapping["<unk>"] = 1

        return mapping

    def __len__(self) -> int:
        return len(self.sequences)

    def __convert_to_one_hot(self, sequence: str) -> torch.Tensor:
        # Converts a given amino acid sequence to a one-hot encoding. The
        # output is a torch.Tensor of shape [num_labels, n_samples].
        amino_ids = []

        for amino in sequence[:self.max_seq_len]:
            amino_ids.append(self.amino2id.get(amino, self.amino2id['<unk>']))

        if len(amino_ids) < self.max_seq_len:
            remaining = self.max_seq_len - len(amino_ids)
            amino_ids += [self.amino2id['<pad>'] for _ in range(remaining)]

        amino_ids = torch.as_tensor(amino_ids)
        amino_ids_one_hot = one_hot(amino_ids, num_classes=len(self.amino2id))
        amino_ids_one_hot = amino_ids_one_hot.permute(1, 0)

        return amino_ids_one_hot

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        amino_ids_one_hot = self.__convert_to_one_hot(
            self.sequences.iloc[index])
        label_id = self.label2id.get(self.labels.iloc[index],
                                     self.label2id['<unk>'])
        label_id = torch.as_tensor(label_id)

        return {"sequence": amino_ids_one_hot, "label": label_id}
