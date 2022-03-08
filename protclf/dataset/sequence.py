import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import torch

from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from typing import List, Dict
from pandas.core.frame import DataFrame, Series
from collections import Counter


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
        self.split = split

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
        """

        Returns:
            int: The number of unique family labels in the dataset.
        """
        return len(self.label2id)

    def get_num_unique_aminos(self) -> int:
        """

        Returns:
            int: The number of unique amino acid codes in the dataset.
        """
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
        """
        Returns the item with a given index from the dataset.

        Args:
            index (int): Index value.

        Returns:
            dict: Dictionary containing the amino acid sequence (represented as
                one-hot encoding) and its corresponding family label (as int).
        """
        amino_ids_one_hot = self.__convert_to_one_hot(
            self.sequences.iloc[index])
        label_id = self.label2id.get(self.labels.iloc[index],
                                     self.label2id['<unk>'])
        label_id = torch.as_tensor(label_id)

        return {"sequence": amino_ids_one_hot, "label": label_id}

    # Note: the methods below could also be moved outside of the class should
    # there be some assumptions about the shape and type of the data we are
    # dealing with. For example:
    # - They could be moved to a common/visual.py module which contained some
    #   helper methods for plotting data.
    # - They could be moved to a base class (e.g., BaseSequenceDataset) and
    #   have a CSVSequenceDataset derive from it.

    def plot_label_distribution(self) -> None:
        _, ax = plt.subplots(figsize=(8, 5))

        sorted_labels = self.labels.groupby(
            self.labels).size().sort_values(ascending=False)
        sns.histplot(sorted_labels.values, kde=True, log_scale=True, ax=ax)

        plt.title("Distribution of family sizes for the '" + self.split +
                  "' split")
        plt.xlabel("Family size (log scale)")
        plt.ylabel("# Families")
        plt.show()

    def plot_seq_len_distribution(self) -> None:
        _, ax = plt.subplots(figsize=(8, 5))

        sequence_lengths = self.sequences.str.len()
        median = sequence_lengths.median()
        mean = sequence_lengths.mean()

        sns.histplot(sequence_lengths.values,
                     kde=True,
                     log_scale=True,
                     bins=60,
                     ax=ax)

        ax.axvline(mean, color='r', linestyle='-', label=f"Mean = {mean:.1f}")
        ax.axvline(median,
                   color='g',
                   linestyle='-',
                   label=f"Median = {median:.1f}")

        plt.title("Distribution of sequence lengths")
        plt.xlabel("Sequence' length (log scale)")
        plt.ylabel("# Sequences")
        plt.legend(loc="best")
        plt.show()

    def plot_amino_distribution(self) -> None:
        _, ax = plt.subplots(figsize=(8, 5))

        def get_amino_acid_frequencies(data):
            aa_counter = Counter()

            for sequence in data:
                aa_counter.update(sequence)

            return pd.DataFrame({
                'AA': list(aa_counter.keys()),
                'Frequency': list(aa_counter.values())
            })

        amino_acid_counter = get_amino_acid_frequencies(self.sequences)

        sns.barplot(x='AA',
                    y='Frequency',
                    data=amino_acid_counter.sort_values(by=['Frequency'],
                                                        ascending=False),
                    ax=ax)

        plt.title("Distribution of AAs' frequencies in the '" + self.split +
                  "' split")
        plt.xlabel("Amino acid codes")
        plt.ylabel("Frequency (log scale)")
        plt.yscale("log")
        plt.show()
