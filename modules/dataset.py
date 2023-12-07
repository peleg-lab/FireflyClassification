import os

import ast
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RealFlashPatterns(Dataset):
    def __init__(self, data_root, num_species, augmentations, n_classes, data_path):
        super().__init__()
        self._data_root = data_root
        self._data_path = data_path
        self._num_species = num_species
        self.augmentations = augmentations
        self.n_classes = n_classes
        self._load_flash_data(True)

    def _load_flash_data(self, ignore_single_flashes=True):
        # Read flash info
        dp = 'real_data/{}'.format(self._data_path)

        data_path = os.path.join(self._data_root, dp)
        data = pd.read_csv(data_path)

        assert self.n_classes == len(data['species_label'].value_counts()), \
            'Mismatch detected between expected number of classes {} and number of classes in training set {}'.format(
                self.n_classes, len(data['species_label'].value_counts())
            )

        if ignore_single_flashes:
            data = data.loc[data['num_flashes'] > 1]
            print('Ignoring sequences with < 2 flashes in the dataset')
        binary_df = data['timeseries']
        flash_df = data[['species', 'species_label', 'Dataset']]

        binary_data, flash_df = self._embed(binary_df, flash_df)

        self._data = binary_data
        self._meta_data = flash_df

    @staticmethod
    def _embed(bd, fd):
        word_set = set([1.0, 0.0])
        word_list = list(word_set) + ['<pad>']
        word2idx = {word: idx for idx, word in enumerate(word_list)}
        x = [torch.FloatTensor([word2idx[float(i)] for i in seq.split(',')]) for seq in bd.values]

        x_padded = pad_sequence(x, batch_first=True, padding_value=word2idx['<pad>'])
        seq_lens = torch.LongTensor(list(map(len, x)))
        binary_data = np.array(x_padded)
        flash_idxs = np.where(binary_data.sum(axis=1) != 0)[0]
        bd = binary_data[flash_idxs]
        if len(flash_idxs) > len(fd):
            fd = fd.iloc[flash_idxs[:-1]]
        else:
            fd = fd.iloc[flash_idxs]

        return bd, fd

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        X = self._data[idx]
        species_label = self._meta_data.iloc[idx].species_label
        species_name = self._meta_data.iloc[idx].species

        # Convert to tensor
        X = torch.tensor(X).float()

        return X, species_label, species_name

    @property
    def species_names(self):
        return list(self._meta_data.species)

    @property
    def n_species(self):
        return len(list(set(self._meta_data.species)))

    @property
    def n_timesteps(self):
        return self._data.shape[1]

    @property
    def dataset(self):
        return self._meta_data.Dataset
