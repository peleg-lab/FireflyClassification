import os

import ast
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import wrangling


class RealFlashPatterns(Dataset):
    def __init__(self, data_root, num_species, augmentations, n_classes, data_path):
        super().__init__()
        self._data_root = data_root
        self._data_path = data_path
        self._num_species = num_species
        self.augmentations = augmentations
        self.n_classes = n_classes
        self._load_flash_data(False)

    def _load_flash_data(self, ignore_single_flashes=True):
        # I think the data path now can be any old file containing all the combined data
        # that means data dir is either a file or a dir, and if it is a dir we get everything inside at once
        data_path = self._data_root
        if not ignore_single_flashes:
            # expects one file, so we need to make sure all the raw data are saved in one file
            # or we make it create one df from a folder path by updating the extract_from_csv logic
            df_cleaned, combined = wrangling.extract_from_csv(data_path)
            df_cleaned = wrangling.calculate_eccentricity(df_cleaned)
            j_to_eccentricity = df_cleaned.set_index('j')['ecc'].to_dict()
            updated_tuples = [
                tup + (j_to_eccentricity.get(tup[wrangling.r_map['traj']], None),)
                for tup in combined
            ]
            combined = updated_tuples

            combined_df, df_cleaned = wrangling.trim_and_collate(combined, df_cleaned)
            data = combined_df
        else:
            data = pd.read_csv(data_path)

        assert self.n_classes == len(data['species_label'].value_counts()), \
            'Mismatch detected between expected number of classes {} and number of classes in training set {}'.format(
                self.n_classes, len(data['species_label'].value_counts())
            )

        if ignore_single_flashes:
            data = data.loc[data['num_flashes'] > 1]
            print('Ignoring sequences with < 2 flashes in the dataset')
        else:
            print('Accounting for sequences of all flash counts')

        features = ['flash_length', 'flash_gap', 'flash_count', 'v', 'totm', 'ecc', 'timeseries']
        flash_df = data[['species', 'species_label', 'Dataset']]

        feature_data = data[features]

        multidimensional_data, flash_df = self._embed(feature_data, flash_df)

        self._data = multidimensional_data
        self._meta_data = flash_df

    @staticmethod
    def _embed(feature_data, flash_df):
        timeseries = feature_data['timeseries']
        numerical_features = feature_data.drop(columns=['timeseries']).values.astype(float)

        word_set = set([1.0, 0.0])
        word_list = list(word_set) + ['<pad>']
        word2idx = {word: idx for idx, word in enumerate(word_list)}
        x = [torch.FloatTensor([word2idx[float(i)] for i in seq.split(',')]) for seq in timeseries.values]

        x_padded = pad_sequence(x, batch_first=True, padding_value=word2idx['<pad>'])
        seq_lens = torch.LongTensor(list(map(len, x)))
        binary_data = np.array(x_padded)

        combined_data = np.hstack((numerical_features, binary_data))

        flash_idxs = np.where(binary_data.sum(axis=1) != 0)[0]
        combined_data = combined_data[flash_idxs]

        if len(flash_idxs) > len(flash_df):
            flash_df = flash_df.iloc[flash_idxs[:-1]]
        else:
            flash_df = flash_df.iloc[flash_idxs]

        return combined_data, flash_df

    def __len__(self):
        return self._data['timeseries'].shape[0]

    def __getitem__(self, idx):
        X = self._data[idx]
        timeseries = X.pop('timeseries')
        species_label = self._meta_data.iloc[idx].species_label
        species_name = self._meta_data.iloc[idx].species

        # Convert to tensor
        X = torch.tensor(X).float()

        return X, timeseries, species_label, species_name

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
