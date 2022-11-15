import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import poisson
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from modules.noise import NoiseHandler


class PreGeneratedFlashPatterns(Dataset):
    def __init__(self, data_tensor, meta_data):
        super().__init__()
        self._data = data_tensor
        self._meta_data = meta_data

    @property
    def species_names(self):
        return list(self._meta_data.species)

    def __len__(self):
        return self._data.shape[0] * self._data.shape[1]

    def __getitem__(self, idx):
        outer_idx = idx % self._data.shape[0]
        inner_idx = idx % self._data.shape[1]
        species_name = self._meta_data.iloc[outer_idx].species
        species_label = self._meta_data.iloc[outer_idx].name
        return self._data[outer_idx][inner_idx], species_label, species_name


class FlashPatterns(Dataset):
    def __init__(self, data_root='data', augmentations={}, class_limit=None):
        super().__init__()
        self._data_root = data_root
        self._class_limit = class_limit
        self._noise_handler = NoiseHandler(augmentations)
        self._load_flash_data()

    @property
    def species_names(self):
        return list(self._meta_data.species)

    @property
    def n_species(self):
        return self._data.shape[0]

    @property
    def n_timesteps(self):
        return self._data.shape[2]

    def _load_flash_data(self, ignore_no_flash_species=True, ignore_single_flashes=True):
        if self._class_limit == 5:
            p = 'real_data/binary_sequences_all.csv'
            mdp = 'real_data/flash_data_all.csv'
            pdp = 'params_5species.csv'
        elif self._class_limit == 4:
            p = 'real_data/binary_sequences_4.csv'
            mdp = 'real_data/flash_data_4.csv'
            pdp = 'params_4species.csv'
        else:
            raise ValueError('Unsupported number of classes! (can only support 4 or 5 as of 08/09)')
            # TODO: subset into the larger one
        # Read binary sequences
        binary_sequence_path = os.path.join(self._data_root, p)
        binary_df = pd.read_csv(binary_sequence_path, header=None)

        # Read flash info
        flash_data_path = os.path.join(self._data_root, mdp)
        flash_df = pd.read_csv(flash_data_path)

        meta_df_path = os.path.join(self._data_root, pdp)
        meta_df = pd.read_csv(meta_df_path, header=None, names=['num_flashes', 'flash_length', 'ipi', 'id'])
        if ignore_single_flashes:
            binary_df = binary_df.loc[meta_df['num_flashes'] > 1]
            flash_df = flash_df.loc[meta_df['num_flashes'] > 1]
            print('Ignoring sequences with < 2 flashes in the dataset')
        binary_data = binary_df.values

        if ignore_no_flash_species:
            flash_idxs = np.where(binary_data.sum(axis=1) != 0)[0]
            binary_data = binary_data[flash_idxs]

            self._no_flash_species = flash_df[flash_df.num_flashes == 0]
            flash_df = flash_df[flash_df.num_flashes > 0]
            flash_df = flash_df.reset_index(drop=True)

        if self._class_limit is not None:
            keep_idxs = np.random.choice(list(range(len(binary_data))),
                                         self._class_limit,
                                         replace=False)
            binary_data = binary_data[keep_idxs]
            flash_df = flash_df.iloc[keep_idxs]

        self._data = binary_data
        self._meta_data = flash_df

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, idx):
        X = self._data[idx]
        species_name = self._meta_data.iloc[idx].species

        # Convert to tensor
        X = torch.tensor(X).float()

        return X, idx, species_name


class VariableLengthFlashPatterns(FlashPatterns):
    def __init__(self, data_root='data', augmentations={}, class_limit=None, num_samples=100, specific_class=None,
                 real_distrib=0):
        self.real_distrib = real_distrib  # ?
        super().__init__(data_root, augmentations, class_limit)
        self.seq_lens = None
        self._noisify(num_samples, specific_class)

    def _print(self, idx):
        X = self._data[idx]
        species_name = self._meta_data.iloc[idx].species
        print(species_name, X)

    def get_seq_lens(self):
        return self.seq_lens

    def _load_flash_data(self, ignore_no_flash_species=False, ignore_single_flashes=True):
        if self._class_limit == 5:
            p = 'real_data/binary_sequences_all.csv'
            mdp = 'real_data/flash_data_all.csv'
            pdp = 'params_5species.csv'
        elif self._class_limit == 4:
            p = 'real_data/binary_sequences_4.csv'
            mdp = 'real_data/flash_data_4.csv'
            pdp = 'params_4species.csv'
        else:
            raise ValueError('Unsupported number of classes! (can only support 4 or 5 as of 08/09)')
            # TODO: subset into the larger one

        binary_sequence_path = os.path.join(self._data_root, p)
        binary_df = pd.read_csv(binary_sequence_path, header=None, sep='\n')

        # Read flash info
        flash_data_path = os.path.join(self._data_root, mdp)
        flash_df = pd.read_csv(flash_data_path)

        meta_df_path = os.path.join(self._data_root, pdp)
        meta_df = pd.read_csv(meta_df_path, header=None, names=['num_flashes', 'flash_length', 'ipi', 'id'])

        if ignore_single_flashes:
            binary_df = binary_df.loc[meta_df['num_flashes'] > 1]
            flash_df = flash_df.loc[meta_df['num_flashes'] > 1]
            print('Ignoring sequences with < 2 flashes in the dataset')
        binary_data = binary_df.values

        if ignore_no_flash_species:
            flash_idxs = np.where(binary_data.sum(axis=1) != 0)[0]
            binary_data = binary_data[flash_idxs]

            self._no_flash_species = flash_df[flash_df.num_flashes == 0]
            flash_df = flash_df[flash_df.num_flashes > 0]
            flash_df = flash_df.reset_index(drop=True)

        self._data = binary_data
        self._meta_data = flash_df

        if self.real_distrib != 0:
            self._param_distrib = meta_df

    def _noisify(self, num_samples, specific_class):
        # generate, if specified, otherwise noisify
        if self._noise_handler.whether_to_generate:
            print('...generating {} samples/species...'.format(num_samples))
            # pad sequences
            word_set = set([1, 0])
            word_list = list(word_set) + ['<pad>']
            word2idx = {word: idx for idx, word in enumerate(word_list)}
            x = [torch.FloatTensor([word2idx[int(float(word))] for word in seq[0].split(',')]) for seq in self._data]
            x_padded = pad_sequence(x, batch_first=True, padding_value=word2idx['<pad>'])
            self.seq_lens = torch.LongTensor(list(map(len, x)))
            ml = max(self.seq_lens)
            new_data = torch.empty(size=(self.n_species, num_samples, ml))

        else:
            print('...noisifying {} samples...'.format(num_samples))
            x_padded = torch.tensor(self._data)
            new_data = torch.empty(size=(self.n_species, num_samples, self._data.shape[1]))

        if specific_class is None:
            for i in range(self.n_species):
                for j in range(num_samples):
                    # generate sequence here from meta data
                    X = x_padded[i].clone().detach()
                    if self._noise_handler.whether_to_generate:
                        if self.real_distrib != 0:
                            meta_data_i = self._param_distrib[self._param_distrib['id'] == i]
                            meta_data_dists = self._get_meta_data_dists(meta_data_i,
                                                                        num_samples)
                        else:
                            meta_data_i = self._meta_data.iloc(0)[i]
                            meta_data_dists = self._get_meta_data_dists(meta_data_i,
                                                                        num_samples)
                        Y = self._noise_handler._generate(len(X), meta_data_dists, do_fpi=False)
                        new_data[i][j] = torch.tensor(Y)
                    else:
                        meta_data_dists = {'ipi': meta_data_i.ipi,
                                           'fpi': meta_data_i.fpi,
                                           'fl': meta_data_i.flash_duration,
                                           'fnum': meta_data_i.num_flashes
                                           }
                        Y = self._noise_handler._generate(len(X), meta_data_dists, do_fpi=True)
                        new_data[i][j] = torch.tensor(Y)
            self._data = new_data.reshape(new_data.shape[0] * new_data.shape[1], new_data.shape[2])
            self._meta_data = self._meta_data[['species', 'species_label']]
            _md = pd.DataFrame(np.repeat(self._meta_data.values, num_samples, axis=0))
            _md.columns = self._meta_data.columns
            self._meta_data = _md
        else:
            new_data = torch.empty(size=(1, num_samples, ml))
            for i in specific_class:
                for j in range(num_samples):
                    # generate sequence here from meta data
                    X = x_padded[i].clone().detach()
                    if self._noise_handler.whether_to_generate:
                        if self.real_distrib != 0:  # sample from real distributions
                            meta_data_i = self._param_distrib[self._param_distrib['id'] == i]
                            meta_data_dists = self._get_meta_data_dists(meta_data_i,
                                                                        num_samples)
                        else:
                            # sample from gaussian distrib
                            meta_data_i = self._meta_data.iloc(0)[i]
                            meta_data_dists = self._get_meta_data_dists(meta_data_i,
                                                                        num_samples)
                        Y = self._noise_handler._generate(len(X), meta_data_dists, do_fpi=False)
                        new_data[0][j] = torch.tensor(Y)
                    else:
                        meta_data_dists = {'ipi': meta_data_i.ipi,
                                           'fpi': meta_data_i.fpi,
                                           'fl': meta_data_i.flash_duration,
                                           'fnum': meta_data_i.num_flashes
                                           }
                        Y = self._noise_handler._generate(len(X), meta_data_dists, do_fpi=True)
                        new_data[0][j] = torch.tensor(Y)
            self._data = new_data.reshape(new_data.shape[0] * new_data.shape[1], new_data.shape[2])
            self._meta_data = self._meta_data[['species', 'species_label']]
            if specific_class is not None:
                _md = pd.DataFrame(np.repeat(
                    self._meta_data.loc[self._meta_data['species_label'].isin(specific_class)].values,
                    num_samples, axis=0)
                )
            else:
                _md = pd.DataFrame(np.repeat(self._meta_data.values, num_samples, axis=0))
            _md.columns = self._meta_data.columns
            self._meta_data = _md
        print('done')

    def __len__(self):
        return self._data.shape[0]

    def _get_meta_data_dists(self, meta_data, num_samples):
        if self.real_distrib != 0:
            ipi = np.random.choice(meta_data['ipi'].to_numpy())
            fpi = 1  # no support for fpi
            flash_length = np.random.choice(meta_data['flash_length'].to_numpy())
            num_flashes = np.random.choice(meta_data['num_flashes'].to_numpy())
        else:
            ipi = np.random.normal(meta_data.ipi, meta_data.ipi_sd)
            fpi = np.random.normal(meta_data.fpi, meta_data.ipi_sd)
            flash_length = np.random.normal(meta_data.flash_duration, meta_data.duration_sd)

            # control for one-flashers
            if meta_data.num_flashes > 3:
                num_flashes = poisson.rvs(mu=meta_data.num_flashes, size=num_samples)[0]
            else:
                num_flashes = int(meta_data.num_flashes)

        # control for divide-by-0
        if num_flashes == 0:
            num_flashes += 1
        if ipi <= 0:
            ipi += 1
        if flash_length <= 0:
            flash_length += 1
        if fpi <= 0:
            fpi += 1

        # return dict of noisified durations
        return {'ipi': ipi,
                'fpi': fpi,
                'fl': flash_length,
                'fnum': num_flashes
                }

    def __getitem__(self, idx):
        X = self._data[idx]
        species_name = self._meta_data.iloc[idx].species

        # Convert to tensor
        X = torch.tensor(X).float()

        return X, idx, species_name


class RealFlashPatterns(Dataset):
    def __init__(self, data_root, num_species, augmentations, num_samples, n_classes):
        super().__init__()
        self._data_root = data_root
        self._num_species = num_species
        self.augmentations = augmentations
        self.num_samples = num_samples
        self.n_classes = n_classes
        self._noise_handler = NoiseHandler(augmentations)
        self._load_flash_data(True)

    def _load_flash_data(self, ignore_single_flashes=True):
        # Read flash info
        if self.n_classes == 5:
            p = 'real_data/binary_sequences_all.csv'
            mdp = 'real_data/flash_data_all.csv'
            pdp = 'params_5species.csv'
        elif self.n_classes == 4:
            p = 'real_data/binary_sequences_4.csv'
            mdp = 'real_data/flash_data_4.csv'
            pdp = 'params_4species.csv'
        else:
            raise ValueError('Unsupported number of classes! (can only support 4 or 5 as of 10/03)')

        binary_sequence_path = os.path.join(self._data_root, p)
        binary_df = pd.read_csv(binary_sequence_path, header=None, sep='\n')
        flash_data_path = os.path.join(self._data_root, mdp)
        flash_df = pd.read_csv(flash_data_path)

        meta_df_path = os.path.join(self._data_root, pdp)
        meta_df = pd.read_csv(meta_df_path, header=None, names=['num_flashes', 'flash_length', 'ipi', 'id'])
        if ignore_single_flashes:
            binary_df = binary_df.loc[meta_df['num_flashes'] > 1]
            flash_df = flash_df.loc[meta_df['num_flashes'] > 1]
            print('Ignoring sequences with < 2 flashes in the dataset')

        binary_data, flash_df = self._embed(binary_df, flash_df)

        self._data = binary_data
        self._meta_data = flash_df

    @staticmethod
    def _embed(bd, fd):
        word_set = set([1.0, 0.0])
        word_list = list(word_set) + ['<pad>']
        word2idx = {word: idx for idx, word in enumerate(word_list)}
        x = [torch.FloatTensor([word2idx[float(i)] for i in seq.split(',')]) for seq in bd[0].values]

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
