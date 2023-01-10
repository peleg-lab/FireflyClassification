import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

from modules.dataset import RealFlashPatterns


class FireflyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, augmentations, class_limit, batch_size, val_split, gen_seed, downsample):
        super().__init__()
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.class_limit = class_limit
        self.batch_size = batch_size
        self.gen_seed = gen_seed
        self.val_split = val_split
        self.downsample = downsample
        self.data_path = 'default'

        # 1. Create full dataset
        self.full = RealFlashPatterns(data_root=self.data_dir,
                                      num_species=self.class_limit,
                                      augmentations=augmentations,
                                      n_classes=self.class_limit)

        # 2. Split into train, val, test data sets
        self.train = None
        self.val = None
        self.test = None
        self.setup_datasets()

    def setup_datasets(self):
        self.train, self.val, self.test = self.train_test_val_split(self.full,
                                                                    bs=self.batch_size,
                                                                    downsample=self.downsample)

    @staticmethod
    def find_shortest(dataset, train_dataset, max_balance):
        train_class_balance = dataset[train_dataset.indices][1].value_counts()
        specific_classes = []
        for i in list(set(dataset[train_dataset.indices][1].values)):
            if train_class_balance[i] < max_balance:
                specific_classes.append(i)
        return specific_classes

    def downsample_dataset(self, dataset):
        np.random.seed(self.gen_seed)
        counts = dataset.dataset[dataset.indices][1].value_counts()
        min_balance = min(counts)
        new_indices = []

        for sp in np.unique(dataset.dataset._meta_data['species_label']):
            counter = min_balance
            np.random.shuffle(dataset.indices)
            good_indices = np.where(dataset.dataset[dataset.indices][1] == sp)[0]
            new_indices.extend(list(np.array(dataset.indices)[good_indices[:counter]]))

        return new_indices

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=True, shuffle=False)

    def train_test_val_split(self, dataset, bs, downsample):
        val_split = self.val_split if self.val_split is not None else 0
        gen_seed = self.gen_seed
        dataset_size = len(dataset)
        if val_split > 0:

            print('spitting into test/train/val sets... (split % = {})'.format(val_split * 100))
            n_train = int(dataset_size * (1 - (2 * val_split)))
            n_val = int(dataset_size * val_split)
            n_test = int(dataset_size * val_split)

            diff = dataset_size - (n_train + n_val + n_test)

            try:
                assert diff == 0
            except AssertionError:
                while diff > 0:
                    n_train += 1
                    diff = dataset_size - (n_train + n_val + n_test)
        else:
            n_train = 0
            n_val = 0
            n_test = len(dataset)
            if self.gen_seed is None:
                gen_seed = 42.0

        train_dataset, valid_dataset, test_dataset = random_split(
            dataset=dataset, lengths=[n_train, n_val, n_test], generator=torch.Generator().manual_seed(gen_seed)
        )

        print('done splitting data into {} train/{} validation/{} test sequences...'.format(len(train_dataset),
                                                                                            len(valid_dataset),
                                                                                            len(test_dataset)))
        if downsample != 0:
            print("Downsampling selected")
            train_dataset.indices = self.downsample_dataset(train_dataset)
            valid_dataset.indices = self.downsample_dataset(valid_dataset)
            test_dataset.indices = self.downsample_dataset(test_dataset)
            print("Downsampling done, {} samples per species in training set".format(
                len(train_dataset.indices) / self.class_limit)
            )

        return train_dataset, valid_dataset, test_dataset


class RealFireflyData(FireflyDataModule):
    def __init__(self, data_dir, augmentations, class_limit, batch_size, val_split, gen_seed, downsample):
        super().__init__(data_dir, augmentations, class_limit, batch_size, val_split, gen_seed, downsample)

    def train_test_val_split(self, dataset, bs, downsample):
        val_split = self.val_split if self.val_split is not None else 0
        gen_seed = self.gen_seed
        dataset_size = len(dataset)
        if val_split > 0:
            print('spitting into test/train/val sets... (split % = {})'.format(val_split * 100))
            n_train = int(dataset_size * (1 - (2 * val_split)))
            n_val = int(dataset_size * val_split)
            n_test = int(dataset_size * val_split)
            diff = dataset_size - (n_train + n_val + n_test)
            try:
                assert diff == 0
            except AssertionError:
                while diff > 0:
                    n_train += 1
                    diff = dataset_size - (n_train + n_val + n_test)
        else:
            n_train = 0
            n_val = 0
            n_test = len(dataset)
            if self.gen_seed is None:
                gen_seed = 42.0

        train_dataset, valid_dataset, test_dataset = random_split(
            dataset=dataset, lengths=[n_train, n_val, n_test], generator=torch.Generator().manual_seed(gen_seed)
        )
        if downsample != 0:
            print("Downsampling selected")
            former_indices = [x for x in train_dataset.indices]
            train_dataset.indices = self.downsample_dataset(train_dataset)
            s = set(train_dataset.indices)
            extra_test = [x for x in former_indices if x not in s]
            test_dataset.indices.extend(extra_test)

            print("Downsampling done, {} samples per species in training set".format(
                len(train_dataset.indices) / self.class_limit)
            )

        print('done splitting data into {} train/{} validation/{} test sequences...'.format(len(train_dataset),
                                                                                            len(valid_dataset),
                                                                                            len(test_dataset)))
        return train_dataset, valid_dataset, test_dataset
