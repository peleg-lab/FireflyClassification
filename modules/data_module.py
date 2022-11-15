import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

from modules.dataset import VariableLengthFlashPatterns, RealFlashPatterns


class FireflyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, augmentations, class_limit, batch_size, num_samples,
                 augment_train, val_split, gen_seed, downsample):
        super().__init__()
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.class_limit = class_limit
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.gen_seed = gen_seed
        self.val_split = val_split
        self.augment_train = augment_train
        self.downsample = downsample

        # 1. Create full dataset
        self.full = RealFlashPatterns(data_root=self.data_dir,
                                      num_species=self.class_limit,
                                      augmentations=augmentations,
                                      num_samples=num_samples,
                                      n_classes=self.class_limit)

        # 2. Split into train, val, test data sets
        self.train = None
        self.val = None
        self.test = None
        self.setup_datasets()

    def setup_datasets(self):
        self.train, self.val, self.test = self.train_test_val_split(self.full,
                                                                    bs=self.batch_size,
                                                                    augment_train=self.augment_train,
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
            for index in dataset.indices:
                if counter == 0:
                    break
                else:
                    if dataset.dataset[index][1] == sp:
                        new_indices.append(index)
                        counter -= 1
        return new_indices

    def augment_dataset(self, n_samples, specific_class=None):
        generated_patterns = VariableLengthFlashPatterns(data_root=self.data_dir,
                                                         class_limit=self.class_limit,
                                                         augmentations=self.augmentations,
                                                         num_samples=n_samples,
                                                         specific_class=specific_class)
        return generated_patterns

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=True, shuffle=False)

    def train_test_val_split(self, dataset, bs, augment_train, downsample):
        val_split = self.val_split
        dataset_size = len(dataset)

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

        train_dataset, valid_dataset, test_dataset = random_split(
            dataset=dataset, lengths=[n_train, n_val, n_test], generator=torch.Generator().manual_seed(self.gen_seed)
        )
        assert len(valid_dataset) >= bs, \
            'Validation dataset has length {} while batch_size is {}. Please increase your sample size'.format(
                len(valid_dataset), bs)

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

        if augment_train != 0:
            print("Augmentation selected")
            len_train_data = len(train_dataset.dataset._data)
            all_counts = dataset[train_dataset.indices][1].value_counts()
            max_balance = max(all_counts)
            specific_classes = self.find_shortest(dataset, train_dataset, max_balance)
            if len(specific_classes) != 0:
                print("Class imbalance detected, augmenting...")
            else:
                print("Augmentation not needed, skipping...")
                return train_dataset, valid_dataset, test_dataset

            for sp in specific_classes:
                n_samples = max_balance - all_counts[sp]
                generated_augmentation = self.augment_dataset(n_samples, specific_class=[sp])
                train_dataset.indices.extend(
                    list(range(len_train_data, len_train_data + n_samples))
                )
                train_dataset.dataset._data = np.vstack(
                    (train_dataset.dataset._data, np.array(generated_augmentation._data)))
                train_dataset.dataset._meta_data = train_dataset.dataset._meta_data.append(
                    generated_augmentation._meta_data, ignore_index=True)
                len_train_data = len(train_dataset.dataset._data)
                print('Done augmenting train dataset with {} samples for sp. {}, generated from distributions'.format(
                    n_samples, sp)
                )
        return train_dataset, valid_dataset, test_dataset


class RealFireflyData(FireflyDataModule):
    def __init__(self, data_dir, augmentations, class_limit, batch_size, num_samples,
                 augment_train, val_split, gen_seed, downsample):
        super().__init__(data_dir, augmentations, class_limit, batch_size, num_samples,
                         augment_train, val_split, gen_seed, downsample)

    def train_test_val_split(self, dataset, bs, augment_train, downsample):
        val_split = self.val_split
        dataset_size = len(dataset)

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

        train_dataset, valid_dataset, test_dataset = random_split(
            dataset=dataset, lengths=[n_train, n_val, n_test], generator=torch.Generator().manual_seed(self.gen_seed)
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
        assert len(valid_dataset) >= bs, \
            'Validation dataset has length {} while batch_size is {}. Please increase your sample size'.format(
                len(valid_dataset), bs)

        print('done splitting data into {} train/{} validation/{} test sequences...'.format(len(train_dataset),
                                                                                            len(valid_dataset),
                                                                                            len(test_dataset)))
        return train_dataset, valid_dataset, test_dataset
