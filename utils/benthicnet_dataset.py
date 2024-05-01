import os
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd
import PIL.Image
import torch.utils.data
from sklearn.model_selection import train_test_split


class BenthicNetDataset(torch.utils.data.Dataset):
    """BenthicNet dataset."""

    def __init__(
        self,
        annotations=None,
        transform=None,
        local=None,
    ):
        """
        Dataset for BenthicNet data.

        Parameters
        ----------
        tar_dir : str
            Directory with all the images.
        annotations : str
            Dataframe with annotations.
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
        self.dataframe = annotations.copy()
        self.dataframe.loc[:, "tarname"] = self.dataframe.loc[:, "dataset"] + ".tar"
        self.transform = transform
        self.local = local

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img_name = row["image"] + ".jpg"

        path = row["dataset"] + "/" + row["site"] + "/" + img_name

        if self.local:
            node_file_path = self.local + "/" + path
        else:
            node_file_path = os.path.join(os.environ["SLURM_TMPDIR"], path)

        sample = PIL.Image.open(node_file_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, (
            row["catami_biota"],
            row["catami_substrate"],
            row["catami_relief"],
            row["catami_bedforms"],
            row["colour_qualifier"],
            row["biota_mask"],
            row["substrate_mask"],
            row["relief_mask"],
            row["bedforms_mask"],
        )


class OneHotBenthicNetDataset(torch.utils.data.Dataset):
    """BenthicNet dataset."""

    def __init__(
        self, annotations=None, transform=None, lab_col="catami_substrate", local=None
    ):
        """
        Dataset for BenthicNet data.

        Parameters
        ----------
        tar_dir : str
            Directory with all the images.
        annotations : str
            Dataframe with annotations.
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
        self.dataframe = annotations.copy()
        self.dataframe.loc[:, "tarname"] = self.dataframe.loc[:, "dataset"] + ".tar"
        self.transform = transform
        self.lab_col = lab_col
        self.local = local

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img_name = row["image"] + ".jpg"

        path = row["dataset"] + "/" + row["site"] + "/" + img_name

        if self.local:
            node_file_path = self.local + "/" + path
        else:
            node_file_path = os.path.join(os.environ["SLURM_TMPDIR"], path)

        sample = PIL.Image.open(node_file_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, row[self.lab_col]


def random_partition_df(df, val_size, test_size, seed):
    # First, split the data into train and test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    # Then, split the train data into train and validation
    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=seed)

    return df_train, df_val, df_test


def gen_datasets(
    df,
    transform,
    random_partition,
    one_hot=False,
    local=None,
    lab_col="catami_substrate",
    val_size=0.25,
    test_size=0.2,
    seed=0,
):
    if random_partition:
        df_train, df_val, df_test = random_partition_df(
            df, val_size=val_size, test_size=test_size, seed=seed
        )
    else:
        df_train = df[df["partition"] == "train"]
        df_test = df[df["partition"] == "test"]

        # To ensure 60% of data is in training
        # Here, test_size is expected test size,
        # which may have been violated by geo-spatial partitioning
        # Exact
        # val_percent = (0.4 - len(df_test) / len(df)) / (1 - len(df_test) / len(df))
        # Approximate
        val_percent = (0.4 - len(df_test) / len(df)) / (1 - test_size)

        assert val_percent > 0, "Test size is too large for the given dataset."
        df_train, df_val = train_test_split(
            df_train, test_size=val_percent, random_state=seed
        )
    if one_hot:
        train_dataset = OneHotBenthicNetDataset(
            df_train, transform=transform[0], local=local, lab_col=lab_col
        )
        val_dataset = OneHotBenthicNetDataset(
            df_val, transform=transform[1], local=local, lab_col=lab_col
        )
        test_dataset = OneHotBenthicNetDataset(
            df_test, transform=transform[1], local=local, lab_col=lab_col
        )
    else:
        train_dataset = BenthicNetDataset(df_train, transform=transform[0], local=local)
        val_dataset = BenthicNetDataset(df_val, transform=transform[1], local=local)
        test_dataset = BenthicNetDataset(df_test, transform=transform[1], local=local)

    return train_dataset, val_dataset, test_dataset
