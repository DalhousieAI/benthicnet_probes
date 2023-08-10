import os
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd
import PIL.Image
import torch.utils.data
from sklearn.model_selection import train_test_split

from utils.benthicnet.io import row2basename


class BenthicNetDataset(torch.utils.data.Dataset):
    """BenthicNet dataset."""

    def __init__(
        self,
        tar_dir,
        annotations=None,
        transform=None,
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
        self.tar_dir = tar_dir
        self.dataframe = annotations.copy()
        self.dataframe.loc[:, "tarname"] = self.dataframe.loc[:, "dataset"] + ".tar"
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        split_img_name = row2basename(row, use_url_extension=True).split(".")

        if len(split_img_name) > 1:
            img_name = ".".join(split_img_name[:-1]) + ".jpg"
        else:
            img_name = split_img_name[0] + ".jpg"

        path = row["dataset"] + "/" + row["site"] + "/" + img_name
        node_file_path = os.path.join(os.environ["SLURM_TMPDIR"], path)
        sample = PIL.Image.open(node_file_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, (
            row["CATAMI Biota"],
            row["CATAMI Substrate"],
            row["CATAMI Relief"],
            row["CATAMI Bedforms"],
            row["Colour-qualifier"],
            row["Biota Mask"],
            row["Substrate Mask"],
            row["Relief Mask"],
            row["Bedforms Mask"],
        )


def random_partition_df(df, val_size, test_size, seed):
    # First, split the data into train and test
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=seed)

    # Then, split the train data into train and validation
    df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=seed)

    return df_train, df_val, df_test


def gen_datasets(
    df, tar_dir, transform, random_partition, val_size=0.25, test_size=0.2, seed=0
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
        val_percent = (0.4 - len(df_test) / len(df)) / (1 - test_size)

        assert val_percent > 0, "Test size is too large for the given dataset."
        df_train, df_val = train_test_split(
            df_train, test_size=val_percent, random_state=seed
        )

    train_dataset = BenthicNetDataset(tar_dir, df_train, transform=transform[0])
    val_dataset = BenthicNetDataset(tar_dir, df_val, transform=transform[1])
    test_dataset = BenthicNetDataset(tar_dir, df_test, transform=transform[1])

    return train_dataset, val_dataset, test_dataset
