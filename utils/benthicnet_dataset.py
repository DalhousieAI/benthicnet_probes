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
        self.dataframe = annotations
        # self.dataframe = self.dataframe.head(64)
        self.dataframe["tarname"] = self.dataframe["dataset"] + ".tar"
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # start_time = time.time()

        split_img_name = row2basename(row, use_url_extension=False).split(".")

        if len(split_img_name) > 1:
            img_name = ".".join(split_img_name[:-1]) + ".jpg"
        else:
            img_name = split_img_name[0] + ".jpg"

        path = row["dataset"] + "/" + row["site"] + "/" + img_name
        node_file_path = os.path.join(os.environ["SLURM_TMPDIR"], path)
        if os.path.isfile(node_file_path):
            sample = PIL.Image.open(node_file_path)
        else:
            # Need to load the file from the tarball over the network
            with tarfile.open(
                os.path.join(self.tar_dir, row["tarname"]), mode="r"
            ) as t:
                # print(row)
                sample = PIL.Image.open(t.extractfile(path))
                # PIL.Image has lazy data loading. But PIL won't be able to access
                # the data from the tarball once we've left this context, so we have
                # to manually trigger the loading of the data now.
                sample.load()
                # print(time.time() - start_time)

            # Other workers might try to access the same image at the same
            # time, creating a race condition. If we've started writing the
            # output, there will be a partially written file which can't be
            # loaded. To avoid another worker trying to read our partially
            # written file, we write the output to a temp file and
            # then move the file to the target location once it is done.
            with tempfile.TemporaryDirectory() as dir_tmp:
                # Write to a temporary file
                node_file_temp_path = os.path.join(dir_tmp, os.path.basename(path))
                sample.save(node_file_temp_path)
                # Move our temporary file to the destination
                os.makedirs(os.path.dirname(node_file_path), exist_ok=True)
                shutil.move(node_file_temp_path, node_file_path)
        # load_time = time.time() - start_time
        # print("load time:", load_time)

        if self.transform:
            sample = self.transform(sample)

        return (
            sample,
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


def gen_datasets(df, tar_dir, transform, val_size=0.25, test_size=0.2, seed=0):
    df_train, df_val, df_test = random_partition_df(
        df, val_size=val_size, test_size=test_size, seed=seed
    )

    train_dataset = BenthicNetDataset(tar_dir, df_train, transform=transform[0])
    val_dataset = BenthicNetDataset(tar_dir, df_val, transform=transform[1])
    test_dataset = BenthicNetDataset(tar_dir, df_test, transform=transform[1])

    return train_dataset, val_dataset, test_dataset


# Deprecated functions

# def gen_random_partition(data, size, replace):
#     secondary_idxs = np.random.choice(np.arange(len(data)), round(len(data)*size), replace=replace)
#     secondary_data = data[data['index'].isin(secondary_idxs)]
#     data = data[~data['index'].isin(secondary_idxs)]

#     return data, secondary_data

# def ds_by_stations_lite(file, validation_size=0.25, test_size=0.2, replace=False):
#     dataset = pd.read_csv(file, low_memory=False)
#     dataset = dataset.drop_duplicates()
#     dataset = dataset[dataset["dst"]!='nrcan']
#     #dataset = dataset[dataset["dst"]!='Chesterfield']
#     #dataset = dataset[dataset["dst"]!='Wager']
#     dataset.dropna(how='all', inplace=True)
#     dataset.dropna(subset=['label'], inplace=True)

#     dataset['index'] = dataset.index
#     test_loc_data = dataset[dataset["partition"].isin(["test"])]
#     training_data = dataset[dataset["partition"].isin(["train"])]

#     # Test subset has to be removed prior to partitioning validation
#     training_data, test_r_data = gen_random_partition(training_data, test_size, replace)
#     training_data, validation_data = gen_random_partition(training_data, validation_size, replace)

#     return training_data, validation_data, test_r_data, test_loc_data

# def ds_by_stations_full(
#     file,
#     class_graph,
#     class_dict,
#     R,
#     validation_size=0.25,
#     test_size=0.2,
#     replace=False,
#     missing_images=[
#         "16_DSCN3879",
#         "16_DSCN3880",
#         "22_DSCN3248",
#         "055_4_IMG0131"
#         ]
#     ):
#     training_data = pd.read_csv(file, low_memory=False)
#     training_data = training_data.drop_duplicates()
#     training_data = training_data[training_data["dataset"]!="pangaea-b"]
#     training_data = training_data[training_data["dataset"]!="pangaea-897047"]
#     training_data = training_data[training_data["image"].str.contains("nrcan-2013", regex=False)==False]
#     training_data = training_data[training_data["image"].str.contains("(1)", regex=False)==False]
#     for img in missing_images:
#         training_data = training_data[training_data["image"].str.contains(img, regex=False)==False]

#     training_data.dropna(how='all', subset=["CATAMI Substrate"], inplace=True)

#     training_data['index'] = training_data.index

#     labels, masks = parse_labels(df=training_data, class_graph=class_graph, class_dict=class_dict, R=R)
#     training_data['label_id'] = labels
#     training_data["mask"] = masks

#     test_loc_data = training_data[training_data["partition"].isin(["test"])]
#     training_data = training_data[training_data["partition"].isin(["train"])]

#     training_data, test_r_data = gen_random_partition(training_data, test_size, replace)
#     training_data, validation_data = gen_random_partition(training_data, validation_size, replace)

#     return training_data, validation_data, test_r_data, test_loc_data
