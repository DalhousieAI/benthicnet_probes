import os
import tarfile

import torch

from utils.benthicnet.io import row2basename
from utils.utils import get_df


def extract_filepaths(tar_dir):
    filepaths = set()

    # Iterate over all the tar files in the directory
    for filename in os.listdir(tar_dir):
        filepath = os.path.join(tar_dir, filename)

        # Check if the item in the directory is a tar file
        if os.path.isfile(filepath) and filepath.endswith(".tar"):
            # Open the tar file
            with tarfile.open(filepath, "r") as tar:
                # Iterate over each member (file or folder) in the tar file
                for member in tar.getmembers():
                    # Extract the file path
                    filepaths.add(member.name)

    return filepaths


class BenthicNetDatasetSkeleton(torch.utils.data.Dataset):
    """BenthicNet dataset."""

    def __init__(
        self,
        tar_dir,
        annotations=None,
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
        self.dataframe = annotations
        self.valid_filepaths = extract_filepaths(tar_dir)

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

        # Need to load the file from the tarball over the network
        if path in self.valid_filepaths:
            return path, True
        return path, False


def save_list_to_txt(list_data, file_path):
    with open(file_path, "w") as file:
        for item in list_data:
            file.write(str(item) + "\n")


def main():
    problem_img_paths = []
    tar_dir = "/lustre06/project/6012565/become/benthicnet-compiled/compiled_unlabelled_512px/tar"
    csv_path = "/lustre06/project/6012565/isaacxu/benthicnet_probes/data_csv/benthicnet_unlabelled_sub_eval.csv"

    df = get_df(csv_path)
    print("Loaded:", csv_path)

    dataset = BenthicNetDatasetSkeleton(tar_dir, df)

    dataset_len = len(dataset)
    batch_size = 8192

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
    )

    del dataset

    for batch_idx, (batch_paths, batch_found_flags) in enumerate(dataloader):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, dataset_len)

        completion_progress = end_idx / dataset_len * 100
        print(f"Processing images: {completion_progress:.2f}%", end="\r")

        problem_img_paths.extend(
            [path for path, found in zip(batch_paths, batch_found_flags) if not found]
        )

    print("\nTotal number of encountered problem images:", len(problem_img_paths))
    save_list_to_txt(
        problem_img_paths,
        "/lustre06/project/def-ttt/isaacxu/benthicnet_probes/data_csv/unlabelled_problem_imgs.txt",
    )


if __name__ == "__main__":
    main()
