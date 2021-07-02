from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, sqrt
import pandas as pd  # needed for the df format
from hfcnn.lib import files
from numpy import integer, issubdtype
import os


def make_dict(df: pd.DataFrame, drop_neg_values: bool, mean: float, std: float):
    """Makes a dict from the following parameters.

    Args:
        df (str or pd.DataFrame): a dataframe object cotaining possible labels.

        img_dir (str): link to the data directory

        drop_neg_values (bool): Drops negative pixel values

        mean (float): Standardization parameter for data.
        std (float): Standardization parameter for data.

    Returns:
        dict: dict with the above parameters
    """
    my_dict = {
        "img_labels": df,
        "drop_neg_values": drop_neg_values,
        "mean": mean,
        "std": std,
    }
    return my_dict


class HeatLoadDataset(Dataset):
    def __init__(
        self,
        df: str or pd.DataFrame,
        img_dir: str,
        mean: float = None,
        std: float = None,
        drop_neg_values: bool = True,
    ):
        """Creates at HeatloadDatatset object from a dataframe or link to a dataframe

        Args:
            df (str or pd.DataFrame): a dataframe object cotaining possible labels
            or a directory for the dataframe object and dict stored in .hkl format.

            img_dir (str): link to the data directory

        Raises:
            TypeError: [description]
        """
        if type(df) == str:
            data = files.import_file_from_local_cache(df)
            if isinstance(data, dict):
                self.img_labels = data.pop("img_labels")
                self.img_labels = self.img_labels.reset_index(drop=True)
                self.mean = data.pop("mean")
                self.std = data.pop("std")
                self.drop_neg_values = data.pop("drop_neg_values")
            elif isinstance(data, pd.DataFrame):
                self.img_labels = data
                self.img_labels = self.img_labels.reset_index(drop=True)
                self.mean = mean
                self.std = std
                self.drop_neg_values = drop_neg_values
        elif type(df) == pd.DataFrame:
            self.img_labels = df.copy()
            self.img_labels = self.img_labels.reset_index(drop=True)
            self.mean = mean
            self.std = std
            self.drop_neg_values = drop_neg_values
        else:
            raise TypeError("Input must be a str or df")

        if not os.path.isdir(img_dir):
            raise ValueError("Expected path to directory from img_dir")

        self.img_dir = img_dir

    def __len__(self):
        """Returns the number of data points in the set

        Returns:
            length: number of data points in the set
        """
        return self.img_labels.shape[0]

    def __getitem__(self, idx: int, label_names: list = False):
        """Returns a data point with "image" and "label" where the labels are
        pulled from the label_names, a list of column names from the data frame.

        Args:
            idx (int): the timestamp of the heatload image data OR an index.
            label_names (list, optional): A list containing data from the
            corresponding column names of the dataframe to return as the "label"
            in the sample.
            Defaults to ['PC1'].

        Returns:
            Sample (dic): Returns a sample with "image" and "label"
        """
        # Assuming there is no more than max_number_of_samples worth of samples.
        # This is to help differentiate between timestamps and index values.
        max_number_of_samples = 9999999999

        # Sets the default option if left blank (needed because list is permutable)
        if not label_names:
            label_names = ["PC1"]

        # If idx is too small to be a timestamp, find the timestamp that
        # corresponds to the index.
        if (idx <= max_number_of_samples) and issubdtype(type(idx), integer):
            idx = self.img_labels.iloc[idx]["times"]
        elif not issubdtype(type(idx), integer):
            raise TypeError("idx needs to be an int")

        # find the row that matches the timestamp
        row = self.img_labels[self.img_labels["times"] == idx]

        # load the image from the local cache
        timestamp, port = row["times"].values[0], row["port"].values[0]
        img_path = files.generate_file_path(timestamp, port, self.img_dir)
        image = files.import_file_from_local_cache(img_path)
        image = from_numpy(image)
        if self.drop_neg_values:
            image = image.clip(min=0)

        # generate the labels
        label = row[label_names].values[0]
        label = from_numpy(label)

        # return sample
        sample = {"image": image, "label": label}
        return sample

    def apply(self, filter_fn):
        """Applies a filter to the dataset and removes any elements that
        don't pass the filter criteria. Filters do no change the content of the
        images. Returns a HeatLoadDataset object with the filtered dataset.

        Args:
            filter_fn (function): A function designed to take in a row from
            the self.img_labels dataframe (pd.Series) and return a Boolean.
        """
        filter_for_df = self.img_labels.apply(filter_fn, axis=1)
        return HeatLoadDataset(self.img_labels[filter_for_df], self.img_dir)

    def program_nums(self):
        """Returns the unique program numbers from the data set
        """
        return self.img_labels["program_num"].unique()

    def to_file(self, path_to_file):
        """Exports the data set as a Pandas dataframe and a dict to hkl.

        Args:
            path_to_file ([type]): path to file (should end in .hkl)
        """
        # construct the dict

        files.export_data_to_local_cache(
            make_dict(self.img_labels, self.drop_neg_values, self.mean, self.std),
            path_to_file,
        )
        print("Export Complete")

    def split_by_program_num(self, prog_num_list: list):
        """Generates a new copy of the data set with the subset of data that
        whose program_nums match the ones in the prog_num_list.

        Args:
            prog_num_list (list): [description]

        Returns:
            [type]: [description]
        """
        filter_for_df = self.img_labels.program_num.isin(prog_num_list)

        return HeatLoadDataset(self.img_labels[filter_for_df], self.img_dir)

    def normalize(self, new_img_dir: str = None):
        """Calculates the normalization parameters of the dataset across all
        images. If drop_neg_values, will set all values below zero to zero
        before normalizing the data. The normalized version data can be saved if
        new_img_dir is provided.

        Args:
            new_img_dir (str, optional): path to saved normalized (or zeroed)
            data. Defaults to None.

        Returns:
            float: mean and std
        """
        # get the total number of pixels in the entire data set
        x, y = self.__getitem__(1)["image"].size()
        num_of_pixels = self.__len__() * x * y

        bs = min(50, self.__len__())

        # set up dataloader
        temploader = DataLoader(self, batch_size=bs, num_workers=12)

        # solve for mean
        total_sum = 0
        for batch in temploader:
            total_sum += batch["image"].sum()
        self.mean = total_sum / num_of_pixels

        # solve for std
        sum_of_squared_error = 0
        for batch in temploader:
            sum_of_squared_error += ((batch["image"] - self.mean).pow(2)).sum()
        self.std = sqrt(sum_of_squared_error / num_of_pixels)

        return self.mean, self.std

def main():
    raw_data = HeatLoadDataset('data/raw/test_df.hkl', 'data/raw/')
    print('huh?')


if __name__ == "__main__":
    main()