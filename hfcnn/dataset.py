from __future__ import annotations
import functools
import pandas as pd  # needed for the df format
import numpy as np
from methodtools import lru_cache
from collections import OrderedDict
from numpy import integer, issubdtype
from typing import Callable, Tuple, Union, List
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, float16, from_numpy
from torchvision import transforms
from hfcnn import files, filters, utils
from mlxtend.preprocessing import standardize
from tqdm import tqdm
from hfcnn.utils import get_logger


def check_defaults(source1, source2):
    """
    Checks the correct default settings.
    """
    default_settings = {
        "img_dir": "./data/raw/",
        "label_list": [],
        "transforms": None,
        "drop_neg_values": True,
        "norm_param": {},
    }
    for setting, value in default_settings.items():
        # check to see if setting is in source1
        if setting in source1:
            default_settings[setting] = source1[setting]
        # otherwise check to see if the setting is in source2
        else:
            if setting in source2:
                default_settings[setting] = source2[setting]

    return default_settings


class HeatLoadDataset(Dataset):
    def __init__(self, df: str or pd.DataFrame, **kwargs):
        """Creates at HeatloadDatatset object from a dataframe or link to a dataframe

        Args:
            df (str or pd.DataFrame): a dataframe object cotaining possible labels
            or a directory for the dataframe object and dict stored in .hkl format.


        kwargs: (kwargs will supersede any stored settings)
            img_dir (str): link to the data directory
            Default: ./data/raw/

            transforms: any transformations of the data (applied on get item)
                Transformations are applied prior to standardization.
            Default: None

            norm_param: the standardization parameters for any data or labels.
            Default: {}

            drop_neg_values: sets any pixels below zero to zero
            Default: True

            label_list: list of df column labels to use
            Default: [PC1]

        Raises:
            TypeError: [description]
        """
        # if a string is passed check to see if df is .csv or .pkl
        if isinstance(df, str):
            if df.endswith(".csv"):
                data = pd.read_csv(
                    df,
                    converters={
                        "pressure": lambda x: utils.str_to_array(x),
                        "iota": lambda x: utils.str_to_array(x),
                        "vol": lambda x: utils.str_to_array(x),
                        "phi_edge": lambda x: utils.str_to_array(x),
                    },
                )
            elif df.endswith(".pkl"):
                data = files.import_file_from_local_cache(df)
            else:
                raise ValueError("File must be .csv or .pkl")
            # check to see if the data is a dataframe or dict
            if isinstance(data, dict):
                self.img_labels = data.pop("img_labels")
                self.img_labels = self.img_labels.reset_index(drop=True)
                # default value check
                self.settings = check_defaults(kwargs, data)
            elif isinstance(data, pd.DataFrame):
                self.img_labels = data
                self.img_labels = self.img_labels.reset_index(drop=True)
                # default value check
                self.settings = check_defaults(kwargs, {})
        elif isinstance(df, pd.DataFrame):
            self.img_labels = df.copy()
            self.img_labels = self.img_labels.reset_index(drop=True)
            # default value check
            self.settings = check_defaults(kwargs, {})
        else:
            raise TypeError("Input must be a str (path) or df")

        if "cache" in kwargs:
            self.cache = kwargs["cache"]
        else:
            self.cache = False

        # To enable cache without worrying about garbage collection
        # see: https://rednafi.github.io/reflections/dont-wrap-instance-methods-with-functoolslru_cache-decorator-in-python.html
        # self._get_cached_item__ = functools.cache(self._getitem)

    def __getitem__(self, idx: int) -> dict[Tensor]:
        """Returns a data point from the dataset

        Args:
            idx (int): index of the data point

        Returns:
            Tuple[Tensor, Tensor]: data point and label
        """
        if self.cache:
            return self._getcacheditem(idx)
        else:
            return self._getitem(idx)

    @lru_cache(maxsize=None)
    def _getcacheditem(self, idx: int) -> dict[Tensor]:
        """Returns a data point from the dataset

        Args:
            idx (int): index of the data point

        Returns:
            Tuple[Tensor, Tensor]: data point and label
        """
        return self._getitem(idx)

    def _getitem(self, idx: int) -> dict[Tensor]:
        """Returns a data point with "image" and "label" where the labels are
        pulled from the label_list, a list of column names from the data frame.

        Args:
            idx (int): the timestamp of the heatload image data OR an index.

        Returns:
            Sample (dic): Returns a sample with "image" and "label"
        """
        # Assuming there is no more than max_number_of_samples worth of samples.
        # This is to help differentiate between timestamps and index values.
        max_number_of_samples = 9999999999

        # if a list is passed then assume it is a list of idx values
        if issubdtype(type(idx), list):
            return [self.__getitem__(i) for i in idx]

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
        img_path = files.generate_file_path(timestamp, port, self.settings["img_dir"])
        image = files.import_file_from_local_cache(img_path)
        image = image.clip(min=0)
        image = image.astype(np.float32)
        image = from_numpy(image)

        # add channel for tensor
        if image.ndim < 3:
            image = image[None, :, :]

        # apply any provided transformations of the data
        if not (self.settings["transforms"] == None):
            image = self.settings["transforms"](image)

        # standardize the data
        if "image_labels" in self.settings["norm_param"]:
            img_mean = self.settings["norm_param"]["image_labels"][0]
            img_std = self.settings["norm_param"]["image_labels"][1]
            image = transforms.Normalize(mean=(img_mean), std=(img_std))(image)

        # get and standardize the labels
        label = OrderedDict()

        for label_function in self.settings["label_list"]:
            rawvalue = self.__getattribute__(label_function)(row["times"].item())

            # if the normalization parameters are provided then standardize
            if label_function in self.settings["norm_param"]:
                label[label_function] = (
                    rawvalue - self.settings["norm_param"][label_function][0]
                ) / self.settings["norm_param"][label_function][1]
            else:
                label[label_function] = rawvalue

        label = [value for key, value in label.items()]

        label = Tensor(label)

        # return sample
        sample = {"image": image, "label": label}
        return sample

    def __len__(self) -> int:
        """Returns the number of data points in the set

        Returns:
            length: number of data points in the set
        """
        return self.img_labels.shape[0]

    # TODO: use @classmethod
    def apply(self, filter_fn: Union[Callable, list]) -> HeatLoadDataset:
        """Applies a list of filter to the dataset and removes any elements that
        don't pass the filter criteria. Filters do no change the content of the
        images. Returns a HeatLoadDataset object with the filtered dataset.

        Args:
            filter_fn (Callable or list): A Callable (or list of Callables) designed to take in a row from
            the self.img_labels dataframe (pd.Series) and return a Boolean.
        """

        log = get_logger(__name__)

        if isinstance(filter_fn, Callable):
            filter_fn = [filter_fn]
        num_of_filters = len(filter_fn)

        tqdm.pandas()
        temp_df = self.img_labels.copy()
        for i, filter in enumerate(filter_fn):

            log.info(f"Applying filter {i+1}/{num_of_filters}:")

            # Pass the img path to the filter
            filter.set_img_path(self.settings["img_dir"])

            # apply the filter to the dataframe
            filter_for_df = temp_df.progress_apply(filter.row_filter(), axis=1)

            # save the file
            temp_df = temp_df[filter_for_df]
        return HeatLoadDataset(temp_df, **self.settings)

    def program_nums(self):
        """Returns the unique program numbers from the data set"""
        return self.img_labels["program_num"].unique()

    def to_file(self, path_to_file: str) -> None:
        """Exports the data set as a Pandas dataframe and a dict to hkl.

        Args:
            path_to_file ([type]): path to file (should end in .hkl)
        """
        log = get_logger(__name__)

        # construct the dict
        export_data = self.settings
        export_data["img_labels"] = self.img_labels

        files.export_data_to_local_cache(
            export_data,
            path_to_file,
        )
        log.info("Export Complete")

    def split_by_program_num(self, prog_num_list: List[int]) -> HeatLoadDataset:
        """Generates a new copy of the data set with the subset of data that
        whose program_nums match the ones in the prog_num_list.

        Args:
            prog_num_list (list): [description]

        Returns:
            [type]: [description]
        """
        filter_for_df = self.img_labels.program_num.isin(prog_num_list)

        return HeatLoadDataset(self.img_labels[filter_for_df], **self.settings)

    def validation_split(self, ratio: float) -> Tuple[HeatLoadDataset, HeatLoadDataset]:
        """Takes in a training ratio for train/validation(or test) and returns
        two HeatLoadDatasets with the corresponding ratios of programs.

        Args:
            ratio (float): Ratio of the validation(train). (0-1)

        Returns:
            Training_data (HeatLoadDataset), Validation_data (HeatLoadDataset)
        """
        # genarate the two sets of program numbers to use
        program_num_split = filters.split(self.program_nums(), ratio)

        # generate two datasets from the program numbers.
        return self.split_by_program_num(
            program_num_split[1]
        ), self.split_by_program_num(program_num_split[0])

    def normalize_data(self) -> Tuple[float, float]:
        """Calculates the normalization parameters of the dataset across all
        images. If drop_neg_values, will set all values below zero to zero
        before normalizing the data. The normalized version data can be saved if
        new_img_dir is provided.

        Returns:
            float: mean and std
        """
        log = get_logger(__name__)

        # solve for mean
        tqdm.pandas()
        log.info("Calculating mean of the images")

        sums = self.img_labels.progress_apply(lambda x: self.image_mean(x), axis=1)
        sums = sums / self.__len__()
        mean = sums.sum().item()

        # solve for std
        log.info("Calculating std. of the images")
        sum_of_squared_error = self.img_labels.progress_apply(
            lambda x: self.image_sum_squared_error(x, mean), axis=1
        )
        sum_of_squared_error = sum_of_squared_error.sum()
        std = np.sqrt(sum_of_squared_error / self.__len__())
        std = std.item()

        self.settings["norm_param"]["image_labels"] = (mean, std)

        return mean, std

    def normalize_labels(self, labels: list[str]) -> None:
        """Calculates the standardization parameters of the dataset across all
        labels.

        Args:
            labels (list[str]): List of strings of labels.
        """
        log = get_logger(__name__)
        tqdm.pandas()

        for label in labels:
            log.info(f"Calculating mean and std. of feature {label}")
            norm_params = standardize(
                self.img_labels.progress_apply(
                    lambda x: self.__getattribute__(label)(x["times"]), axis=1
                ),
                return_params=True,
            )
            self.settings["norm_param"][label] = (
                norm_params[1]["avgs"].item(),
                norm_params[1]["stds"].item(),
            )

    def import_settings(self, settings: dict) -> None:
        """Imports the settings.

        Args:
            settings (dict): self.settings from a Heatloaddataset object.
        """
        self.settings = settings

    def get_output_dim(self) -> int:
        """Returns the output dimension of the dataset.

        Returns:
            int: output dimension
        """
        return self.__getitem__(0)["label"].size(dim=0)

    def get_input_dim(self) -> List[int]:
        """Returns the input dimension of the dataset.

        Returns:
            int: input dimension
        """
        return list(self.__getitem__(0)["image"].size())

    def image_mean(self, row: pd.Series):
        """Sum over the pixels in the image."""
        # load the image file from the pandas row
        data = self.__getitem__(row["times"])
        image = data["image"]

        # remove values below zero convert to int
        current_image = image.numpy().astype(np.int64).clip(min=0)

        # sum the pixels
        image_mean = np.sum(current_image) / current_image.size

        return image_mean

    def image_sum_squared_error(self, row: pd.Series, mean: float):
        """Sum of squared error of the pixels of the image."""
        # load the image file from the pandas row
        data = self.__getitem__(row["times"])
        image = data["image"]

        # remove values below zero convert to int
        current_image = image.numpy().astype(np.int64).clip(min=0)

        # sum of squared error
        sum_of_squared_error = np.square(current_image - mean).sum()
        sum_of_squared_error = sum_of_squared_error / current_image.size

        return sum_of_squared_error

    ##################
    #### features ####
    ##################

    def IA(self, timestamp: str = None) -> float:
        """Returns iota at the edge of the plasma."""

        # find the row that matches the timestamp and return IA
        return self.img_labels.loc[self.img_labels["times"] == timestamp, "IA"].item()

    def pressure(self, timestamp: str = None) -> List[float]:
        """Returns list of pressures across the plasma from r=0 to r=1."""

        # find the row that matches the timestamp and returns list of pressures across the plasma
        return self.img_labels.loc[
            self.img_labels["times"] == timestamp, "pressure"
        ].item()

    def pressure_at_edge(self, timestamp: str = None) -> float:
        """Returns pressure at the edge of the plasma."""

        # load the image file from the pandas row
        pressure_list = self.pressure(timestamp)

        # find the row that matches the timestamp and return pressure
        return pressure_list[-1]

    def W_dia(self, timestamp: str = None) -> float:
        """Returns the diameter of the plasma."""

        # find the row that matches the timestamp and return W_dia
        return self.img_labels.loc[
            self.img_labels["times"] == timestamp, "W_dia"
        ].item()

    def iota_edge(self, timestamp: str = None) -> float:
        """Returns the diameter of the plasma."""

        # find the row that matches the timestamp and return W_dia
        return self.img_labels.loc[
            self.img_labels["times"] == timestamp, "iota_edge"
        ].item()
