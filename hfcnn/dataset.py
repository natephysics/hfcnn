from __future__ import annotations
import os
import pandas as pd  # needed for the df format
from numpy import integer, issubdtype
from typing import Callable, Tuple, Union, List
from torch.utils.data import Dataset, DataLoader
from torch import sqrt, Tensor
from torchvision import transforms
from hfcnn import files, filters, utils
from mlxtend.preprocessing import standardize
from tqdm import tqdm



def check_defaults(source1, source2):
    """
    Checks the correct default settings. 
    """
    default_settings = {
        'img_dir': './data/raw/',
        'label_list': ['PC1'],
        'transforms': None,
        'drop_neg_values': True,
        'norm_param': {}
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
    def __init__(
        self,
        df: str or pd.DataFrame,
        **kwargs
    ):
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
            if df.endswith('.csv'):
                data = pd.read_csv(df, converters={
                    'pressure': lambda x: utils.str_to_array(x),
                     'iota': lambda x: utils.str_to_array(x)
                     })
            elif df.endswith('.pkl'):
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


    def __len__(self) -> int:
        """Returns the number of data points in the set

        Returns:
            length: number of data points in the set
        """
        return self.img_labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
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
        img_path = files.generate_file_path(timestamp, port, self.settings['img_dir'])
        image = files.import_file_from_local_cache(img_path)
        image = Tensor(image)
        
        # set any pixels below zero to zero
        if self.settings['drop_neg_values']:
            image = image.clip(min=0)

                # add channel for tensor 
        if image.ndim < 3:
            image = image[None, :, :]

        # apply any provided transformations of the data
        if not (self.settings['transforms'] == None):
            image = self.settings['transforms'](image)
        
        # standardize the data
        if 'image_labels' in self.settings['norm_param']:
            img_mean = self.settings['norm_param']['image_labels'][0]
            img_std = self.settings['norm_param']['image_labels'][1]
            image = transforms.Normalize(mean=(img_mean), std=(img_std))(image)

        # generate the labels
        label = row[self.settings['label_list']].copy()
        for label_name in self.settings['label_list']:
            if label_name in self.settings['norm_param'].keys():
                rawvalue = label[label_name].values
                label[label_name] = (rawvalue - self.settings['norm_param'][label_name][0]) /\
                    self.settings['norm_param'][label_name][1]

        label = label.to_numpy()
        if len(label) == 1:
            label = label.flatten()
        
        label = Tensor(label)


        # return sample
        sample = {'image': image, 'label': label}
        return sample

    def apply(self, filter_fn: Union[Callable, list]) -> HeatLoadDataset:
        """Applies a list of filter to the dataset and removes any elements that
        don't pass the filter criteria. Filters do no change the content of the
        images. Returns a HeatLoadDataset object with the filtered dataset.

        Args:
            filter_fn (Callable or list): A Callable (or list of Callables) designed to take in a row from
            the self.img_labels dataframe (pd.Series) and return a Boolean.
        """
        if isinstance(filter_fn, Callable):
            filter_fn = [filter_fn]
        num_of_filters = len(filter_fn)

        tqdm.pandas()
        temp_df = self.img_labels.copy()
        for i, filter in enumerate(filter_fn):  
            print(f'Applying filter {i+1}/{num_of_filters}:')

            # Pass the img path to the filter
            filter.set_img_path(self.settings['img_dir'])

            # apply the filter to the dataframe 
            filter_for_df = temp_df.progress_apply(filter.row_filter(), axis=1)

            # save the file
            temp_df = temp_df[filter_for_df]
        return HeatLoadDataset(temp_df, **self.settings)

    def program_nums(self):
        """Returns the unique program numbers from the data set
        """
        return self.img_labels["program_num"].unique()

    def to_file(self, path_to_file: str) -> None:
        """Exports the data set as a Pandas dataframe and a dict to hkl.

        Args:
            path_to_file ([type]): path to file (should end in .hkl)
        """
        # construct the dict
        export_data = self.settings
        export_data['img_labels'] = self.img_labels

        files.export_data_to_local_cache(
            export_data,
            path_to_file,
        )
        print("Export Complete")

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
        return self.split_by_program_num(program_num_split[1]), self.split_by_program_num(program_num_split[0])

    def normalize_data(self) -> Tuple[float, float]:
        """Calculates the normalization parameters of the dataset across all
        images. If drop_neg_values, will set all values below zero to zero
        before normalizing the data. The normalized version data can be saved if
        new_img_dir is provided.

        Returns:
            float: mean and std
        """
        # get the total number of pixels in the entire data set
        _, x, y = self.__getitem__(1)["image"].size()
        num_of_pixels = self.__len__() * x * y

        # set the batch size
        bs = min(50, self.__len__())

        # set up dataloader
        temploader = DataLoader(self, batch_size=bs)

        # solve for mean
        total_sum = 0
        print('Calculating mean of dataset')
        for batch in tqdm(temploader):
            total_sum += batch["image"].sum()
        mean = total_sum / num_of_pixels
        mean = mean.item()

        # solve for std
        print('Calculating std. of dataset')
        sum_of_squared_error = 0
        for batch in tqdm(temploader):
            sum_of_squared_error += ((batch["image"] - mean).pow(2)).sum()
        std = sqrt(sum_of_squared_error / num_of_pixels)
        std = std.item()

        self.settings['norm_param']['image_labels'] = (mean, std)

        return mean, std

    def normalize_labels(self, labels: list[str]) -> None:
        """Calculates the standardization parameters of the dataset across all
        labels.

        Args:
            labels (list[str]): List of strings that reprepset the labels of the df.
        """
        norm_data = standardize(self.img_labels[labels], columns=labels, return_params=True)

        for label in labels:
            self.settings['norm_param'][label] = (norm_data[1]['avgs'][label], norm_data[1]['stds'][label])

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
        return self.__getitem__(0)['label'].size(dim=0)

    def get_input_dim(self) -> int:
        """Returns the input dimension of the dataset.

        Returns:
            int: input dimension
        """
        return list(self.__getitem__(0)['image'].size())