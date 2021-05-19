from torch.utils.data import Dataset
import pandas as pd # needed for the df format
from hfcnn.lib import files
from numpy import integer, issubdtype
import os

class HeatLoadDataset(Dataset):
    def __init__(self, df: str or pd.DataFrame, img_dir: str):
        """Creates at HeatloadDatatset object from a dataframe or link to a dataframe

        Args:
            df (str or pd.DataFrame): a dataframe object or a directory for the
            dataframe object stored in .hkl format.

            img_dir (str): link to the data directory

        Raises:
            TypeError: [description]
        """
        if type(df) == str:
            self.img_labels = files.import_file_from_local_cache(df)
            self.img_labels = self.img_labels.reset_index(drop=True)
        elif type(df) == pd.DataFrame:
            self.img_labels = df.copy()
            self.img_labels = self.img_labels.reset_index(drop=True)
        else:
            raise TypeError('Input must be a str or df')
        
        if not os.path.isdir(img_dir):
            raise ValueError('Expected path to directory from img_dir')

        self.img_dir = img_dir

    def __len__(self):
        """Returns the number of data points in the set

        Returns:
            length: number of data points in the set
        """
        return self.img_labels.shape[0]

    def __getitem__(self, idx: int, label_names: list=False):
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
        if label_names == False:
            label_names = ['PC1']

        # If idx is too small to be a timestamp, find the timestamp that
        # corresponds to the index.
        if (idx <= max_number_of_samples) and issubdtype(type(idx), integer):
            idx = self.img_labels.iloc[idx]['times']
        elif not issubdtype(type(idx), integer):
            raise TypeError('idx needs to be an int')

        # find the row that matches the timestamp
        row = self.img_labels[self.img_labels['times'] == idx]
            
        # load the image from the local cache
        timestamp, port = row['times'].values[0], row['port'].values[0]
        img_path = files.generate_file_path(timestamp, port, self.img_dir)
        image = files.import_file_from_local_cache(img_path)
        
        # generate a sample
        label = row[label_names].values[0]
        sample = {"image": image, "label": label}
        return sample

    def apply(self, filter_fn):
        """Applies a filter to the dataset and removes any elements that
        don't pass the filter criteria. Returns a HeatLoadDataset object
        with the filtered dataset. 

        Args:
            filter_fn (function): A function designed to take in a row from
            the self.img_labels dataframe (pd.Series) and return a Boolean.
        """
        filter_for_df = self.img_labels.apply(filter_fn, axis=1)
        return HeatLoadDataset(
            self.img_labels[filter_for_df], 
            self.img_dir
            )
            
    def program_nums(self):
        """Returns the unique program numbers from the data set
        """
        return self.img_labels['program_num'].unique()

    def to_file(self, path_to_file):
        """Exports the data set as a Pandas dataframe to hkl.

        Args:
            path_to_file ([type]): path to file (should end in .hkl)
        """
        files.export_data_to_local_cache(self.img_labels, path_to_file)