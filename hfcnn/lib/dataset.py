from six import string_types
from torch.utils.data import Dataset
import pandas as pd # needed for the df format
from hfcnn.lib import files
import copy

class HeatLoadDataset(Dataset):
    def __init__(self, df: str or pd.DataFrame, img_dir: str, transform=None, target_transform=None):
        """HeatLoadDataset expects string with a path to a dataframe stored in 
        a hickle format. For more details see the readme."""
        if type(df) == str:
            self.img_labels = files.import_file_from_local_cache(df)
        elif type(df) == pd.DataFrame:
            self.img_labels = df.copy()
        else:
            raise TypeError('Input must be a str or df')
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_labels.shape[0]

    def __getitem__(self, time: int, label_names: list=False):
        """Returns a single sample with labels based on the list label_names"""
        if label_names == False:
            label_names = ['PC1']
        row = self.img_labels[self.img_labels['times'] == time]
        timestamp, port = row['times'].values[0], row['port'].values[0]
        img_path = files.generate_file_path(timestamp, port, self.img_dir)
        image = files.import_file_from_local_cache(img_path)
        label = row[label_names].values[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample