from torch.utils.data import Dataset
import pandas as pd # needed for the df format
from hfcnn.lib import files

class HeatLoadDataset(Dataset):
    def __init__(self, df: str, img_dir: str, transform=None, target_transform=None):
        """HeatLoadDataset expects string with a path to a dataframe stored in 
        a hickle format. For more details see the readme."""
        self.img_labels = files.import_file_from_local_cache(df)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.img_labels.size

    def __getitem__(self, time, label_names=['PC1']):
        row = self.img_labels[self.img_labels['times'] == time]
        timestamp, port = row[1]['times'], row[1]['port']
        img_path = files.generate_file_path(timestamp, port, self.img_dir)
        image = files.import_file_from_local_cache(img_path)
        label = self.img_labels[label_names]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample