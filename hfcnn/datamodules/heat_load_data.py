import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hfcnn import dataset, config
from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

default_paths = config.build_default_paths()

class HeatLoadDataModule(LightningDataModule):
    def __init__(
        self, 
        # Data root should be the path to the data folder, which contains raw and processed.
        data_root, 
        params_file_path=None,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        batch_size: Optional[int] = 32,
        pin_memory: Optional[bool] = True,
        num_workers: Optional[int] = 1,
        shuffle: Optional[bool] = True,
        ):
        super().__init__(
            train_transforms=train_transforms, 
            val_transforms=val_transforms, 
            test_transforms=test_transforms, 
            )
        self.data_root = data_root
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.params_file_path = params_file_path
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: Optional[str] = None) -> None:
        """ Method to import the datasets.
        """
        if stage == "fit" or stage is None:
            self.train_data = dataset.HeatLoadDataset(
                # Path to processed dataframe
                os.path.join(self.data_root, default_paths['train']),
                # Path to the raw image files
                img_dir = os.path.join(self.data_root, default_paths['raw_folder'])
                )
            self.val_data = dataset.HeatLoadDataset(
                os.path.join(self.data_root, default_paths['validation']),
                img_dir = os.path.join(self.data_root, default_paths['raw_folder'])
                )

        if stage == "test" or stage is None:
            self.test_data = dataset.HeatLoadDataset(
                os.path.join(self.data_root, default_paths['test']),
                img_dir = os.path.join(self.data_root, default_paths['raw_folder'])
                )


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
        self.train_data, 
        batch_size=self.batch_size, 
        shuffle=self.shuffle, 
        pin_memory=self.pin_memory,
        num_workers=self.num_workers
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
        self.val_data, 
        batch_size=self.batch_size, 
        shuffle=False, 
        pin_memory=self.pin_memory,
        num_workers=self.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
        self.test_data, 
        batch_size=self.batch_size, 
        shuffle=False, 
        pin_memory=self.pin_memory,
        num_workers=self.num_workers
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(root={self.data_root}, "
            + f"batch_size={self.batch_size}, "
            + f"params_file_path={self.params_file_path}, "
            + f"pin_memory={self.pin_memory}, "
            + f"num_workers={self.num_workers}, shuffle={self.shuffle})"
        )

    def save_data(self, directory: str) -> None:
        """Saves a copy of the data sets to the path. 
        """
        self.train_data.to_file(os.path.join(directory, default_paths['train']))
        self.val_data.to_file(os.path.join(directory, default_paths['validation']))

        if os.path.isfile(os.path.join(directory, default_paths['test'])):       
            self.test_data.to_file(os.path.join(directory, default_paths['test']))
