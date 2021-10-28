import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hfcnn import dataset
from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

class HeatLoadDataModule(LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str=None,
        data_root=None,
        params_file_path=None,
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        batch_size: Optional[int] = 32,
        pin_memory: Optional[bool] = True,
        num_workers: Optional[int] = 4,
        shuffle: Optional[bool] = True,
        ):
        super().__init__(
            train_transforms=train_transforms, 
            val_transforms=val_transforms, 
            test_transforms=test_transforms, 
            )
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
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
        # TODO: Add options to specify data_root
        if stage == "fit" or stage is None:
            self.train_data = dataset.HeatLoadDataset(
                # Path to processed dataframe
                self.test_data_path,
                # Path to the raw image files
                img_dir = self.data_root
                )
            self.val_data = dataset.HeatLoadDataset(
                self.val_data_path,
                img_dir = self.data_root
                )

        if stage == "test" or stage is None:
            self.test_data = dataset.HeatLoadDataset(
                self.test_data_path,
                img_dir = self.data_root
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
            + f"batch_size={self.batch_size}, "
            + f"params_file_path={self.params_file_path}, "
            + f"pin_memory={self.pin_memory}, "
            + f"num_workers={self.num_workers}, shuffle={self.shuffle})"
        )

    def save_data(self, directory: str) -> None:
        """Saves a copy of the data sets to the path. 
        """
        self.train_data.to_file(os.path.join(directory, 'train.pkl'))
        self.val_data.to_file(os.path.join(directory, 'vaildation.pkl'))

        # if test set exists, copy as well. 
        if self.test_data_path is not None:       
            self.test_data.to_file(os.path.join(directory, 'test.pkl'))
