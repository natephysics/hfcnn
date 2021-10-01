from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hfcnn.utils import get_logger
from hfcnn import dataset, config
from typing import Optional
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

# import the default path_options
path_options = config.construct_options_dict()

class HeatLoadDataModule(LightningDataModule):

    def __init__(
        self, 
        train_transforms=None, 
        val_transforms=None, 
        test_transforms=None, 
        dims=None,
        batch_size: Optional[int] = 32,
        pin_memory: Optional[bool] = False,
        num_workers: Optional[int] = 1,
        shuffle: Optional[bool] = True,

        ):
        super().__init__(
            train_transforms=train_transforms, 
            val_transforms=val_transforms, 
            test_transforms=test_transforms, 
            dims=dims
            )
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: Optional[str] = None) -> None:
        
        """
        Method to import the datasets
        
        """
        if stage == "fit" or stage is None:
            self.train_data = dataset.HeatLoadDataset(path_options["train_df_path"])
            self.val_data = dataset.HeatLoadDataset(path_options["val_df_path"])

        if stage == "test" or stage is None:
            self.test_data = dataset.HeatLoadDataset(path_options["test_df_path"])


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
        self.train_data, 
        batch_size=self.batch_size, 
        shuffle=self.shuffle, 
        pin_memory=self.pin_memory,
        num_workers=self.num_workers,
        persistent_workers=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
        self.val_data, 
        batch_size=self.batch_size, 
        shuffle=self.shuffle, 
        pin_memory=self.pin_memory,
        num_workers=self.num_workers,
        persistent_workers=True
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
        self.test_data, 
        batch_size=self.batch_size, 
        shuffle=self.shuffle, 
        pin_memory=self.pin_memory,
        num_workers=self.num_workers,
        persistent_workers=True
        )