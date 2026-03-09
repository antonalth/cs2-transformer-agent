import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional

from dataset import DatasetRoot, TrainingSample, GroundTruth, cs2_collate_fn
from config import GlobalConfig

class CS2DataModule(pl.LightningDataModule):
    def __init__(self, global_cfg: GlobalConfig):
        super().__init__()
        self.global_cfg = global_cfg
        self.ds_root: Optional[DatasetRoot] = None
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None):
        # Initialize the dataset root (loads manifest, etc.)
        if self.ds_root is None:
            self.ds_root = DatasetRoot(self.global_cfg.dataset)

        # Build datasets once (deterministic sliding window)
        if self.train_ds is None:
            self.train_ds = self.ds_root.build_dataset("train")
        if self.val_ds is None:
            self.val_ds = self.ds_root.build_dataset("val")

    def train_dataloader(self):
        if self.ds_root is None:
             self.ds_root = DatasetRoot(self.global_cfg.dataset)
        if self.train_ds is None:
            self.train_ds = self.ds_root.build_dataset("train")
        
        return DataLoader(
            self.train_ds,
            batch_size=self.global_cfg.train.batch_size,
            shuffle=True, # Shuffle training data
            num_workers=self.global_cfg.train.num_workers,
            collate_fn=cs2_collate_fn,
            pin_memory=True,
            persistent_workers=False
        )

    def val_dataloader(self):
        if self.ds_root is None:
             self.ds_root = DatasetRoot(self.global_cfg.dataset)
        if self.val_ds is None:
             self.val_ds = self.ds_root.build_dataset("val")
        
        return DataLoader(
            self.val_ds,
            batch_size=self.global_cfg.train.batch_size,
            shuffle=False,
            num_workers=self.global_cfg.train.num_workers,
            collate_fn=cs2_collate_fn,
            pin_memory=True,
            persistent_workers=False
        )
