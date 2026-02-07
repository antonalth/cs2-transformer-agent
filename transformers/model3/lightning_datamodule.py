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

        # We assume the "epoch" concept from the original code is mapped to 
        # PyTorch Lightning's idea of an epoch. 
        # The original code rebuilt the dataset every epoch with a new seed.
        # In PL, setup is called once (usually). 
        # To support dynamic epoch regeneration, we might need to do it in 
        # on_train_epoch_start in the LightningModule or finding a way here.
        # However, for simplicity and standard usage, let's build the initial epoch here.
        # If we need re-sampling every epoch, we can use a custom DataLoader that 
        # re-samples in its __iter__, or use the ReloadDataloaderEveryEpoch flag in Trainer.
        
        # For now, let's build epoch 0. The original code's "Epoch" class implies 
        # it's just a container for samples.
        # To match the original behavior of "random windows per round", we ideally want 
        # this to refresh.
        # We will handle the epoch index in train_dataloader to rebuild if needed, 
        # but standard PL pattern usually keeps the dataset static or uses an IterableDataset.
        
        # Current strategy: Build epoch 0 here. 
        # If the user wants dynamic sampling, we can implement it by checking trainer.current_epoch 
        # inside the dataloader creation if we pass trainer to this module or access it.
        pass

    def train_dataloader(self):
        # We can access the current epoch from the trainer if attached
        epoch_idx = 0
        if self.trainer:
            epoch_idx = self.trainer.current_epoch
        
        # Re-build the dataset for the current epoch to get new random windows
        # as per original implementation logic
        if self.ds_root is None:
             self.ds_root = DatasetRoot(self.global_cfg.dataset)
             
        self.train_ds = self.ds_root.build_epoch("train", epoch_idx)
        
        return DataLoader(
            self.train_ds,
            batch_size=self.global_cfg.train.batch_size,
            shuffle=True,
            num_workers=self.global_cfg.train.num_workers,
            collate_fn=cs2_collate_fn,
            pin_memory=True,
            persistent_workers=self.global_cfg.train.num_workers > 0
        )

    def val_dataloader(self):
        # Validation set usually shouldn't change, but consistent with original code:
        epoch_idx = 0
        if self.trainer:
            epoch_idx = self.trainer.current_epoch
            
        if self.ds_root is None:
             self.ds_root = DatasetRoot(self.global_cfg.dataset)

        self.val_ds = self.ds_root.build_epoch("val", epoch_idx)
        
        return DataLoader(
            self.val_ds,
            batch_size=self.global_cfg.train.batch_size,
            shuffle=False,
            num_workers=self.global_cfg.train.num_workers,
            collate_fn=cs2_collate_fn,
            pin_memory=True,
             persistent_workers=self.global_cfg.train.num_workers > 0
        )
