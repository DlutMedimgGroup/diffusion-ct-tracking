# data_loader/dataloader_base.py

import os
import torch
from abc import ABC, abstractmethod
from monai.data import CacheDataset, PersistentDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class DataloaderBase(ABC):
    """
    An abstract base class for all dataloader modules in the framework.

    This class provides a common structure for creating MONAI DataLoaders and
    handles common logic such as performance settings, configurable dataset types
    (CacheDataset vs. PersistentDataset), and distributed (DDP) training.
    """
    def __init__(self, config: dict, inference: bool, is_ddp: bool = False):
        """
        Initializes the base dataloader.

        Args:
            config (dict): The complete configuration dictionary for the task.
            inference (bool): Flag indicating if the dataloader is for inference.
            is_ddp (bool): Flag indicating if running in DDP mode.
        """
        self.is_ddp_mode = is_ddp
        if not hasattr(self, '_configs_loaded'):
            # This is a fallback. The most specific child class should call this.
            self._load_configs(config, inference)

        self.inference = inference
        self.collate_fn = None

        # --- Setup common parameters from the parsed config sections ---
        if not self.inference:
            self.cache_rate = self.dataloader_config.get('cache_rate', 1.0)
            self.num_workers = self.dataloader_config.get('num_workers', 4)
            self.batch_size = self.dataloader_config.get('batch_size', 2)
            self.base_shuffle = True
            self.ddp = self.is_ddp_mode
        else:
            self.cache_rate, self.num_workers, self.batch_size, self.base_shuffle, self.ddp = 0.0, 1, 1, False, False

        if self.dataloader_config.get('debug_model', False):
            print("INFO: Dataloader running in DEBUG MODE (num_workers=0, no cache/shuffle).")
            self.cache_rate, self.num_workers, self.base_shuffle, self.ddp = 0.0, 0, False, False

        self.partial_dataset = self.dataloader_config.get('partial_dataset', -1)

        self.init_data_list()
        self.init_transforms()

    def _load_configs(self, config: dict, inference: bool):
        """
        Parses the main config dictionary to set up instance-level config attributes.
        """
        if inference:
            inference_dataloader_name = config['Inference'].get('inference_dataloader_name')
            self.data_config = {
                'dataloader_name': inference_dataloader_name,
                inference_dataloader_name: config['Inference'].get(inference_dataloader_name),
            }
            self.model_config = config['Model']
        else:
            self.data_config = config['Data']
            self.model_config = config['Model']
            
        self.task_config = config['Task']
        dataloader_name = self.data_config['dataloader_name']
        self.dataloader_config = self.data_config[dataloader_name]
        self._configs_loaded = True

    @abstractmethod
    def init_data_list(self):
        """Abstract method to initialize train/val/test data lists."""
        raise NotImplementedError

    @abstractmethod
    def init_transforms(self):
        """Abstract method to initialize MONAI data transformations."""
        raise NotImplementedError

    def _create_dataset(self, data_list: list, transform: callable, split_name: str):
        """
        Internal factory method to create a MONAI Dataset instance based on the
        'dataset_type' specified in the configuration ('Persistent' or 'Cache').
        """
        dataset_type = self.data_config.get("dataset_type", "Cache").lower()
        is_main_process = not self.ddp or torch.distributed.get_rank() == 0

        if dataset_type == "persistent":
            cache_dir = self.data_config.get('cache_dir')
            if not cache_dir:
                raise ValueError("`cache_dir` must be specified in the [Data] config when using PersistentDataset.")
            
            persistent_cache_dir = os.path.join(cache_dir, self.task_config['task_name'], split_name)
            if is_main_process:
                print(f"INFO: Using PersistentDataset with cache directory: {persistent_cache_dir}")
            
            return PersistentDataset(data=data_list, transform=transform, cache_dir=persistent_cache_dir)
        
        # Default to CacheDataset
        if dataset_type != "cache" and is_main_process:
            print(f"Warning: Unknown dataset_type '{dataset_type}'. Defaulting to 'Cache'.")
        
        if is_main_process:
            print(f"INFO: Using CacheDataset with cache_rate: {self.cache_rate}")

        return CacheDataset(
            data=data_list, transform=transform,
            cache_rate=self.cache_rate, num_workers=self.num_workers
        )

    def get_train_loader(self) -> DataLoader:
        """
        Constructs the DataLoader for the training set. In DDP mode, it uses a
        DistributedSampler to partition the data across processes.
        """
        if not hasattr(self, 'train_list') or not self.train_list: return None
            
        train_ds = self._create_dataset(self.train_list, self.train_transform, "train")
        sampler = DistributedSampler(train_ds, shuffle=self.base_shuffle) if self.ddp else None
        
        return DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=(sampler is None and self.base_shuffle),
            num_workers=self.num_workers, sampler=sampler, collate_fn=self.collate_fn, pin_memory=True
        )

    def get_val_loader(self) -> DataLoader:
        """
        Constructs the DataLoader for the validation set. Validation is typically
        not distributed to simplify metric calculation.
        """
        if not hasattr(self, 'val_list') or not self.val_list: return None
            
        val_ds = self._create_dataset(self.val_list, self.val_transform, "val")
        
        return DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers,
            collate_fn=self.collate_fn, pin_memory=True
        )

    def get_test_loader(self) -> DataLoader:
        """Constructs the DataLoader for the test set."""
        if not hasattr(self, 'test_list') or not self.test_list: return None

        # Testing usually doesn't need heavy caching, but we can still use the factory
        test_ds = self._create_dataset(self.test_list, self.val_transform, "test")

        return DataLoader(
            test_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, collate_fn=self.collate_fn
        )
    