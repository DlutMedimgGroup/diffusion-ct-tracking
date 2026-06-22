# engine/trainer_base.py

import os
import sys
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from abc import ABC, abstractmethod
from monai.utils import set_determinism

def get_class(class_path: str):
    """
    Dynamically imports a class from a string path (e.g., 'package.ClassName').
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Use basic print here as logging might not be set up yet.
        print(f"FATAL: Could not import {class_path}: {e}")
        sys.exit(1)

class TrainerBase(ABC):
    """
    An abstract base class for trainer modules, with support for DDP.

    Handles component instantiation, device setup, logging, checkpointing, and
    the core training loop structure. It differentiates between single-GPU and
    multi-GPU (DDP) execution based on the `is_ddp` flag.
    """
    def __init__(self, config: dict, is_ddp: bool = False):
        """
        Initializes the base trainer.

        Args:
            config (dict): The complete configuration dictionary for the task.
            is_ddp (bool): Flag indicating if running in DDP mode.
        """
        self.config = config
        self.task_config = config['Task']
        self.run_config = config['Run']
        
        # --- DDP Setup ---
        self.is_ddp = is_ddp
        self.rank = 0 # Default rank is 0 for single-GPU mode
        if self.is_ddp:
            self._setup_ddp()
        
        # --- Path and Directory Setup ---
        self.project_path = self.task_config['project_path']
        self.task_name = self.task_config['task_name']
        self.output_dir = os.path.join(self.project_path, 'trained_models', self.task_name)
        
        # Only the main process should create directories.
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # --- Logging, Device, and Reproducibility ---
        self._setup_logging()
        self.device = torch.device(self.rank if self.is_ddp else self.run_config.get('device', 'cuda:0'))
        if self.run_config.get('reproducibility', False):
            set_determinism(seed=42)

        # --- Component Placeholders ---
        self.dataloader, self.model, self.loss_fn = None, None, None
        self.optimizer, self.lr_scheduler, self.start_epoch = None, None, 0

    def _setup_ddp(self):
        """Initializes the distributed process group and sets ranks."""
        if not dist.is_available():
            raise RuntimeError("Distributed training is not available.")
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        torch.cuda.set_device(self.rank)
        print(f"DDP Initialized: Rank {self.rank}/{self.world_size} on device cuda:{self.rank}")

    def _setup_logging(self):
        """Sets up logging. Only the main process (rank 0) writes to files and console."""
        self.logger = logging.getLogger(f"{self.task_name}_rank{self.rank}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        
        # Only the main process gets handlers that write output.
        if self.rank == 0:
            formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            # File Handler
            file_handler = logging.FileHandler(os.path.join(self.output_dir, "training.log"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            # Console Handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        else:
            # Other processes get a NullHandler to suppress messages.
            self.logger.addHandler(logging.NullHandler())

    def _setup(self):
        """Sets up all components for training."""
        self.logger.info("--- Setting up training components ---")
        
        dataloader_name = self.config['Data']['dataloader_name']
        dataloader_class = get_class(f"data_loader.{dataloader_name}")
        self.dataloader = dataloader_class(config=self.config, inference=False, is_ddp=self.is_ddp)
        self.train_loader = self.dataloader.get_train_loader()
        self.val_loader = self.dataloader.get_val_loader()
        self.logger.info(f"Dataloader '{dataloader_name}' loaded.")

        model_name = self.config['Model']['model_name']
        
        # Check if a pre-instantiated model is provided (useful for testing)
        if 'model_instance' in self.config['Model']:
            model = self.config['Model']['model_instance'].to(self.device)
            self.logger.info(f"Using pre-instantiated model: {model.__class__.__name__}")
        else:
            model_class = get_class(f"model.{model_name}")
            model = model_class(config=self.config).to(self.device)

        if self.is_ddp:
            # find_unused_parameters=True might be safer for complex models where some
            # parameters might not be used in every forward pass.
            self.model = DDP(model, device_ids=[self.rank], find_unused_parameters=False)
        else:
            self.model = model
        
        # This assumes the model has a get_total_params method. Let's make it optional.
        if hasattr(self.model, 'get_total_params'):
            total_params = self.model.module.get_total_params() if self.is_ddp else self.model.get_total_params()
            self.logger.info(f"Model '{model_name}' loaded with {total_params:,} parameters.")
        else:
            self.logger.info(f"Model '{model_name}' loaded.")

        # Allow dummy loss for testing
        loss_name = self.config['Loss']['loss_name']
        if loss_name != "DummyLoss":
            loss_class = get_class(f"loss.{loss_name}")
            self.loss_fn = loss_class(config=self.config).to(self.device)
            self.logger.info(f"Loss function '{loss_name}' loaded.")
        else:
            self.loss_fn = None # No loss needed for this test
            self.logger.info("Using DummyLoss (no loss function loaded).")


        model_to_configure = self.model.module if self.is_ddp else self.model
        # Check if model has this method, as our dummy model won't
        if hasattr(model_to_configure, 'configure_optimizers'):
             self.optimizer, self.lr_scheduler = model_to_configure.configure_optimizers()
        else:
            # Create a default optimizer for the dummy model
            opt_config = self.config['Model'].get('Optimizer', {})
            opt_name = opt_config.get('name', 'Adam')
            opt_lr = opt_config.get('learning_rate', 1e-4)
            if opt_name == 'Adam':
                 self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_lr)
            # Create a dummy scheduler
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100)


        if self.run_config.get('continue_training', False):
            self._load_checkpoint()
        
        if self.is_ddp: dist.barrier()

    def _load_checkpoint(self):
        """Loads state to resume training. Only rank 0 loads from disk."""
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"'continue_training' is true, but checkpoint not found. Starting fresh.")
            return

        map_location = {'cuda:0': f'cuda:{self.rank}'} if self.is_ddp else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")

        model_to_load = self.model.module if self.is_ddp else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        
        # Restore trainer-specific state only on the main process
        if self.rank == 0:
            self.best_val_tre = checkpoint.get('best_val_tre', float('inf'))
            self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
            self.pth_best_tre = checkpoint.get('pth_best_tre', "")
            self.pth_best_dice = checkpoint.get('pth_best_dice', "")
            self.logger.info(f"Resumed from epoch {self.start_epoch}.")

    @abstractmethod
    def train_epoch(self, epoch: int):
        raise NotImplementedError

    @abstractmethod
    def validate_epoch(self, epoch: int):
        raise NotImplementedError

    def run(self):
        """The main entry point to start the training process."""
        self._setup()
        start_message = f"--- Starting DDP Training on {self.world_size} GPUs ---" if self.is_ddp else f"--- Starting Training on device {self.device} ---"
        self.logger.info(start_message)
        
        for epoch in range(self.start_epoch, self.run_config['max_epochs']):
            self.logger.info(f"\n===== Epoch {epoch}/{self.run_config['max_epochs'] - 1} =====")
            self.train_epoch(epoch)
            
            val_interval = self.run_config.get('val_interval', 1)
            if val_interval > 0 and (epoch + 1) % val_interval == 0:
                self.validate_epoch(epoch)
        
        if self.is_ddp:
            dist.destroy_process_group()
        
        self.logger.info("\n--- Training Finished ---")
        