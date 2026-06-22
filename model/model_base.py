# models/model_base.py

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class ModelBase(nn.Module, ABC):
    """
    An abstract base class for all models in the framework.

    It ensures that every model is a `torch.nn.Module` and defines a consistent
    initialization and optimizer configuration pattern. The most specific child
    class is expected to call `_load_configs` to parse the configuration.
    """
    def __init__(self, config: dict):
        """
        Initializes the base model.

        Args:
            config (dict): The complete configuration dictionary for the task.
        """
        super().__init__()
        # If the `_configs_loaded` flag is not set by a child class, this base
        # class will load them as a fallback.
        if not hasattr(self, '_configs_loaded'):
            print("Warning: ModelBase is loading configs. This should ideally be handled by a child class.")
            self._load_configs(config)

    def _load_configs(self, config: dict):
        """
        Parses the main config to set up model-related instance config attributes.
        This is intended to be called once by the most specific child class.
        """
        # We assume training mode here. For inference, a flag would be needed
        # or the calling context would pass the correct config section.
        self.model_cfg_section = config.get('Model', config.get('Inference', {}).get('Model'))
        if self.model_cfg_section is None:
            raise KeyError("Config must contain a [Model] or [Inference.Model] section.")
            
        model_name = self._get_model_name()
        self.model_config = self.model_cfg_section[model_name]
        self.run_config = config.get('Run', config.get('Inference'))
        
        # Set the flag to prevent parent classes from re-loading configs.
        self._configs_loaded = True

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Abstract forward pass method."""
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self) -> (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler):
        """Abstract method to configure and return the model's optimizer and LR scheduler."""
        raise NotImplementedError

    def get_total_params(self) -> int:
        """Calculates and returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)