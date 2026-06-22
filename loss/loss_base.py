# losses/loss_base.py

from abc import ABC, abstractmethod
import torch.nn as nn

class LossBase(nn.Module, ABC):
    """
    An abstract base class for all loss function modules in the framework.

    It ensures every loss function is a `torch.nn.Module` and defines a
    consistent initialization pattern. The most specific child class is expected
    to call `_load_configs` to parse the configuration.
    """
    def __init__(self, config: dict):
        """
        Initializes the base loss function module.

        Args:
            config (dict): The complete configuration dictionary for the task.
        """
        super().__init__()
        # If the `_configs_loaded` flag is not set by a child class, this base
        # class will load them as a fallback.
        if not hasattr(self, '_configs_loaded'):
            print("Warning: LossBase is loading configs. This should ideally be handled by a child class.")
            self._load_configs(config)

    def _load_configs(self, config: dict):
        """
        Parses the main config to set up loss-related instance config attributes.
        This is intended to be called once by the most specific child class.
        """
        loss_cfg_section = config.get('Loss')
        if loss_cfg_section is None:
            raise KeyError("Config must contain a [Loss] section.")
            
        loss_name = self._get_loss_name()
        self.loss_config = loss_cfg_section[loss_name]
        
        # Set the flag to prevent parent classes from re-loading configs.
        self._configs_loaded = True

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Abstract forward pass method to compute the loss."""
        raise NotImplementedError