from typing import Optional, Sequence, Tuple, Union, Any
import json
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet
from model.model_base import ModelBase

class BottleneckTransformer3D(nn.Module):
    def __init__(self, channels=512, num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(channels)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B,C,D,H,W]
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)  # [B,N,C]
        x = self.norm_in(x)
        x = self.encoder(x)  # [B,N,C]
        x = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()    # [B,C,D,H,W]
        return x

class AE_unet_coarse(UNet):
    def __init__(
        self,
        out_channels: int = 32,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        in_channels: int = 1,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        transformer_bottleneck = True,
    ) -> None:
        
        super(UNet, self).__init__()

        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.out_channels = out_channels
        self.dimensions = 3
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.transformer_bottleneck = transformer_bottleneck

        self.down_layer_1 = self._get_down_layer(in_channels=in_channels, out_channels=64, strides=2, is_top=True)
        self.down_layer_2 = self._get_down_layer(in_channels=64, out_channels=128, strides=2, is_top=False)
        self.down_layer_3 = self._get_down_layer(in_channels=128, out_channels=256, strides=2, is_top=False)
        self.down_layer_4 = self._get_down_layer(in_channels=256, out_channels=512, strides=2, is_top=False)
        self.bottom_layer = self._get_bottom_layer(in_channels=512, out_channels=512)
        self.up_layer_4 = self._get_up_layer(in_channels=1024, out_channels=256, strides=2, is_top=False)
        self.up_layer_3 = self._get_up_layer(in_channels=256+256, out_channels=self.out_channels, strides=2, is_top=True)

        if self.transformer_bottleneck:
            self.bottleneck_tf = BottleneckTransformer3D(channels=512, num_layers=2, num_heads=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_1 = self.down_layer_1(x)
        down_2 = self.down_layer_2(down_1)
        down_3 = self.down_layer_3(down_2)
        down_4 = self.down_layer_4(down_3)
        bottom = self.bottom_layer(down_4)
        if self.transformer_bottleneck:
            bottom = self.bottleneck_tf(bottom)
        up_4 = self.up_layer_4(torch.cat([bottom, down_4], dim=1))
        up_3 = self.up_layer_3(torch.cat([up_4, down_3], dim=1))

        coarse_out = F.normalize(up_3, p=2, dim=1)
        
        return coarse_out

class model_coarse_wodiff(ModelBase):
    """

    """
    def __init__(self, config: dict):

        self._load_configs(config)
        self.setup_params()
        super().__init__(config)
        self.setup_network()

    def _get_model_name(self):
        return "model_coarse_wodiff"

    def setup_params(self):
        self.in_channels = self.model_config.get('in_channels', 1)
        self.num_res_units = self.model_config.get('num_res_units', 2)
        self.target_feature_size = self.model_config.get('target_feature_size', 32)
        self.max_epochs = self.run_config.get('max_epochs', 1000)
        self.spacing = self.model_config.get('spacing')
        self.transformer_bottleneck = self.model_config.get('transformer_bottleneck', False)
    
    def setup_network(self):
        self.network = AE_unet_coarse(
            out_channels=self.target_feature_size,
            num_res_units=self.num_res_units,
            in_channels=self.in_channels,
            transformer_bottleneck=self.transformer_bottleneck
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = image_tensor.device
        B, C, D, H, W = image_tensor.shape
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            student_features = self.network(image_tensor)

        return {
            "correspondence_output": student_features,
        }
    
    def inference(self, image_tensor: torch.Tensor):
        correspondence_features = self.network(image_tensor)
        return {
            "correspondence_output": correspondence_features,
        }

    def configure_optimizers(self) -> (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler):
        """
        Configures the optimizer and scheduler from the config file.
        This provides a default implementation for most common use cases.
        """
        optimizer_cfg = self.model_cfg_section.get('Optimizer')
        scheduler_cfg = self.model_cfg_section.get('LRScheduler')

        if not optimizer_cfg:
            raise KeyError("[Model.Optimizer] section is missing from the config.")

        optimizer_name = optimizer_cfg.get('name', 'Adam').lower()
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=optimizer_cfg.get('learning_rate', 1e-4),
                weight_decay=optimizer_cfg.get('weight_decay', 1e-5)
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=optimizer_cfg.get('learning_rate', 1e-4), 
                weight_decay=optimizer_cfg.get('weight_decay', 1e-5)
            )
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' is not yet supported in the base config.")

        if not scheduler_cfg:
            print("Warning: [Model.LRScheduler] not found. No learning rate scheduler will be used.")
            return optimizer, None
            
        scheduler_name = scheduler_cfg.get('name', 'CosineAnnealingLR').lower()
        if scheduler_name == 'cosineannealinglr':
            # Get T_max from the scheduler config, defaulting to 0.
            t_max_from_config = scheduler_cfg.get('T_max', 0)
            
            # Get max_epochs from the run config.
            max_epochs = self.run_config.get('max_epochs')
            if max_epochs is None:
                raise KeyError("`max_epochs` must be defined in the [Run] config section to use CosineAnnealingLR.")

            # If T_max is set to 0 (or not set), automatically use max_epochs.
            if t_max_from_config == 0:
                final_t_max = max_epochs
                print(f"INFO: 'T_max' for scheduler not specified or set to 0. Defaulting to 'max_epochs' ({final_t_max}).")
            else:
                final_t_max = t_max_from_config
                print(f"INFO: Using specified 'T_max' for scheduler: {final_t_max}.")

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_t_max)
        else:
            raise NotImplementedError(f"Scheduler '{scheduler_name}' is not yet supported in the base config.")
        
        print(f"Optimizer '{optimizer_name}' and Scheduler '{scheduler_name}' configured.")
        return optimizer, scheduler