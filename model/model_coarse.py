from typing import Optional, Sequence, Tuple, Union, Any
import json
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet
from monai.bundle import ConfigParser
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from model.model_base import ModelBase

def define_instance(args: Namespace, instance_def_key: str) -> Any:
    """
    Define and instantiate an object based on the provided arguments and instance definition key.

    This function uses a ConfigParser to parse the arguments and instantiate an object
    defined by the instance_def_key.

    Args:
        args: An object containing the arguments to be parsed.
        instance_def_key (str): The key used to retrieve the instance definition from the parsed content.

    Returns:
        The instantiated object as defined by the instance_def_key in the parsed configuration.
    """
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)

FEATURE_STORE = {}

def get_hook(name):
    def hook(model, inp, outp):
        FEATURE_STORE[name] = outp.detach()
    return hook

def register_hooks(unet, layers_to_hook):
    """
    layers_to_hook: list of dicts:
        [{ "name": "fine", "layer_path": "up_blocks.3.resnets.0" }, ...]
    """
    for layer_info in layers_to_hook:
        name, path = layer_info["name"], layer_info["layer_path"]
        module = unet
        for p in path.split("."):
            module = getattr(module, p)
        module.register_forward_hook(get_hook(name))
        print(f"✔ Hook 注册: {name} → {path}")


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

class Adapter_for_diffusion(UNet):
    def __init__(
        self,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        out_channels: int = 32,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: int = 3,
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

        self.dimensions = 3
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.transformer_bottleneck = transformer_bottleneck
        
        self.down_layer_3 = self._get_down_layer(in_channels=64, out_channels=128, strides=2, is_top=True)
        self.down_layer_4 = self._get_down_layer(in_channels=128, out_channels=256, strides=2, is_top=False)
        self.bottom_layer = self._get_bottom_layer(in_channels=256+256, out_channels=256)

        self.up_layer_4 = self._get_up_layer(in_channels=512, out_channels=128, strides=2, is_top=False)
        self.up_layer_3 = self._get_up_layer(in_channels=256, out_channels=self.out_channels, strides=2, is_top=False)

        if self.transformer_bottleneck:
            self.bottleneck_tf = BottleneckTransformer3D(channels=256, num_layers=2, num_heads=8)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        dift_feat_fine, dift_feat_coarse = x
        down_3 = self.down_layer_3(dift_feat_fine)
        down_4 = self.down_layer_4(down_3)
        bottom = self.bottom_layer(torch.cat([down_4, dift_feat_coarse], dim=1))
        if self.transformer_bottleneck:
            bottom = self.bottleneck_tf(bottom)
        up_4 = self.up_layer_4(torch.cat([bottom, down_4], dim=1))
        up_3 = self.up_layer_3(torch.cat([up_4, down_3], dim=1))
        return F.normalize(up_3)

class model_coarse(ModelBase):
    """

    """
    def __init__(self, config: dict):

        self._load_configs(config)
        self.setup_params()
        super().__init__(config)
        self.setup_network()

    def _get_model_name(self):
        return "model_coarse"

    def setup_params(self):
        self.in_channels = self.model_config.get('in_channels', 1)
        self.num_res_units = self.model_config.get('num_res_units', 2)
        self.target_feature_size = self.model_config.get('target_feature_size', 32)
        self.max_epochs = self.run_config.get('max_epochs', 1000)
        self.spacing = self.model_config.get('spacing')
        self.DIFF_TIMESTEP = self.model_config['noise_timestep']
        self.AE_WEIGHTS = self.model_config['AE_WEIGHTS']
        self.UNET_WEIGHTS = self.model_config['UNET_WEIGHTS']
        self.CONFIG_JSON = self.model_config['CONFIG_JSON']
        self.LAYERS_TO_HOOK = [
            {"name": "coarse", "layer_path": "up_blocks.1.resnets.0"},
            {"name": "fine",   "layer_path": "up_blocks.3.resnets.0"}
        ]
        self.transformer_bottleneck = self.model_config.get('transformer_bottleneck', False)
    
    def setup_network(self):
        self.network = AE_unet_coarse(
            out_channels=self.target_feature_size,
            num_res_units=self.num_res_units,
            in_channels=self.in_channels,
            transformer_bottleneck=self.transformer_bottleneck
        )
        self.adapter = Adapter_for_diffusion(
            out_channels=self.target_feature_size,
            num_res_units=self.num_res_units,
            transformer_bottleneck=self.transformer_bottleneck
        )
        ae_model, unet, scheduler, scaling_factor, args = self.load_maisi_models(self.AE_WEIGHTS, self.UNET_WEIGHTS, self.CONFIG_JSON)
        self.ae_model = ae_model
        self.unet = unet
        self.scaling_factor = scaling_factor
        object.__setattr__(self, "scheduler", scheduler)
        object.__setattr__(self, "args", args)

        for p in self.ae_model.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        # （可选）确保它们处于 eval
        self.ae_model.eval()
        self.unet.eval()

        register_hooks(self.unet, self.LAYERS_TO_HOOK)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = image_tensor.device
        B, C, D, H, W = image_tensor.shape

        FEATURE_STORE.clear()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # 1. VAE 编码
                z_mu, z_sigma = self.ae_model.encode(image_tensor)
                latent = self.ae_model.sampling(z_mu, z_sigma)
                latent = latent * self.scaling_factor.to(device)

                # 2. 通过 U-Net 获取不同层次的特征
                noise = torch.randn_like(latent)
                t = torch.full((B,), self.DIFF_TIMESTEP, device=device, dtype=torch.long)
                noisy_latent = self.scheduler.add_noise(
                    original_samples=latent,
                    noise=noise,
                    timesteps=t
                )

                spacing_tensor = torch.tensor(self.spacing, device=device, dtype=torch.float16).unsqueeze(0).expand(B, -1) * 1e2
                modality_tensor = torch.tensor([0], device=device, dtype=torch.long)

                inputs = {
                    "x": noisy_latent,
                    "timesteps": t,
                    "spacing_tensor": spacing_tensor,
                    "class_labels": modality_tensor
                }

                _ = self.unet(**inputs)

                feat = FEATURE_STORE.copy()
                FEATURE_STORE.clear()
                dift_feat_fine = feat['fine'].detach()
                dift_feat_coarse = feat['coarse'].detach()
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # 3. 通过 Adapter 获取最终的对应特征
            teacher_features = self.adapter((dift_feat_fine, dift_feat_coarse))
            # 4. 通过 Coarse U-Net 获取粗对应特征
            student_features = self.network(image_tensor)

        return {
            "correspondence_output": student_features,
            "maisi_output": teacher_features
        }
    
    def load_maisi_models(
            self,
            ae_weights_path: str,
            unet_weights_path: str,
            config_json_path: str
        ):
        print("\n=== 正在加载模型 (Autoencoder + MAISI U-Net) ===")

        # -------------------------
        # 1. 加载 AE
        # -------------------------
        ae_def = {
            "spatial_dims": 3, "in_channels": 1, "out_channels": 1, "latent_channels": 4,
            "num_channels": [64, 128, 256], "num_res_blocks": [2, 2, 2],
            "norm_num_groups": 32, "norm_eps": 1e-06, "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False, "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False, "use_convtranspose": False, "norm_float16": True,
            "num_splits": 4, "dim_split": 1
        }

        ae_model = AutoencoderKlMaisi(**ae_def)
        ae_state_dict = torch.load(ae_weights_path, weights_only=False)
        ae_model.load_state_dict(ae_state_dict)
        ae_model.eval()
        print("✔ 自编码器加载成功")

        # -------------------------
        # 2. 加载 UNet + scheduler + args
        # -------------------------
        class ArgsNamespace:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)

        with open(config_json_path, "r") as f:
            args = ArgsNamespace(**json.load(f))

        unet = define_instance(args, "diffusion_unet_def")
        unet_ckpt = torch.load(unet_weights_path, weights_only=False)
        unet.load_state_dict(unet_ckpt["unet_state_dict"])
        unet.eval()
        print("✔ U-Net 加载成功")

        scale_factor = unet_ckpt["scale_factor"]

        # scheduler
        noise_scheduler = define_instance(args, "noise_scheduler")

        return ae_model, unet, noise_scheduler, scale_factor, args
    
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