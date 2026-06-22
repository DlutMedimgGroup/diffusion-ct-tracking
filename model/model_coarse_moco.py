from typing import Optional, Sequence, Tuple, Union, Any, Dict
import copy
import json
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.layers.factories import Act, Norm
from monai.networks.nets import UNet
from monai.bundle import ConfigParser
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.schedulers.rectified_flow import RFlowScheduler
from monai.networks.nets.diffusion_model_unet import UpBlock, AttnUpBlock, CrossAttnUpBlock
from monai.utils.type_conversion import convert_to_tensor

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

    _ = parser.get_parsed_content(instance_def_key, instantiate=False)
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

class DiffusionModelUNetMaisiWithFeatures(DiffusionModelUNetMaisi):
    """
    在不使用 forward hook 的前提下，精确返回 up_blocks[i].resnets[j] 的输出特征。

    默认行为与父类一致：return_features=False 时，只返回 h_tensor。
    当 return_features=True 时，返回 (h_tensor, feats_dict)，例如：

        output, {"coarse": feat_1_1, "fine": feat_3_1}

    其中 feat_1_1 表示 up_blocks[1].resnets[1] 的输出（attention 之前）。
    """

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        top_region_index_tensor: Optional[torch.Tensor] = None,
        bottom_region_index_tensor: Optional[torch.Tensor] = None,
        spacing_tensor: Optional[torch.Tensor] = None,
        return_features: bool = False,
        # 默认：coarse = up_blocks[1].resnets[1], fine = up_blocks[3].resnets[1]
        feature_positions: Dict[str, Tuple[int, int]] = None,
    ):
        if feature_positions is None:
            feature_positions = {"coarse": (1, 1), "fine": (3, 1)}

        # ---- 与原 forward 一致的部分 ----
        emb = self._get_time_and_class_embedding(x, timesteps, class_labels)
        emb = self._get_input_embeddings(emb, top_region_index_tensor, bottom_region_index_tensor, spacing_tensor)

        h = self.conv_in(x)
        h, down_res = self._apply_down_blocks(h, emb, context, down_block_additional_residuals)
        h = self.middle_block(h, emb, context)

        if mid_block_additional_residual is not None:
            h += mid_block_additional_residual

        # 不需要特征：直接走原来的 up path
        if not return_features:
            h = self._apply_up_blocks(h, emb, context, down_res)
            h = self.out(h)
            h_tensor: torch.Tensor = convert_to_tensor(h)
            return h_tensor

        # 需要特征：用自定义的 up path
        h, feats = self._apply_up_blocks_with_features(
            h,
            emb,
            context,
            down_res,
            feature_positions=feature_positions,
        )

        h = self.out(h)
        h_tensor: torch.Tensor = convert_to_tensor(h)
        return h_tensor, feats

    def _apply_up_blocks_with_features(
        self,
        h: torch.Tensor,
        emb: torch.Tensor,
        context: Optional[torch.Tensor],
        down_block_res_samples,
        feature_positions: Dict[str, Tuple[int, int]],
    ):
        """
        仿照 DiffusionModelUNetMaisi._apply_up_blocks，但展开每个 up_block 的 forward，
        在指定的 (up_idx, res_idx) 位置记录 resnet 输出。
        """
        feats: Dict[str, torch.Tensor] = {k: None for k in feature_positions.keys()}

        for up_idx, up_block in enumerate(self.up_blocks):
            # 与原 _apply_up_blocks 相同：每个 up_block 取对应数目的 skip features
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            # 根据 up_block 的具体类型分别展开 forward
            if isinstance(up_block, UpBlock):
                h = self._forward_upblock_no_attn(
                    up_block, up_idx, h, res_samples, emb, context, feature_positions, feats
                )
            elif isinstance(up_block, AttnUpBlock):
                h = self._forward_upblock_with_self_attn(
                    up_block, up_idx, h, res_samples, emb, context, feature_positions, feats
                )
            elif isinstance(up_block, CrossAttnUpBlock):
                h = self._forward_upblock_with_cross_attn(
                    up_block, up_idx, h, res_samples, emb, context, feature_positions, feats
                )
            else:
                # 理论上不会走到这里；如果 MONAI 将来新增类型，可在此加分支
                h = up_block(
                    hidden_states=h,
                    res_hidden_states_list=res_samples,
                    temb=emb,
                    context=context,
                )

        return h, feats

    # ------------------ 三种 up_block 的展开实现 ------------------ #

    def _maybe_record_feature(
        self,
        feats: Dict[str, torch.Tensor],
        feature_positions: Dict[str, Tuple[int, int]],
        up_idx: int,
        res_idx: int,
        h: torch.Tensor,
    ):
        """
        检查当前 (up_idx, res_idx) 是否是我们想要的 feature 位置，是的话记录 h。
        """
        for name, (target_up, target_res) in feature_positions.items():
            if up_idx == target_up and res_idx == target_res:
                feats[name] = h

    def _forward_upblock_no_attn(
        self,
        up_block: UpBlock,
        up_idx: int,
        h: torch.Tensor,
        res_samples: list,
        emb: torch.Tensor,
        context: Optional[torch.Tensor],
        feature_positions: Dict[str, Tuple[int, int]],
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del context
        for res_idx, resnet in enumerate(up_block.resnets):
            res_hidden_states = res_samples[-1]
            res_samples = res_samples[:-1]

            h = torch.cat([h, res_hidden_states], dim=1)
            h = resnet(h, emb)  # ✅ 位置参数

            self._maybe_record_feature(feats, feature_positions, up_idx, res_idx, h)

        if up_block.upsampler is not None:
            h = up_block.upsampler(h, emb)  # ✅ 位置参数

        return h


    def _forward_upblock_with_self_attn(
        self,
        up_block: AttnUpBlock,
        up_idx: int,
        h: torch.Tensor,
        res_samples: list,
        emb: torch.Tensor,
        context: Optional[torch.Tensor],
        feature_positions: Dict[str, Tuple[int, int]],
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        del context
        for res_idx, (resnet, attn) in enumerate(zip(up_block.resnets, up_block.attentions)):
            res_hidden_states = res_samples[-1]
            res_samples = res_samples[:-1]

            h = torch.cat([h, res_hidden_states], dim=1)
            h = resnet(h, emb)  # ✅ 位置参数

            self._maybe_record_feature(feats, feature_positions, up_idx, res_idx, h)

            h = attn(h).contiguous()

        if up_block.upsampler is not None:
            h = up_block.upsampler(h, emb)  # ✅ 位置参数

        return h


    def _forward_upblock_with_cross_attn(
        self,
        up_block: CrossAttnUpBlock,
        up_idx: int,
        h: torch.Tensor,
        res_samples: list,
        emb: torch.Tensor,
        context: Optional[torch.Tensor],
        feature_positions: Dict[str, Tuple[int, int]],
        feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        for res_idx, (resnet, attn) in enumerate(zip(up_block.resnets, up_block.attentions)):
            res_hidden_states = res_samples[-1]
            res_samples = res_samples[:-1]

            h = torch.cat([h, res_hidden_states], dim=1)
            h = resnet(h, emb)  # ✅ 位置参数

            self._maybe_record_feature(feats, feature_positions, up_idx, res_idx, h)

            h = attn(h, context=context)

        if up_block.upsampler is not None:
            h = up_block.upsampler(h, emb)  # ✅ 位置参数

        return h

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

class model_coarse_moco(ModelBase):
    """

    """
    def __init__(self, config: dict):

        self._load_configs(config)
        self.setup_params()
        super().__init__(config)
        self.setup_network()

    def _get_model_name(self):
        return "model_coarse_moco"

    def setup_params(self):
        self.in_channels = self.model_config.get('in_channels', 1)
        self.num_res_units = self.model_config.get('num_res_units', 2)
        self.target_feature_size = self.model_config.get('target_feature_size', 32)
        self.max_epochs = self.run_config.get('max_epochs', 1000)
        self.spacing = self.model_config.get('spacing')
        self.DIFF_TIMESTEP = self.model_config['noise_timestep']
        # --- diffusion teacher weights (optional) ---
        self.AE_WEIGHTS = self.model_config.get("AE_WEIGHTS", "")
        self.UNET_WEIGHTS = self.model_config.get("UNET_WEIGHTS", "")

        def _valid_path(p):
            return isinstance(p, str) and len(p.strip()) > 0 and os.path.exists(p)

        # If either path is empty or missing, do NOT load diffusion teacher (inference/lightweight mode)
        self.enable_diffusion_teacher = _valid_path(self.AE_WEIGHTS) and _valid_path(self.UNET_WEIGHTS)

        if self.enable_diffusion_teacher:
            print(f"✔ Diffusion teacher ENABLED (AE='{self.AE_WEIGHTS}', UNet='{self.UNET_WEIGHTS}')")
        else:
            print("⚠ Diffusion teacher DISABLED (AE_WEIGHTS/UNET_WEIGHTS empty or not found).")
        self.LAYERS_TO_HOOK = [
            {"name": "coarse", "layer_path": "up_blocks.1.resnets.0"},
            {"name": "fine",   "layer_path": "up_blocks.3.resnets.0"}
        ]
        self.transformer_bottleneck = self.model_config.get('transformer_bottleneck', False)
        self.moco_m = self.model_config.get("moco_m", 0.999)
    
    def setup_network(self):
        # 1) student: coarse UNet
        self.network_q = AE_unet_coarse(
            out_channels=self.target_feature_size,
            num_res_units=self.num_res_units,
            in_channels=self.in_channels,
            transformer_bottleneck=self.transformer_bottleneck
        )
        self.network_k = copy.deepcopy(self.network_q)
        for p in self.network_k.parameters():
            p.requires_grad_(False)
        self.network_k.eval()

        # 2) adapter + diffusion parts (optional)
        if self.enable_diffusion_teacher:
            # adapter_q participates in training
            self.adapter_q = Adapter_for_diffusion(
                out_channels=self.target_feature_size,
                num_res_units=self.num_res_units,
                transformer_bottleneck=self.transformer_bottleneck,
            )
            self.adapter_k = copy.deepcopy(self.adapter_q)
            for p in self.adapter_k.parameters():
                p.requires_grad_(False)
            self.adapter_k.eval()

            # frozen diffusion parts
            ae_model, unet, scheduler, scaling_factor = self.load_maisi_models(
                self.AE_WEIGHTS, self.UNET_WEIGHTS
            )
            self.ae_model = ae_model
            self.unet = unet
            self.scaling_factor = scaling_factor
            object.__setattr__(self, "scheduler", scheduler)

            for p in self.ae_model.parameters():
                p.requires_grad_(False)
            for p in self.unet.parameters():
                p.requires_grad_(False)
            self.ae_model.eval()
            self.unet.eval()

            register_hooks(self.unet, self.LAYERS_TO_HOOK)

        else:
            # lightweight inference mode (no diffusion teacher)
            self.adapter_q = None
            self.adapter_k = None
            self.ae_model = None
            self.unet = None
            self.scaling_factor = None
            object.__setattr__(self, "scheduler", None)

    def forward(self, image_tensor: torch.Tensor):
        # --- lightweight path: no diffusion teacher ---
        if not self.enable_diffusion_teacher:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                student_q = self.network_q(image_tensor)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    student_k = self.network_k(image_tensor)

            return {
                "student_q": student_q,
                "student_k": student_k,
                "teacher_q": None,
                "teacher_k": None,
            }
        device = image_tensor.device
        B, C, D, H, W = image_tensor.shape

        FEATURE_STORE.clear()

        # ---- A) frozen diffusion 提取中间特征（不动）----
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_mu, z_sigma = self.ae_model.encode(image_tensor)
                latent = self.ae_model.sampling(z_mu, z_sigma)
                latent = latent * self.scaling_factor.to(device)

                noise = torch.randn_like(latent)
                t = torch.full((B,), self.DIFF_TIMESTEP, device=device, dtype=torch.long)
                noisy_latent = self.scheduler.add_noise(latent, noise, t)

                spacing_tensor = torch.tensor(self.spacing, device=device, dtype=torch.float16).unsqueeze(0).expand(B, -1) * 1e2
                modality_tensor = torch.tensor([0], device=device, dtype=torch.long)

                _ = self.unet(
                    x=noisy_latent,
                    timesteps=t,
                    spacing_tensor=spacing_tensor,
                    class_labels=modality_tensor
                )

                feat = FEATURE_STORE.copy()
                FEATURE_STORE.clear()
                dift_feat_fine = feat["fine"].detach()
                dift_feat_coarse = feat["coarse"].detach()

        # ---- B) adapter q/k ----
        # adapter_q 参与训练
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            teacher_q = self.adapter_q((dift_feat_fine, dift_feat_coarse))

        # adapter_k 不参与训练
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                teacher_k = self.adapter_k((dift_feat_fine, dift_feat_coarse))

        # ---- C) student q/k ----
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            student_q = self.network_q(image_tensor)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                student_k = self.network_k(image_tensor)

        return {
            "student_q": student_q,
            "student_k": student_k,
            "teacher_q": teacher_q,
            "teacher_k": teacher_k,
        }
    
    def load_maisi_models(
            self,
            ae_weights_path: str,
            unet_weights_path: str,
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
        
        unet_def = {
            "spatial_dims": 3, "in_channels": 4, "out_channels": 4,
            "num_channels": [64, 128, 256, 512],
            "attention_levels": [False, False, True, True],
            "num_head_channels": [0, 0, 32, 32],
            "num_res_blocks": 2,
            "use_flash_attention": True,
            "include_top_region_index_input": False,
            "include_bottom_region_index_input": False,
            "include_spacing_input": True,
            "num_class_embeds": 128,
            "resblock_updown": True,
            "include_fc": True
        }
        unet = DiffusionModelUNetMaisiWithFeatures(**unet_def)
        unet_ckpt = torch.load(unet_weights_path, weights_only=False)
        unet.load_state_dict(unet_ckpt["unet_state_dict"])
        unet.eval()
        print("✔ U-Net 加载成功")

        scale_factor = unet_ckpt["scale_factor"]

        # scheduler
        scheduler_def = {
            "num_train_timesteps": 1000,
            "use_discrete_timesteps": False,
            "use_timestep_transform": True,
            "sample_method": "uniform",
            "scale":1.4
        }
        noise_scheduler = RFlowScheduler(**scheduler_def)

        return ae_model, unet, noise_scheduler, scale_factor
    
    def inference(self, image_tensor: torch.Tensor):
        correspondence_features = self.network_q(image_tensor)
        return {
            "correspondence_output": correspondence_features,
        }

    @torch.no_grad()
    def _momentum_update(self):
        # student
        m_s = self.moco_m
        for p_q, p_k in zip(self.network_q.parameters(), self.network_k.parameters()):
            p_k.data.mul_(m_s).add_(p_q.data, alpha=(1.0 - m_s))

        # adapter
        m_a = self.moco_m
        for p_q, p_k in zip(self.adapter_q.parameters(), self.adapter_k.parameters()):
            p_k.data.mul_(m_a).add_(p_q.data, alpha=(1.0 - m_a))

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