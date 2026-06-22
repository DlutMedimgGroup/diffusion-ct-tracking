from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from model.model_base import ModelBase
from model.model_coarse_moco import DiffusionModelUNetMaisiWithFeatures


class model_diffusion_feature(ModelBase):
    """
    仅用于测试扩散模型中间特征效果的模型。

    功能：
        inference(image) -> 返回 diffusion UNet 的中间特征

    不可训练。
    不包含 student / adapter。
    """

    def __init__(self, config: dict):
        self._load_configs(config)
        self.setup_params()
        super().__init__(config)
        self.setup_network()

    def _get_model_name(self):
        return "model_diffusion_feature"

    # ---------------------------------------------------
    # 1️⃣ 读取配置
    # ---------------------------------------------------

    def setup_params(self):
        self.in_channels = self.model_config.get("in_channels", 1)
        self.spacing = self.model_config["spacing"]
        self.DIFF_TIMESTEP = self.model_config["noise_timestep"]
        self.AE_WEIGHTS = self.model_config["AE_WEIGHTS"]
        self.UNET_WEIGHTS = self.model_config["UNET_WEIGHTS"]

        # 默认提取 coarse / fine
        self.feature_positions = {
            "coarse": (1, 0),   # up_blocks[1].resnets[0]
            "fine": (3, 0),     # up_blocks[3].resnets[0]
        }

    # ---------------------------------------------------
    # 2️⃣ 加载冻结 diffusion 模型
    # ---------------------------------------------------

    def setup_network(self):

        ae_model, unet, scheduler, scale_factor = self.load_maisi_models(
            self.AE_WEIGHTS,
            self.UNET_WEIGHTS
        )

        self.ae_model = ae_model
        self.unet = unet

        # ❗关键修改：不注册 scheduler 为子模块，避免 eval()/to() 报错
        self.__dict__["scheduler"] = scheduler

        self.scaling_factor = scale_factor

        for p in self.ae_model.parameters():
            p.requires_grad_(False)

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.eval()

    # ---------------------------------------------------
    # 3️⃣ Forward（不建议训练时调用）
    # ---------------------------------------------------

    def forward(self, image_tensor: torch.Tensor):
        raise RuntimeError(
            "This model is inference-only. "
            "Call inference() instead."
        )

    # ---------------------------------------------------
    # 4️⃣ Inference：返回 diffusion 中间特征
    # ---------------------------------------------------

    @torch.no_grad()
    def inference(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:

        device = image_tensor.device
        B = image_tensor.shape[0]

        with torch.autocast(device_type="cuda", dtype=torch.float16):

            # ---- AE 编码 ----
            z_mu, z_sigma = self.ae_model.encode(image_tensor)
            latent = self.ae_model.sampling(z_mu, z_sigma)
            latent = latent * self.scaling_factor.to(device)

            # ---- 加噪 ----
            noise = torch.randn_like(latent)
            t = torch.full((B,), self.DIFF_TIMESTEP, device=device, dtype=torch.long)
            noisy_latent = self.scheduler.add_noise(latent, noise, t)

            spacing_tensor = (
                torch.tensor(self.spacing, device=device, dtype=torch.float16)
                .unsqueeze(0)
                .expand(B, -1)
                * 1e2
            )

            modality_tensor = torch.zeros(B, device=device, dtype=torch.long)

            # ---- diffusion UNet forward + 返回特征 ----
            _, feats = self.unet(
                x=noisy_latent,
                timesteps=t,
                spacing_tensor=spacing_tensor,
                class_labels=modality_tensor,
                return_features=True,
                feature_positions=self.feature_positions,
            )

        return {
            "diffusion_coarse": feats["coarse"],
            "diffusion_fine": feats["fine"],
        }

    # ---------------------------------------------------
    # 5️⃣ 加载 MAISI
    # ---------------------------------------------------

    def load_maisi_models(self, ae_weights_path, unet_weights_path):

        print("\n=== Loading diffusion feature extractor ===")

        # ---------- AE ----------
        ae_def = {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "num_channels": [64, 128, 256],
            "num_res_blocks": [2, 2, 2],
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": True,
            "num_splits": 4,
            "dim_split": 1,
        }

        ae_model = AutoencoderKlMaisi(**ae_def)
        ae_model.load_state_dict(torch.load(ae_weights_path, weights_only=False))
        ae_model.eval()
        print("✔ AE loaded")

        # ---------- UNet ----------
        unet_def = {
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
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
            "include_fc": True,
        }

        unet = DiffusionModelUNetMaisiWithFeatures(**unet_def)
        ckpt = torch.load(unet_weights_path, weights_only=False)
        unet.load_state_dict(ckpt["unet_state_dict"])
        unet.eval()
        print("✔ Diffusion UNet loaded")

        scale_factor = ckpt["scale_factor"]

        # ---------- Scheduler ----------
        scheduler_def = {
            "num_train_timesteps": 1000,
            "use_discrete_timesteps": False,
            "use_timestep_transform": True,
            "sample_method": "uniform",
            "scale": 1.4,
        }

        scheduler = RFlowScheduler(**scheduler_def)

        return ae_model, unet, scheduler, scale_factor

    # ---------------------------------------------------
    # 6️⃣ 禁用优化器
    # ---------------------------------------------------

    def configure_optimizers(self):
        raise RuntimeError(
            "This model is inference-only. "
            "Optimizer is not supported."
        )