# models/model_correspondence_3d2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase  # 你们框架里的 ModelBase

# ---------------------------
# 1) 基础 3D 卷积块
# ---------------------------

class ConvNormAct3d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm="instance", act="leakyrelu"):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        if norm == "batch":
            self.norm = nn.BatchNorm3d(out_ch)
        elif norm == "group":
            self.norm = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        else:
            self.norm = nn.InstanceNorm3d(out_ch, affine=True)

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock3d(nn.Module):
    def __init__(self, ch, norm="instance", act="leakyrelu"):
        super().__init__()
        self.c1 = ConvNormAct3d(ch, ch, k=3, s=1, p=1, norm=norm, act=act)
        self.c2 = ConvNormAct3d(ch, ch, k=3, s=1, p=1, norm=norm, act=act)

    def forward(self, x):
        return x + self.c2(self.c1(x))


# ---------------------------
# 2) 3D Backbone (stride=4 or stride=2)
# ---------------------------

class Backbone3D(nn.Module):
    """
    共享 backbone：同时用于 A/B。
    backbone_stride = 4: 32³ -> 8³
    backbone_stride = 2: 32³ -> 16³
    """
    def __init__(self, in_channels=1, base_channels=32, out_channels=128,
                 num_res_units=2, norm="instance", act="leakyrelu",
                 backbone_stride=4):
        super().__init__()
        self.backbone_stride = backbone_stride

        self.stem = nn.Sequential(
            ConvNormAct3d(in_channels, base_channels, norm=norm, act=act),
            *[ResBlock3d(base_channels, norm=norm, act=act) for _ in range(num_res_units)]
        )

        self.down1 = nn.Sequential(
            ConvNormAct3d(base_channels, base_channels * 2, s=2, norm=norm, act=act),
            *[ResBlock3d(base_channels * 2, norm=norm, act=act) for _ in range(num_res_units)]
        )

        if backbone_stride == 4:
            self.down2 = nn.Sequential(
                ConvNormAct3d(base_channels * 2, base_channels * 4, s=2, norm=norm, act=act),
                *[ResBlock3d(base_channels * 4, norm=norm, act=act) for _ in range(num_res_units)]
            )
            proj_in = base_channels * 4
        elif backbone_stride == 2:
            self.down2 = None
            self.feat = nn.Sequential(
                *[ResBlock3d(base_channels * 2, norm=norm, act=act) for _ in range(num_res_units)]
            )
            proj_in = base_channels * 2
        else:
            raise ValueError("backbone_stride must be 2 or 4")

        self.proj = nn.Conv3d(proj_in, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        if self.down2 is not None:
            x = self.down2(x)  # stride=4
        else:
            x = self.feat(x)   # stride=2
        x = self.proj(x)
        return x  # (B, C, D, H, W)


# ---------------------------
# 3) 3D sin-cos 位置编码
# ---------------------------

def build_sincos_3d_pos_embed(d, h, w, dim, device):
    assert dim % 6 == 0, "feat_dim 必须能被 6 整除（如 96/120/192）"

    def get_1d_pos(n, half_dim):
        pos = torch.arange(n, device=device).float()
        omega = torch.arange(half_dim, device=device).float() / half_dim
        omega = 1.0 / (10000 ** omega)
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    half_dim_each = dim // 6
    z_embed = get_1d_pos(d, half_dim_each)
    y_embed = get_1d_pos(h, half_dim_each)
    x_embed = get_1d_pos(w, half_dim_each)

    zz = z_embed[:, None, None, :].expand(d, h, w, -1)
    yy = y_embed[None, :, None, :].expand(d, h, w, -1)
    xx = x_embed[None, None, :, :].expand(d, h, w, -1)

    pos = torch.cat([zz, yy, xx], dim=-1)
    return pos.reshape(d * h * w, dim)  # (N, C)


# ---------------------------
# 4) LoFTR-style Block: SA + CA + FFN
# ---------------------------

class LoFTRBlock3D(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        self.self_attn_A = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_B = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_A = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_B = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1_A = nn.LayerNorm(dim); self.norm1_B = nn.LayerNorm(dim)
        self.norm2_A = nn.LayerNorm(dim); self.norm2_B = nn.LayerNorm(dim)
        self.norm3_A = nn.LayerNorm(dim); self.norm3_B = nn.LayerNorm(dim)

        self.ffn_A = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim), nn.Dropout(dropout)
        )
        self.ffn_B = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim), nn.Dropout(dropout)
        )

    def forward(self, tok_A, tok_B):
        # SA
        xA = self.norm1_A(tok_A)
        xB = self.norm1_B(tok_B)
        sa_A, _ = self.self_attn_A(xA, xA, xA)
        sa_B, _ = self.self_attn_B(xB, xB, xB)
        tok_A = tok_A + sa_A
        tok_B = tok_B + sa_B

        # CA
        xA2 = self.norm2_A(tok_A)
        xB2 = self.norm2_B(tok_B)
        ca_A, _ = self.cross_attn_A(xA2, xB2, xB2)  # A queries B
        ca_B, _ = self.cross_attn_B(xB2, xA2, xA2)  # B queries A
        tok_A = tok_A + ca_A
        tok_B = tok_B + ca_B

        # FFN
        yA = self.norm3_A(tok_A)
        yB = self.norm3_B(tok_B)
        tok_A = tok_A + self.ffn_A(yA)
        tok_B = tok_B + self.ffn_B(yB)

        return tok_A, tok_B


class LocalFeatureTransformer3D(nn.Module):
    """
    LoFTR 原文的 Local Feature Transformer（3D 版）
    """
    def __init__(self, dim=128, num_layers=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            LoFTRBlock3D(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tok_A, tok_B):
        for blk in self.layers:
            tok_A, tok_B = blk(tok_A, tok_B)
        return tok_A, tok_B


# ---------------------------
# 5) 终极 Pair 模型：吃两张图，吐两张图特征
# ---------------------------

class CorrespondenceLoFTR3D(nn.Module):
    """
    输入:
      img_A, img_B: (B, 1, 32, 32, 32)
    输出:
      feat_A_tr, feat_B_tr: (B, C, D, H, W)
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=32,
                 feat_dim=120,              # 建议 6 的倍数
                 num_res_units=2,
                 backbone_stride=4,         # 4 或 2
                 transf_layers=4,
                 transf_heads=4,
                 transf_dropout=0.0):
        super().__init__()
        self.backbone = Backbone3D(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=feat_dim,
            num_res_units=num_res_units,
            backbone_stride=backbone_stride
        )
        self.transformer = LocalFeatureTransformer3D(
            dim=feat_dim,
            num_layers=transf_layers,
            num_heads=transf_heads,
            dropout=transf_dropout
        )

    def forward(self, img_A, img_B):
        # 1) backbone
        feat_A = self.backbone(img_A)  # (B, C, D, H, W)
        feat_B = self.backbone(img_B)

        B, C, D, H, W = feat_A.shape
        N = D * H * W

        # 2) flatten to tokens
        tok_A = feat_A.flatten(2).transpose(1, 2)  # (B, N, C)
        tok_B = feat_B.flatten(2).transpose(1, 2)

        # 3) add pos embed
        pos = build_sincos_3d_pos_embed(D, H, W, C, device=feat_A.device)  # (N, C)
        tok_A = tok_A + pos[None, :, :]
        tok_B = tok_B + pos[None, :, :]

        # 4) SA+CA transformer
        tok_A, tok_B = self.transformer(tok_A, tok_B)

        # 5) reshape back
        feat_A_tr = tok_A.transpose(1, 2).reshape(B, C, D, H, W)
        feat_B_tr = tok_B.transpose(1, 2).reshape(B, C, D, H, W)

        feat_A_tr = F.normalize(feat_A_tr, p=2, dim=1)
        feat_B_tr = F.normalize(feat_B_tr, p=2, dim=1)
        return feat_A_tr, feat_B_tr

# ---------------------------
# 6) 框架封装：ModelBase 子类
# ---------------------------

class model_local(ModelBase):
    """
    对外接口和你给的 model_AE_unet 一致。
    """
    def __init__(self, config: dict):
        self._load_configs(config)
        self.setup_params()
        super().__init__(config)
        self.setup_network()

    def _get_model_name(self):
        return "model_local"

    def setup_params(self):
        mc = self.model_config
        rc = self.run_config

        self.in_channels = mc.get('in_channels', 1)
        self.base_channels = mc.get('base_channels', 32)
        self.target_feature_size = mc.get('target_feature_size', 128)
        self.num_res_units = mc.get('num_res_units', 2)

        # Transformer 相关
        self.backbone_stride = mc.get('backbone_stride', 2)
        self.transf_layers = mc.get('transf_layers', 2)
        self.transf_heads = mc.get('transf_heads', 4)
        self.transf_dropout = mc.get('transf_dropout', 0.0)

        self.max_epochs = rc.get('max_epochs', 1000)

    def setup_network(self):
        self.network = CorrespondenceLoFTR3D(
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            feat_dim=self.target_feature_size,
            num_res_units=self.num_res_units,
            backbone_stride=self.backbone_stride,
            transf_layers=self.transf_layers,
            transf_heads=self.transf_heads,
            transf_dropout=self.transf_dropout
        )

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: (B, 1, 32, 32, 32)
        return: (B, C, 8, 8, 8)
        """
        return self.network(img_A, img_B)

    def inference(self, image_tensor: torch.Tensor):
        corr_feat = self.network(image_tensor)
        return {"correspondence_output": corr_feat}

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
