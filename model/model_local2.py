# models/model_correspondence_3d2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import ModelBase

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
# 2) Encoder3D: stride=2/4/8 + skip features
# ---------------------------

class Encoder3D(nn.Module):
    """
    共享 Encoder：用于 A/B。
    backbone_stride = 2: 32³ -> 16³
    backbone_stride = 4: 32³ -> 8³
    backbone_stride = 8: 32³ -> 4³
    返回：
      feat: (B, C, D, H, W)
      skips: list of tensors，按分辨率从高到低（stem, down1, down2）
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=32,
                 out_channels=128,
                 num_res_units=2,
                 norm="instance",
                 act="leakyrelu",
                 backbone_stride=4):
        super().__init__()
        if backbone_stride not in (2, 4, 8):
            raise ValueError("backbone_stride must be 2 or 4 or 8")
        self.backbone_stride = backbone_stride

        # stem: 32 -> 32
        self.stem = nn.Sequential(
            ConvNormAct3d(in_channels, base_channels, norm=norm, act=act),
            *[ResBlock3d(base_channels, norm=norm, act=act) for _ in range(num_res_units)]
        )

        # down1: /2 (32->16)
        self.down1 = nn.Sequential(
            ConvNormAct3d(base_channels, base_channels * 2, s=2, norm=norm, act=act),
            *[ResBlock3d(base_channels * 2, norm=norm, act=act) for _ in range(num_res_units)]
        )

        # down2: /4 (32->8) if needed
        self.down2 = None
        self.down3 = None
        ch_after = base_channels * 2

        if backbone_stride >= 4:
            self.down2 = nn.Sequential(
                ConvNormAct3d(base_channels * 2, base_channels * 4, s=2, norm=norm, act=act),
                *[ResBlock3d(base_channels * 4, norm=norm, act=act) for _ in range(num_res_units)]
            )
            ch_after = base_channels * 4

        # down3: /8 (32->4) if needed
        if backbone_stride >= 8:
            self.down3 = nn.Sequential(
                ConvNormAct3d(base_channels * 4, base_channels * 8, s=2, norm=norm, act=act),
                *[ResBlock3d(base_channels * 8, norm=norm, act=act) for _ in range(num_res_units)]
            )
            ch_after = base_channels * 8

        self.proj = nn.Conv3d(ch_after, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        skips = []
        x0 = self.stem(x)   # 32^3
        skips.append(x0)

        x1 = self.down1(x0) # 16^3
        skips.append(x1)

        x = x1
        if self.down2 is not None:
            x2 = self.down2(x1)  # 8^3
            skips.append(x2)
            x = x2

        if self.down3 is not None:
            x3 = self.down3(x2)  # 4^3
            skips.append(x3)
            x = x3

        feat = self.proj(x)
        return feat, skips


# ---------------------------
# 3) 对称 Decoder3D：把 transformer 后的低分辨率特征上采样回更高分辨率
# ---------------------------

class Decoder3D(nn.Module):
    """
    对称解码器：按 backbone_stride 逐级上采样，并与 encoder skip 融合。
    只注册实际会用到的 skip_proj，避免 DDP unused params。
    """
    def __init__(self,
                 feat_dim=128,
                 base_channels=32,
                 num_res_units=1,
                 norm="instance",
                 act="leakyrelu",
                 backbone_stride=4):
        super().__init__()
        if backbone_stride not in (2, 4, 8):
            raise ValueError("backbone_stride must be 2 or 4 or 8")
        self.backbone_stride = backbone_stride
        self.num_ups = {2: 1, 4: 2, 8: 3}[backbone_stride]

        # upsample blocks
        self.up = nn.ModuleList([
            nn.ConvTranspose3d(feat_dim, feat_dim, kernel_size=2, stride=2)
            for _ in range(self.num_ups)
        ])

        # ✅ 关键：只为会用到的 skip 分辨率创建 proj
        # stride=2: fuse [stem]                 -> channels = base
        # stride=4: fuse [down1, stem]          -> channels = 2base, base
        # stride=8: fuse [down2, down1, stem]   -> channels = 4base, 2base, base
        if backbone_stride == 2:
            fuse_inchs = [base_channels]
        elif backbone_stride == 4:
            fuse_inchs = [base_channels * 2, base_channels]
        else:  # 8
            fuse_inchs = [base_channels * 4, base_channels * 2, base_channels]

        self.skip_proj = nn.ModuleList([
            nn.Conv3d(in_ch, feat_dim, kernel_size=1, bias=True)
            for in_ch in fuse_inchs
        ])

        self.refine = nn.ModuleList([
            nn.Sequential(*[ResBlock3d(feat_dim, norm=norm, act=act) for _ in range(num_res_units)])
            for _ in range(self.num_ups)
        ])

    def forward(self, x_feat: torch.Tensor, skips: list):
        # 取 fuse 的 skip（由高到低的 skips: [stem, down1, down2, down3?]）
        if self.backbone_stride == 2:
            fuse_skips = [skips[0]]                 # stem
        elif self.backbone_stride == 4:
            fuse_skips = [skips[1], skips[0]]       # down1, stem
        else:  # 8
            fuse_skips = [skips[2], skips[1], skips[0]]  # down2, down1, stem

        for i in range(self.num_ups):
            x_feat = self.up[i](x_feat)
            x_feat = x_feat + self.skip_proj[i](fuse_skips[i])
            x_feat = self.refine[i](x_feat)

        return x_feat


# ---------------------------
# 4) 3D sin-cos 位置编码
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
# 5) LoFTR-style Block: SA + CA + FFN
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
# 6) Pair 模型：Encoder -> Transformer -> Decoder
# ---------------------------

class CorrespondenceLoFTR3D(nn.Module):
    """
    输入:
      img_A, img_B: (B, 1, D, H, W) 例如 32³ patch
    输出:
      feat_A_out, feat_B_out: (B, C, D, H, W) 现在是更高分辨率（默认回到输入分辨率）
    """
    def __init__(self,
                 in_channels=1,
                 base_channels=32,
                 feat_dim=120,              # 建议 6 的倍数
                 num_res_units=2,
                 backbone_stride=4,         # 2 / 4 / 8
                 transf_layers=4,
                 transf_heads=4,
                 transf_dropout=0.0,
                 decoder_res_units=1):
        super().__init__()
        self.encoder = Encoder3D(
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

        self.decoder = Decoder3D(
            feat_dim=feat_dim,
            base_channels=base_channels,
            num_res_units=decoder_res_units,
            backbone_stride=backbone_stride
        )

    def forward(self, img_A, img_B):
        # 1) encode + skips
        feat_A, skips_A = self.encoder(img_A)  # (B, C, d, h, w)
        feat_B, skips_B = self.encoder(img_B)

        B, C, d, h, w = feat_A.shape

        # 2) flatten tokens
        tok_A = feat_A.flatten(2).transpose(1, 2)  # (B, N, C)
        tok_B = feat_B.flatten(2).transpose(1, 2)

        # 3) pos embed（在低分辨率 token 空间）
        pos = build_sincos_3d_pos_embed(d, h, w, C, device=feat_A.device)  # (N, C)
        tok_A = tok_A + pos[None, :, :]
        tok_B = tok_B + pos[None, :, :]

        # 4) transformer
        tok_A, tok_B = self.transformer(tok_A, tok_B)

        # 5) reshape back to low-res feat
        feat_A_tr = tok_A.transpose(1, 2).reshape(B, C, d, h, w)
        feat_B_tr = tok_B.transpose(1, 2).reshape(B, C, d, h, w)

        # 6) decode to higher resolution (symmetric)
        feat_A_out = self.decoder(feat_A_tr, skips_A)
        feat_B_out = self.decoder(feat_B_tr, skips_B)

        # 7) normalize along channel
        feat_A_out = F.normalize(feat_A_out, p=2, dim=1)
        feat_B_out = F.normalize(feat_B_out, p=2, dim=1)
        return feat_A_out, feat_B_out


# ---------------------------
# 7) 框架封装：ModelBase 子类
# ---------------------------

class model_local2(ModelBase):
    """
    LoFTR3D + 对称解码器：输出更高分辨率特征
    """
    def __init__(self, config: dict):
        self._load_configs(config)
        self.setup_params()
        super().__init__(config)
        self.setup_network()

    def _get_model_name(self):
        return "model_local2"

    def setup_params(self):
        mc = self.model_config
        rc = self.run_config

        self.in_channels = mc.get('in_channels', 1)
        self.base_channels = mc.get('base_channels', 32)
        self.target_feature_size = mc.get('target_feature_size', 120)  # 建议 6 的倍数
        self.num_res_units = mc.get('num_res_units', 2)

        # ✅ Encoder stride 支持 2/4/8
        self.backbone_stride = mc.get('backbone_stride', 4)

        # Transformer
        self.transf_layers = mc.get('transf_layers', 2)
        self.transf_heads = mc.get('transf_heads', 4)
        self.transf_dropout = mc.get('transf_dropout', 0.0)

        # Decoder
        self.decoder_res_units = mc.get('decoder_res_units', 1)

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
            transf_dropout=self.transf_dropout,
            decoder_res_units=self.decoder_res_units
        )

    def forward(self, img_A: torch.Tensor, img_B: torch.Tensor):
        """
        img_A, img_B: (B, 1, D, H, W) 例如 32³ patch
        return:
          feat_A_out, feat_B_out: (B, C, D, H, W)  # ✅ 已通过解码器回到更高分辨率
        """
        return self.network(img_A, img_B)

    @torch.no_grad()
    def inference(self, img_A: torch.Tensor, img_B: torch.Tensor):
        """
        保持接口正确：inference 必须给 A/B 两张
        """
        feat_A_out, feat_B_out = self.network(img_A, img_B)
        return {"feat_A": feat_A_out, "feat_B": feat_B_out}

    def configure_optimizers(self):
        optimizer_cfg = self.model_cfg_section.get('Optimizer')
        scheduler_cfg = self.model_cfg_section.get('LRScheduler')

        if not optimizer_cfg:
            raise KeyError("[Model.Optimizer] section is missing from the config.")

        optimizer_name = optimizer_cfg.get('name', 'Adam').lower()
        lr = optimizer_cfg.get('learning_rate', 1e-4)
        wd = optimizer_cfg.get('weight_decay', 1e-5)

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            raise NotImplementedError(f"Optimizer '{optimizer_name}' is not supported.")

        if not scheduler_cfg:
            print("Warning: [Model.LRScheduler] not found. No scheduler will be used.")
            return optimizer, None

        scheduler_name = scheduler_cfg.get('name', 'CosineAnnealingLR').lower()
        if scheduler_name == 'cosineannealinglr':
            t_max_from_config = scheduler_cfg.get('T_max', 0)
            max_epochs = self.run_config.get('max_epochs')
            if max_epochs is None:
                raise KeyError("`max_epochs` must be defined in [Run] to use CosineAnnealingLR.")

            final_t_max = max_epochs if (t_max_from_config == 0) else t_max_from_config
            if t_max_from_config == 0:
                print(f"INFO: Scheduler T_max not set. Use max_epochs={final_t_max}.")
            else:
                print(f"INFO: Scheduler T_max={final_t_max}.")

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_t_max)
        else:
            raise NotImplementedError(f"Scheduler '{scheduler_name}' is not supported.")

        print(f"Optimizer '{optimizer_name}' and Scheduler '{scheduler_name}' configured.")
        return optimizer, scheduler