import os
import json
import hashlib
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.bundle import ConfigParser
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi


# ==============================================================================
# 0. 配置区：把这里改成你的路径/参数（不使用 argparse）
# ==============================================================================

# 你的 1mm 预处理输出目录（包含 images/）
OUTPUT_FOLDER = "/mnt/nvme3n1/mingrui/dataset/FLARE23_1mm"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

# 新增缓存输出目录
CACHE_ROOT = os.path.join(OUTPUT_FOLDER, "maisi_teacher_feats")
FINE_DIR   = os.path.join(CACHE_ROOT, "fine")
COARSE_DIR = os.path.join(CACHE_ROOT, "coarse")
META_DIR   = os.path.join(CACHE_ROOT, "meta")
os.makedirs(FINE_DIR, exist_ok=True)
os.makedirs(COARSE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

# MAISI 权重与配置
AE_WEIGHTS_PATH   = "/home/mingrui/disk1/projects/20250620_MAISI/maisi/models/autoencoder_epoch273.pt"
UNET_WEIGHTS_PATH = "/home/mingrui/disk1/projects/20250620_MAISI/maisi/models/diff_unet_3d_rflow.pt"
CONFIG_JSON_PATH  = "/home/mingrui/disk1/projects/20250620_MAISI/maisi/configs/config_maisi3d-rflow.json"

# teacher 特征提取超参（必须与你训练一致）
DIFF_TIMESTEP = 50
SPACING = (1.0, 1.0, 1.0)   # 你训练里乘了 1e2
MODALITY_LABEL = 0

# pad 规则：你已确认需要能被 32 整除
PAD_MULTIPLE = 32
PAD_MODE = "replicate"  # "replicate" 或 "constant"
PAD_CONSTANT_VALUE = 0.0

# hook 层（按你 model_coarse 的版本）
LAYERS_TO_HOOK = [
    {"name": "coarse", "layer_path": "up_blocks.1.resnets.0"},
    {"name": "fine",   "layer_path": "up_blocks.3.resnets.0"},
]

# 保存 dtype
SAVE_NPY_DTYPE = np.float16

# 设备 & autocast
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
AUTOCAST_DTYPE = torch.float16

# 是否跳过已存在的缓存
SKIP_IF_EXISTS = True


# ==============================================================================
# 1. Hook 与工具函数
# ==============================================================================

FEATURE_STORE: Dict[str, torch.Tensor] = {}

def get_hook(name: str):
    def hook(module, inp, outp):
        FEATURE_STORE[name] = outp.detach()
    return hook

def register_hooks(unet: nn.Module, layers_to_hook: List[Dict[str, str]]):
    for info in layers_to_hook:
        module = unet
        for p in info["layer_path"].split("."):
            module = getattr(module, p)
        module.register_forward_hook(get_hook(info["name"]))
        print(f"✔ Hook 注册: {info['name']} -> {info['layer_path']}")

class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def define_instance(args_obj: Any, instance_def_key: str):
    parser = ConfigParser(vars(args_obj))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)

def fixed_noise_like(x: torch.Tensor, key: str) -> torch.Tensor:
    """
    兼容旧 torch：不用 randn_like(generator=...)
    """
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    seed = int(h, 16) % (2**32 - 1)
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    return torch.randn(
        x.shape,
        dtype=x.dtype,
        device=x.device,
        generator=g,
    )

def pad_to_multiple_3d(x: torch.Tensor, multiple: int, mode: str = "replicate", value: float = 0.0):
    """
    x: [B,C,D,H,W]
    pad only on the "right" side: (Wl,Wr,Hl,Hr,Dl,Dr) = (0,pw,0,ph,0,pd)
    """
    assert x.ndim == 5, f"expect 5D tensor [B,C,D,H,W], got {x.shape}"
    B, C, D, H, W = x.shape

    def need(n: int) -> int:
        r = n % multiple
        return 0 if r == 0 else (multiple - r)

    pd, ph, pw = need(D), need(H), need(W)
    pad = (0, pw, 0, ph, 0, pd)

    if pd == 0 and ph == 0 and pw == 0:
        return x, pad

    if mode == "constant":
        x_pad = F.pad(x, pad, mode=mode, value=value)
    else:
        x_pad = F.pad(x, pad, mode=mode)

    return x_pad, pad

def crop_feat_to_unpadded(feat_pad: torch.Tensor, orig_img_shape_dhw: Tuple[int, int, int], pad_img_shape_dhw: Tuple[int, int, int]):
    """
    按比例把 pad 后 feature 裁回原始区域（不依赖 stride 假设）
    feat_pad: [B,C,Df,Hf,Wf]
    """
    B, C, Df, Hf, Wf = feat_pad.shape
    D, H, W = orig_img_shape_dhw
    Dp, Hp, Wp = pad_img_shape_dhw

    # ratio crop length
    Df0 = max(1, int(round(Df * (D / Dp))))
    Hf0 = max(1, int(round(Hf * (H / Hp))))
    Wf0 = max(1, int(round(Wf * (W / Wp))))

    return feat_pad[:, :, :Df0, :Hf0, :Wf0]


# ==============================================================================
# 2. 加载 MAISI 模型
# ==============================================================================

def load_maisi_models():
    print("\n=== Loading MAISI AE + diffusion UNet + scheduler ===")

    # AE
    ae_def = {
        "spatial_dims": 3, "in_channels": 1, "out_channels": 1, "latent_channels": 4,
        "num_channels": [64, 128, 256], "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32, "norm_eps": 1e-06,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False, "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False, "use_convtranspose": False,
        "norm_float16": True,
        "num_splits": 4, "dim_split": 1,
    }
    ae = AutoencoderKlMaisi(**ae_def).to(DEVICE)
    ae_state = torch.load(AE_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    ae.load_state_dict(ae_state)
    ae.eval()

    # UNet + scheduler
    with open(CONFIG_JSON_PATH, "r") as f:
        args = ArgsNamespace(**json.load(f))

    unet = define_instance(args, "diffusion_unet_def").to(DEVICE)
    unet_ckpt = torch.load(UNET_WEIGHTS_PATH, map_location="cpu", weights_only=False)
    unet.load_state_dict(unet_ckpt["unet_state_dict"])
    unet.eval()

    scheduler = define_instance(args, "noise_scheduler")

    # hooks
    register_hooks(unet, LAYERS_TO_HOOK)

    # freeze（保险）
    for p in ae.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    return ae, unet, scheduler


# ==============================================================================
# 3. 单个病例：整幅特征缓存
# ==============================================================================

@torch.no_grad()
def compute_and_save_one(ae, unet, scheduler, image_path: str):
    fname = os.path.basename(image_path)
    stem = fname.replace(".nii.gz", "").replace(".nii", "")

    fine_out   = os.path.join(FINE_DIR, stem + ".npy")
    coarse_out = os.path.join(COARSE_DIR, stem + ".npy")
    meta_out   = os.path.join(META_DIR, stem + ".json")

    if SKIP_IF_EXISTS and os.path.exists(fine_out) and os.path.exists(coarse_out) and os.path.exists(meta_out):
        return "skip"

    # load image: [D,H,W] -> [1,1,D,H,W]
    img_np = nib.load(image_path).get_fdata(dtype=np.float32)
    img = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(DEVICE)  # float32

    orig_shape = tuple(img.shape[-3:])  # (D,H,W)

    # pad to multiple 32
    img_pad, pad_tuple = pad_to_multiple_3d(
        img,
        multiple=PAD_MULTIPLE,
        mode=PAD_MODE,
        value=PAD_CONSTANT_VALUE,
    )
    pad_shape = tuple(img_pad.shape[-3:])

    FEATURE_STORE.clear()

    # forward teacher (AE -> latent -> add_noise -> UNet)
    with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", dtype=AUTOCAST_DTYPE):
        z_mu, z_sigma = ae.encode(img_pad)
        latent = ae.sampling(z_mu, z_sigma)

        # deterministic noise
        noise = fixed_noise_like(latent, key=stem)

        t = torch.full((1,), DIFF_TIMESTEP, device=DEVICE, dtype=torch.long)
        noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=t)

        spacing_tensor = (torch.tensor(SPACING, device=DEVICE, dtype=torch.float16).unsqueeze(0) * 1e2)
        modality_tensor = torch.tensor([MODALITY_LABEL], device=DEVICE, dtype=torch.long)

        _ = unet(
            x=noisy_latent,
            timesteps=t,
            spacing_tensor=spacing_tensor,
            class_labels=modality_tensor,
        )

    # check hooks
    missing = [k for k in ("fine", "coarse") if k not in FEATURE_STORE]
    if missing:
        keys = list(FEATURE_STORE.keys())
        FEATURE_STORE.clear()
        raise RuntimeError(f"Hook missing keys {missing}. Got keys={keys}. Check LAYERS_TO_HOOK paths.")

    fine_pad = FEATURE_STORE["fine"]
    coarse_pad = FEATURE_STORE["coarse"]
    FEATURE_STORE.clear()

    # crop back to original region (recommended)
    fine = crop_feat_to_unpadded(fine_pad, orig_shape, pad_shape)
    coarse = crop_feat_to_unpadded(coarse_pad, orig_shape, pad_shape)

    # save
    np.save(fine_out, fine.squeeze(0).cpu().numpy().astype(SAVE_NPY_DTYPE))
    np.save(coarse_out, coarse.squeeze(0).cpu().numpy().astype(SAVE_NPY_DTYPE))

    meta = {
        "image_path": image_path,
        "orig_shape_DHW": list(orig_shape),
        "pad_shape_DHW": list(pad_shape),
        "pad_tuple_WHW_D": list(pad_tuple),  # (0,pw,0,ph,0,pd)
        "pad_multiple": PAD_MULTIPLE,
        "pad_mode": PAD_MODE,
        "diff_timestep": DIFF_TIMESTEP,
        "spacing": list(SPACING),
        "layers_to_hook": LAYERS_TO_HOOK,
        "fine_shape_BCDHW": list(fine.shape),
        "coarse_shape_BCDHW": list(coarse.shape),
        "saved_dtype": str(SAVE_NPY_DTYPE),
        "noise_fixed_by": "sha1(stem) -> torch.Generator",
    }
    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    return "ok"


# ==============================================================================
# 4. 主程序
# ==============================================================================

def main():
    assert os.path.isdir(IMAGE_FOLDER), f"IMAGE_FOLDER not found: {IMAGE_FOLDER}"
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".nii.gz") or f.endswith(".nii")])
    print(f"Found {len(image_files)} images in: {IMAGE_FOLDER}")
    print(f"Cache root: {CACHE_ROOT}")
    print(f"PAD_MULTIPLE={PAD_MULTIPLE}, PAD_MODE={PAD_MODE}, DIFF_TIMESTEP={DIFF_TIMESTEP}, SPACING={SPACING}")
    print(f"Device: {DEVICE}")

    ae, unet, scheduler = load_maisi_models()

    n_ok = n_skip = n_fail = 0

    for f in tqdm(image_files, desc="Caching teacher feats"):
        p = os.path.join(IMAGE_FOLDER, f)
        try:
            status = compute_and_save_one(ae, unet, scheduler, p)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
        except Exception as e:
            n_fail += 1
            print("\n[FAIL]", p)
            print("Error:", repr(e))
            print(traceback.format_exc())

    print("\n=== Done ===")
    print(f"ok: {n_ok}, skip: {n_skip}, fail: {n_fail}")
    print(f"Outputs:")
    print(f"  fine  : {FINE_DIR}")
    print(f"  coarse: {COARSE_DIR}")
    print(f"  meta  : {META_DIR}")


if __name__ == "__main__":
    main()