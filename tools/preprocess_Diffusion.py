# ==========================================
# correspondence_debug.py  (Part 1 / 4)
# ==========================================

import os
import sys
import json
import math
import torch
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from typing import List, Dict, Mapping, Hashable

import torch.nn.functional as F
from monai.transforms import MapTransform, Compose, ToTensord, LoadImaged
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.bundle import ConfigParser

# 固定设备
DEVICE = torch.device("cuda:3")

# ---------------------------------------------------------
# LoadPreprocessed (⚠ 必须使用你提供的版本，已保持完全一致)
# ---------------------------------------------------------

class LoadPreprocessed(MapTransform):
    """
    Load DeepLesion preprocessed data: image.npy + meta.pkl
    Keep mmap_mode reading to reduce memory.
    """
    def __init__(self, keys):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        d = dict(data)
        basename = os.path.basename(d[self.first_key(d)])
        basename = basename.split('.')[0]

        # load meta .pkl
        meta_path = os.path.join(os.path.dirname(d[self.first_key(d)]), basename + '.pkl')
        with open(meta_path, 'rb') as f:
            d['image_meta_dict'] = pickle.load(f)

        # load each image array (mmap)
        for key in self.keys:
            d[key] = np.load(d[key], mmap_mode='r')

        return d

# ---------------------------------------------------------
# LoadPointCloudTxt: 读取 landmark_1, landmark_2 (txt)
# ---------------------------------------------------------

class LoadPointCloudTxt(MapTransform):
    """
    Loads Nx3 landmarks from txt file containing:
    x y z
    x y z
    ...
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            pts = np.loadtxt(d[key]).astype(np.float32)
            if pts.ndim == 1:
                pts = pts.reshape(1, 3)
            d[key] = pts
        return d

# ---------------------------------------------------------
# Simple JSON Dataset Loader
# ---------------------------------------------------------

class DeepLesionDataset:
    """
    Lightweight loader for JSON list:
    {
        "test": [
            {
                "image_1": "/path/caseXX/image_1.npy",
                "image_2": "/path/caseXX/image_2.npy",
                "landmark_1": "/path/caseXX/landmark_1.txt",
                "landmark_2": "/path/caseXX/landmark_2.txt"
            },
            ...
        ]
    }
    """

    def __init__(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.test_list = data.get("test", [])
        if len(self.test_list) == 0:
            print("⚠ Warning: test list is empty")

        # build transforms
        self.transform = Compose([
            LoadPreprocessed(keys=["image_1", "image_2"]),
            LoadPointCloudTxt(keys=["landmark_1", "landmark_2"]),
            ToTensord(keys=["image_1", "image_2"], allow_missing_keys=True)
        ])

    def __len__(self):
        return len(self.test_list)

    def __getitem__(self, idx):
        item = self.test_list[idx]
        return self.transform(item)
    

# ==========================================
# correspondence_debug.py  (Part 2 / 4)
# ==========================================

def define_instance(args, key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(key, instantiate=True)

# ---------------------------------------------------------
# 模型加载（沿用你原来的结构）
# ---------------------------------------------------------

def load_models(
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

    ae_model = AutoencoderKlMaisi(**ae_def).to(DEVICE)
    ae_state_dict = torch.load(ae_weights_path, map_location=DEVICE, weights_only=False)
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

    unet = define_instance(args, "diffusion_unet_def").to(DEVICE)
    unet_ckpt = torch.load(unet_weights_path, map_location=DEVICE, weights_only=False)
    unet.load_state_dict(unet_ckpt["unet_state_dict"])
    unet.eval()
    print("✔ U-Net 加载成功")

    # scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")

    return ae_model, unet, noise_scheduler, args


# ---------------------------------------------------------
# 预处理 image（numpy/mmap → torch tensor）
# ---------------------------------------------------------

def preprocess_image(image_np, landmarks_np=None):
    """
    统一调整图像到 MAISI 要求的 64 倍数尺寸。
    若给出 landmarks，则一起调整坐标。
    返回:
        img_tensor: [1,1,D,H,W]
        adj_landmarks: Nx3 或 None
    """
    if landmarks_np is None:
        # 生成空 landmarks，避免分支重复
        _, landmarks_np = adjust_to_maisi_size(image_np, np.zeros((0,3)))
        img_np, _ = adjust_to_maisi_size(image_np, np.zeros((0,3)))
        adj_landmarks = None
    else:
        img_np, adj_landmarks = adjust_to_maisi_size(image_np, landmarks_np)

    img = torch.from_numpy(img_np.astype(np.float32))[None,None].to(DEVICE)

    return img, adj_landmarks


# ---------------------------------------------------------
# AE 编码：得到 latent
# ---------------------------------------------------------

def get_latent_representation(image_tensor, ae_model):
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z_mu, z_sigma = ae_model.encode(image_tensor)
            latent = ae_model.sampling(z_mu, z_sigma)
    return latent


# ---------------------------------------------------------
# Hook 机制（multi-level DIFT）
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# DIFT 特征提取（复用你的 extract_dift_features）
# ---------------------------------------------------------

def extract_dift_features(latent, unet, scheduler, t, args, target_spacing):
    """
    latent: [1,4,D,H,W]
    t: timestep
    return: dict { layer_name: tensor }
    """
    FEATURE_STORE.clear()

    noise = torch.randn_like(latent)
    noisy_latent = scheduler.add_noise(
        original_samples=latent,
        noise=noise,
        timesteps=torch.tensor([t], device=DEVICE)
    )

    spacing_tensor = torch.tensor(target_spacing, device=DEVICE, dtype=torch.half).unsqueeze(0) * 1e2
    modality_tensor = torch.tensor([0], device=DEVICE, dtype=torch.long)

    inputs = {
        "x": noisy_latent,
        "timesteps": torch.tensor([t], device=DEVICE),
        "spacing_tensor": spacing_tensor,
        "class_labels": modality_tensor
    }

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            _ = unet(**inputs)

    return FEATURE_STORE.copy()

def nearest_64_multiple(n):
    """返回离 n 最近但 <= n 的 64 整数倍；最小为 64"""
    return max(64, (n // 64) * 64)


def adjust_to_maisi_size(img_np, landmarks):
    """
    输入：
        img_np: [D,H,W] numpy array
        landmarks: Nx3 numpy array
    输出：
        new_img, new_landmarks
    """
    D, H, W = img_np.shape
    new_sizes = [
        nearest_64_multiple(D),
        nearest_64_multiple(H),
        nearest_64_multiple(W)
    ]

    img = img_np.numpy().copy()
    lm = landmarks.copy()

    # ===== 每个轴分别处理 =====
    for axis, (orig, target) in enumerate(zip([D, H, W], new_sizes)):

        if orig == target:
            continue

        # ---- 1) 过大 → 居中裁剪 ----
        if orig > target:
            start = (orig - target) // 2
            end = start + target

            if axis == 0:
                img = img[start:end, :, :]
                lm[:, 0] -= start
            elif axis == 1:
                img = img[:, start:end, :]
                lm[:, 1] -= start
            elif axis == 2:
                img = img[:, :, start:end]
                lm[:, 2] -= start

        # ---- 2) 太小 → 居中 padding ----
        else:
            pad_total = target - orig
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before

            if axis == 0:
                img = np.pad(img, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
                lm[:, 0] += pad_before
            elif axis == 1:
                img = np.pad(img, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
                lm[:, 1] += pad_before
            elif axis == 2:
                img = np.pad(img, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
                lm[:, 2] += pad_before

    return img, lm

# ==========================================
# correspondence_debug.py  (Part 3 / 4)
# ==========================================

# ---------------------------------------------------------
# 多层特征匹配 — coarse + fine (cos similarity)
# ---------------------------------------------------------

def compute_feature_downsample(img_shape_xyz, feat_shape):
    """
    img_shape_xyz: 原图大小 (X,Y,Z)
    feat_shape: 特征图 (C, D, H, W)
    return: downsample factors (fx, fy, fz)
    """
    feat_shape_xyz = (feat_shape[3], feat_shape[2], feat_shape[1])
    return [img / feat for img, feat in zip(img_shape_xyz, feat_shape_xyz)]


def map_point_to_feature_grid(point_xyz, downsample_factors):
    """
    point_xyz: 原图坐标 (x,y,z)
    downsample_factors: fx, fy, fz
    return: feature map index (x_f, y_f, z_f)
    """
    return [
        int(math.floor(coord / f))
        for coord, f in zip(point_xyz, downsample_factors)
    ]


def compute_cosine_similarity_map(source_vec, target_feat):
    """
    source_vec: [C]
    target_feat: [C, D, H, W]
    return: similarity_map [D, H, W]
    """
    C, D, H, W = target_feat.shape
    target_flat = target_feat.reshape(C, -1)
    sim_flat = F.cosine_similarity(source_vec.unsqueeze(1), target_flat, dim=0)
    return sim_flat.reshape(D, H, W)


# ---------------------------------------------------------
# 单点匹配：源点 → 目标点
# ---------------------------------------------------------

def find_corresponding_point(
    source_point_xyz,
    source_features,
    target_features,
    img_shape_xyz,
    coarse_weight=0.0,
    fine_weight=1.0
):
    """
    source_point_xyz: (x, y, z)
    source_features/target_features:
        {"fine": tensor[C,D,H,W], "coarse": tensor[C,D,H,W]}
    """

    # ------------------------------
    # 1. 提取 fine 特征
    # ------------------------------
    fine_src = source_features["fine"].squeeze(0)    # C,D,H,W
    fine_tgt = target_features["fine"].squeeze(0)

    VAE_DOWNSAMPLE = 4

    px_f = math.floor(source_point_xyz[0] / VAE_DOWNSAMPLE)
    py_f = math.floor(source_point_xyz[1] / VAE_DOWNSAMPLE)
    pz_f = math.floor(source_point_xyz[2] / VAE_DOWNSAMPLE)
    source_vec_fine = fine_src[:, px_f, py_f, pz_f]

    sim_fine = compute_cosine_similarity_map(source_vec_fine, fine_tgt)

    # ------------------------------
    # 2. 提取 coarse 特征
    # ------------------------------
    coarse_src = source_features["coarse"].squeeze(0)
    coarse_tgt = target_features["coarse"].squeeze(0)

    VAE_DOWNSAMPLE = 16

    px_c = math.floor(source_point_xyz[0] / VAE_DOWNSAMPLE)
    py_c = math.floor(source_point_xyz[1] / VAE_DOWNSAMPLE)
    pz_c = math.floor(source_point_xyz[2] / VAE_DOWNSAMPLE)
    source_vec_coarse = coarse_src[:, px_c, py_c, pz_c]

    sim_coarse = compute_cosine_similarity_map(source_vec_coarse, coarse_tgt)

    # coarse 上采样到 fine 尺寸
    coarse_up = F.interpolate(
        sim_coarse.unsqueeze(0).unsqueeze(0),
        size=sim_fine.shape,
        mode="trilinear",
        align_corners=False
    ).squeeze(0).squeeze(0)

    # ------------------------------
    # 3. 融合
    # ------------------------------
    combined = coarse_weight * coarse_up + fine_weight * sim_fine

    # ------------------------------
    # 4. 取最大值位置
    # ------------------------------
    best_index = torch.argmax(combined)
    x_f, y_f, z_f = np.unravel_index(best_index.cpu().numpy(), combined.shape)

    # ------------------------------
    # 5. 反映射回原始坐标
    # ------------------------------
    VAE_DOWNSAMPLE = 4

    tgt_xyz = [
        round((x_f+0.5) * VAE_DOWNSAMPLE),
        round((y_f+0.5) * VAE_DOWNSAMPLE),
        round((z_f+0.5) * VAE_DOWNSAMPLE)
    ]

    return tuple(tgt_xyz), combined


# ---------------------------------------------------------
# landmark 批量对应：landmark_1 → landmark_pred
# ---------------------------------------------------------

def match_landmarks_for_case(
    img_1,                 # tensor [1,1,D,H,W]
    img_2,                 # tensor [1,1,D,H,W]
    landmarks_1_adj,       # Nx3 (已调整)
    ae_model,
    unet,
    scheduler,
    args,
    layers_to_hook,
    timestep,
    target_spacing=[1.5,1.5,1.5],
    coarse_weight=0.0,
    fine_weight=1.0
):
    """
    返回：
        pred_landmarks: [N,3]
    """

    # ------------------------------
    # 1. 注册 hooks
    # ------------------------------
    register_hooks(unet, layers_to_hook)

    # ------------------------------
    # 2. AE latent
    # ------------------------------
    latent_1 = get_latent_representation(img_1, ae_model)
    latent_2 = get_latent_representation(img_2, ae_model)

    # ------------------------------
    # 3. DIFT features
    # ------------------------------
    print("  → 提取 source 特征...")
    feat_1 = extract_dift_features(latent_1, unet, scheduler, timestep, args, target_spacing)

    print("  → 提取 target 特征...")
    feat_2 = extract_dift_features(latent_2, unet, scheduler, timestep, args, target_spacing)

    # ------------------------------
    # 4. 执行 landmark-level 对应点搜索
    # ------------------------------
    pred_landmarks = []
    img_shape_xyz = img_1.shape[2:]               # (D,H,W)

    print(f"  → 共 {len(landmarks_1_adj)} 个点需要匹配")

    for i, p in enumerate(landmarks_1_adj):
        x, y, z = map(float, p)

        tgt_xyz, _ = find_corresponding_point(
            (x, y, z),
            feat_1,
            feat_2,
            img_shape_xyz,
            coarse_weight=coarse_weight,
            fine_weight=fine_weight
        )

        pred_landmarks.append(tgt_xyz)
        print(f"    点 #{i}: {p} → {tgt_xyz}")

    return np.array(pred_landmarks, dtype=np.float32)


# ==========================================
# correspondence_debug.py  (Part 4 / 4)
# ==========================================

# ---------------------------------------------------------
# 可视化: 保存 NIfTI 图像 + landmark voxel map
# ---------------------------------------------------------

def save_nifti(img_np, path):
    """保存 [D,H,W] numpy 到 NIfTI"""
    nib.save(nib.Nifti1Image(img_np.astype(np.float32), np.eye(4)), path)


def save_landmark_nifti(landmarks, img_shape, path, cube_size=2):
    """
    landmarks: [N,3]
    img_shape: (D,H,W)
    """
    label = np.zeros(img_shape, dtype=np.uint8)

    for p in landmarks:
        x, y, z = [int(v) for v in p]
        for ix in range(max(0, x-cube_size), min(img_shape[0], x+cube_size+1)):
            for iy in range(max(0, y-cube_size), min(img_shape[1], y+cube_size+1)):
                for iz in range(max(0, z-cube_size), min(img_shape[2], z+cube_size+1)):
                    label[ix, iy, iz] = 1

    nib.save(nib.Nifti1Image(label, np.eye(4)), path)


# ---------------------------------------------------------
# 主程序 main()
# ---------------------------------------------------------
# 数据路径（你已有的 output_folder）
OUTPUT_FOLDER = "/home/mingrui/disk1/dataset/Flare2023_15mm"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

TEACHER_FEAT_ROOT = os.path.join(OUTPUT_FOLDER, "maisi_teacher_feats")
FINE_DIR   = os.path.join(TEACHER_FEAT_ROOT, "fine")
COARSE_DIR = os.path.join(TEACHER_FEAT_ROOT, "coarse")
META_DIR   = os.path.join(TEACHER_FEAT_ROOT, "meta")
os.makedirs(FINE_DIR, exist_ok=True)
os.makedirs(COARSE_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def main():

    # ===============================================
    # 用户需要填的 3 个路径（与你原程序一致）
    # ===============================================
    AE_WEIGHTS   = "/home/mingrui/disk1/projects/20250620_MAISI/maisi/models/autoencoder_epoch273.pt"
    UNET_WEIGHTS = "/home/mingrui/disk1/projects/20250620_MAISI/maisi/models/diff_unet_3d_rflow.pt"
    CONFIG_JSON  = "/home/mingrui/disk1/projects/20251103_DiffusionCorr/diffusioncorr/configs/config_maisi3d-rflow.json"


    OUTPUT_ROOT = "/home/mingrui/disk1/projects/20250620_MAISI/maisi/my_correspondence/debug_data/DeepLesion_results"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # MAISI DIFT settings
    DIFF_TIMESTEP = 50
    TARGET_SPACING = [1.5, 1.5, 1.5]

    LAYERS_TO_HOOK = [
        {"name": "coarse", "layer_path": "up_blocks.1.resnets.0"},
        {"name": "fine",   "layer_path": "up_blocks.3.resnets.0"}
    ]

    COARSE_WEIGHT = 1.0
    FINE_WEIGHT   = 1.0

    # ===============================================
    # 1) 加载模型
    # ===============================================

    ae_model, unet, scheduler, args = load_models(
        AE_WEIGHTS,
        UNET_WEIGHTS,
        CONFIG_JSON
    )

    # ===============================================
    # 2) 加载 DeepLesion JSON 数据集
    # ===============================================

    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".nii.gz") or f.endswith(".nii")])
    print(f"\nFound {len(image_files)} images in: {IMAGE_FOLDER}")

    for fname in tqdm(image_files, desc="Precomputing teacher features"):
        stem = fname.replace(".nii.gz", "").replace(".nii", "")

        fine_out   = os.path.join(FINE_DIR,   stem + ".npy")
        coarse_out = os.path.join(COARSE_DIR, stem + ".npy")
        meta_out   = os.path.join(META_DIR,   stem + ".json")

        if os.path.exists(fine_out) and os.path.exists(coarse_out) and os.path.exists(meta_out):
            continue

        path = os.path.join(IMAGE_FOLDER, fname)

        # 读入 float32，后面 autocast 到 fp16
        img_np = nib.load(path).get_fdata(dtype=np.float32)         # [D,H,W]
        img = torch.from_numpy(img_np)[None, None].to(DEVICE)       # [1,1,D,H,W]

        latent = get_latent_representation(img, ae_model)

        pause = 1


    # ===============================================
    # 3) 遍历每一个 case 做 landmark 对应点预测
    # ===============================================
    tre_records = []

    for idx in range(len(dataset)):


        # ===============================================
        # landmark 对应点预测（核心 DIFT+MAISI 部分）
        # ===============================================
        print("→ 开始 landmark 对应点匹配...")

        pred_landmarks = match_landmarks_for_case(
            img_1=img_1,
            img_2=img_2,
            landmarks_1_adj=landmarks_1_adj,
            ae_model=ae_model,
            unet=unet,
            scheduler=scheduler,
            args=args,
            layers_to_hook=LAYERS_TO_HOOK,
            timestep=DIFF_TIMESTEP,
            target_spacing=TARGET_SPACING,
            coarse_weight=COARSE_WEIGHT,
            fine_weight=FINE_WEIGHT
        )

        for pid in range(len(pred_landmarks)):
            gt = landmarks_2_adj[pid]   # 目标点（处理域）
            pred = pred_landmarks[pid]

            tre = np.linalg.norm(pred - gt)

            tre_records.append({
                "case": case_name,
                "point_id": pid,
                "gt_x": gt[0], "gt_y": gt[1], "gt_z": gt[2],
                "pred_x": pred[0], "pred_y": pred[1], "pred_z": pred[2],
                "tre": tre
            })

        if idx < 0:
            # 保存预测 landmark
            print("→ 保存预测 landmark")
            save_landmark_nifti(pred_landmarks, img_shape, os.path.join(case_dir, "landmark_pred.nii.gz"))

            # 也保存数值
            np.savetxt(os.path.join(case_dir, "landmark_pred.txt"), pred_landmarks)

            print(f"★ Case {case_name} 完成！输出保存在 {case_dir}")

        if idx >= 9:
            print("⚠ 仅测试前 100 个 case，跳出循环。")
            break

    # =============================
    # 输出所有 CASE 的 TRE 结果
    # =============================
    df = pd.DataFrame(tre_records)
    excel_path = os.path.join(OUTPUT_ROOT, "TRE_results.xlsx")
    df.to_excel(excel_path, index=False)

    print("\n>>> TRE 结果已保存：", excel_path)
    print(df.describe())

    print("\n==============================================")
    print("全部测试样本处理完成！")
    print(f"输出目录：{OUTPUT_ROOT}")
    print("==============================================\n")


# ---------------------------------------------------------
# 程序入口
# ---------------------------------------------------------

if __name__ == "__main__":
    main()




# # =========================
# # ====== 主流程 ==========
# # =========================
# @torch.no_grad()
# def main():
#     ae, unet, scheduler = load_models()

#     image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".nii.gz") or f.endswith(".nii")])
#     print(f"\nFound {len(image_files)} images in: {IMAGE_FOLDER}")

#     for fname in tqdm(image_files, desc="Precomputing teacher features"):
#         stem = fname.replace(".nii.gz", "").replace(".nii", "")

#         fine_out   = os.path.join(FINE_DIR,   stem + ".npy")
#         coarse_out = os.path.join(COARSE_DIR, stem + ".npy")
#         meta_out   = os.path.join(META_DIR,   stem + ".json")

#         if os.path.exists(fine_out) and os.path.exists(coarse_out) and os.path.exists(meta_out):
#             continue

#         path = os.path.join(IMAGE_FOLDER, fname)

#         # 读入 float32，后面 autocast 到 fp16
#         img_np = nib.load(path).get_fdata(dtype=np.float32)         # [D,H,W]
#         img = torch.from_numpy(img_np)[None, None].to(DEVICE)       # [1,1,D,H,W]

#         FEATURE_STORE.clear()

#         with torch.autocast(device_type="cuda" if DEVICE.type == "cuda" else "cpu", dtype=AUTOCAST_DTYPE):
#             # --- AE encode ---
#             z_mu, z_sigma = ae.encode(img)
#             latent = ae.sampling(z_mu, z_sigma)

#             # --- add noise at timestep ---
#             noise = fixed_noise_like(latent, stem)
#             t = torch.full((1,), DIFF_TIMESTEP, device=DEVICE, dtype=torch.long)

#             noisy_latent = scheduler.add_noise(
#                 original_samples=latent,
#                 noise=noise,
#                 timesteps=t,
#             )

#             # --- UNet forward (hook 捕获中间特征) ---
#             spacing_tensor = torch.tensor(SPACING, device=DEVICE, dtype=torch.float16).unsqueeze(0) * 1e2
#             modality_tensor = torch.tensor([MODALITY_LABEL], device=DEVICE, dtype=torch.long)

#             _ = unet(
#                 x=noisy_latent,
#                 timesteps=t,
#                 spacing_tensor=spacing_tensor,
#                 class_labels=modality_tensor,
#             )

#         # hook 校验
#         if "fine" not in FEATURE_STORE or "coarse" not in FEATURE_STORE:
#             keys = list(FEATURE_STORE.keys())
#             FEATURE_STORE.clear()
#             raise RuntimeError(f"Hook missing. Got keys={keys}. Check LAYERS_TO_HOOK paths.")

#         fine_feat   = FEATURE_STORE["fine"].detach().cpu().numpy().astype(SAVE_DTYPE_NP)
#         coarse_feat = FEATURE_STORE["coarse"].detach().cpu().numpy().astype(SAVE_DTYPE_NP)
#         FEATURE_STORE.clear()

#         np.save(fine_out, fine_feat)
#         np.save(coarse_out, coarse_feat)

#         meta = {
#             "image_path": path,
#             "image_shape_DHW": list(img_np.shape),
#             "diff_timestep": int(DIFF_TIMESTEP),
#             "spacing": list(SPACING),
#             "layers_to_hook": LAYERS_TO_HOOK,
#             "fine_shape": list(fine_feat.shape),
#             "coarse_shape": list(coarse_feat.shape),
#             "dtype_saved": "float16",
#             "noise": "fixed by sha1(stem)",
#             "device": str(DEVICE),
#             "cuda_current_device": int(torch.cuda.current_device()) if DEVICE.type == "cuda" else None,
#         }
#         with open(meta_out, "w") as f:
#             json.dump(meta, f, indent=2)

#     print("✔ MAISI teacher 特征预计算完成")


# if __name__ == "__main__":
#     main()