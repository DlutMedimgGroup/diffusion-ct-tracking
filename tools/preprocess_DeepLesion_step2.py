import os
import json
import pickle  # ==== NEW ====
from glob import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)

# ================= 配置开关 & 参数 =================

# 是否保存 .npy
SAVE_NPY = False

# 是否保存 .nii.gz（affine = identity）
SAVE_NII = False

# 是否保存 .pkl（与 .npy 同名，保存 MONAI 的 meta 信息） ==== NEW ====
SAVE_PKL = True

# 并行进程数（None 表示使用 os.cpu_count()）
NUM_WORKERS = 2

# 目标 spacing
TARGET_SPACING = (1.0, 1.0, 1.0)


# ============================================================
# 1. 解析 DLT JSON：配对 + 去掉 A->B / B->A 重复
#    注意：JSON 中的 center 是原始 voxel 坐标 (x, y, z)
# ============================================================

def parse_dlt_pairs(json_folder, nifti_root_dir):
    json_files = glob(os.path.join(json_folder, "*.json"))
    pair_list = []
    seen_pairs = set()

    print(f"正在解析 JSON 文件: {json_files}")

    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]  # train / test / valid
        with open(jf, "r") as f:
            data = json.load(f)

        for pid, ct in data.items():
            s_name = ct["source"]
            t_name = ct["target"]

            # DLT 的 center 是像素坐标 (x, y, z)
            s_center_vox = np.array(ct["source center"], dtype=np.float32)
            t_center_vox = np.array(ct["target center"], dtype=np.float32)

            s_key = tuple(np.round(s_center_vox, 3))
            t_key = tuple(np.round(t_center_vox, 3))

            s_path = os.path.join(nifti_root_dir, s_name)
            t_path = os.path.join(nifti_root_dir, t_name)

            if not (os.path.exists(s_path) and os.path.exists(t_path)):
                continue

            key = (s_name, t_name, s_key, t_key)
            rev_key = (t_name, s_name, t_key, s_key)

            # 去掉 A->B / B->A 的重复
            if key in seen_pairs or rev_key in seen_pairs:
                continue

            seen_pairs.add(key)

            pair_list.append(
                {
                    "pair_id": f"{base}_{pid}",
                    "source_image": s_path,
                    "target_image": t_path,
                    "source_voxel": [s_center_vox.tolist()],  # 统一成 N×3
                    "target_voxel": [t_center_vox.tolist()],
                }
            )

    print(f"解析完成，共得到 {len(pair_list)} 个去重后的配对。")
    return pair_list


# ============================================================
# 2. 图像预处理：只变图，不碰点
#    输出的图像 shape 为 (1, Y, X, Z)，取 [0] 后是 (Y, X, Z)
# ============================================================

def build_image_transform(target_spacing=(1.0, 1.0, 1.0)):
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image"], axcodes="LPS"),
            Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
            EnsureTyped(keys=["image"]),
        ]
    )


# ============================================================
# 3. voxel_old -> voxel_new : A_new^-1 · A_old
#    这里的 voxel 是 [x, y, z]
# ============================================================

def map_voxel_old_to_new(voxel_points, affine_old, affine_new):
    """
    world = A_old @ [x_old, y_old, z_old, 1]^T
    [x_new, y_new, z_new, 1]^T = A_new^{-1} @ world
    """
    voxel_points = np.asarray(voxel_points, dtype=np.float64).reshape(-1, 3)

    affine_old = np.asarray(affine_old, dtype=np.float64)
    affine_new = np.asarray(affine_new, dtype=np.float64)

    n = voxel_points.shape[0]
    idx_old_h = np.hstack([voxel_points, np.ones((n, 1), dtype=np.float64)])  # (N,4)
    world = (affine_old @ idx_old_h.T).T                                      # (N,4)
    A_new_inv = np.linalg.inv(affine_new)
    idx_new_h = (A_new_inv @ world.T).T                                       # (N,4)
    idx_new = idx_new_h[:, :3]
    return idx_new


# ============================================================
# 4. 单个 pair 的处理函数（供多进程调用）
# ============================================================

def process_single_pair(args):
    """
    args: (idx, total, pair, out_img_dir, out_npy_dir, out_lm_dir, target_spacing,
           save_npy, save_nii, save_pkl)  # ==== NEW ====
    """
    (
        idx,
        total,
        pair,
        out_img_dir,
        out_npy_dir,
        out_lm_dir,
        target_spacing,
        save_npy,
        save_nii,
        save_pkl,   # ==== NEW ====
    ) = args

    pair_id = pair["pair_id"]
    try:
        print(f"[{idx + 1}/{total}] 开始处理 pair {pair_id} ...")

        # 每个进程各自构建 transform，避免跨进程对象共享问题
        img_transform = build_image_transform(target_spacing=target_spacing)

        # =================== SOURCE ===================
        src_path = pair["source_image"]
        voxel_old_src = np.array(pair["source_voxel"]).reshape(-1, 3)[0]  # (x,y,z)

        src_nii_orig = nib.load(src_path)
        affine_old_src = src_nii_orig.affine

        src_res = img_transform({"image": src_path})
        src_img_new = src_res["image"]               # MetaTensor: (1, Y, X, Z)
        src_vol_new = src_img_new[0].numpy()         # (Y, X, Z)
        affine_new_src = np.asarray(src_img_new.meta["affine"], dtype=np.float64)

        voxel_new_src = map_voxel_old_to_new(
            voxel_old_src, affine_old_src, affine_new_src
        )[0]  # (3,)

        # 保存 NIfTI
        if save_nii:
            src_nii_path = os.path.join(out_img_dir, f"{pair_id}_source.nii.gz")
            # 为了让 voxel 坐标直接对应数组 index，这里写 identity affine
            nib.save(nib.Nifti1Image(src_vol_new, np.eye(4)), src_nii_path)

        # 保存 NPY & PKL  ==== NEW (PKL) ====
        if save_npy or save_pkl:
            src_npy_path = os.path.join(out_npy_dir, f"{pair_id}_source.npy")

        if save_npy:
            np.save(src_npy_path, src_vol_new)

        if save_pkl:
            # 优先用 MONAI pipeline 中的 image_meta_dict，和你第二段代码一致
            meta_dict = src_res.get("image_meta_dict", None)
            if meta_dict is None:
                # 兜底：从 MetaTensor 的 .meta 里取
                meta_dict = dict(src_img_new.meta)
            meta_dict['pair_id'] = pair_id 
            src_pkl_path = src_npy_path[:-4] + ".pkl"
            with open(src_pkl_path, "wb") as f:
                pickle.dump(meta_dict, f)

        # 保存 voxel 坐标 txt
        # 注意：这里沿用你验证通过的写法：voxel_new_src[[1, 0, 2]]
        src_lm_path = os.path.join(out_lm_dir, f"{pair_id}_source.txt")
        np.savetxt(src_lm_path, voxel_new_src[[1, 0, 2]].reshape(1, 3), fmt="%.3f")

        # =================== TARGET ===================
        tgt_path = pair["target_image"]
        voxel_old_tgt = np.array(pair["target_voxel"]).reshape(-1, 3)[0]

        tgt_nii_orig = nib.load(tgt_path)
        affine_old_tgt = tgt_nii_orig.affine

        tgt_res = img_transform({"image": tgt_path})
        tgt_img_new = tgt_res["image"]
        tgt_vol_new = tgt_img_new[0].numpy()
        affine_new_tgt = np.asarray(tgt_img_new.meta["affine"], dtype=np.float64)

        voxel_new_tgt = map_voxel_old_to_new(
            voxel_old_tgt, affine_old_tgt, affine_new_tgt
        )[0]

        if save_nii:
            tgt_nii_path = os.path.join(out_img_dir, f"{pair_id}_target.nii.gz")
            nib.save(nib.Nifti1Image(tgt_vol_new, np.eye(4)), tgt_nii_path)

        if save_npy or save_pkl:
            tgt_npy_path = os.path.join(out_npy_dir, f"{pair_id}_target.npy")

        if save_npy:
            np.save(tgt_npy_path, tgt_vol_new)

        if save_pkl:
            meta_dict = tgt_res.get("image_meta_dict", None)
            if meta_dict is None:
                meta_dict = dict(tgt_img_new.meta)
            meta_dict['pair_id'] = pair_id 
            tgt_pkl_path = tgt_npy_path[:-4] + ".pkl"
            with open(tgt_pkl_path, "wb") as f:
                pickle.dump(meta_dict, f)

        tgt_lm_path = os.path.join(out_lm_dir, f"{pair_id}_target.txt")
        np.savetxt(tgt_lm_path, voxel_new_tgt[[1, 0, 2]].reshape(1, 3), fmt="%.3f")

        print(f"[{idx + 1}/{total}] 完成 pair {pair_id}")

    except Exception as e:
        # 避免某个样本异常直接把整个并行任务干掉
        print(f"[{idx + 1}/{total}] 处理 pair {pair_id} 出错: {e}")


# ============================================================
# 5. 主流程：并行处理所有 pair
# ============================================================

def main():
    # ---------- 请按需修改这些路径 ----------
    nifti_input_dir = "/home/mingrui/disk1/datasets/DeepLesion/Images_nifti"
    json_annotation_dir = "/home/mingrui/disk1/datasets/DeepLesion/DLT_Annotations"
    output_root = "/home/mingrui/disk1/processed_dataset/DeepLesion_1mm"
    # ----------------------------------------

    out_img_dir = os.path.join(output_root, "images")      # 保存 nii
    out_npy_dir = os.path.join(output_root, "npy")         # 保存 npy & pkl
    out_lm_dir  = os.path.join(output_root, "landmarks")   # 保存 voxel txt

    # 根据开关创建目录
    if SAVE_NII:
        os.makedirs(out_img_dir, exist_ok=True)
    if SAVE_NPY or SAVE_PKL:
        os.makedirs(out_npy_dir, exist_ok=True)
    os.makedirs(out_lm_dir,  exist_ok=True)

    # 1) 解析配对
    pairs = parse_dlt_pairs(json_annotation_dir, nifti_input_dir)
    if len(pairs) == 0:
        print("未找到任何配对，请检查路径。")
        return

    total = len(pairs)
    print(f"开始并行处理 {total} 个配对数据...")
    num_workers = NUM_WORKERS
    print(f"使用进程数: {num_workers}")

    # 2) 组织参数并行执行
    args_list = [
        (
            i,
            total,
            pair,
            out_img_dir,
            out_npy_dir,
            out_lm_dir,
            TARGET_SPACING,
            SAVE_NPY,
            SAVE_NII,
            SAVE_PKL,
        )
        for i, pair in enumerate(pairs)
    ]

    if num_workers == 1:
        # 单进程调试模式
        for args in args_list:
            process_single_pair(args)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(process_single_pair, args_list))

    print("全部配对处理完成。输出目录：", output_root)


if __name__ == "__main__":
    main()