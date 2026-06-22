import os
import json
from glob import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)

# ======================= 1. 解析 DLT 配对（只拿第一对） =======================

def parse_dlt_pairs(json_folder, nifti_root_dir):
    json_files = glob(os.path.join(json_folder, "*.json"))
    pair_list = []
    seen_pairs = set()

    print(f"正在解析 JSON 文件: {json_files}")

    for jf in json_files:
        base = os.path.splitext(os.path.basename(jf))[0]  # train / val / test
        with open(jf, "r") as f:
            data = json.load(f)

        for pid, ct in data.items():
            s_name = ct["source"]
            t_name = ct["target"]

            s_center_vox = np.array(ct["source center"], dtype=np.float32)  # (x,y,z)
            t_center_vox = np.array(ct["target center"], dtype=np.float32)

            s_key = tuple(np.round(s_center_vox, 3))
            t_key = tuple(np.round(t_center_vox, 3))

            s_path = os.path.join(nifti_root_dir, s_name)
            t_path = os.path.join(nifti_root_dir, t_name)

            if not (os.path.exists(s_path) and os.path.exists(t_path)):
                continue

            key = (s_name, t_name, s_key, t_key)
            rev_key = (t_name, s_name, t_key, s_key)
            if key in seen_pairs or rev_key in seen_pairs:
                continue

            seen_pairs.add(key)

            pair_list.append({
                "pair_id": f"{base}_{pid}",
                "source_image": s_path,
                "target_image": t_path,
                "source_voxel": [s_center_vox.tolist()],
                "target_voxel": [t_center_vox.tolist()],
            })

    print(f"解析完成，共得到 {len(pair_list)} 个去重后的配对。")
    return pair_list


# ======================= 2. 两种预处理：before(LPS) / after(LPS+Spacing) =======================

def build_before_lps_transform():
    """只把原始图像重排到 LPS，不改 spacing。"""
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="LPS"),
        EnsureTyped(keys=["image"]),
    ])

def build_after_lps_transform(target_spacing=(1.0, 1.0, 1.0)):
    """和之前一样：LPS + Spacing + 强度归一化。"""
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys="image",
            a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        Orientationd(keys=["image"], axcodes="LPS"),
        Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
        EnsureTyped(keys=["image"]),
    ])


# ======================= 3. voxel_old -> voxel_new : A_new^-1 · A_old =======================

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


# ======================= 4. 保存 2D PNG =======================

def save_slice_with_marker(volume, voxel, out_path, title=""):
    """
    volume: (Y, X, Z)
    voxel:  (x, y, z) —— (列, 行, 层)
    """
    vol = np.asarray(volume)
    y_dim, x_dim, z_dim = vol.shape

    vx = np.asarray(voxel, dtype=np.float64).reshape(-1)
    x, y, z = vx.tolist()

    x_int = int(round(x))
    y_int = int(round(y))
    z_int = int(round(z))

    print(f"保存切片 {out_path}: voxel = ({x:.2f}, {y:.2f}, {z:.2f}) -> 索引 = (x={x_int}, y={y_int}, z={z_int})")
    print(f"体数据 shape = (Y={y_dim}, X={x_dim}, Z={z_dim})")

    if not (0 <= x_int < x_dim and 0 <= y_int < y_dim and 0 <= z_int < z_dim):
        print(f"[WARN] 标注点超出范围，无法在切片中正确显示！")

    z_plot = min(max(z_int, 0), z_dim - 1)
    slice_img = vol[:, :, z_plot]

    plt.figure(figsize=(5, 5))
    plt.imshow(slice_img, cmap="gray", origin="lower")
    if 0 <= x_int < x_dim and 0 <= y_int < y_dim:
        plt.scatter([x_int], [y_int], c="r", s=40)
    plt.title(title)
    plt.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ======================= 5. 保存 3D NIfTI 并在体数据中标记坐标 =======================

def save_nii_with_marker(volume, voxel, affine, out_path, cube_radius=1):
    """
    volume: (Y, X, Z)
    voxel:  (x, y, z)
    affine: LPS 下的 affine
    """
    vol = np.asarray(volume).astype(np.float32)
    y_dim, x_dim, z_dim = vol.shape

    vx = np.asarray(voxel, dtype=np.float64).reshape(-1)
    x, y, z = vx.tolist()
    x_int = int(round(x))
    y_int = int(round(y))
    z_int = int(round(z))

    print(f"保存 NIfTI {out_path}: voxel = ({x:.2f}, {y:.2f}, {z:.2f}) -> 索引 = (x={x_int}, y={y_int}, z={z_int})")
    print(f"NIfTI 体数据 shape = (Y={y_dim}, X={x_dim}, Z={z_dim})")

    vol_marked = vol.copy()
    marker_val = float(vol.max() * 1.2) if vol.max() > 0 else 1.0

    if 0 <= x_int < x_dim and 0 <= y_int < y_dim and 0 <= z_int < z_dim:
        y0 = max(0, y_int - cube_radius)
        y1 = min(y_dim, y_int + cube_radius + 1)
        x0 = max(0, x_int - cube_radius)
        x1 = min(x_dim, x_int + cube_radius + 1)
        z0 = max(0, z_int - cube_radius)
        z1 = min(z_dim, z_int + cube_radius + 1)
        vol_marked[y0:y1, x0:x1, z0:z1] = marker_val
    else:
        print("[WARN] 标注点超出范围，NIfTI 中不写 marker。")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    affine = np.asarray(affine, dtype=np.float64)
    nib.save(nib.Nifti1Image(vol_marked, affine), out_path)


# ======================= 6. 主逻辑：所有 NIfTI 都是 LPS =======================

def main():
    # ---------- 路径 ----------
    nifti_input_dir = "/home/mingrui/disk1/dataset/DeepLesion/Images_nifti"
    json_annotation_dir = "/home/mingrui/disk1/dataset/DeepLesion/DLT_Annotations"
    debug_out_dir = "/home/mingrui/disk1/dataset/DeepLesion/debug_output"
    # -------------------------

    png_dir = os.path.join(debug_out_dir, "png")
    nii_dir = os.path.join(debug_out_dir, "nii")
    lm_dir  = os.path.join(debug_out_dir, "landmarks") 
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(nii_dir, exist_ok=True)
    os.makedirs(lm_dir,  exist_ok=True)

    # 1) 解析配对，只关心第一对
    pairs = parse_dlt_pairs(json_annotation_dir, nifti_input_dir)
    if len(pairs) == 0:
        print("未找到任何配对，请检查路径。")
        return

    pair = pairs[2]
    pair_id = pair["pair_id"]
    print(f"只处理第一对数据: {pair_id}")

    # 2) 两种 transform
    tf_before_lps = build_before_lps_transform()
    tf_after_lps  = build_after_lps_transform(target_spacing=(1.0, 1.0, 1.0))

    # -------------------------- SOURCE --------------------------
    print("\n===== SOURCE =====")
    src_path = pair["source_image"]
    src_voxel_old = np.array(pair["source_voxel"]).reshape(-1, 3)[0]

    # 原始 affine（只用于映射）
    src_nii_orig = nib.load(src_path)
    affine_old_src = src_nii_orig.affine
    print("Source 原始 affine:")
    print(affine_old_src)

    # -------- BEFORE：只做 Orientation 到 LPS --------
    src_before_res = tf_before_lps({"image": src_path})
    src_before_img = src_before_res["image"]
    src_vol_before_lps = src_before_img[0].numpy()  # (Y, X, Z)
    affine_before_lps = np.asarray(src_before_img.meta["affine"], dtype=np.float64)

    print("Source BEFORE LPS affine:")
    print(affine_before_lps)
    print("Source BEFORE LPS shape:", src_vol_before_lps.shape)

    # voxel_old -> voxel_before_lps
    src_voxel_before_lps = map_voxel_old_to_new(
        src_voxel_old, affine_old_src, affine_before_lps
    )[0]

    np.savetxt(
        os.path.join(lm_dir, f"{pair_id}_source_before_LPS.txt"),
        src_voxel_before_lps.reshape(1, 3),
        fmt="%.3f"
    )

    print("Source BEFORE LPS voxel:", src_voxel_before_lps)

    save_slice_with_marker(
        src_vol_before_lps,
        src_voxel_before_lps,
        os.path.join(png_dir, f"{pair_id}_source_before_LPS.png"),
        title=f"{pair_id} source BEFORE_LPS\nvoxel={src_voxel_before_lps}"
    )

    save_nii_with_marker(
        src_vol_before_lps,
        src_voxel_before_lps,
        affine_before_lps,
        os.path.join(nii_dir, f"{pair_id}_source_before_LPS_marked.nii.gz"),
        cube_radius=1,
    )

    # -------- AFTER：LPS + Spacing（和之前一致） --------
    src_after_res = tf_after_lps({"image": src_path})
    src_after_img = src_after_res["image"]
    src_vol_after_lps = src_after_img[0].numpy()
    affine_after_lps = np.asarray(src_after_img.meta["affine"], dtype=np.float64)

    print("Source AFTER LPS+Spacing affine:")
    print(affine_after_lps)
    print("Source AFTER LPS+Spacing shape:", src_vol_after_lps.shape)

    src_voxel_after_lps = map_voxel_old_to_new(
        src_voxel_old, affine_old_src, affine_after_lps
    )[0]

    np.savetxt(
        os.path.join(lm_dir, f"{pair_id}_source_after_LPS.txt"),
        src_voxel_after_lps.reshape(1, 3),
        fmt="%.3f"
    )

    print("Source AFTER LPS+Spacing voxel:", src_voxel_after_lps)

    save_slice_with_marker(
        src_vol_after_lps,
        src_voxel_after_lps,
        os.path.join(png_dir, f"{pair_id}_source_after_LPS.png"),
        title=f"{pair_id} source AFTER_LPS\nvoxel={src_voxel_after_lps}"
    )

    save_nii_with_marker(
        src_vol_after_lps,
        src_voxel_after_lps,
        affine_after_lps,
        os.path.join(nii_dir, f"{pair_id}_source_after_LPS_marked.nii.gz"),
        cube_radius=1,
    )

    # -------------------------- TARGET --------------------------
    print("\n===== TARGET =====")
    tgt_path = pair["target_image"]
    tgt_voxel_old = np.array(pair["target_voxel"]).reshape(-1, 3)[0]

    tgt_nii_orig = nib.load(tgt_path)
    affine_old_tgt = tgt_nii_orig.affine
    print("Target 原始 affine:")
    print(affine_old_tgt)

    # BEFORE: LPS
    tgt_before_res = tf_before_lps({"image": tgt_path})
    tgt_before_img = tgt_before_res["image"]
    tgt_vol_before_lps = tgt_before_img[0].numpy()
    affine_before_tgt_lps = np.asarray(tgt_before_img.meta["affine"], dtype=np.float64)

    print("Target BEFORE LPS affine:")
    print(affine_before_tgt_lps)
    print("Target BEFORE LPS shape:", tgt_vol_before_lps.shape)

    tgt_voxel_before_lps = map_voxel_old_to_new(
        tgt_voxel_old, affine_old_tgt, affine_before_tgt_lps
    )[0]
    np.savetxt(
        os.path.join(lm_dir, f"{pair_id}_target_before_LPS.txt"),
        tgt_voxel_before_lps.reshape(1, 3),
        fmt="%.3f"
    )

    print("Target BEFORE LPS voxel:", tgt_voxel_before_lps)

    save_slice_with_marker(
        tgt_vol_before_lps,
        tgt_voxel_before_lps,
        os.path.join(png_dir, f"{pair_id}_target_before_LPS.png"),
        title=f"{pair_id} target BEFORE_LPS\nvoxel={tgt_voxel_before_lps}"
    )

    save_nii_with_marker(
        tgt_vol_before_lps,
        tgt_voxel_before_lps,
        affine_before_tgt_lps,
        os.path.join(nii_dir, f"{pair_id}_target_before_LPS_marked.nii.gz"),
        cube_radius=1,
    )

    # AFTER: LPS + Spacing
    tgt_after_res = tf_after_lps({"image": tgt_path})
    tgt_after_img = tgt_after_res["image"]
    tgt_vol_after_lps = tgt_after_img[0].numpy()
    affine_after_tgt_lps = np.asarray(tgt_after_img.meta["affine"], dtype=np.float64)

    print("Target AFTER LPS+Spacing affine:")
    print(affine_after_tgt_lps)
    print("Target AFTER LPS+Spacing shape:", tgt_vol_after_lps.shape)

    tgt_voxel_after_lps = map_voxel_old_to_new(
        tgt_voxel_old, affine_old_tgt, affine_after_tgt_lps
    )[0]

    np.savetxt(
        os.path.join(lm_dir, f"{pair_id}_target_after_LPS.txt"),
        tgt_voxel_after_lps.reshape(1, 3),
        fmt="%.3f"
    )
    
    print("Target AFTER LPS+Spacing voxel:", tgt_voxel_after_lps)

    save_slice_with_marker(
        tgt_vol_after_lps,
        tgt_voxel_after_lps,
        os.path.join(png_dir, f"{pair_id}_target_after_LPS.png"),
        title=f"{pair_id} target AFTER_LPS\nvoxel={tgt_voxel_after_lps}"
    )

    save_nii_with_marker(
        tgt_vol_after_lps,
        tgt_voxel_after_lps,
        affine_after_tgt_lps,
        os.path.join(nii_dir, f"{pair_id}_target_after_LPS_marked.nii.gz"),
        cube_radius=1,
    )

    print("\nPNG 输出目录：", png_dir)
    print("NIfTI(LPS) 输出目录：", nii_dir)


if __name__ == "__main__":
    main()