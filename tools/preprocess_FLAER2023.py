import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import nibabel as nib
from functools import partial
import concurrent.futures
from tqdm import tqdm

# 让子进程保存 png 时不需要 GUI
import matplotlib
matplotlib.use("Agg")
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

from utility import *  # maybe_mkdir_p, subfiles, join 等


# ------------------------------------------------------------------------------
# 1) 判断是否“像 CT HU”的启发式检测
# ------------------------------------------------------------------------------
def is_ct_hu_like(img_dhw: np.ndarray):
    """
    返回 (ok: bool, reason: str)
    规则（可按需要调整）：
      - 必须是有限数值
      - 使用 p1/p99 判断：
          * p1 < -200  (应包含明显负值：空气/肺/背景)
          * p99 > 200  (应包含组织/骨的上端范围)
      - 另外排除常见“已归一化”情况：整体落在 [0,1.5] 且无明显负值
    """
    if img_dhw.size == 0:
        return False, "empty array"

    if not np.isfinite(img_dhw).all():
        return False, "non-finite values (nan/inf)"

    vmin = float(img_dhw.min())
    vmax = float(img_dhw.max())

    # 常见非 CT：已经归一化到 0~1 或 0~1.x
    if vmin >= 0.0 and vmax <= 1.5:
        return False, f"looks normalized (min={vmin:.3f}, max={vmax:.3f})"

    # 用分位数更稳健（避免极少数异常点）
    p1, p99 = np.percentile(img_dhw, [1, 99])

    # 过窄动态范围也可疑（比如 MRI 或被裁剪过的非 HU）
    if (p99 - p1) < 300:
        return False, f"too small dynamic range (p1={p1:.1f}, p99={p99:.1f})"

    if p1 >= -200:
        return False, f"no enough negatives for HU (p1={p1:.1f})"

    if p99 <= 200:
        return False, f"upper tail too low for HU (p99={p99:.1f})"

    # 也可以加一个“合理 HU 上限”的软约束（不强制）
    if vmax > 10000:
        return False, f"too large max suspicious (max={vmax:.1f})"

    return True, "ok"


# ------------------------------------------------------------------------------
# 2) 保存居中冠状面 PNG（用于人工检查）
# ------------------------------------------------------------------------------
def save_mid_coronal_png(img_1dhw: np.ndarray, out_png_path: str):
    """
    img_1dhw: [1, D, H, W] 或 [D,H,W] 均可
    保存居中冠状面：沿 H 取中间 -> 得到 [D, W]
    """
    if img_1dhw.ndim == 4:
        img_dhw = img_1dhw[0]
    elif img_1dhw.ndim == 3:
        img_dhw = img_1dhw
    else:
        raise ValueError(f"unexpected shape for png saving: {img_1dhw.shape}")

    D, H, W = img_dhw.shape
    mid_h = H // 2
    coronal = img_dhw[:, mid_h, :]  # [D, W]

    # 为了更直观：把 D 放到纵轴，W 放到横轴；origin 用 lower 避免倒置观感
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(coronal, cmap="gray", origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    plt.close()


# ------------------------------------------------------------------------------
# 子进程任务：处理单个样本并保存
# ------------------------------------------------------------------------------
def process_and_save(
    data_dict,
    pre_transforms_raw,     # 用于检测 CT HU
    pre_transforms_final,   # 用于最终保存（含 scale/spacing/orient 等）
    output_image_folder,
    output_npy_folder,
    output_png_folder,
    SAVE_IMAGE,
    SAVE_NPY,
    SAVE_PNG,
):
    original_image_path = data_dict["image"]
    filename = os.path.basename(original_image_path)

    try:
        # --- (A) 先做“原始强度”的检测（不做 ScaleIntensityRanged） ---
        raw = pre_transforms_raw(data_dict)
        raw_img = raw["image"]  # torch tensor [1,D,H,W]
        raw_dhw = raw_img[0].cpu().numpy()

        ok, reason = is_ct_hu_like(raw_dhw)
        if not ok:
            return f"Skipped {filename}: {reason}"

        # --- (B) 通过检测后，做正式预处理（含归一化/spacing/orientation等） ---
        processed = pre_transforms_final(data_dict)

        img_chwd = processed["image"]           # torch Tensor [1,D,H,W]
        img_dhw = img_chwd[0].cpu().numpy()     # np [D,H,W]

        # 1) 保存 NIfTI（可选）
        if SAVE_IMAGE:
            out_path = os.path.join(output_image_folder, filename)
            nib.save(nib.Nifti1Image(img_dhw, np.eye(4)), out_path)

        # 2) 保存 NPY + meta（可选）
        if SAVE_NPY:
            npy_path = os.path.join(
                output_npy_folder,
                filename.replace(".nii.gz", ".npy").replace(".nii", ".npy")
            )
            np_array = img_chwd.cpu().numpy().astype(np.float32)  # [1,D,H,W]
            np.save(npy_path, np_array)

            if np_array.min() > 0.05:
                print(f"Warning: min value > 0.05 in {filename}")

            meta = processed.get("image_meta_dict", {})
            with open(npy_path[:-4] + ".pkl", "wb") as f:
                pickle.dump(meta, f)

        # 3) 保存居中冠状面 PNG（建议始终开着）
        if SAVE_PNG:
            png_path = os.path.join(
                output_png_folder,
                filename.replace(".nii.gz", ".png").replace(".nii", ".png")
            )
            # 这里用“最终处理后”的强度（已经 0~1）做可视化，更方便人工检查
            save_mid_coronal_png(img_chwd.cpu().numpy(), png_path)

        return f"Successfully processed {filename}"

    except Exception as e:
        return f"Failed to process {original_image_path}: {e}"


def main():
    # -------------------------
    # 输入 / 输出参数（变量方式）
    # -------------------------
    image_data_folder = "/home/mingrui/disk1/datasets/Flare2023"

    SAVE_IMAGE = False
    SAVE_NPY = True
    SAVE_PNG = False

    output_folder = "/home/mingrui/ssd1/processed_dataset/Flare2023_1mm"

    output_image_folder = None
    output_npy_folder = None
    output_png_folder = None

    if SAVE_IMAGE:
        output_image_folder = os.path.join(output_folder, "images")
        maybe_mkdir_p(output_image_folder)
    if SAVE_NPY:
        output_npy_folder = os.path.join(output_folder, "npy")
        maybe_mkdir_p(output_npy_folder)
    if SAVE_PNG:
        output_png_folder = os.path.join(output_folder, "png_coronal_mid")
        maybe_mkdir_p(output_png_folder)

    # -------------------------
    # 1) 收集数据列表
    # -------------------------
    file_list_2200 = subfiles(join(image_data_folder, "imagesTr2200"), join=True, suffix=".nii.gz")
    file_list_1800 = subfiles(join(image_data_folder, "unlabelTr1800"), join=True, suffix=".nii.gz")
    file_list = file_list_2200 + file_list_1800
    data_dicts = [{"image": p} for p in sorted(file_list)]

    # -------------------------
    # 2) 两套 transforms
    #   - raw：用于 CT HU 检测（不做 ScaleIntensityRanged）
    #   - final：用于最终保存（含 scale / spacing / orientation）
    # -------------------------
    pre_transforms_raw = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])

    pre_transforms_final = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        Orientationd(keys=["image"], axcodes="LPS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),
        EnsureTyped(keys=["image"]),
    ])

    # -------------------------
    # 3) 多进程并行处理
    # -------------------------
    num_workers = 4
    print(f"Using {num_workers} processes for parallel execution.")
    print(f"Input cases: {len(data_dicts)}")
    print(f"Output folder: {output_folder}")
    print("Note: No DivisiblePadd. CT HU check enabled. Mid-coronal PNG will be saved.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        task = partial(
            process_and_save,
            pre_transforms_raw=pre_transforms_raw,
            pre_transforms_final=pre_transforms_final,
            output_image_folder=output_image_folder,
            output_npy_folder=output_npy_folder,
            output_png_folder=output_png_folder,
            SAVE_IMAGE=SAVE_IMAGE,
            SAVE_NPY=SAVE_NPY,
            SAVE_PNG=SAVE_PNG,
        )
        results = list(tqdm(executor.map(task, data_dicts), total=len(data_dicts)))

    print("\nProcessing finished. Summary:")
    n_ok = sum(r.startswith("Successfully") for r in results)
    n_skip = sum(r.startswith("Skipped") for r in results)
    n_fail = sum(r.startswith("Failed") for r in results)
    print(f"  Successfully: {n_ok}")
    print(f"  Skipped:      {n_skip}")
    print(f"  Failed:       {n_fail}")

    if n_skip or n_fail:
        print("\nDetails (Skipped/Failed):")
        for r in results:
            if r.startswith("Skipped") or r.startswith("Failed"):
                print(r)


if __name__ == "__main__":
    main()