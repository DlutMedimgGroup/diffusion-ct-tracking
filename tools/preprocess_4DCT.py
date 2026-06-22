import sys
import torch
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
import os
from monai.transforms import Transform
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    apply_transform,
    Lambdad,
    SaveImaged,
    ApplyTransformToPointsd,
    MapTransform
)
from monai.data import decollate_batch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utility import *

class LoadLandmarks(Transform):
    """
    自定义变换，用于从txt文件加载标志点。
    每行为一个点的三维坐标 (x, y, z)。
    """
    def __call__(self, data):
        d = dict(data)
        # 从txt文件读取坐标
        landmarks = np.loadtxt(d["landmarks"])
        # 确保是N x 3的形状
        d["landmarks"] = landmarks.reshape(-1, 3)
        return d

def save_landmarks(landmarks, original_path, output_dir):
    """保存更新后的标志点到txt文件。"""
    filename = os.path.basename(original_path)
    new_filename = filename.replace("_landmarks.txt", "_landmarks_processed.txt")
    output_path = os.path.join(output_dir, new_filename)
    np.savetxt(output_path, landmarks, fmt="%.6f")

class VoxelToPhysicald(MapTransform):
    """
    将输入的像素(Voxel)坐标根据参考图像的仿射矩阵转换为物理(Physical)坐标。
    """
    def __init__(self, keys, refer_keys):
        super().__init__(keys)
        self.refer_keys = refer_keys

    def __call__(self, data):
        d = dict(data)
        for key, refer_key in self.key_iterator(d, self.refer_keys):
            if key not in d or refer_key not in d:
                continue
            
            # 获取原始图像的仿射矩阵
            # 'original_affine' 是 LoadImaged 保留的变换前的仿射
            affine = d[refer_key].meta.get("original_affine")
            if affine is None:
                # 如果没有 'original_affine', 就用当前的，但前者更安全
                affine = d[refer_key].affine

            points = d[key]
            # 将点转换为齐次坐标 (N, 4)
            points_homog = np.hstack([points, np.ones((points.shape[0], 1))])
            
            # 应用仿射变换
            points_physical_homog = (affine @ points_homog.T).T
            
            # 转换回非齐次坐标 (N, 3)
            d[key] = points_physical_homog[:, :3]
        return d

def main():
    # input folder
    source_data_folder = '/home/mingrui/disk1/datasets/DIRLab_4DCT/Processed_Dataset_with_Landmarks'
    image_data_folder = os.path.join(source_data_folder, 'image')
    landmark_data_folder = os.path.join(source_data_folder, 'landmark')

    SAVE_IMAGE = True
    SAVE_NPY = True

    # output folder
    output_folder = '/home/mingrui/disk1/processed_dataset/DIRLab_4DCT_1mm'
    if SAVE_IMAGE:
        output_image_folder = join(output_folder, 'images')
        maybe_mkdir_p(output_image_folder)
    if SAVE_NPY:
        output_npy_folder = join(output_folder, 'npy')
        maybe_mkdir_p(output_npy_folder)
    output_landmark_folder = join(output_folder, 'landmarks')
    maybe_mkdir_p(output_landmark_folder)

    # 1. 准备数据列表
    data_dicts = []
    for data_name in subfiles(image_data_folder, join=False):
        image_path = join(image_data_folder, data_name)
        landmark_name = data_name.replace('.nii.gz', '.txt')
        landmark_path = join(landmark_data_folder, landmark_name)
        data_dicts.append({
            'image': image_path,
            'landmarks': landmark_path
        })
    
    # 2. 定义预处理变换
    pre_transforms = Compose([
        LoadImaged(keys=["image"],image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        LoadLandmarks(),
        VoxelToPhysicald(keys=["landmarks"], refer_keys=["image"]),
        EnsureChannelFirstd(keys=["landmarks"], channel_dim="no_channel"),
        Lambdad(keys="image", func=lambda x: x - 1024),
        ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        Orientationd(keys=["image"], axcodes="RPS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear"
        ),
        ApplyTransformToPointsd(
            keys=["landmarks"],    # 要变换的点
            refer_keys=["image"],  # 参考的图像，从中获取变换历史
            dtype=torch.float32,
        ),
        EnsureTyped(keys=["image"]),
    ])
    
    # 3. 处理数据并应用变换
    processed_data = []
    for data in data_dicts:
        # 整个流水线一次性应用
        processed_dict = pre_transforms(data)
        processed_data.append(processed_dict)
    
    # 4. 保存结果
    for i, item in enumerate(processed_data):
        # 保存图像
        if SAVE_IMAGE:
            original_image_path = data_dicts[i]["image"]
            filename = os.path.basename(original_image_path)
            output_image_path = os.path.join(output_image_folder, filename)
            nib.save(nib.Nifti1Image(item["image"][0].numpy(), np.eye(4)), output_image_path)
        if SAVE_NPY:
            npy_output_path = os.path.join(output_npy_folder, filename.replace('.nii.gz', '.npy'))
            np.save(npy_output_path, item["image"].numpy())
            dict = item['image_meta_dict']
            with open(npy_output_path[:-4]+'.pkl', 'wb') as f:
                pickle.dump(dict, f)
        # 保存标定点
        output_path = os.path.join(output_landmark_folder, filename.replace('.nii.gz', '.txt'))
        np.savetxt(output_path, item["landmarks"][0], fmt="%.6f")
        print(f"Saved processed landmarks to {output_path}")

if __name__ == "__main__":
    main()
