import torch
import pickle
import numpy as np
import nibabel as nib
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
    Lambdad,
    ApplyTransformToPointsd,
    MapTransform
)
# 保留 utility 引用
from utility import *

class LoadLandmarks(Transform):
    """
    自定义变换，用于从txt文件加载标志点。
    每行为一个点的三维坐标 (x, y, z)。
    """
    def __call__(self, data):
        d = dict(data)
        # 从txt文件读取坐标
        landmarks = np.loadtxt(d["landmarks"], delimiter=",")
        # 确保是N x 3的形状
        d["landmarks"] = landmarks.reshape(-1, 3)
        return d

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
            affine = d[refer_key].meta.get("original_affine")
            if affine is None:
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
    # ================= 路径配置 =================
    # Liver DIR 原始数据文件夹路径（所有 img 和 landmarks 都在这里）
    source_data_folder = '/home/mingrui/disk1/dataset/Liver DIR Landmark Dataset/NIfTI'
    
    # 输出路径
    output_folder = '/home/mingrui/disk1/dataset/LiverDIR_1mm'

    SAVE_IMAGE = True
    SAVE_NPY = True

    if SAVE_IMAGE:
        output_image_folder = join(output_folder, 'images')
        maybe_mkdir_p(output_image_folder)
    if SAVE_NPY:
        output_npy_folder = join(output_folder, 'npy')
        maybe_mkdir_p(output_npy_folder)
    output_landmark_folder = join(output_folder, 'landmarks')
    maybe_mkdir_p(output_landmark_folder)

    # ================= 1. 准备数据列表 =================
    data_dicts = []
    
    # 获取源文件夹下所有文件名
    all_files = subfiles(source_data_folder, join=False)
    
    # 筛选出图像文件 (.nii)
    # Liver DIR 通常没有 .gz 后缀，如果是 .nii.gz 请自行调整
    image_files = [f for f in all_files if f.endswith('.nii')]
    image_files.sort()

    for data_name in image_files:
        # data_name example: case10_img1.nii
        image_path = join(source_data_folder, data_name)
        
        # 构造对应的 landmark 文件名
        # 规则: 替换 'img' 为 'landmarks', 替换 '.nii' 为 '.txt'
        # example: case10_img1.nii -> case10_landmarks1.txt
        landmark_name = data_name.replace('img', 'landmarks').replace('.nii', '.txt')
        landmark_path = join(source_data_folder, landmark_name)
        
        # 检查对应的 landmark 文件是否存在
        if os.path.exists(landmark_path):
            data_dicts.append({
                'image': image_path,
                'landmarks': landmark_path,
                'id': data_name # 用于保存时的文件名索引
            })
        else:
            print(f"Warning: Landmark file {landmark_name} not found for {data_name}, skipping.")
            
    print(f"Total paired data found: {len(data_dicts)}")

    # ================= 2. 定义预处理变换 =================
    pre_transforms = Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        LoadLandmarks(),
        
        # 关键：先转物理坐标，防止 Spacing/Orientation 改变导致坐标错位
        VoxelToPhysicald(keys=["landmarks"], refer_keys=["image"]),
        EnsureChannelFirstd(keys=["landmarks"], channel_dim="no_channel"),
        
        # --- 修改点：灰度值修正 ---
        # Liver DIR 偏移了 1000
        Lambdad(keys="image", func=lambda x: x - 1000),
        
        # 截断范围建议：虽然是肝脏，但通常保留 -1000 到 1000 作为一个较宽的全身范围
        ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000,
            b_min=0.0, b_max=1.0, clip=True
        ),
        
        Orientationd(keys=["image"], axcodes="LPS"),
        
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0), # 重采样到 1mm
            mode="bilinear"
        ),
        
        ApplyTransformToPointsd(
            keys=["landmarks"],    
            refer_keys=["image"],  
            dtype=torch.float32,
        ),
        EnsureTyped(keys=["image"]),
    ])
    
    # ================= 3. 处理数据并即时保存 =================
    # 这一步合并了原来的处理和保存步骤，处理完一个保存一个，节省内存
    for i, data in enumerate(data_dicts):
        filename = data['id']  # 获取文件名，例如 case10_img1.nii
        print(f"Processing {i+1}/{len(data_dicts)}: {filename}")
        
        try:
            # 1. 应用变换 (Loading -> Spacing -> Orientation -> ...)
            item = pre_transforms(data)
            
            # 2. 保存图像
            if SAVE_IMAGE:
                output_image_path = join(output_image_folder, filename.replace('.nii', '.nii.gz'))
                # 按照要求：强制使用 np.eye(4) 作为 affine 矩阵
                # item["image"][0] 取出数据，形状为 (H, W, D)
                nib.save(nib.Nifti1Image(item["image"][0].cpu().numpy(), np.eye(4)), output_image_path)
            
            # 3. 保存 NPY 和 PKL (元数据)
            if SAVE_NPY:
                npy_output_path = join(output_npy_folder, filename.replace('.nii', '.npy'))
                np.save(npy_output_path, item["image"].cpu().numpy())
                
                # 处理元数据字典
                meta_dict = item['image_meta_dict']
                    
                with open(npy_output_path[:-4]+'.pkl', 'wb') as f:
                    pickle.dump(meta_dict, f)
            
            # 4. 保存标定点
            # 构造 landmark 输出文件名
            landmark_filename = filename.replace('img', 'landmarks').replace('.nii', '.txt')
            output_path = join(output_landmark_folder, landmark_filename)
            
            # 保存变换后的坐标
            # item["landmarks"] 形状为 (1, N, 3)，取 [0] 得到 (N, 3)
            np.savetxt(output_path, item["landmarks"][0].cpu().numpy(), fmt="%.6f")
            
            print(f"  Saved successfully -> {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # 如果需要调试详细错误，可以取消下面两行的注释
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    main()