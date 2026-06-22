import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path

def create_landmark_label_map(image_path, landmark_path, cube_size=3):
    """
    根据给定的图像和地标点文件，创建一个标签图。

    Args:
        image_path (str): 处理后的 NIfTI 图像文件路径。
        landmark_path (str): 包含像素坐标的地标点 .txt 文件路径。
        cube_size (int): 在每个地标点处绘制的立方体的大小 (奇数)。

    Returns:
        nib.Nifti1Image: 一个 NIfTI 格式的标签图对象。
                         如果出错则返回 None。
    """
    try:
        # 1. 加载处理后的图像以获取其尺寸和空间信息
        img_nib = nib.load(image_path)
        img_dims = img_nib.shape
        img_affine = img_nib.affine # 使用图像的仿射矩阵，确保可以完美叠加

        # 2. 创建一个与图像同样大小的空白标签图
        label_map = np.zeros(img_dims, dtype=np.uint8)

        # 3. 加载地标点的像素坐标
        # ndmin=2 确保即使只有一个点也能正确读取为 2D 数组
        landmarks_vox = np.loadtxt(landmark_path, ndmin=2)

        # 确保有地标点可处理
        if landmarks_vox.size == 0:
            print(f"  - Warning: Landmark file is empty: {landmark_path}")
            # 返回一个空的标签图
            return nib.Nifti1Image(label_map, img_affine, img_nib.header)

        # 4. 在每个地标点的位置绘制一个立方体
        offset = cube_size // 2
        for point in landmarks_vox:
            # 将浮点坐标四舍五入为整数索引
            center_coords = np.round(point).astype(int)
            
            # 计算立方体的边界，并进行边界检查，防止索引越界
            z_start = max(0, center_coords[2] - offset)
            z_end = min(img_dims[2], center_coords[2] + offset + 1)
            
            y_start = max(0, center_coords[1] - offset)
            y_end = min(img_dims[1], center_coords[1] + offset + 1)

            x_start = max(0, center_coords[0] - offset)
            x_end = min(img_dims[0], center_coords[0] + offset + 1)
            
            # 在标签图中将立方体区域赋值为 1
            label_map[x_start:x_end, y_start:y_end, z_start:z_end] = 1
        
        # 5. 使用原始图像的仿射矩阵和头文件创建新的 NIfTI 对象
        label_map_nib = nib.Nifti1Image(label_map, img_affine, img_nib.header)
        return label_map_nib

    except FileNotFoundError:
        print(f"  - Error: File not found. Skipping pair: Image='{image_path}', Landmarks='{landmark_path}'")
        return None
    except Exception as e:
        print(f"  - An unexpected error occurred for {image_path}: {e}")
        return None


def main():
    # --- 1. 定义输入和输出文件夹 ---
    
    # 输入文件夹（即您上一个脚本的输出）
    processed_folder = '/home/mingrui/disk1/dataset/DIRLab_4DCT_1mm'
    processed_image_folder = os.path.join(processed_folder, 'images')
    processed_landmark_folder = os.path.join(processed_folder, 'landmarks')

    # # 输入文件夹（即您上一个脚本的输出）
    # processed_folder = '/home/mingrui/disk1/dataset/LiverDIR_1mm'
    # processed_image_folder = os.path.join(processed_folder, 'images')
    # processed_landmark_folder = os.path.join(processed_folder, 'landmarks')

    # # 输入文件夹（即您上一个脚本的输出）
    # processed_folder = '/home/mingrui/disk1/dataset/DeepLesion/Processed_1mm'
    # processed_image_folder = os.path.join(processed_folder, 'images')
    # processed_landmark_folder = os.path.join(processed_folder, 'landmarks')
    
    # 验证结果的输出文件夹
    validation_output_folder = '/home/mingrui/disk1/projects/20251103_DiffusionCorr/debug'
    os.makedirs(validation_output_folder, exist_ok=True)
    
    print("--- Starting Validation Process ---")
    print(f"Reading processed images from: {processed_image_folder}")
    print(f"Reading processed landmarks from: {processed_landmark_folder}")
    print(f"Saving validation label maps to: {validation_output_folder}")
    print("-" * 35)

    # --- 2. 查找所有处理过的地标点文件 ---
    landmark_files = sorted(glob.glob(os.path.join(processed_landmark_folder, "*.txt")))
    
    if not landmark_files:
        print("Error: No landmark files found in the specified directory. Please check the path.")
        return

    # --- 3. 循环处理 ---
    count = 0
    for landmark_path in landmark_files:
        landmark_filename = os.path.basename(landmark_path)
        
        # ================== 兼容性核心逻辑 ==================
        # 定义可能的图像文件名候选列表
        
        candidates = []
        
        # 候选 1: 旧数据集模式 (文件名完全一致，仅后缀不同)
        # e.g., Case1.txt -> Case1.nii.gz
        candidates.append(landmark_filename.replace('.txt', '.nii.gz'))
        
        # 候选 2: Liver DIR 模式 (landmarks 替换为 img)
        # e.g., case10_landmarks1.txt -> case10_img1.nii.gz
        if 'landmarks' in landmark_filename:
            candidates.append(landmark_filename.replace('landmarks', 'img').replace('.txt', '.nii.gz'))
            
        # 候选 3: 防止有些人命名为 Case1_landmarks.txt -> Case1.nii.gz (去后缀模式)
        if '_landmarks.txt' in landmark_filename:
            candidates.append(landmark_filename.replace('_landmarks.txt', '.nii.gz'))

        # 检查哪个候选文件真实存在
        image_path = None
        target_image_name = None
        
        for cand in candidates:
            cand_path = os.path.join(processed_image_folder, cand)
            if os.path.exists(cand_path):
                image_path = cand_path
                target_image_name = cand
                break # 找到了就停止尝试
        
        # ===================================================

        if image_path is None:
            print(f"Skipping: Could not find image for {landmark_filename}. Tried: {candidates}")
            continue

        print(f"Processing: {target_image_name} <-> {landmark_filename}")

        # 生成可视化标签
        label_map_nib = create_landmark_label_map(image_path, landmark_path, cube_size=3)
        
        if label_map_nib:
            # 输出文件名：使用图像名 + 后缀，保证一一对应
            output_filename = target_image_name.replace('.nii.gz', '_vis_landmarks.nii.gz')
            output_path = os.path.join(validation_output_folder, output_filename)
            
            nib.save(label_map_nib, output_path)
            print(f"  -> Saved: {output_filename}")
            count += 1

    print(f"\n--- Validation complete! Processed {count} pairs. ---")

if __name__ == "__main__":
    main()