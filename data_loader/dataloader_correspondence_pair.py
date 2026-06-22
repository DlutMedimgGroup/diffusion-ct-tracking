import json
import os
import pickle
from typing import Dict, Hashable, Mapping, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Invertd,
    InvertibleTransform,
    LoadImaged,
    MapTransform,
    Orientationd,
    Randomizable,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)

from data_loader.dataloader_base import DataloaderBase
from utility import *


class LoadPreprocessed(MapTransform):
    """Load preprocessed .npy volumes (mmap) and a shared meta dict (.pkl) for a pair."""

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        basename = os.path.basename(d[self.first_key(d)])
        basename = basename.split(".")[0]

        meta_path = join(os.path.dirname(d[self.first_key(d)]), f"{basename}.pkl")
        with open(meta_path, "rb") as f:
            d["image_meta_dict"] = pickle.load(f)
            d["image_meta_dict"]["image_1_path"] = d["image_1"]
            d["image_meta_dict"]["image_2_path"] = d["image_2"]
            d["image_meta_dict"]["pair_id"] = basename[:9]

        for key in self.keys:
            d[key] = np.load(d[key], mmap_mode="r")

        return d


class RandomPairDataset(Dataset):
    """
    Create random image pairs on-the-fly from a list of single-image records.

    Each item in `data_list` should be like:
        {"image": "path/to/img.npy"}
    """

    def __init__(self, data_list: list, transform: callable = None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        item1 = self.data_list[idx]

        idx2 = np.random.randint(0, len(self.data_list))
        while idx2 == idx:
            idx2 = np.random.randint(0, len(self.data_list))

        item2 = self.data_list[idx2]

        # Expected input format for downstream transforms.
        paired_data = {"image": (item1["image"], item2["image"])}

        if self.transform is not None:
            paired_data = self.transform(paired_data)

        return paired_data


class RandSamPatchFourd(MapTransform, InvertibleTransform, Randomizable):
    """Sample two overlapping patches from each image (4 patches total) and generate positive point pairs."""

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        n_pos: float,
        allow_missing_keys: bool = False,
        scaling_factor: float = 0.1,
        overlap_factor: float = 0.5,  # 1.0 means full overlap
    ):
        super().__init__(keys, allow_missing_keys)
        self.offset_factor = 1 - overlap_factor
        self.roi_size = roi_size
        self.scaling_factor = scaling_factor
        self.n_pos = n_pos

    def _get_overlap_range(self, patch_range):
        r = np.zeros((1, 6))
        r[0, :3] = np.maximum(patch_range[0, :3], patch_range[1, :3])
        r[0, 3:] = np.minimum(patch_range[0, 3:], patch_range[1, 3:])
        return r

    def _get_positive_pos(self, patch_range, overlap_range):
        p = np.random.rand(self.n_pos, 3) * (overlap_range[:, 3:] - overlap_range[:, :3]) + overlap_range[:, :3]

        size0 = (patch_range[:1, 3:] - patch_range[:1, :3]).astype(np.float32)
        size1 = (patch_range[1:, 3:] - patch_range[1:, :3]).astype(np.float32)

        den0 = np.maximum(size0 - 1.0, 1.0)
        den1 = np.maximum(size1 - 1.0, 1.0)

        positive_pos = np.zeros((self.n_pos, 2, 3), dtype=np.float32)
        positive_pos[:, 0, :] = (p - patch_range[:1, :3]) / den0
        positive_pos[:, 1, :] = (p - patch_range[1:, :3]) / den1

        positive_pos = positive_pos * 2 - 1
        positive_pos = np.clip(positive_pos, -1.0, 1.0).astype(np.float32)
        return positive_pos

    def __call__(self, data):
        d = dict(data)
        assert len(self.keys) == 1, "error keys form RandSamPatchPaird."

        ori_image_1 = d["image_pair_0"]
        ori_image_2 = d["image_pair_1"]

        numpy_mode = isinstance(ori_image_1, np.ndarray)
        if numpy_mode:
            ori_shape_1 = ori_image_1.shape
            ori_shape_2 = ori_image_2.shape
        else:
            ori_shape_1 = ori_image_1.size()
            ori_shape_2 = ori_image_2.size()

        assert ori_shape_1[0] == 1, "error channel form RandSamPatchPaird."
        ori_shape_1 = ori_shape_1[1:]
        ori_shape_2 = ori_shape_2[1:]

        result = torch.zeros([4] + self.roi_size)

        # Generate patch ranges.
        roi_size_11 = np.array(self.roi_size)
        roi_size_12 = np.array(self.roi_size)
        roi_size_21 = np.array(self.roi_size)
        roi_size_22 = np.array(self.roi_size)

        rand = torch.rand(4, 3)
        patch_range_1 = torch.zeros(2, 6)
        patch_range_2 = torch.zeros(2, 6)

        for dim in range(3):
            roi_size_11[dim] = torch.round(
                self.roi_size[dim] * (1 + rand[0, dim] * 2 * self.scaling_factor - self.scaling_factor)
            )
            roi_size_12[dim] = torch.round(
                self.roi_size[dim] * (1 + rand[1, dim] * 2 * self.scaling_factor - self.scaling_factor)
            )
            roi_size_21[dim] = torch.round(
                self.roi_size[dim] * (1 + rand[0, dim] * 2 * self.scaling_factor - self.scaling_factor)
            )
            roi_size_22[dim] = torch.round(
                self.roi_size[dim] * (1 + rand[1, dim] * 2 * self.scaling_factor - self.scaling_factor)
            )

            # First patch.
            patch_range_1[0, dim] = np.round((ori_shape_1[dim] - roi_size_11[dim]) * rand[2, dim])
            patch_range_1[0, dim + 3] = patch_range_1[0, dim] + roi_size_11[dim]
            patch_range_2[0, dim] = np.round((ori_shape_2[dim] - roi_size_21[dim]) * rand[2, dim])
            patch_range_2[0, dim + 3] = patch_range_2[0, dim] + roi_size_21[dim]

            # Second patch (constrained by overlap).
            max_offset_1 = np.round(min(roi_size_11[dim], roi_size_12[dim]) * self.offset_factor)
            max_offset_2 = np.round(min(roi_size_21[dim], roi_size_22[dim]) * self.offset_factor)

            r_1 = [
                patch_range_1[0, dim] - max_offset_1,
                patch_range_1[0, dim + 3] + max_offset_1 - roi_size_12[dim],
            ]
            r_1[0] = max(0, r_1[0])
            r_1[1] = min(ori_shape_1[dim] - roi_size_12[dim], r_1[1])
            patch_range_1[1, dim] = np.round(r_1[0] + (r_1[1] - r_1[0]) * rand[3, dim])
            patch_range_1[1, dim + 3] = patch_range_1[1, dim] + roi_size_12[dim]

            r_2 = [
                patch_range_2[0, dim] - max_offset_2,
                patch_range_2[0, dim + 3] + max_offset_2 - roi_size_22[dim],
            ]
            r_2[0] = max(0, r_2[0])
            r_2[1] = min(ori_shape_2[dim] - roi_size_22[dim], r_2[1])
            patch_range_2[1, dim] = np.round(r_2[0] + (r_2[1] - r_2[0]) * rand[3, dim])
            patch_range_2[1, dim + 3] = patch_range_2[1, dim] + roi_size_22[dim]

        patch_range_1 = np.array([aa.tolist() for aa in patch_range_1]).astype(np.int16)
        patch_range_2 = np.array([aa.tolist() for aa in patch_range_2]).astype(np.int16)

        # Crop patches and resize to roi_size.
        for c in range(2):
            patch_1 = torch.tensor(
                ori_image_1[
                    :,
                    patch_range_1[c, 0] : patch_range_1[c, 3],
                    patch_range_1[c, 1] : patch_range_1[c, 4],
                    patch_range_1[c, 2] : patch_range_1[c, 5],
                ]
            )
            result[c, :, :, :] = F.interpolate(patch_1.unsqueeze(0), self.roi_size, mode="trilinear", align_corners=True)

            patch_2 = torch.tensor(
                ori_image_2[
                    :,
                    patch_range_2[c, 0] : patch_range_2[c, 3],
                    patch_range_2[c, 1] : patch_range_2[c, 4],
                    patch_range_2[c, 2] : patch_range_2[c, 5],
                ]
            )
            result[c + 2, :, :, :] = F.interpolate(
                patch_2.unsqueeze(0), self.roi_size, mode="trilinear", align_corners=True
            )

        d["image"] = result
        d["image_meta_dict_0"]["patch_range"] = patch_range_1
        d["image_meta_dict_1"]["patch_range"] = patch_range_2

        overlap_range_1 = self._get_overlap_range(patch_range_1)
        d["image_meta_dict_0"]["positive_pos"] = self._get_positive_pos(patch_range_1, overlap_range_1)

        overlap_range_2 = self._get_overlap_range(patch_range_2)
        d["image_meta_dict_1"]["positive_pos"] = self._get_positive_pos(patch_range_2, overlap_range_2)

        del d["image_pair_0"]
        del d["image_pair_1"]

        return d


class LoadPreprocessePared(MapTransform):
    """Load a preprocessed pair: two mmap'd .npy volumes and per-image meta dict (.pkl)."""

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Mapping[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.keys:
            image_path_1, image_path_2 = d[key]

            for i, image_path in enumerate([image_path_1, image_path_2]):
                basename = os.path.basename(image_path).split(".")[0]
                meta_path = join(os.path.dirname(image_path), f"{basename}.pkl")

                d[f"image_pair_{i}"] = np.load(image_path, mmap_mode="r")

                meta_key = f"{key}_meta_dict_{i}"
                try:
                    with open(meta_path, "rb") as f:
                        d[meta_key] = pickle.load(f)
                except FileNotFoundError:
                    print(f"[Warning] Metadata file not found: {meta_path}. Using an empty meta_dict for '{key}'.")
                    d[meta_key] = {}

        return d


class DataloaderCorrespondencePair(DataloaderBase):
    def __init__(self, config: dict, inference: bool = False, is_ddp: bool = False):
        self._load_configs(config, inference)

        self.roi_size = self.model_config.get("roi_size")
        if self.roi_size is None:
            raise ValueError("`roi_size` must be defined in the [Model] config section.")

        self.n_pos = self.dataloader_config.get("n_pos", 1024)
        self.overlap_factor = self.dataloader_config.get("overlap_factor", 0.5)
        self.scaling_factor = self.dataloader_config.get("scaling_factor", 0.1)
        self.pixdim = self.dataloader_config.get("pixdim", [1.5, 1.5, 1.5])

        # DataloaderBase will call init_data_list() and init_transforms().
        super().__init__(config, inference, is_ddp)

    def init_data_list(self):
        """Load train/validation/test splits from the JSON file specified by `dataset_json`."""
        json_path = self.dataloader_config.get("dataset_json")
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(
                f"dataset_json not found at path: {json_path}. Please specify a valid path in your config file."
            )

        with open(json_path, "r") as f:
            data_lists = json.load(f)

        self.train_list = data_lists.get("train", [])
        self.val_list = data_lists.get("validation", [])
        self.test_list = data_lists.get("test", [])

        if not self.train_list:
            print("[Warning] Training list is empty.")
        if not self.val_list:
            print("[Warning] Validation list is empty.")

    def _create_dataset(self, data_list: list, transform: callable, split_name: str):
        """Create a dataset instance for a given split."""
        is_main_process = (not self.ddp) or torch.distributed.get_rank() == 0
        if is_main_process:
            print(f"INFO: Using RandomPairDataset with cache_rate: {self.cache_rate}")

        return RandomPairDataset(data_list=data_list, transform=transform)

    def init_transforms(self):
        """Initialize training and validation transforms."""
        self.train_transform = Compose(
            [
                LoadPreprocessePared(keys=["image"]),
                RandSamPatchFourd(
                    keys=["image"],
                    roi_size=self.roi_size,
                    n_pos=self.n_pos,
                    scaling_factor=self.scaling_factor,
                    overlap_factor=self.overlap_factor,
                ),
            ]
        )

        self.val_transform = Compose(
            [
                LoadImaged(keys=["image"], image_only=False),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                Spacingd(keys=["image"], pixdim=self.pixdim, mode="bilinear"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000,
                    a_max=1000,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                ToTensord(keys=["image"]),
            ]
        )

    def get_post_transforms(self, pred_keys=("pred",), image_keys=("image",)) -> Compose:
        return Compose(
            [
                Invertd(
                    keys=pred_keys,
                    transform=self.val_transform,
                    orig_keys=image_keys,
                    meta_keys=[f"{k}_meta_dict" for k in pred_keys],
                    orig_meta_keys=[f"{k}_meta_dict" for k in image_keys],
                    nearest_interp=True,
                    to_tensor=True,
                    device="cpu",
                )
            ]
        )


# ==============================================================================
# Unit test
# ==============================================================================
if __name__ == "__main__":
    import time
    from pathlib import Path

    import nibabel as nib
    import toml

    print("--- Running DataloaderCorrespondencePair Unit Test ---")

    config_path = "/home/mingrui/disk1/projects/20251103_DiffusionCorr/diffusioncorr/configs/AE_IDRI_15mm_ae_maisi.toml"
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Test config file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    config = toml.load(config_path)

    dataloader_name = config["Data"]["dataloader_name"]
    print(f"Instantiating Dataloader: {dataloader_name}...")

    try:
        dataloader_module = DataloaderCorrespondencePair(config=config, inference=False)
    except Exception as e:
        print(f"\n[ERROR] Failed to instantiate dataloader: {e}")
        import traceback

        traceback.print_exc()
        raise

    print("Dataloader instantiated successfully.")

    print("Getting train loader...")
    train_loader = dataloader_module.get_train_loader()
    print(f"Train loader created. Number of workers: {train_loader.num_workers}")

    print("\n--- Iterating through one batch of data ---")
    start_time = time.time()

    try:
        batch_data = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Dataloader is empty. Check your dataset_json file and paths.")
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch a batch from the dataloader: {e}")
        import traceback

        traceback.print_exc()
        raise

    end_time = time.time()

    print(f"\nTime to fetch one batch: {end_time - start_time:.4f} seconds")
    print("Batch loaded successfully. Verifying contents...")

    expected_keys = ["image", "image_meta_dict_0", "image_meta_dict_1"]
    for key in expected_keys:
        assert key in batch_data, f"Missing key '{key}' in the batch."
    print("✔ All expected keys are present.")

    config_data_section = config["Data"][dataloader_name]
    config_batch_size = config_data_section["batch_size"]
    config_roi_size = np.array(config["Model"]["roi_size"])
    config_n_pos = config_data_section["n_pos"]

    image_tensor = batch_data["image"]
    expected_shape = (config_batch_size, 4, *config_roi_size)
    assert image_tensor.shape == expected_shape, (
        f"Incorrect image tensor shape. Expected {expected_shape}, but got {image_tensor.shape}"
    )
    print(f"✔ Image tensor shape is correct: {image_tensor.shape}")

    expected_pos_shape_per_pair = (config_batch_size, config_n_pos, 2, 3)
    for i in range(2):
        meta_dict_key = f"image_meta_dict_{i}"
        meta_dict = batch_data[meta_dict_key]

        assert "positive_pos" in meta_dict, f"Missing 'positive_pos' in {meta_dict_key}."
        positive_pos_tensor = meta_dict["positive_pos"]

        assert positive_pos_tensor.shape == expected_pos_shape_per_pair, (
            f"Incorrect positive_pos tensor shape in {meta_dict_key}. "
            f"Expected {expected_pos_shape_per_pair}, but got {positive_pos_tensor.shape}"
        )
        print(f"✔ Positive points tensor shape is correct for pair {i}: {positive_pos_tensor.shape}")

        assert positive_pos_tensor.min() >= -1.0 and positive_pos_tensor.max() <= 1.0, (
            f"Positive points values are out of the expected [-1, 1] range in {meta_dict_key}."
        )
        print(f"✔ Positive points values are within the [-1, 1] range for pair {i}.")

        assert "patch_range" in meta_dict, f"Missing 'patch_range' in {meta_dict_key}"
        print(f"✔ 'patch_range' is present in {meta_dict_key}")

    print("\n--- Basic Dataloader checks PASSED! ---")

    print("\n--- Performing Visualization Check ---")

    debug_dir = Path("/home/mingrui/disk1/projects/20251103_DiffusionCorr/debug")
    debug_dir.mkdir(exist_ok=True)
    print(f"Saving visualization files to: {debug_dir.resolve()}")

    batch_idx = 0

    patch_0 = image_tensor[batch_idx, 0, ...].cpu().numpy()
    patch_1 = image_tensor[batch_idx, 1, ...].cpu().numpy()
    patch_2 = image_tensor[batch_idx, 2, ...].cpu().numpy()
    patch_3 = image_tensor[batch_idx, 3, ...].cpu().numpy()

    pos_tensor_1 = batch_data["image_meta_dict_0"]["positive_pos"]
    pos_pair_1_relative = pos_tensor_1[batch_idx, 0, ...].cpu().numpy()

    pos_tensor_2 = batch_data["image_meta_dict_1"]["positive_pos"]
    pos_pair_2_relative = pos_tensor_2[batch_idx, 0, ...].cpu().numpy()

    def relative_to_pixel(relative_coords, size):
        return np.round((relative_coords + 1.0) / 2.0 * (size - 1)).astype(int)

    coords_0_pixel = relative_to_pixel(pos_pair_1_relative[0], config_roi_size)
    coords_1_pixel = relative_to_pixel(pos_pair_1_relative[1], config_roi_size)
    coords_2_pixel = relative_to_pixel(pos_pair_2_relative[0], config_roi_size)
    coords_3_pixel = relative_to_pixel(pos_pair_2_relative[1], config_roi_size)

    masks = [np.zeros_like(patch_0, dtype=np.uint8) for _ in range(4)]

    def create_marker(mask, coords, marker_size=3):
        coords = np.clip(coords, marker_size, np.array(mask.shape) - marker_size - 1)
        x, y, z = coords
        mask[x - marker_size : x + marker_size + 1, y - marker_size : y + marker_size + 1, z - marker_size : z + marker_size + 1] = 1
        return mask

    mask_0 = create_marker(masks[0], coords_0_pixel)
    mask_1 = create_marker(masks[1], coords_1_pixel)
    mask_2 = create_marker(masks[2], coords_2_pixel)
    mask_3 = create_marker(masks[3], coords_3_pixel)

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(patch_0, affine), debug_dir / "pair1_patch0.nii.gz")
    nib.save(nib.Nifti1Image(patch_1, affine), debug_dir / "pair1_patch1.nii.gz")
    nib.save(nib.Nifti1Image(patch_2, affine), debug_dir / "pair2_patch2.nii.gz")
    nib.save(nib.Nifti1Image(patch_3, affine), debug_dir / "pair2_patch3.nii.gz")

    nib.save(nib.Nifti1Image(mask_0, affine), debug_dir / "pair1_mask0.nii.gz")
    nib.save(nib.Nifti1Image(mask_1, affine), debug_dir / "pair1_mask1.nii.gz")
    nib.save(nib.Nifti1Image(mask_2, affine), debug_dir / "pair2_mask2.nii.gz")
    nib.save(nib.Nifti1Image(mask_3, affine), debug_dir / "pair2_mask3.nii.gz")

    print("\nSuccessfully saved 4 patches and 4 masks.")
    print("Please verify the correspondences in a 3D viewer.")
    print("\n--- Dataloader unit test with visualization PASSED! ---")