import json
import os
import pickle
from typing import Dict, Hashable, Mapping, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    Invertd,
    InvertibleTransform,
    MapTransform,
    Randomizable,
)
from data_loader.dataloader_base import DataloaderBase
from utility import *

# ==============================================================================
# Custom MONAI transforms
# ==============================================================================


class LoadPreprocessed(MapTransform):
    """Load a preprocessed volume (.npy, mmap) and its metadata (.pkl)."""

    def __init__(self, keys: KeysCollection) -> None:
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        basename = os.path.basename(d[self.first_key(d)]).split(".")[0]
        meta_path = join(os.path.dirname(d[self.first_key(d)]), f"{basename}.pkl")

        with open(meta_path, "rb") as f:
            d["image_meta_dict"] = pickle.load(f)

        for key in self.keys:
            d[key] = np.load(d[key], mmap_mode="r")

        return d


class RandSamPatchPaird(MapTransform, InvertibleTransform, Randomizable):
    """Sample two overlapping 3D patches from a single volume and generate positive point pairs."""

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

        positive_pos = np.zeros((self.n_pos, 2, 3), dtype=np.float32)
        positive_pos[:, 0, :] = (p - patch_range[:1, :3]) / (patch_range[:1, 3:] - patch_range[:1, :3])
        positive_pos[:, 1, :] = (p - patch_range[1:, :3]) / (patch_range[1:, 3:] - patch_range[1:, :3])
        positive_pos = positive_pos * 2 - 1

        return positive_pos

    def __call__(self, data):
        d = dict(data)
        assert len(self.keys) == 1, "error keys form RandSamPatchPaird."

        ori_image = d[self.keys[0]]
        numpy_mode = isinstance(ori_image, np.ndarray)
        if numpy_mode:
            ori_shape = ori_image.shape
        else:
            ori_shape = ori_image.size()

        assert ori_shape[0] == 1, "error channel form RandSamPatchPaird."
        ori_shape = ori_shape[1:]

        result = torch.zeros([2] + self.roi_size)

        roi_size_1 = np.array(self.roi_size)
        roi_size_2 = np.array(self.roi_size)

        rand = torch.rand(4, 3)
        patch_range = torch.zeros(2, 6)

        for dim in range(3):
            roi_size_1[dim] = torch.round(
                self.roi_size[dim] * (1 + rand[0, dim] * 2 * self.scaling_factor - self.scaling_factor)
            )
            roi_size_2[dim] = torch.round(
                self.roi_size[dim] * (1 + rand[1, dim] * 2 * self.scaling_factor - self.scaling_factor)
            )

            max_offset = np.round(min(roi_size_1[dim], roi_size_2[dim]) * self.offset_factor)

            patch_range[0, dim] = np.round((ori_shape[dim] - roi_size_1[dim]) * rand[2, dim])
            patch_range[0, dim + 3] = patch_range[0, dim] + roi_size_1[dim]

            r = [
                patch_range[0, dim] - max_offset,
                patch_range[0, dim + 3] + max_offset - roi_size_2[dim],
            ]
            r[0] = max(0, r[0])
            r[1] = min(ori_shape[dim] - roi_size_2[dim], r[1])

            patch_range[1, dim] = np.round(r[0] + (r[1] - r[0]) * rand[3, dim])
            patch_range[1, dim + 3] = patch_range[1, dim] + roi_size_2[dim]

        patch_range = np.array([aa.tolist() for aa in patch_range]).astype(np.int16)

        for c in range(2):
            patch = torch.tensor(
                ori_image[
                    :,
                    patch_range[c, 0] : patch_range[c, 3],
                    patch_range[c, 1] : patch_range[c, 4],
                    patch_range[c, 2] : patch_range[c, 5],
                ]
            )
            result[c, :, :, :] = F.interpolate(patch.unsqueeze(0), self.roi_size, mode="trilinear", align_corners=True)

        d["image"] = result
        d["image_meta_dict"]["patch_range"] = patch_range

        overlap_range = self._get_overlap_range(patch_range)
        d["image_meta_dict"]["positive_pos"] = self._get_positive_pos(patch_range, overlap_range)

        return d


class RandIntensityAugPaird(MapTransform, Randomizable):
    """
    Optional intensity augmentation for paired patches in d['image'].

    Expected shape:
        d[key] = [2, D, H, W]

    Supports:
        scale, shift, gamma, and optional Gaussian noise.

    If `same_on_pair=True`, both patches share the same augmentation parameters.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        same_on_pair: bool = True,
        scale_range: float = 0.10,
        shift_range: float = 0.10,
        gamma_range: float = 0.20,
        noise_std: float = 0.00,
        clip: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.prob = float(prob)
        self.same_on_pair = bool(same_on_pair)
        self.scale_range = float(scale_range)
        self.shift_range = float(shift_range)
        self.gamma_range = float(gamma_range)
        self.noise_std = float(noise_std)
        self.clip = bool(clip)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self._do = False
        self._s0 = self._b0 = self._g0 = None
        self._s1 = self._b1 = self._g1 = None

    def randomize(self, data=None):
        self._do = self.R.rand() < self.prob

        def sample_params():
            s = 1.0 + (2.0 * self.R.rand() - 1.0) * self.scale_range
            b = (2.0 * self.R.rand() - 1.0) * self.shift_range
            g = 1.0 + (2.0 * self.R.rand() - 1.0) * self.gamma_range
            return float(s), float(b), float(g)

        if not self._do:
            self._s0 = self._b0 = self._g0 = None
            self._s1 = self._b1 = self._g1 = None
            return

        s, b, g = sample_params()
        self._s0, self._b0, self._g0 = s, b, g

        if self.same_on_pair:
            self._s1, self._b1, self._g1 = s, b, g
        else:
            s, b, g = sample_params()
            self._s1, self._b1, self._g1 = s, b, g

    def _apply_one(self, x: torch.Tensor, s: float, b: float, g: float):
        y = x * s + b
        y = torch.clamp(y, min=0.0)  # gamma requires non-negative values
        y = y.pow(g)

        if self.noise_std > 0:
            y = y + torch.randn_like(y) * float(self.noise_std)

        if self.clip:
            y = torch.clamp(y, float(self.clip_min), float(self.clip_max))

        return y

    def __call__(self, data):
        d = dict(data)
        key = self.keys[0]

        if key not in d:
            return d

        img = d[key]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)

        if img.ndim != 4 or img.shape[0] != 2:
            raise ValueError(f"RandIntensityAugPaird expects image shape [2,D,H,W], got {tuple(img.shape)}")

        self.randomize(None)
        if not self._do:
            d[key] = img
            return d

        out0 = self._apply_one(img[0], self._s0, self._b0, self._g0)
        out1 = self._apply_one(img[1], self._s1, self._b1, self._g1)

        d[key] = torch.stack([out0, out1], dim=0)
        return d


# ==============================================================================
# Dataloader
# ==============================================================================


class DataloaderCorrespondence(DataloaderBase):
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
        """Load splits from the configured JSON file."""
        json_path = self.dataloader_config.get("dataset_json")
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(
                f"dataset_json not found at path: {json_path}. Please specify a valid path in your config file."
            )

        with open(json_path, "r") as f:
            data_lists = json.load(f)

        # Keep the original behavior.
        self.train_list = data_lists.get("train", [])
        self.val_list = data_lists.get("train", [])
        self.test_list = data_lists.get("train", [])

        if not self.train_list:
            print("[Warning] Training list is empty.")
        if not self.val_list:
            print("[Warning] Validation list is empty.")

    def init_transforms(self):
        """Initialize transforms (and optional intensity augmentation)."""
        aug_cfg = self.dataloader_config.get("intensity_aug", {})
        enable_aug = bool(aug_cfg.get("enable", False))

        tfs = [
            LoadPreprocessed(keys=["image"]),
            RandSamPatchPaird(
                keys=["image"],
                roi_size=self.roi_size,
                n_pos=self.n_pos,
                scaling_factor=self.scaling_factor,
                overlap_factor=self.overlap_factor,
            ),
        ]

        if enable_aug:
            tfs.append(
                RandIntensityAugPaird(
                    keys=["image"],
                    prob=float(aug_cfg.get("prob", 0.5)),
                    same_on_pair=bool(aug_cfg.get("same_on_pair", True)),
                    scale_range=float(aug_cfg.get("scale_range", 0.10)),
                    shift_range=float(aug_cfg.get("shift_range", 0.10)),
                    gamma_range=float(aug_cfg.get("gamma_range", 0.20)),
                    noise_std=float(aug_cfg.get("noise_std", 0.00)),
                    clip=bool(aug_cfg.get("clip", True)),
                    clip_min=float(aug_cfg.get("clip_min", 0.0)),
                    clip_max=float(aug_cfg.get("clip_max", 1.0)),
                )
            )

        self.train_transform = Compose(tfs)
        self.val_transform = self.train_transform

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

    print("--- Running DataloaderCorrespondence Unit Test ---")

    config_path = "/home/mingrui/disk1/projects/20251103_DiffusionCorr/diffusioncorr/configs/AE_1mm_local_3.toml"
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Test config file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    config = toml.load(config_path)

    print("Instantiating DataloaderCorrespondence...")
    try:
        dataloader_module = DataloaderCorrespondence(config=config, inference=False)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate dataloader: {e}")

    print("Dataloader instantiated successfully.")

    print("Getting train loader...")
    train_loader = dataloader_module.get_train_loader()
    print(f"Train loader created. Number of workers: {train_loader.num_workers}")

    print("\n--- Iterating through one batch of data ---")
    start_time = time.time()

    try:
        batch_data = next(iter(train_loader))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch a batch from the dataloader: {e}")

    end_time = time.time()

    print(f"\nTime to fetch one batch: {end_time - start_time:.4f} seconds")
    print("Batch loaded successfully. Verifying contents...")

    expected_keys = ["image", "image_meta_dict"]
    for key in expected_keys:
        assert key in batch_data, f"Missing key '{key}' in the batch."
    print("✔ All expected keys are present.")

    image_tensor = batch_data["image"]
    config_batch_size = config["Data"]["DataloaderCorrespondence"]["batch_size"]
    config_roi_size = np.array(config["Model"]["roi_size"])

    expected_shape = (config_batch_size, 2, *config_roi_size)
    assert image_tensor.shape == expected_shape, (
        f"Incorrect image tensor shape. Expected {expected_shape}, but got {image_tensor.shape}"
    )
    print(f"✔ Image tensor shape is correct: {image_tensor.shape}")

    meta_dict = batch_data["image_meta_dict"]
    assert "positive_pos" in meta_dict, "Missing 'positive_pos' in meta_dict."

    positive_pos_tensor = meta_dict["positive_pos"]
    config_n_pos = config["Data"]["DataloaderCorrespondence"]["n_pos"]

    expected_pos_shape = (config_batch_size, config_n_pos, 2, 3)
    assert positive_pos_tensor.shape == expected_pos_shape, (
        f"Incorrect positive_pos tensor shape. Expected {expected_pos_shape}, but got {positive_pos_tensor.shape}"
    )
    print(f"✔ Positive points tensor shape is correct: {positive_pos_tensor.shape}")

    assert positive_pos_tensor.min() >= -1.0 and positive_pos_tensor.max() <= 1.0, (
        "Positive points values are out of the expected [-1, 1] range."
    )
    print("✔ Positive points values are within the [-1, 1] range.")

    print("\n--- Basic Dataloader checks PASSED! ---")

    print("\n--- Performing Visualization Check ---")

    debug_dir = Path("/home/mingrui/disk1/projects/20251103_DiffusionCorr/debug")
    debug_dir.mkdir(exist_ok=True)
    print(f"Saving visualization files to: {debug_dir.resolve()}")

    batch_idx = 0

    patch_0 = image_tensor[batch_idx, 0, ...].cpu().numpy()
    patch_1 = image_tensor[batch_idx, 1, ...].cpu().numpy()

    first_pos_pair_relative = positive_pos_tensor[batch_idx, 0, ...].cpu().numpy()

    coords_0_pixel = np.round((first_pos_pair_relative[0] + 1.0) / 2.0 * (config_roi_size - 1)).astype(int)
    coords_1_pixel = np.round((first_pos_pair_relative[1] + 1.0) / 2.0 * (config_roi_size - 1)).astype(int)

    print(f"Patch 0 - Relative Coords: {first_pos_pair_relative[0]}")
    print(f"Patch 0 - Pixel Coords (H,W,D): {coords_0_pixel}")
    print(f"Patch 1 - Relative Coords: {first_pos_pair_relative[1]}")
    print(f"Patch 1 - Pixel Coords (H,W,D): {coords_1_pixel}")

    mask_0 = np.zeros_like(patch_0, dtype=np.uint8)
    mask_1 = np.zeros_like(patch_1, dtype=np.uint8)

    marker_size = 1

    h, w, d = coords_0_pixel
    mask_0[h - marker_size : h + marker_size + 1, w - marker_size : w + marker_size + 1, d - marker_size : d + marker_size + 1] = 1

    h, w, d = coords_1_pixel
    mask_1[h - marker_size : h + marker_size + 1, w - marker_size : w + marker_size + 1, d - marker_size : d + marker_size + 1] = 1

    affine = np.eye(4)

    nib.save(nib.Nifti1Image(patch_0, affine), debug_dir / "patch_0.nii.gz")
    nib.save(nib.Nifti1Image(patch_1, affine), debug_dir / "patch_1.nii.gz")
    nib.save(nib.Nifti1Image(mask_0, affine), debug_dir / "mask_0.nii.gz")
    nib.save(nib.Nifti1Image(mask_1, affine), debug_dir / "mask_1.nii.gz")

    print("\nSuccessfully saved patch_0, patch_1, mask_0, mask_1.")
    print("Open these files in ITK-SNAP or 3D Slicer to verify correspondences.")
    print("\n--- Dataloader unit test with visualization PASSED! ---")