import json
import os
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose, Invertd, MapTransform, ToTensord

from data_loader.dataloader_base import DataloaderBase
from data_loader.dataloader_correspondence import LoadPreprocessed
from utility import *

# ==============================================================================
# Transforms
# ==============================================================================


class LoadPointCloudNumpyd(MapTransform):
    """Load a point cloud from a text file and convert it to a float32 torch Tensor."""

    def __init__(self, keys, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            point_cloud = np.loadtxt(d[key])
            d[key] = torch.from_numpy(point_cloud).float()
        return d


# ==============================================================================
# Dataloader
# ==============================================================================


class DataloaderCorrLandmark(DataloaderBase):
    def __init__(self, config: dict, inference: bool = False, is_ddp: bool = False):
        self._load_configs(config, inference)

        self.roi_size = self.model_config.get("roi_size")
        if self.roi_size is None:
            raise ValueError("`roi_size` must be defined in the [Model] config section.")

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

        self.train_list = data_lists.get("train", [])
        self.val_list = data_lists.get("validation", [])
        self.test_list = data_lists.get("test", [])

        if not self.test_list:
            print("[Warning] Test list is empty.")

    def init_transforms(self):
        """Initialize transforms for (inference) landmark correspondence evaluation."""
        self.val_transform = Compose(
            [
                LoadPreprocessed(keys=["image_1", "image_2"]),
                LoadPointCloudNumpyd(keys=["landmark_1", "landmark_2"]),
                ToTensord(keys=["image_1", "image_2"], allow_missing_keys=True),
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
    import nibabel as nib
    import time
    import toml

    print("--- Running DataloaderCorrLandmark Unit Test ---")

    config_path = "/home/mingrui/disk1/projects/20251103_DiffusionCorr/diffusioncorr/configs/AE_IDRI_15mm_ae_maisi.toml"
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"Test config file not found at: {config_path}")

    print(f"Loading configuration from: {config_path}")
    config = toml.load(config_path)

    print("Instantiating DataloaderCorrLandmark...")
    try:
        dataloader_module = DataloaderCorrLandmark(config=config, inference=True)
    except Exception as e:
        raise RuntimeError(f"Failed to instantiate dataloader: {e}")

    print("Dataloader instantiated successfully.")

    print("Getting test loader...")
    test_loader = dataloader_module.get_test_loader()
    print(f"Test loader created. Number of workers: {test_loader.num_workers}")

    print("\n--- Iterating through one batch of data ---")
    start_time = time.time()
    try:
        batch_data = next(iter(test_loader))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch a batch from the dataloader: {e}")
    end_time = time.time()

    print(f"\nTime to fetch one batch: {end_time - start_time:.4f} seconds")
    print("Batch loaded successfully. Verifying contents...")

    expected_keys = ["image_1", "image_2", "landmark_1", "landmark_2"]
    for key in expected_keys:
        assert key in batch_data, f"Missing key '{key}' in the batch."
    print("✔ All expected keys are present.")

    print("\n--- Performing Visualization Check ---")

    debug_dir = Path("/home/mingrui/disk1/projects/20251103_DiffusionCorr/debug")
    debug_dir.mkdir(exist_ok=True)
    print(f"Saving visualization files to: {debug_dir.resolve()}")

    image_1 = batch_data["image_1"].numpy()
    image_2 = batch_data["image_2"].numpy()
    landmark_1 = batch_data["landmark_1"].numpy()
    landmark_2 = batch_data["landmark_2"].numpy()

    def visualize(img_arr, landmarks_vox, base_name: str):
        """Save the image and a landmark label map (small cubes) as NIfTI."""
        cube_size = 3
        img_dims = img_arr.shape[2:]
        img_affine = np.eye(4)

        label_map = np.zeros(img_dims, dtype=np.uint8)

        if landmarks_vox.size != 0:
            offset = cube_size // 2
            for point in landmarks_vox:
                center = np.round(point).astype(int)

                z_start = max(0, center[2] - offset)
                z_end = min(img_dims[2], center[2] + offset + 1)

                y_start = max(0, center[1] - offset)
                y_end = min(img_dims[1], center[1] + offset + 1)

                x_start = max(0, center[0] - offset)
                x_end = min(img_dims[0], center[0] + offset + 1)

                label_map[x_start:x_end, y_start:y_end, z_start:z_end] = 1

        label_map_nib = nib.Nifti1Image(label_map, img_affine)
        label_path = debug_dir / f"{base_name}_landmarks.nii.gz"
        nib.save(label_map_nib, str(label_path))
        print(f"  -> Saved label map: {label_path}")

        image_nib = nib.Nifti1Image(img_arr[0, 0, :], np.eye(4))
        image_path = debug_dir / f"{base_name}_image.nii.gz"
        nib.save(image_nib, str(image_path))
        print(f"  -> Saved image: {image_path}")

    fullname = batch_data["image_meta_dict"]["filename_or_obj"][0]
    filename = os.path.basename(fullname)

    visualize(image_1, landmark_1[0], filename[:-13] + "_00")
    visualize(image_2, landmark_2[0], filename[:-13] + "_50")