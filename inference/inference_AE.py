import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.networks.utils import one_hot
from tqdm import tqdm

from .inference_base import InferenceBase


class inference_AE(InferenceBase):
    """
    Inference + evaluation pipeline for correspondence / registration-style models.

    Features:
      - Run inference on each test pair.
      - Compute TRE (and optionally other metrics kept for compatibility).
      - Save per-landmark visualizations (moving/fixed GT, fixed prediction + heatmap).
      - Export per-case landmark results to Excel and a summary spreadsheet.
    """

    def __init__(self, config: dict):
        """Initialize the inference runner and bookkeeping variables."""
        super().__init__(config)

        # Metric accumulators / logs (kept for compatibility with existing workflow).
        self.total_tre = 0.0
        self.results_log_console = []
        self.results_data_excel = []

        # Sequential case id used by some pipelines.
        self.case_counter = 1

        # For segmentation-style metrics (not always used in this class).
        self.num_classes = self.model_config.get("num_classes", 2)

        self.roi_size = np.array(self.inference_config.get("roi_size"))
        self.spacing = self.inference_config.get("spacing")

    @staticmethod
    def _calculate_tre(pred_kpts: torch.Tensor, target_kpts: torch.Tensor, vx: torch.Tensor) -> float:
        """Compute TRE in physical space (mm) using voxel spacing."""
        diff = (pred_kpts - target_kpts) * vx
        return diff.pow(2).sum(dim=-1).sqrt()

    @staticmethod
    def _floor_to_multiple(x: int, m: int = 16) -> int:
        """Floor an integer to the nearest lower multiple of `m`."""
        return (int(x) // m) * m

    def _adaptive_roi_size(self, shape1, shape2, base_roi, multiple: int = 16):
        """
        Compute an ROI size that:
          - does not exceed `base_roi`,
          - fits within both inputs,
          - and is floored to a per-dimension multiple.

        Returns:
            np.ndarray[int] or None: ROI size, or None if any dim becomes too small.
        """
        shape1 = np.array(shape1, dtype=np.int64)
        shape2 = np.array(shape2, dtype=np.int64)
        base_roi = np.array(base_roi, dtype=np.int64)

        max_allow = np.minimum(shape1, shape2)
        roi = np.minimum(base_roi, max_allow)
        roi = np.array([self._floor_to_multiple(v, multiple) for v in roi], dtype=np.int64)

        if np.any(roi < multiple):
            return None
        return roi

    def _forward_with_model(self, x, roi_size):
        """Forward the model and upsample correspondence features to `roi_size`."""
        with torch.amp.autocast(device_type=self.device.type):
            y = self.model.inference(x)

        roi_size_tuple = tuple(int(v) for v in roi_size.tolist())
        local_feat_up = F.interpolate(
            y["correspondence_output"],
            size=roi_size_tuple,
            mode="trilinear",
            align_corners=True,
        )
        return local_feat_up

    def _get_correspondence_points(
        self,
        reference_array,  # moving image, torch.Tensor (D,H,W) on GPU
        reference_points,  # moving GT landmarks, np.ndarray (N,3) in (z,y,x)
        target_array,  # fixed image, torch.Tensor (D,H,W) on GPU
        target_points_gt,  # fixed GT landmarks, np.ndarray (N,3) in (z,y,x)
        data_name,
        roi_size,
    ):
        """
        Predict correspondences on `target_array` (fixed) for GT landmarks in `reference_points` (moving).

        For each landmark, we save:
          1) Moving image slice with GT point
          2) Fixed image slice with GT point
          3) Fixed image slice with predicted point and aligned similarity heatmap
        """

        def _save_slice_with_point(vol3d_t: torch.Tensor, point_zyx, out_path: str, cmap="gray"):
            """Save an x-slice (z,y plane) with a single point overlay (display: rot90 + flip)."""
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            vol = vol3d_t.detach().cpu().numpy()
            z, y, x = [int(round(v)) for v in point_zyx]

            z = int(np.clip(z, 0, vol.shape[0] - 1))
            y = int(np.clip(y, 0, vol.shape[1] - 1))
            x = int(np.clip(x, 0, vol.shape[2] - 1))

            img2d = vol[:, :, x]
            img2d_disp = np.flip(np.rot90(img2d), 0)

            # After rot90+flip: (z,y) maps to (row=y, col=z).
            rr, cc = y, z

            plt.figure(figsize=(5, 5))
            plt.imshow(img2d_disp, cmap=cmap)
            plt.scatter([cc], [rr], s=35)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close()

        def _save_slice_with_point_and_heatmap_aligned(
            vol3d_t: torch.Tensor,  # (D,H,W) fixed image, z,y,x
            point_zyx,  # (z,y,x) in fixed global coords
            sim_patch_t: torch.Tensor,  # (1,1,Dp,Hp,Wp) similarity map in patch coords
            patch_origin_zyx,  # (z0,y0,x0) patch origin in fixed global coords
            out_path: str,
            img_cmap="gray",
            heat_cmap="hot",
            alpha_max=0.45,
            thr=0.35,
            gamma=1.0,
        ):
            """Overlay a patch similarity heatmap back onto the full fixed x-slice and save."""
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            vol = vol3d_t.detach().cpu().numpy()
            D, H, W = vol.shape

            z, y, x = [int(round(v)) for v in point_zyx]
            z = int(np.clip(z, 0, D - 1))
            y = int(np.clip(y, 0, H - 1))
            x = int(np.clip(x, 0, W - 1))

            img2d = vol[:, :, x]
            img2d_disp = np.flip(np.rot90(img2d), 0)

            rr, cc = y, z

            # Build a full-slice heat canvas in (D,H) (z,y) space.
            heat_full = np.zeros((D, H), dtype=np.float32)

            z0, y0, x0 = [int(v) for v in patch_origin_zyx.tolist()]
            sim = sim_patch_t.detach().float().cpu().numpy()[0, 0]  # (Dp,Hp,Wp)
            Dp, Hp, Wp = sim.shape

            x_in_patch = x - x0
            if 0 <= x_in_patch < Wp:
                heat_patch_2d = sim[:, :, x_in_patch]  # (Dp,Hp)
                z1 = min(z0 + Dp, D)
                y1 = min(y0 + Hp, H)
                dz = z1 - z0
                dy = y1 - y0
                if dz > 0 and dy > 0:
                    heat_full[z0:z1, y0:y1] = heat_patch_2d[:dz, :dy]

            heat_disp = np.flip(np.rot90(heat_full), 0)

            hmin, hmax = float(np.min(heat_disp)), float(np.max(heat_disp))
            if hmax > hmin:
                h = (heat_disp - hmin) / (hmax - hmin + 1e-8)
            else:
                h = np.zeros_like(heat_disp, dtype=np.float32)

            if gamma != 1.0:
                h = np.power(h, gamma)

            alpha = (h - float(thr)) / (1.0 - float(thr) + 1e-8)
            alpha = np.clip(alpha, 0.0, 1.0) * float(alpha_max)

            plt.figure(figsize=(5, 5))
            plt.imshow(img2d_disp, cmap=img_cmap)
            plt.imshow(heat_disp, cmap=heat_cmap, alpha=alpha)
            plt.scatter([cc], [rr], s=35)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close()

        # ---------------------------------------------------------------------
        # 1) Extract a per-landmark feature vector on the moving image.
        # ---------------------------------------------------------------------
        vector_dict = {}
        for i, landmark in enumerate(reference_points):
            seed_pos = np.round(landmark).astype(np.int32)  # (z,y,x)

            pos_r = torch.round(torch.tensor(seed_pos) - 0.5 * torch.tensor(roi_size))
            pos_r = torch.maximum(pos_r, torch.tensor([0, 0, 0]))
            pos_r = torch.minimum(pos_r, torch.tensor(reference_array.shape) - torch.tensor(roi_size))
            pos_r = pos_r.int()

            val_input = reference_array[
                pos_r[0] : pos_r[0] + roi_size[0],
                pos_r[1] : pos_r[1] + roi_size[1],
                pos_r[2] : pos_r[2] + roi_size[2],
            ].to(self.device)

            val_output = self._forward_with_model(val_input.unsqueeze(0).unsqueeze(0), roi_size=roi_size)

            offset_l = torch.tensor(seed_pos) - pos_r
            local_vector = val_output[0, :, offset_l[0], offset_l[1], offset_l[2]]  # (C,)
            vector_dict[str(i)] = {"local_vector": local_vector}

        num_landmarks = len(vector_dict)
        max_sim = np.zeros((num_landmarks), dtype=np.float32)
        max_sim_pos = np.zeros((num_landmarks, 3), dtype=np.float32)

        # Shape: (N,C,1,1,1) for dot-product similarity with local_feature.
        local_vector = torch.cat([vector_dict[str(i)]["local_vector"].unsqueeze(0) for i in range(num_landmarks)], dim=0)
        local_vector = local_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # ---------------------------------------------------------------------
        # 2) Sliding-window coarse search on the fixed image.
        # ---------------------------------------------------------------------
        patch_pos = -roi_size.copy()
        flag = [True, True, True]
        image_shape = target_array.shape

        while flag[0]:
            patch_pos[0] = patch_pos[0] + roi_size[0]
            if patch_pos[0] + roi_size[0] > image_shape[0]:
                patch_pos[0] = image_shape[0] - roi_size[0]
                flag[0] = False
            flag[1] = True

            while flag[1]:
                patch_pos[1] = patch_pos[1] + roi_size[1]
                if patch_pos[1] + roi_size[1] > image_shape[1]:
                    patch_pos[1] = image_shape[1] - roi_size[1]
                    flag[1] = False
                flag[2] = True

                while flag[2]:
                    patch_pos[2] = patch_pos[2] + roi_size[2]
                    if patch_pos[2] + roi_size[2] > image_shape[2]:
                        patch_pos[2] = image_shape[2] - roi_size[2]
                        flag[2] = False

                    val_input = target_array[
                        patch_pos[0] : patch_pos[0] + roi_size[0],
                        patch_pos[1] : patch_pos[1] + roi_size[1],
                        patch_pos[2] : patch_pos[2] + roi_size[2],
                    ].to(self.device)

                    local_feature = self._forward_with_model(val_input.unsqueeze(0).unsqueeze(0), roi_size=roi_size)

                    for i in range(num_landmarks):
                        sim_total = torch.sum(local_feature * local_vector[i : i + 1], dim=1)  # (1,D,H,W)
                        cur_max = torch.max(sim_total).item()

                        if cur_max > float(max_sim[i]):
                            max_sim[i] = cur_max

                            pos = torch.nonzero(sim_total == torch.max(sim_total))
                            pos = pos[0, 1:]  # inside patch
                            pos = pos + 0.5
                            max_sim_pos[i] = patch_pos + pos.detach().cpu().numpy()

                patch_pos[2] = -roi_size[2]
            patch_pos[1] = -roi_size[1]
        patch_pos[0] = -roi_size[0]

        max_sim_pos = torch.tensor(max_sim_pos)

        # ---------------------------------------------------------------------
        # 3) Local refinement around coarse maxima + visualization export.
        # ---------------------------------------------------------------------
        result_dict = []
        for i, landmark in enumerate(reference_points):
            pos_r = torch.round(max_sim_pos[i] - 0.5 * torch.tensor(roi_size))
            pos_r = torch.maximum(pos_r, torch.tensor([0, 0, 0]))
            pos_r = torch.minimum(pos_r, torch.tensor(target_array.shape) - torch.tensor(roi_size))
            pos_r = pos_r.int()

            val_input = target_array[
                pos_r[0] : pos_r[0] + roi_size[0],
                pos_r[1] : pos_r[1] + roi_size[1],
                pos_r[2] : pos_r[2] + roi_size[2],
            ].to(self.device)

            local_feature = self._forward_with_model(val_input.unsqueeze(0).unsqueeze(0), roi_size=roi_size)

            sim_local = torch.sum(local_feature * local_vector[i : i + 1], dim=1).unsqueeze(0)  # (1,1,D,H,W)
            sim_map = sim_local

            pos = torch.nonzero(sim_map == torch.max(sim_map))
            pos = pos[0, 2:].to("cpu")  # (z,y,x) inside patch
            pos = pos + pos_r  # global fixed coords
            pred_zyx = pos.tolist()
            result_dict.append(pred_zyx)

            moving_gt_zyx = np.round(landmark).astype(np.int32).tolist()
            fixed_gt_zyx = np.round(target_points_gt[i]).astype(np.int32).tolist()

            base = f"{data_name[:-7]}_lm{i:03d}"
            _save_slice_with_point(
                reference_array,
                moving_gt_zyx,
                os.path.join(self.output_dir, "png", f"{base}_moving_GT.png"),
            )
            _save_slice_with_point(
                target_array,
                fixed_gt_zyx,
                os.path.join(self.output_dir, "png", f"{base}_fixed_GT.png"),
            )
            _save_slice_with_point_and_heatmap_aligned(
                target_array,
                pred_zyx,
                sim_map,
                pos_r,
                os.path.join(self.output_dir, "png", f"{base}_fixed_PRED_heat.png"),
                img_cmap="gray",
                heat_cmap="hot",
                alpha_max=0.45,
                thr=0.35,
                gamma=1.5,
            )

        return result_dict

    def predict(self, batch_data: dict, data_name, roi_size):
        """Run correspondence prediction for a single case and export per-case Excel."""
        fixed_image = batch_data["image_1"].to(self.device)
        fixed_keypoints = batch_data["landmark_1"].numpy()
        moving_image = batch_data["image_2"].to(self.device)
        moving_keypoints = batch_data["landmark_2"].numpy()

        if moving_keypoints.ndim == 1:
            moving_keypoints = moving_keypoints[np.newaxis, :]
        if fixed_keypoints.ndim == 1:
            fixed_keypoints = fixed_keypoints[np.newaxis, :]

        group_size = min(moving_keypoints.shape[0], 50)

        moved_keypoints = []
        for i in range(0, len(moving_keypoints), group_size):
            keypoints_batch = moving_keypoints[i : i + group_size]
            fixed_batch = fixed_keypoints[i : i + group_size]
            moved_keypoints_batch = self._get_correspondence_points(
                moving_image,
                keypoints_batch,
                fixed_image,
                fixed_batch,
                data_name,
                roi_size=roi_size,
            )
            moved_keypoints.extend(moved_keypoints_batch)

        moved_keypoints = np.array(moved_keypoints)
        tre = self._calculate_tre(
            torch.tensor(fixed_keypoints),
            torch.tensor(moved_keypoints),
            vx=self.spacing[0],
        )

        dict_out = {}
        array_out = np.concatenate((moving_keypoints, moved_keypoints, fixed_keypoints), axis=1)
        for i, title in enumerate(
            [
                "moving: X",
                "moving: Y",
                "moving: Z",
                "moved: X",
                "moved: Y",
                "moved: Z",
                "fixed: X",
                "fixed: Y",
                "fixed: Z",
            ]
        ):
            dict_out[title] = array_out[:, i]

        df = pd.DataFrame(dict_out)
        excel_path = os.path.join(self.output_dir, data_name[:-7] + ".xlsx")
        df.to_excel(excel_path, index=False)

        self.case_counter += 1
        return tre

    def run(self):
        """Run inference on the full test set and export a summary spreadsheet."""
        self._setup_components()
        print(f"\n--- Running inference for task '{self.task_config['task_name']}' ---")
        print(f"Outputs will be saved to: {self.output_dir}")
        os.makedirs(os.path.join(self.output_dir, "png"), exist_ok=True)

        tre_results = {}
        with torch.no_grad():
            for batch_data in tqdm(self.dataloader, desc="Inference"):
                batch_data = decollate_batch(batch_data)
                for one_val_data in batch_data:
                    data_name = os.path.basename(one_val_data["image_meta_dict"]["filename_or_obj"])
                    case_name = data_name[:-7]

                    if one_val_data["image_1"].shape[0] == 1:
                        one_val_data["image_1"] = one_val_data["image_1"][0, :]
                    if one_val_data["image_2"].shape[0] == 1:
                        one_val_data["image_2"] = one_val_data["image_2"][0, :]

                    base_roi = self.roi_size
                    roi_size_case = self._adaptive_roi_size(
                        one_val_data["image_1"].shape,
                        one_val_data["image_2"].shape,
                        base_roi,
                        multiple=16,
                    )

                    if roi_size_case is None:
                        print(
                            f"Skip {case_name}: image too small for ROI multiple. "
                            f"image_1={tuple(one_val_data['image_1'].shape)}, "
                            f"image_2={tuple(one_val_data['image_2'].shape)}"
                        )
                        continue

                    tre = self.predict(one_val_data, data_name, roi_size=roi_size_case)
                    tre_results[data_name[:-7]] = tre.numpy()

        df = pd.DataFrame(tre_results)
        excel_path = os.path.join(self.output_dir, "inference_summary.xlsx")
        df.to_excel(excel_path, index=False)

        print(f"Saved summary Excel to: {excel_path}")