import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from tqdm import tqdm

from .inference_base import InferenceBase


class inference_local(InferenceBase):
    def __init__(self, config: dict):
        super().__init__(config)

        self.roi_size = np.array(self.inference_config.get("roi_size"))
        self.spacing = self.inference_config.get("spacing")  # scalar or [3] in (z, y, x)

        self.search_radius = int(self.inference_config.get("search_radius", 16))
        self.match_tau = float(self.inference_config.get("match_tau", 0.02))
        self.refine_radius = int(self.inference_config.get("refine_radius", 2))
        self.corse_path = str(self.inference_config.get("corse_path", ""))

        # Default success threshold in voxels (e.g., < 10 voxels).
        self.ok_thresh_vx = float(self.inference_config.get("ok_thresh_vx", 10.0)) / 1.0

        self.case_counter = 1

    # -------------------- helpers --------------------
    def _save_nii(self, path, vol_zyx, dtype=None, affine=None):
        """Save a 3D volume in (z, y, x) order as NIfTI."""
        import nibabel as nib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if affine is None:
            affine = np.eye(4)

        arr = np.asarray(vol_zyx)
        if dtype is not None:
            arr = arr.astype(dtype)

        nib.save(nib.Nifti1Image(arr, affine), path)

    def _make_point_mask(self, shape_zyx, p_zyx, radius=2, value=1, mask=None):
        """
        Create a small cubic marker around a point in a label map.

        Args:
            shape_zyx: Volume shape (D, H, W).
            p_zyx: Point coordinate (z, y, x) in voxel space.
            radius: Half size of the cube marker.
            value: Label value to write (e.g., 1=GT, 2=Pred, 3=Coarse).
            mask: Optional existing mask to draw on.

        Note:
            If markers overlap, later writes overwrite earlier values.
        """
        D, H, W = shape_zyx
        p = np.round(np.asarray(p_zyx)).astype(np.int32)
        z, y, x = int(p[0]), int(p[1]), int(p[2])

        z = int(np.clip(z, 0, D - 1))
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))

        if mask is None:
            mask = np.zeros((D, H, W), dtype=np.uint8)

        r = int(max(0, radius))
        z0, z1 = max(0, z - r), min(D - 1, z + r)
        y0, y1 = max(0, y - r), min(H - 1, y + r)
        x0, x1 = max(0, x - r), min(W - 1, x + r)

        mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = np.uint8(value)
        return mask

    def _debug_save_first_landmark(
        self,
        case_name,
        moving_img,
        fixed_img,
        mov_pts,
        fix_gt_pts,
        out_dir,
        fix_pred_pts=None,  # refined points, [N,3] zyx
        radius=2,
    ):
        """
        Save moving/fixed images and point masks for the first landmark.

        The fixed label map encodes:
          - 1: GT landmark
          - 2: predicted landmark (overwrites GT if overlapping)
        """
        if mov_pts.shape[0] == 0 or fix_gt_pts.shape[0] == 0:
            return

        p_mov0 = mov_pts[0]
        p_fix_gt0 = fix_gt_pts[0]

        mov_label = self._make_point_mask(moving_img.shape, p_mov0, radius=radius, value=1)

        fix_label = None
        fix_label = self._make_point_mask(fixed_img.shape, p_fix_gt0, radius=radius, value=1, mask=fix_label)

        if fix_pred_pts is not None and len(fix_pred_pts) > 0:
            p_fix_pred0 = fix_pred_pts[0]
            fix_label = self._make_point_mask(fixed_img.shape, p_fix_pred0, radius=radius, value=2, mask=fix_label)

        self._save_nii(
            os.path.join(out_dir, f"{case_name}_moving_img.nii.gz"),
            moving_img,
            dtype=np.float32,
        )
        self._save_nii(
            os.path.join(out_dir, f"{case_name}_moving_label.nii.gz"),
            mov_label,
            dtype=np.uint8,
        )

        self._save_nii(
            os.path.join(out_dir, f"{case_name}_fixed_img.nii.gz"),
            fixed_img,
            dtype=np.float32,
        )
        self._save_nii(
            os.path.join(out_dir, f"{case_name}_fixed_label.nii.gz"),
            fix_label,
            dtype=np.uint8,
        )

    @staticmethod
    def _to_numpy_volume(x):
        """Convert input to a numpy volume with shape [D, H, W]."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        assert x.ndim == 3, f"Expect 3D volume, got {x.shape}"
        return x

    @staticmethod
    def _as_zyx_points(x):
        """Convert input to numpy float32 points with shape [N, 3]."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return x

    @staticmethod
    def _calculate_tre_mm(pred_zyx: np.ndarray, gt_zyx: np.ndarray, spacing):
        """
        Compute TRE in mm and voxels.

        Args:
            pred_zyx, gt_zyx: [N, 3] in voxel coordinates (z, y, x).
            spacing: scalar or [3] in mm per voxel (z, y, x).

        Returns:
            tre_mm: [N]
            tre_vx: [N]
            tre_x_mm, tre_y_mm, tre_z_mm: per-axis absolute error in mm (currently computed for the first point only).
        """
        pred = np.asarray(pred_zyx, dtype=np.float32)
        gt = np.asarray(gt_zyx, dtype=np.float32)

        diff_vx = pred - gt
        tre_vx = np.linalg.norm(diff_vx, axis=-1)

        if isinstance(spacing, (float, int)):
            scale = np.array([spacing, spacing, spacing], dtype=np.float32)
        else:
            scale = np.asarray(spacing, dtype=np.float32)
            if scale.shape != (3,):
                scale = scale.reshape(3,)

        diff_mm = diff_vx * scale[None, :]
        tre_mm = np.linalg.norm(diff_mm, axis=-1)

        # Keep the original behavior (axis errors for the first sample).
        tre_x_mm = np.absolute(diff_mm[0, 0])
        tre_y_mm = np.absolute(diff_mm[0, 1])
        tre_z_mm = np.absolute(diff_mm[0, 2])

        return tre_mm, tre_vx, tre_x_mm, tre_y_mm, tre_z_mm

    def _pad_to_roi(self, vol_zyx: np.ndarray, roi_zyx: np.ndarray):
        """
        Pad a volume so that each dimension is at least roi_size.

        Returns:
            vol_pad: padded volume
            pad_before: [pz, py, px], to be added to coordinates in the padded space
        """
        D, H, W = vol_zyx.shape
        rz, ry, rx = map(int, roi_zyx.tolist())

        pad_before = [0, 0, 0]
        pad_after = [0, 0, 0]

        for i, (s, r) in enumerate([(D, rz), (H, ry), (W, rx)]):
            if s < r:
                total = r - s
                pb = total // 2
                pa = total - pb
                pad_before[i] = pb
                pad_after[i] = pa

        if sum(pad_before) == 0 and sum(pad_after) == 0:
            return vol_zyx, np.array([0, 0, 0], dtype=np.int32)

        vol_pad = np.pad(
            vol_zyx,
            pad_width=(
                (pad_before[0], pad_after[0]),
                (pad_before[1], pad_after[1]),
                (pad_before[2], pad_after[2]),
            ),
            mode="edge",
        )
        return vol_pad, np.array(pad_before, dtype=np.int32)

    def _compute_patch_origin(self, center_zyx: np.ndarray, vol_shape_zyx, roi_size_zyx):
        """Compute a clamped patch origin (zyx) given a center point."""
        p = torch.tensor(center_zyx, dtype=torch.float32)
        roi = torch.tensor(roi_size_zyx, dtype=torch.float32)

        pos = torch.floor(p - 0.5 * roi)
        pos = torch.maximum(pos, torch.tensor([0, 0, 0], dtype=torch.float32))
        pos = torch.minimum(pos, torch.tensor(vol_shape_zyx, dtype=torch.float32) - roi)

        return pos.int().cpu().numpy()

    def _crop_patch_numpy(self, vol_zyx: np.ndarray, pos_zyx: np.ndarray):
        """Crop a patch from a numpy volume given origin (zyx)."""
        z0, y0, x0 = pos_zyx.tolist()
        dz, dy, dx = self.roi_size.tolist()
        return vol_zyx[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]

    def _forward_with_model(self, patch_mov: torch.Tensor, patch_fix: torch.Tensor):
        """
        Forward model on two patches.

        Args:
            patch_mov, patch_fix: [D, H, W] (numpy or torch)

        Returns:
            feat_mov_up, feat_fix_up: [1, C, D, H, W], upsampled to roi_size
        """
        if not isinstance(patch_mov, torch.Tensor):
            patch_mov = torch.from_numpy(patch_mov)
        if not isinstance(patch_fix, torch.Tensor):
            patch_fix = torch.from_numpy(patch_fix)

        patch_mov = patch_mov.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, D, H, W]
        patch_fix = patch_fix.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
            feat_mov, feat_fix = self.model(patch_mov, patch_fix)  # [1, C, d, h, w] x 2

        feat_mov_up = F.interpolate(
            feat_mov,
            size=tuple(self.roi_size.tolist()),
            mode="trilinear",
            align_corners=True,
        )
        feat_fix_up = F.interpolate(
            feat_fix,
            size=tuple(self.roi_size.tolist()),
            mode="trilinear",
            align_corners=True,
        )
        return feat_mov_up, feat_fix_up

    def _refine_one_landmark(
        self,
        moving_img_zyx: np.ndarray,  # [D, H, W]
        fixed_img_zyx: np.ndarray,  # [D, H, W]
        p_mov_zyx: np.ndarray,  # [3] in voxel zyx
        p_fix_coarse_zyx: np.ndarray,  # [3] in voxel zyx
    ) -> np.ndarray:
        """
        Refine a landmark given a coarse estimate.

        Procedure:
          1) Crop moving/fixed patches (roi_size) around p_mov and p_fix_coarse.
          2) Compute cosine similarity only within a window around the coarse point.
          3) Find the integer argmax in the search window.
          4) Apply soft-argmax in a smaller cube around the argmax.

        Returns:
            Refined point in global voxel coordinates (zyx), float32.
        """
        p_mov = np.round(np.asarray(p_mov_zyx)).astype(np.int32)
        p_fix_c = np.round(np.asarray(p_fix_coarse_zyx)).astype(np.int32)

        pos_m = self._compute_patch_origin(p_mov, moving_img_zyx.shape, self.roi_size)
        pos_f = self._compute_patch_origin(p_fix_c, fixed_img_zyx.shape, self.roi_size)

        patch_mov_np = self._crop_patch_numpy(moving_img_zyx, pos_m).astype(np.float32)
        patch_fix_np = self._crop_patch_numpy(fixed_img_zyx, pos_f).astype(np.float32)

        off_m = (p_mov - pos_m).astype(np.int32)
        off_c = (p_fix_c - pos_f).astype(np.int32)

        patch_mov = torch.from_numpy(patch_mov_np).to(self.device)
        patch_fix = torch.from_numpy(patch_fix_np).to(self.device)

        feat_mov, feat_fix = self._forward_with_model(patch_mov, patch_fix)
        feat_mov = F.normalize(feat_mov, p=2, dim=1)
        feat_fix = F.normalize(feat_fix, p=2, dim=1)

        dz, dy, dx = map(int, self.roi_size.tolist())

        zm, ym, xm = map(int, off_m.tolist())
        zm = int(np.clip(zm, 0, dz - 1))
        ym = int(np.clip(ym, 0, dy - 1))
        xm = int(np.clip(xm, 0, dx - 1))

        zc, yc, xc = map(int, off_c.tolist())
        zc = int(np.clip(zc, 0, dz - 1))
        yc = int(np.clip(yc, 0, dy - 1))
        xc = int(np.clip(xc, 0, dx - 1))

        local_vec = feat_mov[0, :, zm, ym, xm]  # [C]
        local_vec = F.normalize(local_vec, p=2, dim=0)

        # ----------------------------
        # Limited search around coarse (in fixed patch coordinates).
        # ----------------------------
        r_search = int(getattr(self, "search_radius", 16))
        if r_search is None or r_search <= 0:
            z0, z1 = 0, dz
            y0, y1 = 0, dy
            x0, x1 = 0, dx
        else:
            z0 = max(zc - r_search, 0)
            z1 = min(zc + r_search + 1, dz)
            y0 = max(yc - r_search, 0)
            y1 = min(yc + r_search + 1, dy)
            x0 = max(xc - r_search, 0)
            x1 = min(xc + r_search + 1, dx)

        search_feat = feat_fix[0, :, z0:z1, y0:y1, x0:x1]  # [C, Zw, Yw, Xw]
        sim_win = (search_feat * local_vec.view(-1, 1, 1, 1)).sum(dim=0)  # [Zw, Yw, Xw]

        # Stage 1: argmax in the search window.
        flat_idx = torch.argmax(sim_win.reshape(-1))
        Zw, Yw, Xw = sim_win.shape

        iz = flat_idx // (Yw * Xw)
        rem = flat_idx % (Yw * Xw)
        iy = rem // Xw
        ix = rem % Xw

        p0 = torch.stack([iz + z0, iy + y0, ix + x0], dim=0)  # patch-space zyx (int)

        # Stage 2: soft-argmax around p0.
        r_ref = int(self.refine_radius)

        zz0 = int(max(p0[0].item() - r_ref, z0))
        zz1 = int(min(p0[0].item() + r_ref + 1, z1))
        yy0 = int(max(p0[1].item() - r_ref, y0))
        yy1 = int(min(p0[1].item() + r_ref + 1, y1))
        xx0 = int(max(p0[2].item() - r_ref, x0))
        xx1 = int(min(p0[2].item() + r_ref + 1, x1))

        local_feat = feat_fix[0, :, zz0:zz1, yy0:yy1, xx0:xx1]  # [C, rz, ry, rx]
        local_sim = (local_feat * local_vec.view(-1, 1, 1, 1)).sum(dim=0)  # [rz, ry, rx]

        logits = local_sim.reshape(-1) / float(self.match_tau)
        prob = F.softmax(logits, dim=0)

        zz = torch.arange(zz0, zz1, device=self.device)
        yy = torch.arange(yy0, yy1, device=self.device)
        xx = torch.arange(xx0, xx1, device=self.device)
        Z, Y, X = torch.meshgrid(zz, yy, xx, indexing="ij")
        coords_patch = torch.stack([Z, Y, X], dim=-1).reshape(-1, 3).float()  # [K, 3] zyx

        refined_patch = (prob.view(-1, 1) * coords_patch).sum(dim=0)  # [3] zyx float
        refined_global = refined_patch + torch.tensor(pos_f, device=self.device, dtype=torch.float32)

        return refined_global.detach().cpu().numpy().astype(np.float32)

    # -------------------- main predict/run --------------------
    def predict(self, batch_data: dict, data_name: str):
        """
        Run refinement for one image pair.

        Convention:
          - moving: image_2, landmark_2
          - fixed:  image_1, landmark_1
          - coarse: batch_data["landmark_coarse"] if provided, otherwise simulated from GT

        Returns:
            tre_mm [N], tre_vx [N], tre_x_mm, tre_y_mm, tre_z_mm
        """
        moving_img = self._to_numpy_volume(batch_data["image_2"]).astype(np.float32)
        fixed_img = self._to_numpy_volume(batch_data["image_1"]).astype(np.float32)

        moving_img, pad_m = self._pad_to_roi(moving_img, self.roi_size)
        fixed_img, pad_f = self._pad_to_roi(fixed_img, self.roi_size)

        mov_pts = self._as_zyx_points(batch_data["landmark_2"]) + pad_m[None, :]
        fix_gt = self._as_zyx_points(batch_data["landmark_1"]) + pad_f[None, :]

        # ---------------------------
        # 1) Coarse source: real coarse or simulated coarse
        # ---------------------------
        if "landmark_coarse" in batch_data:
            fix_coarse = self._as_zyx_points(batch_data["landmark_coarse"]) + pad_f[None, :]
        else:
            enable = bool(self.inference_config.get("coarse_sim_enable", True))
            max_err = float(self.inference_config.get("coarse_sim_max_error_vx", 0))
            mode = str(self.inference_config.get("coarse_sim_mode", "uniform_cube"))
            seed = int(self.inference_config.get("coarse_sim_seed", 0))

            if (not enable) or max_err <= 0:
                fix_coarse = fix_gt.copy()
            else:
                rng = np.random.default_rng(seed)

                if mode == "uniform_sphere":
                    v = rng.normal(size=fix_gt.shape).astype(np.float32)
                    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
                    r = (rng.random(size=(fix_gt.shape[0], 1)).astype(np.float32) ** (1.0 / 3.0)) * max_err
                    jitter = v * r
                else:
                    jitter = rng.uniform(-max_err, max_err, size=fix_gt.shape).astype(np.float32)

                fix_coarse = fix_gt + jitter

                D, H, W = fixed_img.shape
                fix_coarse[:, 0] = np.clip(fix_coarse[:, 0], 0, D - 1)
                fix_coarse[:, 1] = np.clip(fix_coarse[:, 1], 0, H - 1)
                fix_coarse[:, 2] = np.clip(fix_coarse[:, 2], 0, W - 1)

        N = min(len(mov_pts), len(fix_gt), len(fix_coarse))
        mov_pts = mov_pts[:N]
        fix_gt = fix_gt[:N]
        fix_coarse = fix_coarse[:N]

        # ---------------------------
        # 2) Refine
        # ---------------------------
        refined = []
        for i in range(N):
            refined_i = self._refine_one_landmark(
                moving_img_zyx=moving_img,
                fixed_img_zyx=fixed_img,
                p_mov_zyx=mov_pts[i],
                p_fix_coarse_zyx=fix_coarse[i],
            )
            refined.append(refined_i)

        refined = np.stack(refined, axis=0) if N > 0 else np.zeros((0, 3), dtype=np.float32)

        # ---------------------------
        # 3) Optional debug visualization
        # ---------------------------
        debug_enable = bool(self.inference_config.get("debug_vis_enable", True))
        debug_max_cases = int(self.inference_config.get("debug_vis_max_cases", 999999))
        debug_radius = int(self.inference_config.get("debug_vis_radius", 2))

        if debug_enable and self.case_counter <= debug_max_cases:
            vis_dir = os.path.join(self.output_dir, "vis_nii")
            self._debug_save_first_landmark(
                case_name=data_name[:-7],
                moving_img=moving_img,
                fixed_img=fixed_img,
                mov_pts=mov_pts,
                fix_gt_pts=fix_gt,
                fix_pred_pts=refined,
                out_dir=vis_dir,
                radius=debug_radius,
            )

        # ---------------------------
        # 4) Metrics: refined->GT and coarse->GT
        # ---------------------------
        tre_mm, tre_vx, tre_x_mm, tre_y_mm, tre_z_mm = self._calculate_tre_mm(refined, fix_gt, self.spacing)
        coarse_err_vx = (
            np.linalg.norm((fix_coarse - fix_gt).astype(np.float32), axis=1) if N > 0 else np.zeros((0,), np.float32)
        )

        # ---------------------------
        # 5) Save per-case Excel
        # ---------------------------
        if N > 0:
            out_arr = np.concatenate([mov_pts, fix_coarse, refined, fix_gt], axis=1)
            titles = [
                "moving: Z",
                "moving: Y",
                "moving: X",
                "coarse: Z",
                "coarse: Y",
                "coarse: X",
                "refined: Z",
                "refined: Y",
                "refined: X",
                "fixed_GT: Z",
                "fixed_GT: Y",
                "fixed_GT: X",
            ]
            df = pd.DataFrame({t: out_arr[:, i] for i, t in enumerate(titles)})
            df["coarse_err_vx"] = coarse_err_vx
            df["refined_err_vx"] = tre_vx
            df["refined_err_mm"] = tre_mm
            df["refined_err_x_mm"] = tre_x_mm
            df["refined_err_y_mm"] = tre_y_mm
            df["refined_err_z_mm"] = tre_z_mm
        else:
            df = pd.DataFrame()

        excel_path = os.path.join(self.output_dir, data_name[:-7] + ".xlsx")
        os.makedirs(self.output_dir, exist_ok=True)
        df.to_excel(excel_path, index=False)

        return tre_mm, tre_vx, tre_x_mm, tre_y_mm, tre_z_mm

    def run(self):
        """Run inference for the whole test set and export summary Excel."""
        self._setup_components()
        os.makedirs(self.output_dir, exist_ok=True)

        print("\n--- Running inference (local refinement) ---")
        print(f"Output dir: {self.output_dir}")

        per_case = {}
        all_tre_mm = []
        all_tre_vx = []
        all_ok = []

        radius_csv_path = self.inference_config.get("radius_csv", "")
        radius_dict = {}

        if radius_csv_path != "" and os.path.exists(radius_csv_path):
            save_radius = True
            df_radius = pd.read_csv(radius_csv_path)
            for _, row in df_radius.iterrows():
                pair_id = row["pair_id"]
                if pair_id[:4] != "test":
                    continue
                radius_dict[pair_id] = {
                    "source_name": row["source"],
                    "target_name": row["target"],
                    "source_radius": row["source_radius_mm"],
                    "target_radius": row["target_radius_mm"],
                }
            print(f"Loaded lesion radius from: {radius_csv_path}")
        else:
            save_radius = False

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="inference"):
                for one in decollate_batch(batch):
                    data_name = os.path.basename(one["image_meta_dict"]["filename_or_obj"])
                    case_name = data_name[:-7]
                    pair_id = one["image_meta_dict"].get("pair_id", "")

                    if save_radius:
                        radius = radius_dict[pair_id]["target_radius"]

                    # Load coarse predictions (stage-1 outputs) if provided.
                    if self.corse_path != "":
                        corse_name = os.path.join(self.corse_path, case_name + ".xlsx")
                        if not os.path.exists(corse_name):
                            print(f"Coarse file not found: {corse_name}")
                            continue
                        df = pd.read_excel(corse_name)
                        coords = df[["moved: X", "moved: Y", "moved: Z"]].to_numpy()
                        one["landmark_coarse"] = torch.from_numpy(coords.astype(np.float32))

                    tre_mm, tre_vx, tre_x_mm, tre_y_mm, tre_z_mm = self.predict(one, data_name)

                    tre_mm = np.asarray(tre_mm, dtype=np.float32)
                    tre_vx = np.asarray(tre_vx, dtype=np.float32)

                    ok_mask = tre_vx < self.ok_thresh_vx
                    ok_10 = float(ok_mask.mean()) if tre_vx.size > 0 else np.nan

                    if save_radius:
                        ok_ratio = float((tre_vx < radius).mean()) if tre_vx.size > 0 else np.nan
                    else:
                        ok_ratio = np.nan

                    per_case[case_name] = {
                        "N": int(tre_vx.size),
                        "Mean_TRE_mm": float(np.nanmean(tre_mm)) if tre_mm.size > 0 else np.nan,
                        "Mean_TRE_x_mm": float(np.nanmean(tre_x_mm)) if np.size(tre_x_mm) > 0 else np.nan,
                        "Mean_TRE_y_mm": float(np.nanmean(tre_y_mm)) if np.size(tre_y_mm) > 0 else np.nan,
                        "Mean_TRE_z_mm": float(np.nanmean(tre_z_mm)) if np.size(tre_z_mm) > 0 else np.nan,
                        "Mean_TRE_vx": float(np.nanmean(tre_vx)) if tre_vx.size > 0 else np.nan,
                        "Std_TRE_vx": float(np.nanstd(tre_vx)) if tre_vx.size > 0 else np.nan,
                        "Ratio_<10vx": ok_10,
                        "Ratio_<radius": ok_ratio,
                    }

                    if tre_mm.size > 0:
                        all_tre_mm.append(tre_mm)
                        all_tre_vx.append(tre_vx)
                        all_ok.append(ok_mask.astype(np.float32))

        if len(per_case) == 0:
            print("No inference results.")
            return

        df_cases = pd.DataFrame.from_dict(per_case, orient="index")
        df_cases.index.name = "Case"

        out_path = os.path.join(self.output_dir, "inference_summary_detailed.xlsx")
        with pd.ExcelWriter(out_path) as writer:
            df_cases.to_excel(writer, sheet_name="per_case")

        print(f"\nSaved summary to: {out_path}")