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

from .inference_AE import inference_AE


class inference_AE_diffusion(inference_AE):
    """Inference wrapper that builds cosine-normalized features from diffusion outputs."""

    def __init__(self, config: dict):
        """Initialize the diffusion-based AE inference runner."""
        super().__init__(config)

    def _forward_with_model(self, x, roi_size):
        """Run model inference and return normalized features at the requested ROI size."""
        with torch.amp.autocast(device_type=self.device.type):
            y = self.model.inference(x)

        coarse = y["diffusion_coarse"]
        fine = y["diffusion_fine"]

        # Match spatial resolution to `fine` before concatenation.
        coarse_up = F.interpolate(coarse, size=fine.shape[2:], mode="trilinear", align_corners=False)
        feat = torch.cat([coarse_up, fine], dim=1)

        roi_size_tuple = tuple(int(v) for v in roi_size.tolist())
        feat_up = F.interpolate(feat, size=roi_size_tuple, mode="trilinear", align_corners=True)

        # Normalize channel-wise for cosine similarity downstream.
        feat_up = F.normalize(feat_up, dim=1)
        return feat_up

    def _adaptive_roi_size(self, shape1, shape2, base_roi, multiple: int = 16):
        """
        Compute an ROI size that:
          - does not exceed `base_roi`,
          - fits within both images,
          - and is floored to a fixed multiple per dimension.

        Returns:
            np.ndarray[int] or None: ROI size, or None if any dim becomes too small.
        """
        multiple = 64  # Keep behavior unchanged (overrides the default argument).

        shape1 = np.array(shape1, dtype=np.int64)
        shape2 = np.array(shape2, dtype=np.int64)
        base_roi = np.array(base_roi, dtype=np.int64)

        max_allow = np.minimum(shape1, shape2)
        roi = np.minimum(base_roi, max_allow)
        roi = np.array([self._floor_to_multiple(v, multiple) for v in roi], dtype=np.int64)

        if np.any(roi < multiple):
            return None
        return roi