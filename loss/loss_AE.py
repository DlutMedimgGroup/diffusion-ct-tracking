import torch
import torch.nn as nn

# --- Import base loss class ---
from loss.loss_base import LossBase


class loss_AE(LossBase):
    """
    Basic InfoNCE-style contrastive loss.

    This loss samples one positive key and multiple random negative keys
    from the feature map and applies cross-entropy over similarity logits.
    """

    def __init__(self, config: dict):
        """
        Initialize loss with configuration.

        Args:
            config (dict): Full experiment configuration.
        """
        # Load loss-specific config first
        self._load_configs(config)
        super().__init__(config)

        self.tau = self.loss_config.get("tau", 0.1)
        self.n_neg = self.loss_config.get("n_neg", 100000)

        self.loss_func = nn.CrossEntropyLoss()

    def _get_loss_name(self):
        """Return the name of this loss."""
        return "loss_AE"

    def _get_feature_vector(self, embedding_tensor, pos_zyx, dim=3):
        """
        Extract feature vectors at specified normalized coordinates.

        Args:
            embedding_tensor:
                dim=3: [B, C, D, H, W]
                dim=2: [B, C, H, W]
            pos_zyx:
                dim=3: [B, N, 3] in [-1, 1], order (z, y, x)
                dim=2: [B, N, 2] in [-1, 1], order (y, x)
            dim (int): Spatial dimensionality.

        Returns:
            Tensor:
                dim=3: [B, C, N]
                dim=2: [B, C, N]
        """
        if dim == 3:
            B, C, D, H, W = embedding_tensor.shape
            assert pos_zyx.shape[-1] == 3

            # Map normalized coordinates [-1,1] to voxel indices
            z = torch.round((pos_zyx[..., 0] + 1) * 0.5 * (D - 1)).long()
            y = torch.round((pos_zyx[..., 1] + 1) * 0.5 * (H - 1)).long()
            x = torch.round((pos_zyx[..., 2] + 1) * 0.5 * (W - 1)).long()

            z = z.clamp(0, D - 1)
            y = y.clamp(0, H - 1)
            x = x.clamp(0, W - 1)

            # Flatten spatial dimension and gather
            fm = embedding_tensor.view(B, C, D * H * W)
            lin = (z * (H * W) + y * W + x).unsqueeze(1).expand(-1, C, -1)
            f = torch.gather(fm, 2, lin)
            return f

        elif dim == 2:
            B, C, H, W = embedding_tensor.shape

            # Support both [B,N,3] and [B,N,2]
            if pos_zyx.shape[-1] == 3:
                pos_yx = pos_zyx[..., 1:3]
            else:
                pos_yx = pos_zyx

            y = torch.round((pos_yx[..., 0] + 1) * 0.5 * (H - 1)).long()
            x = torch.round((pos_yx[..., 1] + 1) * 0.5 * (W - 1)).long()

            y = y.clamp(0, H - 1)
            x = x.clamp(0, W - 1)

            fm = embedding_tensor.view(B, C, H * W)
            lin = (y * W + x).unsqueeze(1).expand(-1, C, -1)
            f = torch.gather(fm, 2, lin)
            return f

        else:
            raise ValueError(f"Unsupported dim={dim}")

    def _get_negative_samples(self, feature_map):
        """
        Randomly sample negative feature vectors from the feature map.

        Args:
            feature_map (Tensor): [B, C, D, H, W]

        Returns:
            Tensor: Negative samples of shape [B, C, n_neg]
        """
        B, C, D, H, W = feature_map.shape
        num_spatial_locations = D * H * W

        # Flatten spatial dimension: [B, C, D*H*W]
        fm_flat = feature_map.view(B, C, num_spatial_locations)

        # Random indices per sample in batch
        rand_indices = torch.randint(
            0,
            num_spatial_locations,
            (B, self.n_neg),
            device=feature_map.device,
        )

        # Expand indices for gather
        rand_indices_expanded = rand_indices.unsqueeze(1).expand(-1, C, -1)

        # Gather negative samples: [B, C, n_neg]
        negative_samples = torch.gather(fm_flat, 2, rand_indices_expanded)

        return negative_samples

    def forward(self, embedding_1, embedding_2, positive_pos_1, positive_pos_2, dim=3):
        """
        Compute InfoNCE contrastive loss.

        Args:
            embedding_1 (Tensor): Query feature map [B, C, ...]
            embedding_2 (Tensor): Key feature map   [B, C, ...]
            positive_pos_1 (Tensor): Query positive coordinates [B, N, 3]
            positive_pos_2 (Tensor): Key positive coordinates   [B, N, 3]
            dim (int): Spatial dimensionality.

        Returns:
            dict: {"total_loss": loss}
        """
        F_l_q = embedding_1
        F_l_k = embedding_2

        # --- 1. Positive feature pairs ---
        f_q = self._get_feature_vector(F_l_q, positive_pos_1, dim)      # [B, C, N]
        f_k_pos = self._get_feature_vector(F_l_k, positive_pos_2, dim)  # [B, C, N]

        # --- 2. Negative samples ---
        f_k_neg = self._get_negative_samples(F_l_k)  # [B, C, n_neg]

        # --- 3. Similarity logits ---
        # Positive similarity: dot product
        l_pos = (f_q * f_k_pos).sum(dim=1)  # [B, N]

        # Negative similarity via batch matrix multiplication
        # [B, C, N] -> [B, N, C]
        # [B, N, C] @ [B, C, n_neg] -> [B, N, n_neg]
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k_neg)

        # --- 4. Concatenate logits ---
        logits = torch.cat([l_pos.unsqueeze(-1), l_neg], dim=-1)  # [B, N, 1+n_neg]

        # --- 5. Temperature scaling ---
        logits = logits / self.tau

        # --- 6. Cross-entropy loss ---
        B, N, _ = logits.shape
        labels = torch.zeros(B * N, dtype=torch.long, device=logits.device)

        loss = self.loss_func(logits.view(B * N, -1), labels)

        return {"total_loss": loss}