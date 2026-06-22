import torch
import torch.nn.functional as F

from loss.loss_base import LossBase


class loss_AE_loca_distance(LossBase):
    """
    Location-distance supervision for correspondence embeddings.

    Key idea (v2):
      - Avoid subtracting coordinates across different patch frames.
      - Build two RBF similarity targets:
          * Row target uses pos2 (keys) coordinates.
          * Col target uses pos1 (queries) coordinates.
      - Add an InfoNCE diagonal constraint so i ↔ i is the strongest match.

    Inputs:
      - embedding_1, embedding_2: [B, C, D, H, W]
      - positive_pos_1, positive_pos_2: [B, N, 3] in [-1, 1] (grid_sample coords, order z,y,x)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._load_configs(config)

        self.sigma = float(self.loss_config.get("sigma", 1.0))
        self.tau = float(self.loss_config.get("tau", 0.1))  # InfoNCE temperature

        self.w_mse = float(self.loss_config.get("w_mse", 1.0))
        self.w_nce = float(self.loss_config.get("w_nce", 1.0))

        self.symmetric_nce = bool(self.loss_config.get("symmetric_nce", True))

    def _get_loss_name(self):
        return "loss_AE_loca_distance"

    @staticmethod
    def _sample_trilinear(feature_map: torch.Tensor, pos_zyx: torch.Tensor) -> torch.Tensor:
        """
        Trilinear sampling of features at normalized coordinates.

        Args:
            feature_map: [B, C, D, H, W]
            pos_zyx:     [B, N, 3] in [-1, 1], order (z, y, x)

        Returns:
            Sampled features: [B, C, N]
        """
        B, C, D, H, W = feature_map.shape

        # grid_sample expects (x, y, z)
        pos_xyz = pos_zyx[..., [2, 1, 0]]
        grid = pos_xyz.view(B, -1, 1, 1, 3)  # [B, N, 1, 1, 3]

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",  # trilinear for 5D inputs
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).squeeze(-1)  # [B, C, N]

    @staticmethod
    def _pairwise_rbf_from_same_coords(pos: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Build an RBF similarity matrix from coordinates within the same frame.

        Args:
            pos:   [B, N, 3]
            sigma: RBF bandwidth

        Returns:
            sim_gt: [B, N, N]
        """
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1)             # [B, N, N]
        return torch.exp(-(dist**2) / (2.0 * (sigma**2)))

    @staticmethod
    def _diag_offdiag_stats(sim: torch.Tensor):
        """
        Compute simple diagonal/off-diagonal statistics.

        Args:
            sim: [B, N, N]

        Returns:
            diag_mean, offdiag_mean, margin
        """
        B, N, _ = sim.shape
        eye = torch.eye(N, device=sim.device, dtype=torch.bool).unsqueeze(0)  # [1, N, N]

        diag = sim[eye.expand(B, -1, -1)].view(B, N)
        off = sim[~eye.expand(B, -1, -1)].view(B, N * (N - 1))

        diag_mean = diag.mean()
        off_mean = off.mean()
        margin = diag_mean - off_mean
        return diag_mean, off_mean, margin

    def forward(self, embedding_1, embedding_2, positive_pos_1, positive_pos_2, dim=3):
        assert dim == 3, "Only support dim=3."

        B, C, D, H, W = embedding_1.shape
        _, N, _ = positive_pos_1.shape

        # 1) Sample features at positive locations
        f_q = self._sample_trilinear(embedding_1, positive_pos_1)  # [B, C, N]
        f_k = self._sample_trilinear(embedding_2, positive_pos_2)  # [B, C, N]

        # 2) Normalize features (cosine similarity)
        f_q = F.normalize(f_q, dim=1)
        f_k = F.normalize(f_k, dim=1)

        # 3) Similarity matrix
        sim_emb = torch.bmm(f_q.transpose(1, 2), f_k)  # [B, N, N]
        logits = sim_emb / self.tau

        # 4) RBF targets (row uses pos2, col uses pos1)
        sim_gt_row = self._pairwise_rbf_from_same_coords(positive_pos_2, self.sigma)  # [B, N, N]
        sim_gt_col = self._pairwise_rbf_from_same_coords(positive_pos_1, self.sigma)  # [B, N, N]

        # 5) Losses
        loss_mse_row = F.mse_loss(sim_emb, sim_gt_row)
        loss_mse_col = F.mse_loss(sim_emb, sim_gt_col)
        loss_mse = 0.5 * (loss_mse_row + loss_mse_col)

        labels = torch.arange(N, device=logits.device).view(1, N).repeat(B, 1)  # [B, N]
        loss_nce_row = F.cross_entropy(logits.reshape(B * N, N), labels.reshape(B * N))

        if self.symmetric_nce:
            loss_nce_col = F.cross_entropy(
                logits.transpose(1, 2).reshape(B * N, N),
                labels.reshape(B * N),
            )
            loss_nce = 0.5 * (loss_nce_row + loss_nce_col)
        else:
            loss_nce_col = None
            loss_nce = loss_nce_row

        total = self.w_mse * loss_mse + self.w_nce * loss_nce

        # ---- Monitoring metrics (no grad) ----
        with torch.no_grad():
            pred_row = sim_emb.argmax(dim=-1)  # [B, N]
            acc_row = (pred_row == labels).float().mean()

            pred_col = sim_emb.argmax(dim=-2)  # [B, N]
            acc_col = (pred_col == labels).float().mean()

            diag_mean, off_mean, margin = self._diag_offdiag_stats(sim_emb)

            prob_row = F.softmax(logits, dim=-1)                     # [B, N, N]
            diag_prob = prob_row.diagonal(dim1=1, dim2=2)            # [B, N]
            diag_prob_mean = diag_prob.mean()

            eps = 1e-12
            entropy_row = -(prob_row * prob_row.clamp_min(eps).log()).sum(dim=-1)  # [B, N]
            entropy_row_mean = entropy_row.mean()

        out = {
            "total_loss": total,
            "loss_mse": loss_mse.detach(),
            "loss_mse_row": loss_mse_row.detach(),
            "loss_mse_col": loss_mse_col.detach(),
            "loss_nce": loss_nce.detach(),
            "loss_nce_row": loss_nce_row.detach(),
            "acc_row@1": acc_row,
            "acc_col@1": acc_col,
            "diag_sim_mean": diag_mean,
            "offdiag_sim_mean": off_mean,
            "diag_margin": margin,
            "diag_prob_mean": diag_prob_mean,
            "entropy_row_mean": entropy_row_mean,
        }
        if loss_nce_col is not None:
            out["loss_nce_col"] = loss_nce_col.detach()

        return out