import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.loss_AE import loss_AE
from loss.loss_base import LossBase


class loss_AE_cross(LossBase):
    """
    Cross-distillation loss for correspondence learning.

    Components:
      - AE loss on teacher features (correspondence_output)
      - AE loss on student/decoder features (maisi_output)
      - Cross-pair KL distillation between student similarities and teacher similarities
    """

    def __init__(self, config: dict):
        self._load_configs(config)
        super().__init__(config)

        self.ae_loss = loss_AE(config)
        self.loss_func = nn.CrossEntropyLoss()

        self.wight_ae = self.loss_config.get("wight_ae", 1.0)
        self.wight_maisidecoder = self.loss_config.get("wight_maisidecoder", 1.0)
        self.wight_cross = self.loss_config.get("wight_cross", 1.0)

        self.cross_tau = self.loss_config.get("cross_tau", 1.0)
        self.distillation_temp = self.loss_config.get("distillation_temp", 1.0)

    def _get_loss_name(self):
        return "loss_AE_cross"

    def _get_feature_vector(self, embedding_tensor, pos_zyx, dim=3):
        """
        Gather features at normalized positions.

        Args:
            embedding_tensor:
              - dim=3: [B, C, D, H, W]
              - dim=2: [B, C, H, W]
            pos_zyx:
              - dim=3: [B, N, 3] in [-1, 1], order (z, y, x)
              - dim=2: [B, N, 2] in [-1, 1], order (y, x)
                If a 3D (z,y,x) tensor is passed, the last two dims are used.
            dim: 2 or 3

        Returns:
            [B, C, N]
        """
        if dim == 3:
            B, C, D, H, W = embedding_tensor.shape
            assert pos_zyx.shape[-1] == 3

            # [-1, 1] -> [0, size-1] (align_corners=True semantics)
            z = torch.round((pos_zyx[..., 0] + 1) * 0.5 * (D - 1)).long()
            y = torch.round((pos_zyx[..., 1] + 1) * 0.5 * (H - 1)).long()
            x = torch.round((pos_zyx[..., 2] + 1) * 0.5 * (W - 1)).long()

            z = z.clamp(0, D - 1)
            y = y.clamp(0, H - 1)
            x = x.clamp(0, W - 1)

            fm = embedding_tensor.view(B, C, D * H * W)
            lin = (z * (H * W) + y * W + x).unsqueeze(1).expand(-1, C, -1)  # [B, C, N]
            f = torch.gather(fm, 2, lin)
            return f

        if dim == 2:
            B, C, H, W = embedding_tensor.shape

            pos_yx = pos_zyx[..., 1:3] if pos_zyx.shape[-1] == 3 else pos_zyx
            y = torch.round((pos_yx[..., 0] + 1) * 0.5 * (H - 1)).long()
            x = torch.round((pos_yx[..., 1] + 1) * 0.5 * (W - 1)).long()

            y = y.clamp(0, H - 1)
            x = x.clamp(0, W - 1)

            fm = embedding_tensor.view(B, C, H * W)
            lin = (y * W + x).unsqueeze(1).expand(-1, C, -1)  # [B, C, N]
            f = torch.gather(fm, 2, lin)
            return f

        raise ValueError(f"Unsupported dim={dim}")

    def cross_loss(self, embedding_1, embedding_2, positive_pos_1, positive_pos_2, dim=3):
        """
        Cross-pair KL distillation loss.

        Student similarity distribution (maisi_output) is aligned to teacher similarity
        distribution (correspondence_output).

        Args:
            embedding_1: dict with keys {"maisi_output", "correspondence_output"}
            embedding_2: dict with keys {"maisi_output", "correspondence_output"}
            positive_pos_1: [B, N, 3] for embedding_1
            positive_pos_2: [B, N, 3] for embedding_2
            dim: 2 or 3
        """
        # Sample features at positive locations
        f_student_1 = self._get_feature_vector(embedding_1["maisi_output"], positive_pos_1, dim)  # [B, C, N]
        f_student_2 = self._get_feature_vector(embedding_2["maisi_output"], positive_pos_2, dim)  # [B, C, N]

        f_teacher_1 = self._get_feature_vector(
            embedding_1["correspondence_output"], positive_pos_1, dim
        )  # [B, C, N]
        f_teacher_2 = self._get_feature_vector(
            embedding_2["correspondence_output"], positive_pos_2, dim
        )  # [B, C, N]

        # [B, C, N] -> [B, N, C]
        f_student_1 = f_student_1.permute(0, 2, 1)
        f_student_2 = f_student_2.permute(0, 2, 1)
        f_teacher_1 = f_teacher_1.permute(0, 2, 1)
        f_teacher_2 = f_teacher_2.permute(0, 2, 1)

        # Similarity matrices: [B, N, N]
        logits_student = torch.bmm(f_student_1, f_student_2.transpose(1, 2))
        logits_teacher = torch.bmm(f_teacher_1, f_teacher_2.transpose(1, 2))

        # KLDivLoss expects (log p_student, p_teacher)
        log_prob_student = F.log_softmax(logits_student / self.distillation_temp, dim=-1)
        with torch.no_grad():
            prob_teacher = F.softmax(logits_teacher / self.distillation_temp, dim=-1)
        prob_teacher.detach()  # keep original behavior

        loss_kl = F.kl_div(log_prob_student, prob_teacher, reduction="batchmean")
        scaled_loss = loss_kl * (self.distillation_temp**2)
        return scaled_loss

    def forward(
        self,
        embedding_11,
        embedding_12,
        embedding_21,
        embedding_22,
        positive_pos_1,
        positive_pos_2,
        dim=3,
    ):
        """
        Compute total loss over two paired samples.

        Args:
            embedding_11, embedding_12: pair-1 embeddings (dicts)
            embedding_21, embedding_22: pair-2 embeddings (dicts)
            positive_pos_1: [B, N, 2, 3] for pair-1 (0/1 index selects patch)
            positive_pos_2: [B, N, 2, 3] for pair-2 (0/1 index selects patch)
            dim: 2 or 3
        """
        loss_ae = (
            self.ae_loss(
                embedding_11["correspondence_output"],
                embedding_12["correspondence_output"],
                positive_pos_1[:, :, 0, :],
                positive_pos_1[:, :, 1, :],
                dim,
            )["total_loss"]
            + self.ae_loss(
                embedding_21["correspondence_output"],
                embedding_22["correspondence_output"],
                positive_pos_2[:, :, 0, :],
                positive_pos_2[:, :, 1, :],
                dim,
            )["total_loss"]
        )
        loss_ae = loss_ae * self.wight_ae / 2.0

        loss_maisidecoder = (
            self.ae_loss(
                embedding_11["maisi_output"],
                embedding_12["maisi_output"],
                positive_pos_1[:, :, 0, :],
                positive_pos_1[:, :, 1, :],
                dim,
            )["total_loss"]
            + self.ae_loss(
                embedding_21["maisi_output"],
                embedding_22["maisi_output"],
                positive_pos_2[:, :, 0, :],
                positive_pos_2[:, :, 1, :],
                dim,
            )["total_loss"]
        )
        loss_maisidecoder = loss_maisidecoder * self.wight_maisidecoder / 2.0

        loss_cross = (
            self.cross_loss(
                embedding_11,
                embedding_21,
                positive_pos_1[:, :, 0, :],
                positive_pos_2[:, :, 0, :],
            )
            + self.cross_loss(
                embedding_12,
                embedding_22,
                positive_pos_1[:, :, 1, :],
                positive_pos_2[:, :, 1, :],
            )
        )
        loss_cross = loss_cross * self.wight_cross / 2.0

        loss = loss_ae + loss_maisidecoder + loss_cross
        return {
            "total_loss": loss,
            "ae_loss": loss_ae,
            "maisidecoder_loss": loss_maisidecoder,
            "cross loss": loss_cross,
        }