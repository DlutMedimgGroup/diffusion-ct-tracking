import torch
import torch.nn.functional as F

from loss.loss_base import LossBase


class loss_AE_cross_moco(LossBase):
    """
    Cross-MoCo loss for correspondence learning.

    Goals:
      1) Patch-level contrastive learning with MoCo queues (more/global negatives).
      2) Teacher-student alignment via matching similarity distributions to their
         respective memory banks (KL distillation).
    """

    def __init__(self, config: dict):
        self._load_configs(config)
        super().__init__(config)

        # -------------------- weights --------------------
        self.wight_ae = self.loss_config.get("wight_ae", 1.0)
        self.wight_maisidecoder = self.loss_config.get("wight_maisidecoder", 1.0)

        self.wight_moco_student = self.loss_config.get("wight_moco_student", 1.0)
        self.wight_moco_teacher = self.loss_config.get("wight_moco_teacher", 1.0)

        self.wight_bank_distill = self.loss_config.get("wight_bank_distill", 1.0)
        self.bank_distill_T = self.loss_config.get("bank_distill_T", 1.0)  # temperature for KL

        # MoCo temperatures
        self.T_student = self.loss_config.get("moco_T_student", 0.2)
        self.T_teacher = self.loss_config.get("moco_T_teacher", 0.2)

        # Queue sizes
        self.queue_size_student = int(self.loss_config.get("queue_size_student", 65536))
        self.queue_size_teacher = int(self.loss_config.get("queue_size_teacher", 65536))

        # Feature dimension
        self.feat_dim = int(
            self.loss_config.get(
                "feat_dim",
                config.get("Model", {}).get("target_feature_size", 32),
            )
        )

        # -------------------- queues --------------------
        # Store keys from *_k branch with shape [C, K].
        self.register_buffer(
            "queue_student",
            F.normalize(torch.randn(self.feat_dim, self.queue_size_student), dim=0),
        )
        self.register_buffer(
            "queue_teacher",
            F.normalize(torch.randn(self.feat_dim, self.queue_size_teacher), dim=0),
        )
        self.register_buffer("ptr_student", torch.zeros(1, dtype=torch.long))
        self.register_buffer("ptr_teacher", torch.zeros(1, dtype=torch.long))

    def _get_loss_name(self):
        return "loss_AE_cross_moco"

    # -------------------------
    # pos -> feature vectors
    # -------------------------
    def _get_feature_vector(self, embedding_tensor, pos_zyx, dim=3):
        """
        Sample voxel features at given normalized coordinates.

        Args:
            embedding_tensor: [B, C, D, H, W]
            pos_zyx: [B, N, 3] in [-1, 1], (z, y, x)
            dim: spatial dimension (only 3 is supported)

        Returns:
            [B, C, N]
        """
        if dim != 3:
            raise ValueError("This implementation assumes dim=3.")

        B, C, D, H, W = embedding_tensor.shape

        z = torch.round((pos_zyx[..., 0] + 1) * 0.5 * (D - 1)).long().clamp(0, D - 1)
        y = torch.round((pos_zyx[..., 1] + 1) * 0.5 * (H - 1)).long().clamp(0, H - 1)
        x = torch.round((pos_zyx[..., 2] + 1) * 0.5 * (W - 1)).long().clamp(0, W - 1)

        fm = embedding_tensor.view(B, C, D * H * W)
        lin = (z * (H * W) + y * W + x).unsqueeze(1).expand(-1, C, -1)  # [B, C, N]
        out = torch.gather(fm, 2, lin)  # [B, C, N]

        # Keep original behavior (no functional change).
        return out.as_tensor() if hasattr(out, "as_tensor") else out

    @torch.no_grad()
    def _concat_all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """All-gather tensors in DDP to keep queues consistent across ranks."""
        if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
            return x
        world_size = torch.distributed.get_world_size()
        xs = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(xs, x.contiguous())
        return torch.cat(xs, dim=0)

    @torch.no_grad()
    def _enqueue(self, keys: torch.Tensor, which: str):
        """
        Enqueue keys into the selected queue.

        Args:
            keys: [M, C] (will be L2-normalized)
            which: "student" or "teacher"
        """
        keys = self._concat_all_gather(keys)
        keys = F.normalize(keys, dim=1)

        if which == "student":
            queue, ptr_buf = self.queue_student, self.ptr_student
        elif which == "teacher":
            queue, ptr_buf = self.queue_teacher, self.ptr_teacher
        else:
            raise ValueError(which)

        K = queue.shape[1]
        ptr = int(ptr_buf.item())
        M = keys.shape[0]

        if M >= K:
            keys = keys[:K]
            M = K

        end = ptr + M
        if end <= K:
            queue[:, ptr:end] = keys.T
        else:
            first = K - ptr
            queue[:, ptr:] = keys[:first].T
            queue[:, : end - K] = keys[first:].T

        ptr_buf[0] = (ptr + M) % K

    # -------------------------
    # MoCo InfoNCE with queue
    # -------------------------
    def _moco_loss(self, q_map, k_map, pos_q, pos_k, which: str, T: float, dim=3):
        """
        MoCo InfoNCE loss using a memory queue.

        Args:
            q_map: [B, C, D, H, W] from *_q branch (trainable)
            k_map: [B, C, D, H, W] from *_k branch (momentum, no grad)
            pos_q/pos_k: [B, N, 3] normalized in [-1, 1]
            which: selects queue ("student" or "teacher")
            T: temperature
        """
        q = self._get_feature_vector(q_map, pos_q, dim=dim)  # [B, C, N]
        k = self._get_feature_vector(k_map, pos_k, dim=dim)  # [B, C, N]

        # [B, C, N] -> [M, C]
        q = F.normalize(q.permute(0, 2, 1).reshape(-1, q.shape[1]), dim=1)
        k = F.normalize(k.permute(0, 2, 1).reshape(-1, k.shape[1]), dim=1)

        l_pos = torch.sum(q * k, dim=1, keepdim=True)  # [M, 1]

        queue = self.queue_student if which == "student" else self.queue_teacher
        l_neg = torch.mm(q, queue.detach())  # [M, K]

        logits = torch.cat([l_pos, l_neg], dim=1) / T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # Update queue with current keys.
        with torch.no_grad():
            if self.training:
                self._enqueue(k.detach(), which=which)

        return loss

    # -------------------------
    # bank-distribution distillation
    # -------------------------
    def _bank_distill(self, q_teacher_map, q_student_map, pos_t, pos_s, dim=3):
        """
        KL distillation between teacher/student similarity distributions w.r.t their banks.

        Teacher distribution:
            q_teacher vs teacher_queue
        Student distribution:
            q_student vs student_queue

        To compute KL, both distributions must share the same dimension, so we
        truncate to K = min(K_teacher, K_student).
        """
        qt = self._get_feature_vector(q_teacher_map, pos_t, dim=dim)  # [B, C, N]
        qs = self._get_feature_vector(q_student_map, pos_s, dim=dim)  # [B, C, N]

        qt = F.normalize(qt.permute(0, 2, 1).reshape(-1, qt.shape[1]), dim=1)  # [M, C]
        qs = F.normalize(qs.permute(0, 2, 1).reshape(-1, qs.shape[1]), dim=1)  # [M, C]

        logit_t = torch.mm(qt, self.queue_teacher.clone().detach())  # [M, Kt]
        logit_s = torch.mm(qs, self.queue_student.clone().detach())  # [M, Ks]

        K = min(logit_t.shape[1], logit_s.shape[1])
        logit_t = logit_t[:, :K]
        logit_s = logit_s[:, :K]

        T = self.bank_distill_T
        log_p_s = F.log_softmax(logit_s / T, dim=1)
        with torch.no_grad():
            p_t = F.softmax(logit_t / T, dim=1)

        kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T**2)
        return kl

    # -------------------------
    # forward
    # -------------------------
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
        Symmetric MoCo on two patch pairs:
          - pair 1: 11 <-> 12
          - pair 2: 21 <-> 22
        """
        # =========================================================
        # (1) Student MoCo (symmetric q <-> k)
        # =========================================================
        loss_moco_student = (
            self._moco_loss(
                q_map=embedding_11["student_q"],
                k_map=embedding_12["student_k"],
                pos_q=positive_pos_1[:, :, 0, :],
                pos_k=positive_pos_1[:, :, 1, :],
                which="student",
                T=self.T_student,
                dim=dim,
            )
            + self._moco_loss(
                q_map=embedding_12["student_q"],
                k_map=embedding_11["student_k"],
                pos_q=positive_pos_1[:, :, 1, :],
                pos_k=positive_pos_1[:, :, 0, :],
                which="student",
                T=self.T_student,
                dim=dim,
            )
            + self._moco_loss(
                q_map=embedding_21["student_q"],
                k_map=embedding_22["student_k"],
                pos_q=positive_pos_2[:, :, 0, :],
                pos_k=positive_pos_2[:, :, 1, :],
                which="student",
                T=self.T_student,
                dim=dim,
            )
            + self._moco_loss(
                q_map=embedding_22["student_q"],
                k_map=embedding_21["student_k"],
                pos_q=positive_pos_2[:, :, 1, :],
                pos_k=positive_pos_2[:, :, 0, :],
                which="student",
                T=self.T_student,
                dim=dim,
            )
        ) * (self.wight_moco_student / 4.0)

        # =========================================================
        # (2) Teacher MoCo (symmetric q <-> k)
        # =========================================================
        loss_moco_teacher = (
            self._moco_loss(
                q_map=embedding_11["teacher_q"],
                k_map=embedding_12["teacher_k"],
                pos_q=positive_pos_1[:, :, 0, :],
                pos_k=positive_pos_1[:, :, 1, :],
                which="teacher",
                T=self.T_teacher,
                dim=dim,
            )
            + self._moco_loss(
                q_map=embedding_12["teacher_q"],
                k_map=embedding_11["teacher_k"],
                pos_q=positive_pos_1[:, :, 1, :],
                pos_k=positive_pos_1[:, :, 0, :],
                which="teacher",
                T=self.T_teacher,
                dim=dim,
            )
            + self._moco_loss(
                q_map=embedding_21["teacher_q"],
                k_map=embedding_22["teacher_k"],
                pos_q=positive_pos_2[:, :, 0, :],
                pos_k=positive_pos_2[:, :, 1, :],
                which="teacher",
                T=self.T_teacher,
                dim=dim,
            )
            + self._moco_loss(
                q_map=embedding_22["teacher_q"],
                k_map=embedding_21["teacher_k"],
                pos_q=positive_pos_2[:, :, 1, :],
                pos_k=positive_pos_2[:, :, 0, :],
                which="teacher",
                T=self.T_teacher,
                dim=dim,
            )
        ) * (self.wight_moco_teacher / 4.0)

        # =========================================================
        # (3) Bank distribution distillation
        # =========================================================
        loss_bank_distill = (
            self._bank_distill(
                embedding_11["teacher_q"],
                embedding_11["student_q"],
                positive_pos_1[:, :, 0, :],
                positive_pos_1[:, :, 0, :],
                dim,
            )
            + self._bank_distill(
                embedding_12["teacher_q"],
                embedding_12["student_q"],
                positive_pos_1[:, :, 1, :],
                positive_pos_1[:, :, 1, :],
                dim,
            )
            + self._bank_distill(
                embedding_21["teacher_q"],
                embedding_21["student_q"],
                positive_pos_2[:, :, 0, :],
                positive_pos_2[:, :, 0, :],
                dim,
            )
            + self._bank_distill(
                embedding_22["teacher_q"],
                embedding_22["student_q"],
                positive_pos_2[:, :, 1, :],
                positive_pos_2[:, :, 1, :],
                dim,
            )
        ) * (self.wight_bank_distill / 4.0)

        total = loss_moco_student + loss_moco_teacher + loss_bank_distill

        return {
            "total_loss": total,
            "moco_student_loss": loss_moco_student,
            "moco_teacher_loss": loss_moco_teacher,
            "bank_distill_loss": loss_bank_distill,
        }