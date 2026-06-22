import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.loss_base import LossBase


class loss_AE_moco(LossBase):
    """
    Student-only MoCo loss on patch correspondence features.

    Inputs (expected by forward):
      embedding_11/12/21/22: dict with keys:
        - "student_q": [B,C,D,H,W]
        - "student_k": [B,C,D,H,W]   (momentum branch, no grad)
      positive_pos_1/2: [B,N,2,3] in [-1,1] (z,y,x), where [:,:,0,:] matches query in first patch,
                                            [:,:,1,:] matches key in second patch.

    Output:
      dict(total_loss, moco_student_loss, optional monitor stats)
    """

    def __init__(self, config: dict):
        self._load_configs(config)
        super().__init__(config)

        self.wight_moco_student = float(self.loss_config.get("wight_moco_student", 1.0))
        self.T_student = float(self.loss_config.get("moco_T_student", 0.2))

        self.queue_size_student = int(self.loss_config.get("queue_size_student", 65536))
        self.feat_dim = int(
            self.loss_config.get(
                "feat_dim",
                config.get("Model", {}).get("target_feature_size", 32),
            )
        )

        # queue: [C, K]
        q = torch.randn(self.feat_dim, self.queue_size_student)
        self.register_buffer("queue_student", F.normalize(q, dim=0))
        self.register_buffer("ptr_student", torch.zeros(1, dtype=torch.long))

        # if you ever want voxel coords instead of [-1,1] coords
        self.pos_is_normalized = bool(self.loss_config.get("pos_is_normalized", True))

    def _get_loss_name(self):
        return "loss_AE_moco"

    # -------------------------
    # DDP helper
    # -------------------------
    @torch.no_grad()
    def _concat_all_gather(self, x: torch.Tensor) -> torch.Tensor:
        if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
            return x
        world_size = torch.distributed.get_world_size()
        xs = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(xs, x.contiguous())
        return torch.cat(xs, dim=0)

    @torch.no_grad()
    def _enqueue_student(self, keys: torch.Tensor):
        """
        keys: [M, C] (will be gathered & normalized)
        """
        keys = self._concat_all_gather(keys)
        keys = F.normalize(keys, dim=1)

        queue = self.queue_student
        ptr_buf = self.ptr_student

        K = queue.shape[1]
        ptr = int(ptr_buf.item())
        M = int(keys.shape[0])

        if M <= 0:
            return

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
    # pos -> feature vectors
    # -------------------------
    def _get_feature_vector(self, embedding_tensor: torch.Tensor, pos_zyx: torch.Tensor, dim=3) -> torch.Tensor:
        """
        embedding_tensor: [B,C,D,H,W]
        pos_zyx:
          - if pos_is_normalized=True: [B,N,3] in [-1,1] (z,y,x)
          - else: [B,N,3] in voxel index space (z,y,x)
        return: [B,C,N]
        """
        if dim != 3:
            raise ValueError("This implementation assumes dim=3.")
        B, C, D, H, W = embedding_tensor.shape

        if self.pos_is_normalized:
            z = torch.round((pos_zyx[..., 0] + 1) * 0.5 * (D - 1))
            y = torch.round((pos_zyx[..., 1] + 1) * 0.5 * (H - 1))
            x = torch.round((pos_zyx[..., 2] + 1) * 0.5 * (W - 1))
        else:
            z, y, x = pos_zyx[..., 0], pos_zyx[..., 1], pos_zyx[..., 2]

        z = z.long().clamp(0, D - 1)
        y = y.long().clamp(0, H - 1)
        x = x.long().clamp(0, W - 1)

        fm = embedding_tensor.view(B, C, D * H * W)
        lin = (z * (H * W) + y * W + x).unsqueeze(1).expand(-1, C, -1)  # [B,C,N]
        out = torch.gather(fm, 2, lin)  # [B,C,N]
        return out

    # -------------------------
    # MoCo InfoNCE with queue
    # -------------------------
    def _moco_loss_student(self, q_map, k_map, pos_q, pos_k, T: float, dim=3):
        """
        q_map: [B,C,D,H,W] from student_q (trainable)
        k_map: [B,C,D,H,W] from student_k (momentum, no grad)
        pos_*: [B,N,3]
        """
        q = self._get_feature_vector(q_map, pos_q, dim=dim)  # [B,C,N]
        k = self._get_feature_vector(k_map, pos_k, dim=dim)  # [B,C,N]

        # [B,C,N] -> [M,C]
        q = F.normalize(q.permute(0, 2, 1).reshape(-1, q.shape[1]), dim=1)
        k = F.normalize(k.permute(0, 2, 1).reshape(-1, k.shape[1]), dim=1)

        # positive logits: [M,1]
        l_pos = torch.sum(q * k, dim=1, keepdim=True)

        # negative logits: [M,K]
        l_neg = torch.mm(q, self.queue_student.detach())

        logits = torch.cat([l_pos, l_neg], dim=1) / float(T)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        # monitor
        with torch.no_grad():
            top1 = (logits.argmax(dim=1) == 0).float().mean()
            pos_mean = l_pos.mean()
            neg_mean = l_neg.mean()

        # enqueue keys
        if self.training:
            with torch.no_grad():
                self._enqueue_student(k.detach())

        return loss, top1, pos_mean, neg_mean

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
        symmetric MoCo:
          pair1: 11 <-> 12
          pair2: 21 <-> 22
        """
        loss_sum = 0.0
        acc_sum = 0.0
        posm_sum = 0.0
        negm_sum = 0.0

        # pair 1
        l, a, pm, nm = self._moco_loss_student(
            q_map=embedding_11["student_q"],
            k_map=embedding_12["student_k"],
            pos_q=positive_pos_1[:, :, 0, :],
            pos_k=positive_pos_1[:, :, 1, :],
            T=self.T_student,
            dim=dim,
        )
        loss_sum = loss_sum + l; acc_sum += a; posm_sum += pm; negm_sum += nm

        l, a, pm, nm = self._moco_loss_student(
            q_map=embedding_12["student_q"],
            k_map=embedding_11["student_k"],
            pos_q=positive_pos_1[:, :, 1, :],
            pos_k=positive_pos_1[:, :, 0, :],
            T=self.T_student,
            dim=dim,
        )
        loss_sum = loss_sum + l; acc_sum += a; posm_sum += pm; negm_sum += nm

        # pair 2
        l, a, pm, nm = self._moco_loss_student(
            q_map=embedding_21["student_q"],
            k_map=embedding_22["student_k"],
            pos_q=positive_pos_2[:, :, 0, :],
            pos_k=positive_pos_2[:, :, 1, :],
            T=self.T_student,
            dim=dim,
        )
        loss_sum = loss_sum + l; acc_sum += a; posm_sum += pm; negm_sum += nm

        l, a, pm, nm = self._moco_loss_student(
            q_map=embedding_22["student_q"],
            k_map=embedding_21["student_k"],
            pos_q=positive_pos_2[:, :, 1, :],
            pos_k=positive_pos_2[:, :, 0, :],
            T=self.T_student,
            dim=dim,
        )
        loss_sum = loss_sum + l; acc_sum += a; posm_sum += pm; negm_sum += nm

        loss_moco_student = loss_sum * (self.wight_moco_student / 4.0)

        return {
            "total_loss": loss_moco_student,
            "moco_student_loss": loss_moco_student,

            # monitors (averaged over the 4 symmetric terms)
            "moco_top1@pos_is_class0": (acc_sum / 4.0).detach(),
            "pos_logit_mean": (posm_sum / 4.0).detach(),
            "neg_logit_mean": (negm_sum / 4.0).detach(),
            "queue_ptr": self.ptr_student.detach().clone(),
        }