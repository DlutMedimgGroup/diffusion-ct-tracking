# engine/trainer_vae.py

import os
import time
import math
import torch
import torch.distributed as dist
from collections import defaultdict
from trainer.trainer_AE import trainer_AE
from torch.utils.tensorboard import SummaryWriter

class trainer_AE_local(trainer_AE):

    def __init__(self, config: dict, is_ddp: bool = False):
        """初始化训练器。"""
        super().__init__(config, is_ddp)

    def _run_epoch(self, epoch: int, is_training: bool):
        """
        运行一个完整的 epoch，无论是训练还是验证。
        它能动态处理并累加损失字典中的所有项。
        """
        self.model.train(is_training)
        
        # 在DDP训练模式下，为sampler设置epoch
        if is_training and self.is_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
        loader = self.train_loader if is_training else self.val_loader
        
        # 使用 defaultdict 来自动处理新的损失键
        epoch_losses = defaultdict(float)
        step_count = 0
        start_time = time.time()

        # 循环的核心逻辑
        for batch_data in loader:
            step_count += 1
            image = batch_data['image'].to(self.device)            
            # --- 前向传播 ---
            with torch.set_grad_enabled(is_training):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    embedding_1, embedding_2 = self.model(image[:,:1,...], image[:,1:,...])

                    meta_dict = batch_data['image_meta_dict']
                    positive_pos = meta_dict['positive_pos'].to(self.device).detach()
                    loss_dict = self.loss_fn(embedding_1, embedding_2, positive_pos[:,:,0,:], positive_pos[:,:,1,:], dim=3)
                    
            # --- 反向传播 (仅在训练时) ---
            if is_training:
                self.optimizer.zero_grad()
                total_loss_key = 'total_loss' if 'total_loss' in loss_dict else 'loss'
                if total_loss_key not in loss_dict:
                    raise KeyError(f"损失字典中找不到 'total_loss' 或 'loss' 键。可用的键: {loss_dict.keys()}")

                self.scaler.scale(loss_dict[total_loss_key]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # --- 累加所有损失项 ---
            for key, value in loss_dict.items():
                # 确保值是标量
                if torch.is_tensor(value):
                    epoch_losses[key] += value.item()
        
        if is_training:
            self.lr_scheduler.step()
        
        # --- 计算平均损失 ---
        avg_losses = {key: value / step_count for key, value in epoch_losses.items()}
        avg_losses['time'] = time.time() - start_time
        
        return avg_losses
    