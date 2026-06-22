# engine/trainer_vae.py

import os
import time
import math
from tqdm import tqdm
import torch
import torch.distributed as dist
from collections import defaultdict
from trainer.trainer_AE import trainer_AE
from torch.utils.tensorboard import SummaryWriter

class trainer_AE_cross(trainer_AE):

    def __init__(self, config: dict, is_ddp: bool = False):
        """初始化训练器。"""
        super().__init__(config, is_ddp)

        self.max_steps_per_epoch = self.config['Run'].get('max_steps_per_epoch', -1)


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
        step_count = 0
        pbar = tqdm(loader, desc=f"{'Train' if is_training else 'Val'} Epoch {epoch}", ncols=120)
        for batch_data in pbar:
            step_count += 1
            image = batch_data['image'].to(self.device)            
            # --- 前向传播 ---
            with torch.set_grad_enabled(is_training):
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    embedding_11 = self.model(image[:,:1,...])
                    embedding_12 = self.model(image[:,1:2,...])
                    embedding_21 = self.model(image[:,2:3,...])
                    embedding_22 = self.model(image[:,3:,...])

                    positive_pos_1 = batch_data['image_meta_dict_0']['positive_pos'].to(self.device).detach()
                    positive_pos_2 = batch_data['image_meta_dict_1']['positive_pos'].to(self.device).detach()

                    loss_dict = self.loss_fn(embedding_11, 
                                             embedding_12, 
                                             embedding_21, 
                                             embedding_22,
                                             positive_pos_1, 
                                             positive_pos_2, dim=3)
                    
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

            # ★ 更新 tqdm 进度条显示当前 loss ★
            pbar.set_postfix({k: f"{v/step_count:.4f}" for k, v in epoch_losses.items()})

            if self.max_steps_per_epoch > 0 and step_count > self.max_steps_per_epoch:
                break
        
        if is_training:
            self.lr_scheduler.step()
        
        # --- 计算平均损失 ---
        avg_losses = {key: value / step_count for key, value in epoch_losses.items()}
        avg_losses['time'] = time.time() - start_time
        
        return avg_losses

    