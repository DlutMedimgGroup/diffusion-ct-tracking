# engine/trainer_vae.py

import os
import time
import math
import torch
import torch.distributed as dist
from collections import defaultdict
from .trainer_base import TrainerBase
from torch.utils.tensorboard import SummaryWriter

class trainer_AE(TrainerBase):

    def __init__(self, config: dict, is_ddp: bool = False):
        """初始化训练器。"""
        super().__init__(config, is_ddp)
        
        self.amp_enabled = self.run_config.get('train_amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.writer = SummaryWriter(log_dir=self.output_dir) if self.rank == 0 else None
        
        self.best_val_loss = float('inf')
        self.pth_best_model = ""

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
                    embedding_1 = self.model(image[:,:1,...])
                    embedding_2 = self.model(image[:,1:,...])
                    embedding_1 = embedding_1['correspondence_output']
                    embedding_2 = embedding_2['correspondence_output']

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

    def train_epoch(self, epoch: int):
        """运行单次训练。"""
        avg_losses = self._run_epoch(epoch, is_training=True)
        
        if self.rank == 0:
            # 动态地将所有损失项写入TensorBoard
            if self.writer:
                for key, value in avg_losses.items():
                    if key != 'time': 
                        self.writer.add_scalar(f"train/{key}", value, epoch)

                self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], epoch)

            # 动态构建日志字符串
            # 使用 'total_loss' 或 'loss' 来显示主要损失
            main_loss_key = 'total_loss' if 'total_loss' in avg_losses else 'loss'
            log_str_parts = [f"Loss: {avg_losses.get(main_loss_key, 0):.4f}"]
            
            # 添加其他子损失到日志
            for key, val in avg_losses.items():
                if key not in [main_loss_key, 'time', 'perplexity']:
                    log_str_parts.append(f"{key.replace('_', ' ').title()}: {val:.4f}")

            # 如果存在，添加Perplexity
            if 'perplexity' in avg_losses:
                log_str_parts.append(f"Perplexity: {avg_losses['perplexity']:.2f}")

            log_str = " | ".join(log_str_parts)
            self.logger.info(f"  Avg. Train: {log_str} | Time: {avg_losses['time']:.2f}s")

    def validate_epoch(self, epoch: int):
        """运行单次验证。"""
        if self.rank == 0:
            self._save_checkpoint(epoch, "latest")

    def _save_checkpoint(self, epoch: int, save_type: str):
        model_state = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
        checkpoint = { 'epoch': epoch, 'model_state_dict': model_state, 'optimizer_state_dict': self.optimizer.state_dict(), 'scheduler_state_dict': self.lr_scheduler.state_dict(), 'best_val_loss': self.best_val_loss, 'pth_best_model': self.pth_best_model }
        if save_type == "latest": torch.save(checkpoint, os.path.join(self.output_dir, "checkpoint_latest.pth"))
        elif save_type == "best_loss":
            if self.pth_best_model and os.path.exists(self.pth_best_model): os.remove(self.pth_best_model)
            filename = f"checkpoint_best_loss_epoch{epoch}_loss{self.best_val_loss:.4f}.pth"
            self.pth_best_model = os.path.join(self.output_dir, filename)
            torch.save(checkpoint, self.pth_best_model)
            return f"    -> Saved new best model: {filename}"
        return None