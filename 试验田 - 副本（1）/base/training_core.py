"""
训练核心模块
整合SAM训练器、PANet训练器、损失函数和训练工具
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
try:
    from torch.amp import GradScaler, autocast
    # 新版本PyTorch
    NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    # 旧版本PyTorch
    NEW_AMP_API = False
from torch.utils.checkpoint import checkpoint
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import json
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# 检测PyTorch版本并设置兼容性
PYTORCH_VERSION = torch.__version__
IS_PYTORCH_2_PLUS = int(PYTORCH_VERSION.split('.')[0]) >= 2

# 全局抑制PyTorch相关的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.amp")

# TensorBoard支持
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboard import SummaryWriter
    except ImportError:
        # 如果没有tensorboard，创建一个虚拟的SummaryWriter
        class DummySummaryWriter:
            def __init__(self, *args, **kwargs):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def add_image(self, *args, **kwargs):
                pass
            def add_histogram(self, *args, **kwargs):
                pass
            def close(self):
                pass
        SummaryWriter = DummySummaryWriter

# ============================================================================
# 损失函数
# ============================================================================

class BoundaryDiceLoss(nn.Module):
    """
    边界感知Dice损失函数 - 专门解决心脏超声边界模糊问题
    基于医学图像分割的专业实践
    """
    
    def __init__(self, num_classes: int = 5, boundary_weight: float = 0.8):
        super(BoundaryDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
    
    def extract_boundaries(self, target: torch.Tensor) -> torch.Tensor:
        """提取边界区域（使用形态学操作，避免Canny的复杂性）"""
        boundaries = []
        for i in range(target.size(0)):
            # 转换为numpy进行边界提取
            mask = target[i].cpu().numpy().astype(np.uint8)
            
            # 对每个类别提取边界
            boundary = np.zeros_like(mask)
            for class_id in range(1, self.num_classes):  # 跳过背景
                class_mask = (mask == class_id).astype(np.uint8)
                if class_mask.sum() == 0:
                    continue
                
                # 使用形态学操作提取边界
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eroded = cv2.erode(class_mask, kernel, iterations=1)
                boundary_region = class_mask - eroded
                boundary[boundary_region > 0] = class_id
            
            boundaries.append(boundary)
        
        return torch.from_numpy(np.array(boundaries)).long().to(target.device)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测结果 (B, C, H, W)
            target: 真实标签 (B, H, W)
        """
        # 1. 转换为one-hot编码
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        pred_soft = F.softmax(pred, dim=1)
        
        # 2. 提取边界区域
        boundary_mask = self.extract_boundaries(target)  # (B, H, W)
        
        # 3. 创建边界权重图
        boundary_weight_map = torch.zeros_like(target).float()
        for class_id in range(1, self.num_classes):
            boundary_weight_map[boundary_mask == class_id] = 1.0
        
        # 4. 计算标准Dice
        intersection_std = (pred_soft * target_onehot).sum(dim=(2, 3))
        union_std = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice_std = (2.0 * intersection_std + 1e-6) / (union_std + 1e-6)
        
        # 5. 计算边界Dice（仅在边界区域）
        boundary_weight_expanded = boundary_weight_map.unsqueeze(1).expand_as(pred_soft)
        pred_boundary = pred_soft * boundary_weight_expanded
        target_boundary = target_onehot * boundary_weight_expanded
        
        intersection_boundary = (pred_boundary * target_boundary).sum(dim=(2, 3))
        union_boundary = pred_boundary.sum(dim=(2, 3)) + target_boundary.sum(dim=(2, 3))
        dice_boundary = (2.0 * intersection_boundary + 1e-6) / (union_boundary + 1e-6)
        
        # 6. 组合损失：重点关注边界
        dice_loss_std = 1 - dice_std.mean()
        dice_loss_boundary = 1 - dice_boundary.mean()
        
        # 边界损失权重更高
        total_loss = (1 - self.boundary_weight) * dice_loss_std + self.boundary_weight * dice_loss_boundary
        
        return total_loss

class FocalDiceLoss(nn.Module):
    """
    Focal Dice损失函数
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 dice_weight: float = 0.7,
                 ce_weight: float = 0.3,
                 class_weights: Optional[torch.Tensor] = None,
                 smooth: float = 1e-6,
                 use_hard_mining: bool = True,
                 mining_threshold: float = 0.3):
        """
        初始化Focal Dice损失
        
        Args:
            num_classes: 类别数量
            focal_alpha: Focal loss的alpha参数
            focal_gamma: Focal loss的gamma参数
            dice_weight: Dice损失的权重
            ce_weight: 交叉熵损失的权重
            class_weights: 类别权重
            smooth: 平滑因子
            use_hard_mining: 是否使用困难样本挖掘
            mining_threshold: 困难样本挖掘阈值
        """
        super(FocalDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.class_weights = class_weights
        self.smooth = smooth
        self.use_hard_mining = use_hard_mining
        self.mining_threshold = mining_threshold
              
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 预测结果 [B, C, H, W]
            targets: 真实标签 [B, H, W]
        
        Returns:
            损失值
        """
        # 确保输入格式正确
        if len(inputs.shape) == 4 and inputs.shape[1] != self.num_classes:
            inputs = inputs.permute(0, 3, 1, 2)
        
        # 计算交叉熵
        ce_loss = self.ce_loss(inputs, targets)
        
        # 计算Focal Loss
        focal_loss = self._focal_loss(inputs, targets)
        
        # 计算Dice Loss
        dice_loss = self._dice_loss(inputs, targets)
        
        # 组合损失
        total_loss = (self.ce_weight * ce_loss + 
                     self.focal_alpha * focal_loss + 
                     self.dice_weight * dice_loss)
        
        return total_loss
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Focal Loss"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 计算focal权重
        focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
        
        # 应用困难样本挖掘
        if self.use_hard_mining:
            # 选择困难样本
            hard_mask = ce_loss > self.mining_threshold
            focal_weight = focal_weight * hard_mask.float()
        
        return (focal_weight * ce_loss).mean()
    
    def _dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss - 数值稳定版本
        """
        # 检查输入是否有效
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("⚠️ 检测到无效的输入logits，使用备用计算")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # 转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # 使用数值稳定的softmax
        inputs_soft = F.softmax(inputs, dim=1)
        
        # 检查softmax结果
        if torch.isnan(inputs_soft).any():
            print("⚠️ softmax产生NaN，使用备用计算")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # 计算Dice系数 - 增强数值稳定性
        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # 增大smooth项防止除零
        dice = (2.0 * intersection + self.smooth * 10) / (union + self.smooth * 10)
        
        # 检查dice结果
        if torch.isnan(dice).any():
            print("⚠️ dice计算产生NaN，使用备用值")
            return torch.tensor(0.5, device=inputs.device, requires_grad=True)
        
        dice_loss = 1 - dice.mean()
        
        # 最终检查
        if torch.isnan(dice_loss) or torch.isinf(dice_loss):
            return torch.tensor(0.5, device=inputs.device, requires_grad=True)
        
        return dice_loss

class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合多种损失函数，提供更稳定的训练
    """
    
    def __init__(self,
                 num_classes: int = 5,
                 ce_weight: float = 0.4,
                 dice_weight: float = 0.4,
                 focal_weight: float = 0.2,
                 class_weights: Optional[torch.Tensor] = None,
                 smooth: float = 1e-6):
        """
        初始化组合损失
        
        Args:
            num_classes: 类别数量
            ce_weight: 交叉熵损失权重
            dice_weight: Dice损失权重
            focal_weight: Focal损失权重
            class_weights: 类别权重
            smooth: 平滑因子
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 预测结果 [B, C, H, W]
            targets: 真实标签 [B, H, W]
        
        Returns:
            损失值
        """
        ce_loss = self.ce_loss(inputs, targets)
        
        # Dice损失
        dice_loss = self._dice_loss(inputs, targets)
        
        # Focal损失
        focal_loss = self._focal_loss(inputs, targets)
        
        # 组合损失
        total_loss = (self.ce_weight * ce_loss + 
                     self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss
    
    def _dice_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Dice Loss - 数值稳定版本
        """
        # 检查输入是否有效
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print("⚠️ 检测到无效的输入logits，使用备用计算")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # 转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # 使用数值稳定的softmax
        inputs_soft = F.softmax(inputs, dim=1)
        
        # 检查softmax结果
        if torch.isnan(inputs_soft).any():
            print("⚠️ softmax产生NaN，使用备用计算")
            return torch.tensor(1.0, device=inputs.device, requires_grad=True)
        
        # 计算Dice系数 - 增强数值稳定性
        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # 增大smooth项防止除零
        dice = (2.0 * intersection + self.smooth * 10) / (union + self.smooth * 10)
        
        # 检查dice结果
        if torch.isnan(dice).any():
            print("⚠️ dice计算产生NaN，使用备用值")
            return torch.tensor(0.5, device=inputs.device, requires_grad=True)
        
        dice_loss = 1 - dice.mean()
        
        # 最终检查
        if torch.isnan(dice_loss) or torch.isinf(dice_loss):
            return torch.tensor(0.5, device=inputs.device, requires_grad=True)
        
        return dice_loss
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算Focal Loss"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # 计算focal权重
        focal_weight = (1 - pt) ** 2
        
        return (focal_weight * ce_loss).mean()

class DiversityEnhancedLoss(nn.Module):
    """
    多样性增强损失函数 - 解决单一类别预测问题
    """
    
    def __init__(self, main_loss, diversity_weight: float = 0.1):
        super().__init__()
        self.main_loss = main_loss
        self.diversity_weight = diversity_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 预测结果 [B, C, H, W]
            targets: 真实标签 [B, H, W]
        
        Returns:
            组合损失值
        """
        # 主损失
        main_loss = self.main_loss(inputs, targets)
        
        # 多样性正则化损失
        diversity_loss = self._diversity_regularization(inputs)
        
        # 组合损失
        total_loss = main_loss + self.diversity_weight * diversity_loss
        
        return total_loss
    
    def _diversity_regularization(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        多样性正则化 - 惩罚单一类别预测
        """
        # 计算预测概率
        probs = F.softmax(inputs, dim=1)  # [B, C, H, W]
        
        # 计算每个类别的平均预测概率
        class_probs = probs.mean(dim=(0, 2, 3))  # [C]
        
        # 目标：每个类别都应该有一定的预测概率
        target_prob = 1.0 / inputs.shape[1]  # 均匀分布
        
        # 使用KL散度惩罚偏离均匀分布
        uniform_dist = torch.full_like(class_probs, target_prob)
        kl_div = F.kl_div(
            torch.log(class_probs + 1e-8), 
            uniform_dist, 
            reduction='sum'
        )
        
        # 额外的熵正则化 - 鼓励预测的不确定性
        entropy_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        entropy_regularization = -entropy_loss  # 负号因为我们想最大化熵
        
        return kl_div + 0.1 * entropy_regularization

# ============================================================================
# 训练工具类
# ============================================================================

class TrainingUtils:
    """
    训练工具类
    提供训练过程中的各种辅助功能
    """
    
    def __init__(self):
        """初始化训练工具"""
        pass
    
    @staticmethod
    def save_checkpoint(model: nn.Module,
                       optimizer,
                       scheduler,
                       epoch: int,
                       loss: float,
                       save_path: str,
                       **kwargs):
        """
        保存训练检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            loss: 当前损失
            save_path: 保存路径
            **kwargs: 其他需要保存的信息
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            **kwargs
        }
        
        torch.save(checkpoint, save_path)
        print(f"检查点已保存: {save_path}")
    
    @staticmethod
    def load_checkpoint(model: nn.Module,
                       optimizer=None,
                       scheduler=None,
                       checkpoint_path: str = None,
                       device: str = 'cuda'):
        """
        加载训练检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            checkpoint_path: 检查点路径
            device: 设备
        
        Returns:
            加载的epoch和损失
        """
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return 0, float('inf')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"检查点已加载: {checkpoint_path}, epoch={epoch}, loss={loss:.4f}")
        return epoch, loss
    
    @staticmethod
    def calculate_metrics(predictions: torch.Tensor, 
                         targets: torch.Tensor,
                         num_classes: int = 5) -> Dict[str, float]:
        """
        计算评估指标 - 统一使用argmax方法
        
        Args:
            predictions: 预测结果 [B, H, W] 或 [B, C, H, W] 或 [B, 1, H, W]
            targets: 真实标签 [B, H, W]
            num_classes: 类别数量
        
        Returns:
            指标字典
        """
        # 转换为numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # 处理不同的输入格式
        if len(predictions.shape) == 4:
            if predictions.shape[1] == 1:
                # [B, 1, H, W] -> [B, H, W]
                pred_classes = predictions.squeeze(1).astype(int)
            else:
                # [B, C, H, W] -> [B, H, W]
                pred_classes = np.argmax(predictions, axis=1)
        else:
            # [B, H, W]
            pred_classes = predictions.astype(int)
        
        # 计算Dice系数
        dice_scores = []
        for i in range(num_classes):
            pred_mask = (pred_classes == i)
            target_mask = (targets == i)
            
            if target_mask.sum() > 0:
                intersection = (pred_mask & target_mask).sum()
                union = pred_mask.sum() + target_mask.sum()
                dice = 2.0 * intersection / (union + 1e-8)
                dice_scores.append(dice)
            else:
                dice_scores.append(0.0)
        
        # 计算平均Dice
        mean_dice = np.mean(dice_scores)
        
        # 计算像素准确率
        pixel_accuracy = (pred_classes == targets).mean()
        
        return {
            'mean_dice': mean_dice,
            'pixel_accuracy': pixel_accuracy,
            'dice_per_class': dice_scores
        }
    
    @staticmethod
    def plot_training_curves(train_losses: List[float],
                           val_losses: List[float],
                           train_metrics: List[Dict[str, float]] = None,
                           val_metrics: List[Dict[str, float]] = None,
                           save_path: str = None):
        """
        绘制训练曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_metrics: 训练指标列表
            val_metrics: 验证指标列表
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(train_losses, label='Train Loss')
        axes[0, 0].plot(val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice系数曲线
        if train_metrics and val_metrics:
            train_dice = [m['mean_dice'] for m in train_metrics]
            val_dice = [m['mean_dice'] for m in val_metrics]
            
            axes[0, 1].plot(train_dice, label='Train Dice')
            axes[0, 1].plot(val_dice, label='Val Dice')
            axes[0, 1].set_title('Mean Dice Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 像素准确率曲线
        if train_metrics and val_metrics:
            train_acc = [m['pixel_accuracy'] for m in train_metrics]
            val_acc = [m['pixel_accuracy'] for m in val_metrics]
            
            axes[1, 0].plot(train_acc, label='Train Accuracy')
            axes[1, 0].plot(val_acc, label='Val Accuracy')
            axes[1, 0].set_title('Pixel Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 学习率曲线（如果有的话）
        axes[1, 1].text(0.5, 0.5, 'Learning Rate Curve\n(Not Available)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        
        plt.show()

# ============================================================================
# SAM训练器
# ============================================================================

class SAMTrainer:
    """
    SAM模型训练器
    实现SAM模型的训练、验证和测试功能
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 test_loader: DataLoader = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 num_epochs: int = 100,
                 save_dir: str = 'shared/checkpoints',
                 use_amp: bool = True,
                 use_checkpoint: bool = False,
                 accumulation_steps: int = 1,
                 early_stopping_patience: int = 5,
                 criterion: nn.Module = None):
        """
        初始化SAM训练器
        
        Args:
            model: SAM模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            num_epochs: 训练轮数
            save_dir: 保存目录
            use_amp: 是否使用混合精度训练
            use_checkpoint: 是否使用梯度检查点
            accumulation_steps: 梯度累积步数
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_amp = use_amp
        self.use_checkpoint = use_checkpoint
        self.accumulation_steps = accumulation_steps
        
        # 早停机制
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        self.best_epoch = 0
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # 初始化混合精度训练
        if use_amp:
            if NEW_AMP_API:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 初始化损失函数
        if criterion is not None:
            self.criterion = criterion.to(device)
        else:
            self.criterion = self._create_loss_function()
        
        # 初始化TensorBoard - 确保输出到shared目录
        if not save_dir.startswith('shared/'):
            save_dir = f'shared/{save_dir}'
        self.writer = SummaryWriter(os.path.join(save_dir, 'logs'))
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
        # 静默初始化完成
    
    def _create_loss_function(self):
        """创建损失函数 - 关键修复：使用组合损失"""
        class_weights = self._calculate_class_weights()
        
        # **关键修复：使用组合损失函数(Dice+CE)**
        # 静默使用组合损失函数(Dice+CE)
        
        # 根据模型类别数动态创建损失函数
        if hasattr(self.model, 'num_classes'):
            num_classes = self.model.num_classes
        else:
            num_classes = 5
        
        # 创建组合损失函数
        return CombinedLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            ce_weight=0.5,  # 交叉熵权重
            dice_weight=0.5,  # Dice权重
            focal_weight=0.0  # 暂时不使用Focal
        )
        
        # 对于2类和3类，不需要多样性损失（问题较简单）
        if num_classes <= 3:
            return main_loss
        else:
            # 只有5类时才使用多样性增强
            return DiversityEnhancedLoss(main_loss, diversity_weight=0.5)
    
    def _calculate_class_weights(self):
        """计算类别权重 - 根据当前模型的类别数动态调整"""
        
        # 根据模型的实际类别数设置权重
        if hasattr(self.model, 'num_classes'):
            num_classes = self.model.num_classes
        else:
            num_classes = 5  # 默认5类
        
        if num_classes == 2:
            # 2类：背景 vs 心脏 - 合理权重
            weights = torch.tensor([0.4, 1.5], dtype=torch.float32)
            class_names = ['背景', '心脏']
        elif num_classes == 3:
            # 3类：背景 + 左心 + 右心
            weights = torch.tensor([0.4, 1.5, 1.5], dtype=torch.float32)
            class_names = ['背景', '左心', '右心']
        else:
            # 5类：完整分割
            weights = torch.tensor([
                0.5,   # 背景：适中权重
                2.0,   # 左心室：提高权重但不过度
                2.5,   # 右心室：最高权重（最稀少类别）
                2.0,   # 左心房：提高权重但不过度
                1.8    # 右心房：适中提高权重
            ], dtype=torch.float32)
            class_names = ['背景', '左心室', '右心室', '左心房', '右心房']
        
        # 静默设置类别权重
        
        return weights.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
            
            # 前向传播
            if self.use_amp and self.scaler:
                if NEW_AMP_API:
                    with autocast(device_type='cuda'):
                        # MedSAM优化：使用模型内置的默认prompt生成
                        # MedSAM会自动处理prompt生成，无需手动创建虚拟prompts
                        batch_size = images.shape[0]
                        points = None  # MedSAM会自动生成默认prompts
                        boxes = None
                        masks = None
                        
                        if self.use_checkpoint:
                            outputs = checkpoint(self.model, images, points, boxes, masks)
                        else:
                            outputs = self.model(images, points, boxes, masks)
                        
                        # MedSAM优化：处理新的输出格式
                        if isinstance(outputs, dict):
                            # 优先使用cardiac_logits，这是MedSAM的主要输出
                            outputs = outputs.get('cardiac_logits', outputs.get('predictions', outputs.get('sam_masks', outputs)))
                        
                        loss = self.criterion(outputs, labels)
                        loss = loss / self.accumulation_steps
                else:
                    with autocast():
                        # MedSAM优化：使用模型内置的默认prompt生成
                        # MedSAM会自动处理prompt生成，无需手动创建虚拟prompts
                        batch_size = images.shape[0]
                        points = None  # MedSAM会自动生成默认prompts
                        boxes = None
                        masks = None
                        
                        if self.use_checkpoint:
                            outputs = checkpoint(self.model, images, points, boxes, masks)
                        else:
                            outputs = self.model(images, points, boxes, masks)
                        
                        # MedSAM优化：处理新的输出格式
                        if isinstance(outputs, dict):
                            # 优先使用cardiac_logits，这是MedSAM的主要输出
                            outputs = outputs.get('cardiac_logits', outputs.get('predictions', outputs.get('sam_masks', outputs)))
                        
                        loss = self.criterion(outputs, labels)
                        loss = loss / self.accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # MedSAM优化：使用模型内置的默认prompt生成
                batch_size = images.shape[0]
                points = None  # MedSAM会自动生成默认prompts
                boxes = None
                masks = None
                
                if self.use_checkpoint:
                    outputs = checkpoint(self.model, images, points, boxes, masks)
                else:
                    outputs = self.model(images, points, boxes, masks)
                
                # MedSAM优化：处理新的输出格式
                if isinstance(outputs, dict):
                    # 优先使用cardiac_logits，这是MedSAM的主要输出
                    outputs = outputs.get('cardiac_logits', outputs.get('predictions', outputs.get('sam_masks', outputs)))
                
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # 更频繁的内存清理
                del outputs, loss
                if batch_idx % 5 == 0:  # 每5个批次清理一次
                    torch.cuda.empty_cache()
                    
                # 限制训练速度，给GPU喘息时间
                if batch_idx % 20 == 0:
                    import time
                    time.sleep(0.1)
            
            # 更新统计 - 修复NaN问题
            current_loss = loss.item() * self.accumulation_steps
            
            # 检查损失是否有效 - 增强检查
            if torch.isnan(loss) or torch.isinf(loss) or current_loss > 100:
                print(f"⚠️ 检测到异常损失值: {current_loss}, 跳过此batch")
                # 清理异常状态
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            
            total_loss += current_loss
            total_samples += images.size(0)
            
            # 更新进度条 - 修复NaN显示
            avg_loss = total_loss / (batch_idx + 1) if (batch_idx + 1) > 0 else 0
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
            
            # 添加梯度裁剪防止梯度爆炸
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'mean_dice': 0.0, 'pixel_accuracy': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 准备数据
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                
                # 前向传播
                if self.use_amp and self.scaler:
                    if NEW_AMP_API:
                        with autocast(device_type='cuda'):
                            # MedSAM优化：使用模型内置的默认prompt生成
                            batch_size = images.shape[0]
                            points = None  # MedSAM会自动生成默认prompts
                            boxes = None
                            masks = None
                            
                            outputs = self.model(images, points, boxes, masks)
                            
                            # 处理字典输出
                            if isinstance(outputs, dict):
                                outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                            
                            loss = self.criterion(outputs, labels)
                    else:
                        with autocast():
                            # MedSAM优化：使用模型内置的默认prompt生成
                            batch_size = images.shape[0]
                            points = None  # MedSAM会自动生成默认prompts
                            boxes = None
                            masks = None
                            
                            outputs = self.model(images, points, boxes, masks)
                            
                            # 处理字典输出
                            if isinstance(outputs, dict):
                                outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                            
                            loss = self.criterion(outputs, labels)
                else:
                    # MedSAM优化：使用模型内置的默认prompt生成
                    batch_size = images.shape[0]
                    points = None  # MedSAM会自动生成默认prompts
                    boxes = None
                    masks = None
                    
                    outputs = self.model(images, points, boxes, masks)
                    
                    # 处理字典输出
                    if isinstance(outputs, dict):
                        outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                    
                    loss = self.criterion(outputs, labels)
                
                # 更新统计
                total_loss += loss.item()
                
                # 计算指标 - 使用与推理一致的argmax方法
                pred_classes = torch.argmax(outputs, dim=1)  # 直接使用argmax
                metrics = TrainingUtils.calculate_metrics(pred_classes.unsqueeze(1), labels)
                for key, value in metrics.items():
                    if key in total_metrics:
                        total_metrics[key] += value
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {key: value / len(self.val_loader) for key, value in total_metrics.items()}
        
        return {'val_loss': avg_loss, **avg_metrics}
    
    def train(self):
        """开始训练"""
        # 开始训练
        
        # 记录训练历史
        train_losses = []
        val_losses = []
        val_dices = []
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录指标
            all_metrics = {**train_metrics, **val_metrics}
            for key, value in all_metrics.items():
                self.writer.add_scalar(key, value, epoch)
            
            # 记录历史
            train_losses.append(train_metrics.get('train_loss', 0))
            val_losses.append(val_metrics.get('val_loss', 0))
            val_dices.append(val_metrics.get('mean_dice', 0))
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.num_epochs}: "
                  f"Train Loss: {train_metrics.get('train_loss', 0):.4f}, "
                  f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, "
                  f"Val Dice: {val_metrics.get('mean_dice', 0):.4f}")
            
            # 早停机制检查
            current_dice = val_metrics.get('mean_dice', 0)
            if current_dice > self.best_val_dice:
                self.best_val_dice = current_dice
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(is_best=True, suffix='_best_dice')
                print(f"🎯 新的最佳Dice: {current_dice:.4f}")
            else:
                self.patience_counter += 1
                print(f"⏳ 早停计数器: {self.patience_counter}/{self.early_stopping_patience}")
            
            # 保存最佳损失模型
            if val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                self.save_checkpoint(is_best=True)
            
            # 早停检查
            if self.patience_counter >= self.early_stopping_patience:
                print(f"🛑 早停触发! 最佳Dice: {self.best_val_dice:.4f} (第{self.best_epoch+1}轮)")
                print(f"   已连续{self.patience_counter}轮无改善，停止训练")
                break
        
        print("SAM模型训练完成!")
        self.writer.close()
        
        # 返回训练结果
        return {
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'final_val_dice': val_dices[-1] if val_dices else 0,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices
        }
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = ''):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'sam_checkpoint_epoch_{self.current_epoch}{suffix}.pth'
        )
        
        TrainingUtils.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            loss=self.best_val_loss,
            save_path=checkpoint_path,
            best_val_dice=self.best_val_dice
        )

# ============================================================================
# PANet训练器
# ============================================================================

class PANetTrainer:
    """
    PANet模型训练器
    实现PANet模型的训练、验证和测试功能
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader = None,
                 test_loader: DataLoader = None,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 num_epochs: int = 100,
                 save_dir: str = 'shared/checkpoints',
                 use_amp: bool = True):
        """
        初始化PANet训练器
        
        Args:
            model: PANet模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            device: 设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            num_epochs: 训练轮数
            save_dir: 保存目录
            use_amp: 是否使用混合精度训练
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.use_amp = use_amp
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 初始化学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # 初始化混合精度训练
        if use_amp:
            if NEW_AMP_API:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 初始化损失函数 - 支持动态类别数
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        else:
            num_classes = 5
        self.criterion = FocalDiceLoss(num_classes=num_classes)
        
        # 初始化TensorBoard - 确保输出到shared目录
        if not save_dir.startswith('shared/'):
            save_dir = f'shared/{save_dir}'
        self.writer = SummaryWriter(os.path.join(save_dir, 'logs'))
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        
        print(f"PANet训练器初始化完成: device={device}, epochs={num_epochs}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            if isinstance(batch, dict):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
            else:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
            
            # 前向传播
            if self.use_amp and self.scaler:
                with autocast():
                    # MedSAM优化：使用模型内置的默认prompt生成
                    batch_size = images.shape[0]
                    points = None  # MedSAM会自动生成默认prompts
                    boxes = None
                    masks = None
                    
                    outputs = self.model(images, points, boxes, masks)
                    
                    # 处理字典输出
                    if isinstance(outputs, dict):
                        outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                    
                    loss = self.criterion(outputs, labels)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                # MedSAM优化：使用模型内置的默认prompt生成
                batch_size = images.shape[0]
                points = None  # MedSAM会自动生成默认prompts
                boxes = None
                masks = None
                
                outputs = self.model(images, points, boxes, masks)
                
                # MedSAM优化：处理新的输出格式
                if isinstance(outputs, dict):
                    # 优先使用cardiac_logits，这是MedSAM的主要输出
                    outputs = outputs.get('cardiac_logits', outputs.get('predictions', outputs.get('sam_masks', outputs)))
                
                loss = self.criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 更新统计
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_metrics = {'mean_dice': 0.0, 'pixel_accuracy': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # 准备数据
                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                else:
                    images, labels = batch
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                
                # 前向传播
                if self.use_amp and self.scaler:
                    if NEW_AMP_API:
                        with autocast(device_type='cuda'):
                            # MedSAM优化：使用模型内置的默认prompt生成
                            batch_size = images.shape[0]
                            points = None  # MedSAM会自动生成默认prompts
                            boxes = None
                            masks = None
                            
                            outputs = self.model(images, points, boxes, masks)
                            
                            # 处理字典输出
                            if isinstance(outputs, dict):
                                outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                            
                            loss = self.criterion(outputs, labels)
                    else:
                        with autocast():
                            # MedSAM优化：使用模型内置的默认prompt生成
                            batch_size = images.shape[0]
                            points = None  # MedSAM会自动生成默认prompts
                            boxes = None
                            masks = None
                            
                            outputs = self.model(images, points, boxes, masks)
                            
                            # 处理字典输出
                            if isinstance(outputs, dict):
                                outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                            
                            loss = self.criterion(outputs, labels)
                else:
                    # MedSAM优化：使用模型内置的默认prompt生成
                    batch_size = images.shape[0]
                    points = None  # MedSAM会自动生成默认prompts
                    boxes = None
                    masks = None
                    
                    outputs = self.model(images, points, boxes, masks)
                    
                    # 处理字典输出
                    if isinstance(outputs, dict):
                        outputs = outputs.get('cardiac_logits', outputs.get('sam_masks', outputs))
                    
                    loss = self.criterion(outputs, labels)
                
                # 更新统计
                total_loss += loss.item()
                
                # 计算指标 - 使用与推理一致的argmax方法
                pred_classes = torch.argmax(outputs, dim=1)  # 直接使用argmax
                metrics = TrainingUtils.calculate_metrics(pred_classes.unsqueeze(1), labels)
                for key, value in metrics.items():
                    if key in total_metrics:
                        total_metrics[key] += value
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {key: value / len(self.val_loader) for key, value in total_metrics.items()}
        
        return {'val_loss': avg_loss, **avg_metrics}
    
    def train(self):
        """开始训练"""
        print("开始PANet模型训练...")
        
        # 记录训练历史
        train_losses = []
        val_losses = []
        val_dices = []
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录指标
            all_metrics = {**train_metrics, **val_metrics}
            for key, value in all_metrics.items():
                self.writer.add_scalar(key, value, epoch)
            
            # 记录历史
            train_losses.append(train_metrics.get('train_loss', 0))
            val_losses.append(val_metrics.get('val_loss', 0))
            val_dices.append(val_metrics.get('mean_dice', 0))
            
            # 打印进度
            print(f"Epoch {epoch+1}/{self.num_epochs}: "
                  f"Train Loss: {train_metrics.get('train_loss', 0):.4f}, "
                  f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, "
                  f"Val Dice: {val_metrics.get('mean_dice', 0):.4f}")
            
            # 保存最佳模型
            if val_metrics.get('val_loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics.get('val_loss', float('inf'))
                self.save_checkpoint(is_best=True)
            
            if val_metrics.get('mean_dice', 0) > self.best_val_dice:
                self.best_val_dice = val_metrics.get('mean_dice', 0)
                self.save_checkpoint(is_best=True, suffix='_best_dice')
        
        print("PANet模型训练完成!")
        self.writer.close()
        
        # 返回训练结果
        return {
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'final_val_dice': val_dices[-1] if val_dices else 0,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_dices': val_dices
        }
    
    def save_checkpoint(self, is_best: bool = False, suffix: str = ''):
        """保存检查点"""
        checkpoint_path = os.path.join(
            self.save_dir, 
            f'panet_checkpoint_epoch_{self.current_epoch}{suffix}.pth'
        )
        
        TrainingUtils.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            loss=self.best_val_loss,
            save_path=checkpoint_path,
            best_val_dice=self.best_val_dice
        )

# ============================================================================
# 工厂函数
# ============================================================================

def create_sam_trainer(model: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader = None,
                      test_loader: DataLoader = None,
                      device: str = 'cuda',
                      learning_rate: float = 1e-4,
                      weight_decay: float = 1e-4,
                      num_epochs: int = 100,
                      save_dir: str = 'shared/checkpoints',
                      use_amp: bool = True,
                      use_checkpoint: bool = False,
                      accumulation_steps: int = 1,
                      early_stopping_patience: int = 5,
                      criterion: nn.Module = None) -> SAMTrainer:
    """
    创建SAM训练器
    
    Args:
        model: SAM模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        num_epochs: 训练轮数
        save_dir: 保存目录
        use_amp: 是否使用混合精度训练
        use_checkpoint: 是否使用梯度检查点
        accumulation_steps: 梯度累积步数
        early_stopping_patience: 早停耐心值（连续多少轮无改善后停止）
    
    Returns:
        SAMTrainer对象
    """
    return SAMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        save_dir=save_dir,
        use_amp=use_amp,
        use_checkpoint=use_checkpoint,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        criterion=criterion
    )

def create_panet_trainer(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader = None,
                        test_loader: DataLoader = None,
                        device: str = 'cuda',
                        learning_rate: float = 1e-4,
                        weight_decay: float = 1e-4,
                        num_epochs: int = 100,
                        save_dir: str = 'shared/checkpoints',
                        use_amp: bool = True) -> PANetTrainer:
    """
    创建PANet训练器
    
    Args:
        model: PANet模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        device: 设备
        learning_rate: 学习率
        weight_decay: 权重衰减
        num_epochs: 训练轮数
        save_dir: 保存目录
        use_amp: 是否使用混合精度训练
    
    Returns:
        PANetTrainer对象
    """
    return PANetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        save_dir=save_dir,
        use_amp=use_amp
    )
