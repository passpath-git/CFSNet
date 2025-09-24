#!/usr/bin/env python3
"""
心脏四腔分割训练主脚本
集成SAM、PANet和心脏特定特征的端到端训练
支持两阶段训练策略：冻结SAM主干 + 分层学习率微调

🚀 快速开始示例：
# 基础训练（推荐，自动两阶段训练）
python scripts/train_cardiac.py --data_root database/database_nifti --model_type cardiac_sam

# 完整参数训练
python scripts/train_cardiac.py --data_root database/database_nifti --model_type cardiac_sam --epochs 50 --batch_size 1 --image_size 256 256 --accumulate_steps 8 --use_checkpoint --num_workers 0 --learning_rate 1e-5 --weight_decay 1e-4

# 自定义两阶段训练
python scripts/train_cardiac.py --data_root database/database_nifti --model_type cardiac_sam --epochs 100 --stage1_epochs 10 --stage1_learning_rate 2e-4

📋 两阶段训练策略：
阶段一：冻结SAM主干，快速训练PANet和分割头（5-10 epochs）
阶段二：解冻SAM主干，分层学习率微调（剩余epochs）

🎯 推荐配置：
- epochs: 50-100
- batch_size: 1-2
- accumulate_steps: 8
- image_size: 256x256 或 512x512
- learning_rate: 1e-5（自动调整）
- weight_decay: 1e-4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import json
import logging
from datetime import datetime
import warnings

# 抑制所有nibabel警告
warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")
warnings.filterwarnings("ignore", message="pixdim.*qfac.*should be 1.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# 直接屏蔽nibabel的INFO日志
import logging
logging.getLogger('nibabel').setLevel(logging.WARNING)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.cardiac_config import CardiacConfig
from configs.model_config import ModelConfig
from models.cardiac_sam import CardiacSAM
from models.cardiac_panet import CardiacPANet
from base.data_processing import create_cardiac_dataloader
from base.training_core import create_sam_trainer, create_panet_trainer
from utils.visualization import create_visualization_utils
from utils.metrics import create_cardiac_metrics
from utils.losses import CardiacLoss, ExtremeImbalanceLoss, GradientMonitor, online_hard_example_mining


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.debug(f"日志文件: {log_file}")
    
    return logger


def setup_device(device: str) -> torch.device:
    """设置训练设备"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'
    
        # 设备信息已静默设置
    
    return torch.device(device)


def create_model(model_type: str, config: Dict[str, Any], device: torch.device) -> nn.Module:
    """创建模型"""
    # 静默创建模型
    if model_type == 'cardiac_sam':
        model = CardiacSAM(config=config)
    elif model_type == 'cardiac_panet':
        model = CardiacPANet(
            num_classes=config['num_classes'],
            feature_dim=config['feature_dim'],
            use_sam_features=config.get('use_sam_features', True)
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model.to(device)


def setup_two_stage_training(model: nn.Module, config: Dict[str, Any], device: torch.device):
    """
    设置两阶段训练策略
    
    阶段一：冻结SAM主干，快速训练PANet和分割头
    阶段二：解冻SAM，使用分层学习率进行微调
    """
    # 静默设置两阶段训练
    if hasattr(model, 'sam_model') and hasattr(model.sam_model, 'image_encoder'):
        for param in model.sam_model.image_encoder.parameters():
            param.requires_grad = False
        config['stage1_learning_rate'] = config.get('stage1_learning_rate', 1e-4)
        config['stage1_epochs'] = config.get('stage1_epochs', 5)
        return True
    else:
        return False


def create_trainer(model: nn.Module, 
                  model_type: str,
                  train_loader, 
                  val_loader, 
                  config: Dict[str, Any], 
                  device: torch.device,
                  checkpoint_dir: str,
                  criterion: nn.Module = None):
    """创建训练器"""
    if model_type == 'cardiac_sam':
        trainer = create_sam_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=checkpoint_dir,
            num_epochs=config.get('epochs', 100),
            use_amp=config.get('use_amp', True),
            early_stopping_patience=config.get('early_stopping_patience', 15),  # 放宽早停期限到15
            criterion=criterion  # 传递外部创建的损失函数
        )
    elif model_type == 'cardiac_panet':
        trainer = create_panet_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            num_epochs=config.get('epochs', 20),
            save_dir=checkpoint_dir
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return trainer


def execute_classification_training(model, train_loader, val_loader, config, device, logger, output_dir):
    """执行三阶段分类训练策略"""
    
    # 三阶段分类训练配置 - 优化渐进式提升策略
    # 动态计算三个阶段的比例，确保每个阶段有足够的训练时间
    total_epochs = config.get('epochs', 50)
    stage1_ratio = 15 / 50  # 阶段1比例：20/65 ≈ 0.31 (更多时间建立基础)
    stage2_ratio = 15 / 50  # 阶段2比例：20/65 ≈ 0.31 (充分学习左右心区分)
    stage3_ratio = 20 / 50  # 阶段3比例：25/65 ≈ 0.38 (精细分割需要更多时间)
    
    stage1_epochs = max(3, int(total_epochs * stage1_ratio))  # 至少3个epoch
    stage2_epochs = max(3, int(total_epochs * stage2_ratio))  # 至少3个epoch
    stage3_epochs = max(5, total_epochs - stage1_epochs - stage2_epochs)  # 至少5个epoch
    
    classification_stages = [
        {
            'name': '阶段1: 背景vs心脏整体',
            'num_classes': 2,
            'epochs': stage1_epochs,
            'freeze_sam': True,
            'learning_rate': 5e-4,  # 进一步提高学习率，快速建立基础
            'class_weights': [0.01, 10.0],  # 大幅提升心脏权重，基于78:1的不平衡比例
            'description': '让模型先学会"哪里是心脏"，建立基础分割能力',
            'expected_dice_range': (0.1, 0.4),  # 期望Dice范围
            'warmup_epochs': 2  # 预热轮数
        },
        {
            'name': '阶段2: 四腔粗分（左心vs右心）',
            'num_classes': 3,
            'epochs': stage2_epochs,
            'freeze_sam': 'partial',  # 解冻最后2个Transformer Block
            'learning_rate': 3e-4,  # 提高学习率，加强左右心区分
            'class_weights': [0.01, 8.0, 8.0],  # 大幅提升左右心权重，基于160:1:1的不平衡比例
            'description': '区分左右心系统，建立对称性感知，利用心脏左右对称性',
            'expected_dice_range': (0.3, 0.6),  # 期望Dice范围
            'warmup_epochs': 1  # 预热轮数
        },
        {
            'name': '阶段3: 完整五类精细分割',
            'num_classes': 5,
            'epochs': stage3_epochs,
            'freeze_sam': False,  # 解冻整个SAM
            'learning_rate': 1e-4,  # 提高学习率，加强精细分割
            'class_weights': [0.01, 15.0, 20.0, 12.0, 10.0],  # 基于实际数据分布调整权重
            'description': '精细分割LV、RV、LA、RA、背景，联合训练SAM+PANet',
            'expected_dice_range': (0.5, 0.8),  # 期望Dice范围
            'warmup_epochs': 1  # 预热轮数
        }
    ]
    
    all_results = {}
    
    for stage_idx, stage in enumerate(classification_stages):
        # 静默设置阶段信息，只保留关键输出
        print(f"阶段{stage_idx + 1}: {stage['name']}")
        print(f"期望Dice范围: {stage['expected_dice_range'][0]:.1f} - {stage['expected_dice_range'][1]:.1f}")
        print(f"{stage['num_classes']}类分割, {stage['epochs']}轮训练")
        
        # 调整模型输出类别数 - 只在类别数真正改变时才重建头部
        in_features = model.cardiac_head[-1].in_channels
        current_classes = model.cardiac_head[-1].out_channels
        
        if current_classes != stage['num_classes']:
            # 保存当前头部权重（如果存在）
            old_head_weight = None
            old_head_bias = None
            if hasattr(model.cardiac_head[-1], 'weight') and model.cardiac_head[-1].weight is not None:
                old_head_weight = model.cardiac_head[-1].weight.data.clone()
                old_head_bias = model.cardiac_head[-1].bias.data.clone()
            
            # 创建新的头部
            model.cardiac_head[-1] = nn.Conv2d(in_features, stage['num_classes'], 1).to(device)
            
            # 尝试继承权重 - 改进权重继承策略
            if old_head_weight is not None:
                new_weight = model.cardiac_head[-1].weight.data
                new_bias = model.cardiac_head[-1].bias.data
                
                # 如果新类别数 >= 旧类别数，可以部分继承权重
                if new_weight.shape[0] >= old_head_weight.shape[0]:
                    # 继承现有权重
                    new_weight[:old_head_weight.shape[0]] = old_head_weight
                    new_bias[:old_head_bias.shape[0]] = old_head_bias
                    
                    # 对于新增的类别，使用现有权重的平均值进行初始化
                    if new_weight.shape[0] > old_head_weight.shape[0]:
                        avg_weight = old_head_weight.mean(dim=0, keepdim=True)
                        avg_bias = old_head_bias.mean()
                        for i in range(old_head_weight.shape[0], new_weight.shape[0]):
                            new_weight[i] = avg_weight + torch.randn_like(avg_weight) * 0.1
                            new_bias[i] = avg_bias + torch.randn_like(avg_bias) * 0.1
                    
                    # 静默继承权重
                else:
                    # 如果新类别数 < 旧类别数，选择最重要的类别权重
                    if stage_idx == 1:  # 阶段2：选择背景和心脏权重
                        new_weight[0] = old_head_weight[0]  # 背景
                        new_weight[1] = old_head_weight[1:].mean(dim=0)  # 心脏整体
                        new_bias[0] = old_head_bias[0]
                        new_bias[1] = old_head_bias[1:].mean()
                    else:  # 其他情况，选择前几个类别
                        new_weight[:] = old_head_weight[:new_weight.shape[0]]
                        new_bias[:] = old_head_bias[:new_bias.shape[0]]
                    # 静默选择性继承权重
            
            # **关键修复：更新模型的num_classes属性**
            model.num_classes = stage['num_classes']
        else:
            # 静默保持当前头部
            pass
        
        # 重新创建损失函数以匹配新的类别数
        class_weights = torch.tensor(stage['class_weights']).to(device)
        criterion = ExtremeImbalanceLoss(
            num_classes=stage['num_classes'],
            alpha=0.25,
            gamma=2.0,
            dice_weight=1.0,
            focal_weight=1.0,
            use_class_weights=True,
            use_focal=True,
            use_dice=True
        ).to(device)
        # 静默创建损失函数
        
        # 应用SAM冻结策略 - 静默执行
        if stage['freeze_sam'] == True:
            # 冻结所有SAM参数
            for param in model.sam_model.image_encoder.parameters():
                param.requires_grad = False
        elif stage['freeze_sam'] == 'partial':
            # 只解冻最后2个Transformer Block
            if hasattr(model.sam_model.image_encoder, 'blocks'):
                total_blocks = len(model.sam_model.image_encoder.blocks)
                for i, block in enumerate(model.sam_model.image_encoder.blocks):
                    for param in block.parameters():
                        param.requires_grad = i >= (total_blocks - 2)  # 解冻最后2层
        else:
            # 解冻整个SAM主干（前70%层）
            if hasattr(model.sam_model.image_encoder, 'blocks'):
                total_blocks = len(model.sam_model.image_encoder.blocks)
                freeze_blocks = int(total_blocks * 0.3)  # 只冻结前30%
                for i, block in enumerate(model.sam_model.image_encoder.blocks):
                    for param in block.parameters():
                        param.requires_grad = i >= freeze_blocks
        
        # 创建优化器 - 添加学习率预热
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=stage['learning_rate'], weight_decay=1e-4)
        
        # 学习率预热调度器
        warmup_epochs = stage['warmup_epochs']
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs  # 线性预热
                else:
                    return 1.0  # 正常学习率
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage['epochs'])
        
        # 损失函数已在上面重新创建
        
        # 标签转换函数 - 修复标签映射问题
        def convert_labels_for_stage(labels, num_classes):
            if num_classes == 2:
                # 背景(0) vs 心脏整体(1)：合并所有心脏结构
                # 确保标签值在[0, 1]范围内
                converted = (labels > 0).long()
                return torch.clamp(converted, 0, 1)
            elif num_classes == 3:
                # 背景(0) + 左心(1) + 右心(2)
                new_labels = labels.clone()
                # 左心室(1) + 左心房(3) -> 左心(1)
                new_labels[labels == 1] = 1  # 左心室 -> 左心
                new_labels[labels == 3] = 1  # 左心房 -> 左心
                # 右心室(2) + 右心房(4) -> 右心(2)
                new_labels[labels == 2] = 2  # 右心室 -> 右心
                new_labels[labels == 4] = 2  # 右心房 -> 右心
                # 确保标签值在[0, 2]范围内
                return torch.clamp(new_labels, 0, 2)
            else:
                # 5类：保持原样，确保标签值在[0, 4]范围内
                return torch.clamp(labels, 0, 4)
        
        # 使用原有的训练器框架，但修改损失函数和标签转换
        from base.training_core import SAMTrainer
        
        # 创建包装的数据加载器
        class ClassificationDataLoader:
            def __init__(self, original_loader, num_classes):
                self.original_loader = original_loader
                self.num_classes = num_classes
                self.dataset = original_loader.dataset
            
            def __iter__(self):
                for batch in self.original_loader:
                    if isinstance(batch, dict):
                        images = batch['image']
                        labels = convert_labels_for_stage(batch['label'], self.num_classes)
                        yield {'image': images, 'label': labels}
                    else:
                        images, labels = batch
                        labels = convert_labels_for_stage(labels, self.num_classes)
                        yield images, labels
            
            def __len__(self):
                return len(self.original_loader)
        
        # 包装数据加载器
        wrapped_train_loader = ClassificationDataLoader(train_loader, stage['num_classes'])
        wrapped_val_loader = ClassificationDataLoader(val_loader, stage['num_classes'])
        
        # 创建阶段专用训练器，直接传递损失函数
        stage_trainer = SAMTrainer(
            model=model,
            train_loader=wrapped_train_loader,
            val_loader=wrapped_val_loader,
            device=device,
            learning_rate=stage['learning_rate'],
            num_epochs=stage['epochs'],
            save_dir=output_dir,
            use_amp=True,
            accumulation_steps=4,
            early_stopping_patience=15,
            criterion=criterion  # 直接传递损失函数
        )
        
        # **关键修复：更新训练器的类别数信息**
        stage_trainer.num_classes = stage['num_classes']
        
        # 替换优化器和调度器
        stage_trainer.optimizer = optimizer
        stage_trainer.scheduler = scheduler
        
        # 静默开始训练
        best_dice_so_far = 0.0
        dice_history = []
        
        # 自定义训练循环，保留进度条和关键输出
        for epoch in range(stage['epochs']):
            # 训练一个epoch
            train_metrics = stage_trainer.train_epoch(epoch)
            
            # 验证一个epoch
            val_metrics = stage_trainer.validate_epoch(epoch)
            
            # 更新学习率
            stage_trainer.scheduler.step()
            
            # 记录Dice历史
            current_dice = val_metrics.get('mean_dice', 0)
            dice_history.append(current_dice)
            
            # 更新最佳Dice
            if current_dice > best_dice_so_far:
                best_dice_so_far = current_dice
            
            # 打印训练进度和关键数据
            print(f"Epoch {epoch+1}/{stage['epochs']}: "
                  f"Train Loss: {train_metrics.get('train_loss', 0):.4f}, "
                  f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, "
                  f"Val Dice: {current_dice:.4f}")
        
        # 保存阶段结果
        stage_results = {
            'final_train_loss': train_metrics.get('train_loss', 0),
            'final_val_loss': val_metrics.get('val_loss', 0),
            'final_val_dice': current_dice,
            'best_val_dice': best_dice_so_far,
            'dice_history': dice_history,
            'expected_range': stage['expected_dice_range'],
            'achieved_target': best_dice_so_far >= stage['expected_dice_range'][0]
        }
        
        # 保存阶段检查点
        stage_checkpoint_path = os.path.join(output_dir, f'classification_stage_{stage_idx + 1}_best.pth')
        torch.save({
            'stage': stage_idx + 1,
            'stage_name': stage['name'],
            'num_classes': stage['num_classes'],
            'model_state_dict': model.state_dict(),
            'results': stage_results
        }, stage_checkpoint_path)
        
        all_results[f'classification_stage_{stage_idx + 1}'] = stage_results
        
        # 阶段完成总结 - 只保留关键输出
        print(f"阶段{stage_idx + 1}完成! 最佳Dice: {best_dice_so_far:.4f}")
        
        # 阶段间过渡提示 - 只保留关键输出
        if stage_idx < len(classification_stages) - 1:
            next_stage = classification_stages[stage_idx + 1]
            print(f"准备进入阶段{stage_idx + 2}: {next_stage['name']}")
            print(f"期望Dice范围: {next_stage['expected_dice_range'][0]:.1f} - {next_stage['expected_dice_range'][1]:.1f}")
    
    logger.debug("三阶段分类训练完成！")
    return all_results


def execute_two_stage_training(trainer, config: Dict[str, Any], logger):
    """
    执行两阶段训练
    
    阶段一：冻结SAM主干，快速训练
    阶段二：解冻SAM，分层学习率微调
    """
    print("\n开始两阶段训练...")
    
    # 阶段一：冻结训练
    stage1_epochs = config.get('stage1_epochs', 5)
    print(f"\n阶段一：冻结SAM主干训练 ({stage1_epochs} epochs)")
    print("目标：让PANet和分割头快速适应任务")
    
    # 设置阶段一的学习率
    original_lr = trainer.optimizer.param_groups[0]['lr']
    stage1_lr = config.get('stage1_learning_rate', 1e-4)
    
    # 更新学习率
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = stage1_lr
    
    print(f"  - 阶段一学习率: {stage1_lr}")
    print(f"  - 原始学习率: {original_lr}")
    
    # 执行阶段一训练
    # 临时修改训练器的epochs属性
    original_epochs = trainer.num_epochs
    trainer.num_epochs = stage1_epochs
    stage1_results = trainer.train()
    # 恢复原始num_epochs
    trainer.num_epochs = original_epochs
    
    print(f"阶段一训练完成！")
    
    # 安全地获取和打印结果
    final_train_loss = stage1_results.get('final_train_loss', 'N/A')
    final_val_loss = stage1_results.get('final_val_loss', 'N/A')
    final_val_dice = stage1_results.get('final_val_dice', 'N/A')
    
    if isinstance(final_train_loss, (int, float)):
        print(f"  - 最终训练损失: {final_train_loss:.4f}")
    else:
        print(f"  - 最终训练损失: {final_train_loss}")
    
    if isinstance(final_val_loss, (int, float)):
        print(f"  - 最终验证损失: {final_val_loss:.4f}")
    else:
        print(f"  - 最终验证损失: {final_val_loss}")
    
    if isinstance(final_val_dice, (int, float)):
        print(f"  - 最终验证Dice: {final_val_dice:.4f}")
    else:
        print(f"  - 最终验证Dice: {final_val_dice}")
    
    # 阶段二：解冻微调
    print(f"\n阶段二：解冻SAM主干，分层学习率微调")
    print("目标：让整个模型协同优化，追求更高精度")
    
    # 解冻SAM的image_encoder
    if hasattr(trainer.model, 'sam_model') and hasattr(trainer.model.sam_model, 'image_encoder'):
        for param in trainer.model.sam_model.image_encoder.parameters():
            param.requires_grad = True
        print("SAM Image Encoder已解冻")
    
    # 设置分层学习率
    stage2_lr_sam = config.get('stage2_lr_sam', 1e-6)      # SAM主干：极低学习率
    stage2_lr_panet = config.get('stage2_lr_panet', 1e-5)  # PANet：中等学习率
    stage2_lr_head = config.get('stage2_lr_head', 1e-5)    # 分割头：中等学习率
    
    # 创建分层学习率优化器
    param_groups = []
    
    # SAM主干参数组
    if hasattr(trainer.model, 'sam_model') and hasattr(trainer.model.sam_model, 'image_encoder'):
        sam_params = list(trainer.model.sam_model.image_encoder.parameters())
        param_groups.append({'params': sam_params, 'lr': stage2_lr_sam})
        print(f"  - SAM主干学习率: {stage2_lr_sam}")
    
    # PANet参数组
    if hasattr(trainer.model, 'panet_fusion'):
        panet_params = list(trainer.model.panet_fusion.parameters())
        param_groups.append({'params': panet_params, 'lr': stage2_lr_panet})
        print(f"  - PANet学习率: {stage2_lr_panet}")
    
    # 分割头参数组
    if hasattr(trainer.model, 'cardiac_head'):
        head_params = list(trainer.model.cardiac_head.parameters())
        param_groups.append({'params': head_params, 'lr': stage2_lr_head})
        print(f"  - 分割头学习率: {stage2_lr_head}")
    
    # 其他参数使用默认学习率
    other_params = []
    for name, param in trainer.model.named_parameters():
        if not any(name.startswith(prefix) for prefix in ['sam_model.image_encoder', 'panet_fusion', 'cardiac_head']):
            other_params.append(param)
    
    if other_params:
        param_groups.append({'params': other_params, 'lr': original_lr})
        print(f"  - 其他参数学习率: {original_lr}")
    
    # 创建新的优化器
    trainer.optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=config.get('weight_decay', 3e-5)
    )
    
    # 重新创建学习率调度器
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=trainer.num_epochs,
        eta_min=1e-7
    )
    
    print(" 分层学习率优化器已设置")
    
    # 计算剩余训练轮数
    total_epochs = config.get('epochs', 100)
    remaining_epochs = total_epochs - stage1_epochs
    
    print(f"  - 剩余训练轮数: {remaining_epochs}")
    
    # 执行阶段二训练
    # 临时修改训练器的num_epochs属性
    original_epochs = trainer.num_epochs
    trainer.num_epochs = remaining_epochs
    stage2_results = trainer.train()
    # 恢复原始num_epochs
    trainer.num_epochs = original_epochs
    
    print(f"✅ 阶段二训练完成！")
    
    # 安全地获取和打印结果
    final_train_loss = stage2_results.get('final_train_loss', 'N/A')
    final_val_loss = stage2_results.get('final_val_loss', 'N/A')
    final_val_dice = stage2_results.get('final_val_dice', 'N/A')
    
    if isinstance(final_train_loss, (int, float)):
        print(f"  - 最终训练损失: {final_train_loss:.4f}")
    else:
        print(f"  - 最终训练损失: {final_train_loss}")
    
    if isinstance(final_val_loss, (int, float)):
        print(f"  - 最终验证损失: {final_val_loss:.4f}")
    else:
        print(f"  - 最终验证损失: {final_val_loss}")
    
    if isinstance(final_val_dice, (int, float)):
        print(f"  - 最终验证Dice: {final_val_dice:.4f}")
    else:
        print(f"  - 最终验证Dice: {final_val_dice}")
    
    # 合并结果
    combined_results = {
        'stage1': stage1_results,
        'stage2': stage2_results,
        'total_epochs': total_epochs,
        'stage1_epochs': stage1_epochs,
        'stage2_epochs': remaining_epochs
    }
    
    return combined_results


def verify_fixes(model: nn.Module, device: torch.device, logger) -> Dict[str, bool]:
    """静默验证关键问题是否已解决"""
    verification_results = {}
    
    try:
        # 简单验证特征维度
        test_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            model.eval()
            outputs = model(test_input)
            verification_results['feature_dimension_fix'] = 'cardiac_logits' in outputs and outputs['cardiac_logits'].shape == (1, 5, 512, 512)
        
        # 简单验证损失函数
        class_weights = torch.tensor([0.20, 2.50, 2.80, 2.20, 2.00]).to(device)
        cardiac_loss = CardiacLoss(weights=class_weights).to(device)
        test_pred = torch.randn(1, 5, 64, 64).to(device)
        test_target = torch.randint(0, 5, (1, 64, 64)).to(device)
        loss = cardiac_loss(test_pred, test_target)
        verification_results['class_imbalance_fix'] = torch.isfinite(loss) and loss.item() > 0
        
        # 简单验证梯度
        model.train()
        outputs = model(test_input)
        loss = cardiac_loss(outputs['cardiac_logits'], torch.randint(0, 5, (1, 512, 512)).to(device))
        loss.backward()
        grad_monitor = GradientMonitor(model)
        grad_norm = grad_monitor.check_gradients()
        verification_results['gradient_monitoring_fix'] = grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm))
        model.zero_grad()
        
        verification_results['all_fixes_verified'] = all(verification_results.values())
        
    except Exception as e:
        logger.error(f"验证失败: {e}")
        verification_results = {'all_fixes_verified': False}
    
    return verification_results


def main():
    """主函数"""

    
    parser = argparse.ArgumentParser(description='心脏四腔分割训练 - 支持两阶段训练策略')
    
    # 基本参数
    parser.add_argument('--model_type', type=str, default='cardiac_sam',
                       choices=['cardiac_sam', 'cardiac_panet'],
                       help='模型类型')
    parser.add_argument('--use_classification_training', type=bool, default=True,
                       help='使用三阶段分类训练策略（背景vs心脏→左心vs右心→完整五类）（默认启用）')
    parser.add_argument('--data_root', type=str, required=True,
                       help='数据根目录')
    parser.add_argument('--output_dir', type=str, default='shared/outputs',
                       help='输出目录（默认：shared/outputs）')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='训练设备')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='总训练轮数（推荐：50-100）')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='批次大小（推荐：2-4，平衡训练效率和显存）')
    parser.add_argument('--accumulate_steps', type=int, default=4,
                       help='梯度累积步数（推荐：4，有效批次大小 = batch_size × accumulate_steps）')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='基础学习率（推荐：1e-5，两阶段训练会自动调整）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减（推荐：1e-4，适中的正则化）')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='使用混合精度训练（节省内存，但可能影响稳定性）')
    parser.add_argument('--use_checkpoint', action='store_true', default=True,
                       help='使用梯度检查点（推荐启用，节省内存）')
    
    # 两阶段训练参数（自动优化，通常无需手动调整）
    parser.add_argument('--use_two_stage', action='store_true', default=True,
                       help='使用两阶段训练策略（强烈推荐，自动冻结SAM主干+分层学习率）')
    parser.add_argument('--stage1_epochs', type=int, default=5,
                       help='阶段一训练轮数（冻结SAM主干，推荐：5-10）')
    parser.add_argument('--stage1_learning_rate', type=float, default=1e-4,
                       help='阶段一学习率（冻结阶段，推荐：1e-4，快速学习新参数）')
    parser.add_argument('--stage2_lr_sam', type=float, default=1e-6,
                       help='阶段二SAM主干学习率（解冻阶段，推荐：1e-6，极低学习率精修）')
    parser.add_argument('--stage2_lr_panet', type=float, default=1e-5,
                       help='阶段二PANet学习率（解冻阶段，推荐：1e-5，中等学习率）')
    parser.add_argument('--stage2_lr_head', type=float, default=1e-5,
                       help='阶段二分割头学习率（解冻阶段，推荐：1e-5，中等学习率）')
    
    # 数据参数
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512],
                       help='图像尺寸 (H W)（推荐：512x512，平衡性能和显存）')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载器工作进程数（默认：0，避免多进程问题）')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='使用数据增强（已默认启用，此参数保留用于向后兼容）')
    parser.add_argument('--split_file_dir', type=str, default='database/database_split',
                       help='数据集分割文件目录')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='特征维度')
    parser.add_argument('--use_hq', action='store_true',
                       help='使用HQSAM')
    parser.add_argument('--use_sam_features', action='store_true',
                       help='使用SAM特征（PANet模型）')
    parser.add_argument('--use_prompts', action='store_true',
                       help='使用真实Prompt生成（基于分割掩码生成点和边界框）')
    
    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建带时间戳的输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'shared/outputs/training_{timestamp}'
    # 创建专门的checkpoint文件夹，按时间戳命名
    checkpoint_dir = f'shared/checkpoints/training_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 静默设置日志
    logger = setup_logging(output_dir)
    
    # 设置设备
    device = setup_device(args.device)
    
    # 设置PyTorch环境变量以抑制警告
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    

    
    # 抑制PyTorch内部警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
    
    # 静默加载配置
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = CardiacConfig().get_model_config(args.model_type)
    
    # 更新配置
    config.update({
        'model_type': args.model_type,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'accumulate_steps': args.accumulate_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'image_size': tuple(args.image_size),
        'feature_dim': args.feature_dim,
        'use_hq': args.use_hq,
        'use_sam_features': args.use_sam_features,
        'use_prompts': args.use_prompts,
        'use_amp': args.use_amp,
        'use_checkpoint': args.use_checkpoint,
        'output_dir': output_dir,
        'checkpoint_dir': checkpoint_dir,
        'log_dir': os.path.join(output_dir, 'logs')
    })
    
    # 静默配置
    if args.use_two_stage:
        config.update({
            'use_two_stage': True,
            'stage1_epochs': args.stage1_epochs,
            'stage1_learning_rate': args.stage1_learning_rate,
            'stage2_lr_sam': args.stage2_lr_sam,
            'stage2_lr_panet': args.stage2_lr_panet,
            'stage2_lr_head': args.stage2_lr_head
        })
    
    # 静默保存配置
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    try:
        # 静默创建数据和模型
        dataloader = create_cardiac_dataloader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=tuple(args.image_size),
            use_augmentation=True,  # 默认启用数据增强，提高模型泛化能力
            use_prompts=args.use_prompts,
            split_file_dir=args.split_file_dir
        )
        
        train_loader = dataloader.create_dataloader('train')
        val_loader = dataloader.create_dataloader('validation')
        model = create_model(args.model_type, config, device)
        
        # 静默准备训练
        verification_results = verify_fixes(model, device, logger)
        if args.use_two_stage:
            setup_two_stage_training(model, config, device)
        
        trainer = create_trainer(
            model=model,
            model_type=args.model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        if args.resume and os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)
        
        if args.use_classification_training:
            # 执行三阶段分类训练
            training_results = execute_classification_training(model, train_loader, val_loader, config, device, logger, output_dir)
        elif args.use_two_stage:
            # 执行两阶段训练
            training_results = execute_two_stage_training(trainer, config, logger)
        else:
            # 执行标准训练
            training_results = trainer.train()
        
        # 保存训练结果
        results_file = os.path.join(output_dir, 'training_results.json')
        # 转换训练结果为JSON可序列化格式
        serializable_results = {}
        for key, value in training_results.items():
            if isinstance(value, list):
                serializable_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        # 处理字典类型
                        serializable_item = {}
                        for k, v in item.items():
                            if isinstance(v, (int, float, str, bool)) or v is None:
                                serializable_item[k] = v
                            elif isinstance(v, torch.Tensor):
                                serializable_item[k] = v.item() if v.numel() == 1 else v.tolist()
                            else:
                                serializable_item[k] = str(v)
                        serializable_results[key].append(serializable_item)
                    else:
                        serializable_results[key].append(str(item))
            elif isinstance(value, dict):
                # 处理字典类型
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        serializable_results[key][k] = v
                    elif isinstance(v, torch.Tensor):
                        serializable_results[key][k] = v.item() if v.numel() == 1 else v.tolist()
                    else:
                        serializable_results[key][k] = str(v)
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.debug(f"训练结果已保存: {results_file}")
        
        logger.debug("训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
