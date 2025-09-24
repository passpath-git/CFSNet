#!/usr/bin/env python3
"""
心脏四腔分割PANet模型
基于PANet架构，专门用于心脏超声图像的四腔分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.panet.panet_model import PANetModule


class CardiacPANet(nn.Module):
    """
    心脏四腔分割PANet模型
    
    基于PANet的特征融合架构，
    专门用于心脏超声图像的四腔分割任务
    """
    
    def __init__(self, 
                 num_classes: int = 5,
                 feature_dim: int = 256,
                 use_sam_features: bool = True,
                 config: Dict[str, Any] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.use_sam_features = use_sam_features
        self.config = config or {}
        
        # 特征提取器（模拟SAM特征）
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3),  # 输入单通道图像
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # PANet特征融合模块
        self.panet_fusion = PANetModule()
        
        # 心脏分割头
        self.cardiac_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, 1)
        )
        
        # 如果使用SAM特征，添加特征投影层
        if self.use_sam_features:
            self.sam_feature_proj = nn.Conv2d(256, self.feature_dim, 1)
        else:
            self.sam_feature_proj = None
    
    def forward(self, images: torch.Tensor,
                sam_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 输入图像 [B, C, H, W]
            sam_features: SAM特征 [B, 256, H, W] (可选)
        
        Returns:
            包含预测结果的字典
        """
        # 特征提取
        if sam_features is not None and self.use_sam_features:
            # 使用SAM特征
            if self.sam_feature_proj is not None:
                features = self.sam_feature_proj(sam_features)
            else:
                features = sam_features
        else:
            # 使用自己的特征提取器
            features = self.feature_extractor(images)
        
        # 确保特征图尺寸与输入图像匹配
        if features.shape[-2:] != images.shape[-2:]:
            features = F.interpolate(
                features,
                size=images.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # PANet特征融合
        fused_features = self.panet_fusion([features])
        
        # 心脏分割预测
        cardiac_logits = self.cardiac_head(fused_features)
        
        # 构建输出
        outputs = {
            'cardiac_logits': cardiac_logits,
            'features': features,
            'fused_features': fused_features
        }
        
        # 为了兼容训练脚本，添加标准预测输出
        outputs['predictions'] = cardiac_logits
        
        return outputs
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'use_sam_features': self.use_sam_features
        }
