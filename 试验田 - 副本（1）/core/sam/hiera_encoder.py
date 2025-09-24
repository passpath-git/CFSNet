# 简化版Hiera Backbone编码器（基于ViT，内存优化）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math
from configs.model_config import ModelConfig

class SimpleHieraViTEncoder(nn.Module):
    """简化版Hiera Backbone编码器 - 内存优化版本"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = ModelConfig.SAM_IMAGE_ENCODER
            
        self.embed_dim = config.get('embed_dim', 256)
        self.image_size = config.get('image_size', 512)
        self.patch_size = config.get('patch_size', 16)
        self.num_heads = config.get('num_heads', 8)
        self.out_chans = config.get('out_chans', 256)
        
        # 简化的多尺度补丁嵌入
        self.patch_embed_8x8 = nn.Conv2d(3, 128, kernel_size=8, stride=8)  # 中分辨率
        self.patch_embed_16x16 = nn.Conv2d(3, 256, kernel_size=16, stride=16)  # 低分辨率
        
        # 简化的多尺度编码器
        self.encoder_8x8 = self._build_simple_encoder(128, 2, 4)
        self.encoder_16x16 = self._build_simple_encoder(256, 4, 8)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(128 + 256, self.embed_dim, 1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim, self.out_chans, 1)
        )
        
        # 下采样到期望尺寸 (32x32 for 512x512 input)
        self.downsample = nn.AdaptiveAvgPool2d((32, 32))
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _build_simple_encoder(self, dim: int, depth: int, num_heads: int):
        """构建简化的编码器"""
        encoder = nn.ModuleList()
        for _ in range(depth):
            encoder.append(nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,  # 减少MLP维度
                dropout=0.1,
                batch_first=True
            ))
        return encoder
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 简化的多尺度特征提取
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            features: 融合后的多尺度特征
        """
        B, C, H, W = x.shape
        
        # 多尺度特征提取
        # 8x8补丁 - 中分辨率特征
        feat_8x8 = self.patch_embed_8x8(x)  # (B, 128, H/8, W/8)
        feat_8x8 = self._apply_simple_transformer(feat_8x8, self.encoder_8x8)
        
        # 16x16补丁 - 低分辨率语义
        feat_16x16 = self.patch_embed_16x16(x)  # (B, 256, H/16, W/16)
        feat_16x16 = self._apply_simple_transformer(feat_16x16, self.encoder_16x16)
        
        # 上采样到统一尺寸
        target_size = feat_8x8.shape[-2:]
        feat_16x16 = F.interpolate(feat_16x16, size=target_size, mode='bilinear', align_corners=False)
        
        # 多尺度特征融合
        fused_features = torch.cat([feat_8x8, feat_16x16], dim=1)
        fused_features = self.fusion(fused_features)
        
        # 输出层
        output = self.neck(fused_features)
        
        # 下采样到期望尺寸
        output = self.downsample(output)
        
        return output
    
    def _apply_simple_transformer(self, x: torch.Tensor, encoder: nn.ModuleList) -> torch.Tensor:
        """应用简化的Transformer编码器"""
        B, C, H, W = x.shape
        
        # 转换为序列格式
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 应用Transformer编码器
        for layer in encoder:
            x = layer(x)
        
        # 转换回空间格式
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x
