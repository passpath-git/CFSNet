# PANet主模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .attention import MultiHeadAttention, SpatialAttention, ChannelAttention
from .fusion import FeatureFusion
from configs.model_config import ModelConfig

class PANetModule(nn.Module):
    """PANet主模块，实现真正的多尺度特征融合"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = ModelConfig.PANET_FUSION
            
        self.feature_dim = config['output_dim']
        self.use_attention = config.get('use_attention', True)
        
        # 侧向连接层（1x1卷积调整通道数）
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(self.feature_dim, self.feature_dim, 1) for _ in range(4)  # 支持4个尺度
        ])
        
        # 输出投影层
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # 注意力机制
        if self.use_attention:
            self.spatial_attention = SpatialAttention(self.feature_dim)
            self.channel_attention = ChannelAttention(self.feature_dim)
            
        # 特征融合
        self.feature_fusion = FeatureFusion()
        
        # 最终输出投影
        self.final_output = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, self.feature_dim, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播 - 实现PANet的多尺度融合"""
        # 如果输入是单个特征图，直接返回
        if not isinstance(features, list):
            features = [features]
            
        if len(features) == 1:
            return features[0]
        
        # 确保有足够的侧向连接层
        num_features = min(len(features), len(self.lateral_convs))
        features = features[:num_features]
        
        # 1. 自顶向下路径（FPN）
        # 应用侧向连接
        laterals = []
        for i, feat in enumerate(features):
            lateral = self.lateral_convs[i](feat)
            laterals.append(lateral)
        
        # 自顶向下融合（从高分辨率到低分辨率）
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样高分辨率特征
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            # 特征融合（相加）
            laterals[i] = laterals[i] + upsampled
        
        # 2. 自底向上路径（PANet）
        # 应用输出卷积
        for i in range(len(laterals)):
            laterals[i] = self.output_convs[i](laterals[i])
        
        # 自底向上融合（从低分辨率到高分辨率）
        for i in range(1, len(laterals)):
            # 上采样低分辨率特征
            upsampled = F.interpolate(
                laterals[i - 1], 
                size=laterals[i].shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
            # 特征融合（相加）
            laterals[i] = laterals[i] + upsampled
        
        # 3. 使用最高分辨率特征作为输出
        output = laterals[-1]
        
        # 4. 应用注意力机制
        if self.use_attention:
            output = self.spatial_attention(output)
            output = self.channel_attention(output)
        
        # 5. 最终输出投影
        output = self.final_output(output)
        
        return output

class PANetBlock(nn.Module):
    """PANet块，包含多个PANet模块"""
    
    def __init__(self, feature_dim: int, num_blocks: int = 3):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_blocks = num_blocks
        
        # 多个PANet模块
        self.panet_modules = nn.ModuleList([
            PANetModule() for _ in range(num_blocks)
        ])
        
        # 输出归一化
        self.norm = nn.BatchNorm2d(feature_dim)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        current_features = features
        
        # 依次通过每个PANet模块
        for panet_module in self.panet_modules:
            current_features = [panet_module(current_features)]
            
        # 归一化
        output = self.norm(current_features[0])
        
        return output

class PANetPyramid(nn.Module):
    """PANet金字塔结构，处理多尺度特征"""
    
    def __init__(self, feature_dims: List[int], output_dim: int):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # 为每个尺度创建投影层
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in feature_dims
        ])
        
        # PANet模块
        self.panet_module = PANetModule()
        
        # 输出融合
        self.output_fusion = nn.Conv2d(output_dim * len(feature_dims), output_dim, 1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        # 投影到相同维度
        projected_features = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            projected_features.append(projected)
            
        # PANet特征融合
        fused = self.panet_module(projected_features)
        
        # 多尺度特征拼接
        aligned_features = []
        for feature in projected_features:
            if feature.shape[-2:] != fused.shape[-2:]:
                feature = F.interpolate(feature, size=fused.shape[-2:], mode='bilinear', align_corners=False)
            aligned_features.append(feature)
            
        # 拼接所有特征
        concatenated = torch.cat(aligned_features, dim=1)
        
        # 输出融合
        output = self.output_fusion(concatenated)
        
        return output

class PANetWithSAM(nn.Module):
    """PANet与SAM结合的模块"""
    
    def __init__(self, sam_feature_dim: int, panet_feature_dim: int, output_dim: int):
        super().__init__()
        self.sam_feature_dim = sam_feature_dim
        self.panet_feature_dim = panet_feature_dim
        self.output_dim = output_dim
        
        # SAM特征投影
        self.sam_projection = nn.Conv2d(sam_feature_dim, output_dim, 1)
        
        # PANet特征投影
        self.panet_projection = nn.Conv2d(panet_feature_dim, output_dim, 1)
        
        # 特征融合
        self.fusion = PANetModule()
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 1)
        )
        
    def forward(self, sam_features: torch.Tensor, panet_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 特征投影
        sam_projected = self.sam_projection(sam_features)
        panet_projected = self.panet_projection(panet_features)
        
        # 特征融合
        fused = self.fusion([sam_projected, panet_projected])
        
        # 输出头
        output = self.output_head(fused)
        
        return output
