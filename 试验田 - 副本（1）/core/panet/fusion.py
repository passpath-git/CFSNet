# PANet特征融合模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from configs.model_config import ModelConfig

class FeatureFusion(nn.Module):
    """PANet特征融合模块"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = ModelConfig.PANET_FUSION
            
        self.fusion_type = config['fusion_type']
        self.output_dim = config['output_dim']
        self.use_skip_connection = config['use_skip_connection']
        self.normalization = config['normalization']
        
        # 根据融合类型选择融合策略
        if self.fusion_type == 'adaptive':
            self.fusion_layer = AdaptiveFusion(self.output_dim)
        elif self.fusion_type == 'concatenation':
            self.fusion_layer = ConcatenationFusion(self.output_dim)
        elif self.fusion_type == 'weighted':
            self.fusion_layer = WeightedFusion(self.output_dim)
        elif self.fusion_type == 'attention':
            self.fusion_layer = AttentionFusion(self.output_dim)
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")
            
        # 归一化层
        if self.normalization == 'batch_norm':
            self.norm = nn.BatchNorm2d(self.output_dim)
        elif self.normalization == 'layer_norm':
            self.norm = nn.LayerNorm(self.output_dim)
        elif self.normalization == 'instance_norm':
            self.norm = nn.InstanceNorm2d(self.output_dim)
        else:
            self.norm = nn.Identity()
            
        # 输出投影
        self.output_proj = nn.Conv2d(self.output_dim, self.output_dim, 1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """前向传播"""
        if len(features) == 1:
            return features[0]
            
        # 特征融合
        fused_features = self.fusion_layer(features)
        
        # 归一化
        fused_features = self.norm(fused_features)
        
        # 输出投影
        output = self.output_proj(fused_features)
        
        # 跳跃连接
        if self.use_skip_connection and len(features) > 0:
            # 使用第一个特征作为跳跃连接
            skip_feature = features[0]
            if skip_feature.shape[1] != self.output_dim:
                skip_feature = F.interpolate(skip_feature, size=output.shape[-2:], mode='bilinear', align_corners=False)
            output = output + skip_feature
            
        return output

class AdaptiveFusion(nn.Module):
    """自适应特征融合"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_dim, output_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim // 4, output_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 确保所有特征具有相同的空间尺寸
        target_size = features[0].shape[-2:]
        aligned_features = []
        
        for feature in features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feature)
            
        # 特征求和
        fused = sum(aligned_features)
        
        # 自适应权重
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        return fused

class ConcatenationFusion(nn.Module):
    """拼接特征融合"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.conv = nn.Conv2d(output_dim, output_dim, 1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 确保所有特征具有相同的空间尺寸
        target_size = features[0].shape[-2:]
        aligned_features = []
        
        for feature in features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feature)
            
        # 通道维度拼接
        concatenated = torch.cat(aligned_features, dim=1)
        
        # 投影到目标维度
        fused = self.conv(concatenated)
        
        return fused

class WeightedFusion(nn.Module):
    """加权特征融合"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.ones(len(features)) / len(features))
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 确保所有特征具有相同的空间尺寸
        target_size = features[0].shape[-2:]
        aligned_features = []
        
        for feature in features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feature)
            
        # 加权融合
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, aligned_features))
        
        return fused

class AttentionFusion(nn.Module):
    """注意力特征融合"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 确保所有特征具有相同的空间尺寸
        target_size = features[0].shape[-2:]
        aligned_features = []
        
        for feature in features:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feature)
            
        # 重塑为序列形式
        batch_size, channels, height, width = aligned_features[0].shape
        aligned_features = [f.view(batch_size, channels, -1).transpose(1, 2) for f in aligned_features]
        
        # 注意力融合
        query = aligned_features[0]
        key = torch.cat(aligned_features, dim=1)
        value = torch.cat(aligned_features, dim=1)
        
        fused, _ = self.attention(query, key, value)
        
        # 重塑回空间形式
        fused = fused.transpose(1, 2).view(batch_size, channels, height, width)
        
        return fused

class MultiScaleFusion(nn.Module):
    """多尺度特征融合"""
    
    def __init__(self, feature_dims: List[int], output_dim: int):
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # 为每个尺度创建投影层
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in feature_dims
        ])
        
        # 融合层
        self.fusion = FeatureFusion()
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 投影到相同维度
        projected_features = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature)
            projected_features.append(projected)
            
        # 特征融合
        fused = self.fusion(projected_features)
        
        return fused
