# PANet注意力机制模块
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from configs.model_config import ModelConfig

class MultiHeadAttention(nn.Module):
    """PANet多头注意力机制"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = ModelConfig.PANET_ATTENTION
            
        self.feature_dim = config['feature_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.use_relative_pos = config['use_relative_pos']
        self.attention_type = config['attention_type']
        
        self.head_dim = self.feature_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # 注意力层
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout)
        
        # 相对位置编码
        if self.use_relative_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * 64 - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * 64 - 1, self.head_dim))
            
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        if key is None:
            key = query
        if value is None:
            value = key
            
        batch_size, seq_len, _ = query.shape
        
        # 投影查询、键、值
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 添加相对位置编码
        if self.use_relative_pos:
            attn_scores = self._add_relative_pos(attn_scores, q, k)
            
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # 应用注意力
        output = torch.matmul(attn_probs, v)
        
        # 重塑输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        output = self.out_proj(output)
        
        return output
    
    def _add_relative_pos(self, attn_scores: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """添加相对位置编码"""
        # 简化实现，实际项目中需要完整的相对位置编码
        return attn_scores

class SelfAttention(MultiHeadAttention):
    """自注意力机制"""
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(x, x, x, mask)

class CrossAttention(MultiHeadAttention):
    """交叉注意力机制"""
    
    def forward(self, query: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().forward(query, context, context, mask)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_dim, feature_dim // 8, 1)
        self.conv2 = nn.Conv2d(feature_dim // 8, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 空间注意力权重
        attention = self.conv1(x)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力
        return x * attention

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, feature_dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // reduction, feature_dim, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # 注意力权重
        attention = self.sigmoid(avg_out + max_out)
        
        # 应用注意力
        return x * attention
