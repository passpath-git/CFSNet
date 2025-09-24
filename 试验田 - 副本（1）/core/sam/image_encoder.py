# MedSAM图像编码器（基于SAM2的医学图像编码器）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from configs.model_config import ModelConfig

class ImageEncoder(nn.Module):
    """MedSAM图像编码器，基于SAM2的医学图像特征提取"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = ModelConfig.SAM_IMAGE_ENCODER
            
        self.embed_dim = config.get('embed_dim', 256)
        self.image_size = config.get('image_size', 512)
        self.patch_size = config.get('patch_size', 16)
        self.num_layers = config.get('num_layers', 12)
        self.num_heads = config.get('num_heads', 8)
        self.global_attn_indexes = config.get('global_attn_indexes', [2, 5, 8, 11])
        self.window_size = config.get('window_size', 14)
        self.use_rel_pos = config.get('use_rel_pos', True)
        self.out_chans = config.get('out_chans', 256)
        
        # 补丁嵌入层
        self.patch_embed = PatchEmbed(
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            in_chans=3,
            embed_dim=self.embed_dim,
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.embed_dim, self.image_size // self.patch_size, self.image_size // self.patch_size)
        )
        
        # Transformer编码器层
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                use_rel_pos=self.use_rel_pos,
                window_size=self.window_size if i not in self.global_attn_indexes else 0,
                input_size=(self.image_size // self.patch_size, self.image_size // self.patch_size),
            )
            for i in range(self.num_layers)
        ])
        
        # 输出投影层
        self.neck = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.out_chans, 1, bias=False),
            nn.BatchNorm2d(self.out_chans),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        B, C, H, W = x.shape
        
        # 补丁嵌入
        x = self.patch_embed(x)  # (B, H*W//patch_size^2, embed_dim)
        
        # 动态调整位置编码以匹配输入尺寸
        H_patch = H // self.patch_size
        W_patch = W // self.patch_size
        
        if self.pos_embed.shape[2:] != (H_patch, W_patch):
            # 如果位置编码尺寸不匹配，进行插值调整
            pos_embed = torch.nn.functional.interpolate(
                self.pos_embed, size=(H_patch, W_patch), mode='bilinear', align_corners=False
            )
        else:
            pos_embed = self.pos_embed
        
        # 将位置编码转换为序列格式
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (1, H_patch*W_patch, embed_dim)
        
        # 添加位置编码
        x = x + pos_embed
        
        # Transformer编码
        for block in self.blocks:
            x = block(x)
        
        # 重塑为空间特征
        x = x.permute(0, 2, 1).view(B, self.embed_dim, H_patch, W_patch)
        
        # 输出投影
        x = self.neck(x)
        
        return x

class PatchEmbed(nn.Module):
    """补丁嵌入层"""
    
    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int, int], 
                 in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W//patch_size^2, embed_dim)
        return x

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, use_rel_pos: bool = True,
                 window_size: int = 0, input_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias, use_rel_pos, window_size, input_size
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 use_rel_pos: bool = True, window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos and input_size is not None:
            # 相对位置编码
            window_size = 14  # 与官方SAM保持一致
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size - 1, self.head_dim))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if self.use_rel_pos:
            attn = self._add_rel_pos(attn, q, k)
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
    def _add_rel_pos(self, attn: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """添加相对位置编码"""
        # 简化实现，实际项目中需要完整的相对位置编码
        return attn

class MLPBlock(nn.Module):
    """MLP块"""
    
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x