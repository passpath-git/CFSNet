# MedSAM双向Transformer（基于SAM2的医学图像Transformer）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type
import math

class TwoWayTransformer(nn.Module):
    """双向Transformer，用于MedSAM的掩码解码"""
    
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int,
                 activation: Type[nn.Module] = nn.ReLU, attention_downsample_rate: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
    
    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor,
                point_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        
        # 准备查询
        queries = point_embedding
        keys = image_embedding
        
        # 应用transformer块和最终层归一化
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        
        # 应用从点到图像的最终注意力层
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        
        return queries, keys

class TwoWayAttentionBlock(nn.Module):
    """双向注意力块"""
    
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU, attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        
        self.skip_first_layer_pe = skip_first_layer_pe
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, 
                query_pe: torch.Tensor, key_pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 自注意力块
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        
        # 交叉注意力块，token关注图像嵌入
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        
        # MLP块
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        
        # 交叉注意力块，图像嵌入关注token
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys

class Attention(nn.Module):
    """注意力层，允许在投影到查询、键和值后下采样嵌入大小"""
    
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1,
                 dropout: float = 0.0, kv_in_dim: int = None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        
        self.dropout_p = dropout
    
    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head
    
    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # 输入投影
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # 分离为头
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        
        dropout_p = self.dropout_p if self.training else 0.0
        # 注意力
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        
        return out

class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int, activation: nn.Module = nn.ReLU, 
                 sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
