# MedSAM提示编码器（基于SAM2的医学图像提示编码器）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Type
from configs.model_config import ModelConfig

class PromptEncoder(nn.Module):
    """MedSAM提示编码器，基于SAM2的医学图像提示处理"""
    
    def __init__(self, embed_dim: int, image_embedding_size: Tuple[int, int], 
                 input_image_size: Tuple[int, int], mask_in_chans: int,
                 activation: Type[nn.Module] = nn.GELU, config: dict = None):
        super().__init__()
        if config is None:
            config = ModelConfig.SAM_PROMPT_ENCODER
            
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        
        # 位置编码
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        
        # 点嵌入
        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        
        # 掩码输入处理
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
    
    def get_dense_pe(self) -> torch.Tensor:
        """获取密集位置编码"""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)
    
    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, 
                     pad: bool) -> torch.Tensor:
        """嵌入点提示"""
        points = points + 0.5  # 移动到像素中心
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding
    
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """嵌入框提示"""
        boxes = boxes + 0.5  # 移动到像素中心
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding
    
    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """嵌入掩码输入"""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding
    
    def _get_batch_size(self, points: Optional[Tuple[torch.Tensor, torch.Tensor]],
                        boxes: Optional[torch.Tensor], masks: Optional[torch.Tensor]) -> int:
        """获取批次大小"""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1
    
    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device
    
    def forward(self, points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                batch_size: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        if batch_size == -1:
            bs = self._get_batch_size(points, boxes, masks)
        else:
            bs = batch_size
        
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        
        # 处理点提示
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        
        # 处理框提示
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        
        # 处理掩码提示
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        
        return sparse_embeddings, dense_embeddings

class PositionEmbeddingRandom(nn.Module):
    """随机位置编码"""
    
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
    
    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """位置编码点，归一化到[0,1]"""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    
    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """为指定大小的网格生成位置编码"""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
    
    def forward_with_coords(self, coords_input: torch.Tensor, 
                           image_size: Tuple[int, int]) -> torch.Tensor:
        """位置编码未归一化到[0,1]的点"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class LayerNorm2d(nn.Module):
    """2D LayerNorm"""
    
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x