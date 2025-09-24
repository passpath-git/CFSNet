# MedSAM模型（基于SAM2的医学图像分割）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import sys
import os

from configs.cardiac_config import CardiacConfig

class SAMModel(nn.Module):
    """MedSAM模型，基于SAM2的医学图像分割，专为3D医学图像优化"""
    
    def __init__(self, config: dict = None):
        super().__init__()
        if config is None:
            config = CardiacConfig.SAM_CONFIG
            
        self.config = config
        self.image_size = config.get('image_size', 512)
        self.embed_dim = config.get('embed_dim', 256)
        
        # 图像预处理参数
        self.pixel_mean = torch.tensor(config.get('pixel_mean', [0.485, 0.456, 0.406])).view(-1, 1, 1)
        self.pixel_std = torch.tensor(config.get('pixel_std', [0.229, 0.224, 0.225])).view(-1, 1, 1)
        
        # 初始化MedSAM组件
        self._init_medsam_components()
        
        # 心脏分割头 - 将MedSAM特征转换为心脏类别预测
        self.cardiac_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 5, 1)  # 5个类别：背景(0) + 左心室(1) + 右心室(2) + 左心房(3) + 右心房(4)
        )
    
    def _init_medsam_components(self):
        """初始化MedSAM核心组件"""
        # 导入MedSAM组件
        from .image_encoder import ImageEncoder
        from .mask_decoder import MaskDecoder
        from .prompt_encoder import PromptEncoder
        
        # 图像编码器
        self.image_encoder = ImageEncoder()
        
        # 提示编码器
        if isinstance(self.image_size, (list, tuple)):
            img_size = self.image_size[0] if len(self.image_size) > 0 else self.image_size
        else:
            img_size = self.image_size
            
        self.prompt_encoder = PromptEncoder(
            embed_dim=self.embed_dim,
            image_embedding_size=(img_size // 16, img_size // 16),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )
        
        # 掩码解码器
        self.mask_decoder = MaskDecoder(
            transformer_dim=self.embed_dim,
            transformer=self._build_transformer(),
            num_multimask_outputs=3,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    
    def _build_transformer(self):
        """构建双向Transformer"""
        from .transformer import TwoWayTransformer
        
        return TwoWayTransformer(
            depth=2,
            embedding_dim=self.embed_dim,
            mlp_dim=2048,
            num_heads=8,
        )
    
    def forward(self, images: torch.Tensor, 
                points: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                multimask_output: bool = True,
                target_size: Optional[Tuple[int, int]] = None) -> Dict[str, torch.Tensor]:
        """前向传播"""
        # 图像预处理
        images = self.preprocess_images(images)
        
        # 图像编码
        image_embeddings = self.image_encoder(images)
        
        # 处理3D医学图像 - 如果是3D，取中间切片
        if len(images.shape) == 5:  # (B, C, D, H, W)
            # 取中间切片进行2D处理
            mid_slice = images.shape[2] // 2
            images_2d = images[:, :, mid_slice, :, :]
            image_embeddings = self.image_encoder(images_2d)
        
        # 提示编码 - 如果没有提示，创建默认提示
        if points is None and boxes is None and masks is None:
            # 创建默认提示：图像中心点作为前景点
            batch_size = images.shape[0]
            center_point = torch.tensor([[self.image_size//2, self.image_size//2]], 
                                      device=images.device, dtype=torch.float32)
            center_label = torch.tensor([[1]], device=images.device, dtype=torch.int32)
            points = (center_point.expand(batch_size, -1, -1),
                     center_label.expand(batch_size, -1))
        
        # 编码提示
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks
        )
        
        # 掩码解码
        masks, iou_pred, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
        )
        
        # 心脏分割预测 - 使用图像特征
        cardiac_features = image_embeddings
        
        # 确保特征图尺寸正确 - 上采样到正确的输出尺寸
        if target_size is not None:
            cardiac_features = F.interpolate(
                cardiac_features, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            # 回退到原始输入图像尺寸
            original_size = images.shape[-2:]
            cardiac_features = F.interpolate(
                cardiac_features, 
                size=original_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # 通过心脏分割头
        cardiac_logits = self.cardiac_head(cardiac_features)
        
        # 构建输出
        outputs = {
            'masks': masks,
            'iou_pred': iou_pred,
            'cardiac_logits': cardiac_logits,  # 添加心脏分割输出
            'predictions': cardiac_logits,  # 为了兼容训练脚本
        }
        
        return outputs
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """图像预处理"""
        # 确保图像在正确的设备上
        if self.pixel_mean.device != images.device:
            self.pixel_mean = self.pixel_mean.to(images.device)
            self.pixel_std = self.pixel_std.to(images.device)
            
        # 处理3D医学图像
        if len(images.shape) == 5:  # (B, C, D, H, W)
            # 取中间切片进行2D处理
            mid_slice = images.shape[2] // 2
            images = images[:, :, mid_slice, :, :]
        
        # 归一化 - 确保维度匹配
        if images.shape[1] != self.pixel_mean.shape[0]:
            # 如果通道数不匹配，调整pixel_mean和pixel_std的维度
            if images.shape[1] == 1:
                # 单通道图像，复制到3通道
                pixel_mean = self.pixel_mean.mean().expand(1, 1, 1)
                pixel_std = self.pixel_std.mean().expand(1, 1, 1)
            elif images.shape[1] == 3:
                # 3通道图像，使用原始的pixel_mean和pixel_std
                pixel_mean = self.pixel_mean
                pixel_std = self.pixel_std
            else:
                # 其他情况，使用平均值
                pixel_mean = self.pixel_mean.mean().expand(images.shape[1], 1, 1)
                pixel_std = self.pixel_std.mean().expand(images.shape[1], 1, 1)
        else:
            pixel_mean = self.pixel_mean
            pixel_std = self.pixel_std
            
        images = (images - pixel_mean) / pixel_std
        
        # 调整尺寸
        if isinstance(self.image_size, (list, tuple)):
            target_size = self.image_size
        else:
            target_size = (self.image_size, self.image_size)
            
        if images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, 
                mode='bilinear', align_corners=False
            )
            
        return images
    
    def generate_masks(self, images: torch.Tensor, 
                      points: Optional[torch.Tensor] = None,
                      boxes: Optional[torch.Tensor] = None,
                      masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """生成掩码"""
        with torch.no_grad():
            outputs = self.forward(images, points, boxes, masks)
            return outputs['masks']
    
    def get_image_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """获取图像嵌入（用于缓存）"""
        images = self.preprocess_images(images)
        return self.image_encoder(images)
    
    def predict_masks_from_embeddings(self, image_embeddings: torch.Tensor,
                                     points: Optional[torch.Tensor] = None,
                                     boxes: Optional[torch.Tensor] = None,
                                     masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """从预计算的图像嵌入预测掩码"""
        # 提示编码
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks
        )
        
        # 掩码解码
        masks, iou_pred, sam_output_tokens, object_score_logits = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
        )
        
        outputs = {
            'masks': masks,
            'iou_pred': iou_pred,
        }
            
        return outputs
    
    def set_image(self, images: torch.Tensor):
        """设置图像（用于交互式分割）"""
        self.image_embeddings = self.get_image_embeddings(images)
        
    def predict(self, points: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """预测（需要先调用set_image）"""
        if not hasattr(self, 'image_embeddings'):
            raise RuntimeError("请先调用set_image设置图像")
            
        return self.predict_masks_from_embeddings(
            self.image_embeddings, points, boxes, masks
        )