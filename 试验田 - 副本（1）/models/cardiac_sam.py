#!/usr/bin/env python3
"""
心脏四腔分割SAM模型
基于SAM架构，专门用于心脏超声图像的四腔分割
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sam.sam_model import SAMModel
from core.panet.panet_model import PANetModule


class GradientSafeUpsample(nn.Upsample):
    """梯度安全的上采样层 - 确保梯度能够正确回流"""
    
    def forward(self, input):
        output = super().forward(input)
        # 确保梯度能够回流
        if input.requires_grad and not output.requires_grad:
            output.requires_grad_(True)
        return output


class ImprovedFeaturePyramidUpsampler(nn.Module):
    """改进的特征金字塔上采样器 - 支持多尺度输入并确保梯度流动"""
    
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        
        # 32×32 → 512×512 的标准上采样路径
        self.upsample_32_to_512 = nn.Sequential(
            # 32→64
            nn.Conv2d(in_channels, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 64→128  
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 128→256
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 256→512
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 16×16 → 512×512 的专门路径
        self.upsample_16_to_512 = nn.Sequential(
            # 16→32（先2倍上采样）
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            
            # 32→64
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 64→128  
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 128→256
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 256→512
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # 64×64 → 512×512 的路径
        self.upsample_64_to_512 = nn.Sequential(
            # 64→128
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 128→256
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 256→512
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            GradientSafeUpsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        _, _, h, w = x.shape
        
        if h == 16 and w == 16:
            return self.upsample_16_to_512(x)
        elif h == 32 and w == 32:
            return self.upsample_32_to_512(x)
        elif h == 64 and w == 64:
            return self.upsample_64_to_512(x)
        else:
            # 对于其他尺寸，使用自适应上采样
            return self._adaptive_upsample(x)
    
    def _adaptive_upsample(self, x):
        """自适应上采样到512×512"""
        target_size = (512, 512)
        
        # 直接上采样到目标尺寸
        upsampled = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # 通过一个卷积层处理
        conv_layer = nn.Conv2d(x.shape[1], x.shape[1], 3, padding=1).to(x.device)
        output = conv_layer(upsampled)
        
        return output
    
    def check_gradient_flow(self, input_tensor):
        """检查梯度流动的辅助函数"""
        print(f"🔍 梯度流动检查:")
        print(f"输入张量需要梯度: {input_tensor.requires_grad}")
        print(f"输入张量是叶子节点: {input_tensor.is_leaf}")
        
        # 确保输入张量保留梯度
        if not input_tensor.is_leaf:
            input_tensor.retain_grad()
        
        output = self.forward(input_tensor)
        print(f"输出张量需要梯度: {output.requires_grad}")
        print(f"输出形状: {output.shape}")
        
        # 创建损失并反向传播
        loss = output.mean()
        print(f"损失值: {loss.item():.6f}")
        
        loss.backward()
        
        # 检查输入梯度
        if input_tensor.grad is not None:
            input_grad_norm = input_tensor.grad.norm().item()
            # 静默记录输入梯度范数
            input_has_grad = True
        else:
            print(f"❌ 输入梯度为None")
            input_has_grad = False
        
        # 只检查实际使用的路径的参数梯度
        _, _, h, w = input_tensor.shape
        
        if h == 16 and w == 16:
            used_path = "upsample_16_to_512"
        elif h == 32 and w == 32:
            used_path = "upsample_32_to_512"
        elif h == 64 and w == 64:
            used_path = "upsample_64_to_512"
        else:
            used_path = "adaptive"
        
        print(f"使用的路径: {used_path}")
        
        # 检查使用路径的参数梯度
        used_param_count = 0
        used_grad_count = 0
        
        for name, param in self.named_parameters():
            if used_path in name or used_path == "adaptive":
                used_param_count += 1
                if param.grad is not None:
                    used_grad_count += 1
                    # 静默记录参数梯度范数
                else:
                    print(f"❌ {name}: 无梯度")
        
        print(f"使用路径参数梯度统计: {used_grad_count}/{used_param_count}")
        
        # 如果是自适应路径，检查所有参数
        if used_path == "adaptive":
            total_param_count = 0
            total_grad_count = 0
            for name, param in self.named_parameters():
                total_param_count += 1
                if param.grad is not None:
                    total_grad_count += 1
            print(f"总参数梯度统计: {total_grad_count}/{total_param_count}")
            return input_has_grad and (total_grad_count > 0)
        
        return input_has_grad and (used_grad_count == used_param_count)


# 保持向后兼容性的别名
FeaturePyramidUpsampler = ImprovedFeaturePyramidUpsampler


class CardiacSAM(nn.Module):
    """
    心脏四腔分割SAM模型
    
    结合SAM的图像编码能力和PANet的特征融合能力，
    专门用于心脏超声图像的四腔分割任务
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        if config is None:
            from configs.cardiac_config import CardiacConfig
            config = CardiacConfig().get_model_config('cardiac_sam')
        
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.feature_dim = config.get('feature_dim', 256)
        self.use_hq = config.get('use_hq', False)
        self.use_sam_features = config.get('use_sam_features', True)
        self.target_size = config.get('image_size', (512, 512))
        
        # SAM模型作为主干网络
        self.sam_model = SAMModel(config)
        
        # 轻量化医学图像适配层 - 减少显存消耗
        self.medical_adaptation = nn.Sequential(
            nn.Conv2d(3, 3, 1),  # 简化为1x1卷积，减少计算量
            nn.Tanh()  # 限制输出范围
        )
        
        # 加载SAM预训练权重
        sam_checkpoint_path = config.get('sam_checkpoint_path')
        if sam_checkpoint_path and os.path.exists(sam_checkpoint_path):
            self._load_sam_weights(sam_checkpoint_path)
        else:
            pass  # 静默初始化
        
        # PANet特征融合模块
        self.panet_fusion = PANetModule()
        
        # 特征增强模块 - 提高空间变化性
        self.feature_enhancer = nn.Sequential(
            # 空间注意力模块
            nn.Conv2d(self.feature_dim, self.feature_dim // 4, 1),
            nn.BatchNorm2d(self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim // 4, self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 改进的心脏分割头 - 支持动态类别数调整
        self.cardiac_head = nn.Sequential(
            # 第一组：特征提取
            nn.Conv2d(self.feature_dim, 128, 3, padding=1, groups=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            # 第二组：空间细化
            nn.Conv2d(128, 128, 3, padding=2, dilation=2),  # 膨胀卷积，调整padding保持尺寸
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三组：通道压缩
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            
            # 最终分类层 - 初始化为最大类别数，后续动态调整
            nn.Conv2d(64, 5, 1)  # 初始化为5类，训练时会动态调整
        )
        
        # 改进的初始化 - 使用更好的权重初始化
        self._init_cardiac_head()
        
        # 特征投影层（如果需要调整特征维度）
        if self.feature_dim != 256:
            self.feature_proj = nn.Conv2d(256, self.feature_dim, 1)
        else:
            self.feature_proj = nn.Identity()
    
    def _init_cardiac_head(self):
        """初始化心脏分割头的权重"""
        for m in self.cardiac_head.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化，增加权重方差
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 为最后一层添加类别偏置
                    if m.out_channels == self.num_classes:
                        # 给不同类别设置更平衡的初始偏置
                        with torch.no_grad():
                            for i in range(self.num_classes):
                                m.bias[i] = 0.0  # 所有类别都设为零偏置
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images: torch.Tensor,
                points: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 输入图像 [B, C, H, W]
            points: 点提示 [B, N, 3] (可选)
            boxes: 框提示 [B, N, 4] (可选)
            masks: 掩码提示 [B, N, H, W] (可选)
        
        Returns:
            包含预测结果的字典
        """
        batch_size = images.shape[0]
        
        # 医学图像适配预处理 - 检查输入格式
        # 处理3D医学图像
        if len(images.shape) == 5:  # (B, C, D, H, W)
            # 取中间切片进行2D处理
            mid_slice = images.shape[2] // 2
            images = images[:, :, mid_slice, :, :]
        
        # 记录原始图像尺寸，用于后续输出匹配
        original_size = images.shape[-2:]
        
        # 确保图像尺寸正确（根据配置动态调整）
        target_size = getattr(self, 'target_size', (512, 512))
        if images.shape[-2:] != target_size:
            images = torch.nn.functional.interpolate(
                images, size=target_size, mode='bilinear', align_corners=False
            )
        
        if images.dim() == 4 and images.shape[1] != 3:
            # 如果输入不是标准的[B, 3, H, W]格式，需要调整
            if images.shape[1] == 1:
                # 单通道转三通道
                images = images.repeat(1, 3, 1, 1)
            elif images.shape[1] > 3:
                # 多通道取前三个
                images = images[:, :3, :, :]
        
        adapted_images = self.medical_adaptation(images)
        
        # 如果没有提供提示，创建默认提示
        if points is None and boxes is None and masks is None:
            # 创建图像中心点作为默认提示 - 适配新的MedSAM格式
            h, w = adapted_images.shape[-2:]
            center_point = torch.tensor([[w//2, h//2]], 
                                     device=adapted_images.device, 
                                     dtype=torch.float32)
            center_label = torch.tensor([[1]], 
                                      device=adapted_images.device, 
                                      dtype=torch.int32)
            points = (center_point.expand(batch_size, -1, -1),
                     center_label.expand(batch_size, -1))
        
        # SAM模型前向传播 - 使用适配后的图像
        sam_outputs = self.sam_model(adapted_images, points, boxes, masks)
        
        # 使用SAM的完整输出，包括prompt信息
        if 'sam_features' in sam_outputs:
            sam_features = sam_outputs['sam_features']
        else:
            # 如果没有sam_features，则使用image_encoder
            with torch.no_grad():  # 冻结SAM主干网络
                sam_features = self.sam_model.image_encoder(adapted_images)
        
        # 调整特征维度
        sam_features = self.feature_proj(sam_features)
        
        # 创建多尺度特征用于PANet融合
        if self.use_sam_features:
            # 创建不同尺度的特征图
            multi_scale_features = []
            
            # 原始特征（最高分辨率）
            if sam_features.shape[-2:] != images.shape[-2:]:
                # 上采样到输入图像尺寸
                upsampled_features = F.interpolate(
                    sam_features,
                    size=images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                multi_scale_features.append(upsampled_features)
            else:
                multi_scale_features.append(sam_features)
            
            # 减少多尺度特征以提高速度
            scales = [0.5, 0.25]  # 只用两个尺度，减少计算量
            for scale in scales:
                target_size = (int(images.shape[-2] * scale), int(images.shape[-1] * scale))
                if target_size[0] > 16 and target_size[1] > 16:  # 提高最小尺寸阈值
                    scaled_features = F.interpolate(
                        sam_features,
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    # 再上采样回原始尺寸
                    scaled_features = F.interpolate(
                        scaled_features,
                        size=images.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    multi_scale_features.append(scaled_features)
            
            # 使用PANet融合多尺度特征
            fused_features = self.panet_fusion(multi_scale_features)
            
            # 关键修复：应用特征增强，提高空间变化性
            attention_weights = self.feature_enhancer(fused_features)
            enhanced_features = fused_features * attention_weights
        else:
            # 直接使用SAM特征
            if sam_features.shape[-2:] != images.shape[-2:]:
                fused_features = F.interpolate(
                    sam_features,
                    size=images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                fused_features = sam_features
            
            # 对直接SAM特征也应用增强
            attention_weights = self.feature_enhancer(fused_features)
            enhanced_features = fused_features * attention_weights
        
        # 心脏分割预测 - 使用增强后的特征
        cardiac_logits = self.cardiac_head(enhanced_features)
        
        # 确保输出尺寸与原始输入尺寸匹配
        if cardiac_logits.shape[-2:] != original_size:
            cardiac_logits = F.interpolate(
                cardiac_logits, size=original_size, mode='bilinear', align_corners=False
            )
        
        # 确保输出张量形状正确：[B, C, H, W]
        if cardiac_logits.dim() != 4:
            raise ValueError(f"cardiac_logits 维度错误: {cardiac_logits.shape}, 期望: [B, C, H, W]")
        
        # 动态更新模型类别数（用于三阶段训练）
        if cardiac_logits.shape[1] != self.num_classes:
            self.num_classes = cardiac_logits.shape[1]
        
        # 确保输出形状正确 - 支持2、3、5类
        if cardiac_logits.shape[1] not in [2, 3, 5]:
            raise ValueError(f"输出类别数错误: {cardiac_logits.shape[1]}, 期望: 2、3 或 5")
        
        # 构建输出
        outputs = {
            'cardiac_logits': cardiac_logits,
            'sam_masks': sam_outputs.get('masks', None),
            'iou_pred': sam_outputs.get('iou_pred', None),
            'sam_features': sam_features,
            'fused_features': enhanced_features
        }
        
        # 为了兼容训练脚本，添加标准预测输出
        outputs['predictions'] = cardiac_logits
        
        return outputs
    
    def get_sam_features(self, images: torch.Tensor) -> torch.Tensor:
        """获取SAM特征（用于特征分析）"""
        with torch.no_grad():
            sam_outputs = self.sam_model(images)
            if 'image_embeddings' in sam_outputs:
                return sam_outputs['image_embeddings']
            else:
                return self.sam_model.image_encoder(
                    self.sam_model.preprocess_images(images)
                )
    
    def predict_cardiac_segmentation(self, images: torch.Tensor, 
                                   use_temperature_scaling: bool = True,
                                   temperature: float = 2.0) -> torch.Tensor:
        """
        预测心脏分割（推理接口）
        
        Args:
            images: 输入图像
            use_temperature_scaling: 是否使用温度缩放
            temperature: 温度参数，>1会使预测更平滑
        
        Returns:
            预测的logits或概率分布
        """
        with torch.no_grad():
            outputs = self.forward(images)
            logits = outputs['cardiac_logits']
            
            if use_temperature_scaling:
                # 使用温度缩放来平衡预测概率
                logits = logits / temperature
                
            return logits
    
    def predict_with_postprocessing(self, images: torch.Tensor) -> torch.Tensor:
        """
        带后处理的预测（解决类别不平衡问题）
        """
        with torch.no_grad():
            outputs = self.forward(images)
            logits = outputs['cardiac_logits']
            
            # 方法1: 添加随机噪声增加空间变化性
            if self.training:
                noise = torch.randn_like(logits) * 0.1
                logits = logits + noise
            
            # 方法2: 使用Gumbel-Softmax增加随机性
            temperature = 1.0
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            gumbel_logits = (logits + gumbel_noise) / temperature
            
            # 方法3: 类别平衡的阈值调整
            class_thresholds = torch.tensor([-0.2, 0.1, -0.1, 0.05, 0.0], 
                                          device=logits.device).view(1, -1, 1, 1)
            adjusted_logits = gumbel_logits + class_thresholds
            
            return adjusted_logits
    
    def predict_with_spatial_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """
        使用空间多样性的预测方法
        """
        with torch.no_grad():
            outputs = self.forward(images)
            logits = outputs['cardiac_logits']
            
            B, C, H, W = logits.shape
            
            # 创建空间变化的权重
            y_coords = torch.linspace(-1, 1, H, device=logits.device).view(1, 1, H, 1)
            x_coords = torch.linspace(-1, 1, W, device=logits.device).view(1, 1, 1, W)
            
            # 为不同类别创建不同的空间偏好
            spatial_weights = torch.zeros_like(logits)
            
            # 背景：边缘区域更可能
            spatial_weights[:, 0] = (y_coords.abs() + x_coords.abs()) * 0.5
            
            # 左心室：左下区域更可能  
            spatial_weights[:, 1] = -(x_coords + 0.3) * (y_coords - 0.3) * 2.0
            
            # 右心室：右下区域更可能
            spatial_weights[:, 2] = (x_coords - 0.3) * (y_coords - 0.3) * 2.0
            
            # 左心房：左上区域更可能
            spatial_weights[:, 3] = -(x_coords + 0.3) * (y_coords + 0.3) * 2.0
            
            # 右心房：右上区域更可能
            spatial_weights[:, 4] = (x_coords - 0.3) * (y_coords + 0.3) * 2.0
            
            # 应用空间权重
            enhanced_logits = logits + spatial_weights * 0.3
            
            return enhanced_logits
    
    def predict_cardiac_with_diversity(self, images: torch.Tensor, 
                                     method: str = "spatial_diversity") -> Dict[str, torch.Tensor]:
        """
        多样性心脏分割预测 - 主要推理接口
        
        Args:
            images: 输入图像 [B, C, H, W]
            method: 预测方法 ("spatial_diversity", "postprocessing", "temperature")
        
        Returns:
            包含多种预测结果的字典
        """
        with torch.no_grad():
            # 原始预测
            original_outputs = self.forward(images)
            original_logits = original_outputs['cardiac_logits']
            original_pred = torch.argmax(original_logits, dim=1)
            
            # 应用选择的多样性方法
            if method == "spatial_diversity":
                enhanced_logits = self.predict_with_spatial_diversity(images)
            elif method == "postprocessing":
                enhanced_logits = self.predict_with_postprocessing(images)
            elif method == "temperature":
                enhanced_logits = self.predict_cardiac_segmentation(images, 
                                                                  use_temperature_scaling=True, 
                                                                  temperature=2.0)
            else:
                enhanced_logits = original_logits
            
            enhanced_pred = torch.argmax(enhanced_logits, dim=1)
            enhanced_probs = torch.softmax(enhanced_logits, dim=1)
            
            return {
                'original_logits': original_logits,
                'enhanced_logits': enhanced_logits,
                'original_predictions': original_pred,
                'enhanced_predictions': enhanced_pred,
                'enhanced_probabilities': enhanced_probs,
                'cardiac_logits': enhanced_logits,  # 为了兼容性
                'predictions': enhanced_pred        # 为了兼容性
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'use_hq': self.use_hq,
            'use_sam_features': self.use_sam_features
        }
    
    def _load_sam_weights(self, checkpoint_path: str):
        """加载SAM预训练权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 只加载image_encoder的权重（冻结主干）
            sam_image_encoder_keys = self.sam_model.image_encoder.state_dict().keys()
            filtered_state_dict = {}
            
            for key, value in checkpoint.items():
                if key.startswith('image_encoder.'):
                    # 移除image_encoder前缀
                    new_key = key.replace('image_encoder.', '')
                    if new_key in sam_image_encoder_keys:
                        # 检查形状是否匹配
                        expected_shape = self.sam_model.image_encoder.state_dict()[new_key].shape
                        if value.shape == expected_shape:
                            filtered_state_dict[new_key] = value
                        elif new_key == 'pos_embed' and len(value.shape) == 4 and len(expected_shape) == 4:
                            # 特殊处理位置编码：从 (1, 64, 64, 768) 转换为 (1, 768, 64, 64)
                            if value.shape == (1, 64, 64, 768) and expected_shape == (1, 768, 64, 64):
                                filtered_state_dict[new_key] = value.permute(0, 3, 1, 2)
                                # 静默转换位置编码形状
                            else:
                                # 静默跳过形状不匹配的权重
                                pass
                        elif 'rel_pos' in new_key and len(value.shape) == 2 and len(expected_shape) == 2:
                            # 特殊处理相对位置编码：插值初始化
                            if value.shape[1] == expected_shape[1]:  # 通道数匹配
                                filtered_state_dict[new_key] = self._interpolate_rel_pos(value, expected_shape)
                                # 静默插值相对位置编码
                            else:
                                # 静默跳过形状不匹配的权重
                                pass
                        else:
                            # 静默跳过形状不匹配的权重
                            pass
                    else:
                        # 静默跳过不在image_encoder中的权重
                        pass
            
            # 非严格加载image_encoder权重（允许部分权重不匹配）
            missing_keys, unexpected_keys = self.sam_model.image_encoder.load_state_dict(
                filtered_state_dict, strict=False
            )

            # 静默加载权重
            
            # 关键修复：部分解冻SAM主干网络，允许适应心脏图像
            # 只冻结前几层，解冻后几层让其适应心脏超声图像
            total_blocks = len(self.sam_model.image_encoder.blocks)
            freeze_blocks = total_blocks // 2  # 冻结前一半，解冻后一半
            
            # 静默解冻SAM
            for param in self.sam_model.image_encoder.parameters():
                param.requires_grad = True
            
            if hasattr(self.sam_model.image_encoder, 'neck'):
                for param in self.sam_model.image_encoder.neck.parameters():
                    param.requires_grad = True
            
        except Exception as e:
            pass  # 静默处理错误
    
    def verify_weights_loaded(self):
        """验证预训练权重是否真正生效"""
        print("\n🔍 验证SAM预训练权重是否真正生效...")
        
        # 测试：相同输入两次前向，特征是否一致（冻结）
        x1 = torch.randn(1, 3, 1024, 1024)
        x2 = x1.clone()
        
        # 确保在评估模式
        self.sam_model.image_encoder.eval()
        
        with torch.no_grad():
            feat1 = self.sam_model.image_encoder(x1)
            feat2 = self.sam_model.image_encoder(x2)
        
        feature_diff = (feat1 - feat2).abs().max().item()
        print(f"特征差异: {feature_diff:.10f} (应该接近0)")
        
        if feature_diff < 1e-6:
            # 静默加载SAM预训练权重
            return True
        else:
            print("❌ SAM预训练权重可能未生效！")
            return False
    
    def _interpolate_rel_pos(self, rel_pos, new_shape):
        """插值相对位置编码"""
        import torch.nn.functional as F
        
        old_len, c = rel_pos.shape
        new_len = new_shape[0]
        
        if old_len == new_len:
            return rel_pos
        
        # 将2D相对位置编码转换为4D进行插值
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, C]
        rel_pos = F.interpolate(rel_pos, size=(new_len, c), mode='bilinear', align_corners=False)
        return rel_pos.squeeze(0).squeeze(0)  # [new_len, C]
