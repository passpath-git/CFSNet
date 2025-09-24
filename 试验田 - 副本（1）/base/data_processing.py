#!/usr/bin/env python3
"""
心脏数据处理模块
整合数据集、数据加载器、预处理和数据验证功能
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging
import cv2
import json
from datetime import datetime
import warnings

# 尝试导入可选依赖
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    warnings.warn("nibabel未安装，无法处理NIfTI文件")

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    # 创建虚拟的albumentations模块

# 导入prompt生成相关模块
try:
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_dilation, disk
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image未安装，无法使用prompt生成功能")
    class DummyAlbumentations:
        def Compose(self, transforms):
            def dummy_transform(image=None, mask=None):
                for transform in transforms:
                    if image is not None:
                        if hasattr(transform, '__call__'):
                            try:
                                if hasattr(transform, '__code__') and 'image' in transform.__code__.co_varnames:
                                    image = transform(image=image)
                                    if isinstance(image, dict) and 'image' in image:
                                        image = image['image']
                                else:
                                    image = transform(image)
                            except AttributeError:
                                pass
                    if mask is not None:
                        if hasattr(transform, '__call__'):
                            try:
                                if hasattr(transform, '__code__') and 'mask' in transform.__code__.co_varnames:
                                    mask = transform(mask=mask)
                                    if isinstance(mask, dict) and 'mask' in mask:
                                        mask = mask['mask']
                                else:
                                    mask = transform(mask)
                            except AttributeError:
                                pass
                return {'image': image, 'mask': mask}
            return dummy_transform
    A = DummyAlbumentations()

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL未安装，某些图像处理功能可能不可用")

logger = logging.getLogger(__name__)

# ============================================================================
# 数据集类
# ============================================================================

class CardiacDataset(Dataset):
    """心脏超声数据集类"""
    
    def __init__(self, 
                 data_root: str,
                 split_file: str,
                 transform=None,
                 target_transform=None,
                 image_size: Tuple[int, int] = (512, 512),
                 use_augmentation: bool = False,
                 use_prompts: bool = False,
                 num_classes: int = 5):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            split_file: 分割文件路径
            transform: 图像变换
            target_transform: 标签变换
            image_size: 目标图像尺寸
            use_augmentation: 是否使用数据增强
            use_prompts: 是否使用prompt生成
            num_classes: 类别数量
        """
        self.data_root = data_root
        self.split_file = split_file
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.use_prompts = use_prompts
        self.num_classes = num_classes
        
        # 加载患者ID列表
        self.patient_ids = self._load_patient_ids()
        
        # 构建文件路径列表
        self.image_paths, self.label_paths = self._build_file_paths()
        
        logger.debug(f"数据集初始化完成: {len(self.patient_ids)} 个患者")
    
    def _load_patient_ids(self) -> List[str]:
        """加载患者ID列表"""
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"分割文件不存在: {self.split_file}")
        
        with open(self.split_file, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
        
        return patient_ids
    
    def _build_file_paths(self) -> Tuple[List[str], List[str]]:
        """构建图像和标签文件路径列表"""
        image_paths = []
        label_paths = []
        
        for patient_id in self.patient_ids:
            # 尝试不同的患者ID格式
            possible_ids = [
                patient_id,  # 原始格式
                patient_id.zfill(8),  # 补零到8位
                f"patient{int(patient_id.replace('patient', '')):04d}" if patient_id.startswith('patient') else patient_id,  # 4位数字格式
            ]
            
            image_files = []
            label_files = []
            
            for pid in possible_ids:
                # 查找图像文件
                image_pattern = os.path.join(self.data_root, pid, f"{pid}_4CH_ED.nii.gz")
                if os.path.exists(image_pattern):
                    image_files = [image_pattern]
                    break
                
                # 也尝试不带patient前缀的格式
                if pid.startswith('patient'):
                    simple_id = pid.replace('patient', '')
                    image_pattern = os.path.join(self.data_root, pid, f"{simple_id}_4CH_ED.nii.gz")
                    if os.path.exists(image_pattern):
                        image_files = [image_pattern]
                        break
            
            for pid in possible_ids:
                # 查找标签文件
                label_pattern = os.path.join(self.data_root, pid, f"{pid}_4CH_ED_gt.nii.gz")
                if os.path.exists(label_pattern):
                    label_files = [label_pattern]
                    break
                
                # 也尝试不带patient前缀的格式
                if pid.startswith('patient'):
                    simple_id = pid.replace('patient', '')
                    label_pattern = os.path.join(self.data_root, pid, f"{simple_id}_4CH_ED_gt.nii.gz")
                    if os.path.exists(label_pattern):
                        label_files = [label_pattern]
                        break
            
            if image_files and label_files:
                image_paths.append(image_files[0])
                label_paths.append(label_files[0])
            else:
                logger.warning(f"患者 {patient_id} 的文件不完整")
        
        return image_paths, label_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据项"""
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # **修复：先加载标签确定最佳切片，再加载对应的图像切片**
        label = self._load_label(label_path)
        image = self._load_image_with_slice(image_path, getattr(self, '_best_slice_idx', None))
        
        # 应用变换
        if self.transform:
            if ALBUMENTATIONS_AVAILABLE and self.use_augmentation:
                # 使用albumentations的命名参数
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']
            elif ALBUMENTATIONS_AVAILABLE and not self.use_augmentation:
                # 验证阶段，只使用ToTensorV2
                if hasattr(self.transform, 'transforms') and len(self.transform.transforms) > 0:
                    # 只使用ToTensorV2变换
                    for transform in self.transform.transforms:
                        if hasattr(transform, '__class__') and 'ToTensorV2' in str(transform.__class__):
                            transformed = transform(image=image, mask=label)
                            image = transformed['image']
                            label = transformed['mask']
                            break
                    else:
                        # 如果没有ToTensorV2，直接使用原始数据
                        pass
            else:
                # 使用简单的变换
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        # 生成prompt（如果启用）
        prompts = {}
        if self.use_prompts and SKIMAGE_AVAILABLE:
            try:
                prompt_generator = CardiacPromptGenerator(
                    num_classes=self.num_classes,
                    point_perturbation=0.1,
                    box_expansion=0.1
                )
                # **增强prompt生成**：为每个类别生成更多点
                prompts = prompt_generator.generate_multi_prompt(
                    label.numpy() if isinstance(label, torch.Tensor) else label,
                    include_points=True,
                    include_boxes=True
                )
                
                # 为稀有类别生成额外的边界点
                if prompts and 'points' in prompts:
                    additional_points = prompt_generator.generate_boundary_points(
                        label.numpy() if isinstance(label, torch.Tensor) else label,
                        classes=[1, 3, 4],  # 为左心室、左心房、右心房生成更多点
                        points_per_class=2
                    )
                    if additional_points:
                        prompts['points'].extend(additional_points)
            except Exception as e:
                logger.warning(f"生成Prompt失败: {e}")
                prompts = {}
        
        result = {
            'image': image,
            'label': label,
            'image_path': image_path,
            'label_path': label_path
        }
        
        # 添加prompt信息
        if prompts:
            result.update(prompts)
        
        return result
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像"""
        try:
            if NIBABEL_AVAILABLE:
                # 抑制qfac警告 - 使用stderr重定向
                import sys
                import io
                from contextlib import redirect_stderr
                
                with redirect_stderr(io.StringIO()):
                    img = nib.load(image_path)
                    image = img.get_fdata()
            else:
                # 使用SimpleITK作为备选
                img = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(img)
            
            # MedSAM优化：改进3D医学图像处理
            if len(image.shape) == 3:
                # 使用与标签相同的最佳切片选择策略
                if hasattr(self, '_best_slice_idx') and self._best_slice_idx is not None:
                    # 使用标签确定的最佳切片
                    image = image[:, :, self._best_slice_idx]
                else:
                    # 如果没有最佳切片索引，使用中间切片
                    mid_slice = image.shape[2] // 2
                    image = image[:, :, mid_slice]
            
            # 调整尺寸
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # 归一化 - 防止除零错误
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = np.zeros_like(image)  # 如果图像全为同一值，设为0
            
            # 确保返回的是 (H, W) 格式的灰度图像
            if len(image.shape) == 3:
                # 如果是3D，取中间切片
                image = image[:, :, image.shape[2] // 2]
            
            # 转换为3通道以匹配SAM模型期望
            if len(image.shape) == 2:
                # 灰度图像转换为3通道，并调整维度顺序为 (C, H, W)
                image = np.stack([image, image, image], axis=0)
            elif len(image.shape) == 3 and image.shape[-1] == 3:
                # 如果已经是3通道，调整维度顺序为 (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            
            # **关键修复：归一化到[0,1]范围**
            image = image.astype(np.float32)
            if image.max() > 1.0:  # 如果图像值范围是0-255
                image = image / 255.0  # 归一化到[0,1]
            return image
        
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            # 返回零图像
            return np.zeros(self.image_size, dtype=np.float32)
    
    def _load_image_with_slice(self, image_path: str, best_slice_idx: int = None) -> np.ndarray:
        """加载图像，使用与标签相同的切片索引"""
        try:
            if NIBABEL_AVAILABLE:
                # 抑制qfac警告
                import io
                from contextlib import redirect_stderr
                
                with redirect_stderr(io.StringIO()):
                    img = nib.load(image_path)
                    image = img.get_fdata()
            else:
                # 使用SimpleITK作为备选
                img = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(img)
            
            # MedSAM优化：改进3D医学图像处理
            if len(image.shape) == 3:
                if best_slice_idx is not None:
                    # 使用标签确定的最佳切片
                    image = image[:, :, best_slice_idx]
                else:
                    # 如果没有最佳切片索引，使用中间切片
                    mid_slice = image.shape[2] // 2
                    image = image[:, :, mid_slice]
            
            # 调整尺寸
            if image.shape != self.image_size:
                image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # 归一化 - 防止除零错误
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            else:
                image = np.zeros_like(image)  # 如果图像全为同一值，设为0
            
            # 确保返回的是 (H, W) 格式的灰度图像
            if len(image.shape) == 3:
                # 如果是3D，取中间切片
                image = image[:, :, image.shape[2] // 2]
            
            # 转换为3通道以匹配SAM模型期望
            if len(image.shape) == 2:
                # 灰度图像转换为3通道，并调整维度顺序为 (C, H, W)
                image = np.stack([image, image, image], axis=0)
            elif len(image.shape) == 3 and image.shape[-1] == 3:
                # 如果已经是3通道，调整维度顺序为 (C, H, W)
                image = np.transpose(image, (2, 0, 1))
            
            # **关键修复：归一化到[0,1]范围**
            image = image.astype(np.float32)
            if image.max() > 1.0:  # 如果图像值范围是0-255
                image = image / 255.0  # 归一化到[0,1]
            return image
        
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            # 返回零图像
            return np.zeros((3,) + self.image_size, dtype=np.float32)
    
    def _load_label(self, label_path: str) -> np.ndarray:
        """加载标签"""
        try:
            if NIBABEL_AVAILABLE:
                # 抑制qfac警告 - 使用stderr重定向
                import sys
                import io
                from contextlib import redirect_stderr
                
                with redirect_stderr(io.StringIO()):
                    img = nib.load(label_path)
                    label = img.get_fdata()
            else:
                img = sitk.ReadImage(label_path)
                label = sitk.GetArrayFromImage(img)
            
            # 检查标签是否为空
            if label.size == 0:
                logger.error(f"标签文件为空: {label_path}")
                return np.zeros(self.image_size, dtype=np.int64)
            
            # 转换为2D（选择有最多非零标签的切片）
            if len(label.shape) == 3:
                # 找到有最多非零标签的切片
                non_zero_counts = []
                for z in range(label.shape[2]):
                    slice_data = label[:, :, z]
                    non_zero_count = np.count_nonzero(slice_data)
                    non_zero_counts.append(non_zero_count)
                
                # 选择非零标签最多的切片
                best_slice_idx = np.argmax(non_zero_counts)
                self._best_slice_idx = best_slice_idx  # 保存最佳切片索引
                label = label[:, :, best_slice_idx]
            
            # 调整尺寸
            if label.shape != self.image_size:
                label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # 先转换为整数，再检查值范围
            label = np.round(label).astype(np.int64)
            
            # 检查标签值范围
            unique_values = np.unique(label)
            if len(unique_values) > 1:
                # 有非零标签，检查是否在有效范围内
                if np.max(label) > 4 or np.min(label) < 0:
                    logger.warning(f"标签值超出范围 [0,4]: {label_path}, 范围: [{np.min(label)}, {np.max(label)}]")
                    label = np.clip(label, 0, 4)
            else:
                # 只有背景值，检查是否真的是背景
                if unique_values[0] != 0:
                    logger.warning(f"标签只有单一非零值: {label_path}, 值: {unique_values[0]}")
                    # 将非零值设为背景
                    label = np.zeros_like(label)
                else:
                    # 确实只有背景值，这是正常的，不需要警告
                    pass
            
            # 最终验证标签（只在有异常时警告）
            final_unique = np.unique(label)
            if len(final_unique) == 1 and final_unique[0] == 0:
                # 只有背景值是正常的，不需要警告
                pass
            
            return label
        
        except Exception as e:
            # 特殊处理压缩文件错误
            if "Compressed file ended before the end-of-stream marker was reached" in str(e):
                logger.error(f"❌ 标签文件损坏: {label_path}")
                logger.error(f"   错误详情: {e}")
                logger.error(f"   建议: 跳过此文件，使用其他数据")
                # 返回零标签，让训练继续
                return np.zeros(self.image_size, dtype=np.int64)
            else:
                logger.error(f"加载标签失败 {label_path}: {e}")
                # 返回零标签
                return np.zeros(self.image_size, dtype=np.int64)

# ============================================================================
# 数据加载器类
# ============================================================================

class CardiacDataLoader:
    """
    心脏数据加载器
    管理训练、验证和测试数据的加载
    """
    
    def __init__(self,
                 data_root: Optional[str] = None,
                 split_file_dir: Optional[str] = None,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 image_size: Tuple[int, int] = (512, 512),
                 num_classes: int = 5,
                 use_augmentation: bool = True,
                 use_prompts: bool = False,
                 dataset_type: str = 'auto',
                 config: Optional[Dict] = None):
        """
        初始化数据加载器
        
        Args:
            data_root: 数据根目录
            split_file_dir: 数据集分割文件目录
            batch_size: 批次大小
            num_workers: 工作进程数
            image_size: 图像尺寸
            num_classes: 类别数量
            use_augmentation: 是否使用数据增强
            use_prompts: 是否使用提示
            dataset_type: 数据集类型
            config: 配置参数
        """
        # 使用配置或默认值
        if config:
            self.data_root = config.get('data_root', data_root)
            self.split_file_dir = config.get('split_file_dir', split_file_dir)
            self.batch_size = config.get('batch_size', batch_size)
            self.num_workers = config.get('num_workers', num_workers)
            self.image_size = tuple(config.get('image_size', image_size))
            self.num_classes = config.get('num_classes', num_classes)
            self.use_augmentation = config.get('use_augmentation', use_augmentation)
            self.use_prompts = config.get('use_prompts', use_prompts)
            self.dataset_type = config.get('dataset_type', dataset_type)
        else:
            self.data_root = data_root
            self.split_file_dir = split_file_dir
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.image_size = image_size
            self.num_classes = num_classes
            self.use_augmentation = use_augmentation
            self.use_prompts = use_prompts
            self.dataset_type = dataset_type
        
        # 验证参数
        if not self.data_root:
            raise ValueError("data_root 不能为空")
        if not self.split_file_dir:
            raise ValueError("split_file_dir 不能为空")
        
        # 创建数据预处理器
        self.preprocessor = CardiacPreprocessor(
            image_size=self.image_size,
            use_augmentation=self.use_augmentation
        )
        
        # 创建数据变换
        self.transform = self.preprocessor.get_transform()
        self.target_transform = self.preprocessor.get_target_transform()
        
        # 分割文件映射
        self.split_files = {
            'train': 'subgroup_training.txt',
            'validation': 'subgroup_validation.txt',
            'test': 'subgroup_testing.txt'
        }
        
        logger.debug(f"数据加载器初始化完成: batch_size={self.batch_size}, image_size={self.image_size}")
    
    def create_dataloader(self, split: str = 'train') -> DataLoader:
        """
        创建数据加载器
        
        Args:
            split: 数据集分割 ('train', 'validation', 'test')
        
        Returns:
            DataLoader对象
        """
        split_file = os.path.join(self.split_file_dir, self.split_files[split])
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"分割文件不存在: {split_file}")
        
        # 创建数据集
        dataset = CardiacDataset(
            data_root=self.data_root,
            split_file=split_file,
            transform=self.transform,
            target_transform=self.target_transform,
            image_size=self.image_size,
            use_augmentation=self.use_augmentation and split == 'train'
        )
        
        # 创建采样器（仅用于训练）
        sampler = None
        if split == 'train':
            # 暂时禁用加权采样，使用随机采样
            # 这样可以避免复杂的权重计算问题
            pass
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == 'train' and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        logger.debug(f"创建 {split} 数据加载器: {len(dataset)} 个样本")
        return dataloader
    
    def _calculate_class_weights(self, dataset: CardiacDataset) -> Optional[torch.Tensor]:
        """计算类别权重"""
        try:
            # 统计类别分布
            class_counts = torch.zeros(self.num_classes)
            
            for i in range(len(dataset)):
                sample = dataset[i]
                if isinstance(sample, dict):
                    label = sample['label']
                else:
                    _, label = sample
                    
                if isinstance(label, torch.Tensor):
                    unique, counts = torch.unique(label, return_counts=True)
                    for cls, count in zip(unique, counts):
                        cls_val = cls.item()
                        if 0 <= cls_val < self.num_classes:
                            class_counts[cls_val] += count
            
            # 计算权重
            total = class_counts.sum()
            if total > 0:
                weights = total / (self.num_classes * class_counts + 1e-8)
                return weights
            
        except Exception as e:
            logger.warning(f"计算类别权重失败: {e}")
        
        return None

# ============================================================================
# 数据预处理器类
# ============================================================================

class CardiacPreprocessor:
    """
    心脏数据预处理器
    实现图像预处理、标准化和增强功能
    """
    
    def __init__(self,
                 image_size: Tuple[int, int] = (512, 512),
                 use_augmentation: bool = True,
                 normalization_method: str = 'imagenet'):  # MedSAM优化：默认使用ImageNet标准化
        """
        初始化预处理器
        
        Args:
            image_size: 目标图像尺寸
            use_augmentation: 是否使用数据增强
            normalization_method: 归一化方法 ('z_score', 'min_max', 'robust')
        """
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.normalization_method = normalization_method
        
        # 创建变换
        self.transform = self._create_transform()
        self.target_transform = self._create_target_transform()
        
        logger.debug(f"预处理器初始化完成: image_size={image_size}, augmentation={use_augmentation}")
    
    def _create_transform(self) -> Callable:
        """创建图像变换"""
        if ALBUMENTATIONS_AVAILABLE and self.use_augmentation:
            # MedSAM优化：针对Hiera Backbone的数据增强策略
            # 针对医学图像和Hiera Backbone的特定增强
            augmentation_transforms = [
                # 强制尺寸一致
                A.Resize(self.image_size[0], self.image_size[1]),
                
                # MedSAM优化：适合Hiera Backbone的增强策略
                A.HorizontalFlip(p=0.5),  # 水平翻转，保持心脏解剖结构
                A.Rotate(limit=5, p=0.2, border_mode=cv2.BORDER_CONSTANT),  # 轻微旋转±5°，避免过度变形
                
                # 医学图像特定的增强
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),  # 轻微亮度对比度调整
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # 轻微噪声，模拟医学图像噪声
                
                # 归一化和转换 - 适配MedSAM的输入要求
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
                ToTensorV2()
            ]
            
            return A.Compose(
                augmentation_transforms,
                additional_targets={'mask': 'mask'}, 
                is_check_shapes=False
            )
        else:
            # 使用类方法而不是局部函数，避免pickle问题
            return self._simple_transform
    
    def _simple_transform(self, image):
        """简单的图像变换方法（避免pickle问题）"""
        # 调整尺寸 - 修复尺寸比较逻辑
        target_h, target_w = self.image_size
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # MedSAM优化：使用ImageNet标准化
        if self.normalization_method == 'imagenet':
            # 使用ImageNet标准化，适配MedSAM预训练权重
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        elif self.normalization_method == 'z_score':
            image = (image - image.mean()) / (image.std() + 1e-8)
        elif self.normalization_method == 'min_max':
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        elif self.normalization_method == 'robust':
            q75, q25 = np.percentile(image, [75, 25])
            image = (image - np.median(image)) / (q75 - q25 + 1e-8)
        
        # 转换为tensor - 确保是 (C, H, W) 格式
        if len(image.shape) == 2:
            # 灰度图像，添加通道维度并复制为3通道
            image = np.stack([image, image, image], axis=0)  # (C, H, W)
        elif len(image.shape) == 3:
            # 彩色图像，从 (H, W, C) 转换为 (C, H, W)
            if image.shape[2] == 1:
                # 单通道，复制为3通道
                image = np.repeat(image, 3, axis=2)  # (H, W, 3)
            # 转换维度顺序从 (H, W, C) 到 (C, H, W)
            image = np.transpose(image, (2, 0, 1))
        
        return torch.from_numpy(image).float()
    
    def _create_target_transform(self) -> Callable:
        """创建标签变换"""
        return self._target_transform
    
    def _target_transform(self, label):
        """标签变换方法（避免pickle问题）"""
        # 标签已经在_load_label中调整了尺寸，这里只需要转换为tensor
        if isinstance(label, np.ndarray):
            return torch.from_numpy(label).long()
        else:
            return label.long()
    
    def get_transform(self) -> Callable:
        """获取图像变换"""
        return self.transform
    
    def get_target_transform(self) -> Callable:
        """获取标签变换"""
        return self.target_transform

# ============================================================================
# 数据验证器类
# ============================================================================

class DataValidator:
    """数据验证器类"""
    
    def __init__(self, data_root: str, split_file_dir: str):
        """
        初始化数据验证器
        
        Args:
            data_root: 数据根目录
            split_file_dir: 分割文件目录
        """
        self.data_root = data_root
        self.split_file_dir = split_file_dir
        
        # 验证配置
        self.validation_config = {
            'expected_formats': ['.nii.gz', '.nii'],
            'expected_image_patterns': ['*_4CH_ED.nii.gz'],
            'expected_label_patterns': ['*_4CH_ED_gt.nii.gz'],
            'required_files': ['Info_*.cfg'],
            'min_file_size': 1024,  # 最小文件大小（字节）
            'max_image_dimensions': (2048, 2048),  # 最大图像尺寸
            'min_image_dimensions': (64, 64),  # 最小图像尺寸
            'expected_label_values': [0, 1, 2, 3, 4]  # 期望的标签值（5个类别）
        }
        
        # 分割文件映射
        self.split_files = {
            'train': 'subgroup_training.txt',
            'validation': 'subgroup_validation.txt',
            'test': 'subgroup_testing.txt'
        }
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        验证整个数据集
        
        Returns:
            验证结果字典
        """
        logger.info("开始验证数据集...")
        
        results = {
            'overall_status': 'unknown',
            'total_patients': 0,
            'valid_patients': 0,
            'invalid_patients': 0,
            'split_validation': {},
            'common_issues': [],
            'recommendations': []
        }
        
        # 验证每个分割
        for split_name, split_file in self.split_files.items():
            split_file_path = os.path.join(self.split_file_dir, split_file)
            if os.path.exists(split_file_path):
                split_result = self._validate_split(split_name, split_file_path)
                results['split_validation'][split_name] = split_result
                results['total_patients'] += split_result['total_patients']
                results['valid_patients'] += split_result['valid_patients']
                results['invalid_patients'] += split_result['invalid_patients']
            else:
                logger.warning(f"分割文件不存在: {split_file_path}")
                results['split_validation'][split_name] = {
                    'status': 'missing_file',
                    'total_patients': 0,
                    'valid_patients': 0,
                    'invalid_patients': 0,
                    'issues': [f"分割文件不存在: {split_file}"]
                }
        
        # 计算整体状态
        if results['invalid_patients'] == 0:
            results['overall_status'] = 'valid'
        elif results['valid_patients'] > 0:
            results['overall_status'] = 'partial'
        else:
            results['overall_status'] = 'invalid'
        
        # 生成建议
        results['recommendations'] = self._generate_recommendations(results)
        
        logger.info(f"数据集验证完成: {results['overall_status']}")
        return results
    
    def _validate_split(self, split_name: str, split_file_path: str) -> Dict[str, Any]:
        """验证单个分割"""
        logger.info(f"验证 {split_name} 分割...")
        
        # 加载患者ID
        try:
            with open(split_file_path, 'r') as f:
                patient_ids = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"读取分割文件失败: {e}")
            return {
                'status': 'error',
                'total_patients': 0,
                'valid_patients': 0,
                'invalid_patients': 0,
                'issues': [f"读取分割文件失败: {e}"]
            }
        
        valid_patients = 0
        invalid_patients = 0
        issues = []
        
        for patient_id in patient_ids:
            patient_result = self._validate_patient(patient_id)
            if patient_result['valid']:
                valid_patients += 1
            else:
                invalid_patients += 1
                issues.extend(patient_result['issues'])
        
        status = 'valid' if invalid_patients == 0 else 'partial' if valid_patients > 0 else 'invalid'
        
        return {
            'status': status,
            'total_patients': len(patient_ids),
            'valid_patients': valid_patients,
            'invalid_patients': invalid_patients,
            'issues': issues
        }
    
    def _validate_patient(self, patient_id: str) -> Dict[str, Any]:
        """验证单个患者的数据"""
        issues = []
        
        # 查找图像文件
        image_pattern = os.path.join(self.data_root, f"*{patient_id}*_4CH_ED.nii.gz")
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            issues.append(f"患者 {patient_id} 缺少图像文件")
        else:
            # 验证图像文件
            for image_file in image_files:
                if not self._validate_file(image_file, 'image'):
                    issues.append(f"患者 {patient_id} 图像文件无效: {image_file}")
        
        # 查找标签文件
        label_pattern = os.path.join(self.data_root, f"*{patient_id}*_4CH_ED_gt.nii.gz")
        label_files = glob.glob(label_pattern)
        
        if not label_files:
            issues.append(f"患者 {patient_id} 缺少标签文件")
        else:
            # 验证标签文件
            for label_file in label_files:
                if not self._validate_file(label_file, 'label'):
                    issues.append(f"患者 {patient_id} 标签文件无效: {label_file}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_file(self, file_path: str, file_type: str) -> bool:
        """验证单个文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return False
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size < self.validation_config['min_file_size']:
                return False
            
            # 检查文件格式
            if file_type == 'image':
                # 验证图像文件
                if NIBABEL_AVAILABLE:
                    img = nib.load(file_path)
                    data = img.get_fdata()
                else:
                    img = sitk.ReadImage(file_path)
                    data = sitk.GetArrayFromImage(img)
                
                # 检查图像尺寸
                if len(data.shape) >= 2:
                    height, width = data.shape[:2]
                    if (height < self.validation_config['min_image_dimensions'][0] or 
                        height > self.validation_config['max_image_dimensions'][0] or
                        width < self.validation_config['min_image_dimensions'][1] or 
                        width > self.validation_config['max_image_dimensions'][1]):
                        return False
                
            elif file_type == 'label':
                # 验证标签文件
                if NIBABEL_AVAILABLE:
                    img = nib.load(file_path)
                    data = img.get_fdata()
                else:
                    img = sitk.ReadImage(file_path)
                    data = sitk.GetArrayFromImage(img)
                
                # 检查标签值
                unique_values = np.unique(data)
                for value in unique_values:
                    if value not in self.validation_config['expected_label_values']:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"验证文件失败 {file_path}: {e}")
            return False
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if results['overall_status'] == 'invalid':
            recommendations.append("数据集存在严重问题，建议检查数据完整性")
        elif results['overall_status'] == 'partial':
            recommendations.append("数据集部分有效，建议修复无效的患者数据")
        
        if results['invalid_patients'] > 0:
            recommendations.append(f"发现 {results['invalid_patients']} 个无效患者，建议检查其数据文件")
        
        # 检查常见问题
        common_issues = set()
        for split_result in results['split_validation'].values():
            if 'issues' in split_result:
                common_issues.update(split_result['issues'])
        
        if common_issues:
            recommendations.append("常见问题: " + "; ".join(list(common_issues)[:5]))
        
        return recommendations

# ============================================================================
# 工厂函数
# ============================================================================

def create_cardiac_dataloader(data_root: str, 
                            split_file_dir: str,
                            batch_size: int = 8,
                            num_workers: int = 4,
                            image_size: Tuple[int, int] = (512, 512),
                            use_augmentation: bool = True,
                            use_prompts: bool = False,
                            config: Optional[Dict] = None) -> CardiacDataLoader:
    """
    创建心脏数据加载器
    
    Args:
        data_root: 数据根目录
        split_file_dir: 分割文件目录
        batch_size: 批次大小
        num_workers: 工作进程数
        image_size: 图像尺寸
        use_augmentation: 是否使用数据增强
        use_prompts: 是否使用真实Prompt生成
        config: 配置参数
    
    Returns:
        CardiacDataLoader对象
    """
    return CardiacDataLoader(
        data_root=data_root,
        split_file_dir=split_file_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        use_augmentation=use_augmentation,
        use_prompts=use_prompts,
        config=config
    )

def create_cardiac_preprocessor(image_size: Tuple[int, int] = (512, 512),
                              use_augmentation: bool = True,
                              normalization_method: str = 'z_score') -> CardiacPreprocessor:
    """
    创建心脏数据预处理器
    
    Args:
        image_size: 目标图像尺寸
        use_augmentation: 是否使用数据增强
        normalization_method: 归一化方法
    
    Returns:
        CardiacPreprocessor对象
    """
    return CardiacPreprocessor(
        image_size=image_size,
        use_augmentation=use_augmentation,
        normalization_method=normalization_method
    )

def create_data_validator(data_root: str, split_file_dir: str) -> DataValidator:
    """
    创建数据验证器
    
    Args:
        data_root: 数据根目录
        split_file_dir: 分割文件目录
    
    Returns:
        DataValidator对象
    """
    return DataValidator(data_root=data_root, split_file_dir=split_file_dir)


# ============================================================================
# Prompt生成功能
# ============================================================================

class CardiacPromptGenerator:
    """心脏分割Prompt生成器"""
    
    def __init__(self, 
                 num_classes: int = 5,
                 point_perturbation: float = 0.1,
                 box_expansion: float = 0.1):
        """
        初始化Prompt生成器
        
        Args:
            num_classes: 类别数量
            point_perturbation: 点扰动强度
            box_expansion: 框扩展比例
        """
        self.num_classes = num_classes
        self.point_perturbation = point_perturbation
        self.box_expansion = box_expansion
        
        # 类别名称映射
        self.class_names = ['background', 'left_ventricle', 'right_ventricle', 
                           'left_atrium', 'right_atrium']
    
    def generate_points_from_mask(self, 
                                 mask: np.ndarray, 
                                 num_points: int = 3,
                                 use_centroid: bool = True,
                                 use_contour: bool = True) -> List[np.ndarray]:
        """
        从掩码生成关键点
        
        Args:
            mask: 分割掩码 (H, W)
            num_points: 每个区域生成的点数
            use_centroid: 是否使用质心
            use_contour: 是否使用轮廓点
            
        Returns:
            点列表，每个点为 [x, y, label]
        """
        if not SKIMAGE_AVAILABLE:
            return []
            
        points = []
        
        # 为每个非背景类别生成点
        for class_id in range(1, self.num_classes):
            class_mask = (mask == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
                
            # 连通域分析
            labeled_mask = label(class_mask)
            regions = regionprops(labeled_mask)
            
            for region in regions:
                region_points = []
                
                # 添加质心点
                if use_centroid:
                    centroid = region.centroid
                    region_points.append([int(np.round(centroid[1])), int(np.round(centroid[0])), class_id])
                
                # 添加轮廓点
                if use_contour and len(region_points) < num_points:
                    coords = np.round(region.coords).astype(np.int32)
                    contour = self._extract_contour_points(coords, num_points - len(region_points))
                    for point in contour:
                        region_points.append([int(np.round(point[1])), int(np.round(point[0])), class_id])
                
                points.extend(region_points)
        
        return points
    
    def generate_boxes_from_mask(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        从掩码生成边界框
        
        Args:
            mask: 分割掩码 (H, W)
            
        Returns:
            边界框列表，每个框为 [x1, y1, x2, y2]
        """
        if not SKIMAGE_AVAILABLE:
            return []
            
        boxes = []
        h, w = mask.shape
        
        # 为每个非背景类别生成边界框
        for class_id in range(1, self.num_classes):
            class_mask = (mask == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
                
            # 连通域分析
            labeled_mask = label(class_mask)
            regions = regionprops(labeled_mask)
            
            for region in regions:
                minr, minc, maxr, maxc = region.bbox
                minr, minc, maxr, maxc = int(np.round(minr)), int(np.round(minc)), int(np.round(maxr)), int(np.round(maxc))
                
                # 扩展边界框
                expand_h = int(float(maxr - minr) * self.box_expansion)
                expand_w = int(float(maxc - minc) * self.box_expansion)
                
                minr = int(np.clip(minr - expand_h, 0, h-1))
                minc = int(np.clip(minc - expand_w, 0, w-1))
                maxr = int(np.clip(maxr + expand_h, 0, h-1))
                maxc = int(np.clip(maxc + expand_w, 0, w-1))
                
                boxes.append(np.array([minc, minr, maxc, maxr], dtype=np.int32))
        
        return boxes
    
    def _extract_contour_points(self, coords: np.ndarray, num_points: int) -> List[np.ndarray]:
        """提取轮廓点"""
        if len(coords) <= num_points:
            return coords.astype(np.int32).tolist()
        
        # 计算轮廓
        contour_points = []
        for i in range(0, len(coords), max(1, len(coords) // num_points)):
            contour_points.append(coords[i])
        
        return contour_points[:num_points]
    
    def generate_multi_prompt(self, 
                            mask: np.ndarray, 
                            include_points: bool = True,
                            include_boxes: bool = True) -> Dict[str, Any]:
        """
        生成多种类型的Prompt
        
        Args:
            mask: 分割掩码
            include_points: 是否包含点
            include_boxes: 是否包含边界框
            
        Returns:
            Prompt字典
        """
        prompts = {}
        
        if include_points:
            points = self.generate_points_from_mask(mask)
            prompts['points'] = points
        
        if include_boxes:
            boxes = self.generate_boxes_from_mask(mask)
            prompts['boxes'] = boxes
        
        return prompts
    
    def generate_boundary_points(self, mask: np.ndarray, classes: List[int], points_per_class: int = 2) -> List[List[int]]:
        """
        为指定类别生成边界点，增强稀有类别的prompt
        
        Args:
            mask: 分割掩码
            classes: 需要生成边界点的类别列表
            points_per_class: 每个类别生成的点数
            
        Returns:
            边界点列表 [[x, y, class_id], ...]
        """
        if not SKIMAGE_AVAILABLE:
            return []
        
        boundary_points = []
        
        for class_id in classes:
            class_mask = (mask == class_id).astype(np.uint8)
            if class_mask.sum() == 0:
                continue
            
            # 查找边界
            from skimage.segmentation import find_boundaries
            boundaries = find_boundaries(class_mask, mode='inner')
            
            if boundaries.sum() > 0:
                # 获取边界坐标
                boundary_coords = np.argwhere(boundaries)
                
                if len(boundary_coords) >= points_per_class:
                    # 随机选择边界点
                    selected_indices = np.random.choice(
                        len(boundary_coords), 
                        size=points_per_class, 
                        replace=False
                    )
                    
                    for idx in selected_indices:
                        y, x = boundary_coords[idx]
                        # 添加小的扰动
                        x = int(np.clip(x + np.random.normal(0, self.point_perturbation * 10), 0, mask.shape[1]-1))
                        y = int(np.clip(y + np.random.normal(0, self.point_perturbation * 10), 0, mask.shape[0]-1))
                        boundary_points.append([x, y, class_id])
        
        return boundary_points
