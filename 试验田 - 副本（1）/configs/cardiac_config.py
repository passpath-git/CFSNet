# 心脏分割配置文件
# 融合HQSAM和PANet的配置参数
# 支持多种数据集格式和结构

import os
from typing import Dict, List, Tuple, Optional, Any

class CardiacConfig:
    """心脏分割配置类"""
    
    # 数据集配置
    DATASET_CONFIG = {
        'name': 'cardiac_ultrasound',  # 数据集名称
        'type': 'nifti',  # 支持 'nifti', 'image', 'mixed'
        'data_root': 'database/database_nifti',  # 数据根目录
        'split_file_dir': 'database/database_split',  # 数据集分割文件目录
        'output_root': 'shared/outputs',  # 输出根目录
        'checkpoint_dir': 'shared/checkpoints',  # 检查点目录
        
        # 数据分割配置
        'train_split_file': 'subgroup_training.txt',
        'val_split_file': 'subgroup_validation.txt', 
        'test_split_file': 'subgroup_testing.txt',
        
        # 数据格式配置 - 只保留NIfTI格式
        'image_extensions': ['.nii.gz', '.nii'],
        'label_extensions': ['.nii.gz', '.nii'],
        
        # 文件命名模式 - 只保留4CH_ED格式
        'file_patterns': {
            'image': ['*_4CH_ED.nii.gz'],
            'label': ['*_4CH_ED_gt.nii.gz']
        }
    }
    
    # MedSAM模型配置（基于SAM2的医学图像分割）
    SAM_CONFIG = {
        'model_type': 'medsam',  # 使用MedSAM模型
        'image_size': 512,  # 医学图像使用512x512分辨率
        'embed_dim': 256,  # MedSAM使用256维特征
        'num_classes': 5,  # 根据实际数据，包含5个类别
        'feature_dim': 256,
        'pixel_mean': [0.485, 0.456, 0.406],  # ImageNet预训练权重归一化
        'pixel_std': [0.229, 0.224, 0.225],   # ImageNet预训练权重归一化
        'use_high_res_features': False,  # 医学图像通常不需要高分辨率特征
        'use_checkpoint': True,
        'sam_checkpoint_path': 'shared/checkpoints/sam/sam_vit_b_01ec64.pth'  # SAM预训练权重路径
    }
    
    # PANet特征融合配置
    PANET_CONFIG = {
        'feature_dim': 256,
        'num_heads': 8,
        'dropout': 0.1,
        'use_attention': True,
        'fusion_strategy': 'adaptive',  # adaptive, concatenation, weighted
    }
    
    # 心脏分割配置 - 优化训练参数
    CARDIAC_CONFIG = {
        'num_classes': 5,  # 根据实际数据，包含5个类别
        'image_size': (512, 512),  # 优化：使用512x512平衡性能和显存
        'batch_size': 2,  # 优化：增加批次大小到2-4，提升训练效率
        'learning_rate': 2e-4,  # 提高学习率，促进模型学习
        'epochs': 100,
        'weight_decay': 3e-5,  # 回退到之前有效的权重衰减
        'accumulate_steps': 4,  # 减少梯度累积步数，配合更大batch_size
        'feature_dim': 256,
        'use_sam_features': True,
        'use_amp': True,  # 启用混合精度训练，节省显存
        'use_checkpoint': True,
        
        # 新增优化配置
        'optimizer': 'adamw',  # 使用AdamW优化器
        'scheduler': 'cosine_annealing',  # 使用余弦退火调度器
        'warmup_epochs': 3,  # 回退到之前有效的预热训练
        'min_lr': 5e-6,  # 回退到之前有效的最小学习率
        'max_lr': 5e-4,  # 回退到之前有效的最大学习率
        'use_warmup': True,  # 启用预热调度器
        
        # 正则化配置
        'dropout_rate': 0.15,  # 适度dropout，避免过度正则化
        'label_smoothing': 0.05,  # 减少标签平滑，保持标签信息
        'mixup_alpha': 0.1,  # 减少Mixup强度，避免过度增强
        
        # 损失函数配置
        'loss_type': 'focal_dice',  # 使用Focal Dice损失
        'focal_alpha': 0.25,  # Focal loss参数
        'focal_gamma': 2.0,  # Focal loss参数
        'dice_weight': 0.6,  # 平衡Dice和CE权重，避免过度偏向
        'ce_weight': 0.4,  # 增加CE权重，促进分类学习
        
        # 强数据增强配置 - 专门为小数据集设计，目标扩充到1000+样本
        'augmentation': {
            'rotation_range': 45,  # 大幅增加旋转范围
            'scale_range': (0.6, 1.5),  # 大幅增加缩放范围
            'brightness_range': (0.5, 1.5),  # 大幅增加亮度范围
            'contrast_range': (0.5, 1.5),  # 大幅增加对比度范围
            'noise_std': 0.05,  # 增加噪声强度
            'elastic_alpha': 5.0,  # 大幅增加弹性变换
            'elastic_sigma': 50.0,  # 弹性变换参数
            'horizontal_flip': True,  # 水平翻转
            'vertical_flip': True,  # 启用垂直翻转
            'gaussian_blur_prob': 0.4,  # 增加模糊概率
            'gaussian_blur_sigma': (0.3, 1.5),  # 扩大模糊范围
            'gamma_range': (0.7, 1.4),  # 伽马校正
            'hue_shift_range': (-0.1, 0.1),  # 色调偏移
            'saturation_range': (0.8, 1.2),  # 饱和度调整
            'cutout_prob': 0.3,  # 随机擦除
            'cutout_size': (0.05, 0.15),  # 擦除区域大小
            'mixup_alpha': 0.2,  # Mixup增强
            'cutmix_alpha': 0.3,  # CutMix增强
            'augmentation_multiplier': 8,  # 每个样本生成8个增强版本
        }
    }
    
    # 类别映射 - 根据实际数据，包含5个类别
    CLASS_MAPPING = {
        0: "background",
        1: "left_ventricle",    # 左心室
        2: "right_ventricle",   # 右心室
        3: "left_atrium",       # 左心房
        4: "right_atrium"       # 右心房
    }
    
    # 输出配置 - 简化结构，直接保存在一级目录下
    OUTPUT_CONFIG = {
        'results_dir': 'shared/outputs/results',  # 推理结果
        'predictions_dir': 'shared/outputs/predictions',  # 预测结果
        'visualizations_dir': 'shared/outputs/visualizations',  # 可视化结果
        'metrics_dir': 'shared/outputs/metrics',  # 评估指标
        'logs_dir': 'shared/outputs/logs',  # 训练日志
        'checkpoints_dir': 'shared/checkpoints',  # 模型检查点
        
        # 文件命名规则
        'naming_convention': {
            'results': '{patient_id}_{image_type}_{timestamp}',
            'predictions': '{patient_id}_{image_type}_pred',
            'visualizations': '{patient_id}_{image_type}_viz',
            'metrics': '{patient_id}_{image_type}_metrics'
        }
    }
    
    # 数据预处理配置
    PREPROCESSING_CONFIG = {
        'normalization': 'minmax',  # minmax, zscore, none
        'augmentation': {
            'rotation': [-15, 15],  # 旋转角度范围
            'translation': [-0.1, 0.1],  # 平移范围
            'scale': [0.9, 1.1],  # 缩放范围
            'flip_horizontal': True,  # 水平翻转
            'flip_vertical': False,  # 垂直翻转
            'brightness': [0.8, 1.2],  # 亮度调整
            'contrast': [0.8, 1.2]  # 对比度调整
        },
        'target_size': (512, 512),  # 目标图像尺寸
        'interpolation': 'bilinear'  # 插值方法
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'optimizer': 'adamw',  # adam, adamw, sgd
        'scheduler': 'cosine',  # step, cosine, plateau
        'early_stopping': True,  # 早停
        'patience': 10,  # 早停耐心值
        'save_best': True,  # 保存最佳模型
        'save_last': True,  # 保存最后一个模型
        'log_interval': 10,  # 日志记录间隔
        'eval_interval': 1,  # 评估间隔
        'checkpoint_interval': 5  # 检查点保存间隔
    }
    
    # 评估配置
    EVALUATION_CONFIG = {
        'metrics': ['dice', 'iou', 'hausdorff', 'surface_distance'],
        'threshold': 0.5,  # 二值化阈值
        'save_predictions': True,  # 保存预测结果
        'save_visualizations': True,  # 保存可视化结果
        'overlay_alpha': 0.6  # 叠加透明度
    }
    
    # 推理配置
    INFERENCE_CONFIG = {
        'batch_size': 1,  # 推理批次大小
        'use_tta': False,  # 使用测试时增强
        'tta_transforms': ['flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270'],
        'ensemble': False,  # 使用模型集成
        'save_format': 'nii.gz'  # 保存格式
    }
    
    # 系统配置
    SYSTEM_CONFIG = {
        'num_workers': 4,  # 数据加载器工作进程数
        'pin_memory': True,  # 固定内存
        'prefetch_factor': 2,  # 预取因子
        'persistent_workers': True,  # 持久化工作进程
        'max_memory_usage': 0.8  # 最大内存使用率
    }

    def __init__(self):
        """初始化配置"""
        pass

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取指定模型类型的配置"""
        if model_type == 'cardiac_sam':
            config = self.SAM_CONFIG.copy()
            config.update(self.CARDIAC_CONFIG)
        elif model_type == 'cardiac_panet':
            config = self.PANET_CONFIG.copy()
            config.update(self.CARDIAC_CONFIG)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return config

    def get_dataset_config(self) -> Dict[str, Any]:
        """获取数据集配置"""
        return self.DATASET_CONFIG.copy()

    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.OUTPUT_CONFIG.copy()

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """获取预处理配置"""
        return self.PREPROCESSING_CONFIG.copy()

    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.TRAINING_CONFIG.copy()

    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.EVALUATION_CONFIG.copy()

    def get_inference_config(self) -> Dict[str, Any]:
        """获取推理配置"""
        return self.INFERENCE_CONFIG.copy()

    def get_system_config(self) -> Dict[str, Any]:
        """获取系统配置"""
        return self.SYSTEM_CONFIG.copy()

    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必要的目录是否存在
            required_dirs = [
                self.DATASET_CONFIG['data_root'],
                self.DATASET_CONFIG['split_file_dir']
            ]
            
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    print(f"警告: 目录不存在: {dir_path}")
                    return False
            
            # 检查分割文件是否存在
            split_files = [
                self.DATASET_CONFIG['train_split_file'],
                self.DATASET_CONFIG['val_split_file'],
                self.DATASET_CONFIG['test_split_file']
            ]
            
            for split_file in split_files:
                split_path = os.path.join(self.DATASET_CONFIG['split_file_dir'], split_file)
                if not os.path.exists(split_path):
                    print(f"警告: 分割文件不存在: {split_path}")
                    return False
            
            print("配置验证通过！")
            return True
            
        except Exception as e:
            print(f"配置验证失败: {str(e)}")
            return False


def get_cardiac_config() -> Dict[str, Any]:
    """获取心脏分割配置"""
    config = CardiacConfig()
    return {
        'SAM_CONFIG': config.SAM_CONFIG,
        'PANET_CONFIG': config.PANET_CONFIG,
        'TRAINING_CONFIG': config.TRAINING_CONFIG,
        'DATASET_CONFIG': config.DATASET_CONFIG
    }
