# 模型配置文件
# 定义SAM和PANet的模型架构参数

class ModelConfig:
    """模型架构配置类"""
    
    # MedSAM图像编码器配置 - 基于SAM2的医学图像编码器，支持Hiera Backbone
    SAM_IMAGE_ENCODER = {
        'embed_dim': 256,  # MedSAM使用256维特征
        'image_size': 512,  # 医学图像通常使用512x512
        'patch_size': 16,
        'num_layers': 12,  # 保持12层
        'num_heads': 8,    # MedSAM使用8个注意力头
        'global_attn_indexes': [2, 5, 8, 11],  # 全局注意力索引
        'window_size': 14,
        'use_rel_pos': True,
        'out_chans': 256,
        
        # Hiera Backbone配置
        'use_hiera': True,  # 启用Hiera Backbone
        'hiera_depths': [2, 2, 6, 2],  # 各阶段深度
        'hiera_dims': [64, 128, 256, 512],  # 各阶段维度
        'hiera_num_heads': [2, 4, 8, 16],  # 各阶段头数
        'hiera_window_sizes': [7, 7, 7, 7],  # 各阶段窗口大小
    }
    
    # MedSAM掩码解码器配置（基于SAM2）
    SAM_MASK_DECODER = {
        'transformer_dim': 256,
        'transformer_depth': 2,
        'transformer_mlp_dim': 2048,
        'transformer_num_heads': 8,
        'num_multimask_outputs': 3,
        'iou_head_depth': 3,
        'iou_head_hidden_dim': 256,
        'use_high_res_features': False,  # 医学图像通常不需要高分辨率特征
        'iou_prediction_use_sigmoid': False,
        'pred_obj_scores': False,
        'pred_obj_scores_mlp': False,
        'use_multimask_token_for_obj_ptr': False,
    }
    
    # MedSAM提示编码器配置
    SAM_PROMPT_ENCODER = {
        'embed_dim': 256,
        'image_embedding_size': (32, 32),  # 512/16 = 32
        'input_image_size': (512, 512),
        'mask_in_chans': 16,  # SAM2使用16个掩码输入通道
    }
    
    # PANet注意力机制配置
    PANET_ATTENTION = {
        'feature_dim': 256,
        'num_heads': 8,
        'dropout': 0.1,
        'use_relative_pos': True,
        'attention_type': 'multi_head',  # multi_head, self_attention, cross_attention
    }
    
    # PANet特征融合配置
    PANET_FUSION = {
        'fusion_type': 'adaptive',  # adaptive, concatenation, weighted, attention
        'output_dim': 256,
        'use_skip_connection': True,
        'normalization': 'batch_norm',  # batch_norm, layer_norm, instance_norm
    }
