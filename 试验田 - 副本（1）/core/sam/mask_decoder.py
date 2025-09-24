# MedSAM掩码解码器（基于SAM2的医学图像掩码解码器）
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Type
from configs.model_config import ModelConfig

class MaskDecoder(nn.Module):
    """MedSAM掩码解码器，基于SAM2的医学图像掩码预测"""
    
    def __init__(self, config: dict = None, **kwargs):
        super().__init__()
        if config is None:
            config = ModelConfig.SAM_MASK_DECODER
            
        # 从kwargs或config获取参数
        self.transformer_dim = kwargs.get('transformer_dim', config.get('transformer_dim', 256))
        self.transformer = kwargs.get('transformer', None)
        self.num_multimask_outputs = kwargs.get('num_multimask_outputs', config.get('num_multimask_outputs', 3))
        self.activation = kwargs.get('activation', nn.GELU)
        self.iou_head_depth = kwargs.get('iou_head_depth', config.get('iou_head_depth', 3))
        self.iou_head_hidden_dim = kwargs.get('iou_head_hidden_dim', config.get('iou_head_hidden_dim', 256))
        self.use_high_res_features = kwargs.get('use_high_res_features', config.get('use_high_res_features', False))
        self.iou_prediction_use_sigmoid = kwargs.get('iou_prediction_use_sigmoid', config.get('iou_prediction_use_sigmoid', False))
        self.pred_obj_scores = kwargs.get('pred_obj_scores', config.get('pred_obj_scores', False))
        self.pred_obj_scores_mlp = kwargs.get('pred_obj_scores_mlp', config.get('pred_obj_scores_mlp', False))
        self.use_multimask_token_for_obj_ptr = kwargs.get('use_multimask_token_for_obj_ptr', config.get('use_multimask_token_for_obj_ptr', False))
        
        # IoU token
        self.iou_token = nn.Embedding(1, self.transformer_dim)
        self.num_mask_tokens = self.num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)
        
        # 对象分数预测
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, self.transformer_dim)
        
        # 输出上采样
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.transformer_dim, self.transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(self.transformer_dim // 4),
            self.activation(),
            nn.ConvTranspose2d(
                self.transformer_dim // 4, self.transformer_dim // 8, kernel_size=2, stride=2
            ),
            self.activation(),
        )
        
        # 高分辨率特征
        if self.use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                self.transformer_dim, self.transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                self.transformer_dim, self.transformer_dim // 4, kernel_size=1, stride=1
            )
        
        # 输出超网络MLP
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])
        
        # IoU预测头
        self.iou_prediction_head = MLP(
            self.transformer_dim,
            self.iou_head_hidden_dim,
            self.num_mask_tokens,
            self.iou_head_depth,
            sigmoid_output=self.iou_prediction_use_sigmoid,
        )
        
        # 对象分数预测头
        if self.pred_obj_scores:
            if self.pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(self.transformer_dim, self.transformer_dim, 1, 3)
            else:
                self.pred_obj_score_head = nn.Linear(self.transformer_dim, 1)
    
    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool, repeat_image: bool = False,
                high_res_features: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )
        
        # 选择正确的掩码输出
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]
        
        # 选择SAM输出token
        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape
        
        return masks, iou_pred, sam_tokens_out, object_score_logits
    
    def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                     sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                     repeat_image: bool, high_res_features: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测掩码"""
        # 连接输出token
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat([
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ], dim=0)
            s = 1
        else:
            output_tokens = torch.cat([
                self.iou_token.weight, 
                self.mask_tokens.weight
            ], dim=0)
        
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        
        # 扩展图像数据
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        
        # 运行transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]
        
        # 上采样掩码嵌入并预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)
        
        # 生成掩码
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # 默认对象分数
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        
        return masks, iou_pred, mask_tokens_out, object_score_logits

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