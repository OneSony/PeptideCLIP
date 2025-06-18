# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.data import Dictionary
from unicore.models import (BaseUnicoreModel, register_model,
                            register_model_architecture)
from unicore.modules import LayerNorm
import unicore

from .transformer_encoder_with_pair import TransformerEncoderWithPair
from .unimol import NonLinearHead, UniMolModel, base_architecture

logger = logging.getLogger(__name__)


@register_model("peptideclip")
class PocketAffinityModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--pocket1-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the pocket1 pooler layers",
        )
        parser.add_argument(
            "--pocket2-pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the pocket2 pooler layers",
        )
        parser.add_argument(
            "--pocket1-encoder-layers",
            type=int,
            help="pocket1 encoder layers",
        )
        parser.add_argument(
            "--pocket2-encoder-layers",
            type=int,
            help="pocket2 encoder layers",
        )
        parser.add_argument(
            "--recycling",
            type=int,
            default=1,
            help="recycling nums of decoder",
        )

    def __init__(self, args, pocket_dictionary):
        super().__init__()
        peptideclip_architecture(args)
        self.args = args
        
        
        # 两个口袋编码器（可以共享权重或独立）
        self.pocket1_model = UniMolModel(args.pocket1, pocket_dictionary)
        self.pocket2_model = UniMolModel(args.pocket2, pocket_dictionary)
        
        #for param in self.pocket1_model.parameters():
        #    param.requires_grad = False
        #for param in self.pocket2_model.parameters():
        #    param.requires_grad = False
            
        
        # 投影层
        self.pocket1_project = NonLinearHead(
            args.pocket1.encoder_embed_dim, 128, "relu"
        )
        self.pocket2_project = NonLinearHead(
            args.pocket2.encoder_embed_dim, 128, "relu"
        )
        
        
        #self.pocket1_project = nn.Sequential(
        #    nn.Linear(args.pocket1.encoder_embed_dim, 512),  # 512 -> 512
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(512, 256),                             # 512 -> 256
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(256, 128)                              # 256 -> 128
        #)

        #self.pocket2_project = nn.Sequential(
        #    nn.Linear(args.pocket2.encoder_embed_dim, 512),  # 512 -> 512
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(512, 256),                             # 512 -> 256
        #    nn.ReLU(),
        #    nn.Dropout(0.1),
        #    nn.Linear(256, 128)                              # 256 -> 128
        #)
        
        
        
        # 对比学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(14))
        
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.pocket_dictionary)

    def forward(
        self,
        pocket1_src_tokens,
        pocket1_src_distance,
        pocket1_src_edge_type,
        pocket2_src_tokens,
        pocket2_src_distance,
        pocket2_src_edge_type,
        pocket1_list=None,
        pocket2_list=None,
        encode=False,
        masked_tokens=None,
        features_only=True,
        is_train=True,
        **kwargs
    ):
        def get_dist_features(dist, et, model, flag="pocket"):
            """获取距离特征"""
            n_node = dist.size(-1)
            if flag == "pocket1":
                gbf_feature = self.pocket1_model.gbf(dist, et)
                gbf_result = self.pocket1_model.gbf_proj(gbf_feature)
            else:  # pocket2
                gbf_feature = self.pocket2_model.gbf(dist, et)
                gbf_result = self.pocket2_model.gbf_proj(gbf_feature)
            
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        # 编码第一个口袋
        pocket1_padding_mask = pocket1_src_tokens.eq(self.pocket1_model.padding_idx)
        pocket1_x = self.pocket1_model.embed_tokens(pocket1_src_tokens)
        pocket1_graph_attn_bias = get_dist_features(
            pocket1_src_distance, pocket1_src_edge_type, self.pocket1_model, "pocket1"
        )
        pocket1_outputs = self.pocket1_model.encoder(
            pocket1_x, padding_mask=pocket1_padding_mask, attn_mask=pocket1_graph_attn_bias
        )
        pocket1_encoder_rep = pocket1_outputs[0]

        # 编码第二个口袋
        pocket2_padding_mask = pocket2_src_tokens.eq(self.pocket2_model.padding_idx)
        pocket2_x = self.pocket2_model.embed_tokens(pocket2_src_tokens)
        pocket2_graph_attn_bias = get_dist_features(
            pocket2_src_distance, pocket2_src_edge_type, self.pocket2_model, "pocket2"
        )
        pocket2_outputs = self.pocket2_model.encoder(
            pocket2_x, padding_mask=pocket2_padding_mask, attn_mask=pocket2_graph_attn_bias
        )
        pocket2_encoder_rep = pocket2_outputs[0]

        # 取CLS token作为口袋表示
        pocket1_rep = pocket1_encoder_rep[:, 0, :]
        pocket2_rep = pocket2_encoder_rep[:, 0, :]

        # 通过投影层并归一化
        pocket1_emb = self.pocket1_project(pocket1_rep)
        pocket1_emb = pocket1_emb / pocket1_emb.norm(dim=1, keepdim=True)
        pocket2_emb = self.pocket2_project(pocket2_rep)
        pocket2_emb = pocket2_emb / pocket2_emb.norm(dim=1, keepdim=True)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(pocket1_emb, torch.transpose(pocket2_emb, 0, 1))

        # 处理重复口袋的掩码（避免同一个口袋与自己对比）
        bsz = similarity_matrix.shape[0]
        

        # 处理pocket1重复
        pockets1 = np.array(pocket1_list, dtype=str)
        pockets1 = np.expand_dims(pockets1, 1)
        matrix1_1 = np.repeat(pockets1, len(pockets1), 1)
        matrix1_2 = np.repeat(np.transpose(pockets1), len(pockets1), 0)
        pocket1_duplicate_matrix = matrix1_1 == matrix1_2
        pocket1_duplicate_matrix = 1 * pocket1_duplicate_matrix
        pocket1_duplicate_matrix = torch.tensor(
            pocket1_duplicate_matrix, dtype=similarity_matrix.dtype
        ).to(similarity_matrix.device)

        # 处理pocket2重复
        pockets2 = np.array(pocket2_list, dtype=str)
        pockets2 = np.expand_dims(pockets2, 1)
        matrix2_1 = np.repeat(pockets2, len(pockets2), 1)
        matrix2_2 = np.repeat(np.transpose(pockets2), len(pockets2), 0)
        pocket2_duplicate_matrix = matrix2_1 == matrix2_2
        pocket2_duplicate_matrix = 1 * pocket2_duplicate_matrix
        pocket2_duplicate_matrix = torch.tensor(
            pocket2_duplicate_matrix, dtype=similarity_matrix.dtype
        ).to(similarity_matrix.device)

        # 正样本标签（对角线）
        onehot_labels = torch.eye(bsz).to(similarity_matrix.device)
        
        # 构建掩码矩阵
        indicator_matrix = pocket1_duplicate_matrix + pocket2_duplicate_matrix - 2 * onehot_labels
        
        # 应用温度缩放和掩码
        similarity_matrix = similarity_matrix * self.logit_scale.exp()
        similarity_matrix = indicator_matrix * -1e6 + similarity_matrix

        return similarity_matrix, self.logit_scale.exp()

    def encode_single_pocket(
        self,
        pocket_src_tokens,
        pocket_src_distance,
        pocket_src_edge_type,
        encoder_type="pocket1"
    ):
        """编码单个口袋，用于推理时的表示学习"""
        if encoder_type == "pocket1":
            model = self.pocket1_model
            project = self.pocket1_project
        else:
            model = self.pocket2_model
            project = self.pocket2_project

        # 编码口袋
        pocket_padding_mask = pocket_src_tokens.eq(model.padding_idx)
        pocket_x = model.embed_tokens(pocket_src_tokens)
        
        n_node = pocket_src_distance.size(-1)
        gbf_feature = model.gbf(pocket_src_distance, pocket_src_edge_type)
        gbf_result = model.gbf_proj(gbf_feature)
        graph_attn_bias = gbf_result
        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
        
        pocket_outputs = model.encoder(
            pocket_x, padding_mask=pocket_padding_mask, attn_mask=graph_attn_bias
        )
        pocket_encoder_rep = pocket_outputs[0][:, 0, :]
        
        # 投影并归一化
        pocket_emb = project(pocket_encoder_rep)
        pocket_emb = pocket_emb / pocket_emb.norm(dim=-1, keepdim=True)
        
        return pocket_emb

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


@register_model_architecture("peptideclip", "peptideclip")
def peptideclip_architecture(args):
    """定义PeptideCLIP模型架构参数"""
    
    parser = argparse.ArgumentParser()
    args.pocket1 = parser.parse_args([])
    args.pocket2 = parser.parse_args([])

    # Pocket1 编码器参数
    args.pocket1.encoder_layers = getattr(args, "pocket1_encoder_layers", 15)
    args.pocket1.encoder_embed_dim = getattr(args, "pocket1_encoder_embed_dim", 512)
    args.pocket1.encoder_ffn_embed_dim = getattr(args, "pocket1_encoder_ffn_embed_dim", 2048)
    args.pocket1.encoder_attention_heads = getattr(args, "pocket1_encoder_attention_heads", 64)
    args.pocket1.dropout = getattr(args, "pocket1_dropout", 0.1)
    args.pocket1.emb_dropout = getattr(args, "pocket1_emb_dropout", 0.1)
    args.pocket1.attention_dropout = getattr(args, "pocket1_attention_dropout", 0.1)
    args.pocket1.activation_dropout = getattr(args, "pocket1_activation_dropout", 0.0)
    args.pocket1.pooler_dropout = getattr(args, "pocket1_pooler_dropout", 0.0)
    args.pocket1.max_seq_len = getattr(args, "pocket1_max_seq_len", 512)
    args.pocket1.activation_fn = getattr(args, "pocket1_activation_fn", "gelu")
    args.pocket1.pooler_activation_fn = getattr(args, "pocket1_pooler_activation_fn", "tanh")
    args.pocket1.post_ln = getattr(args, "pocket1_post_ln", False)
    args.pocket1.masked_token_loss = -1.0
    args.pocket1.masked_coord_loss = -1.0
    args.pocket1.masked_dist_loss = -1.0
    args.pocket1.x_norm_loss = -1.0
    args.pocket1.delta_pair_repr_norm_loss = -1.0

    # Pocket2 编码器参数（与Pocket1相同或独立设置）
    args.pocket2.encoder_layers = getattr(args, "pocket2_encoder_layers", 15)
    args.pocket2.encoder_embed_dim = getattr(args, "pocket2_encoder_embed_dim", 512)
    args.pocket2.encoder_ffn_embed_dim = getattr(args, "pocket2_encoder_ffn_embed_dim", 2048)
    args.pocket2.encoder_attention_heads = getattr(args, "pocket2_encoder_attention_heads", 64)
    args.pocket2.dropout = getattr(args, "pocket2_dropout", 0.1)
    args.pocket2.emb_dropout = getattr(args, "pocket2_emb_dropout", 0.1)
    args.pocket2.attention_dropout = getattr(args, "pocket2_attention_dropout", 0.1)
    args.pocket2.activation_dropout = getattr(args, "pocket2_activation_dropout", 0.0)
    args.pocket2.pooler_dropout = getattr(args, "pocket2_pooler_dropout", 0.0)
    args.pocket2.max_seq_len = getattr(args, "pocket2_max_seq_len", 512)
    args.pocket2.activation_fn = getattr(args, "pocket2_activation_fn", "gelu")
    args.pocket2.pooler_activation_fn = getattr(args, "pocket2_pooler_activation_fn", "tanh")
    args.pocket2.post_ln = getattr(args, "pocket2_post_ln", False)
    args.pocket2.masked_token_loss = -1.0
    args.pocket2.masked_coord_loss = -1.0
    args.pocket2.masked_dist_loss = -1.0
    args.pocket2.x_norm_loss = -1.0
    args.pocket2.delta_pair_repr_norm_loss = -1.0

    # 调用基础架构设置
    base_architecture(args)


@register_model_architecture("peptideclip", "peptideclip_base")
def peptideclip_base_architecture(args):
    """基础PeptideCLIP架构"""
    peptideclip_architecture(args)


@register_model_architecture("peptideclip", "peptideclip_shared")
def peptideclip_shared_architecture(args):
    """共享编码器的PeptideCLIP架构"""
    peptideclip_architecture(args)
    # 可以在这里设置共享编码器的特殊参数