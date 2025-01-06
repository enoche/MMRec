# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
ItemKNNCBF
################################################
Reference:
    https://github.com/CRIPAC-DIG/LATTICE
    Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches, ACM RecSys'19
"""


import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood


class ItemKNNCBF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ItemKNNCBF, self).__init__(config, dataset)

        self.knn_k = config['knn_k']
        self.shrink = config['shrink']

        # load dataset info
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        values = interaction_matrix.data
        indices = np.vstack((interaction_matrix.row, interaction_matrix.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = interaction_matrix.shape

        r_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(self.device)

        if self.v_feat is not None and self.t_feat is not None:
            item_fea = torch.cat((self.v_feat, self.t_feat), -1)
        elif self.v_feat is not None:
            item_fea = self.v_feat
        else:
            item_fea = self.t_feat

        self.dummy_embeddings = nn.Parameter(torch.Tensor([0.5, 0.5]))

        # build item-item sim matrix
        item_sim = self.build_item_sim_matrix(item_fea)
        self.scores_matrix = torch.mm(r_matrix, item_sim)

    def build_item_sim_matrix(self, features):
        i_norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        ij_norm = i_norm * i_norm.T + self.shrink
        ij = torch.mm(features, features.T)
        sim = ij.div(ij_norm)

        # top-k
        knn_val, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        weighted_adjacency_matrix = (torch.zeros_like(sim)).scatter_(-1, knn_ind, knn_val)
        return weighted_adjacency_matrix

    def build_item_sim_matrix_with_blocks(self, features, block_size=1000):
        from tqdm import tqdm
        """
        分块计算物品相似矩阵并显示进度条。

        :param features: Tensor, 物品特征向量，形状为 (num_items, feature_dim)
        :param block_size: int, 分块大小，默认 1000
        :return: Tensor, 权重邻接矩阵
        """
        num_items = features.size(0)
        i_norm = torch.norm(features, p=2, dim=-1, keepdim=True)
        shrink = self.shrink

        # 初始化相似矩阵
        weighted_adjacency_matrix = torch.zeros(num_items, num_items, device=features.device)

        # 分块计算
        for start_idx in tqdm(range(0, num_items, block_size), desc="Computing item similarities"):
            end_idx = min(start_idx + block_size, num_items)

            # 当前分块
            block_features = features[start_idx:end_idx]
            block_norm = i_norm[start_idx:end_idx]

            # 计算分块与所有物品的相似性
            ij = torch.mm(block_features, features.T)
            ij_norm = block_norm * i_norm.T + shrink
            sim = ij.div(ij_norm)

            # top-k
            knn_val, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
            weighted_adjacency_matrix[start_idx:end_idx] = (torch.zeros_like(sim)
                                                            .scatter_(-1, knn_ind, knn_val))

        return weighted_adjacency_matrix

    def calculate_loss(self, interaction):
        tmp_v = torch.tensor(0.0)
        return tmp_v

    def full_sort_predict(self, interaction):
        user = interaction[0]
        scores = self.scores_matrix[user]

        return scores

