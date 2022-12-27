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

    def calculate_loss(self, interaction):
        tmp_v = torch.tensor(0.0)
        return tmp_v

    def full_sort_predict(self, interaction):
        user = interaction[0]
        scores = self.scores_matrix[user]

        return scores

