import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import to_undirected

class LightGCNTorch(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightGCNTorch, self).__init__(config, dataset)

        # # load dataset info
        # self.interaction_matrix = dataset.inter_matrix(
        #     form='coo').astype(np.float32)

        # load parameters info
        df = dataset.dataset.df
        uid_field, iid_field = dataset.dataset.uid_field, dataset.dataset.iid_field
        inter_users, inter_items = df[uid_field].tolist(), df[iid_field].tolist()
        edge_index = torch.tensor([inter_users, inter_items], dtype=torch.long).to(self.device)
        self.edge_index = to_undirected(edge_index)

        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(self.n_layers)])
        self.reset_parameters()

        alpha = 1. / (self.n_layers + 1)
        alpha = torch.tensor([alpha] * (self.n_layers + 1))
        self.alpha = alpha
        # self.register_buffer('alpha', alpha)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        embedding_lst = [self.user_embedding, self.item_embedding]
        for embedding in embedding_lst:
            torch.nn.init.xavier_uniform_(embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        out = x * self.alpha[0]
        for i in range(self.n_layers):
            x = self.convs[i](x, self.edge_index)
            out = out + x * self.alpha[i + 1]
        user_out, item_out = out[:self.n_users], out[self.n_users:]
        return user_out, item_out

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user, :]
        posi_embeddings = item_all_embeddings[pos_item, :]
        negi_embeddings = item_all_embeddings[neg_item, :]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding.weight[user, :]
        posi_ego_embeddings = self.item_embedding.weight[pos_item, :]
        negi_ego_embeddings = self.item_embedding.weight[neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :] # interaction[0]은 겹치지 않는 user의 subset을 의미하는 것인가?

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1)) 

        return scores