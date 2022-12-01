# coding: utf-8
"""
https://github.com/jing-1/MVGAE
Paper: Multi-Modal Variational Graph Auto-Encoder for Recommendation Systems
IEEE TMM'21
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform
from torch.autograd import Variable

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization

EPS = 1e-15
MAX_LOGVAR = 10


class MVGAE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MVGAE, self).__init__(config, dataset)
        self.experts = ProductOfExperts()
        #self.dataset = config['dataset']
        self.dataset = 'amazon'
        self.batch_size = config['train_batch_size']
        self.num_user = self.n_users
        self.num_item = self.n_items
        num_user = self.n_users
        num_item = self.n_items
        num_layer = config['n_layers']
        self.aggr_mode = 'mean'
        self.concate = False
        self.dim_x = config['embedding_size']
        self.beta = config['beta']
        self.collaborative = nn.init.xavier_normal_(torch.rand((num_item, self.dim_x), requires_grad=True)).to(self.device)
        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        if self.v_feat is not None:
            self.v_gcn = GCN(self.device, self.v_feat, self.edge_index, self.batch_size, num_user, num_item, self.dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.device, self.t_feat, self.edge_index, self.batch_size, num_user, num_item, self.dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        self.c_gcn = GCN(self.device, self.collaborative, self.edge_index, self.batch_size, num_user, num_item,
                         self.dim_x,
                         self.aggr_mode, self.concate, num_layer=num_layer, dim_latent=128)  # 256)
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user + num_item, self.dim_x))).to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        # ndarray([598918, 2]) for ml-imdb
        return np.column_stack((rows, cols))

    def reparametrize(self, mu, logvar):
        logvar = logvar.clamp(max=MAX_LOGVAR)
        if self.training:
            return mu + torch.randn_like(logvar) * 0.1 * torch.exp(logvar.mul(0.5))
        else:
            return mu

    def dot_product_decode_neg(self, z, user, neg_items, sigmoid=True):
        # multiple negs, for comparison with MAML
        # print('user shape: ',user,user.shape)
        users = torch.unsqueeze(user, 1)
        # print('users shape: ', users,users.shape)
        neg_items = neg_items
        # print('neg_items: ', neg_items,neg_items.shape)
        # print('neg_items.size(1):', neg_items.size(0))
        re_users = users.repeat(1, neg_items.size(0))

        neg_values = torch.sum(z[re_users] * z[neg_items], -1)
        max_neg_value = torch.max(neg_values, dim=-1).values
        return torch.sigmoid(max_neg_value) if sigmoid else max_neg_value

    def dot_product_decode(self, z, edge_index, sigmoid=True):
        value = torch.sum(z[edge_index[0]] * z[edge_index[1]], dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self):
        v_mu, v_logvar = self.v_gcn()
        t_mu, t_logvar = self.t_gcn()
        c_mu, c_logvar = self.c_gcn()
        self.v_logvar = v_logvar
        self.t_logvar = t_logvar
        self.v_mu = v_mu
        self.t_mu = t_mu
        mu = torch.stack([v_mu, t_mu], dim=0)
        logvar = torch.stack([v_logvar, t_logvar], dim=0)

        pd_mu, pd_logvar, _ = self.experts(mu, logvar)
        del mu
        del logvar

        mu = torch.stack([pd_mu, c_mu], dim=0)
        logvar = torch.stack([pd_logvar, c_logvar], dim=0)

        pd_mu, pd_logvar, _ = self.experts(mu, logvar)
        del mu
        del logvar
        z = self.reparametrize(pd_mu, pd_logvar)

        # for more sparse dataset like amazon, use signoid to regulization. for alishop,dont use sigmoid for better results
        if 'amazon' in self.dataset:
            self.result_embed = torch.sigmoid(pd_mu)
        else:
            self.result_embed = pd_mu
        return pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar

    def recon_loss(self, z, pos_edge_index, user, neg_items):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        # for more sparse dataset like amazon, use signoid to regulization. for alishop,dont use sigmoid for better results
        if 'amazon' in self.dataset:
            z = torch.sigmoid(z)

        pos_scores = self.dot_product_decode(z, pos_edge_index, sigmoid=True)
        neg_scores = self.dot_product_decode_neg(z, user, neg_items, sigmoid=True)
        loss = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def kl_loss(self, mu, logvar):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logvar`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logvar (Tensor, optional): The latent space for
                :math:`\log\sigma^2`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        logvar = logvar.clamp(max=MAX_LOGVAR)
        return -0.5 * torch.mean(
            torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1))

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        #user = user.long()
        #pos_items = pos_items.long()
        #neg_items = torch.tensor(neg_items, dtype=torch.long)
        pos_edge_index = torch.stack([user, pos_items], dim=0)
        pd_mu, pd_logvar, z, v_mu, v_logvar, t_mu, t_logvar, c_mu, c_logvar = self.forward()

        z_v = self.reparametrize(v_mu, v_logvar)
        z_t = self.reparametrize(t_mu, t_logvar)
        z_c = self.reparametrize(c_mu, c_logvar)
        recon_loss = self.recon_loss(z, pos_edge_index, user, neg_items)
        kl_loss = self.kl_loss(pd_mu, pd_logvar)
        loss_multi = recon_loss + self.beta * kl_loss
        loss_v = self.recon_loss(z_v, pos_edge_index, user, neg_items) + self.beta * self.kl_loss(v_mu, v_logvar)
        loss_t = self.recon_loss(z_t, pos_edge_index, user, neg_items) + self.beta * self.kl_loss(t_mu, t_logvar)
        loss_c = self.recon_loss(z_c, pos_edge_index, user, neg_items) + self.beta* self.kl_loss(c_mu, c_logvar)
        return loss_multi + loss_v + loss_t + loss_c

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix


class GCN(torch.nn.Module):
    def __init__(self, device, features, edge_index, batch_size, num_user, num_item, dim_id, aggr_mode, concate,
                 num_layer, dim_latent=None):
        super(GCN, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(
                self.device)
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            nn.init.xavier_normal_(self.MLP.weight)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(
                self.device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat + self.dim_id, self.dim_id) if self.concate else nn.Linear(
                self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.g_layer1.weight)

        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        # nn.init.xavier_normal_(self.g_layer2.weight)

        self.conv_embed_4 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_4.weight)
        self.linear_layer4 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer4.weight)
        self.g_layer4 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer4.weight)
        self.conv_embed_5 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_5.weight)
        self.linear_layer5 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer5.weight)
        self.g_layer5 = nn.Linear(self.dim_id + self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id,
                                                                                                         self.dim_id)
        nn.init.xavier_normal_(self.g_layer5.weight)

    def forward(self):
        # print(self.features)
        # print(self.MLP.weight)
        temp_features = self.MLP(self.features) if self.dim_latent else self.features
        # print('temp feature: ',temp_features)
        x = torch.cat((self.preference, temp_features), dim=0)
        # print(x)
        x = F.normalize(x).to(self.device)
        # print(x)

        if self.num_layer > 0:
            h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer1(x))
            x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer1(h))
            del x_hat
            del h

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
            x_hat = F.leaky_relu(self.linear_layer2(x))
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                self.g_layer2(h))
            del h
            del x_hat

        mu = F.leaky_relu(self.conv_embed_4(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer4(x))
        mu = self.g_layer4(torch.cat((mu, x_hat), dim=1)) if self.concate else self.g_layer4(mu) + x_hat
        del x_hat

        logvar = F.leaky_relu(self.conv_embed_5(x, self.edge_index))
        x_hat = F.leaky_relu(self.linear_layer5(x))
        logvar = self.g_layer5(torch.cat((logvar, x_hat), dim=1)) if self.concate else self.g_layer5(logvar) + x_hat
        del x_hat
        return mu, logvar


class ProductOfExperts(torch.nn.Module):
    def __init__(self):
        super(ProductOfExperts, self).__init__()
        """Return parameters for product of independent experts.
        See https://arxiv.org/pdf/1410.7827.pdf for equations.
        @param mu: M x D for M experts
        @param logvar: M x D for M experts
        """

    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar, pd_var


class BaseModel(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(BaseModel, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index.long(), num_nodes=x.size(0))
            edge_index = edge_index.long()
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return F.dropout(aggr_out, p=0.1, training=self.training)

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
