# coding: utf-8
# @email: georgeguo.gzq.cn@gmail.com
r"""
LGMRec
################################################
Reference:
    https://github.com/georgeguo-cn/LGMRec
    AAAI'2024: [LGMRec: Local and Global Graph Learning for Multimodal Recommendation]
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class LGMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LGMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = config['cf_model']
        self.n_mm_layer = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.tau = 0.2

        self.n_nodes = self.n_users + self.n_items

        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        
        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1-self.keep_rate)

        # load item modal features and define hyperedges embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.hyper_num)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))
            
    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)
    
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))
    
    # collaborative graph embedding
    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs
    
    # modality graph embedding
    def mge(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        # user_feats = self.user_embedding.weight
        mge_feats = torch.concat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats
    
    def forward(self):
        # hyperedge dependencies constructing
        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)
        
        # CGE: collaborative graph embedding
        cge_embs = self.cge()
        
        if self.v_feat is not None and self.t_feat is not None:
            # MGE: modal graph embedding
            v_feats = self.mge('v')
            t_feats = self.mge('t')
            # local embeddings = collaborative-related embedding + modality-related embedding
            mge_embs = F.normalize(v_feats) + F.normalize(t_feats)
            lge_embs = cge_embs + mge_embs
            # GHE: global hypergraph embedding
            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:])
            av_hyper_embs = torch.concat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.concat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs
            # local embeddings + alpha * global embeddings
            all_embs = lge_embs + self.alpha * F.normalize(ghe_embs)
        else:
            all_embs = cge_embs

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)

        return u_embs, i_embs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs]
        
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss
    
    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss
    
    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, hyper_embeddings = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + self.ssl_triple_loss(iv_embs[pos_items], it_embs[pos_items], it_embs)
        
        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = batch_bpr_loss + self.cl_weight * batch_hcl_loss + self.reg_weight * batch_reg_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs, _ = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores

class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()

        self.h_layer = n_hyper_layer
    
    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret
