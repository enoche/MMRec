# coding: utf-8
# @email: jiuqiangli@outlook.com
r"""
MSCA
################################################
Reference:
    https://github.com/recomall/MSCA
    WWW'2026: [Multi-view Semantic Contrastive Alignment for Multimodal Recommendation]
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph


class MSCA(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MSCA, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.is_sparse = config['is_sparse']
        self.fusion_coeff = config['fusion_coeff']
        self.tau = config['tau']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # initialize user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.is_sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.is_sparse))

        # construct normalized interaction matrix and adjacency matrix
        self.norm_R, self.norm_adj = self.get_norm_adj_mat()
        # construct normalized item-item structural matrix
        self.struct_original_adj = self.get_struct_adj_mat()

        # construct normalized modality-specific item-item matrix
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.is_sparse, norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.is_sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        # define transformation layers
        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.restore_user_e, self.restore_item_e = None, None

    def get_norm_adj_mat(self):
        R = self.interaction_matrix.tocsr()
        user_degrees = np.array(R.sum(axis=1)).flatten()
        item_degrees = np.array(R.sum(axis=0)).flatten()
        d_user = np.power(user_degrees, -0.5)
        d_user[np.isinf(d_user)] = 0.
        d_item = np.power(item_degrees, -0.5)
        d_item[np.isinf(d_item)] = 0.
        D_user = sp.diags(d_user, shape=(self.n_users, self.n_users))
        D_item = sp.diags(d_item, shape=(self.n_items, self.n_items))
        norm_R = D_user.dot(R).dot(D_item).tocoo()
        top_mat = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), norm_R])
        bottom_mat = sp.hstack([norm_R.T, sp.csr_matrix((self.n_items, self.n_items))])
        norm_adj_mat = sp.vstack([top_mat, bottom_mat]).tocoo()
        norm_R = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([norm_R.row, norm_R.col])),
            torch.FloatTensor(norm_R.data),
            torch.Size(norm_R.shape)
        ).float().to(self.device)
        norm_adj_mat = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([norm_adj_mat.row, norm_adj_mat.col])),
            torch.FloatTensor(norm_adj_mat.data),
            torch.Size(norm_adj_mat.shape)
        ).float().to(self.device)
        return norm_R, norm_adj_mat
    
    def get_struct_adj_mat(self):
        inter_matrix = self.interaction_matrix.tocsr()
        # construct item-item co-occurrence relationships
        item_cooccurrence = (inter_matrix.T @ inter_matrix).tocoo()
        row, col, data = item_cooccurrence.row, item_cooccurrence.col, item_cooccurrence.data
        mask = (row != col) & (data > 1)
        item_cooccurrence = sp.coo_matrix((data[mask], (row[mask], col[mask])), shape=item_cooccurrence.shape)
        item_num = np.bincount(item_cooccurrence.row, minlength=self.n_items).astype(np.int32)
        item_struct_matrix = torch.from_numpy(item_cooccurrence.toarray())
        item_row, item_col = [], []
        for i in range(self.n_items):
            item_row.append(i)
            item_col.append(i)
            k = min(int(item_num[i]), self.knn_k)
            if k > 0:
                _, indices = torch.topk(item_struct_matrix[i], k)
                item_row.extend([i] * k)
                item_col.extend(indices.tolist())
        ii_indices = torch.stack([torch.tensor(item_row, dtype=torch.long), torch.tensor(item_col, dtype=torch.long)], dim=0).to(self.device)
        struct_original_adj = self.compute_normalized_laplacian(indices=ii_indices, adj_size=(self.n_items, self.n_items))
        return struct_original_adj

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def semantic_encode(self, view_ii_adj, initial_item_embeds):
        view_item_embeds = torch.sparse.mm(view_ii_adj, initial_item_embeds)
        view_user_embeds = torch.sparse.mm(self.norm_R, view_item_embeds)
        view_embeds = torch.cat([view_user_embeds, view_item_embeds], dim=0)
        return view_embeds

    def forward(self, test=True):
        if self.v_feat is not None:
            image_item_embeds = self.item_id_embedding.weight * self.gate_v(self.image_trs(self.image_embedding.weight))
        if self.t_feat is not None:
            text_item_embeds = self.item_id_embedding.weight * self.gate_t(self.text_trs(self.text_embedding.weight))

        # user-item collaborative view
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = []
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        collab_embeds = torch.mean(all_embeddings, dim=1, keepdim=False)

        # item-item structural view
        struct_embeds = self.semantic_encode(self.struct_original_adj, self.item_id_embedding.weight)

        # item-item intra-modal view
        image_embeds = self.semantic_encode(self.image_original_adj, image_item_embeds)        
        text_embeds = self.semantic_encode(self.text_original_adj, text_item_embeds)

        # multi-view adaptive fusion
        multi_embed_list = [image_embeds, text_embeds, struct_embeds]
        attn_weights = self.softmax(torch.cat([self.query_common(embed) for embed in multi_embed_list], dim=-1))
        redundant_embeds = sum(w.unsqueeze(1) * embed for w, embed in zip(attn_weights.unbind(dim=1), multi_embed_list))
        multi_embeds = sum(multi_embed_list) - redundant_embeds

        final_embeds = collab_embeds + self.fusion_coeff * multi_embeds
        final_user_embeds, final_item_embeds = torch.split(final_embeds, [self.n_users, self.n_items], dim=0)

        if test:
            return final_user_embeds, final_item_embeds
        return final_user_embeds, final_item_embeds, collab_embeds, struct_embeds, image_embeds, text_embeds

    def cal_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)        
        return mf_loss

    def cal_cl_loss(self, embed_1, embed_2, tau):
        embed_1, embed_2 = F.normalize(embed_1, p=2, dim=1), F.normalize(embed_2, p=2, dim=1)
        pos = torch.sum(embed_1 * embed_2, dim=-1)
        tot = torch.matmul(embed_1, torch.transpose(embed_2, 0, 1))
        gcl_logits = tot - pos[:, None]
        # InfoNCE loss
        clogits = torch.logsumexp(gcl_logits / tau, dim=1)
        infonce_loss = torch.mean(clogits)
        return infonce_loss

    def cal_reg_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += param.norm(2).square()
        return reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        final_user_embeds, final_item_embeds, collab_embeds, struct_embeds, image_embeds, text_embeds = self.forward(test=False)

        batch_user_embeds = final_user_embeds[users]
        batch_pos_item_embeds = final_item_embeds[pos_items]
        batch_neg_item_embeds = final_item_embeds[neg_items]
        bpr_loss = self.cal_bpr_loss(batch_user_embeds, batch_pos_item_embeds, batch_neg_item_embeds)
        reg_loss = self.cal_reg_loss()

        collab_user_embeds, collab_item_embeds = torch.split(collab_embeds, [self.n_users, self.n_items], dim=0)
        struct_user_embeds, struct_item_embeds = torch.split(struct_embeds, [self.n_users, self.n_items], dim=0)
        image_user_embeds, image_item_embeds = torch.split(image_embeds, [self.n_users, self.n_items], dim=0)
        text_user_embeds, text_item_embeds = torch.split(text_embeds, [self.n_users, self.n_items], dim=0)

        # modality-aware semantic contrastive alignment        
        M_u_loss = self.cal_cl_loss(collab_user_embeds[users], image_user_embeds[users], self.tau) + self.cal_cl_loss(collab_user_embeds[users], text_user_embeds[users], self.tau)
        M_i_loss = self.cal_cl_loss(collab_item_embeds[pos_items], image_item_embeds[pos_items], self.tau) + self.cal_cl_loss(collab_item_embeds[pos_items], text_item_embeds[pos_items], self.tau)

        # collaborative semantic contrastive alignment        
        C_u_loss = self.cal_cl_loss(collab_user_embeds[users], struct_user_embeds[users], self.tau)
        C_i_loss = self.cal_cl_loss(collab_item_embeds[pos_items], struct_item_embeds[pos_items], self.tau)

        loss = bpr_loss + self.cl_weight * (C_u_loss + C_i_loss + M_u_loss + M_i_loss) + self.reg_weight * reg_loss

        return loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        self.restore_user_e, self.restore_item_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
        return scores