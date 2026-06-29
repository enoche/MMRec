# coding: utf-8
"""
MMGCF: Multimodal Graph Collaborative Filtering
Late-fusion of LightGCN ID embeddings with multimodal item features.

Fusion modes  : mean | sum | concat
Weighting modes: equal | alpha | normalized

Adapted for the MMRec framework.
NOTE: For non-concat fusion modes, feat_embed_dim must equal embedding_size
      so that ID and modal embeddings can be stacked/added element-wise.
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class MMGCF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGCF, self).__init__(config, dataset)

        self.embedding_dim  = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_ui_layers    = config['n_ui_layers']
        self.reg_weight     = config['reg_weight']
        self.fusion_mode    = config['fusion_mode']   # mean | sum | concat
        self.weighting      = config['weighting']     # equal | alpha | normalized
        self.dropout        = config['dropout']

        self.n_nodes = self.n_users + self.n_items

        # ── interaction matrix & adjacency ────────────────────────────────
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj = None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values  = self.edge_values.to(self.device)

        # ── ID embeddings ─────────────────────────────────────────────────
        self.user_embedding    = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # ── multimodal feature embeddings & projection layers ─────────────
        # Raw pretrained features are stored in freezed Embeddings;
        # a learnable Linear layer projects them to feat_embed_dim on every forward pass.
        self.n_modalities = 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.n_modalities += 1
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.n_modalities += 1

        # ── weighting: learnable α scalar (only used when weighting='alpha') ──
        if self.weighting == 'alpha':
            # α = sigmoid(mm_alpha); item_emb weighted by α, mm feats by (1-α)
            self.mm_alpha = nn.Parameter(torch.tensor(0.0))

        # ── concat fusion layers ──────────────────────────────────────────
        # Three separate Linear layers are used so each concat step has its own
        # fixed input size, avoiding the LazyLinear shape-collision bug in the
        # original code when equal+concat are combined.
        if self.fusion_mode == 'concat' and self.n_modalities > 0:
            # Used by 'alpha' and 'normalized': concat [id_emb, mm1, ..., mmN]
            all_in = self.embedding_dim + self.n_modalities * self.feat_embed_dim
            self.all_concat_layer = nn.Linear(all_in, self.embedding_dim)

            # Used by 'equal' step-1: concat all modality feats (only when >1 mod)
            if self.n_modalities > 1:
                mm_in = self.n_modalities * self.feat_embed_dim
                self.mm_concat_layer = nn.Linear(mm_in, self.feat_embed_dim)

            # Used by 'equal' step-2: concat [id_emb, fused_mm]
            id_mm_in = self.embedding_dim + self.feat_embed_dim
            self.id_mm_concat_layer = nn.Linear(id_mm_in, self.embedding_dim)

    # ══════════════════════════════════════════════════════════════════════
    #  Graph helpers
    # ══════════════════════════════════════════════════════════════════════

    def get_norm_adj_mat(self):
        """Build symmetric normalised Laplacian for the user-item bipartite graph."""
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32,
        )
        inter_M   = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        # fill upper-right and lower-left blocks
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        # A._update(data_dict)
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # D^{-1/2} A D^{-1/2}
        sumArr = (A > 0).sum(axis=1)
        diag   = np.power(np.array(sumArr.flatten())[0] + 1e-7, -0.5)
        D = sp.diags(diag)
        L = sp.coo_matrix(D * A * D)
        i    = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def _normalize_adj_m(self, indices, adj_size):
        """Compute D_u^{-1/2} * D_i^{-1/2} edge weights for a bipartite sub-graph."""
        adj = torch.sparse.FloatTensor(
            indices, torch.ones_like(indices[0], dtype=torch.float32), adj_size
        )
        row_sum = 1e-7 + torch.sparse.sum(adj,   -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        return torch.pow(row_sum, -0.5)[indices[0]] * torch.pow(col_sum, -0.5)[indices[1]]

    def get_edge_info(self):
        rows  = torch.from_numpy(self.interaction_matrix.row)
        cols  = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        vals  = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, vals

    def pre_epoch_processing(self):
        """
        Degree-sensitive edge dropout (disabled when dropout <= 0).
        Edges with higher normalised weight are more likely to be kept,
        encouraging the model to retain informative interactions.
        """
        if self.dropout <= 0.0:
            self.masked_adj = self.norm_adj
            return
        degree_len  = int(self.edge_values.size(0) * (1.0 - self.dropout))
        degree_idx  = torch.multinomial(self.edge_values, degree_len)
        keep_idx    = self.edge_indices[:, degree_idx]
        keep_vals   = self._normalize_adj_m(keep_idx, torch.Size((self.n_users, self.n_items)))
        all_vals    = torch.cat((keep_vals, keep_vals))
        keep_idx[1] += self.n_users   # shift item indices into the joint space
        all_idx     = torch.cat((keep_idx, torch.flip(keep_idx, [0])), dim=1)
        self.masked_adj = torch.sparse.FloatTensor(
            all_idx, all_vals, self.norm_adj.shape
        ).to(self.device)

    # ══════════════════════════════════════════════════════════════════════
    #  LightGCN propagation
    # ══════════════════════════════════════════════════════════════════════

    def lightgcn_propagate(self, adj):
        """
        Standard LightGCN: layer-wise neighbourhood aggregation over the
        user-item graph, followed by mean-pooling across all layer outputs.
        Returns (user_emb, item_emb) each of shape (N, embedding_dim).
        """
        ego = torch.cat(
            [self.user_embedding.weight, self.item_id_embedding.weight], dim=0
        )
        layers = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(adj, ego)
            layers.append(ego)
        # mean of all layer embeddings (LightGCN aggregation)
        out = torch.stack(layers, dim=1).mean(dim=1)
        return torch.split(out, [self.n_users, self.n_items], dim=0)

    # ══════════════════════════════════════════════════════════════════════
    #  Multimodal fusion
    # ══════════════════════════════════════════════════════════════════════

    def _get_mm_feats(self):
        """Project raw pretrained features into feat_embed_dim."""
        feats = []
        if self.v_feat is not None:
            feats.append(self.image_trs(self.image_embedding.weight))
        if self.t_feat is not None:
            feats.append(self.text_trs(self.text_embedding.weight))
        return feats  # list of (n_items, feat_embed_dim) tensors

    def _apply_fusion(self, tensors, concat_layer=None):
        """
        Fuse a list of tensors according to self.fusion_mode.

        mean / sum  — element-wise ops; all tensors must share shape.
        concat      — cat along last dim, then project via concat_layer.
        """
        if self.fusion_mode == 'mean':
            return torch.stack(tensors).mean(dim=0)
        elif self.fusion_mode == 'sum':
            return torch.stack(tensors).sum(dim=0)
        else:  # concat
            return concat_layer(torch.cat(tensors, dim=-1))

    def fuse_item_embeddings(self, item_emb):
        """
        Late-fuse LightGCN item embeddings with multimodal features.

        Weighting strategies
        --------------------
        alpha      : learnable sigmoid gate; item_emb scaled by α,
                     each modal feat scaled by (1−α), then jointly fused.
        normalized : L2-normalise every embedding before fusing; the ID
                     embedding is additionally scaled by n_modalities so
                     all modalities contribute roughly equally.
        equal      : two-stage — first fuse modalities together, then
                     fuse the result with the ID embedding (equal weight).
        """
        mm_feats = self._get_mm_feats()
        if not mm_feats:
            return item_emb

        if self.weighting == 'alpha':
            alpha   = torch.sigmoid(self.mm_alpha)
            tensors = [item_emb * alpha] + [f * (1.0 - alpha) for f in mm_feats]
            return self._apply_fusion(
                tensors,
                concat_layer=self.all_concat_layer if self.fusion_mode == 'concat' else None,
            )

        elif self.weighting == 'normalized':
            tensors = (
                [F.normalize(item_emb) * self.n_modalities]
                + [F.normalize(f) for f in mm_feats]
            )
            return self._apply_fusion(
                tensors,
                concat_layer=self.all_concat_layer if self.fusion_mode == 'concat' else None,
            )

        else:  # equal
            # Step 1: reduce multiple modalities to a single mm embedding
            if len(mm_feats) > 1:
                mm_fused = self._apply_fusion(
                    mm_feats,
                    concat_layer=self.mm_concat_layer if self.fusion_mode == 'concat' else None,
                )
            else:
                mm_fused = mm_feats[0]   # single modality — no reduction needed

            # Step 2: combine ID embedding with the fused modal embedding
            return self._apply_fusion(
                [item_emb, mm_fused],
                concat_layer=self.id_mm_concat_layer if self.fusion_mode == 'concat' else None,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  Forward pass
    # ══════════════════════════════════════════════════════════════════════

    def forward(self, adj):
        user_emb, item_emb = self.lightgcn_propagate(adj)
        item_emb = self.fuse_item_embeddings(item_emb)
        return user_emb, item_emb

    # ══════════════════════════════════════════════════════════════════════
    #  Loss & prediction
    # ══════════════════════════════════════════════════════════════════════

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = (users * pos_items).sum(dim=1)
        neg_scores = (users * neg_items).sum(dim=1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]

        ua_emb, ia_emb = self.forward(self.masked_adj)
        mf_loss = self.bpr_loss(ua_emb[users], ia_emb[pos_items], ia_emb[neg_items])

        # L2 regularisation on the initial (un-propagated) ID embeddings only,
        # consistent with the LightGCN / BPR-MF convention.
        reg_loss = (
            self.user_embedding.weight[users].norm(2).pow(2)
            + self.item_id_embedding.weight[pos_items].norm(2).pow(2)
            + self.item_id_embedding.weight[neg_items].norm(2).pow(2)
        ) / (2 * len(users))

        return mf_loss + self.reg_weight * reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward(self.norm_adj)
        return torch.matmul(user_emb[user], item_emb.t())