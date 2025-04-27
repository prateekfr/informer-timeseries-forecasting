# model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class ProbSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, n_heads=8, d_model=512):
        super().__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.out_projection = nn.Linear(d_model, d_model)


        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)
        M_top = M.topk(n_top, sorted=False)[1]

        return M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            context = V.mean(dim=-2).unsqueeze(-2).expand(B, H, L_Q, D)
        else:
            context = torch.zeros(B, H, L_Q, D, device=V.device)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q):
        if scores is not None:
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, V)
        else:
            context = context_in

        return context


    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        B, L_K, _ = keys.shape

        # Project inputs
        queries = self.query_projection(queries).view(B, L_Q, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        keys = self.key_projection(keys).view(B, L_K, self.n_heads, self.d_k).permute(0, 2, 1, 3)
        values = self.value_projection(values).view(B, L_K, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        # queries, keys, values are now (B, Heads, Seq_len, Dim_per_head)

        # ProbSparse Attention
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        scores_top = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        context = self._get_initial_context(values, L_Q)

        B, H, L_Q, D = queries.shape
        _, _, L_K, _ = keys.shape

        queries = queries.permute(0, 2, 1, 3).contiguous().view(B * L_Q, H, D)
        keys = keys.permute(0, 2, 1, 3).contiguous().view(B * L_K, H, D)
        values = values.permute(0, 2, 1, 3).contiguous().view(B * L_K, H, D)

        

        if self.mask_flag and attn_mask is not None:
            attn_mask = attn_mask.bool()

        context = self._update_context(context, values, None, None, L_Q)

        # Merge heads back
        context = context.reshape(B, L_Q, -1)

        context = self.out_projection(context)  # (batch, seq_len, d_model)


        return context.contiguous(), None
