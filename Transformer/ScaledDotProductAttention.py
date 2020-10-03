"""
@Author:    Pshy Simon
@Date:  2020/10/1 0001 下午 07:36
@Description:
   放缩注意力
"""

import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, mask=None):
        d_model = Q.shape[-1]
        scores = torch.matmul(Q, K.permute(0,1,3,2)) / math.sqrt(d_model)
        # scores = [batch_size, query_len, key_len]
        if mask is not None:
            score = scores.masked_fill(mask == 0, 1e-9)
        attention = torch.nn.functional.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        # out = [batch_size, query_len, d_model]

        return out


