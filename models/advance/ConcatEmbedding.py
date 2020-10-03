"""
将不同粒度的词进行连接
"""
import torch
import torch.nn as nn
from .HighWay import HighWay

class ConcatEmbedding(nn.Module):

    def __init__(self, highway_num_layers, char_emb_dim, word_emb_dim):
        super().__init__()
        self.highway = HighWay(highway_num_layers, char_emb_dim + word_emb_dim)

    def forward(self, char_emb, word_emb):
        char_emb,_ = torch.max(char_emb, dim=2)
        emb = torch.cat((char_emb, word_emb), dim=2)
        emb = self.highway(emb)
        return emb

