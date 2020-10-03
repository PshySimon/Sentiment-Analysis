"""
@Author:    Pshy Simon
@Date:  2020/10/1 0001 下午 08:51
@Description:
   编码器层
"""

import torch
import torch.nn as nn
from .EncoderLayer import EncoderLayer
import copy

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder_layer = EncoderLayer(config)
        self.layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.size)

    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)



