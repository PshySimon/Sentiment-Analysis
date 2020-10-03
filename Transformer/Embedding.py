"""
@Author:    Pshy Simon
@Date:  2020/10/1 0001 下午 07:22
@Description:
    词嵌入组件
"""

import torch
import torch.nn as nn
import math

class Embedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.input_size, config.d_model)
        self.scale = math.sqrt(config.d_model)

    def forward(self, x):
        return self.embedding(x) * self.scale
