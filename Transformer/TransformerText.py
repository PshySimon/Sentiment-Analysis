"""
@Author:    Pshy Simon
@Date:  2020/10/1 0001 下午 08:55
@Description:
   Transformer文本分类
"""

import torch
import torch.nn as nn
from .Embedding import Embedding
from .PositionalEmbedding import PositionalEmbedding
from .Encoder import Encoder

class TransformerText(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.position_embedding = PositionalEmbedding(config)
        self.encoder = Encoder(config)
        self.fc = nn.Linear(config.d_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        out = self.encoder(out)

        features = out[:, -1, :]
        return self.fc(features)


