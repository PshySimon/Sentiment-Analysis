"""
HighWay的实现
公式为：
        Gate(X) = sigmoid(WX + b_W)
        NonLinear(X) = relu(UX + b_U)
        Out = Gate(X) * NonLinear(X) + (1 - Gate(X)) * NonLinear(X)
"""
import torch
import torch.nn as nn

class HighWay(nn.Module):

    def __init__(self, num_layer, emb_dim):
        super().__init__()
        self.num_layer = num_layer
        self.gate = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(self.num_layer)])
        self.linear = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(self.num_layer)])

    def forward(self, X):
        # 输入的是concat了的词向量和字向量，形状为：[batch_size, seq_len, char_emb + word_emb]
        for i in range(self.num_layer):
            Gate = torch.sigmoid(self.gate[i](X))
            NonLinear = torch.relu(self.linear[i](X))
            X = Gate * NonLinear + (1 - Gate) *X
        return X

