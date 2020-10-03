"""
基于Attention的文本分类
"""
import torch
import torch.nn as nn

class TextRNN_Att_Model(nn.Module):
    def __init__(self, input_size, emb_dim, hidden_size1, hidden_size2, dropout, bidirectional, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size1, bidirectional=bidirectional, num_layers = num_layers
                           ,batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.w = nn.Parameter(torch.randn(2 * hidden_size1))
        self.fc1 = nn.Linear(hidden_size1*2, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, num_classes)

    #
    def forward(self, X):
        # X = [seq_len, batch_size]
        X = X.permute(1,0)
        # X = [batch_size, seq_len]
        embedded = self.dropout(self.embedding(X))
        # embedded = [batch_size, seq_len, emb_dim]
        out,(H,C) = self.rnn(embedded)
        # out = [batch_size, seq_len, hidden_size * num_directions]
        # H = [num_layers * num_direction, batch_size, hidden_size]
        # C = [num_layers * num_direction, batch_size, hidden_size]
        # 开始计算Attention，淦！原论文直接给了公式不加解释！
        # 先给他套个激活函数，然后乘上query向量，归一化之后就能拿到attention
        # 然后乘上原信息，进行加权平均
        M = self.tanh1(out)
        score = torch.matmul(out, self.w)
        att = torch.softmax(score, dim=1)
        # att = [batch_size, seq_len]
        out = out * att
        # out = [batch_size, seq_len, hidden_size * 2]
        out = torch.sum(out, 1)
        # out = [batch_size, hidden_size * 2]
        out = torch.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out




