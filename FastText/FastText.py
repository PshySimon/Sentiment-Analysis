"""
fastText巧妙之处在于，它引入了n元语法，同时对n元语法和词向量进行处理
"""
import torch
import torch.nn as nn


class FastText(nn.Module):
    def __init__(self, input_size, emb_dim, max_ngram, hidden_size, num_classes, dropout):
        super().__init__()
        # 词嵌入向量
        self.embedding = nn.Embedding(input_size, emb_dim)
        # 一元语法向量
        self.embedding_bigram = nn.Embedding(max_ngram, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, words, bigram):
        out = self.embedding(words)
        out_bigram = self.embedding_bigram(bigram)

        out = torch.cat((out, out_bigram), dim=-1)

        out = out.mean(dim=1)
        out = self.dropout(out)

        out = self.fc1(out)

        out = nn.functional.relu(out)

        out = self.fc2(out)
        return out





