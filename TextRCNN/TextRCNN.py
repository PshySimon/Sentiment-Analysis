"""
TextRCNN的实现
"""
import torch
import torch.nn as nn

class TextRCNN_Model(nn.Module):

    def __init__(self, input_size, emb_dim, hidden_size, num_classes, dropout, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, bidirectional=True, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.w = nn.Parameter(torch.randn(2*hidden_size + emb_dim, 2 * hidden_size))

    def forward(self, X):
        # X = [seq_len, batch_size]
        embedded = self.embedding(X)
        # embedded = [seq_len, batch_size, hidden_size]
        out,_ = self.rnn(embedded)
        # 将输出和嵌入层连接起来
        # out = [seq_len, batch_size, hidden_size * 2]
        out = out.permute(1,0,2)
        embedded = embedded.permute(1,0,2)
        out = torch.cat((out, embedded), dim=2)
        # out = [batch_size, seq_len, hidden_size * 2 + emb_dim
        out = torch.tanh(torch.matmul(out, self.w))
        # out = [batch_size, seq_len, hidden_size * 2
        out = out.permute(0,2,1)
        # out = [batch_size, hidden_size * 2, seq_len]
        out = nn.functional.max_pool1d(out, out.shape[-1]).squeeze(2)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    X = torch.randint(10,(10,128))
    INPUT_SIZE, EMB_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT, NUM_LAYERS = \
        200, 200, 256, 4, 0.5, 2
    model = TextRCNN_Model(INPUT_SIZE, EMB_DIM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT, NUM_LAYERS)
    y = model(X)
    print(y.shape)


