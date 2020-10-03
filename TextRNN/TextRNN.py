"""
TextRNN就比较容易了，我们将RNN最后的隐藏状态作为Contextualized
"""
import torch
import torch.nn as nn
from sklearn import metrics

class TextRNN_Model(nn.Module):

    def __init__(self, input_size, emb_dim, hidden_size, output_size, num_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*(2 if bidirectional else 1), output_size)

    def forward(self, x):
        # x = [seq_len, batch_size]
        out = self.dropout(self.embedding(x))
        # out = [seq_len, batch_size, emb_dim]
        out,_ = self.rnn(out)
        # out = [seq_len, batch_size, num_directions * hidden_size]
        # out是整句话每个单词输出的hidden state，取最后一个单词的时候所得到的输出
        out = self.fc(out[-1,:,:])
        return out

if __name__ == '__main__':

    from utils.utils import DataIter
    iterator = DataIter()
    train_iter, valid_iter, test_iter = iterator.fetch_iter()
    TEXT = iterator.TEXT
    LABEL = iterator.LABEL

    NUM_EPOCHS = 10
    INPUT_SIZE, EMB_DIM, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT = \
        len(TEXT.vocab), 200, 256, len(LABEL.vocab), 2, True, 0.5
    model = TextRNN_Model(INPUT_SIZE, EMB_DIM, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        total_loss, n, accuracy = 0., 0, 0.
        for (X, length),y in train_iter:
            model.zero_grad()
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optim.step()
            n += y.shape[0]
            total_loss = l.item()
            accuracy = metrics.accuracy_score(torch.argmax(y_hat, dim=1), y)
            print("\r TRAIN epoch:%d, iter:%d, loss:%f, acc:%f" % (epoch, n, total_loss, accuracy), end="")
        print()
        for (X, length),y in valid_iter:
            with torch.no_grad():
                y_hat = model(X)
                l = loss(y_hat, y)
                n = y.shape[0]
                total_loss = l.item()
                accuracy = metrics.accuracy_score(torch.argmax(y_hat, dim=1), y)
                if total_loss < best_loss:
                    torch.save(model.state_dict(), "model.pkt")
                    best_loss = total_loss
            print(" VALID epoch:%d, iter:%d, loss:%f, acc:%f" % (epoch, n, total_loss, accuracy), end="")
        print()

