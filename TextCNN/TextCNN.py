"""
利用卷积神经网络做文本分类
"""
import torch
import torch.nn as nn
from sklearn import metrics

# 一维卷积输入的意义[batch_size, seq_len, emb_dim]
# 经过permute之后，在声明一维卷积
# 一维卷积的参数：in_channels = emb_dim
# out_channels就是一个filter提取多少特征
# kernel_size可以是int也可以是tuple，窗口大小为：[kernel_size, emb_dim]
# batch_size=10, seq_len=15, emb_dim=20

class TextCNN_Model(nn.Module):
    def __init__(self, filter_sizes, num_classes, vocab_size, num_filters, emb_dim, dropout):
        # 首先是嵌入层，这里试试可学习的参数
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # 然后是提取特征
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_filters,x) for x in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        # 最后都要经过池化层，使得输出为
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.relu = nn.ReLU()

    def pool(self, out, conv):
        out = self.relu(conv(out))
        max_pool = nn.MaxPool1d(out.shape[-1])
        out = max_pool(out)
        out = out.squeeze(2)
        return out

    def forward(self, x):
        # x = [batch_size, seq_len]
        embedded = self.embedding(x)
        # 注意这里输入到卷积层中时需要改变维度
        embedded = embedded.permute(0,2,1)
        # embedded = [batch_size, seq_len, emb_dim]
        output = [self.pool(embedded, conv) for conv in self.convs]
        # output = num_filter_sizes * [batch_size, num_filters]
        out = torch.cat(output, dim=1)
        # out = [batch_size, num_filter_sizes * num_filters]
        out = self.dropout(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":

    from utils.utils import DataIter
    iterator = DataIter()
    train_iter, valid_iter, test_iter = iterator.fetch_iter()
    TEXT = iterator.TEXT
    LABEL = iterator.LABEL

    NUM_EPOCHS = 10
    FILTER_SIZES, NUM_CLASSES, VOCAB_SIZE, NUM_FILTERS, EMB_DIM, DROPOUT = \
        [2, 3, 4], len(LABEL.vocab), len(TEXT.vocab), 128, 256, 0.5
    model = TextCNN_Model(FILTER_SIZES, NUM_CLASSES, VOCAB_SIZE, NUM_FILTERS, EMB_DIM, DROPOUT)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        total_loss, n, accuracy = 0., 0, 0.
        for (X, length),y in train_iter:
            model.zero_grad()
            y_hat = model(X.permute(1,0))
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
                y_hat = model(X.permute(1,0))
                my_loss = loss(y_hat, y)
                n = y.shape[0]
                total_loss = my_loss.item()
                accuracy = metrics.accuracy_score(torch.argmax(y_hat, dim=1), y)
            # if total_loss < best_loss:
            #     torch.save(model.state_dict(), "model.pkt")
            #     best_loss = total_loss
            print(" VALID epoch:%d, iter:%d, loss:%f, acc:%f" % (epoch, n, total_loss, accuracy))



