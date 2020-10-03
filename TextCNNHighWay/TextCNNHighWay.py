"""
试试加了HighWay的CNN
"""
import torch
import torch.nn as nn
from models.advance.ConcatEmbedding import ConcatEmbedding


class TextCNNHighWay(nn.Module):

    def __init__(self, char_vocab_size, word_vocab_size, char_emb_dim, word_emb_dim,
                 highway_num_layers, num_filters, filter_sizes, hidden_size, num_classes, dropout):
        super().__init__()
        print("字符级别嵌入层大小为%dx%d" % (char_vocab_size, char_emb_dim))
        print("单词级别嵌入层大小为%dx%d" % (word_vocab_size, word_emb_dim))
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim)
        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim)
        self.emb = ConcatEmbedding(highway_num_layers, char_emb_dim, word_emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(char_emb_dim + word_emb_dim, num_filters, k) for k in filter_sizes])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.dropout = nn.Dropout(dropout)

    def max_pool(self, conv, out):
        out = torch.relu(conv(out))
        pool = nn.MaxPool1d(out.shape[-1])
        out = pool(out)
        out = out.squeeze(2)
        return out

    def forward(self, words, chars):
        # word_embedding = [batch_size, seq_len, word_emb]
        # char_embedding = [batch_size, seq_len, word_len, char_emb]
        # print("1.",words.shape, chars.shape)
        word_embedding = self.word_emb(words)
        char_embedding = self.char_emb(chars)
        # print("2.",word_embedding.shape, char_embedding.shape)
        # char_embedding = [seq_len, batch_size, word_len, char_emb]
        # char_embedding = char_embedding.permute(1, 0, 2, 3)
        # text_emb = [batch_size, seq_len, word_emb + char_emb]
        text_emb = self.emb(char_embedding, word_embedding)
        # print("3.",text_emb.shape)
        text_emb = text_emb.permute(0,2,1)
        # print("4.",text_emb.shape)
        # 放到CNN中
        # 一定要注意CNN的输入是：[batch_size, emb_dim, seq_len]，这样便于过滤器进行过滤，淦！
        out = [self.max_pool(conv_, text_emb) for conv_ in self.convs]
        out = torch.cat(out, dim=1)
        # print("5.",out.shape)
        out = self.dropout(out)
        # print("6.", out.shape)
        # 线性映射
        out = self.fc(out)
        # print("7.", out.shape)
        return out


if __name__ == '__main__':
    CHAR_VOCAB_SIZE, WORD_VOCAB_SIZE, CHAR_EMB_DIM, WORD_EMB_DIM, \
    HIGHWAY_NUM_LAYERS, NUM_FILTERS, FILTER_SIZES, HIDDEN_SIZE, NUM_CLASSES, DROPOUT = \
        94, 1000, 300, 300, 2, 4, [2, 3, 4], 256, 2, 0.5
    BATCH_SIZE, SEQ_LEN, WORD_LEN = 128, 10, 8
    word_idx = torch.randint(1000, (SEQ_LEN, BATCH_SIZE))
    char_idx = torch.randint(94, (BATCH_SIZE, SEQ_LEN, WORD_LEN))

    model = TextCNNHighWay(CHAR_VOCAB_SIZE, WORD_VOCAB_SIZE, CHAR_EMB_DIM, WORD_EMB_DIM,
        HIGHWAY_NUM_LAYERS, NUM_FILTERS, FILTER_SIZES, HIDDEN_SIZE, NUM_CLASSES, DROPOUT)
    model(word_idx, char_idx)










