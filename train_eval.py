"""
训练和测试
"""
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='文本分类')
parser.add_argument("--model", type=str, required=True, help="可选择的模型目前有：TextCNN、TextRNN、TextRCNN、TextRNNAtt")
parser.add_argument("--epoch", type=int, default=10, help="训练的周期，默认为10")

args = parser.parse_args()
# 边测试边验证，并保存验证集的损失，如果碰到损失较小的模型就保存起来
def train(config, model, criterion, optimizer, train_iter, dev_iter, test_iter):
    model.train()
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    best_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        total_loss, step, accuracy = 0., 0, 0.
        msg = "\n TRAIN epoch:{}/{}, train_loss:{}, train_accuracy:{},dev_loss:{},dev_accuracy:{}"
        pbar = tqdm(train_iter, ncols=100)
        for X, y in pbar:
            model.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            step += 1
            total_loss += loss.item()
            labels = y.detach().cpu().numpy()
            preds = np.argmax(y_hat.detach().cpu().numpy(), axis=1)
            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

        accuracy = metrics.accuracy_score(all_preds, all_labels)
        dev_loss, dev_accuracy = evaluation(config, model, DEV_ITER)
        print(msg.format(epoch, NUM_EPOCHS, total_loss / len(train_iter), accuracy, dev_loss, dev_accuracy))

        if dev_loss < best_loss:
            print("Best model so far!")
            best_loss = dev_loss
            torch.save(model.state_dict(), "model.pkl")

        total_loss = 0.
        model.train()

def train_highway(config, model, criterion, optimizer, train_iter, dev_iter, test_iter):
    model.train()
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    best_loss = float("inf")


    msg = "TRAIN epoch:{}/{}, train_loss:{}, train_accuracy:{},dev_loss:{},dev_accuracy:{}"
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_iter, ncols=50)
        total_loss, step, accuracy = 0., 0, 0.
        for (X, (x, length)), y in pbar:
            model.zero_grad()
            y_hat = model(x, X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            step += 1
            total_loss += loss.item()
            labels = y.detach().cpu().numpy()
            preds = np.argmax(y_hat.detach().cpu().numpy(), axis=1)
            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

        accuracy = metrics.accuracy_score(all_preds, all_labels)

        dev_loss, dev_accuracy = evaluation(config, model, DEV_ITER, field = "char")
        print(msg.format(epoch, NUM_EPOCHS, total_loss / len(train_iter), accuracy, dev_loss, dev_accuracy))

        if dev_loss < best_loss:
            print("Best model so far!")
            best_loss = dev_loss
            torch.save(model.state_dict(), "model.pkl")

        total_loss = 0.
        model.train()

def train_fasttext(config, model, criterion, optimizer, train_iter, dev_iter, test_iter):
    model.train()
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    best_loss = float("inf")

    msg = "TRAIN epoch:{}/{}, train_loss:{}, train_accuracy:{},dev_loss:{},dev_accuracy:{}"
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_iter, ncols=50)
        total_loss, step, accuracy = 0., 0, 0.
        for (w, b),y in pbar:
            model.zero_grad()
            y_hat = model(w, b)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            step += 1
            total_loss += loss.item()
            labels = y.detach().cpu().numpy()
            preds = np.argmax(y_hat.detach().cpu().numpy(), axis=1)
            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

        accuracy = metrics.accuracy_score(all_preds, all_labels)

        dev_loss, dev_accuracy = evaluation(config, model, DEV_ITER, field = "bigram")
        print(msg.format(epoch, NUM_EPOCHS, total_loss / len(train_iter), accuracy, dev_loss, dev_accuracy))

        if dev_loss < best_loss:
            print("Best model so far!")
            best_loss = dev_loss
            torch.save(model.state_dict(), "model.pkl")

        total_loss = 0.
        model.train()

def evaluation(config, model, test_iter, field = "word"):
    model.eval()
    loss, accuracy = 0., 0.
    if field == "word":
        with torch.no_grad():
            for X,y in test_iter:
                out = model(X)
                loss = nn.functional.cross_entropy(out, y)
                accuracy = metrics.accuracy_score(torch.argmax(out, dim=1), y)
    elif field == "char":
        with torch.no_grad():
            for (X, (x, length)), y in test_iter:
                out = model(x, X)
                loss = nn.functional.cross_entropy(out, y)
                accuracy = metrics.accuracy_score(torch.argmax(out, dim=1), y)
    elif field == "bigram":
        with torch.no_grad():
            for (w, b), y in test_iter:
                out = model(w, b)
                loss = nn.functional.cross_entropy(out, y)
                accuracy = metrics.accuracy_score(torch.argmax(out, dim=1), y)
    return loss, accuracy

def test(model, test_iter, field = "word"):
    print(model)
    model.load_state_dict(torch.load("model.pkl"))
    return evaluation(None, model, test_iter, field = field)


if __name__ == '__main__':
    from utils.utils import DataIter



    if args.model == "TextCNN":
        iterator = DataIter()
        TRAIN_ITER, DEV_ITER, TEST_ITER = iterator.fetch_iter()
        TEXT = iterator.config.TEXT
        LABEL = iterator.config.LABEL
        from TextCNN.TextCNN import TextCNN_Model
        NUM_EPOCHS = args.epoch
        FILTER_SIZES, NUM_CLASSES, VOCAB_SIZE, NUM_FILTERS, EMB_DIM, DROPOUT = \
            [2, 3, 4], len(LABEL.vocab), len(TEXT.vocab), 128, 256, 0.5
        MODEL = TextCNN_Model(FILTER_SIZES, NUM_CLASSES, VOCAB_SIZE, NUM_FILTERS, EMB_DIM, DROPOUT)
        CRITERION = nn.CrossEntropyLoss()
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
        train(args, MODEL, CRITERION, OPTIMIZER, TRAIN_ITER, DEV_ITER, TEST_ITER)
        test(MODEL, TEST_ITER)
    elif args.model == "TextCNNHighWay":
        iterator = DataIter(field="char", batch_size=64)
        TRAIN_ITER, DEV_ITER, TEST_ITER = iterator.fetch_iter()
        TEXT = iterator.config.TEXT
        CHAR = iterator.config.char_field
        LABEL = iterator.config.LABEL

        from TextCNNHighWay.TextCNNHighWay import TextCNNHighWay
        NUM_EPOCHS = args.epoch
        CHAR_VOCAB_SIZE, WORD_VOCAB_SIZE, CHAR_EMB_DIM, WORD_EMB_DIM, \
            HIGHWAY_NUM_LAYERS, NUM_FILTERS, FILTER_SIZES, HIDDEN_SIZE, NUM_CLASSES, DROPOUT = \
            len(CHAR.vocab), len(TEXT.vocab), 300, 300, 2, 200, [2, 3, 4], 256, len(LABEL.vocab), 0.4
        MODEL = TextCNNHighWay(CHAR_VOCAB_SIZE, WORD_VOCAB_SIZE, CHAR_EMB_DIM, WORD_EMB_DIM,
                               HIGHWAY_NUM_LAYERS, NUM_FILTERS, FILTER_SIZES, HIDDEN_SIZE, NUM_CLASSES, DROPOUT)
        CRITERION = nn.CrossEntropyLoss()
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
        train_highway(args, MODEL, CRITERION, OPTIMIZER, TRAIN_ITER, DEV_ITER, TEST_ITER)
        loss, accuracy = test(MODEL, TEST_ITER, field = "char")
        print("TEST, loss:%f,accuracy:%f" % (loss, accuracy))
    elif args.model == "Transformer":
        iterator = DataIter(batch_size=64)
        TRAIN_ITER, DEV_ITER, TEST_ITER = iterator.fetch_iter()
        TEXT = iterator.config.TEXT
        LABEL = iterator.config.LABEL

        class Config:
            def __init__(self):
                self.input_size = len(TEXT.vocab)
                self.d_model = 512
                self.n_heads = 8
                self.n_layers = 6
                self.dropout = 0.4
                self.max_length = 5000
                self.d_ff = 512
                self.size = self.d_model
                self.num_classes = len(LABEL.vocab)

        NUM_EPOCHS = args.epoch
        config = Config()
        from Transformer.TransformerText import TransformerText
        MODEL = TransformerText(config)
        CRITERION = nn.CrossEntropyLoss()
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
        train(args, MODEL, CRITERION, OPTIMIZER, TRAIN_ITER, DEV_ITER, TEST_ITER)
        test(MODEL, TEST_ITER)
    elif args.model == "FastText":
        iterator = DataIter(field="bigram")
        TRAIN_ITER, DEV_ITER, TEST_ITER = iterator.fetch_iter()
        TEXT = iterator.config.TEXT
        BIGRAM = iterator.config.bigram
        LABEL = iterator.config.LABEL

        INPUT_SIZE, EMB_DIM, MAX_NGRAM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT =\
            len(TEXT.vocab), 300, len(BIGRAM.vocab), 256, len(LABEL.vocab), 0.2

        from FastText.FastText import FastText
        MODEL = FastText(INPUT_SIZE, EMB_DIM, MAX_NGRAM, HIDDEN_SIZE, NUM_CLASSES, DROPOUT)
        NUM_EPOCHS = args.epoch
        CRITERION = nn.CrossEntropyLoss()
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-3)
        train_fasttext(args, MODEL, CRITERION, OPTIMIZER, TRAIN_ITER, DEV_ITER, TEST_ITER)
        loss, accuracy = test(MODEL, TEST_ITER, field="bigram")
        print("TEST, loss:%f,accuracy:%f" % (loss, accuracy))





