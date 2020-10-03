from TextCNN import TextCNN_Model
import torch
import torch.nn as nn
from utils.utils import DataIter
from sklearn import metrics

iterator = DataIter()
train_iter, valid_iter, test_iter = iterator.fetch_iter()
TEXT = iterator.TEXT
LABEL = iterator.LABEL

FILTER_SIZES, NUM_CLASSES, VOCAB_SIZE, NUM_FILTERS, EMB_DIM, DROPOUT = \
    [2, 3, 4], len(LABEL.vocab), len(TEXT.vocab), 128, 256, 0.5
model = TextCNN_Model(FILTER_SIZES, NUM_CLASSES, VOCAB_SIZE, NUM_FILTERS, EMB_DIM, DROPOUT)
model.load_state_dict(torch.load("model.pkt"))
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
print(len(train_iter), len(valid_iter), len(test_iter))

with torch.no_grad():
    total_loss, batch, accuracy = 0., 0, 0.
    for (X, length),y in test_iter:
        batch += 1
        model.eval()
        model.zero_grad()
        y_hat = model(X.permute(1, 0))
        l = loss(y_hat, y)
        total_loss = l.item()
        accuracy = metrics.accuracy_score(torch.argmax(y_hat, dim=1), y)
    print("\r TEST loss:%f, acc:%f" % (total_loss, accuracy), end="")

