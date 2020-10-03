"""
利用传统的方法进行分类，看看效果
1.先提取字典，统计各个字在文本中的出现频率
2.利用卡方检验来获得可以使用的特征
3.利用抽取出来的特征进行分类
"""
import pandas as pd

# 对文本进行统计
def statistic(train_data):
    classes = {}
    freqs = {}
    for i, row in train_data.iterrows():
        sentence, label = row.values
        classes[label] = classes.get(label, 0) + 1
        for w in sentence.split():
            freqs.get(label, set()).add(w)
    return classes,freqs


# 卡方检验某个单词是否可以当做特征
def chi_check(word, label, classes, freqs):
    # 需要统计的量：
    # N1:含有该单词的文档数量
    # N0:不含有该单词的文档数量
    # N11:单词x出现在y类文档的数量
    # N10:单词x出现在除了y类文档以外的数量
    # N01:除了单词x以外其他单词出现在y类文档的数量
    # N00：单词x以外的词不在y类文档中出现的次数
    chi, N1, N0, N11, N10 = [], 0, 0, 0, 0
    for _,x in freqs.items():
        if word in x:
            N1 += 1
            if label == _:
                N11 += 1
            else:
                N10 += 1
        else:
            N0 += 1

    N01 = classes[label] - N1
    N00 = N0 - N01
    print(N0,N1,N10,N11,N01,N00)
    if (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00) == 0:
        return False
    value = ((N11 + N10 + N01 + N00) * (N11 * N00 - N10 * N01) ** 2) / (
                (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00) + 1)
    chi.append(value)
    return True if max(chi) < 10.83 else False

# 经过卡方选择得到的特征
def feature_select():
    train = pd.read_csv("../data/train.tsv", delimiter="\t")
    classes, freqs = statistic(train)
    from utils.utils import DataIter
    iterator = DataIter()
    iterator.fetch_iter()
    vocab = iterator.TEXT.vocab.itos
    labels = [x for x,y in classes.items()]
    features = set()
    for w in vocab:
        for l in labels:
            print("\r 正在校验特征:%s,标签:%d" % (w, l))
            if chi_check(w, l, classes,freqs):
                features.add(w)
    return features

# feature_select()
# 这里数据只有两类，不太适合用卡方选择
# 使用word2vec试试
from gensim.models import word2vec
sentences = []
train = pd.read_csv("../data/train.tsv", delimiter="\t")
for i, row in train.iterrows():
    sentence, label = row.values
    sentences.append(sentence)

model = word2vec.Word2Vec(sentences, workers=4, size=300,min_count=0,window=10,sample=1e-3)

from sklearn.naive_bayes import MultinomialNB

