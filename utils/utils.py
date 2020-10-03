from torchtext import data
import pandas as pd
import torch


# 流程：读取csv -> 数据清洗 -> 定义域 -> 定义数据集 -> 构造数据迭代器
# 这里的数据都是已经清洗过和划分好的

def n_gram_tokenizer(x, n):
    x = x.split()
    result = []
    n_grams = set(zip(*[x[i:] for i in range(n)]))
    for n_gram in n_grams:
        result.append(" ".join(n_gram))
    return result


class Config:
    def __init__(self):
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list)
        self.char_field = data.NestedField(self.CHAR_NESTING, tokenize=lambda x: x.split(), fix_length=60)
        # 构建Field
        self.TEXT = data.Field(batch_first=True, lower=True, tokenize=lambda x: x.split(),
                               fix_length=60)
        self.bigram = data.Field(batch_first=True, lower=True, tokenize=lambda x: n_gram_tokenizer(x, 2),
                                 fix_length=60)
        self.trigram = data.Field(batch_first=True, lower=True, tokenize=lambda x: n_gram_tokenizer(x, 3),
                                  fix_length=60)
        # 标签域一定要加LabelField！！！！气哭
        self.LABEL = data.LabelField(use_vocab=True, dtype=torch.long)
        self.WORD_FIELD = [("sentence_word", self.TEXT), ("label", self.LABEL)]
        self.CHAR_FIELD = [("sentence_char", self.char_field), ("sentence_word", self.TEXT), ("label", self.LABEL)]
        self.BIGRAM_FIELD = [("sentence_word", self.TEXT),("sentence_bigram", self.bigram),("label", self.LABEL)]

class DataIter:
    def __init__(self, field="word", n_gram=False, batch_size=128):
        self.config = Config()
        self.train = pd.read_csv("data/train.tsv", delimiter="\t")
        self.dev = pd.read_csv("data/dev.tsv", delimiter="\t")
        self.test = pd.read_csv("data/test.tsv", delimiter="\t")
        self.batch_size = batch_size
        self.field = field

        print("Read 3 files from disk!")
        print("Shape of train is {}, dev is {}, test is {}".format(self.train.shape, self.dev.shape, self.test.shape))
        # 参数：
        #     sequential  是否为文本序列数据   如果是False，则分词方法时空的  默认：True
        #     use_vocab   是否使用词典        如果是False，则数据应该已经是数值了  默认：True
        #     init_token  起始符              自定义起始符，如果是None则没有   默认：None
        #     eos_token   结束符              自定义结束符，如果是None则没有   默认：None
        #     fix_length  填充长度            自定义填充长度，如果是None则不填充   默认：None
        #     dtype       数据类型            默认：torch.long
        #     preprocessing  预处理管道       在样本经过Field分词过后，数值化之前的一些操作  默认：None
        #     postprocessing  后期处理        在样本数值化之后，转换成Tensor之前的一些操作   默认：None
        #     lower        是否转成小写        默认：False
        #     tokenize     将字符串转换成序列化样本，如果输入"spacy"，则会使用Spacy Tokenizer   默认：string.split()
        #     tokenizer_languages  构建分词器的语言
        #     include_lengths  返回值是否附带句子的长度    默认：False
        #     batch_first  第一维是否为batch_size   默认：False
        #     pad_token 填充符号 默认: <pad>
        #     unk_token 未知符号 默认：<unk>
        #     pad_first 句子开头先填充  默认: False.
        #     truncate_first 句子开头先截断  默认: False
        #     stop_words 停用词  默认: None
        #     is_target 是否为一个目标变量。影响批量迭代。 默认: False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fetch_iter(self):

        train_ds, dev_ds, test_ds = None, None, None
        if self.field == "word":
            train_ds, dev_ds, test_ds = DataSet.splits(self.config.WORD_FIELD, self.train, self.dev, self.test)
            self.config.TEXT.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
            self.config.LABEL.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
        elif self.field == "char":
            train_ds, dev_ds, test_ds = DataSet.splits(self.config.CHAR_FIELD, self.train, self.dev, self.test,
                                                       field="char")
            self.config.TEXT.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
            self.config.LABEL.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
            self.config.char_field.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
        elif self.field == "bigram":
            train_ds, dev_ds, test_ds = DataSet.splits(self.config.BIGRAM_FIELD, self.train, self.dev, self.test,
                                                       field="bigram")
            self.config.TEXT.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
            self.config.LABEL.build_vocab(train_ds, dev_ds, test_ds, max_size=25000)
            self.config.bigram.build_vocab(train_ds, dev_ds, test_ds, max_size=50000)

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_ds, dev_ds, test_ds),
            batch_sizes=(self.batch_size, len(dev_ds), len(test_ds)),
            sort_within_batch=True,
            device=self.device,
        )

        return train_iterator, valid_iterator, test_iterator


# 构建数据集
class DataSet(data.Dataset):
    def __init__(self, ds, fields, is_char=False, is_test=False, field = "word"):
        examples = []
        for i, row in ds.iterrows():
            label = row.label if not is_test else None
            text = row.sentence
            if field == "word":
                examples.append(data.Example.fromlist([text, label], fields))
            elif field == "char":
                examples.append(data.Example.fromlist([text, text, label], fields))
            elif field == "bigram":
                examples.append(data.Example.fromlist([text, text, label], fields))
        super().__init__(examples, fields)

    @staticmethod
    def sort_key(ex):
        return len(ex.sentence_word)

    @classmethod
    def splits(cls, fields, train_df=None, dev_df=None, test_df=None, field = "word", **kwargs):
        train_data, dev_data, test_data = (None, None, None)
        data_fields = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_fields, field=field)
        if dev_df is not None:
            dev_data = cls(dev_df.copy(), data_fields, field=field)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_fields, field=field)
        return tuple(x for x in (train_data, dev_data, test_data) if x is not None)


if __name__ == "__main__":
    Iter = DataIter(n_gram=True)
    a, b, c = Iter.fetch_iter()
    for (X, y), z in a:
        print(X.shape)
        print(y.shape)
        print(z.shape)
        break
    # sen = "she goes to see me yesterday !"
    # config = Config()
    # example = [data.Example.fromlist([sen,sen,0], config.CHAR_FIELD)]
    # dataset = data.Dataset(example, config.CHAR_FIELD)
    # config.char_field.build_vocab(dataset)
    # config.TEXT.build_vocab(dataset)
    # config.LABEL.build_vocab(dataset)
    # iters = data.Iterator(dataset, 1)
    # for (X, (p,q)),y in iters:
    #     print(X.shape, p.shape)
