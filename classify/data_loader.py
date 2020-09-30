import os
home_dir = os.getcwd()
import pandas as pd
import numpy as np

from collections import Counter

data_dir = os.path.join(home_dir, 'classify/data')

PAD = '<PAD>'
UNK = '<PAD>'
PAD_INDEX = 0
UNK_INDEX = 1


def get_train_eval_data():
    seqs_vecs, labels, seqs, seq_ids = load_char_data('train.csv')
    length = len(seqs)
    train_len = 3*length//5
    shuffle_list = np.random.permutation(length)
    shuffle_seqs = seqs[shuffle_list]
    shuffle_labels = labels[shuffle_list]
    shuffle_ids = seq_ids[shuffle_list]

    seqs_train = shuffle_seqs[:length]
    labels_train = shuffle_labels[:length]
    ids_train = shuffle_ids[:length]

    seqs_dev = shuffle_seqs[train_len:]
    labels_dev = shuffle_labels[train_len:]
    ids_dev = shuffle_ids[train_len:]
    train_data = (seqs_train, labels_train, ids_train)
    dev_data = (seqs_dev, labels_dev, ids_dev)
    return train_data, dev_data


def load_char_data(filename):
    word2idx, idx2word = load_char_vocab()
    data_df = pd.read_csv(os.path.join(data_dir, filename))
    labels = []
    seqs = []
    seq_ids = []
    for id, seq_id, cate_0, cate_1, cate_2, cate_3, cate_4, cate_5, seq in data_df.itertuples():
        label = [cate_0, cate_1, cate_2, cate_3, cate_4, cate_5]
        seq_ids.append(seq_id)
        seqs.append(seq)
        labels.append(label)
    seqs_pad = padding(seqs, max_len=300)
    seqs_vec = sentence2id(seqs_pad, word2idx)
    return np.array(seqs_vec), np.array(labels), np.array(seqs), np.array(seq_ids)


def sentence2id(sens_list, sen2label):
    return [list(map(lambda x: sen2label.get(x, UNK_INDEX), seq)) for seq in sens_list]


# 加载字典
def load_char_vocab():
    vocab_path = 'classify/vocab.txt'
    vocab = [line.strip() for line in open(vocab_path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def build_vocab():
    chars = ''
    for path, _, files in  os.walk('classify/data'):
        for file in files:
            data_df = pd.read_csv(os.path.join(path, file))
            seqs = data_df["Question Sentence"].values.tolist()
            for seq in seqs:
                chars += seq
    dic = Counter(list(chars))
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    chars = [item for item, count in dic if count >= 5]
    chars = ['<PAD>', '<UNK>'] + chars
    with open('classify/vocab.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(chars))


def padding(seqs, max_len=300):
    new_seqs = []
    for seq in seqs:
        l = min(max_len, len(seq))
        seq_padding = list(seq[:l]) + [PAD_INDEX] * (max_len - l)
        new_seqs.append(seq_padding)
    return new_seqs


if __name__ == '__main__':
    # seqs, labels = load_char_data('train.csv')
    # all_count = len(seqs)
    # dict = {}
    # max_l = 0
    # for seq in seqs:
    #     l = len(seq)
    #     if l > max_l:
    #         max_l = l
    #     dict[l] = dict.get(l, 0) + 1
    # # dict = sorted(dict.items(), key=lambda x: x[0], reverse=True)
    # count = 0
    # min_l = 0
    # sum = 0
    # while min_l <= max_l:
    #     sum += dict.get(min_l, 0)
    #     if min_l % 30 == 0:
    #         print(min_l, ': %.4f' % (sum / all_count))
    #     min_l += 1
    build_vocab()