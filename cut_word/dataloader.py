import os
import sys
home_dir = os.getcwd()
cur_path = os.path.dirname(__file__)

import pickle
import numpy as np
from collections import Counter


def load_tag_dict():
    tag2label = {'0': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
    label2tag = dict([(index, tag) for (tag, index) in tag2label.items()])
    return tag2label, label2tag


def gen_dict(filename_list, mode='pkl'):
    chars = []
    for filename in filename_list:
        with open(os.path.join(cur_path, filename), 'r', encoding='utf-8') as file:
            chars.extend(list(''.join(file.read().split())))
    count = Counter(chars)
    count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    char2num_dict = dict([(item[0], index + 1) for (index, item) in enumerate(count)])
    char2num_dict['<UNK>'] = len(count) + 1
    char2num_dict['<PAD>'] = 0
    num2char_dict = dict(zip(char2num_dict.values(), char2num_dict.keys()))
    if mode == 'pkl':
        with open(os.path.join(cur_path, r'data\char2num.pkl'), 'wb') as f:
            pickle.dump(char2num_dict, f, pickle.HIGHEST_PROTOCOL)
    return char2num_dict, num2char_dict


def read_word_dict():
    with open(os.path.join(cur_path, 'data/char2num.pkl'), 'rb') as f:
        char2id_dict = pickle.load(f)
    id2char_dict = dict([(id, char) for (char, id) in char2id_dict.items()])
    return char2id_dict, id2char_dict


def shuffle(*arrs):
    """ shuffle

    Shuffle 数据

    Arguments:
        *arrs: 数组数据

    Returns:
        shuffle后的数据

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


'''数据预处理，标注字的位置 '''
def generate_label(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file, 'w', encoding='utf-8') as out:
        for line in lines:
            word_list = line.strip().split()
            for word in word_list:
                if len(word) == 1:
                    out.write(word + "##S ")
                else:
                    out.write(word[0] + "##B ")
                    for w in word[1: len(word) - 1]:
                        out.write(w + "##M ")
                    out.write(word[len(word) - 1] + "##E ")
            out.write("\n")


def get_corpus_data(file):
    word_list = []
    label_list = []
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line != '':
                words = line.split()
                word_list.append(list(map(lambda x: x.split("##")[0], words)))
                label_list.append(list(map(lambda x: x.split("##")[1], words)))
    return word_list, label_list


def tags2id(tags_list, tag2label):
    return [list(map(lambda x: tag2label.get(x), seq)) for seq in tags_list]


def sentence2id(sens_list, sen2label):
    return [list(map(lambda x: sen2label.get(x), seq)) for seq in sens_list]


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post',
                  truncating='post', value=0.):
    """ pad_sequences

    把序列长度转变为一样长的，如果设置了maxlen则长度统一为maxlen，如果没有设置则默认取
    最大的长度。填充和截取包括两种方法，post与pre，post指从尾部开始处理，pre指从头部
    开始处理，默认都是从尾部开始。

    Arguments:
        sequences: 序列
        maxlen: int 最大长度
        dtype: 转变后的数据类型
        padding: 填充方法'pre' or 'post'
        truncating: 截取方法'pre' or 'post'
        value: float 填充的值

    Returns:
        x: numpy array 填充后的序列维度为 (number_of_sequences, maxlen)

    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def load_all_data(file_location):
    # 获得数据
    tag2label, label2tag = load_tag_dict()
    word2id_dict, id2word_dict = read_word_dict()
    sentences_list, tags_list = get_corpus_data(file_location)
    sentences_list, tags_list = shuffle(sentences_list, tags_list)
    # 完成tag向索引的映射
    tags_index_list = tags2id(tags_list, tag2label)
    # 对索引进行填充
    tags_len_list = np.array([len(item) for item in tags_index_list])
    tags_index_pad_list = pad_sequences(tags_index_list)
    print('### tags_index_pad_list', np.shape(tags_index_pad_list))
    print('### tags_len_list', np.shape(tags_len_list))

    # 获得句子中每个字的id
    sen_index_list = sentence2id(sentences_list, word2id_dict)
    # 对句子或标注序列索引进行填充并获得每个句子的长度
    sen_len_list = np.array([len(item) for item in sen_index_list])
    sen_index_pad_list = pad_sequences(sen_index_list)

    print('### sen_index_pad_list', np.shape(sen_index_pad_list))
    print('### sen_len_list', np.shape(sen_len_list))
    for length, x in zip(tags_len_list, sentences_list):
        if length == 0:
            print(x)
    return sentences_list, sen_index_pad_list, sen_len_list, tags_list, tags_index_pad_list, tags_len_list


if __name__ == '__main__':
    input_file1 = 'data//pku_training.utf8'
    input_file2 = 'data//pku_test_gold.utf8'
    out_file1 = 'data//train_corpus.txt'
    out_file2 = 'data//test_corpus.txt'
    generate_label(input_file1, out_file1)
    generate_label(input_file2, out_file2)
    gen_dict([input_file1, input_file2])