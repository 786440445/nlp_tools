import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode

from cut_word.lstm_crf_model import BiLSTM_CRF
from cut_word.dataloader import *
cur_path = os.path.dirname(__file__)
np.set_printoptions(threshold=np.inf)


tf.flags.DEFINE_integer("embedding_size", 300, "每个字向量的维度")
tf.flags.DEFINE_integer("hidden_dim", 300, "LSTM隐藏层细胞的个数")
tf.flags.DEFINE_integer("batch_size", 64, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 100, "训练的轮数")
tf.flags.DEFINE_float("keep_prob", 0.8, "丢失率")
tf.flags.DEFINE_float("clip_grad", 5.0, "梯度的范围")
tf.flags.DEFINE_float("learning_rate", 0.001, "学习率")
FLAGS = tf.flags.FLAGS

# 用于每个字标记向索引的映射，注意“O”对应的必须是0因为标签的填充是以0进行填充的
tag2label, label2tag = load_tag_dict()
num_tags = len(tag2label)
word2id_dict, id2word_dict = read_word_dict()
vocab_size = len(word2id_dict)

model = BiLSTM_CRF(num_tags, vocab_size, FLAGS)
model.build_model()
saver = tf.compat.v1.train.Saver(max_to_keep=3)
ckpt = tf.train.latest_checkpoint(os.path.join(cur_path, 'model'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('### restore model')
saver.restore(sess, ckpt)


def predict(sens):
    char_string = np.array([list(sen) for sen in sens])
    input_ids = np.array(sentence2id(char_string, word2id_dict))
    input_length = np.array([len(item) for item in input_ids])
    feed_dict = {model.input_x: input_ids,
                 model.seq_lengths: input_length,
                 model.keep_prob: 1}

    logits, transition_params = sess.run([model.logits, model.transition_params], feed_dict=feed_dict)

    label_list = []
    # print(sen_len_list_test)
    for logit, seq_len in zip(logits, input_length):
        # viterbi_decode通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
        # viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表.
        # viterbi_score: 序列对应的概率值
        # 这是解码的过程，利用维比特算法结合概率转移矩阵求得最大的可能标注概率
        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
        label_list.append(viterbi_seq)
    tags = [list(map(lambda label: label2tag.get(label), labels)) for labels in label_list]
    return tags


def switch_cut_words(str_list, tag_list):
    cut_words = []
    for inputx, tag in zip(str_list, tag_list):
        if type(inputx) == 'str':
            str_list = list(str_list)
        cut_word = ''
        for index, (item, tag) in enumerate(zip(inputx, tag)):
            if index == 0:
                cut_word += item
            else:
                if tag == 'B':
                    cut_word += ' ' + item
                elif tag == 'M' or tag == 'E':
                    cut_word += item
                elif tag == 'S':
                    cut_word += ' ' + item
                else:
                    pass
        cut_words.append(cut_word.strip())
    return cut_words


if __name__ == "__main__":
    inputx = ['美国今日和我国经济贸易发生冲突！']
    print(list(inputx))
    tags = predict(inputx)
    cut_words = switch_cut_words(inputx, tags)
    print(cut_words)