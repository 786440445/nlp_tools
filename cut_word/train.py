import os
import pickle
import numpy as np
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

input_x_holder = tf.placeholder(tf.int32, [None, None], name='input_x')
target_y_holder = tf.placeholder(tf.int32, [None, None], name='target_y')
seq_lengths_holder = tf.placeholder(tf.int32, shape=[None], name="seq_lengths")

dataset = tf.data.Dataset.from_tensor_slices((input_x_holder, target_y_holder, seq_lengths_holder))
dataset_train = dataset.batch(FLAGS.batch_size).repeat(FLAGS.num_epochs)
iterator_train = dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

dataset_eval = dataset.batch(FLAGS.batch_size)
iterator_eval = dataset_eval.make_initializable_iterator()
next_element_eval = iterator_eval.get_next()


sentences_list_train, sen_index_pad_list_train, sen_len_list_train, tags_list_train, tags_index_pad_list_train, tags_len_list_train = \
    load_all_data(os.path.join(cur_path, 'data//train_corpus.txt'))
sentences_list_test, sen_index_pad_list_test, sen_len_list_test, tags_list_test, tags_index_pad_list_test, tags_len_list_test = \
    load_all_data(os.path.join(cur_path, 'data//test_corpus.txt'))


with tf.Session() as sess:
    model = BiLSTM_CRF(num_tags, vocab_size, FLAGS)
    model.build_model()
    saver = tf.compat.v1.train.Saver(max_to_keep=3)
    ckpt = tf.train.latest_checkpoint(os.path.join(cur_path, 'model'))
    if ckpt is None:
        sess.run(tf.global_variables_initializer())
    else:
        print('### restore model')
        saver.restore(sess, ckpt)
    # 模型的训练
    sess.run(iterator_train.initializer, feed_dict={input_x_holder: sen_index_pad_list_train,
                                                    target_y_holder: tags_index_pad_list_train,
                                                    seq_lengths_holder: tags_len_list_train})
    # 模型的训练
    sess.run(iterator_eval.initializer, feed_dict={input_x_holder: sen_index_pad_list_test,
                                                   target_y_holder: tags_index_pad_list_test,
                                                   seq_lengths_holder: tags_len_list_test})

    steps = int(len(tags_len_list_train) / FLAGS.batch_size)
    old_acc = 0
    for epoch in range(FLAGS.num_epochs):
        for step in range(steps):
            try:
                input_x_batch, target_y_batch, seq_lengths_batch = sess.run(next_element_train)
                feed_dict = {model.input_x: input_x_batch,
                             model.target_y: target_y_batch,
                             model.seq_lengths: seq_lengths_batch,
                             model.keep_prob: FLAGS.keep_prob}

                _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                if step % 5 == 0 or step == steps - 1:
                        print('epoch:', epoch, ' step:', step, ' loss:', loss)
            except tf.errors.OutOfRangeError:
                print('\n')

        # 对测试集进行测试
        eval_steps = int(len(tags_len_list_test) / FLAGS.batch_size)
        total_acc_num = 0
        total_num = 0
        for step in range(eval_steps):
            try:
                input_x_batch, target_y_batch, seq_lengths_batch = sess.run(next_element_eval)
                feed_dict = {model.input_x: input_x_batch,
                             model.target_y: target_y_batch,
                             model.seq_lengths: seq_lengths_batch,
                             model.keep_prob: 1}
                logits, transition_params = sess.run([model.logits, model.transition_params], feed_dict=feed_dict)
                label_list = []
                # print(sen_len_list_test)
                for logit, seq_len in zip(logits, seq_lengths_batch):
                    # viterbi_decode通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
                    # viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表.
                    # viterbi_score: 序列对应的概率值
                    # 这是解码的过程，利用维比特算法结合概率转移矩阵求得最大的可能标注概率
                    viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                    label_list.append(viterbi_seq)

                tags_list = []
                for labels in label_list:
                    tags = []
                    for i in labels:
                        tags.append(i)
                    tags_list.append(tags)

                # 计算精度
                for pre_tags, tags_ids in zip(tags_list, target_y_batch):
                    total_num += len(pre_tags)
                    for pre_tag, test_tag in zip(pre_tags, tags_ids):
                        if pre_tag == test_tag:
                            total_acc_num += 1

            except tf.errors.OutOfRangeError:
                print('\n')

        acc = total_acc_num / total_num
        print('epoch:{0}  eval acc:{1:.6f}   loss:{2}'.format(epoch, acc, loss))
        saver.save(sess, os.path.join(cur_path, 'model/model_{0}_{1:.3f}.ckpt'.format(epoch, acc)))
        if acc > old_acc:
            saver.save(sess, os.path.join(cur_path, 'model/final_model.ckpt'))
            old_acc = acc