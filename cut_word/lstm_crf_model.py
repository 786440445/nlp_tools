# encoding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood


class BiLSTM_CRF():
    def __init__(self, num_tags, vocab_size, args):
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.hidden_dim = args.hidden_dim
        self.learning_rate = args.learning_rate
        self.clip_grad = args.clip_grad
        self.embedding_size = args.embedding_size

    def build_model(self):
        self.init_placeholder()
        self.inference()
        self.calc_loss()

    def init_placeholder(self):
        # 第一维大小为batch_size,第二维是句子的长度是动态获得的没法设置
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.target_y = tf.placeholder(tf.int32, [None, None], name='target_y')
        self.seq_lengths = tf.placeholder(tf.int32, shape=[None], name="seq_lengths")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    def inference(self):
        with tf.name_scope("embedding"):
            embedding_mat = tf.get_variable('lookup_table', dtype=tf.float32,
                                            shape=[self.vocab_size, self.embedding_size],
                                            initializer=tf.contrib.layers.xavier_initializer())
            self.embedding_x = tf.nn.embedding_lookup(embedding_mat, self.input_x)

        with tf.variable_scope('lstm'):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embedding_x,
                                                                        self.seq_lengths, dtype=tf.float32)

            out_put = tf.concat([output_fw, output_bw], axis=-1)  # 对正反向的输出进行合并
            out_put = tf.nn.dropout(out_put, self.keep_prob)  # 防止过拟合

            self.logits = tf.layers.dense(out_put, self.num_tags,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='logits')

        with tf.variable_scope('crf'):
            self.log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                             tag_indices=self.target_y,
                                                                             sequence_lengths=self.seq_lengths)

    def calc_loss(self):
        self.loss = -tf.reduce_mean(self.log_likelihood, name='loss')
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optim.compute_gradients(self.loss)
        grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
        self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)