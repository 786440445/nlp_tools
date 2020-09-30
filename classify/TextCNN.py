import tensorflow as tf

class Graph:
    def __init__(self, args):
        self.embedding_size = args.embedding_size
        self.max_length = args.max_len
        self.vocab_size = args.vocab_size
        self.is_training = args.is_training
        self.class_size = args.class_size
        self.learning_rate = args.learning_rate
        self.num_filters = 64
        self.initializer = None
        self.filter_sizes = [2, 3, 4, 5, 6]
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        self.init_placeholder()
        self.build_model()

    def init_placeholder(self):
        self.seq = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length])
        self.label = tf.placeholder(dtype=tf.int32, shape=[None, self.class_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        self.embedding_table = tf.get_variable(dtype=tf.float32, shape=(self.vocab_size, self.embedding_size),
                                               name='embedding')

    def build_model(self):
        self.input = tf.nn.embedding_lookup(self.embedding_table, self.seq)
        self.pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            conv = tf.layers.conv1d(self.input, self.num_filters, filter_size,
                                    padding='valid',
                                    activation=tf.nn.relu,
                                    kernel_regularizer=self.regularizer)
            # global max pooling
            pooled = tf.layers.max_pooling1d(conv, self.max_length - filter_size + 1, 1)
            self.pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(self.pooled_outputs, 2)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("score"):
            fc = tf.layers.dense(h_pool_flat, self.embedding_size, activation=tf.nn.relu, name='fc1')
            fc = tf.layers.dropout(fc, 1-self.keep_prob)
            # classify
            self.logits = tf.layers.dense(fc, self.class_size, name='fc2')
            self.preds = tf.nn.sigmoid(self.logits)

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            # print(self.label)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits, labels=tf.cast(self.label, tf.float32))

            l2_loss = tf.losses.get_regularization_loss()
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.loss += l2_loss
            # optim
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)





